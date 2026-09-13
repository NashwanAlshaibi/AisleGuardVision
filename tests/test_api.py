"""FastAPI service: endpoints, auth, and what must never be exposed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aisleguardvision.api.server import create_app
from aisleguardvision.api.state import RuntimeState, set_state
from aisleguardvision.camera.manager import CameraManager
from aisleguardvision.core.config import AppConfig, CameraConfig, ZoneConfig
from aisleguardvision.core.types import BehaviorState, ThreatLevel
from aisleguardvision.events.models import build_event
from aisleguardvision.events.recorder import IncidentRecorder, IncidentStore

from test_events import assessment


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("CAM_TEST_RTSP", "rtsp://operator:hunter2@10.0.0.5:554/stream")
    config = AppConfig()
    config.storage.incident_directory = tmp_path / "incidents"

    camera = CameraConfig(
        id="cam_001",
        name="Test Camera",
        source="rtsp://operator:hunter2@10.0.0.5:554/stream",
        enabled=True,
        zones=[
            ZoneConfig(id="shelf_001", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]),
        ],
    )
    config.cameras.cameras.append(camera)

    manager = CameraManager(config)
    manager.add_camera(camera, start=False)

    state = RuntimeState(
        config=config,
        camera_manager=manager,
        incident_store=IncidentStore(config.storage),
        device_label="cpu (test)",
        item_detection_available=False,
    )
    set_state(state)
    yield state
    manager.stop_all()
    set_state(None)


@pytest.fixture
def client(runtime) -> TestClient:
    return TestClient(create_app(runtime.config))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_is_reachable_without_a_credential(client):
    """A load balancer or supervisor must be able to probe this."""
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] in {"healthy", "degraded", "starting"}
    assert body["cameras_total"] == 1
    assert "version" in body


def test_health_states_that_this_is_not_a_theft_determination(client):
    assert "not determine" in client.get("/health").json()["notice"].lower()


def test_health_reports_zone_only_mode(client):
    assert client.get("/health").json()["item_detection_available"] is False


def test_health_without_a_runtime_returns_503():
    set_state(None)
    response = TestClient(create_app(AppConfig()), raise_server_exceptions=False).get("/health")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def test_list_cameras(client):
    body = client.get("/cameras").json()
    assert body["count"] == 1
    assert body["cameras"][0]["camera_id"] == "cam_001"
    assert body["cameras"][0]["zone_count"] == 1


def test_camera_response_never_leaks_the_rtsp_password(client):
    """This is the endpoint most likely to end up in a browser devtools tab."""
    raw = client.get("/cameras").text
    assert "hunter2" not in raw

    source_label = client.get("/cameras").json()["cameras"][0]["source_label"]
    assert "***" in source_label
    assert "10.0.0.5" in source_label


def test_get_one_camera(client):
    body = client.get("/cameras/cam_001").json()
    assert body["camera_id"] == "cam_001"
    assert body["name"] == "Test Camera"


def test_unknown_camera_returns_404(client):
    assert client.get("/cameras/nope").status_code == 404


def test_enable_and_disable_a_camera(client):
    assert client.post("/cameras/cam_001/disable").json()["enabled"] is False
    assert client.post("/cameras/cam_001/enable").json()["enabled"] is True


def test_enable_unknown_camera_returns_404(client):
    assert client.post("/cameras/nope/enable").status_code == 404


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_events_endpoint_is_empty_initially(client):
    body = client.get("/events").json()
    assert body["count"] == 0
    assert body["events"] == []


def test_events_are_listed_and_fetchable(client, runtime):
    recorder = IncidentRecorder(runtime.config.recording, runtime.config.storage)
    event = recorder.record(build_event(assessment()), None, None)

    listing = client.get("/events").json()
    assert listing["count"] == 1
    summary = listing["events"][0]
    assert summary["event_id"] == event.event_id
    assert summary["threat_level"] == ThreatLevel.HIGH_RISK.value
    assert summary["positive_evidence_count"] >= 1

    detail = client.get(f"/events/{event.event_id}").json()
    assert detail["risk_score"] == pytest.approx(91.2)
    assert detail["behavior_state"] == BehaviorState.REVIEW_ALERT.value
    assert detail["evidence_breakdown"], "the full arithmetic must be fetchable"
    assert "human review" in detail["notice"].lower()


def test_events_can_be_filtered(client, runtime):
    recorder = IncidentRecorder(runtime.config.recording, runtime.config.storage)
    recorder.record(build_event(assessment(score=95)), None, None)
    recorder.record(build_event(assessment(score=86)), None, None)

    assert client.get("/events", params={"min_risk": 90}).json()["count"] == 1
    assert client.get("/events", params={"limit": 1}).json()["count"] == 1
    assert client.get("/events", params={"camera_id": "cam_999"}).json()["count"] == 0


def test_unknown_event_returns_404(client):
    assert client.get("/events/does-not-exist").status_code == 404


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_json(client, runtime):
    runtime.metrics.inc("frames_processed", 5, camera_id="cam_001")
    runtime.metrics.observe("inference_latency_ms", 31.0, camera_id="cam_001")

    body = client.get("/metrics").json()
    assert any("frames_processed" in key for key in body["counters"])
    assert any("inference_latency_ms" in key for key in body["histograms"])


def test_metrics_prometheus_format(client, runtime):
    runtime.metrics.inc("alerts_generated", 2, camera_id="cam_001")
    response = client.get("/metrics", params={"format": "prometheus"})

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "# TYPE" in response.text
    assert 'camera_id="cam_001"' in response.text


def test_metrics_rejects_an_unknown_format(client):
    assert client.get("/metrics", params={"format": "xml"}).status_code == 422


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_api_key_is_required_when_configured(runtime):
    runtime.config.api.api_key = "s3cret"
    client = TestClient(create_app(runtime.config))

    assert client.get("/cameras").status_code == 401
    assert client.get("/cameras", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/cameras", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_health_stays_open_even_with_auth_enabled(runtime):
    runtime.config.api.api_key = "s3cret"
    client = TestClient(create_app(runtime.config))
    assert client.get("/health").status_code == 200


def test_empty_api_key_disables_auth(runtime):
    runtime.config.api.api_key = ""
    client = TestClient(create_app(runtime.config))
    assert client.get("/cameras").status_code == 200


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_openapi_schema_is_generated(client):
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    for expected in (
        "/health",
        "/metrics",
        "/cameras",
        "/cameras/{camera_id}",
        "/cameras/{camera_id}/enable",
        "/cameras/{camera_id}/disable",
        "/events",
        "/events/{event_id}",
    ):
        assert expected in paths, f"missing endpoint {expected}"


def test_api_description_states_the_privacy_position(client):
    description = client.get("/openapi.json").json()["info"]["description"].lower()
    assert "no facial" in description
    assert "temporary" in description
