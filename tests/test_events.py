"""Events: cooldown, incident recording, alert dispatch and payload schema."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from aisleguardvision.camera.frame_buffer import CircularFrameBuffer
from aisleguardvision.core.config import (
    AlertsConfig,
    RecordingConfig,
    StorageConfig,
    WebhookConfig,
)
from aisleguardvision.core.types import (
    BehaviorState,
    EventType,
    EvidenceType,
    Frame,
    RiskAssessment,
    RiskContribution,
    SecurityEvent,
    ThreatLevel,
)
from aisleguardvision.events.cooldown import AlertCooldown
from aisleguardvision.events.dispatcher import (
    AlertDispatcher,
    AlertProvider,
    CallbackAlertProvider,
    ConsoleAlertProvider,
    WebhookAlertProvider,
    set_default_dispatcher,
    trigger_alert,
)
from aisleguardvision.events.models import (
    INTERPRETATION_NOTICE,
    SecurityEventModel,
    build_event,
)
from aisleguardvision.events.recorder import IncidentRecorder, IncidentStore, apply_retention


def assessment(score: float = 91.2, person_id: int = 23) -> RiskAssessment:
    return RiskAssessment(
        person_id=person_id,
        camera_id="cam_001",
        timestamp=time.time(),
        risk_score=score,
        threat_level=ThreatLevel.from_score(score),
        behavior_state=BehaviorState.REVIEW_ALERT,
        contributions=[
            RiskContribution(
                EvidenceType.SHELF_INTERACTION, "Shelf interaction detected", 10.0, 1.0, 10.0
            ),
            RiskContribution(
                EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
                "Item became occluded near right waist",
                25.0,
                1.0,
                25.0,
            ),
            RiskContribution(
                EvidenceType.NO_BASKET_PLACEMENT, "No basket placement detected", 0.0, 1.0, 0.0
            ),
            RiskContribution(
                EvidenceType.LOW_POSE_CONFIDENCE, "Pose confidence is low", -20.0, 1.0, -20.0
            ),
        ],
        raw_score=score,
    )


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


def test_first_alert_is_always_allowed():
    cooldown = AlertCooldown(30.0)
    assert cooldown.check("cam", 1, 100.0, 90.0).allowed


def test_repeat_alert_is_suppressed_within_the_cooldown():
    """Without this, one sustained sequence produces hundreds of notifications
    and staff stop reading them."""
    cooldown = AlertCooldown(30.0, escalation_delta=10.0)
    cooldown.record("cam", 1, 100.0, 90.0, "e1")

    decision = cooldown.check("cam", 1, 110.0, 91.0)
    assert not decision.allowed
    assert decision.seconds_remaining == pytest.approx(20.0)


def test_alert_is_allowed_after_the_cooldown_expires():
    cooldown = AlertCooldown(30.0)
    cooldown.record("cam", 1, 100.0, 90.0, "e1")
    assert cooldown.check("cam", 1, 131.0, 90.0).allowed


def test_meaningful_escalation_overrides_the_cooldown():
    """Suppressing a REVIEW alert that has become HIGH RISK would hide exactly
    the escalation staff need to see."""
    cooldown = AlertCooldown(30.0, escalation_delta=10.0)
    cooldown.record("cam", 1, 100.0, 86.0, "e1")

    assert not cooldown.check("cam", 1, 105.0, 90.0).allowed, "4 points is not an escalation"
    decision = cooldown.check("cam", 1, 105.0, 99.0)
    assert decision.allowed and "escalated" in decision.reason


def test_a_distinct_event_type_overrides_the_cooldown():
    cooldown = AlertCooldown(30.0)
    cooldown.record("cam", 1, 100.0, 90.0, "e1", EventType.POSSIBLE_CONCEALMENT)
    assert cooldown.check("cam", 1, 105.0, 90.0, EventType.ELEVATED_BEHAVIOR).allowed


def test_cooldown_is_per_person_and_per_camera():
    cooldown = AlertCooldown(30.0)
    cooldown.record("cam", 1, 100.0, 90.0, "e1")

    assert cooldown.check("cam", 2, 101.0, 90.0).allowed, "a different person"
    assert cooldown.check("cam_other", 1, 101.0, 90.0).allowed, "a different camera"


def test_cooldown_state_is_dropped_for_retired_tracks():
    """Track ids are reused. Stale state would suppress a genuine alert for a
    different shopper who inherited the id."""
    cooldown = AlertCooldown(30.0)
    cooldown.record("cam", 1, 100.0, 90.0, "e1")
    cooldown.record("cam", 2, 100.0, 90.0, "e2")

    cooldown.retain_only("cam", {2})
    assert cooldown.check("cam", 1, 101.0, 90.0).allowed
    assert not cooldown.check("cam", 2, 101.0, 90.0).allowed


def test_forget_clears_one_track():
    cooldown = AlertCooldown(30.0)
    cooldown.record("cam", 1, 100.0, 90.0, "e1")
    cooldown.forget("cam", 1)
    assert cooldown.check("cam", 1, 101.0, 90.0).allowed


# ---------------------------------------------------------------------------
# Event payloads
# ---------------------------------------------------------------------------


def test_build_event_carries_the_evidence():
    event = build_event(assessment())
    assert event.risk_score == pytest.approx(91.2)
    assert event.threat_level is ThreatLevel.HIGH_RISK
    assert "Shelf interaction detected" in event.positive_evidence
    assert "Pose confidence is low" in event.negative_evidence


def test_event_ids_are_unique():
    ids = {build_event(assessment()).event_id for _ in range(50)}
    assert len(ids) == 50


def test_event_model_matches_the_documented_payload_shape():
    model = SecurityEventModel.from_event(build_event(assessment()))
    payload = model.to_webhook_payload()

    assert set(payload) >= {
        "event_id",
        "camera_id",
        "person_id",
        "timestamp",
        "risk_score",
        "event_type",
        "positive_evidence",
        "negative_evidence",
        "snapshot_path",
        "clip_path",
    }
    assert payload["event_type"] == "possible_concealment"
    # Serializable without custom encoders.
    json.dumps(payload)


def test_every_payload_carries_the_interpretation_notice():
    """A consumer must not be able to read one of these without seeing that it
    is not a theft determination."""
    model = SecurityEventModel.from_event(build_event(assessment()))
    assert model.notice == INTERPRETATION_NOTICE
    assert "not a determination" in model.notice.lower()
    assert "human review" in model.to_webhook_payload()["notice"].lower()


def test_event_model_breakdown_is_sorted_by_impact():
    model = SecurityEventModel.from_event(build_event(assessment()))
    points = [abs(item.points) for item in model.evidence_breakdown]
    assert points == sorted(points, reverse=True)


def test_event_model_records_zero_weight_evidence():
    """'Nothing benign was observed' is what a reviewer needs to trust the alert."""
    model = SecurityEventModel.from_event(build_event(assessment()))
    labels = {item.evidence for item in model.evidence_breakdown}
    assert EvidenceType.NO_BASKET_PLACEMENT.value in labels


# ---------------------------------------------------------------------------
# Incident recording
# ---------------------------------------------------------------------------


@pytest.fixture
def recorder(tmp_path) -> IncidentRecorder:
    recording = RecordingConfig(
        pre_event_seconds=1.0, post_event_seconds=0.2, clip_fps=10.0, buffer_max_width=4096
    )
    storage = StorageConfig(incident_directory=tmp_path / "incidents")
    instance = IncidentRecorder(recording, storage)
    instance.start()
    yield instance
    instance.stop(timeout=10)


def filled_buffer(event_time: float) -> CircularFrameBuffer:
    buffer = CircularFrameBuffer(seconds=10.0, max_frames=200, max_width=4096)
    for index in range(30):
        buffer.append(
            Frame(
                camera_id="cam_001",
                frame_id=index,
                timestamp=event_time - 1.0 + index * 0.05,
                image=np.full((120, 160, 3), index * 8 % 256, dtype=np.uint8),
            )
        )
    return buffer


def test_incident_writes_json_metadata(recorder, tmp_path):
    event = recorder.record(build_event(assessment()), None, None)

    assert event.metadata_path is not None
    path = Path(event.metadata_path)
    assert path.exists()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["event_id"] == event.event_id
    assert data["risk_score"] == pytest.approx(91.2)
    assert data["positive_evidence"]
    assert "not a determination" in data["notice"].lower()


def test_incident_directory_is_organized_by_camera_and_date(recorder):
    event = recorder.record(build_event(assessment()), None, None)
    path = Path(event.metadata_path)

    assert path.parent.parent.name == "cam_001"
    assert len(path.parent.name) == len("2026-09-12"), "expected a YYYY-MM-DD directory"
    assert path.stem == event.event_id


def test_incident_writes_a_snapshot_and_a_clip(recorder):
    raw = build_event(assessment())
    snapshot = np.full((120, 160, 3), 128, dtype=np.uint8)
    event = recorder.record(raw, filled_buffer(raw.timestamp), snapshot)

    assert recorder.wait_for_media(event.event_id, timeout=20)
    assert Path(event.snapshot_path).exists()
    assert Path(event.clip_path).exists()
    assert Path(event.clip_path).stat().st_size > 0


def test_recording_does_not_block_the_caller(recorder):
    """Encoding ten seconds of video takes far longer than a frame interval;
    doing it inline would stall the very camera being recorded."""
    raw = build_event(assessment())
    buffer = filled_buffer(raw.timestamp)

    started = time.perf_counter()
    recorder.record(raw, buffer, np.zeros((120, 160, 3), dtype=np.uint8))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"record() blocked for {elapsed:.3f}s"
    recorder.wait_for_media(raw.event_id, timeout=20)


def test_incident_store_lists_and_fetches(recorder, tmp_path):
    events = [recorder.record(build_event(assessment(score=90 + i)), None, None) for i in range(3)]
    store = IncidentStore(StorageConfig(incident_directory=tmp_path / "incidents"))

    assert store.count() == 3
    assert len(store.list_events(limit=10)) == 3
    assert len(store.list_events(limit=2)) == 2
    assert store.get_event(events[0].event_id) is not None
    assert store.get_event("does-not-exist") is None


def test_incident_store_filters_by_camera_and_risk(recorder, tmp_path):
    recorder.record(build_event(assessment(score=95)), None, None)
    recorder.record(build_event(assessment(score=86)), None, None)
    store = IncidentStore(StorageConfig(incident_directory=tmp_path / "incidents"))

    assert len(store.list_events(camera_id="cam_001")) == 2
    assert len(store.list_events(camera_id="cam_999")) == 0
    assert len(store.list_events(min_risk=90.0)) == 1


def test_incident_store_skips_unreadable_records(recorder, tmp_path):
    event = recorder.record(build_event(assessment()), None, None)
    Path(event.metadata_path).write_text("{ not json", encoding="utf-8")

    store = IncidentStore(StorageConfig(incident_directory=tmp_path / "incidents"))
    assert store.list_events() == []


def test_retention_is_disabled_by_default(tmp_path):
    """Silently deleting evidence is a policy decision a store makes, not one a
    default should make for them."""
    storage = StorageConfig(incident_directory=tmp_path)
    assert not storage.retention.enabled
    assert apply_retention(storage) == 0


def test_retention_removes_expired_incidents(recorder, tmp_path):
    event = recorder.record(build_event(assessment()), None, None)
    old = Path(event.metadata_path)
    ancient = time.time() - 90 * 86400
    import os

    os.utime(old, (ancient, ancient))

    storage = StorageConfig(incident_directory=tmp_path / "incidents")
    storage.retention.enabled = True
    storage.retention.max_age_days = 30

    assert apply_retention(storage) == 1
    assert not old.exists()


# ---------------------------------------------------------------------------
# Alert dispatch
# ---------------------------------------------------------------------------


class RecordingProvider(AlertProvider):
    name = "recording"

    def __init__(self, succeed: bool = True) -> None:
        self.events: list[SecurityEventModel] = []
        self.succeed = succeed
        self.ready = threading.Event()

    def send(self, event: SecurityEventModel) -> bool:
        self.events.append(event)
        self.ready.set()
        return self.succeed


def test_dispatcher_delivers_to_providers():
    provider = RecordingProvider()
    dispatcher = AlertDispatcher(AlertsConfig(console=False), providers=[provider])
    dispatcher.start()
    try:
        assert dispatcher.dispatch(build_event(assessment()))
        assert provider.ready.wait(timeout=5)
        assert len(provider.events) == 1
    finally:
        dispatcher.stop(timeout=5)


def test_dispatch_does_not_block_the_caller():
    """A webhook to a store's ticketing system can hang; doing that on the
    inference thread would stall every camera it serves."""

    class SlowProvider(AlertProvider):
        name = "slow"

        def send(self, event):
            time.sleep(0.5)
            return True

    dispatcher = AlertDispatcher(AlertsConfig(console=False), providers=[SlowProvider()])
    dispatcher.start()
    try:
        started = time.perf_counter()
        dispatcher.dispatch(build_event(assessment()))
        assert time.perf_counter() - started < 0.05
    finally:
        dispatcher.stop(timeout=5)


def test_dispatcher_drops_rather_than_blocking_when_the_queue_is_full():
    dispatcher = AlertDispatcher(AlertsConfig(console=False, queue_size=2), providers=[])
    # Never started, so nothing drains the queue.
    assert dispatcher.dispatch(build_event(assessment()))
    assert dispatcher.dispatch(build_event(assessment()))
    assert not dispatcher.dispatch(build_event(assessment()))
    assert dispatcher.dropped == 1


def test_dispatcher_survives_a_failing_provider():
    class BrokenProvider(AlertProvider):
        name = "broken"

        def send(self, event):
            raise RuntimeError("provider exploded")

    good = RecordingProvider()
    dispatcher = AlertDispatcher(
        AlertsConfig(console=False), providers=[BrokenProvider(), good]
    )
    dispatcher.start()
    try:
        dispatcher.dispatch(build_event(assessment()))
        assert good.ready.wait(timeout=5), "one bad provider must not block the others"
    finally:
        dispatcher.stop(timeout=5)


def test_dispatcher_drains_on_stop():
    """An operator stopping the service must not lose the last alert."""
    provider = RecordingProvider()
    dispatcher = AlertDispatcher(AlertsConfig(console=False), providers=[provider])
    dispatcher.start()
    for _ in range(5):
        dispatcher.dispatch(build_event(assessment()))
    dispatcher.stop(timeout=10)
    assert len(provider.events) == 5


def test_console_provider_renders_the_reasoning():
    provider = ConsoleAlertProvider()
    assert provider.send(SecurityEventModel.from_event(build_event(assessment())))


def test_callback_provider_isolates_exceptions():
    def explode(event):
        raise ValueError("nope")

    provider = CallbackAlertProvider(explode)
    assert provider.send(SecurityEventModel.from_event(build_event(assessment()))) is False


def test_webhook_never_logs_its_token(caplog):
    config = WebhookConfig(
        enabled=True, url="https://hooks.example.com/x?token=SECRET", token="BEARER_SECRET"
    )
    provider = WebhookAlertProvider(config)
    assert "SECRET" not in provider.safe_url
    assert "BEARER_SECRET" not in provider.safe_url


def test_trigger_alert_requires_a_configured_dispatcher():
    """The convenience function must not create hidden global state."""
    set_default_dispatcher(None)
    assert trigger_alert(person_id=5, crop_img=None) is None


def test_trigger_alert_routes_through_the_real_architecture():
    provider = RecordingProvider()
    dispatcher = AlertDispatcher(AlertsConfig(console=False), providers=[provider])
    dispatcher.start()
    set_default_dispatcher(dispatcher)
    try:
        event = trigger_alert(person_id=5, crop_img=np.zeros((10, 10, 3), dtype=np.uint8))
        assert event is not None
        assert provider.ready.wait(timeout=5)

        delivered = provider.events[0]
        assert delivered.person_id == 5
        # A hand-made alert must never be presented as an inference.
        assert any("Manually triggered" in text for text in delivered.positive_evidence)
        assert delivered.notice == INTERPRETATION_NOTICE
    finally:
        dispatcher.stop(timeout=5)
        set_default_dispatcher(None)
