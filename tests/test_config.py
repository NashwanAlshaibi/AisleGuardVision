"""Configuration: loading, environment expansion, validation and secrets."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisleguardvision.core.config import (
    AppConfig,
    CameraConfig,
    CamerasConfig,
    ConfigError,
    InferenceConfig,
    ItemTrackerConfig,
    ZoneConfig,
    deep_merge,
    expand_env,
    load_config,
    load_yaml,
)
from aisleguardvision.core.types import ZoneKind
from aisleguardvision.utils.sanitize import sanitize_mapping, sanitize_source, sanitize_url

REPO_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


# ---------------------------------------------------------------------------
# The shipped configuration
# ---------------------------------------------------------------------------


def test_shipped_config_loads():
    """The config committed to the repository must be valid."""
    config = load_config(REPO_CONFIG_DIR)
    assert config.behavior.alert_threshold == 85.0
    assert config.cameras.cameras, "the shipped config should define example cameras"


def test_missing_config_directory_falls_back_to_defaults():
    """A fresh clone with no config must still run."""
    config = load_config("/nonexistent/path/to/config")
    assert config.behavior.alert_threshold == 85.0
    assert config.cameras.cameras == []


def test_defaults_match_the_documented_values():
    config = AppConfig()
    assert config.behavior.temporal_window_seconds == 4.0
    assert config.behavior.item_missing_timeout_seconds == 1.25
    assert config.behavior.cooldown_seconds == 30.0
    assert config.behavior.min_person_track_seconds == 1.0
    assert config.behavior.min_item_association_seconds == 0.4
    assert config.recording.pre_event_seconds == 5.0
    assert config.recording.post_event_seconds == 5.0
    assert config.storage.incident_directory == Path("data/incidents")


# ---------------------------------------------------------------------------
# Environment expansion
# ---------------------------------------------------------------------------


def test_expand_env_substitutes_a_variable():
    assert expand_env("${FOO}", {"FOO": "bar"}) == "bar"


def test_expand_env_supports_a_default():
    assert expand_env("${MISSING:-fallback}", {}) == "fallback"
    assert expand_env("${SET:-fallback}", {"SET": "actual"}) == "actual"


def test_expand_env_unset_without_a_default_becomes_empty():
    """A dev box legitimately has credentials for one camera out of sixty-four;
    the config must still load."""
    assert expand_env("${NOPE}", {}) == ""


def test_expand_env_recurses_into_structures():
    data = {"a": ["${X}", {"b": "${Y}"}], "c": 5}
    assert expand_env(data, {"X": "1", "Y": "2"}) == {"a": ["1", {"b": "2"}], "c": 5}


def test_expand_env_leaves_non_strings_alone():
    assert expand_env(42, {}) == 42
    assert expand_env(None, {}) is None


def test_camera_with_an_unresolved_source_is_detectable():
    camera = CameraConfig(id="cam_x", source="")
    assert not camera.is_resolvable()
    assert CameraConfig(id="cam_y", source="0").is_resolvable()


def test_unresolved_cameras_are_excluded_from_enabled_cameras(monkeypatch):
    monkeypatch.delenv("CAM_001_RTSP", raising=False)
    config = load_config(REPO_CONFIG_DIR)
    assert config.enabled_cameras() == []

    monkeypatch.setenv("CAM_001_RTSP", "rtsp://user:secret@10.0.0.5/stream")
    resolved = load_config(REPO_CONFIG_DIR)
    assert [c.id for c in resolved.enabled_cameras()] == ["cam_001"]


def test_credentials_come_from_the_environment_not_the_repo(monkeypatch):
    """No committed file may contain a real credential."""
    monkeypatch.setenv("CAM_001_RTSP", "rtsp://operator:hunter2@10.0.0.5:554/stream")
    config = load_config(REPO_CONFIG_DIR)
    camera = config.camera("cam_001")

    assert camera is not None
    assert "hunter2" in camera.source, "the credential should reach the runtime object"

    raw = (REPO_CONFIG_DIR / "cameras.yaml").read_text(encoding="utf-8")
    assert "hunter2" not in raw
    assert "${CAM_001_RTSP}" in raw, "the file must reference the variable, not a value"


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------


def test_sanitize_source_removes_the_password():
    cleaned = sanitize_source("rtsp://operator:hunter2@10.0.0.5:554/Streaming/Channels/101")
    assert "hunter2" not in cleaned
    assert "operator" in cleaned, "the username is operationally useful"
    assert "10.0.0.5" in cleaned


def test_sanitize_source_handles_non_url_sources():
    assert sanitize_source(0) == "webcam:0"
    assert sanitize_source("0") == "webcam:0"
    assert sanitize_source("./data/samples/store.mp4") == "./data/samples/store.mp4"
    assert sanitize_source(None) == ""
    assert sanitize_source("") == ""


def test_sanitize_url_strips_userinfo_and_query():
    cleaned = sanitize_url("https://user:tok@hooks.example.com/path?token=abc123")
    assert "tok" not in cleaned.replace("***", "")
    assert "abc123" not in cleaned
    assert "hooks.example.com" in cleaned


def test_sanitize_mapping_redacts_sensitive_keys():
    cleaned = sanitize_mapping(
        {
            "camera_id": "cam_001",
            "password": "hunter2",
            "token": "abc",
            "nested": {"api_key": "xyz"},
            "url": "rtsp://u:p@host/s",
        }
    )
    assert cleaned["camera_id"] == "cam_001"
    assert cleaned["password"] == "***"
    assert cleaned["token"] == "***"
    assert cleaned["nested"]["api_key"] == "***"
    assert "p@host" not in str(cleaned["url"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_key_is_rejected():
    """A typo in a safety-relevant config must fail loudly, not silently keep
    a default."""
    with pytest.raises(ValidationError):
        InferenceConfig(image_sizee=640)


def test_out_of_range_values_are_rejected():
    with pytest.raises(ValidationError):
        InferenceConfig(person_confidence=1.5)
    with pytest.raises(ValidationError):
        InferenceConfig(max_detections=0)


def test_occlusion_ladder_must_be_monotonic():
    with pytest.raises(ValidationError):
        ItemTrackerConfig(
            possibly_occluded_after_seconds=2.0,
            occluded_after_seconds=1.0,
            missing_after_seconds=0.5,
        )


def test_bytetrack_thresholds_must_be_ordered():
    from aisleguardvision.core.config import ByteTrackConfig

    with pytest.raises(ValidationError):
        ByteTrackConfig(high_threshold=0.2, low_threshold=0.8)


def test_association_distance_bounds_must_be_ordered():
    from aisleguardvision.core.config import AssociationConfig

    with pytest.raises(ValidationError):
        AssociationConfig(hand_item_distance_ratio=0.5, hand_item_max_distance_ratio=0.2)


def test_association_weights_are_normalized():
    from aisleguardvision.core.config import AssociationConfig

    config = AssociationConfig(
        weight_distance=2.0, weight_iou=1.0, weight_motion=1.0, weight_persistence=0.0
    )
    weights = config.normalized_weights()
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.5)


def test_zone_polygon_needs_three_vertices():
    with pytest.raises(ValidationError):
        ZoneConfig(id="z", polygon=[(0, 0), (1, 1)])


def test_zone_kinds_are_validated():
    zone = ZoneConfig(id="z", kind="high_value", polygon=[(0, 0), (1, 0), (1, 1)])
    assert zone.kind is ZoneKind.HIGH_VALUE
    with pytest.raises(ValidationError):
        ZoneConfig(id="z", kind="not_a_kind", polygon=[(0, 0), (1, 0), (1, 1)])


def test_webhook_enabled_without_a_url_is_rejected():
    from aisleguardvision.core.config import WebhookConfig

    with pytest.raises(ValidationError):
        WebhookConfig(enabled=True, url="")


def test_duplicate_camera_ids_are_rejected():
    with pytest.raises(ValidationError):
        CamerasConfig(
            cameras=[CameraConfig(id="dup", source="0"), CameraConfig(id="dup", source="1")]
        )


def test_global_zone_referencing_an_unknown_camera_is_rejected():
    with pytest.raises(ValidationError):
        CamerasConfig(
            cameras=[CameraConfig(id="cam_a", source="0")],
            zones=[ZoneConfig(id="z", camera_id="cam_missing", polygon=[(0, 0), (1, 0), (1, 1)])],
        )


def test_global_zones_are_merged_into_their_camera():
    config = CamerasConfig(
        cameras=[CameraConfig(id="cam_a", source="0")],
        zones=[ZoneConfig(id="z1", camera_id="cam_a", polygon=[(0, 0), (1, 0), (1, 1)])],
    )
    assert [z.id for z in config.cameras[0].zones] == ["z1"]


def test_camera_zones_inherit_the_camera_id():
    camera = CameraConfig(
        id="cam_a", source="0", zones=[ZoneConfig(id="z", polygon=[(0, 0), (1, 0), (1, 1)])]
    )
    assert camera.zones[0].camera_id == "cam_a"


def test_camera_name_defaults_to_its_id():
    assert CameraConfig(id="cam_a", source="0").name == "cam_a"


# ---------------------------------------------------------------------------
# Cross-section reconciliation
# ---------------------------------------------------------------------------


def test_item_missing_timeout_drives_the_tracker_ladder():
    """The operator-facing knob wins; the tracker value is derived from it so
    the two can never disagree at runtime."""
    config = AppConfig.model_validate({"behavior": {"item_missing_timeout_seconds": 2.5}})
    assert config.detection.tracking.item.missing_after_seconds == 2.5


def test_a_short_missing_timeout_compresses_the_ladder_without_breaking_it():
    config = AppConfig.model_validate({"behavior": {"item_missing_timeout_seconds": 0.4}})
    item = config.detection.tracking.item
    assert (
        item.possibly_occluded_after_seconds
        <= item.occluded_after_seconds
        <= item.missing_after_seconds
    )
    assert item.missing_after_seconds == 0.4


def test_association_dwell_is_shared_between_sections():
    config = AppConfig.model_validate({"behavior": {"min_item_association_seconds": 0.75}})
    assert config.detection.tracking.association.min_association_seconds == 0.75


def test_alert_threshold_drives_the_high_risk_band():
    config = AppConfig.model_validate({"behavior": {"alert_threshold": 90}})
    assert config.risk.high_risk_threshold == 90.0


def test_alert_threshold_below_the_review_band_is_rejected():
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {"behavior": {"alert_threshold": 50, "zone_only_risk_ceiling": 40}}
        )


# ---------------------------------------------------------------------------
# Loading mechanics
# ---------------------------------------------------------------------------


def test_deep_merge_merges_nested_mappings():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    overlay = {"a": {"c": 9}, "e": 4}
    assert deep_merge(base, overlay) == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}


def test_deep_merge_replaces_lists_wholesale():
    """Half-overriding a camera list would be far more surprising than
    replacing it."""
    assert deep_merge({"x": [1, 2, 3]}, {"x": [9]}) == {"x": [9]}


def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_load_yaml_missing_optional_file_returns_empty(tmp_path):
    assert load_yaml(tmp_path / "nope.yaml", required=False) == {}


def test_load_yaml_missing_required_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "nope.yaml", required=True)


def test_load_yaml_rejects_a_non_mapping(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(path)


def test_invalid_config_raises_config_error(tmp_path):
    (tmp_path / "app.yaml").write_text("behavior:\n  alert_threshold: 500\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(tmp_path)


def test_overrides_are_applied_on_top(tmp_path):
    config = AppConfig.load(tmp_path, overrides={"behavior": {"cooldown_seconds": 7}})
    assert config.behavior.cooldown_seconds == 7.0


def test_config_dir_can_come_from_the_environment(monkeypatch, tmp_path):
    (tmp_path / "app.yaml").write_text("behavior:\n  cooldown_seconds: 11\n", encoding="utf-8")
    monkeypatch.setenv("AISLEGUARD_CONFIG_DIR", str(tmp_path))
    # DEFAULT_CONFIG_DIR is read at import time, so pass it explicitly here;
    # the environment variable is what main.py's --config-dir default uses.
    assert load_config(os.environ["AISLEGUARD_CONFIG_DIR"]).behavior.cooldown_seconds == 11.0
