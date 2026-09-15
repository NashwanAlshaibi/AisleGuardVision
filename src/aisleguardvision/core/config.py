"""Configuration system.

All tunables live in YAML under ``config/`` and are validated into pydantic
models here. No behavioral constant is hard-coded in the source: if a number
influences a decision, it is a field on one of these models.

Three files are merged into a single :class:`AppConfig`:

``config/app.yaml``
    Behavior, risk weights, recording, storage, alerting, API, logging.
``config/detection.yaml``
    Models, inference, scheduling, tracking.
``config/cameras.yaml``
    Camera definitions and their zones.

Values of the form ``${VAR}`` or ``${VAR:-default}`` are resolved from the
process environment at load time. This is how RTSP credentials stay out of the
repository.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .types import EvidenceType, ZoneKind

DEFAULT_CONFIG_DIR = Path(os.environ.get("AISLEGUARD_CONFIG_DIR", "config"))

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed or inconsistent."""


# ---------------------------------------------------------------------------
# Environment substitution
# ---------------------------------------------------------------------------


def expand_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in loaded YAML.

    An unset variable without a default expands to an empty string rather than
    raising, so a config listing 64 cameras stays loadable on a dev box with
    only one camera's credentials exported. Cameras whose source resolves empty
    are reported by :meth:`CameraConfig.is_resolvable` and skipped with a
    warning instead of crashing the process.
    """
    source = os.environ if env is None else env
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return source.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, env) for v in value]
    return value


def load_yaml(path: Path, required: bool = True) -> dict[str, Any]:
    """Load one YAML file with environment expansion applied."""
    if not path.exists():
        if required:
            raise ConfigError(f"configuration file not found: {path}")
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on bad input
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"expected a mapping at the top level of {path}")
    return expand_env(raw)


class _Base(BaseModel):
    """Strict base: an unknown key is a typo, and typos in a safety-relevant
    config should fail loudly rather than silently keep a default."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# detection.yaml
# ---------------------------------------------------------------------------


class ModelsConfig(_Base):
    """Model weights and backend selection."""

    #: Backend implementation key. "ultralytics" today; "tensorrt", "onnx",
    #: "triton" and "deepstream" are the planned additions (see docs/JETSON.md).
    backend: str = "ultralytics"
    detector: str = "yolo11n.pt"
    pose: str = "yolo11n-pose.pt"
    #: Optional custom retail-merchandise model. Empty means "not available";
    #: the pipeline then falls back to zone-driven interaction only.
    product: str = ""
    #: auto | cuda | cuda:0 | mps | cpu
    device: str = "auto"
    #: Half precision. Ignored on devices that do not support it.
    fp16: bool = True
    #: Warm the models with a dummy inference at startup so the first real
    #: frame does not pay CUDA kernel autotuning latency.
    warmup: bool = True


class InferenceConfig(_Base):
    """Model invocation parameters."""

    image_size: int = 640
    person_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    object_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    pose_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    #: Minimum keypoint confidence for a keypoint to be used in reasoning.
    keypoint_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    nms_iou: float = Field(default=0.50, ge=0.0, le=1.0)
    max_detections: int = Field(default=100, ge=1)
    #: Frames from different cameras coalesced into one backend call. 1 for the
    #: single-camera MVP; raised by a multi-camera GPU worker.
    max_batch_size: int = Field(default=1, ge=1)
    #: Max time the batcher waits to fill a batch. Latency/throughput knob.
    batch_timeout_ms: float = Field(default=8.0, ge=0.0)
    #: COCO classes to keep, by name. Person is mandatory; the rest are the
    #: false-positive controls (phone!) and container detection.
    keep_classes: list[str] = Field(
        default_factory=lambda: [
            "person",
            "cell phone",
            "handbag",
            "backpack",
            "suitcase",
            "bottle",
            "cup",
            "book",
        ]
    )


class SchedulerConfig(_Base):
    """Adaptive frame scheduling.

    Decode rate, detection rate and pose rate are three independent budgets.
    This is the single most important knob for multi-camera scaling: 64 cameras
    cannot run every model on every frame.
    """

    detection_fps: float = Field(default=10.0, gt=0.0)
    pose_fps: float = Field(default=8.0, gt=0.0)
    #: Pose rate when nothing interesting is happening on the camera.
    pose_idle_fps: float = Field(default=2.0, gt=0.0)
    #: Pose rate when a person is in a shelf zone / risk is rising.
    pose_active_fps: float = Field(default=10.0, gt=0.0)
    #: Enable demand-driven pose scheduling. When False, pose_fps is constant.
    adaptive_pose: bool = True
    #: Run pose only on people who matter (in/near a zone, or already tracked
    #: in an episode) rather than on every person in frame.
    pose_only_for_relevant_people: bool = True
    #: Distance from a merchandise zone, in body heights, within which a person
    #: is considered "relevant" and gets pose attention.
    pose_relevance_distance_ratio: float = Field(default=1.5, ge=0.0)
    #: Maximum people per frame sent to the pose model, highest priority first.
    max_pose_targets: int = Field(default=8, ge=1)
    #: Reuse the last pose for this long before treating a track as pose-less.
    pose_staleness_seconds: float = Field(default=0.75, gt=0.0)


class ByteTrackConfig(_Base):
    """ByteTrack association parameters (see tracking/person_tracker.py)."""

    #: Detections at or above this go into the high-confidence first pass.
    high_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    #: Detections at or above this join the low-confidence recovery pass.
    low_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    #: IoU needed to match a detection to an existing track.
    match_iou_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    #: IoU for the second (low-confidence) association pass.
    second_match_iou_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    #: Confirmation threshold for reviving a lost track.
    revive_iou_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    #: Consecutive hits before a tentative track is confirmed.
    min_hits: int = Field(default=3, ge=1)
    #: How long a track survives without detections (seconds, not frames --
    #: this is what makes tracking survive brief occlusions identically at
    #: 10 FPS and 30 FPS).
    max_lost_seconds: float = Field(default=1.5, ge=0.0)
    #: Confidence floor for creating a brand-new track.
    new_track_threshold: float = Field(default=0.55, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_thresholds(self) -> ByteTrackConfig:
        if self.low_threshold > self.high_threshold:
            raise ValueError("low_threshold must be <= high_threshold")
        return self


class ItemTrackerConfig(_Base):
    """Merchandise-candidate tracking and the disappearance ladder."""

    match_iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Gate on centre displacement between frames, in item box widths.
    max_center_displacement_ratio: float = Field(default=2.0, gt=0.0)
    #: Detections needed before an item track is considered stable enough to
    #: reason about. Below this, disappearance is scored as tracker noise.
    min_hits_for_stability: int = Field(default=4, ge=1)
    min_age_for_stability_seconds: float = Field(default=0.30, ge=0.0)
    #: Disappearance ladder. VISIBLE -> POSSIBLY_OCCLUDED -> OCCLUDED -> MISSING.
    possibly_occluded_after_seconds: float = Field(default=0.25, ge=0.0)
    occluded_after_seconds: float = Field(default=0.60, ge=0.0)
    #: Time in OCCLUDED before an item may be declared MISSING.
    missing_after_seconds: float = Field(default=1.25, ge=0.0)
    #: Item tracks are deleted this long after going MISSING.
    remove_after_seconds: float = Field(default=20.0, ge=0.0)

    @model_validator(mode="after")
    def _check_ladder(self) -> ItemTrackerConfig:
        if not (
            self.possibly_occluded_after_seconds
            <= self.occluded_after_seconds
            <= self.missing_after_seconds
        ):
            raise ValueError(
                "occlusion ladder must be monotonic: possibly_occluded <= occluded <= missing"
            )
        return self


class AssociationConfig(_Base):
    """Hand/item and pose/person association thresholds.

    Distances are expressed as fractions of the subject's body height so the
    same configuration works for a shopper 3 m from the camera and one 12 m
    away. Avoid fixed pixel thresholds.
    """

    #: Wrist-to-item-centre distance counting as "in hand", in body heights.
    hand_item_distance_ratio: float = Field(default=0.18, gt=0.0)
    #: Distance at which association confidence has decayed to zero.
    hand_item_max_distance_ratio: float = Field(default=0.40, gt=0.0)
    #: Minimum IoU between an item box and a wrist-centred hand box.
    hand_item_min_iou: float = Field(default=0.05, ge=0.0, le=1.0)
    #: Weights of the association evidence channels. Normalized on load.
    weight_distance: float = Field(default=0.40, ge=0.0)
    weight_iou: float = Field(default=0.20, ge=0.0)
    weight_motion: float = Field(default=0.20, ge=0.0)
    weight_persistence: float = Field(default=0.20, ge=0.0)
    #: Association confidence needed to declare a hand/item link.
    min_association_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    #: Continuous seconds of linkage before the association is "stable".
    #: A single frame of overlap is never sufficient.
    min_association_seconds: float = Field(default=0.40, ge=0.0)
    #: Motion-similarity window for trajectory correlation.
    motion_window_seconds: float = Field(default=0.60, gt=0.0)
    #: Pose/person matching: IoU is unreliable for pose boxes, so containment
    #: of keypoints within the person box is the primary signal.
    pose_min_iou: float = Field(default=0.35, ge=0.0, le=1.0)
    pose_min_containment: float = Field(default=0.55, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_distances(self) -> AssociationConfig:
        if self.hand_item_max_distance_ratio <= self.hand_item_distance_ratio:
            raise ValueError("hand_item_max_distance_ratio must exceed hand_item_distance_ratio")
        total = (
            self.weight_distance + self.weight_iou + self.weight_motion + self.weight_persistence
        )
        if total <= 0:
            raise ValueError("association channel weights must sum to a positive value")
        return self

    def normalized_weights(self) -> tuple[float, float, float, float]:
        total = (
            self.weight_distance + self.weight_iou + self.weight_motion + self.weight_persistence
        )
        return (
            self.weight_distance / total,
            self.weight_iou / total,
            self.weight_motion / total,
            self.weight_persistence / total,
        )


class TrackingConfig(_Base):
    person: ByteTrackConfig = Field(default_factory=ByteTrackConfig)
    item: ItemTrackerConfig = Field(default_factory=ItemTrackerConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)


class DetectionConfig(_Base):
    """Root of ``config/detection.yaml``."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


# ---------------------------------------------------------------------------
# app.yaml: behavior + risk
# ---------------------------------------------------------------------------


class StorageRegionConfig(_Base):
    """Geometry of pose-derived storage-region estimates.

    All values are fractions of the subject's body height / shoulder width, so
    regions scale with apparent size instead of using fixed pixel offsets.
    """

    #: Radius of a waist/pocket region, in body heights.
    waist_radius_ratio: float = Field(default=0.10, gt=0.0)
    #: Radius of the torso / jacket region, in body heights. Smaller than it
    #: "should" be anatomically, because hands cross the chest constantly and a
    #: generous torso region turns every gesture into evidence.
    torso_radius_ratio: float = Field(default=0.13, gt=0.0)
    #: Lateral offset of the side-waist regions, in shoulder widths.
    waist_lateral_offset_ratio: float = Field(default=0.55, ge=0.0)
    #: Vertical offset of the jacket region above the hip line, in body heights.
    jacket_offset_ratio: float = Field(default=0.12, ge=0.0)
    #: Radius of an inferred bag/backpack region, in body heights.
    bag_radius_ratio: float = Field(default=0.14, gt=0.0)
    #: Wrist must be within this multiple of a region radius to count as "at"
    #: the region.
    approach_radius_multiplier: float = Field(default=1.25, ge=1.0)


class BehaviorConfig(_Base):
    """Temporal behavior engine parameters."""

    #: Rolling reasoning window. Volatile evidence expires after this.
    temporal_window_seconds: float = Field(default=4.0, gt=0.0)
    #: Risk at or above this creates a reviewable event.
    alert_threshold: float = Field(default=85.0, ge=0.0, le=100.0)
    #: Time an item must stay unobserved before it is called MISSING. Mirrors
    #: tracking.item.missing_after_seconds and overrides it when set.
    item_missing_timeout_seconds: float = Field(default=1.25, ge=0.0)
    #: A person must be tracked this long before any evidence is scored, so
    #: that a one-frame tracker artifact can never produce an alert.
    min_person_track_seconds: float = Field(default=1.0, ge=0.0)
    #: Continuous hand/item association required before it counts.
    min_item_association_seconds: float = Field(default=0.40, ge=0.0)
    #: Per-person alert cooldown.
    cooldown_seconds: float = Field(default=30.0, ge=0.0)
    #: Within cooldown, a second alert still fires if risk rises by this much.
    cooldown_escalation_delta: float = Field(default=10.0, ge=0.0)
    #: Continuous seconds a wrist must satisfy shelf-zone criteria. One frame
    #: of overlap is explicitly insufficient.
    min_shelf_interaction_seconds: float = Field(default=0.35, ge=0.0)
    #: Wrist-to-zone distance, in body heights, counting as "approaching".
    #: Kept tight: an arm hanging at the side is often within a fifth of a body
    #: height of a shelf face the shopper is simply standing next to.
    shelf_approach_distance_ratio: float = Field(default=0.12, ge=0.0)
    #: Overlaps shorter than this are scored as accidental.
    accidental_overlap_seconds: float = Field(default=0.20, ge=0.0)
    #: Pose torso confidence below this raises LOW_POSE_CONFIDENCE.
    low_pose_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    #: Person track confidence below this raises LOW_PERSON_TRACK_CONFIDENCE.
    low_track_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    #: Wrist displacement toward a storage region, in body heights, needed for
    #: the "hand moving to storage" transition.
    storage_approach_travel_ratio: float = Field(default=0.12, ge=0.0)
    #: Continuous seconds the wrist must REMAIN at a storage region before the
    #: approach counts. Without this, a hand sweeping across the body on its
    #: way somewhere else clips the torso region and reads as a concealment
    #: approach -- which is one of the largest false-positive sources there is.
    storage_dwell_seconds: float = Field(default=0.50, ge=0.0)
    #: Downward/inward wrist velocity, in body heights per second, counting as
    #: a concealment motion profile.
    concealment_motion_speed_ratio: float = Field(default=0.25, ge=0.0)
    #: Window used for the concealment *motion* check. Much shorter than the
    #: temporal window: a deliberate hand movement takes well under a second,
    #: and averaging it over 4 s would dilute it below any useful threshold.
    motion_window_seconds: float = Field(default=1.0, gt=0.0)
    #: How long an episode survives with no new evidence before it resets.
    episode_timeout_seconds: float = Field(default=12.0, gt=0.0)
    #: Grace period after a benign terminal state before a new episode may start.
    benign_reset_seconds: float = Field(default=2.0, ge=0.0)
    #: Ceiling on risk when no item detector is available (zone-only mode).
    #: Deliberately below alert_threshold: zone geometry plus wrist kinematics
    #: is NOT sufficient evidence to ask a human to review someone.
    zone_only_risk_ceiling: float = Field(default=59.0, ge=0.0, le=100.0)
    storage_regions: StorageRegionConfig = Field(default_factory=StorageRegionConfig)

    @model_validator(mode="after")
    def _check(self) -> BehaviorConfig:
        if self.zone_only_risk_ceiling >= self.alert_threshold:
            raise ValueError(
                "zone_only_risk_ceiling must stay below alert_threshold: without item "
                "detections the system must not be able to raise an alert"
            )
        return self


def _default_risk_weights() -> dict[EvidenceType, float]:
    """Shipped risk weights.

    Positive weights are deliberately small relative to the negative ones: no
    single positive observation can reach the alert threshold on its own, and
    any one strong benign explanation (basket, return, phone) wipes out a full
    positive sequence.
    """
    return {
        # -- positive ------------------------------------------------------
        EvidenceType.SHELF_INTERACTION: 10.0,
        EvidenceType.HIGH_VALUE_ZONE_INTERACTION: 5.0,
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION: 15.0,
        EvidenceType.ITEM_REMOVED_FROM_SHELF: 10.0,
        EvidenceType.HAND_MOVED_TO_STORAGE_REGION: 15.0,
        EvidenceType.CONCEALMENT_MOTION_PROFILE: 10.0,
        EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE: 25.0,
        EvidenceType.ITEM_REMAINS_MISSING: 20.0,
        # Absence-of-benign-explanation evidence is recorded for the reviewer
        # but scores zero: "nothing was observed" must never add risk on its
        # own, only document what the engine checked.
        EvidenceType.NO_BASKET_PLACEMENT: 0.0,
        EvidenceType.NO_SHELF_RETURN: 0.0,
        # -- negative ------------------------------------------------------
        EvidenceType.ITEM_RETURNED_TO_SHELF: -60.0,
        EvidenceType.ITEM_VISIBLE_IN_HAND: -30.0,
        EvidenceType.ITEM_PLACED_IN_BASKET: -70.0,
        EvidenceType.ITEM_PLACED_IN_CART: -70.0,
        EvidenceType.NORMAL_PHONE_INTERACTION: -70.0,
        EvidenceType.PERSONAL_EFFECT_INTERACTION: -40.0,
        EvidenceType.TEMPORARY_OCCLUSION: -20.0,
        EvidenceType.UNSTABLE_ITEM_TRACK: -25.0,
        EvidenceType.LOW_POSE_CONFIDENCE: -20.0,
        EvidenceType.LOW_PERSON_TRACK_CONFIDENCE: -25.0,
        EvidenceType.CAMERA_OCCLUSION: -20.0,
        EvidenceType.SHORT_ACCIDENTAL_OVERLAP: -15.0,
        EvidenceType.NO_MERCHANDISE_INTERACTION: -40.0,
        EvidenceType.ITEM_DETECTION_UNAVAILABLE: 0.0,
    }


class RiskConfig(_Base):
    """Explainable risk weights and thresholds.

    The final score is a clamped sum of ``weight * confidence`` over the
    evidence active in the current episode. There is no learned
    "shoplifting probability" anywhere in this system.
    """

    weights: dict[EvidenceType, float] = Field(default_factory=_default_risk_weights)
    #: Threat level boundaries. 0-29 LOW, 30-59 ELEVATED, 60-84 REVIEW, 85+ HIGH.
    elevated_threshold: float = Field(default=30.0, ge=0.0, le=100.0)
    review_threshold: float = Field(default=60.0, ge=0.0, le=100.0)
    high_risk_threshold: float = Field(default=85.0, ge=0.0, le=100.0)

    @field_validator("weights", mode="before")
    @classmethod
    def _coerce_weight_keys(cls, value: Any) -> Any:
        """Merge YAML-supplied weights *over the defaults* rather than
        replacing them, so an operator can retune one weight in app.yaml
        without having to restate all of them."""
        if not isinstance(value, dict):
            return value
        merged: dict[EvidenceType, float] = dict(_default_risk_weights())
        for key, weight in value.items():
            try:
                merged[EvidenceType(key)] = float(weight)
            except ValueError as exc:
                raise ValueError(f"unknown evidence type in risk weights: {key!r}") from exc
        return merged

    @model_validator(mode="after")
    def _check_thresholds(self) -> RiskConfig:
        if not (self.elevated_threshold < self.review_threshold < self.high_risk_threshold):
            raise ValueError("risk thresholds must be strictly increasing")
        for evidence in EvidenceType:
            self.weights.setdefault(evidence, 0.0)
        # Sanity: positive evidence must not carry a negative weight and vice
        # versa, or the explanation text ("positive evidence") would lie.
        for evidence, weight in self.weights.items():
            if evidence.is_negative and weight > 0:
                raise ValueError(f"{evidence.value} is negative evidence but has weight {weight}")
            if not evidence.is_negative and weight < 0:
                raise ValueError(f"{evidence.value} is positive evidence but has weight {weight}")
        return self

    def weight_for(self, evidence: EvidenceType) -> float:
        return self.weights.get(evidence, 0.0)


# ---------------------------------------------------------------------------
# app.yaml: recording, storage, alerts, api, logging
# ---------------------------------------------------------------------------


class RecordingConfig(_Base):
    """Circular buffer and incident clip settings."""

    enabled: bool = True
    pre_event_seconds: float = Field(default=5.0, ge=0.0)
    post_event_seconds: float = Field(default=5.0, ge=0.0)
    #: A clip shorter than this is not worth a reviewer's time.
    min_clip_seconds: float = Field(default=2.0, ge=0.0)
    #: Frames per second written into incident clips.
    clip_fps: float = Field(default=15.0, gt=0.0)
    #: FourCC code. mp4v is the portable default; use avc1/h264 where the
    #: local OpenCV build has an H.264 encoder.
    codec: str = "mp4v"
    #: Longest edge of buffered frames. Buffering full 4K costs ~25 MB/s/camera.
    buffer_max_width: int = Field(default=1280, ge=160)
    #: Hard cap on buffered frames per camera, independent of seconds, so a
    #: misconfigured FPS can never exhaust memory.
    buffer_max_frames: int = Field(default=600, ge=1)
    save_snapshot: bool = True
    #: Draw the overlay (boxes, skeleton, evidence) into saved media.
    annotate_media: bool = True


class RetentionConfig(_Base):
    """Incident retention. The MVP persists incidents only, never raw video."""

    enabled: bool = False
    max_age_days: int = Field(default=30, ge=1)
    max_total_gigabytes: float = Field(default=50.0, gt=0.0)
    #: How often the retention sweep runs.
    sweep_interval_seconds: float = Field(default=3600.0, gt=0.0)


class StorageConfig(_Base):
    incident_directory: Path = Path("data/incidents")
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


class WebhookConfig(_Base):
    enabled: bool = False
    url: str = ""
    #: Bearer token. Always sourced from the environment; never logged.
    token: str = ""
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    backoff_base_seconds: float = Field(default=1.0, gt=0.0)
    backoff_max_seconds: float = Field(default=30.0, gt=0.0)
    verify_tls: bool = True

    @model_validator(mode="after")
    def _check(self) -> WebhookConfig:
        if self.enabled and not self.url:
            raise ValueError("webhook is enabled but no url is configured")
        return self


class AlertsConfig(_Base):
    console: bool = True
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    #: Bounded dispatch queue. Delivery never runs on the inference thread; if
    #: the queue fills, alerts are dropped with a logged error rather than
    #: stalling video processing.
    queue_size: int = Field(default=256, ge=1)
    #: Also dispatch below-threshold observations at REVIEW level, for tuning.
    dispatch_review_level: bool = False


class ApiConfig(_Base):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    #: Shared secret required in the X-API-Key header. Empty disables auth,
    #: which is only acceptable on a trusted loopback interface.
    api_key: str = ""
    cors_origins: list[str] = Field(default_factory=list)
    #: Events returned by /events without an explicit limit.
    default_event_limit: int = Field(default=50, ge=1, le=1000)


class LoggingConfig(_Base):
    level: str = "INFO"
    #: text | json. Use json when shipping to a log aggregator.
    format: str = "text"
    file: Path | None = None
    #: Seconds between periodic pipeline statistics lines.
    stats_interval_seconds: float = Field(default=30.0, gt=0.0)

    @field_validator("level")
    @classmethod
    def _upper(cls, value: str) -> str:
        level = value.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {value}")
        return level

    @field_validator("format")
    @classmethod
    def _fmt(cls, value: str) -> str:
        fmt = value.lower()
        if fmt not in {"text", "json"}:
            raise ValueError("log format must be 'text' or 'json'")
        return fmt


class MetricsConfig(_Base):
    enabled: bool = True
    #: Window over which rate meters (FPS) are averaged.
    rate_window_seconds: float = Field(default=5.0, gt=0.0)
    #: Number of samples retained per latency histogram.
    histogram_size: int = Field(default=512, ge=16)


# ---------------------------------------------------------------------------
# cameras.yaml
# ---------------------------------------------------------------------------


class ZoneConfig(_Base):
    """A polygonal region of interest."""

    id: str
    polygon: list[tuple[float, float]]
    kind: ZoneKind = ZoneKind.SHELF
    name: str = ""
    camera_id: str = ""
    risk_multiplier: float = Field(default=1.0, gt=0.0)
    enabled: bool = True

    @field_validator("polygon")
    @classmethod
    def _check_polygon(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(value) < 3:
            raise ValueError("a zone polygon needs at least 3 vertices")
        return value


class CameraConfig(_Base):
    """One camera definition.

    ``source`` accepts a device index (``0``), a file path, or an RTSP/HTTP
    URL. In ``config/cameras.yaml`` it is written as ``${CAM_XXX_RTSP}`` and
    resolved from the environment, so credentials never enter the repository.
    """

    id: str
    name: str = ""
    source: str = ""
    enabled: bool = True
    #: Decode rate cap. 0 means "as fast as the stream delivers".
    target_decode_fps: float = Field(default=0.0, ge=0.0)
    #: Per-camera override of the global detection rate. 0 uses the global.
    target_inference_fps: float = Field(default=0.0, ge=0.0)
    #: Optional (width, height) to request from the device / resize to.
    resolution: tuple[int, int] | None = None
    zones: list[ZoneConfig] = Field(default_factory=list)
    #: Seconds without a frame before the stream is considered stalled.
    stall_timeout_seconds: float = Field(default=10.0, gt=0.0)
    #: Bounded decode queue. Small on purpose: for live video a fresh frame is
    #: worth more than a backlog of stale ones.
    queue_size: int = Field(default=4, ge=1)
    #: Reconnect backoff.
    reconnect_initial_seconds: float = Field(default=1.0, gt=0.0)
    reconnect_max_seconds: float = Field(default=60.0, gt=0.0)
    reconnect_jitter: float = Field(default=0.25, ge=0.0, le=1.0)
    #: Give up after this many consecutive failures. 0 means never give up,
    #: which is the right default for a fixed store camera.
    max_consecutive_failures: int = Field(default=0, ge=0)
    #: Loop file sources. Useful for demos; ignored for live streams.
    loop_file_source: bool = False
    #: Play a file source at its native frame rate instead of as fast as it
    #: decodes. Wanted for a display demo, NOT for batch analysis of recorded
    #: footage, where decoding flat out is the point.
    pace_file_source: bool = False

    @model_validator(mode="after")
    def _default_name(self) -> CameraConfig:
        if not self.name:
            object.__setattr__(self, "name", self.id)
        for zone in self.zones:
            if not zone.camera_id:
                zone.camera_id = self.id
        return self

    def is_resolvable(self) -> bool:
        """False when the source is an unresolved ``${VAR}`` (empty string)."""
        return bool(str(self.source).strip())


class CamerasConfig(_Base):
    cameras: list[CameraConfig] = Field(default_factory=list)
    #: Zones may also be declared globally, each naming its camera_id. Useful
    #: when zone calibration is generated by a tool and kept in its own file.
    zones: list[ZoneConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _merge_global_zones(self) -> CamerasConfig:
        by_id = {cam.id: cam for cam in self.cameras}
        for zone in self.zones:
            if not zone.camera_id:
                raise ValueError(f"global zone {zone.id!r} must declare a camera_id")
            camera = by_id.get(zone.camera_id)
            if camera is None:
                raise ValueError(f"zone {zone.id!r} references unknown camera {zone.camera_id!r}")
            camera.zones.append(zone)
        duplicates = [cid for cid in by_id if [c.id for c in self.cameras].count(cid) > 1]
        if duplicates:
            raise ValueError(f"duplicate camera ids: {sorted(set(duplicates))}")
        return self


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class AppConfig(_Base):
    """Fully validated application configuration."""

    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    cameras: CamerasConfig = Field(default_factory=CamerasConfig)

    @model_validator(mode="after")
    def _cross_section(self) -> AppConfig:
        """Reconcile knobs that appear in more than one section.

        The operator-facing values in ``behavior:`` win; the tracker-level
        values are derived from them so the two can never disagree at runtime.
        Written with ``object.__setattr__`` because these models validate on
        assignment and an intermediate state would transiently violate the
        occlusion-ladder invariant.
        """
        item = self.detection.tracking.item
        timeout = self.behavior.item_missing_timeout_seconds
        occluded = min(item.occluded_after_seconds, timeout)
        possibly = min(item.possibly_occluded_after_seconds, occluded)
        object.__setattr__(item, "possibly_occluded_after_seconds", possibly)
        object.__setattr__(item, "occluded_after_seconds", occluded)
        object.__setattr__(item, "missing_after_seconds", timeout)

        object.__setattr__(
            self.detection.tracking.association,
            "min_association_seconds",
            self.behavior.min_item_association_seconds,
        )
        object.__setattr__(self.risk, "high_risk_threshold", self.behavior.alert_threshold)
        if not (
            self.risk.elevated_threshold
            < self.risk.review_threshold
            < self.risk.high_risk_threshold
        ):
            raise ValueError(
                "behavior.alert_threshold must stay above risk.review_threshold "
                f"({self.behavior.alert_threshold} vs {self.risk.review_threshold})"
            )
        return self

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(
        cls,
        config_dir: Path | str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AppConfig:
        """Load and merge the three YAML files under ``config_dir``.

        Missing files fall back to the validated defaults, so the system runs
        out of the box on a fresh clone.
        """
        directory = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
        app_data = load_yaml(directory / "app.yaml", required=False)
        detection_data = load_yaml(directory / "detection.yaml", required=False)
        cameras_data = load_yaml(directory / "cameras.yaml", required=False)

        merged: dict[str, Any] = dict(app_data)
        if detection_data:
            merged["detection"] = deep_merge(merged.get("detection", {}), detection_data)
        if cameras_data:
            merged["cameras"] = deep_merge(merged.get("cameras", {}), cameras_data)
        if overrides:
            merged = deep_merge(merged, overrides)

        try:
            return cls.model_validate(merged)
        except Exception as exc:  # pydantic ValidationError
            raise ConfigError(f"invalid configuration in {directory}: {exc}") from exc

    def camera(self, camera_id: str) -> CameraConfig | None:
        for cam in self.cameras.cameras:
            if cam.id == camera_id:
                return cam
        return None

    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras.cameras if c.enabled and c.is_resolvable()]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict.

    Lists are replaced wholesale, not concatenated: half-overriding a camera
    list would be far more surprising than replacing it.
    """
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def load_config(
    config_dir: Path | str | None = None, overrides: dict[str, Any] | None = None
) -> AppConfig:
    """Convenience wrapper around :meth:`AppConfig.load`."""
    return AppConfig.load(config_dir, overrides)
