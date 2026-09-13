"""API response models.

Every response is a declared pydantic model, so the schema is generated rather
than implied and a dashboard can be written against it without reading the
server source.

Note what these deliberately never expose: no camera URL (credentials), no
image data, and no identity information of any kind. ``person_id`` is a
temporary per-camera tracking id and is documented as such in the schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import CameraState, ThreatLevel


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="healthy | degraded | starting")
    version: str
    uptime_seconds: float
    cameras_total: int = 0
    cameras_streaming: int = 0
    cameras_reconnecting: int = 0
    cameras_failed: int = 0
    cameras_stalled: list[str] = Field(default_factory=list)
    device: str = ""
    #: False when no merchandise detector is configured on any camera.
    item_detection_available: bool = False
    notice: str = (
        "AisleGuard Vision produces behavioral risk signals for human review. "
        "It does not determine that a theft occurred."
    )


class CameraResponse(BaseModel):
    """Camera status. ``source_label`` is credential-sanitized."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str
    name: str
    state: CameraState
    enabled: bool
    source_label: str = Field(description="Sanitized source; RTSP passwords are never returned")
    resolution: tuple[int, int] | None = None
    decode_fps: float = 0.0
    inference_fps: float = 0.0
    pose_fps: float = 0.0
    frames_received: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    active_tracks: int = 0
    last_frame_timestamp: float | None = None
    last_error: str | None = None
    zone_count: int = 0


class CameraListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cameras: list[CameraResponse]
    count: int


class CameraActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str
    enabled: bool
    state: CameraState
    message: str


class EventSummary(BaseModel):
    """Compact incident record for list views."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    camera_id: str
    person_id: int
    timestamp: float
    timestamp_iso: str
    risk_score: float
    threat_level: ThreatLevel
    behavior_state: str
    event_type: str
    positive_evidence_count: int = 0
    negative_evidence_count: int = 0
    has_clip: bool = False
    has_snapshot: bool = False
    review_status: str = "pending"


class EventListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventSummary]
    count: int
    total_stored: int = 0


class MetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    counters: dict[str, float] = Field(default_factory=dict)
    gauges: dict[str, float] = Field(default_factory=dict)
    rates: dict[str, float] = Field(default_factory=dict)
    histograms: dict[str, dict[str, float]] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detail: str
