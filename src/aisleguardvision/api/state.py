"""Shared runtime state exposed to the API.

The API is a *read-mostly view* of a running pipeline, not an owner of it. This
module is the single, explicit handoff point: the application registers what it
is willing to expose, and the API layer reads only from here.

Keeping the boundary narrow matters. The API runs on uvicorn's threads while
the pipeline runs on its own, and letting request handlers reach arbitrarily
into pipeline internals would make every future endpoint a potential race.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..camera.manager import CameraManager
from ..core.config import AppConfig
from ..core.metrics import MetricsRegistry, get_metrics
from ..core.types import CameraStatus
from ..events.recorder import IncidentStore


@dataclass
class RuntimeState:
    """What a running AisleGuard Vision process exposes to its API."""

    config: AppConfig
    started_at: float = field(default_factory=time.time)
    camera_manager: CameraManager | None = None
    incident_store: IncidentStore | None = None
    metrics: MetricsRegistry = field(default_factory=get_metrics)
    device_label: str = ""
    item_detection_available: bool = False
    #: camera_id -> live per-camera stats published by the pipeline loop.
    _camera_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    # -- publishing (called from the pipeline thread) ----------------------
    def publish_camera_stats(self, camera_id: str, **stats: float) -> None:
        with self._lock:
            self._camera_stats.setdefault(camera_id, {}).update(stats)

    def camera_stats(self, camera_id: str) -> dict[str, float]:
        with self._lock:
            return dict(self._camera_stats.get(camera_id, {}))

    # -- reading (called from request handlers) ----------------------------
    def statuses(self) -> list[CameraStatus]:
        if self.camera_manager is None:
            return []
        statuses = self.camera_manager.all_status()
        for status in statuses:
            stats = self.camera_stats(status.camera_id)
            status.inference_fps = stats.get("inference_fps", 0.0)
            status.pose_fps = stats.get("pose_fps", 0.0)
            status.active_tracks = int(stats.get("active_tracks", 0))
        return statuses

    def status(self, camera_id: str) -> CameraStatus | None:
        for status in self.statuses():
            if status.camera_id == camera_id:
                return status
        return None

    def zone_count(self, camera_id: str) -> int:
        if self.camera_manager is None:
            return 0
        return len(self.camera_manager.zones(camera_id))

    def health(self) -> dict[str, object]:
        if self.camera_manager is None:
            return {
                "cameras_total": 0,
                "cameras_streaming": 0,
                "cameras_reconnecting": 0,
                "cameras_failed": 0,
                "cameras_stalled": [],
                "status": "starting",
            }
        return self.camera_manager.health_summary()


#: Process-wide state, set during startup by main.py or the API entry point.
_STATE: RuntimeState | None = None


def set_state(state: RuntimeState | None) -> None:
    global _STATE
    _STATE = state


def get_state() -> RuntimeState | None:
    return _STATE
