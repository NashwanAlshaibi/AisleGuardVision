"""Camera manager.

Owns the set of cameras, their decode workers, their frame buffers and their
zone registries, and exposes health for the API and the overlay.

Its central guarantee: **one camera's failure is that camera's failure.** A
missing RTSP credential, a dead NVR, an unreadable file or a malformed zone
polygon degrades exactly one camera and is reported on that camera's status.
At 64 cameras this is the difference between a nightly annoyance and a nightly
outage.
"""

from __future__ import annotations

import threading
import time

from ..behavior.zones import ZoneRegistry
from ..core.config import AppConfig, CameraConfig
from ..core.logging import get_logger
from ..core.metrics import MetricsRegistry, get_metrics
from ..core.types import CameraState, CameraStatus
from ..utils.sanitize import sanitize_source
from .frame_buffer import CircularFrameBuffer, FrameQueue
from .worker import CameraDecodeWorker

logger = get_logger(__name__)


class CameraManager:
    """Lifecycle and health for every configured camera."""

    def __init__(self, config: AppConfig, metrics: MetricsRegistry | None = None) -> None:
        self.config = config
        self.metrics = metrics or get_metrics()
        self._workers: dict[str, CameraDecodeWorker] = {}
        self._buffers: dict[str, CircularFrameBuffer] = {}
        self._zones: dict[str, ZoneRegistry] = {}
        self._configs: dict[str, CameraConfig] = {}
        self._lock = threading.RLock()

    # -- registration ------------------------------------------------------
    def add_camera(self, camera: CameraConfig, start: bool = False) -> CameraDecodeWorker | None:
        """Register a camera. Returns its worker, or ``None`` if unusable."""
        with self._lock:
            if camera.id in self._workers:
                raise ValueError(f"camera {camera.id!r} is already registered")

            if not camera.is_resolvable():
                # The source is an unexpanded ${VAR}. Skip loudly rather than
                # failing the whole process: a dev box legitimately has
                # credentials for one camera out of sixty-four.
                logger.warning(
                    "camera source is unset (unresolved environment variable); skipping",
                    extra={"fields": {"camera_id": camera.id, "name": camera.name}},
                )
                return None

            buffer = CircularFrameBuffer.for_recording(self.config.recording)
            queue = FrameQueue(camera.queue_size)
            worker = CameraDecodeWorker(
                config=camera,
                queue=queue,
                buffer=buffer,
                metrics=self.metrics,
            )

            self._configs[camera.id] = camera
            self._workers[camera.id] = worker
            self._buffers[camera.id] = buffer
            self._zones[camera.id] = ZoneRegistry.from_config(camera.id, camera.zones)

            logger.info(
                "camera registered",
                extra={
                    "fields": {
                        "camera_id": camera.id,
                        "name": camera.name,
                        "source": sanitize_source(camera.source),
                        "enabled": camera.enabled,
                        "zones": len(self._zones[camera.id]),
                    }
                },
            )
            if start and camera.enabled:
                worker.start()
            return worker

    def load_from_config(self, start: bool = False) -> list[CameraDecodeWorker]:
        """Register every camera in the configuration."""
        workers: list[CameraDecodeWorker] = []
        for camera in self.config.cameras.cameras:
            try:
                worker = self.add_camera(camera, start=start)
            except Exception:
                logger.exception(
                    "failed to register camera; continuing with the rest",
                    extra={"fields": {"camera_id": camera.id}},
                )
                continue
            if worker is not None:
                workers.append(worker)
        logger.info(
            "cameras loaded",
            extra={
                "fields": {
                    "registered": len(workers),
                    "configured": len(self.config.cameras.cameras),
                }
            },
        )
        return workers

    # -- lifecycle ---------------------------------------------------------
    def start_all(self) -> None:
        with self._lock:
            for camera_id, worker in self._workers.items():
                if worker.is_enabled:
                    try:
                        worker.start()
                    except Exception:
                        logger.exception(
                            "failed to start camera worker",
                            extra={"fields": {"camera_id": camera_id}},
                        )

    def stop_all(self, timeout: float = 5.0) -> None:
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            try:
                worker.stop(timeout=timeout)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "error stopping camera worker",
                    extra={"fields": {"camera_id": worker.camera_id}},
                )

    def enable(self, camera_id: str) -> bool:
        worker = self._workers.get(camera_id)
        if worker is None:
            return False
        worker.enable()
        if not worker.is_running:
            worker.start()
        return True

    def disable(self, camera_id: str) -> bool:
        worker = self._workers.get(camera_id)
        if worker is None:
            return False
        worker.disable()
        return True

    # -- accessors ---------------------------------------------------------
    def worker(self, camera_id: str) -> CameraDecodeWorker | None:
        return self._workers.get(camera_id)

    def buffer(self, camera_id: str) -> CircularFrameBuffer | None:
        return self._buffers.get(camera_id)

    def zones(self, camera_id: str) -> ZoneRegistry:
        registry = self._zones.get(camera_id)
        if registry is None:
            registry = ZoneRegistry(camera_id)
            self._zones[camera_id] = registry
        return registry

    def camera_config(self, camera_id: str) -> CameraConfig | None:
        return self._configs.get(camera_id)

    @property
    def camera_ids(self) -> list[str]:
        return list(self._workers)

    @property
    def workers(self) -> list[CameraDecodeWorker]:
        return list(self._workers.values())

    def status(self, camera_id: str) -> CameraStatus | None:
        worker = self._workers.get(camera_id)
        return worker.status() if worker else None

    def all_status(self) -> list[CameraStatus]:
        return [worker.status() for worker in self._workers.values()]

    def health_summary(self) -> dict[str, object]:
        """Aggregate health, used by ``GET /health``."""
        statuses = self.all_status()
        streaming = sum(1 for s in statuses if s.state is CameraState.STREAMING)
        failed = sum(1 for s in statuses if s.state is CameraState.FAILED)
        reconnecting = sum(1 for s in statuses if s.state is CameraState.RECONNECTING)
        now = time.time()
        stalled: list[str] = []
        for status in statuses:
            camera = self._configs.get(status.camera_id)
            timeout = camera.stall_timeout_seconds if camera else 10.0
            if status.is_stalled(now, timeout):
                stalled.append(status.camera_id)
        return {
            "cameras_total": len(statuses),
            "cameras_streaming": streaming,
            "cameras_reconnecting": reconnecting,
            "cameras_failed": failed,
            "cameras_stalled": stalled,
            # "degraded" rather than "unhealthy": the service is still doing
            # useful work on the cameras that are up.
            "status": "healthy" if failed == 0 and streaming == len(statuses) else "degraded",
        }

    def __enter__(self) -> CameraManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop_all()
