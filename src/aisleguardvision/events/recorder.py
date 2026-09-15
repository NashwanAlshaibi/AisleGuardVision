"""Incident recording.

When an event is raised, three artifacts are persisted:

    data/incidents/<camera_id>/<YYYY-MM-DD>/<event_uuid>.mp4    clip
    data/incidents/<camera_id>/<YYYY-MM-DD>/<event_uuid>.jpg    snapshot
    data/incidents/<camera_id>/<YYYY-MM-DD>/<event_uuid>.json   metadata

Two design points matter here.

**Media writing never happens on the analysis thread.** Encoding ten seconds of
video takes far longer than a frame interval; doing it inline would stall the
camera and drop the very footage being recorded. A background writer thread
owns all encoding, and the JSON record is written immediately so the incident
exists even if encoding later fails.

**Only incidents are persisted, never the continuous stream.** A 64-camera
store retaining raw video would write on the order of a terabyte a day. The
rolling buffer lives in memory and is discarded unless something happens.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from ..camera.frame_buffer import BufferedFrame, CircularFrameBuffer, write_clip
from ..core.config import RecordingConfig, StorageConfig
from ..core.logging import get_logger
from ..core.metrics import MetricNames, MetricsRegistry, get_metrics
from ..core.types import SecurityEvent
from .models import SecurityEventModel

logger = get_logger(__name__)


class IncidentRecorder:
    """Persists incident media and metadata."""

    def __init__(
        self,
        recording: RecordingConfig,
        storage: StorageConfig,
        metrics: MetricsRegistry | None = None,
        annotator: Callable[[np.ndarray, SecurityEvent], np.ndarray] | None = None,
    ) -> None:
        self.recording = recording
        self.storage = storage
        self.metrics = metrics or get_metrics()
        #: Optional overlay renderer, injected by the pipeline so this module
        #: does not need to know about visualization.
        self.annotator = annotator

        self.root = Path(storage.incident_directory)
        self.root.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[tuple | None] = queue.Queue(maxsize=32)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending: dict[str, threading.Event] = {}
        self._pending_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._writer_loop, name="incident-writer", daemon=True
        )
        self._thread.start()
        logger.info(
            "incident recorder started",
            extra={
                "fields": {
                    "directory": str(self.root),
                    "pre_event_s": self.recording.pre_event_seconds,
                    "post_event_s": self.recording.post_event_seconds,
                }
            },
        )

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the writer, draining any queued clips first.

        The drain matters: an operator stopping the service must not lose the
        incident that was being written at the time.
        """
        if self._thread is None:
            return
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover
            pass
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.warning("incident writer did not drain within the timeout")
        self._thread = None

    # -- recording ---------------------------------------------------------
    def record(
        self,
        event: SecurityEvent,
        buffer: CircularFrameBuffer | None,
        snapshot: np.ndarray | None = None,
    ) -> SecurityEvent:
        """Persist an incident.

        The JSON metadata is written synchronously (it is tiny and must not be
        lost); the snapshot and clip are handed to the writer thread. Paths are
        filled in on the event immediately so that the webhook payload can
        reference media that is still being encoded.
        """
        directory = self._directory_for(event)
        directory.mkdir(parents=True, exist_ok=True)

        base = directory / event.event_id
        if self.recording.save_snapshot and snapshot is not None:
            event.snapshot_path = str(base.with_suffix(".jpg"))
        if self.recording.enabled and buffer is not None:
            event.clip_path = str(base.with_suffix(".mp4"))
        event.metadata_path = str(base.with_suffix(".json"))

        self._write_metadata(event)

        if snapshot is not None or buffer is not None:
            done = threading.Event()
            with self._pending_lock:
                self._pending[event.event_id] = done
            try:
                self._queue.put_nowait((event, buffer, snapshot, done))
            except queue.Full:
                # Backed-up writer: drop the media, keep the incident. A record
                # with no clip is far better than blocking the analysis thread.
                with self._pending_lock:
                    self._pending.pop(event.event_id, None)
                self.metrics.inc(MetricNames.CLIP_WRITE_ERRORS, camera_id=event.camera_id)
                logger.error(
                    "incident media queue is full; incident metadata saved without media",
                    extra={"fields": {"event_id": event.event_id, "camera_id": event.camera_id}},
                )
                event.clip_path = None
                event.snapshot_path = None
                self._write_metadata(event)

        self.metrics.inc(MetricNames.EVENTS_CREATED, camera_id=event.camera_id)
        logger.info(
            "incident recorded",
            extra={
                "fields": {
                    "event_id": event.event_id,
                    "camera_id": event.camera_id,
                    "person_id": event.person_id,
                    "risk": event.risk_score,
                    "threat": event.threat_level.value,
                    "directory": str(directory),
                }
            },
        )
        return event

    def wait_for_media(self, event_id: str, timeout: float = 30.0) -> bool:
        """Block until an event's media has been written. For tests and CLI."""
        with self._pending_lock:
            done = self._pending.get(event_id)
        if done is None:
            return True
        return done.wait(timeout)

    # -- internals ---------------------------------------------------------
    def _directory_for(self, event: SecurityEvent) -> Path:
        day = datetime.fromtimestamp(event.timestamp, UTC).strftime("%Y-%m-%d")
        return self.root / event.camera_id / day

    def _write_metadata(self, event: SecurityEvent) -> None:
        path = Path(event.metadata_path) if event.metadata_path else None
        if path is None:
            return
        try:
            model = SecurityEventModel.from_event(event)
            path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        except Exception:
            logger.exception(
                "failed to write incident metadata",
                extra={"fields": {"event_id": event.event_id}},
            )

    def _writer_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            if item is None:
                return
            event, buffer, snapshot, done = item
            try:
                self._write_media(event, buffer, snapshot)
            except Exception:
                self.metrics.inc(MetricNames.CLIP_WRITE_ERRORS, camera_id=event.camera_id)
                logger.exception(
                    "incident media write failed",
                    extra={"fields": {"event_id": event.event_id}},
                )
            finally:
                done.set()
                with self._pending_lock:
                    self._pending.pop(event.event_id, None)
                self._queue.task_done()

    def _write_media(
        self,
        event: SecurityEvent,
        buffer: CircularFrameBuffer | None,
        snapshot: np.ndarray | None,
    ) -> None:
        if snapshot is not None and event.snapshot_path:
            image = snapshot
            if self.recording.annotate_media and self.annotator is not None:
                try:
                    image = self.annotator(snapshot.copy(), event)
                except Exception:
                    logger.exception("snapshot annotation failed; saving the raw frame")
                    image = snapshot
            if not cv2.imwrite(event.snapshot_path, image):
                self.metrics.inc(MetricNames.CLIP_WRITE_ERRORS, camera_id=event.camera_id)
                logger.error(
                    "failed to write incident snapshot",
                    extra={"fields": {"path": event.snapshot_path}},
                )

        if buffer is None or not event.clip_path:
            return

        # Wait out the post-roll so the clip shows what happened next, which is
        # usually what tells a reviewer whether the alert was right.
        deadline = event.timestamp + self.recording.post_event_seconds
        while time.time() < deadline and not self._stop.is_set():
            time.sleep(0.05)

        frames = buffer.snapshot(
            start=event.timestamp - self.recording.pre_event_seconds,
            end=event.timestamp + self.recording.post_event_seconds,
        )
        if not frames:
            self.metrics.inc(MetricNames.CLIP_WRITE_ERRORS, camera_id=event.camera_id)
            logger.error(
                "no buffered frames available for incident clip",
                extra={"fields": {"event_id": event.event_id}},
            )
            return

        span = frames[-1].timestamp - frames[0].timestamp
        if span < self.recording.min_clip_seconds:
            logger.warning(
                "incident clip is shorter than the configured minimum",
                extra={
                    "fields": {
                        "event_id": event.event_id,
                        "span_s": round(span, 2),
                        "minimum_s": self.recording.min_clip_seconds,
                    }
                },
            )

        fps = self._effective_fps(frames)
        if write_clip(frames, event.clip_path, fps, self.recording.codec):
            self.metrics.inc(MetricNames.CLIPS_WRITTEN, camera_id=event.camera_id)
        else:
            self.metrics.inc(MetricNames.CLIP_WRITE_ERRORS, camera_id=event.camera_id)

    def _effective_fps(self, frames: list[BufferedFrame]) -> float:
        """Play the clip back at the rate it was captured.

        Using the configured ``clip_fps`` for a camera that actually delivered
        12 FPS would make every incident look like it happened in fast motion,
        which misleads a reviewer about how deliberate a movement was.
        """
        if len(frames) < 2:
            return self.recording.clip_fps
        span = frames[-1].timestamp - frames[0].timestamp
        if span <= 0:
            return self.recording.clip_fps
        measured = (len(frames) - 1) / span
        return float(np.clip(measured, 1.0, 60.0))


class IncidentStore:
    """Read access to recorded incidents, for the API.

    Backed by the filesystem rather than a database: for the MVP the incident
    count is small and a directory of JSON files is trivially inspectable,
    backed up and moved. A database becomes worthwhile at multi-store scale --
    see ``docs/SCALING.md``.
    """

    def __init__(self, storage: StorageConfig) -> None:
        self.root = Path(storage.incident_directory)

    def list_events(
        self,
        camera_id: str | None = None,
        limit: int = 50,
        min_risk: float = 0.0,
    ) -> list[SecurityEventModel]:
        """Most recent incidents first."""
        if not self.root.exists():
            return []
        pattern = f"{camera_id}/*/*.json" if camera_id else "*/*/*.json"
        paths = sorted(self.root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        events: list[SecurityEventModel] = []
        for path in paths:
            event = self._load(path)
            if event is None or event.risk_score < min_risk:
                continue
            events.append(event)
            if len(events) >= limit:
                break
        return events

    def get_event(self, event_id: str) -> SecurityEventModel | None:
        if not self.root.exists():
            return None
        for path in self.root.glob(f"*/*/{event_id}.json"):
            return self._load(path)
        return None

    @staticmethod
    def _load(path: Path) -> SecurityEventModel | None:
        try:
            return SecurityEventModel.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(
                "skipping unreadable incident record", extra={"fields": {"path": str(path)}}
            )
            return None

    def count(self) -> int:
        if not self.root.exists():
            return 0
        return sum(1 for _ in self.root.glob("*/*/*.json"))


def apply_retention(storage: StorageConfig) -> int:
    """Delete incidents past the retention policy. Returns how many were removed.

    Disabled by default: silently deleting evidence is a decision a store's
    policy makes, not one a default should make for them.
    """
    retention = storage.retention
    if not retention.enabled:
        return 0

    root = Path(storage.incident_directory)
    if not root.exists():
        return 0

    now = time.time()
    max_age_seconds = retention.max_age_days * 86400
    records = sorted(root.glob("*/*/*.json"), key=lambda p: p.stat().st_mtime)

    removed = 0
    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    budget = retention.max_total_gigabytes * (1024**3)

    for record in records:
        age = now - record.stat().st_mtime
        over_budget = total_bytes > budget
        if age <= max_age_seconds and not over_budget:
            break
        for suffix in (".json", ".mp4", ".jpg"):
            path = record.with_suffix(suffix)
            if path.exists():
                total_bytes -= path.stat().st_size
                try:
                    path.unlink()
                except OSError:
                    logger.warning(
                        "could not delete expired incident file",
                        extra={"fields": {"path": str(path)}},
                    )
        removed += 1

    if removed:
        logger.info(
            "retention sweep complete",
            extra={"fields": {"removed": removed, "max_age_days": retention.max_age_days}},
        )
    return removed
