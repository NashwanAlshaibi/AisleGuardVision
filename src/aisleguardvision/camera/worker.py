"""Camera decode worker.

One thread per camera, doing nothing but decoding. Separating acquisition from
inference is what keeps them from interfering: a stalled RTSP read must not
delay another camera's inference, and a slow inference pass must not cause the
decoder to fall behind and accumulate latency.

A thread (not a process) is the right unit here because OpenCV/FFmpeg release
the GIL during decode, so N decoders genuinely run in parallel. At the 64
camera scale the boundary moves to processes for *failure isolation* rather
than for parallelism -- see ``docs/SCALING.md``.

Overload policy is explicit: **drop stale frames, never accumulate latency.**
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from ..core.config import CameraConfig
from ..core.logging import get_logger
from ..core.metrics import MetricNames, MetricsRegistry, get_metrics
from ..core.types import CameraState, CameraStatus, Frame
from ..utils.sanitize import sanitize_source
from .frame_buffer import CircularFrameBuffer, FrameQueue
from .stream import VideoStream

logger = get_logger(__name__)


class CameraDecodeWorker:
    """Decodes one camera into a bounded queue."""

    def __init__(
        self,
        config: CameraConfig,
        queue: FrameQueue | None = None,
        buffer: CircularFrameBuffer | None = None,
        metrics: MetricsRegistry | None = None,
        on_state_change: Callable[[str, CameraState], None] | None = None,
    ) -> None:
        self.config = config
        self.camera_id = config.id
        self.queue = queue or FrameQueue(config.queue_size)
        self.buffer = buffer
        self.metrics = metrics or get_metrics()
        self._on_state_change = on_state_change

        self._stream = VideoStream(config)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._enabled = threading.Event()
        if config.enabled:
            self._enabled.set()

        self._frame_id = 0
        self._state = CameraState.IDLE
        self._decode_rate = self.metrics.rate(MetricNames.DECODE_FPS, {"camera_id": self.camera_id})
        self._labels = {"camera_id": self.camera_id}
        #: Minimum interval between delivered frames when the camera is rate-capped.
        self._min_interval = 1.0 / config.target_decode_fps if config.target_decode_fps > 0 else 0.0
        self._last_delivered = 0.0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"decode-{self.camera_id}", daemon=True
        )
        self._thread.start()
        logger.info(
            "decode worker started",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "source": sanitize_source(self.config.source),
                    "queue_size": self.queue.maxsize,
                }
            },
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self.queue.close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():  # pragma: no cover - only under a hung driver
                logger.warning(
                    "decode worker did not stop cleanly",
                    extra={"fields": {"camera_id": self.camera_id}},
                )
            self._thread = None
        self._stream.close()
        self._set_state(CameraState.IDLE)
        logger.info("decode worker stopped", extra={"fields": {"camera_id": self.camera_id}})

    def enable(self) -> None:
        self._enabled.set()
        logger.info("camera enabled", extra={"fields": {"camera_id": self.camera_id}})

    def disable(self) -> None:
        self._enabled.clear()
        self._stream.close()
        self.queue.clear()
        self._set_state(CameraState.DISABLED)
        logger.info("camera disabled", extra={"fields": {"camera_id": self.camera_id}})

    @property
    def is_enabled(self) -> bool:
        return self._enabled.is_set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- status ------------------------------------------------------------
    def status(self) -> CameraStatus:
        info = self._stream.info
        return CameraStatus(
            camera_id=self.camera_id,
            name=self.config.name,
            state=self._state,
            enabled=self.is_enabled,
            source_label=sanitize_source(self.config.source),
            last_frame_timestamp=self._stream.last_frame_time,
            frames_received=self._stream.frames_read,
            frames_dropped=self.queue.dropped,
            decode_fps=round(self._decode_rate.value, 2),
            reconnect_count=self._stream.reconnect_count,
            consecutive_failures=self._stream.consecutive_failures,
            last_error=self._stream.last_error,
            resolution=info.resolution,
        )

    # -- main loop ---------------------------------------------------------
    def _run(self) -> None:
        """Decode loop. Every failure mode is handled here, none escapes."""
        while not self._stop.is_set():
            if not self._enabled.is_set():
                self._set_state(CameraState.DISABLED)
                self._stop.wait(0.25)
                continue

            if not self._stream.is_open and not self._try_open():
                # Still waiting out the reconnect backoff.
                continue

            if self._stream.is_stalled():
                logger.warning(
                    "camera stream stalled; forcing reconnect",
                    extra={
                        "fields": {
                            "camera_id": self.camera_id,
                            "stall_timeout": self.config.stall_timeout_seconds,
                        }
                    },
                )
                self.metrics.inc(MetricNames.CAMERA_RECONNECTS, **self._labels)
                self._stream.close()
                self._set_state(CameraState.RECONNECTING)
                continue

            ok, image = self._stream.read()
            if not ok or image is None:
                self._set_state(self._stream.state)
                if self._stream.give_up():
                    self._stream.mark_failed()
                    self._set_state(CameraState.FAILED)
                    self.metrics.inc(MetricNames.CAMERA_FAILURES, **self._labels)
                    return
                # Back off without spinning. The stream schedules its own retry.
                self._stop.wait(min(0.5, self._stream.seconds_until_retry() or 0.05))
                continue

            self._deliver(image)

        self._stream.close()

    def _try_open(self) -> bool:
        if not self._stream.should_retry():
            self._stop.wait(min(0.25, self._stream.seconds_until_retry()))
            return False

        first_attempt = self._stream.reconnect_count == 0 and self._stream.frames_read == 0
        opened = self._stream.open() if first_attempt else self._stream.reconnect()
        if not opened:
            self._set_state(CameraState.RECONNECTING)
            if not first_attempt:
                self.metrics.inc(MetricNames.CAMERA_RECONNECTS, **self._labels)
            if self._stream.give_up():
                self._stream.mark_failed()
                self._set_state(CameraState.FAILED)
                self.metrics.inc(MetricNames.CAMERA_FAILURES, **self._labels)
            return False

        if not first_attempt:
            self.metrics.inc(MetricNames.CAMERA_RECONNECTS, **self._labels)
        self._set_state(CameraState.STREAMING)
        # A file played for display should run at its native rate; decoded flat
        # out it is a flipbook. Batch analysis of recorded footage deliberately
        # does not do this -- there, decoding as fast as possible is the point.
        if (
            self.config.pace_file_source
            and not self._stream.is_live
            and self._min_interval == 0.0
            and self._stream.info.fps > 0
        ):
            self._min_interval = 1.0 / self._stream.info.fps
            logger.info(
                "pacing file source to its native frame rate",
                extra={
                    "fields": {
                        "camera_id": self.camera_id,
                        "fps": round(self._stream.info.fps, 2),
                    }
                },
            )
        return True

    def _deliver(self, image) -> None:
        now = time.time()

        # Rate cap: decode as fast as the source delivers, but only forward at
        # the configured rate. Decoding and then discarding is cheaper than
        # letting the driver buffer build up.
        if self._min_interval > 0 and (now - self._last_delivered) < self._min_interval:
            return
        self._last_delivered = now

        self._frame_id += 1
        frame = Frame(
            camera_id=self.camera_id,
            frame_id=self._frame_id,
            timestamp=now,
            image=image,
        )

        if self.buffer is not None:
            # The buffer copies internally; the queue hands off the original,
            # which is safe because OpenCV allocates a fresh array per read on
            # every backend we support.
            self.buffer.append(frame)

        accepted = self.queue.put(frame)
        self._decode_rate.tick()
        self.metrics.inc(MetricNames.FRAMES_RECEIVED, **self._labels)
        self.metrics.set_gauge(MetricNames.QUEUE_DEPTH, self.queue.depth, **self._labels)
        if not accepted:
            self.metrics.inc(MetricNames.FRAMES_DROPPED, **self._labels)

    def _set_state(self, state: CameraState) -> None:
        if state is self._state:
            return
        previous = self._state
        self._state = state
        logger.info(
            "camera state changed",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "from": previous.value,
                    "to": state.value,
                }
            },
        )
        if self._on_state_change is not None:
            try:
                self._on_state_change(self.camera_id, state)
            except Exception:  # pragma: no cover - callback is user code
                logger.exception("camera state callback failed")
