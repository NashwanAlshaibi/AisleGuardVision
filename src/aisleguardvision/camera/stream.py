"""Video stream abstraction with reconnect handling.

IP cameras disconnect. They are rebooted by staff, they drop off congested
store Wi-Fi, their NVR restarts, their PoE switch hiccups, they emit corrupt
frames after a transcoder glitch, and they stall while still holding the TCP
connection open so no read ever errors. All of those are *normal operating
conditions*, not exceptional ones, and the failure of one camera must never
affect another or take down the process.

This class handles:

* device index, file path, RTSP and HTTP sources behind one interface
* exponential reconnect backoff with jitter
* stall detection (open socket, no frames)
* corrupt / empty frame rejection
* credential-safe logging at every point

Credentials never reach a log record: every message uses
:func:`~aisleguardvision.utils.sanitize.sanitize_source`.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.config import CameraConfig
from ..core.logging import RateLimitedLogger, get_logger
from ..core.types import CameraState
from ..utils.sanitize import sanitize_source

logger = get_logger(__name__)
_rate_limited = RateLimitedLogger(logger, interval_seconds=15.0)

#: FFmpeg options applied to RTSP captures unless the operator overrides them.
#: TCP transport is far more reliable than UDP on a congested store LAN, and a
#: socket timeout is what turns a silently stalled stream into a reconnect.
DEFAULT_FFMPEG_OPTIONS = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"


@dataclass(slots=True)
class StreamInfo:
    """What the source reported when it opened."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    backend: str = ""
    is_file: bool = False

    @property
    def resolution(self) -> tuple[int, int] | None:
        return (self.width, self.height) if self.width and self.height else None


def resolve_source(source: str | int) -> int | str:
    """Interpret a configured source.

    ``"0"`` / ``0`` become a webcam index; everything else stays a string
    (path or URL). A path is resolved against the working directory so that
    relative sample paths in config behave predictably.
    """
    if isinstance(source, int):
        return source
    text = str(source).strip()
    if text.isdigit():
        return int(text)
    if "://" not in text:
        path = Path(text).expanduser()
        return str(path)
    return text


def is_live_source(source: int | str) -> bool:
    """True for a webcam or a network stream, False for a file."""
    if isinstance(source, int):
        return True
    return "://" in str(source)


class VideoStream:
    """A single video source with automatic recovery.

    Not thread-safe; one instance belongs to one decode worker.
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.camera_id = config.id
        self.source = resolve_source(config.source)
        self.safe_source = sanitize_source(config.source)
        self.is_live = is_live_source(self.source)

        self._capture: cv2.VideoCapture | None = None
        self._info = StreamInfo(is_file=not self.is_live)
        self._state = CameraState.IDLE
        self._last_frame_time: float | None = None
        self._consecutive_failures = 0
        self._reconnect_count = 0
        self._backoff = config.reconnect_initial_seconds
        self._next_attempt_at = 0.0
        self._frames_read = 0
        self._last_error: str | None = None

    # -- accessors ---------------------------------------------------------
    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def info(self) -> StreamInfo:
        return self._info

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def last_frame_time(self) -> float | None:
        return self._last_frame_time

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> bool:
        """Attempt to open the source. Returns success."""
        self.close()
        self._state = CameraState.CONNECTING
        logger.info(
            "opening camera",
            extra={"fields": {"camera_id": self.camera_id, "source": self.safe_source}},
        )

        if isinstance(self.source, str) and "rtsp" in self.source.lower():
            # OpenCV reads FFmpeg options from this variable at capture-creation
            # time, so it must be set before the VideoCapture is constructed.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", DEFAULT_FFMPEG_OPTIONS)

        try:
            capture = (
                cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                if isinstance(self.source, str) and "://" in self.source
                else cv2.VideoCapture(self.source)
            )
        except Exception as exc:  # pragma: no cover - OpenCV build dependent
            self._fail(f"VideoCapture construction failed: {exc}")
            return False

        if not capture.isOpened():
            capture.release()
            self._fail("source could not be opened")
            return False

        self._capture = capture
        self._configure(capture)
        self._info = self._probe(capture)
        self._state = CameraState.STREAMING
        self._consecutive_failures = 0
        self._backoff = self.config.reconnect_initial_seconds
        self._last_error = None
        self._last_frame_time = time.time()

        logger.info(
            "camera connected",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "source": self.safe_source,
                    "resolution": f"{self._info.width}x{self._info.height}",
                    "source_fps": round(self._info.fps, 2),
                    "live": self.is_live,
                }
            },
        )
        return True

    def _configure(self, capture: cv2.VideoCapture) -> None:
        """Apply capture-side settings."""
        if self.is_live:
            # A one-frame driver buffer is what keeps latency low: for live
            # video a fresh frame is worth more than a queued stale one.
            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # pragma: no cover - not all backends support it
                pass
        if self.config.resolution:
            width, height = self.config.resolution
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))

    @staticmethod
    def _probe(capture: cv2.VideoCapture) -> StreamInfo:
        def _get(prop: int, default: float = 0.0) -> float:
            try:
                value = capture.get(prop)
            except Exception:  # pragma: no cover
                return default
            return value if value and value == value else default  # reject NaN

        fps = _get(cv2.CAP_PROP_FPS)
        # Some RTSP sources report nonsense (0, or 180000). Treat anything
        # outside a plausible range as unknown rather than trusting it.
        if not (1.0 <= fps <= 240.0):
            fps = 0.0
        return StreamInfo(
            width=int(_get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(_get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=int(_get(cv2.CAP_PROP_FRAME_COUNT)),
            backend=capture.getBackendName() if hasattr(capture, "getBackendName") else "",
        )

    def close(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover
                pass
            self._capture = None

    # -- reading -----------------------------------------------------------
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame.

        Returns ``(False, None)`` on any failure, having already scheduled a
        reconnect. Callers loop on this; they never see an exception.
        """
        if self._capture is None:
            return False, None

        try:
            ok, frame = self._capture.read()
        except Exception as exc:  # pragma: no cover - driver level
            self._fail(f"read raised: {exc}")
            return False, None

        if not ok or frame is None:
            if not self.is_live and self.config.loop_file_source:
                # End of a looping file source: rewind rather than reconnect.
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
                if ok and frame is not None:
                    return self._accept(frame)
            self._fail("read returned no frame")
            return False, None

        if frame.size == 0 or frame.ndim != 3:
            # Corrupt frame after a transcoder glitch. Drop it and continue;
            # a single bad frame is not a reason to tear down the connection.
            _rate_limited.warning(
                f"corrupt-frame-{self.camera_id}",
                "discarding corrupt frame",
                extra={"fields": {"camera_id": self.camera_id}},
            )
            return False, None

        return self._accept(frame)

    def _accept(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        self._last_frame_time = time.time()
        self._frames_read += 1
        self._consecutive_failures = 0
        if self._state is not CameraState.STREAMING:
            self._state = CameraState.STREAMING
        return True, frame

    def is_stalled(self, now: float | None = None) -> bool:
        """True when the connection is open but no frames are arriving.

        This is the failure mode that plain error handling misses entirely: the
        socket stays up, ``read()`` blocks or returns stale data, and without an
        explicit check the camera looks healthy forever.
        """
        if self._state is not CameraState.STREAMING or self._last_frame_time is None:
            return False
        reference = now if now is not None else time.time()
        return (reference - self._last_frame_time) > self.config.stall_timeout_seconds

    # -- recovery ----------------------------------------------------------
    def _fail(self, reason: str) -> None:
        self._consecutive_failures += 1
        self._last_error = reason
        self._state = CameraState.RECONNECTING
        _rate_limited.warning(
            f"camera-fail-{self.camera_id}",
            "camera read failure",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "source": self.safe_source,
                    "reason": reason,
                    "consecutive_failures": self._consecutive_failures,
                }
            },
        )
        self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        jitter = 1.0 + random.uniform(-self.config.reconnect_jitter, self.config.reconnect_jitter)
        delay = min(self._backoff * jitter, self.config.reconnect_max_seconds)
        self._next_attempt_at = time.monotonic() + delay
        # Exponential growth, capped. Jitter avoids 64 cameras on one NVR all
        # reconnecting in lockstep after a switch reboot.
        self._backoff = min(self._backoff * 2.0, self.config.reconnect_max_seconds)

    def should_retry(self, now: float | None = None) -> bool:
        """Whether the backoff period has elapsed."""
        reference = now if now is not None else time.monotonic()
        return reference >= self._next_attempt_at

    def seconds_until_retry(self, now: float | None = None) -> float:
        reference = now if now is not None else time.monotonic()
        return max(0.0, self._next_attempt_at - reference)

    def reconnect(self) -> bool:
        """Tear down and reopen. Returns success."""
        self._reconnect_count += 1
        logger.info(
            "reconnecting camera",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "source": self.safe_source,
                    "attempt": self._reconnect_count,
                }
            },
        )
        return self.open()

    def give_up(self) -> bool:
        """Whether the failure budget is exhausted.

        Zero means never give up, which is the right default for a fixed store
        camera: it will come back eventually and nobody wants to restart the
        service when it does.
        """
        limit = self.config.max_consecutive_failures
        return limit > 0 and self._consecutive_failures >= limit

    def mark_failed(self) -> None:
        self._state = CameraState.FAILED
        logger.error(
            "camera failed permanently; giving up",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "source": self.safe_source,
                    "failures": self._consecutive_failures,
                    "last_error": self._last_error,
                }
            },
        )

    def __enter__(self) -> VideoStream:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
