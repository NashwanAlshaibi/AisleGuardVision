"""Circular pre-event frame buffer and bounded frame queue.

Two bounded structures, both designed around the same principle:

    **For live video, a fresh frame is worth more than an old one.**

:class:`CircularFrameBuffer` keeps a rolling window of recent frames so that
when an incident fires, the seconds *before* it are already in memory. A
reviewer needs to see the approach, not just the aftermath.

:class:`FrameQueue` connects the decoder to the analysis pipeline. It is
bounded and **drops the oldest frame** when full, rather than blocking the
decoder or growing without limit. An unbounded queue does not prevent
overload; it converts an overload into unbounded latency and then an OOM,
which is strictly worse.

Memory is bounded twice over -- by seconds *and* by an absolute frame count --
and buffered frames are downscaled, because retaining 4K at 30 FPS for 5
seconds is roughly 1.2 GB per camera.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ..core.config import RecordingConfig
from ..core.logging import get_logger
from ..core.types import Frame

logger = get_logger(__name__)


@dataclass(slots=True)
class BufferedFrame:
    """One retained frame plus the time it was captured."""

    timestamp: float
    image: np.ndarray
    frame_id: int = 0


class CircularFrameBuffer:
    """Rolling window of recent frames for one camera.

    Thread-safe: the decode worker appends while the incident recorder reads.
    """

    def __init__(
        self,
        seconds: float,
        max_frames: int = 600,
        max_width: int = 1280,
    ) -> None:
        self.seconds = seconds
        self.max_frames = max_frames
        self.max_width = max_width
        self._frames: deque[BufferedFrame] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    @classmethod
    def for_recording(cls, config: RecordingConfig) -> CircularFrameBuffer:
        # The buffer must hold the pre-roll plus a margin for the time it takes
        # to decide an incident happened.
        return cls(
            seconds=config.pre_event_seconds + 2.0,
            max_frames=config.buffer_max_frames,
            max_width=config.buffer_max_width,
        )

    def append(self, frame: Frame) -> None:
        """Add a frame, downscaling and copying it first.

        The copy is deliberate and necessary: OpenCV reuses its decode buffer,
        so retaining the array without copying would leave the entire buffer
        pointing at whatever the camera decoded most recently.
        """
        image = self._prepare(frame.image)
        with self._lock:
            self._frames.append(BufferedFrame(frame.timestamp, image, frame.frame_id))
            self._evict_locked(frame.timestamp)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        if image.shape[1] > self.max_width:
            scale = self.max_width / image.shape[1]
            return cv2.resize(
                image,
                (self.max_width, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return image.copy()

    def _evict_locked(self, now: float) -> None:
        cutoff = now - self.seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def snapshot(
        self, start: float | None = None, end: float | None = None
    ) -> list[BufferedFrame]:
        """Frames within ``[start, end]``, oldest first."""
        with self._lock:
            frames = list(self._frames)
        if start is not None:
            frames = [f for f in frames if f.timestamp >= start]
        if end is not None:
            frames = [f for f in frames if f.timestamp <= end]
        return frames

    def latest(self) -> BufferedFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    @property
    def span_seconds(self) -> float:
        with self._lock:
            if len(self._frames) < 2:
                return 0.0
            return self._frames[-1].timestamp - self._frames[0].timestamp

    @property
    def approximate_bytes(self) -> int:
        with self._lock:
            if not self._frames:
                return 0
            return self._frames[0].image.nbytes * len(self._frames)


class FrameQueue:
    """Bounded, drop-oldest queue between the decoder and the pipeline.

    Deliberately not ``queue.Queue``: the semantics needed here are
    "latest wins", and ``Queue.put`` offers only block-or-fail.
    """

    def __init__(self, maxsize: int = 4) -> None:
        if maxsize < 1:
            raise ValueError("frame queue needs room for at least one frame")
        self.maxsize = maxsize
        self._items: deque[Frame] = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped = 0
        self._received = 0
        self._closed = False

    def put(self, frame: Frame) -> bool:
        """Enqueue a frame, evicting the oldest if full.

        Returns ``True`` if nothing had to be dropped.
        """
        with self._not_empty:
            if self._closed:
                return False
            self._received += 1
            dropped = 0
            while len(self._items) >= self.maxsize:
                self._items.popleft()
                dropped += 1
            if dropped:
                self._dropped += dropped
                frame.dropped_before = dropped
            self._items.append(frame)
            self._not_empty.notify()
            return dropped == 0

    def get(self, timeout: float | None = 1.0) -> Frame | None:
        """Pop the oldest queued frame, or ``None`` on timeout/close."""
        with self._not_empty:
            if not self._items:
                self._not_empty.wait(timeout)
            if not self._items:
                return None
            return self._items.popleft()

    def get_latest(self, timeout: float | None = 1.0) -> Frame | None:
        """Pop the newest frame and discard everything older.

        This is the right call for a display or analysis loop that has fallen
        behind: catching up on stale frames only deepens the lag.
        """
        with self._not_empty:
            if not self._items:
                self._not_empty.wait(timeout)
            if not self._items:
                return None
            skipped = len(self._items) - 1
            frame = self._items.pop()
            if skipped:
                self._dropped += skipped
                frame.dropped_before += skipped
            self._items.clear()
            return frame

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._items.clear()
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def received(self) -> int:
        return self._received

    @property
    def is_closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        return self.depth


def write_clip(
    frames: list[BufferedFrame],
    path: str,
    fps: float,
    codec: str = "mp4v",
) -> bool:
    """Write buffered frames to a video file.

    Returns success. Never raises: a failed clip write must be logged and
    counted, not allowed to interrupt video processing.
    """
    if not frames:
        logger.error("refusing to write an empty clip", extra={"fields": {"path": path}})
        return False

    height, width = frames[0].image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(path, fourcc, max(1.0, fps), (width, height))
    if not writer.isOpened():
        logger.error(
            "could not open video writer; is the codec available in this OpenCV build?",
            extra={"fields": {"path": path, "codec": codec}},
        )
        return False

    written = 0
    try:
        for buffered in frames:
            image = buffered.image
            if image.shape[:2] != (height, width):
                # Resolution changed mid-buffer (camera reconnected at a
                # different profile). Normalize rather than abort the clip.
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(image)
            written += 1
    except Exception as exc:
        logger.error(
            "clip write failed", extra={"fields": {"path": path, "error": str(exc)}}, exc_info=True
        )
        return False
    finally:
        writer.release()

    logger.debug(
        "clip written",
        extra={"fields": {"path": path, "frames": written, "fps": round(fps, 2)}},
    )
    return written > 0


def now() -> float:
    """Wall-clock timestamp used for frame acquisition."""
    return time.time()
