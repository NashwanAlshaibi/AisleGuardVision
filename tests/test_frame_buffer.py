"""Circular frame buffer, bounded frame queue and incident clip writing."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from aisleguardvision.camera.frame_buffer import (
    CircularFrameBuffer,
    FrameQueue,
    write_clip,
)
from aisleguardvision.core.config import RecordingConfig
from aisleguardvision.core.types import Frame


def frame(camera_id: str = "cam", frame_id: int = 0, timestamp: float = 0.0, size=(64, 48)):
    width, height = size
    return Frame(
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=timestamp,
        image=np.full((height, width, 3), frame_id % 256, dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Circular buffer
# ---------------------------------------------------------------------------


def test_buffer_retains_recent_frames():
    buffer = CircularFrameBuffer(seconds=5.0, max_frames=100)
    for index in range(10):
        buffer.append(frame(frame_id=index, timestamp=index * 0.1))
    assert len(buffer) == 10


def test_buffer_evicts_frames_older_than_the_window():
    buffer = CircularFrameBuffer(seconds=1.0, max_frames=100)
    for index in range(30):
        buffer.append(frame(frame_id=index, timestamp=index * 0.1))
    assert buffer.span_seconds <= 1.0 + 1e-6
    assert len(buffer) < 30


def test_buffer_respects_the_absolute_frame_cap():
    """A misconfigured frame rate must not be able to exhaust memory."""
    buffer = CircularFrameBuffer(seconds=3600.0, max_frames=10)
    for index in range(100):
        buffer.append(frame(frame_id=index, timestamp=index * 0.01))
    assert len(buffer) == 10


def test_buffer_downscales_large_frames():
    """Retaining 4K at 30 FPS for 5 s is about 1.2 GB per camera."""
    buffer = CircularFrameBuffer(seconds=5.0, max_frames=10, max_width=320)
    buffer.append(frame(size=(1920, 1080)))
    stored = buffer.latest()
    assert stored is not None
    assert stored.image.shape[1] == 320
    # Aspect ratio preserved.
    assert stored.image.shape[0] == pytest.approx(180, abs=2)


def test_buffer_copies_frames():
    """OpenCV reuses its decode buffer; retaining the array without copying
    would leave the whole buffer pointing at the newest frame."""
    buffer = CircularFrameBuffer(seconds=5.0, max_frames=10, max_width=4096)
    original = frame(frame_id=1)
    buffer.append(original)
    original.image[:] = 255

    stored = buffer.latest()
    assert stored is not None
    assert not np.array_equal(stored.image, original.image)


def test_buffer_snapshot_windows_by_time():
    buffer = CircularFrameBuffer(seconds=10.0, max_frames=200)
    for index in range(50):
        buffer.append(frame(frame_id=index, timestamp=index * 0.1))

    window = buffer.snapshot(start=1.0, end=2.0)
    assert window
    assert all(1.0 <= f.timestamp <= 2.0 for f in window)
    assert window == sorted(window, key=lambda f: f.timestamp)


def test_buffer_snapshot_of_an_empty_buffer():
    assert CircularFrameBuffer(seconds=5.0).snapshot() == []
    assert CircularFrameBuffer(seconds=5.0).latest() is None


def test_buffer_is_thread_safe():
    """The decode worker appends while the incident recorder reads."""
    buffer = CircularFrameBuffer(seconds=5.0, max_frames=200)
    errors: list[Exception] = []

    def writer():
        try:
            for index in range(300):
                buffer.append(frame(frame_id=index, timestamp=time.time()))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def reader():
        try:
            for _ in range(300):
                buffer.snapshot()
                buffer.latest()
                len(buffer)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors


def test_buffer_sized_from_recording_config():
    config = RecordingConfig(pre_event_seconds=5.0, buffer_max_frames=123, buffer_max_width=456)
    buffer = CircularFrameBuffer.for_recording(config)
    # Pre-roll plus a margin for the time it takes to decide an incident happened.
    assert buffer.seconds >= config.pre_event_seconds
    assert buffer.max_frames == 123
    assert buffer.max_width == 456


# ---------------------------------------------------------------------------
# Bounded queue
# ---------------------------------------------------------------------------


def test_queue_delivers_in_order():
    queue = FrameQueue(maxsize=4)
    for index in range(3):
        queue.put(frame(frame_id=index, timestamp=index))
    assert [queue.get(timeout=0.1).frame_id for _ in range(3)] == [0, 1, 2]


def test_queue_drops_the_oldest_when_full():
    """An unbounded queue does not prevent overload, it converts an overload
    into unbounded latency and then an OOM."""
    queue = FrameQueue(maxsize=2)
    for index in range(5):
        queue.put(frame(frame_id=index, timestamp=index))

    assert queue.depth == 2
    assert queue.dropped == 3
    # The frames retained are the NEWEST ones.
    assert queue.get(timeout=0.1).frame_id == 3
    assert queue.get(timeout=0.1).frame_id == 4


def test_queue_put_reports_whether_it_dropped():
    queue = FrameQueue(maxsize=1)
    assert queue.put(frame(frame_id=0)) is True
    assert queue.put(frame(frame_id=1)) is False


def test_queue_records_drops_on_the_delivered_frame():
    """The pipeline can see how much it missed."""
    queue = FrameQueue(maxsize=1)
    queue.put(frame(frame_id=0))
    queue.put(frame(frame_id=1))
    delivered = queue.get(timeout=0.1)
    assert delivered.frame_id == 1
    assert delivered.dropped_before == 1


def test_get_latest_skips_the_backlog():
    """Catching up on stale frames only deepens the lag."""
    queue = FrameQueue(maxsize=8)
    for index in range(5):
        queue.put(frame(frame_id=index, timestamp=index))

    latest = queue.get_latest(timeout=0.1)
    assert latest.frame_id == 4
    assert latest.dropped_before == 4
    assert queue.depth == 0


def test_get_times_out_when_empty():
    started = time.monotonic()
    assert FrameQueue(maxsize=2).get(timeout=0.05) is None
    assert time.monotonic() - started < 1.0


def test_queue_blocks_until_a_frame_arrives():
    queue = FrameQueue(maxsize=2)

    def producer():
        time.sleep(0.05)
        queue.put(frame(frame_id=7))

    threading.Thread(target=producer, daemon=True).start()
    delivered = queue.get(timeout=2.0)
    assert delivered is not None and delivered.frame_id == 7


def test_closed_queue_rejects_and_wakes_readers():
    queue = FrameQueue(maxsize=2)
    queue.put(frame(frame_id=0))
    queue.close()

    assert queue.is_closed
    assert queue.put(frame(frame_id=1)) is False
    assert queue.get(timeout=0.1) is None


def test_queue_requires_a_positive_size():
    with pytest.raises(ValueError):
        FrameQueue(maxsize=0)


def test_queue_counts_received_frames():
    queue = FrameQueue(maxsize=2)
    for index in range(5):
        queue.put(frame(frame_id=index))
    assert queue.received == 5


def test_queue_is_thread_safe_under_contention():
    queue = FrameQueue(maxsize=4)
    consumed: list[int] = []
    stop = threading.Event()

    def producer():
        for index in range(400):
            queue.put(frame(frame_id=index, timestamp=index))
        stop.set()

    def consumer():
        while not stop.is_set() or queue.depth:
            item = queue.get(timeout=0.05)
            if item is not None:
                consumed.append(item.frame_id)

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert consumed == sorted(consumed), "frames must never be delivered out of order"
    assert queue.received == 400


# ---------------------------------------------------------------------------
# Clip writing
# ---------------------------------------------------------------------------


def test_write_clip_produces_a_playable_file(tmp_path):
    import cv2

    buffer = CircularFrameBuffer(seconds=10.0, max_frames=100, max_width=4096)
    for index in range(30):
        buffer.append(frame(frame_id=index, timestamp=index * 0.05, size=(160, 120)))

    path = tmp_path / "clip.mp4"
    assert write_clip(buffer.snapshot(), str(path), fps=20.0)
    assert path.exists() and path.stat().st_size > 0

    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    ok, image = capture.read()
    capture.release()
    assert ok and image is not None


def test_write_clip_refuses_an_empty_frame_list(tmp_path):
    assert not write_clip([], str(tmp_path / "empty.mp4"), fps=20.0)


def test_write_clip_normalizes_a_resolution_change(tmp_path):
    """A camera can reconnect at a different profile mid-buffer; that must not
    abort the incident clip."""
    buffer = CircularFrameBuffer(seconds=10.0, max_frames=100, max_width=4096)
    for index in range(5):
        buffer.append(frame(frame_id=index, timestamp=index * 0.05, size=(160, 120)))
    for index in range(5, 10):
        buffer.append(frame(frame_id=index, timestamp=index * 0.05, size=(320, 240)))

    path = tmp_path / "mixed.mp4"
    assert write_clip(buffer.snapshot(), str(path), fps=20.0)
    assert path.exists()
