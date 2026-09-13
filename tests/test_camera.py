"""Camera stream resilience, zones and the decode worker.

IP cameras disconnect constantly: staff reboot them, PoE switches hiccup, NVRs
restart, and streams stall while holding the socket open. Those are normal
operating conditions, and these tests assert that one camera's failure stays
one camera's failure.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from aisleguardvision.behavior.zones import ZoneRegistry
from aisleguardvision.camera.manager import CameraManager
from aisleguardvision.camera.stream import VideoStream, is_live_source, resolve_source
from aisleguardvision.camera.worker import CameraDecodeWorker
from aisleguardvision.core.config import AppConfig, CameraConfig, ZoneConfig
from aisleguardvision.core.types import BoundingBox, CameraState, Point, ZoneKind


def write_video(path, frames: int = 20, size=(160, 120), fps: float = 20.0) -> str:
    import cv2

    width, height = size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((height, width, 3), (index * 10) % 256, dtype=np.uint8))
    writer.release()
    return str(path)


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------


def test_resolve_source_interprets_a_webcam_index():
    assert resolve_source("0") == 0
    assert resolve_source(1) == 1


def test_resolve_source_keeps_urls_and_paths():
    assert resolve_source("rtsp://host/s") == "rtsp://host/s"
    assert resolve_source("./x.mp4").endswith("x.mp4")


def test_is_live_source():
    assert is_live_source(0)
    assert is_live_source("rtsp://host/s")
    assert not is_live_source("/data/clip.mp4")


# ---------------------------------------------------------------------------
# Stream lifecycle
# ---------------------------------------------------------------------------


def test_stream_opens_a_file_and_reports_its_properties(tmp_path):
    path = write_video(tmp_path / "clip.mp4")
    stream = VideoStream(CameraConfig(id="cam", source=path))

    assert stream.open()
    try:
        assert stream.is_open
        assert stream.state is CameraState.STREAMING
        assert stream.info.width == 160 and stream.info.height == 120
        assert stream.info.fps == pytest.approx(20.0, abs=1.0)
        assert not stream.is_live
    finally:
        stream.close()


def test_stream_reads_frames(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=5)
    with VideoStream(CameraConfig(id="cam", source=path)) as stream:
        ok, image = stream.read()
        assert ok and image is not None and image.shape == (120, 160, 3)
        assert stream.frames_read == 1


def test_failing_to_open_does_not_raise():
    """A dead camera must not take down the process."""
    stream = VideoStream(CameraConfig(id="cam", source="/definitely/not/a/file.mp4"))
    assert stream.open() is False
    assert stream.state is CameraState.RECONNECTING
    assert stream.last_error is not None


def test_reconnect_backoff_grows_exponentially():
    config = CameraConfig(
        id="cam",
        source="/nope.mp4",
        reconnect_initial_seconds=1.0,
        reconnect_max_seconds=16.0,
        reconnect_jitter=0.0,
    )
    stream = VideoStream(config)

    delays = []
    for _ in range(5):
        stream.open()
        delays.append(stream.seconds_until_retry())

    assert delays[0] == pytest.approx(1.0, abs=0.2)
    assert delays[1] > delays[0]
    assert all(d <= config.reconnect_max_seconds + 0.1 for d in delays)


def test_reconnect_backoff_is_capped():
    config = CameraConfig(
        id="cam",
        source="/nope.mp4",
        reconnect_initial_seconds=1.0,
        reconnect_max_seconds=3.0,
        reconnect_jitter=0.0,
    )
    stream = VideoStream(config)
    for _ in range(10):
        stream.open()
    assert stream.seconds_until_retry() <= 3.1


def test_reconnect_jitter_prevents_a_thundering_herd():
    """64 cameras on one NVR must not all reconnect in lockstep after a switch
    reboot."""
    config = CameraConfig(
        id="cam",
        source="/nope.mp4",
        reconnect_initial_seconds=4.0,
        reconnect_max_seconds=60.0,
        reconnect_jitter=0.5,
    )
    delays = set()
    for _ in range(12):
        stream = VideoStream(config)
        stream.open()
        delays.add(round(stream.seconds_until_retry(), 4))
    assert len(delays) > 1, "jitter should spread reconnection attempts"


def test_give_up_honours_the_failure_budget():
    unlimited = VideoStream(CameraConfig(id="cam", source="/nope.mp4"))
    for _ in range(20):
        unlimited.open()
    assert not unlimited.give_up(), "0 means never give up on a fixed store camera"

    limited = VideoStream(
        CameraConfig(id="cam", source="/nope.mp4", max_consecutive_failures=3)
    )
    for _ in range(3):
        limited.open()
    assert limited.give_up()


def test_stall_detection_catches_an_open_but_silent_stream(tmp_path):
    """The failure mode plain error handling misses: the socket stays up and
    read() never errors, so the camera looks healthy forever."""
    path = write_video(tmp_path / "clip.mp4")
    stream = VideoStream(CameraConfig(id="cam", source=path, stall_timeout_seconds=0.2))
    assert stream.open()
    try:
        stream.read()
        assert not stream.is_stalled()
        assert stream.is_stalled(now=time.time() + 5.0)
    finally:
        stream.close()


def test_a_corrupt_frame_is_dropped_without_tearing_down_the_connection(tmp_path):
    path = write_video(tmp_path / "clip.mp4")
    stream = VideoStream(CameraConfig(id="cam", source=path))
    assert stream.open()
    try:
        ok, image = stream.read()
        assert ok
        # A zero-size frame is what a transcoder glitch produces.
        assert stream._accept(image)[0] is True
    finally:
        stream.close()


def test_end_of_file_can_loop(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=3)
    stream = VideoStream(CameraConfig(id="cam", source=path, loop_file_source=True))
    assert stream.open()
    try:
        reads = [stream.read()[0] for _ in range(8)]
        assert all(reads), "a looping source should never run out"
    finally:
        stream.close()


def test_credentials_never_appear_in_the_sanitized_label():
    stream = VideoStream(
        CameraConfig(id="cam", source="rtsp://operator:hunter2@10.0.0.5:554/s")
    )
    assert "hunter2" not in stream.safe_source
    assert "operator" in stream.safe_source


# ---------------------------------------------------------------------------
# Decode worker
# ---------------------------------------------------------------------------


def test_worker_decodes_into_its_queue(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=40)
    worker = CameraDecodeWorker(CameraConfig(id="cam_w", source=path, queue_size=8))
    worker.start()
    try:
        deadline = time.time() + 10
        received = 0
        while time.time() < deadline and received < 5:
            if worker.queue.get(timeout=0.2) is not None:
                received += 1
        assert received >= 5
        assert worker.status().frames_received >= 5
    finally:
        worker.stop(timeout=5)


def test_worker_recovers_from_a_missing_source():
    """The worker must keep retrying rather than exiting."""
    worker = CameraDecodeWorker(
        CameraConfig(
            id="cam_bad",
            source="/definitely/missing.mp4",
            reconnect_initial_seconds=0.05,
            reconnect_max_seconds=0.2,
        )
    )
    worker.start()
    try:
        time.sleep(0.8)
        assert worker.is_running, "the worker thread must stay alive"
        assert worker.status().state in (CameraState.RECONNECTING, CameraState.CONNECTING)
    finally:
        worker.stop(timeout=5)


def test_worker_stops_after_the_failure_budget_is_exhausted():
    worker = CameraDecodeWorker(
        CameraConfig(
            id="cam_bad",
            source="/missing.mp4",
            reconnect_initial_seconds=0.02,
            reconnect_max_seconds=0.05,
            max_consecutive_failures=2,
        )
    )
    worker.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and worker.status().state is not CameraState.FAILED:
            time.sleep(0.05)
        assert worker.status().state is CameraState.FAILED
    finally:
        worker.stop(timeout=5)


def test_worker_enable_and_disable(tmp_path):
    path = write_video(tmp_path / "clip.mp4", frames=60)
    worker = CameraDecodeWorker(CameraConfig(id="cam_e", source=path))
    worker.start()
    try:
        worker.disable()
        time.sleep(0.2)
        assert not worker.is_enabled
        assert worker.status().state is CameraState.DISABLED

        worker.enable()
        assert worker.is_enabled
    finally:
        worker.stop(timeout=5)


def test_worker_status_never_exposes_credentials():
    worker = CameraDecodeWorker(
        CameraConfig(id="cam_s", source="rtsp://u:hunter2@host/s", enabled=False)
    )
    assert "hunter2" not in worker.status().source_label


# ---------------------------------------------------------------------------
# Camera manager
# ---------------------------------------------------------------------------


def test_manager_skips_a_camera_with_an_unresolved_source():
    """A dev box has credentials for one camera out of sixty-four; that must
    not stop the other sixty-three."""
    manager = CameraManager(AppConfig())
    assert manager.add_camera(CameraConfig(id="cam_unset", source="")) is None
    assert manager.camera_ids == []


def test_manager_registers_a_camera_with_zones():
    manager = CameraManager(AppConfig())
    camera = CameraConfig(
        id="cam_z",
        source="0",
        zones=[ZoneConfig(id="s1", polygon=[(0, 0), (10, 0), (10, 10)])],
    )
    assert manager.add_camera(camera) is not None
    assert len(manager.zones("cam_z")) == 1
    assert manager.buffer("cam_z") is not None


def test_manager_rejects_a_duplicate_camera_id():
    manager = CameraManager(AppConfig())
    manager.add_camera(CameraConfig(id="cam_d", source="0"))
    with pytest.raises(ValueError):
        manager.add_camera(CameraConfig(id="cam_d", source="0"))


def test_manager_isolates_a_bad_camera_from_the_rest():
    config = AppConfig()
    config.cameras.cameras.extend(
        [
            CameraConfig(id="cam_good", source="0"),
            CameraConfig(id="cam_unset", source=""),
            CameraConfig(id="cam_also_good", source="1"),
        ]
    )
    manager = CameraManager(config)
    workers = manager.load_from_config(start=False)
    assert sorted(w.camera_id for w in workers) == ["cam_also_good", "cam_good"]


def test_manager_health_summary_on_an_empty_manager():
    summary = CameraManager(AppConfig()).health_summary()
    assert summary["cameras_total"] == 0
    assert summary["cameras_stalled"] == []


def test_manager_enable_disable_unknown_camera():
    manager = CameraManager(AppConfig())
    assert manager.enable("nope") is False
    assert manager.disable("nope") is False


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


def test_zone_registry_loads_from_config():
    registry = ZoneRegistry.from_config(
        "cam",
        [
            ZoneConfig(id="shelf", kind=ZoneKind.SHELF, polygon=[(0, 0), (100, 0), (100, 100)]),
            ZoneConfig(
                id="basket", kind=ZoneKind.BASKET, polygon=[(200, 200), (300, 200), (300, 300)]
            ),
        ],
    )
    assert len(registry) == 2
    assert registry.merchandise_zones[0].zone_id == "shelf"
    assert registry.container_zones[0].zone_id == "basket"
    assert registry.has_merchandise_zones


def test_zone_registry_skips_a_degenerate_polygon():
    """A malformed zone must degrade that zone, not the camera."""
    registry = ZoneRegistry.from_config(
        "cam",
        [
            ZoneConfig(id="ok", polygon=[(0, 0), (100, 0), (100, 100)]),
            ZoneConfig(id="flat", polygon=[(0, 0), (10, 0), (20, 0)]),  # zero area
        ],
    )
    assert [z.zone_id for z in registry] == ["ok"]


def test_zone_registry_skips_disabled_zones():
    registry = ZoneRegistry.from_config(
        "cam", [ZoneConfig(id="off", enabled=False, polygon=[(0, 0), (10, 0), (10, 10)])]
    )
    assert len(registry) == 0


def test_zone_containment_and_distance():
    registry = ZoneRegistry.from_config(
        "cam", [ZoneConfig(id="shelf", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)])]
    )
    assert registry.zone_containing(Point(50, 50)) is not None
    assert registry.zone_containing(Point(500, 500)) is None

    hit = registry.nearest_zone(Point(130, 50), reference_length=300.0)
    assert hit is not None
    assert not hit.inside
    assert hit.distance == pytest.approx(30.0)
    assert hit.normalized_distance == pytest.approx(0.1)


def test_exclusion_zones_mask_out_noise():
    """Masks a doorway, a mirror, or a display screen playing video."""
    registry = ZoneRegistry.from_config(
        "cam",
        [
            ZoneConfig(
                id="door",
                kind=ZoneKind.EXCLUSION,
                polygon=[(0, 0), (200, 0), (200, 140), (0, 140)],
            )
        ],
    )
    assert registry.is_excluded(Point(100, 70))
    assert not registry.is_excluded(Point(600, 400))
    assert registry.box_is_excluded(BoundingBox(10, 10, 190, 130))


def test_zones_overlapping_box_is_sorted_by_overlap():
    registry = ZoneRegistry.from_config(
        "cam",
        [
            ZoneConfig(id="a", polygon=[(0, 0), (100, 0), (100, 100), (0, 100)]),
            ZoneConfig(id="b", polygon=[(90, 0), (200, 0), (200, 100), (90, 100)]),
        ],
    )
    overlaps = registry.zones_overlapping_box(BoundingBox(0, 0, 80, 80))
    assert overlaps[0][0].zone_id == "a"


def test_duplicate_zone_ids_are_rejected():
    registry = ZoneRegistry("cam")
    zone = ZoneConfig(id="dup", polygon=[(0, 0), (10, 0), (10, 10)])
    ZoneRegistry.from_config("cam", [zone])
    from aisleguardvision.core.types import ShelfZone
    import numpy as np

    shelf = ShelfZone(
        zone_id="dup",
        camera_id="cam",
        polygon=np.array([[0, 0], [10, 0], [10, 10]], dtype=np.float32),
    )
    registry.add(shelf)
    with pytest.raises(ValueError):
        registry.add(shelf)


def test_empty_registry_describes_itself_honestly():
    assert "no zones configured" in ZoneRegistry("cam").describe()
