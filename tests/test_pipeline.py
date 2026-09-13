"""Pipeline integration, scheduling, backends and metrics.

Runs the real :class:`CameraPipeline` with a scripted backend in place of YOLO,
which exercises the wiring between every stage without needing a GPU.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from aisleguardvision.behavior.zones import ZoneRegistry
from aisleguardvision.core.config import (
    AppConfig,
    CameraConfig,
    InferenceConfig,
    ModelsConfig,
    SchedulerConfig,
    ZoneConfig,
)
from aisleguardvision.core.metrics import Histogram, MetricsRegistry, RateMeter
from aisleguardvision.core.types import (
    BoundingBox,
    Detection,
    Frame,
    ObjectClass,
    PersonTrack,
    Point,
    PoseObservation,
    TrackState,
)
from aisleguardvision.events.dispatcher import AlertDispatcher, AlertProvider
from aisleguardvision.inference.backend import (
    BackendCapabilities,
    DetectorBackend,
    NullBackend,
    create_detector_backend,
    filter_detections,
    map_class_name,
)
from aisleguardvision.inference.detector import PersonDetector
from aisleguardvision.inference.device import DeviceInfo, select_device
from aisleguardvision.inference.pose import PoseEstimator
from aisleguardvision.inference.product_detector import (
    CocoProxyProductDetector,
    StaticProductDetector,
    ZoneOnlyProductDetector,
    create_product_detector,
)
from aisleguardvision.inference.scheduler import BatchAccumulator, FrameScheduler
from aisleguardvision.pipeline import CameraPipeline
from aisleguardvision.simulation.scenarios import SHELF_ZONES, get_scenario, make_person


# ---------------------------------------------------------------------------
# Scripted backend
# ---------------------------------------------------------------------------


class ScriptedBackend(DetectorBackend):
    """Replays scenario detections. Stands in for YOLO, nothing else changes."""

    def __init__(self, models, inference, device=None, pose=False):
        super().__init__(models, inference, device)
        self.is_pose = pose
        self.frames: list[list[Detection]] = []
        self.index = 0
        self.calls = 0

    def load(self) -> None:
        self._loaded = True

    def _infer_batch(self, images):
        self.calls += 1
        results = []
        for _ in images:
            frame = self.frames[self.index % len(self.frames)] if self.frames else []
            self.index += 1
            results.append(frame)
        return results

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_batching=True,
            max_batch_size=8,
            supports_pose=self.is_pose,
            name="scripted-pose" if self.is_pose else "scripted",
        )


def scripted_pipeline(config: AppConfig, scenario_name: str = "POSSIBLE_CONCEALMENT"):
    """Build a pipeline whose detector replays a simulation scenario."""
    scenario = get_scenario(scenario_name)
    models, inference = ModelsConfig(), InferenceConfig()

    detector_backend = ScriptedBackend(models, inference)
    pose_backend = ScriptedBackend(models, inference, pose=True)

    for frame in scenario.frames:
        people = [
            Detection(
                bbox=person.bbox,
                confidence=person.confidence,
                object_class=ObjectClass.PERSON,
                class_name="person",
            )
            for person in frame.persons
        ]
        items = [
            Detection(
                bbox=item.bbox,
                confidence=item.confidence,
                object_class=ObjectClass.BOTTLE,
                class_name="bottle",
            )
            for item in frame.items
        ]
        detector_backend.frames.append(people + items)
        pose_backend.frames.append(
            [
                Detection(
                    bbox=person.bbox,
                    confidence=person.confidence,
                    object_class=ObjectClass.PERSON,
                    pose=PoseObservation.from_array(person.keypoints, person.bbox),
                )
                for person in frame.persons
            ]
        )

    camera = CameraConfig(id="cam_pipe", source="scripted", zones=list(SHELF_ZONES))
    config.cameras.cameras.append(camera)

    pipeline = CameraPipeline(
        camera_id="cam_pipe",
        config=config,
        detector=PersonDetector(detector_backend, config.detection),
        pose_estimator=PoseEstimator(pose_backend, config.detection, crop_mode=False),
        product_detector=CocoProxyProductDetector(detector_backend),
        zones=ZoneRegistry.from_config("cam_pipe", SHELF_ZONES),
    )
    return pipeline, scenario


def frame_at(timestamp: float, frame_id: int) -> Frame:
    return Frame(
        camera_id="cam_pipe",
        frame_id=frame_id,
        timestamp=timestamp,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
    )


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_runs_every_stage_end_to_end():
    config = AppConfig()
    # Run every model on every frame so the scripted detections line up with
    # the scenario's frame indices.
    config.detection.scheduler = SchedulerConfig(
        detection_fps=1000.0, pose_fps=1000.0, adaptive_pose=False,
        pose_only_for_relevant_people=False,
    )
    pipeline, scenario = scripted_pipeline(config)

    results = [pipeline.process(frame_at(f.timestamp, i)) for i, f in enumerate(scenario.frames)]

    assert any(r.tracks for r in results), "people should be tracked"
    assert any(r.items for r in results), "merchandise candidates should be tracked"
    assert any(r.observations for r in results), "the behavior engine should produce observations"
    assert max(r.peak_risk for r in results) > 0


def test_pipeline_generates_an_incident_for_the_concealment_scenario():
    config = AppConfig()
    config.detection.scheduler = SchedulerConfig(
        detection_fps=1000.0, pose_fps=1000.0, adaptive_pose=False,
        pose_only_for_relevant_people=False,
    )
    pipeline, scenario = scripted_pipeline(config)

    events = []
    for index, sim_frame in enumerate(scenario.frames):
        events.extend(pipeline.process(frame_at(sim_frame.timestamp, index)).events)

    assert events, "the full sequence should produce a reviewable incident"
    assert events[0].risk_score >= config.behavior.alert_threshold
    assert events[0].positive_evidence


def test_pipeline_cooldown_limits_repeat_incidents():
    """One sustained sequence must not produce an incident per frame."""
    config = AppConfig()
    config.detection.scheduler = SchedulerConfig(
        detection_fps=1000.0, pose_fps=1000.0, adaptive_pose=False,
        pose_only_for_relevant_people=False,
    )
    pipeline, scenario = scripted_pipeline(config)

    total = 0
    for index, sim_frame in enumerate(scenario.frames):
        total += len(pipeline.process(frame_at(sim_frame.timestamp, index)).events)

    assert total <= 2, f"cooldown should have suppressed repeats, got {total} incidents"


def test_pipeline_does_not_alert_on_benign_scenarios():
    for name in ("NORMAL_BROWSING", "PHONE_INTERACTION", "ITEM_TO_BASKET"):
        config = AppConfig()
        config.detection.scheduler = SchedulerConfig(
            detection_fps=1000.0, pose_fps=1000.0, adaptive_pose=False,
            pose_only_for_relevant_people=False,
        )
        pipeline, scenario = scripted_pipeline(config, name)
        events = []
        for index, sim_frame in enumerate(scenario.frames):
            events.extend(pipeline.process(frame_at(sim_frame.timestamp, index)).events)
        assert not events, f"{name} produced an incident"


def test_pipeline_reports_zone_only_mode():
    config = AppConfig()
    camera = CameraConfig(id="cam_zo", source="x")
    config.cameras.cameras.append(camera)
    pipeline = CameraPipeline(
        camera_id="cam_zo",
        config=config,
        detector=PersonDetector(
            NullBackend(ModelsConfig(), InferenceConfig()), config.detection
        ),
        product_detector=ZoneOnlyProductDetector(),
    )
    assert pipeline.item_detection_available is False


def test_pipeline_survives_a_frame_with_no_detections():
    config = AppConfig()
    camera = CameraConfig(id="cam_empty", source="x")
    config.cameras.cameras.append(camera)
    pipeline = CameraPipeline(
        camera_id="cam_empty",
        config=config,
        detector=PersonDetector(
            NullBackend(ModelsConfig(), InferenceConfig()), config.detection
        ),
    )
    result = pipeline.process(frame_at(time.time(), 1))
    assert result.tracks == []
    assert result.events == []


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def test_null_backend_detects_nothing_and_says_so():
    """Not a placeholder for missing work: it is how the pipeline stays
    runnable with no weights, while correctly reporting zero capability."""
    backend = NullBackend(ModelsConfig(), InferenceConfig())
    assert backend.infer([np.zeros((10, 10, 3), dtype=np.uint8)]) == [[]]
    assert backend.capabilities.name == "null"
    assert not backend.capabilities.supports_pose


def test_backend_infer_is_batch_shaped():
    """The signature a multi-camera GPU worker needs, available from day one."""
    backend = ScriptedBackend(ModelsConfig(), InferenceConfig())
    backend.frames = [[Detection(BoundingBox(0, 0, 10, 10), 0.9, ObjectClass.PERSON)]]
    images = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(8)]

    results = backend.infer(images)
    assert len(results) == 8
    assert backend.calls == 1, "eight camera frames should cost one backend invocation"


def test_backend_isolates_an_inference_failure():
    """A model failure must degrade the camera, not crash the process."""

    class ExplodingBackend(ScriptedBackend):
        def _infer_batch(self, images):
            raise RuntimeError("CUDA is on fire")

    backend = ExplodingBackend(ModelsConfig(), InferenceConfig())
    results = backend.infer([np.zeros((10, 10, 3), dtype=np.uint8)] * 3)

    assert results == [[], [], []]
    assert backend.stats.errors == 1


def test_backend_pads_a_mismatched_result_count():
    class ShortBackend(ScriptedBackend):
        def _infer_batch(self, images):
            return [[]]

    backend = ShortBackend(ModelsConfig(), InferenceConfig())
    assert len(backend.infer([np.zeros((4, 4, 3), dtype=np.uint8)] * 3)) == 3


def test_backend_records_latency():
    backend = ScriptedBackend(ModelsConfig(), InferenceConfig())
    backend.frames = [[]]
    backend.infer([np.zeros((4, 4, 3), dtype=np.uint8)])
    assert backend.stats.invocations == 1
    assert backend.stats.last_latency_ms >= 0.0


def test_unknown_backend_falls_back_to_null_loudly():
    backend = create_detector_backend(
        ModelsConfig(backend="does-not-exist"), InferenceConfig()
    )
    assert backend.capabilities.name == "null"


def test_coco_class_mapping():
    assert map_class_name("person") is ObjectClass.PERSON
    assert map_class_name("cell phone") is ObjectClass.PHONE
    assert map_class_name("Cell Phone") is ObjectClass.PHONE
    assert map_class_name("giraffe") is ObjectClass.UNKNOWN


def test_filter_detections_by_class_and_confidence():
    detections = [
        Detection(BoundingBox(0, 0, 1, 1), 0.9, ObjectClass.PERSON),
        Detection(BoundingBox(0, 0, 1, 1), 0.2, ObjectClass.PERSON),
        Detection(BoundingBox(0, 0, 1, 1), 0.9, ObjectClass.PHONE),
    ]
    kept = filter_detections(detections, {ObjectClass.PERSON}, min_confidence=0.5)
    assert len(kept) == 1


def test_detector_partitions_people_objects_and_containers():
    backend = ScriptedBackend(ModelsConfig(), InferenceConfig())
    backend.frames = [
        [
            Detection(BoundingBox(0, 0, 100, 300), 0.9, ObjectClass.PERSON),
            Detection(BoundingBox(10, 10, 40, 40), 0.8, ObjectClass.PHONE),
            Detection(BoundingBox(50, 50, 90, 90), 0.7, ObjectClass.SHOPPING_CART),
        ]
    ]
    detector = PersonDetector(backend, AppConfig().detection)
    bundle = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8), "cam", 1, 1.0)

    assert len(bundle.people) == 1
    assert len(bundle.objects) == 1
    assert len(bundle.containers) == 1


# ---------------------------------------------------------------------------
# Product detectors
# ---------------------------------------------------------------------------


def test_zone_only_detector_reports_no_merchandise_capability():
    detector = ZoneOnlyProductDetector()
    assert detector.detect(np.zeros((4, 4, 3), dtype=np.uint8), "cam", 1.0) == []
    assert not detector.provides_merchandise_detection
    assert "zone" in detector.info().description.lower()


def test_coco_proxy_only_maps_carryable_classes():
    """It covers bottles, cups and books -- NOT general retail merchandise."""
    detector = CocoProxyProductDetector()
    detections = [
        Detection(BoundingBox(0, 0, 10, 10), 0.9, ObjectClass.BOTTLE, class_name="bottle"),
        Detection(BoundingBox(0, 0, 10, 10), 0.9, ObjectClass.PERSON, class_name="person"),
        Detection(BoundingBox(0, 0, 10, 10), 0.9, ObjectClass.PHONE, class_name="cell phone"),
    ]
    items = detector.from_detections(detections, source="coco_proxy")
    assert len(items) == 1
    assert items[0].object_class is ObjectClass.BOTTLE


def test_create_product_detector_falls_back_to_zone_only_without_a_model():
    detector = create_product_detector(
        ModelsConfig(product=""),
        AppConfig().detection,
        detector_backend=NullBackend(ModelsConfig(), InferenceConfig()),
    )
    assert not detector.provides_merchandise_detection


def test_create_product_detector_uses_the_coco_proxy_when_a_detector_exists():
    detector = create_product_detector(
        ModelsConfig(product=""),
        AppConfig().detection,
        detector_backend=ScriptedBackend(ModelsConfig(), InferenceConfig()),
    )
    assert detector.info().name == "coco_proxy"


def test_static_product_detector_is_marked_as_a_test_double():
    detector = StaticProductDetector([])
    assert detector.info().name == "static"


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def confirmed_track(track_id: int, centre: Point) -> PersonTrack:
    person = make_person(centre, cx=centre.x)
    return PersonTrack(
        track_id=track_id,
        camera_id="cam",
        bbox=person.bbox,
        confidence=0.9,
        first_seen=0.0,
        last_seen=1.0,
        state=TrackState.CONFIRMED,
    )


def test_scheduler_rate_limits_detection():
    scheduler = FrameScheduler(SchedulerConfig(detection_fps=10.0))
    assert scheduler.decide(0.0, []).run_detection
    assert not scheduler.decide(0.05, []).run_detection, "too soon"
    assert scheduler.decide(0.11, []).run_detection


def test_scheduler_rates_are_time_based_not_frame_based():
    """The same configuration must behave identically on a 10 and a 30 FPS
    camera."""

    def count(fps: float) -> int:
        scheduler = FrameScheduler(SchedulerConfig(detection_fps=10.0, adaptive_pose=False))
        step = 1.0 / fps
        return sum(
            1
            for index in range(int(fps * 2))
            if scheduler.decide(index * step, []).run_detection
        )

    assert abs(count(30.0) - count(10.0)) <= 1


def test_scheduler_idles_pose_when_nobody_is_near_merchandise():
    config = SchedulerConfig(pose_idle_fps=2.0, pose_active_fps=10.0, adaptive_pose=True)
    scheduler = FrameScheduler(config)
    assert scheduler.decide(0.0, []).pose_fps == pytest.approx(config.pose_idle_fps)


def test_scheduler_escalates_pose_for_a_person_in_a_zone():
    """64 cameras cannot run pose on everyone; they can run it on the shopper
    who is actually at a shelf."""
    config = SchedulerConfig(pose_idle_fps=2.0, pose_active_fps=10.0, adaptive_pose=True)
    scheduler = FrameScheduler(config)
    zones = ZoneRegistry.from_config("cam", SHELF_ZONES)

    inside = confirmed_track(1, Point(300, 360))
    decision = scheduler.decide(0.0, [inside], zones=zones)
    assert decision.pose_fps == pytest.approx(config.pose_active_fps)


def test_scheduler_escalates_pose_when_risk_is_climbing():
    config = SchedulerConfig(pose_idle_fps=2.0, pose_active_fps=10.0, adaptive_pose=True)
    scheduler = FrameScheduler(config)
    track = confirmed_track(1, Point(1100, 360))
    zones = ZoneRegistry.from_config("cam", SHELF_ZONES)

    decision = scheduler.decide(0.0, [track], zones=zones, risks={1: 55.0})
    assert decision.pose_fps == pytest.approx(config.pose_active_fps)


def test_scheduler_limits_the_number_of_pose_targets():
    config = SchedulerConfig(max_pose_targets=3, adaptive_pose=True)
    scheduler = FrameScheduler(config)
    zones = ZoneRegistry.from_config("cam", SHELF_ZONES)
    tracks = [confirmed_track(i, Point(300 + i * 5, 360)) for i in range(10)]

    decision = scheduler.decide(0.0, tracks, zones=zones)
    assert len(decision.pose_targets) <= 3


def test_scheduler_skips_pose_when_nobody_is_relevant():
    # A tight relevance radius so "far from the shelf" is unambiguous; the
    # shipped default of 1.5 body heights is deliberately generous.
    config = SchedulerConfig(
        adaptive_pose=True,
        pose_only_for_relevant_people=True,
        pose_relevance_distance_ratio=0.3,
    )
    scheduler = FrameScheduler(config)
    zones = ZoneRegistry.from_config("cam", SHELF_ZONES)
    far_away = confirmed_track(1, Point(1250, 400))

    decision = scheduler.decide(0.0, [far_away], zones=zones)
    assert not decision.run_pose or not decision.pose_targets


def test_scheduler_gives_baseline_coverage_when_no_zones_are_configured():
    """With nothing to be near, everybody gets attention rather than nobody."""
    scheduler = FrameScheduler(SchedulerConfig(adaptive_pose=True))
    decision = scheduler.decide(0.0, [confirmed_track(1, Point(600, 400))], zones=ZoneRegistry("cam"))
    assert decision.pose_targets


def test_scheduler_handles_a_backwards_timestamp():
    scheduler = FrameScheduler(SchedulerConfig(detection_fps=10.0))
    scheduler.decide(100.0, [])
    assert scheduler.decide(50.0, []).run_detection, "a seek should re-anchor, not stall"


def test_force_pose_next():
    scheduler = FrameScheduler(SchedulerConfig(pose_fps=1.0, adaptive_pose=False))
    scheduler.decide(0.0, [])
    assert not scheduler.decide(0.1, []).run_pose
    scheduler.force_pose_next()
    assert scheduler.decide(0.11, []).run_pose


def test_batch_accumulator_flushes_when_full():
    accumulator = BatchAccumulator(max_batch_size=3, timeout_ms=1000)
    assert accumulator.add(("a",), 0.0) is None
    assert accumulator.add(("b",), 0.01) is None
    batch = accumulator.add(("c",), 0.02)
    assert batch is not None and len(batch) == 3


def test_batch_accumulator_flushes_on_timeout():
    accumulator = BatchAccumulator(max_batch_size=8, timeout_ms=10)
    accumulator.add(("a",), 0.0)
    assert accumulator.add(("b",), 0.5) is not None


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def test_device_selection_never_assumes_cuda():
    info = select_device("auto", want_fp16=True)
    assert info.kind in {"cuda", "mps", "cpu"}
    assert info.device


def test_explicit_cpu_is_honoured():
    assert select_device("cpu").kind == "cpu"


def test_unavailable_cuda_falls_back_rather_than_raising():
    info = select_device("cuda:99", want_fp16=True)
    assert info.kind in {"cuda", "cpu"}


def test_jetson_detection_is_name_based():
    assert DeviceInfo("cuda:0", "cuda", "Orin", True).is_jetson
    assert DeviceInfo("cuda:0", "cuda", "NVIDIA GeForce RTX 4090", True).is_jetson is False


def test_device_describe_is_loggable():
    assert "device=" in select_device("cpu").describe()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_counter_and_gauge():
    registry = MetricsRegistry()
    registry.inc("frames", 3, camera_id="cam_a")
    registry.inc("frames", 2, camera_id="cam_a")
    registry.set_gauge("queue", 4, camera_id="cam_a")

    snapshot = registry.snapshot()
    assert snapshot["counters"]["frames{camera_id=cam_a}"] == 5
    assert snapshot["gauges"]["queue{camera_id=cam_a}"] == 4


def test_metrics_labels_separate_cameras():
    registry = MetricsRegistry()
    registry.inc("frames", 1, camera_id="cam_a")
    registry.inc("frames", 7, camera_id="cam_b")

    counters = registry.snapshot()["counters"]
    assert counters["frames{camera_id=cam_a}"] == 1
    assert counters["frames{camera_id=cam_b}"] == 7


def test_rate_meter_decays_to_zero_when_a_camera_stalls():
    """A stalled camera must not keep reporting its last healthy rate."""
    meter = RateMeter("fps", window_seconds=0.2)
    now = time.monotonic()
    for index in range(10):
        meter.tick(now=now + index * 0.01)
    assert meter.value > 0

    time.sleep(0.3)
    assert meter.value == 0.0


def test_histogram_reports_percentiles():
    histogram = Histogram("latency", size=100)
    for value in range(1, 101):
        histogram.observe(float(value))

    stats = histogram.snapshot()
    assert stats["count"] == 100
    assert stats["p50"] == pytest.approx(50.5, abs=1.0)
    assert stats["p95"] == pytest.approx(95.0, abs=2.0)
    assert stats["max"] == 100.0


def test_histogram_is_bounded():
    histogram = Histogram("latency", size=10)
    for value in range(1000):
        histogram.observe(float(value))
    assert len(histogram._samples) == 10
    assert histogram.snapshot()["count"] == 1000, "the total count is still tracked"


def test_empty_histogram_snapshot():
    assert Histogram("x").snapshot()["count"] == 0


def test_prometheus_export_is_well_formed():
    registry = MetricsRegistry()
    registry.inc("alerts_generated", 2, camera_id="cam_a")
    registry.observe("inference_latency_ms", 31.0, camera_id="cam_a")

    text = registry.to_prometheus()
    assert "# TYPE alerts_generated counter" in text
    assert 'camera_id="cam_a"' in text
    assert 'quantile="0.95"' in text
    assert text.endswith("\n")
