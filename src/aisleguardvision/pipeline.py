"""Analysis pipeline.

Wires the stages together for one camera:

    Frame -> Detection -> Person tracking -> Conditional pose
          -> Pose/person association -> Product detection -> Item tracking
          -> Hand/item association -> Temporal behavior engine -> Risk
          -> Cooldown -> Incident recorder -> Alert dispatcher

Deliberately separate from :mod:`aisleguardvision.main`, which owns the CLI and
the display loop. One :class:`CameraPipeline` instance per camera; a GPU worker
process holds several of them and shares one detector backend between them,
which is the shape the 64-camera architecture needs (see ``docs/SCALING.md``).

Nothing in this class blocks on I/O: clip encoding and alert delivery are both
handed to background threads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .behavior.engine import TemporalBehaviorEngine
from .behavior.risk import RiskEngine
from .behavior.zones import ZoneRegistry
from .camera.frame_buffer import CircularFrameBuffer
from .core.config import AppConfig
from .core.logging import get_logger
from .core.metrics import MetricNames, MetricsRegistry, get_metrics
from .core.types import (
    BehaviorObservation,
    BoundingBox,
    Frame,
    ItemTrack,
    ObjectClass,
    PersonTrack,
    SecurityEvent,
)
from .events.cooldown import AlertCooldown
from .events.dispatcher import AlertDispatcher
from .events.models import build_event
from .events.recorder import IncidentRecorder
from .inference.detector import PersonDetector
from .inference.pose import PoseEstimator
from .inference.product_detector import ProductDetector, ZoneOnlyProductDetector
from .inference.scheduler import FrameScheduler
from .tracking.association import (
    HandItemAssociator,
    WristHistoryStore,
    associate_poses_with_tracks,
)
from .tracking.item_tracker import ItemTracker
from .tracking.person_tracker import PersonTracker, attach_pose

logger = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    """Everything one analysis pass produced, for display and recording."""

    camera_id: str
    frame: Frame
    tracks: list[PersonTrack] = field(default_factory=list)
    items: list[ItemTrack] = field(default_factory=list)
    observations: dict[int, BehaviorObservation] = field(default_factory=dict)
    events: list[SecurityEvent] = field(default_factory=list)
    ran_detection: bool = False
    ran_pose: bool = False
    inference_latency_ms: float = 0.0
    pose_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def peak_risk(self) -> float:
        if not self.observations:
            return 0.0
        return max(o.risk.risk_score for o in self.observations.values())


class CameraPipeline:
    """Full analysis pipeline for one camera."""

    def __init__(
        self,
        camera_id: str,
        config: AppConfig,
        detector: PersonDetector,
        pose_estimator: PoseEstimator | None = None,
        product_detector: ProductDetector | None = None,
        zones: ZoneRegistry | None = None,
        recorder: IncidentRecorder | None = None,
        dispatcher: AlertDispatcher | None = None,
        buffer: CircularFrameBuffer | None = None,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config
        self.metrics = metrics or get_metrics()

        self.detector = detector
        self.pose_estimator = pose_estimator
        self.product_detector = product_detector or ZoneOnlyProductDetector()
        self.zones = zones or ZoneRegistry(camera_id)
        self.recorder = recorder
        self.dispatcher = dispatcher
        self.buffer = buffer

        detection = config.detection
        camera_config = config.camera(camera_id)
        self.scheduler = FrameScheduler(
            detection.scheduler,
            detection_fps_override=camera_config.target_inference_fps if camera_config else 0.0,
        )
        self.person_tracker = PersonTracker(camera_id, detection.tracking.person)
        self.item_tracker = ItemTracker(camera_id, detection.tracking.item)
        self.associator = HandItemAssociator(detection.tracking.association)
        self.wrists = WristHistoryStore()
        self.risk_engine = RiskEngine(config.risk, config.behavior)
        self.behavior_engine = TemporalBehaviorEngine(camera_id, config, self.risk_engine)
        self.cooldown = AlertCooldown(
            config.behavior.cooldown_seconds, config.behavior.cooldown_escalation_delta
        )

        self._item_detection_available = self.product_detector.provides_merchandise_detection
        self._last_pose_latency = 0.0
        self._last_inference_latency = 0.0
        self._labels = {"camera_id": camera_id}
        self.alerts_generated = 0

        info = self.product_detector.info()
        logger.info(
            "pipeline created",
            extra={
                "fields": {
                    "camera_id": camera_id,
                    "zones": len(self.zones),
                    "product_detector": info.name,
                    "item_evidence": info.provides_merchandise_detection,
                    "detection_fps": round(self.scheduler.detection_fps, 1),
                }
            },
        )
        if not self._item_detection_available:
            logger.warning(
                "camera running in zone-only mode: risk is capped at "
                f"{config.behavior.zone_only_risk_ceiling:.0f} and cannot reach the "
                f"alert threshold of {config.behavior.alert_threshold:.0f}",
                extra={"fields": {"camera_id": camera_id}},
            )

    # -- main entry point --------------------------------------------------
    def process(self, frame: Frame) -> PipelineResult:
        """Run one frame through the pipeline."""
        started = time.perf_counter()
        result = PipelineResult(camera_id=self.camera_id, frame=frame)
        timestamp = frame.timestamp

        decision = self.scheduler.decide(
            timestamp,
            self.person_tracker.confirmed_tracks(),
            zones=self.zones,
            states=self.behavior_engine.states(),
            risks=self.behavior_engine.risks(),
            associated_ids={
                item.associated_person_id
                for item in self.item_tracker.tracks
                if item.associated_person_id is not None
            },
        )

        container_boxes: list[BoundingBox] = []
        tracks: list[PersonTrack] = self.person_tracker.confirmed_tracks()

        if decision.run_detection:
            bundle = self.detector.detect(frame.image, self.camera_id, frame.frame_id, timestamp)
            result.ran_detection = True
            self._last_inference_latency = bundle.latency_ms
            result.inference_latency_ms = bundle.latency_ms

            tracks = self.person_tracker.update(bundle.people, timestamp)
            container_boxes = [d.bbox for d in bundle.containers]

            # Merchandise candidates. The proxy detector reuses the detections
            # we already have rather than running a second model.
            items = self.product_detector.from_detections(bundle.objects, source="coco_proxy")
            if not items and self.product_detector.provides_merchandise_detection:
                items = self.product_detector.detect(frame.image, self.camera_id, timestamp)

            # Personal effects (phones, bags) are tracked as items too: the
            # phone class is what suppresses the most common false positive,
            # so it must reach the associator.
            for detection in bundle.objects:
                if detection.object_class.is_personal_effect:
                    items.append(
                        _as_item(detection, source="coco_personal_effect")
                    )

            self.item_tracker.update(items, timestamp)
            self.metrics.inc(MetricNames.FRAMES_PROCESSED, **self._labels)
        else:
            # Between detections, the Kalman filter carries person tracks
            # forward and the item ladder advances on elapsed time. This is
            # what makes a 10 FPS detector usable on a 30 FPS camera.
            self.item_tracker.tick(timestamp)

        if decision.run_pose and self.pose_estimator is not None and tracks:
            self._run_pose(frame, tracks, timestamp, decision.pose_targets, result)

        item_tracks = self.item_tracker.tracks
        associations = self.associator.associate(
            tracks,
            item_tracks,
            self.wrists,
            timestamp,
            self.config.detection.inference.keypoint_confidence,
        )

        observations = self.behavior_engine.update(
            timestamp=timestamp,
            person_tracks=tracks,
            item_tracks=item_tracks,
            associations=associations,
            zones=self.zones,
            wrists=self.wrists,
            item_tracker=self.item_tracker,
            track_quality={
                track.track_id: self.person_tracker.track_quality(track.track_id)
                for track in tracks
            },
            item_detection_available=self._item_detection_available,
            container_boxes=container_boxes,
        )

        result.tracks = tracks
        result.items = item_tracks
        result.observations = {o.person_id: o for o in observations}
        result.events = self._handle_alerts(observations, frame)

        live_ids = {track.track_id for track in tracks}
        self.wrists.prune(live_ids)
        self.cooldown.retain_only(self.camera_id, live_ids)

        result.pose_latency_ms = self._last_pose_latency
        result.total_latency_ms = (time.perf_counter() - started) * 1000.0

        self.metrics.set_gauge(MetricNames.ACTIVE_TRACKS, len(tracks), **self._labels)
        self.metrics.set_gauge(
            MetricNames.ACTIVE_ITEM_TRACKS, self.item_tracker.active_count(), **self._labels
        )
        self.metrics.observe(
            MetricNames.PIPELINE_LATENCY_MS, result.total_latency_ms, **self._labels
        )
        self.metrics.observe(
            MetricNames.END_TO_END_LATENCY_MS,
            (time.time() - frame.timestamp) * 1000.0,
            **self._labels,
        )
        self.metrics.tick(MetricNames.PIPELINE_FPS, **self._labels)
        return result

    # -- stages ------------------------------------------------------------
    def _run_pose(
        self,
        frame: Frame,
        tracks: list[PersonTrack],
        timestamp: float,
        targets: list[PersonTrack],
        result: PipelineResult,
    ) -> None:
        assert self.pose_estimator is not None
        pose_result = self.pose_estimator.estimate(
            frame.image, self.camera_id, timestamp, targets or tracks
        )
        result.ran_pose = True
        self._last_pose_latency = pose_result.latency_ms
        result.pose_latency_ms = pose_result.latency_ms

        # Geometric matching -- never positional. Pose output order has no
        # relationship to track order.
        matched = associate_poses_with_tracks(
            tracks, pose_result.detections, self.config.detection.tracking.association
        )
        keypoint_confidence = self.config.detection.inference.keypoint_confidence
        for track in tracks:
            pose = matched.get(track.track_id)
            if pose is None:
                continue
            attach_pose(track, pose, timestamp)
            self.wrists.record_pose(track.track_id, pose, timestamp, keypoint_confidence)

    def _handle_alerts(
        self, observations: list[BehaviorObservation], frame: Frame
    ) -> list[SecurityEvent]:
        """Turn qualifying observations into recorded, dispatched incidents."""
        events: list[SecurityEvent] = []
        for observation in observations:
            if not self.risk_engine.should_alert(observation.risk):
                continue

            decision = self.cooldown.check(
                self.camera_id,
                observation.person_id,
                observation.timestamp,
                observation.risk.risk_score,
            )
            if not decision.allowed:
                self.cooldown.note_suppressed()
                self.metrics.inc(MetricNames.ALERTS_SUPPRESSED_COOLDOWN, **self._labels)
                logger.debug(
                    "alert suppressed by cooldown",
                    extra={
                        "fields": {
                            "camera_id": self.camera_id,
                            "person_id": observation.person_id,
                            "reason": decision.reason,
                            "seconds_remaining": round(decision.seconds_remaining, 1),
                        }
                    },
                )
                continue

            event = build_event(
                observation.risk,
                track_metadata=self._track_metadata(observation, frame),
            )
            if self.recorder is not None:
                snapshot = frame.image if self.config.recording.save_snapshot else None
                event = self.recorder.record(event, self.buffer, snapshot)
            if self.dispatcher is not None:
                self.dispatcher.dispatch(event)

            self.cooldown.record(
                self.camera_id,
                observation.person_id,
                observation.timestamp,
                observation.risk.risk_score,
                event.event_id,
            )
            self.alerts_generated += 1
            events.append(event)

            logger.warning(
                "ALERT: possible concealment behavior - human review recommended",
                extra={
                    "fields": {
                        "event_id": event.event_id,
                        "camera_id": self.camera_id,
                        "person_id": observation.person_id,
                        "risk": observation.risk.risk_score,
                        "threat": observation.risk.threat_level.value,
                        "cooldown_reason": decision.reason,
                    }
                },
            )
        return events

    def _track_metadata(
        self, observation: BehaviorObservation, frame: Frame
    ) -> dict[str, object]:
        track = self.person_tracker.get(observation.person_id)
        item = (
            self.item_tracker.get(observation.associated_item_id)
            if observation.associated_item_id is not None
            else None
        )
        region = None
        for event in observation.active_evidence:
            if "region" in event.metadata:
                region = event.metadata["region"]
                break
        return {
            "track_age_seconds": round(track.track_age, 2) if track else 0.0,
            "track_confidence": round(
                self.person_tracker.track_quality(observation.person_id), 3
            ),
            "associated_item_id": observation.associated_item_id,
            "associated_hand": observation.associated_hand.value
            if observation.associated_hand
            else None,
            "association_seconds": round(item.association_duration, 2) if item else 0.0,
            "storage_region": region,
            "frame_width": frame.width,
            "frame_height": frame.height,
            "bbox": [round(v, 1) for v in track.bbox.as_xyxy()] if track else [],
        }

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.person_tracker.reset()
        self.item_tracker.reset()
        self.associator.reset()
        self.wrists.clear()
        self.behavior_engine.reset()
        self.scheduler.reset()

    @property
    def item_detection_available(self) -> bool:
        return self._item_detection_available

    @property
    def last_inference_latency_ms(self) -> float:
        return self._last_inference_latency

    @property
    def last_pose_latency_ms(self) -> float:
        return self._last_pose_latency


def _as_item(detection, source: str):
    """Convert a COCO object detection into a tracked item candidate."""
    from .core.types import ItemDetection

    return ItemDetection(
        bbox=detection.bbox,
        confidence=detection.confidence,
        object_class=detection.object_class,
        class_name=detection.class_name,
        source=source,
    )


def containers_from_detections(detections: list) -> list[BoundingBox]:
    """Boxes of any detected shopping containers."""
    return [d.bbox for d in detections if d.object_class.is_container]


def blank_frame(camera_id: str, width: int = 640, height: int = 480) -> Frame:
    """A black frame. Used by the benchmark harness and by warmup paths."""
    return Frame(
        camera_id=camera_id,
        frame_id=0,
        timestamp=time.time(),
        image=np.zeros((height, width, 3), dtype=np.uint8),
    )


__all__ = [
    "CameraPipeline",
    "PipelineResult",
    "ObjectClass",
    "blank_frame",
    "containers_from_detections",
]
