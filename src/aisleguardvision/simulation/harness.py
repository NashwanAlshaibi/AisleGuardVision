"""Simulation harness.

Drives a :class:`~aisleguardvision.simulation.scenarios.Scenario` through the
**real** tracking, association and behavior stack. Nothing here is a mock: the
same ByteTrack implementation, the same hand/item associator and the same
temporal behavior engine that run against live video also run here. Only the
neural networks are replaced, by scripted detections.

That distinction matters. A simulator built on mocked-out behavior components
would prove nothing; this one catches real regressions in the logic that
decides whether to page a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..behavior.engine import TemporalBehaviorEngine
from ..behavior.risk import RiskEngine
from ..behavior.zones import ZoneRegistry
from ..core.config import AppConfig
from ..core.types import (
    BehaviorObservation,
    BehaviorState,
    BoundingBox,
    Detection,
    ObjectClass,
    PoseObservation,
    ThreatLevel,
)
from ..tracking.association import HandItemAssociator, WristHistoryStore, associate_poses_with_tracks
from ..tracking.item_tracker import ItemTracker
from ..tracking.person_tracker import PersonTracker, attach_pose
from .scenarios import Scenario, SimFrame

SIM_CAMERA_ID = "sim_cam"


@dataclass(slots=True)
class FrameOutcome:
    """What the pipeline concluded for one simulated frame."""

    timestamp: float
    observations: list[BehaviorObservation] = field(default_factory=list)
    person_count: int = 0
    item_count: int = 0


@dataclass(slots=True)
class SimulationResult:
    """Aggregate outcome of one scenario run."""

    scenario: Scenario
    outcomes: list[FrameOutcome] = field(default_factory=list)
    #: Every state transition observed, as (timestamp, person_id, from, to).
    transitions: list[tuple[float, int, BehaviorState, BehaviorState]] = field(default_factory=list)
    peak_risk: float = 0.0
    peak_threat: ThreatLevel = ThreatLevel.LOW
    peak_state: BehaviorState = BehaviorState.IDLE
    #: Observations whose risk crossed the alert threshold with a valid sequence.
    alerts: list[BehaviorObservation] = field(default_factory=list)
    final_observation: BehaviorObservation | None = None

    @property
    def alerted(self) -> bool:
        return bool(self.alerts)

    @property
    def outcome(self) -> str:
        return "ALERT" if self.alerted else "NO_ALERT"

    @property
    def matches_expectation(self) -> bool:
        return self.outcome == self.scenario.expected

    def evidence_summary(self) -> tuple[list[str], list[str]]:
        """Positive and negative evidence from the highest-risk frame."""
        best: BehaviorObservation | None = None
        for outcome in self.outcomes:
            for observation in outcome.observations:
                if best is None or observation.risk.risk_score > best.risk.risk_score:
                    best = observation
        if best is None:
            return [], []
        return best.risk.positive_evidence, best.risk.negative_evidence


class SimulationRunner:
    """Runs scenarios through the production behavior stack."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.reset()

    def reset(self) -> None:
        detection = self.config.detection
        self.person_tracker = PersonTracker(SIM_CAMERA_ID, detection.tracking.person)
        self.item_tracker = ItemTracker(SIM_CAMERA_ID, detection.tracking.item)
        self.associator = HandItemAssociator(detection.tracking.association)
        self.wrists = WristHistoryStore()
        self.risk_engine = RiskEngine(self.config.risk, self.config.behavior)
        self.engine = TemporalBehaviorEngine(SIM_CAMERA_ID, self.config, self.risk_engine)

    def run(self, scenario: Scenario) -> SimulationResult:
        """Execute one scenario from a clean state."""
        self.reset()
        zones = ZoneRegistry.from_config(SIM_CAMERA_ID, scenario.zones)
        result = SimulationResult(scenario=scenario)

        for frame in scenario.frames:
            outcome = self._step(frame, zones, scenario.item_detection_available)
            result.outcomes.append(outcome)

            for observation in outcome.observations:
                if observation.state_changed:
                    result.transitions.append(
                        (
                            observation.timestamp,
                            observation.person_id,
                            observation.previous_state,
                            observation.state,
                        )
                    )
                if observation.risk.risk_score > result.peak_risk:
                    result.peak_risk = observation.risk.risk_score
                    result.peak_threat = observation.risk.threat_level
                if observation.state is not BehaviorState.IDLE:
                    result.peak_state = _higher_state(result.peak_state, observation.state)
                if self.risk_engine.should_alert(observation.risk):
                    result.alerts.append(observation)
                result.final_observation = observation

        return result

    # -- internals ---------------------------------------------------------
    def _step(
        self, frame: SimFrame, zones: ZoneRegistry, item_detection_available: bool
    ) -> FrameOutcome:
        timestamp = frame.timestamp

        # Person detection -> tracking.
        detections = [
            Detection(
                bbox=person.bbox,
                confidence=person.confidence,
                object_class=ObjectClass.PERSON,
                class_name="person",
            )
            for person in frame.persons
        ]
        tracks = self.person_tracker.update(detections, timestamp)

        # Pose -> geometric association with tracks (never positional).
        pose_detections = [
            Detection(
                bbox=person.bbox,
                confidence=person.confidence,
                object_class=ObjectClass.PERSON,
                pose=PoseObservation.from_array(person.keypoints, person.bbox, timestamp),
            )
            for person in frame.persons
        ]
        matched = associate_poses_with_tracks(
            tracks, pose_detections, self.config.detection.tracking.association
        )
        keypoint_confidence = self.config.detection.inference.keypoint_confidence
        for track in tracks:
            pose = matched.get(track.track_id)
            if pose is not None:
                attach_pose(track, pose, timestamp)
                self.wrists.record_pose(track.track_id, pose, timestamp, keypoint_confidence)

        # Item detection -> item tracking -> hand association.
        item_tracks = self.item_tracker.update(frame.items, timestamp)
        associations = self.associator.associate(
            tracks, item_tracks, self.wrists, timestamp, keypoint_confidence
        )

        container_boxes: list[BoundingBox] = [
            item.bbox for item in item_tracks if item.object_class.is_container
        ]

        observations = self.engine.update(
            timestamp=timestamp,
            person_tracks=tracks,
            item_tracks=item_tracks,
            associations=associations,
            zones=zones,
            wrists=self.wrists,
            item_tracker=self.item_tracker,
            track_quality={t.track_id: self.person_tracker.track_quality(t.track_id) for t in tracks},
            item_detection_available=item_detection_available,
            container_boxes=container_boxes,
        )

        self.wrists.prune({t.track_id for t in tracks})
        return FrameOutcome(
            timestamp=timestamp,
            observations=observations,
            person_count=len(tracks),
            item_count=len(item_tracks),
        )


_STATE_RANK = {
    BehaviorState.IDLE: 0,
    BehaviorState.ITEM_RETURNED: 1,
    BehaviorState.ITEM_TO_BASKET: 1,
    BehaviorState.ITEM_TO_CART: 1,
    BehaviorState.SHELF_INTERACTION: 2,
    BehaviorState.ITEM_ASSOCIATED: 3,
    BehaviorState.ITEM_REMOVED_FROM_SHELF: 4,
    BehaviorState.HAND_MOVING_TO_STORAGE: 5,
    BehaviorState.POSSIBLE_CONCEALMENT: 6,
    BehaviorState.ITEM_OCCLUDED: 7,
    BehaviorState.ITEM_MISSING: 8,
    BehaviorState.REVIEW_ALERT: 9,
}


def _higher_state(a: BehaviorState, b: BehaviorState) -> BehaviorState:
    return a if _STATE_RANK.get(a, 0) >= _STATE_RANK.get(b, 0) else b


def run_scenario(scenario: Scenario, config: AppConfig | None = None) -> SimulationResult:
    """Convenience wrapper: run one scenario with a fresh runner."""
    return SimulationRunner(config).run(scenario)
