"""Association: pose-to-person and item-to-wrist.

Two association problems, both of which are easy to get wrong in ways that
silently corrupt everything downstream:

**Pose to person.** The pose model returns poses in its own order. Assuming
that order matches the detector's order -- or the tracker's -- attributes one
shopper's arms to another. Poses are matched geometrically (IoU, containment,
and keypoint-in-box fraction) and unmatched poses are dropped rather than
guessed at.

**Item to wrist.** A single frame of proximity means nothing: hands pass in
front of shelved products constantly. Association combines four independent
channels -- normalized distance, IoU against a body-scaled hand box, motion
similarity, and temporal persistence -- and only becomes "stable" after a
configurable continuous dwell.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from ..behavior.geometry import hand_box, trajectory_similarity
from ..core.config import AssociationConfig
from ..core.logging import get_logger
from ..core.types import (
    BoundingBox,
    Detection,
    Hand,
    ItemStatus,
    ItemTrack,
    PersonTrack,
    Point,
    PoseObservation,
    TrajectoryPoint,
)
from .kalman import greedy_match

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wrist history
# ---------------------------------------------------------------------------


class WristHistoryStore:
    """Per-person, per-hand wrist trajectories.

    Kept out of :class:`PersonTrack` on purpose: wrist history is only
    meaningful for people the pose scheduler actually ran pose on, and the
    behavior engine and the associator both need it. One owner, two readers.
    """

    def __init__(self, max_points: int = 256) -> None:
        self._max_points = max_points
        self._history: dict[tuple[int, Hand], deque[TrajectoryPoint]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )

    def record(self, person_id: int, hand: Hand, position: Point, timestamp: float) -> None:
        self._history[(person_id, hand)].append(TrajectoryPoint(timestamp, position))

    def record_pose(
        self, person_id: int, pose: PoseObservation, timestamp: float, min_confidence: float
    ) -> None:
        for hand in Hand:
            point = pose.wrist(hand, min_confidence)
            if point is not None:
                self.record(person_id, hand, point, timestamp)

    def trajectory(
        self, person_id: int, hand: Hand, window_seconds: float, now: float
    ) -> list[TrajectoryPoint]:
        points = self._history.get((person_id, hand))
        if not points:
            return []
        cutoff = now - window_seconds
        return [p for p in points if p.timestamp >= cutoff]

    def latest(self, person_id: int, hand: Hand) -> TrajectoryPoint | None:
        points = self._history.get((person_id, hand))
        return points[-1] if points else None

    def prune(self, active_person_ids: set[int]) -> None:
        """Drop history for people no longer tracked, bounding memory growth."""
        for key in list(self._history.keys()):
            if key[0] not in active_person_ids:
                del self._history[key]

    def clear(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Pose to person
# ---------------------------------------------------------------------------


def pose_person_score(
    track_box: BoundingBox, pose: PoseObservation, pose_box: BoundingBox | None
) -> float:
    """Geometric agreement between a pose and a person box, in ``[0, 1]``.

    IoU alone is unreliable: a pose model's implied box is often tighter than
    a detector's, and an extended arm stretches one but not the other. The
    fraction of confident keypoints falling inside the person box is a far
    more stable signal, so it carries the most weight.
    """
    box = pose_box if pose_box is not None else pose.bbox
    iou = track_box.iou(box) if box is not None else 0.0
    containment = track_box.containment_of(box) if box is not None else 0.0

    points = [
        (kp.x, kp.y)
        for kp in pose.keypoints.values()
        if kp.confidence >= 0.2 and not (kp.x == 0.0 and kp.y == 0.0)
    ]
    if points:
        array = np.array(points)
        expanded = track_box.expanded(0.15)
        inside = (
            (array[:, 0] >= expanded.x1)
            & (array[:, 0] <= expanded.x2)
            & (array[:, 1] >= expanded.y1)
            & (array[:, 1] <= expanded.y2)
        )
        keypoint_fraction = float(inside.mean())
    else:
        keypoint_fraction = 0.0

    return 0.25 * iou + 0.25 * containment + 0.50 * keypoint_fraction


def associate_poses_with_tracks(
    tracks: list[PersonTrack],
    pose_detections: list[Detection],
    config: AssociationConfig,
) -> dict[int, PoseObservation]:
    """Match pose detections to tracked people.

    Never assumes detection order matches pose order. Returns a mapping of
    ``track_id -> PoseObservation``; poses that match nothing are discarded.
    """
    poses = [(d.pose, d.bbox) for d in pose_detections if d.pose is not None]
    if not tracks or not poses:
        return {}

    scores = np.zeros((len(tracks), len(poses)), dtype=np.float64)
    for i, track in enumerate(tracks):
        for j, (pose, pose_box) in enumerate(poses):
            scores[i, j] = pose_person_score(track.bbox, pose, pose_box)

    # The blended score is dominated by keypoint containment, so the accept
    # threshold is expressed against it rather than against raw IoU.
    threshold = min(config.pose_min_containment, 0.5)
    matches, _, unmatched_poses = greedy_match(scores, threshold)

    if unmatched_poses:
        logger.debug(
            "unmatched poses discarded",
            extra={"fields": {"count": len(unmatched_poses), "tracks": len(tracks)}},
        )

    result: dict[int, PoseObservation] = {}
    for track_index, pose_index in matches:
        pose, pose_box = poses[pose_index]
        if pose.bbox is None:
            pose.bbox = pose_box
        result[tracks[track_index].track_id] = pose
    return result


# ---------------------------------------------------------------------------
# Item to wrist
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AssociationResult:
    """A scored hand/item link for one frame."""

    person_id: int
    item_id: int
    hand: Hand
    confidence: float
    #: Continuous seconds this link has held. Zero on the frame it is created.
    duration: float
    #: Per-channel scores, retained for explainability and tuning.
    channels: dict[str, float] = field(default_factory=dict)
    #: True once ``duration`` exceeds the configured minimum.
    is_stable: bool = False


class HandItemAssociator:
    """Associates merchandise candidates with wrists over time.

    All distances are normalized by the subject's body height, so one
    configuration covers the whole depth of an aisle.
    """

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()
        #: (person_id, hand, item_id) -> timestamp the link was first seen.
        self._link_since: dict[tuple[int, Hand, int], float] = {}
        #: (person_id, hand, item_id) -> timestamp the link was last seen.
        self._link_last: dict[tuple[int, Hand, int], float] = {}

    def associate(
        self,
        tracks: list[PersonTrack],
        items: list[ItemTrack],
        wrists: WristHistoryStore,
        timestamp: float,
        keypoint_confidence: float = 0.3,
    ) -> list[AssociationResult]:
        """Compute this frame's hand/item links.

        Returns at most one link per (person, hand) and at most one per item:
        a hand holding two tracked items at once is possible in reality but
        modelling it would multiply ambiguity for no MVP benefit.
        """
        candidates = self._score_candidates(tracks, items, wrists, timestamp, keypoint_confidence)
        if not candidates:
            self._expire_links(timestamp, keep=set())
            return []

        # Resolve to a one-to-one assignment between hands and items.
        hands = sorted({(c.person_id, c.hand) for c in candidates})
        item_ids = sorted({c.item_id for c in candidates})
        hand_index = {key: i for i, key in enumerate(hands)}
        item_index = {key: i for i, key in enumerate(item_ids)}

        scores = np.zeros((len(hands), len(item_ids)), dtype=np.float64)
        lookup: dict[tuple[int, int], AssociationResult] = {}
        for candidate in candidates:
            i = hand_index[(candidate.person_id, candidate.hand)]
            j = item_index[candidate.item_id]
            if candidate.confidence > scores[i, j]:
                scores[i, j] = candidate.confidence
                lookup[(i, j)] = candidate

        matches, _, _ = greedy_match(scores, self.config.min_association_confidence)

        results: list[AssociationResult] = []
        live_keys: set[tuple[int, Hand, int]] = set()
        for i, j in matches:
            candidate = lookup[(i, j)]
            key = (candidate.person_id, candidate.hand, candidate.item_id)
            live_keys.add(key)

            last = self._link_last.get(key)
            # A gap longer than the dwell requirement restarts the clock:
            # otherwise two brief, unrelated touches minutes apart would sum
            # into a "stable" association.
            if last is None or (timestamp - last) > self.config.min_association_seconds:
                self._link_since[key] = timestamp
            self._link_last[key] = timestamp

            candidate.duration = timestamp - self._link_since[key]
            candidate.is_stable = candidate.duration >= self.config.min_association_seconds
            results.append(candidate)

        self._expire_links(timestamp, keep=live_keys)
        return results

    def link_duration(self, person_id: int, hand: Hand, item_id: int, now: float) -> float:
        since = self._link_since.get((person_id, hand, item_id))
        return 0.0 if since is None else max(0.0, now - since)

    def reset(self) -> None:
        self._link_since.clear()
        self._link_last.clear()

    # -- internals ---------------------------------------------------------
    def _score_candidates(
        self,
        tracks: list[PersonTrack],
        items: list[ItemTrack],
        wrists: WristHistoryStore,
        timestamp: float,
        keypoint_confidence: float,
    ) -> list[AssociationResult]:
        config = self.config
        w_distance, w_iou, w_motion, w_persistence = config.normalized_weights()

        # Only items currently observable can start or sustain an association.
        observable = [
            item
            for item in items
            if item.status in (ItemStatus.VISIBLE, ItemStatus.POSSIBLY_OCCLUDED)
        ]
        if not observable:
            return []

        candidates: list[AssociationResult] = []
        for track in tracks:
            pose = track.pose
            if pose is None:
                continue
            body_height = track.body_height
            if body_height <= 1.0:
                continue

            for hand in Hand:
                wrist = pose.wrist(hand, keypoint_confidence)
                if wrist is None:
                    continue
                hbox = hand_box(wrist, body_height)
                wrist_path = wrists.trajectory(
                    track.track_id, hand, config.motion_window_seconds, timestamp
                )

                for item in observable:
                    distance_ratio = wrist.distance_to(item.center) / body_height
                    if distance_ratio > config.hand_item_max_distance_ratio:
                        continue

                    # Channel 1: normalized proximity, 1.0 at the "in hand"
                    # distance, decaying linearly to 0 at the cut-off.
                    if distance_ratio <= config.hand_item_distance_ratio:
                        distance_score = 1.0
                    else:
                        span = config.hand_item_max_distance_ratio - config.hand_item_distance_ratio
                        distance_score = max(
                            0.0, 1.0 - (distance_ratio - config.hand_item_distance_ratio) / span
                        )

                    # Channel 2: overlap between the hand box and the item box.
                    overlap = hbox.iou(item.bbox)
                    contained = hbox.containment_of(item.bbox)
                    iou_score = float(np.clip(max(overlap / 0.35, contained), 0.0, 1.0))

                    # Channel 3: does the item move *with* the wrist? This is
                    # what separates holding an item from standing in front of
                    # one on a shelf.
                    item_path = [
                        p
                        for p in item.trajectory
                        if p.timestamp >= timestamp - config.motion_window_seconds
                    ]
                    motion_score = trajectory_similarity(wrist_path, item_path, body_height)

                    # Channel 4: how long this exact link has already held.
                    prior = self.link_duration(track.track_id, hand, item.item_id, timestamp)
                    persistence_score = (
                        float(np.clip(prior / max(config.min_association_seconds, 1e-3), 0.0, 1.0))
                        if config.min_association_seconds > 0
                        else 1.0
                    )

                    confidence = (
                        w_distance * distance_score
                        + w_iou * iou_score
                        + w_motion * motion_score
                        + w_persistence * persistence_score
                    )
                    # A link with no spatial support at all is not a link,
                    # regardless of how the other channels score.
                    if distance_score <= 0.0 and iou_score <= 0.0:
                        continue

                    candidates.append(
                        AssociationResult(
                            person_id=track.track_id,
                            item_id=item.item_id,
                            hand=hand,
                            confidence=float(np.clip(confidence, 0.0, 1.0)),
                            duration=prior,
                            channels={
                                "distance": round(distance_score, 3),
                                "iou": round(iou_score, 3),
                                "motion": round(motion_score, 3),
                                "persistence": round(persistence_score, 3),
                                "distance_ratio": round(distance_ratio, 3),
                            },
                        )
                    )
        return candidates

    def _expire_links(self, timestamp: float, keep: set[tuple[int, Hand, int]]) -> None:
        """Forget links that have not been observed for well over the dwell window."""
        horizon = max(self.config.min_association_seconds * 4.0, 2.0)
        for key, last in list(self._link_last.items()):
            if key in keep:
                continue
            if (timestamp - last) > horizon:
                self._link_last.pop(key, None)
                self._link_since.pop(key, None)
