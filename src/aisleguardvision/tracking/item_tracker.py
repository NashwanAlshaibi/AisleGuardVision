"""Merchandise-candidate tracking and the disappearance ladder.

The critical design rule of this module:

    **An object disappearing is never, by itself, evidence of concealment.**

Objects leave the frame, get occluded by the shopper's own body, get occluded
by another shopper, fall below the detector's confidence threshold for a few
frames, or were never a stable track to begin with. So disappearance is
modelled as a timed ladder --

    VISIBLE -> POSSIBLY_OCCLUDED -> OCCLUDED -> MISSING

-- and reaching MISSING is one input among many to the behavior engine, which
additionally requires that the item was reliably tracked, was consistently
associated with the same wrist, moved with that wrist, and vanished near a
plausible storage region with no basket placement and no shelf return.

This module owns the ladder and the association bookkeeping. It deliberately
knows nothing about zones or risk: benign resolutions (RETURNED, IN_BASKET,
IN_CART) are pushed in by the behavior engine via :meth:`ItemTracker.resolve`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..core.config import ItemTrackerConfig
from ..core.logging import get_logger
from ..core.types import (
    Hand,
    ItemDetection,
    ItemStatus,
    ItemTrack,
    ObjectClass,
    TrajectoryPoint,
)
from .kalman import greedy_match, iou_matrix

logger = get_logger(__name__)


@dataclass(slots=True)
class ItemStatusChange:
    """Emitted when an item track changes observation state."""

    item_id: int
    previous: ItemStatus
    current: ItemStatus
    timestamp: float
    #: Seconds the item has been unobserved at the time of the change.
    unobserved_for: float


class ItemTracker:
    """Tracks merchandise candidates for one camera.

    Association gating combines IoU with a centre-displacement limit expressed
    in item widths. Retail items are small and the detector's boxes jitter, so
    IoU alone either matches nothing (tight threshold) or matches two adjacent
    products on a shelf face to each other (loose threshold).
    """

    def __init__(
        self,
        camera_id: str,
        config: ItemTrackerConfig | None = None,
        trajectory_length: int = 256,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or ItemTrackerConfig()
        self._trajectory_length = trajectory_length
        self._tracks: dict[int, ItemTrack] = {}
        self._next_id = 1
        self._last_changes: list[ItemStatusChange] = []

    # -- public API --------------------------------------------------------
    def update(self, detections: list[ItemDetection], timestamp: float) -> list[ItemTrack]:
        """Advance the tracker by one detection frame."""
        self._last_changes = []
        tracks = list(self._tracks.values())

        matches, unmatched_tracks, unmatched_detections = self._associate(tracks, detections)

        for track_index, detection_index in matches:
            self._update_track(tracks[track_index], detections[detection_index], timestamp)

        for track_index in unmatched_tracks:
            self._advance_ladder(tracks[track_index], timestamp)

        for detection_index in unmatched_detections:
            self._create_track(detections[detection_index], timestamp)

        self._retire(timestamp)
        return list(self._tracks.values())

    @property
    def tracks(self) -> list[ItemTrack]:
        return list(self._tracks.values())

    @property
    def status_changes(self) -> list[ItemStatusChange]:
        """Status transitions produced by the most recent :meth:`update`."""
        return list(self._last_changes)

    def get(self, item_id: int) -> ItemTrack | None:
        return self._tracks.get(item_id)

    def active_count(self) -> int:
        return sum(
            1 for t in self._tracks.values() if t.status in (ItemStatus.VISIBLE, ItemStatus.POSSIBLY_OCCLUDED)
        )

    def resolve(self, item_id: int, status: ItemStatus, timestamp: float) -> None:
        """Record a benign resolution determined by the behavior engine.

        Only the benign terminal statuses may be pushed in this way; the
        occlusion ladder itself is owned here and is time-driven.
        """
        if not status.is_resolved_benign:
            raise ValueError(f"{status} is not a benign resolution; the ladder owns the rest")
        track = self._tracks.get(item_id)
        if track is None or track.status is status:
            return
        previous = track.status
        track.status = status
        self._last_changes.append(
            ItemStatusChange(item_id, previous, status, timestamp, unobserved_for=0.0)
        )
        logger.debug(
            "item resolved benignly",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "item_id": item_id,
                    "from": previous.value,
                    "to": status.value,
                }
            },
        )

    def set_association(
        self,
        item_id: int,
        person_id: int | None,
        hand: Hand | None,
        confidence: float,
        timestamp: float,
    ) -> None:
        """Record (or clear) which wrist an item is believed to be held by.

        ``association_since`` is only reset when the *identity* of the
        association changes, so that a stable link accumulates dwell time
        across frames -- that dwell time is what distinguishes holding an item
        from a hand passing in front of one.
        """
        track = self._tracks.get(item_id)
        if track is None:
            return
        if person_id is None:
            track.associated_person_id = None
            track.associated_hand = None
            track.association_confidence = 0.0
            track.association_since = None
            return
        changed = track.associated_person_id != person_id or track.associated_hand != hand
        track.associated_person_id = person_id
        track.associated_hand = hand
        track.association_confidence = confidence
        if changed or track.association_since is None:
            track.association_since = timestamp

    def reset(self) -> None:
        self._tracks.clear()
        self._last_changes = []

    # -- internals ---------------------------------------------------------
    def _associate(
        self, tracks: list[ItemTrack], detections: list[ItemDetection]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        scores = iou_matrix([t.bbox for t in tracks], [d.bbox for d in detections])

        # Centre-displacement gate, in item widths. Kills the "adjacent product
        # on the same shelf face" match that pure IoU is prone to.
        max_ratio = self.config.max_center_displacement_ratio
        for i, track in enumerate(tracks):
            reference = max(track.bbox.width, track.bbox.height, 1.0)
            for j, detection in enumerate(detections):
                if scores[i, j] <= 0:
                    continue
                displacement = track.center.distance_to(detection.bbox.center)
                if displacement > reference * max_ratio:
                    scores[i, j] = 0.0
                # Different semantic classes never match: a phone must never
                # inherit a merchandise track's history, or the phone
                # false-positive suppression could be bypassed by a swap.
                if detection.object_class is not track.object_class:
                    scores[i, j] = 0.0

        return greedy_match(scores, self.config.match_iou_threshold)

    def _update_track(self, track: ItemTrack, detection: ItemDetection, timestamp: float) -> None:
        previous_status = track.status
        track.bbox = detection.bbox
        track.confidence = detection.confidence
        track.last_seen = timestamp
        track.hit_count += 1
        track.trajectory.append(TrajectoryPoint(timestamp, detection.bbox.center))
        track.last_visible_position = detection.bbox.center
        track.last_visible_timestamp = timestamp
        track.disappeared_at = None

        # Re-observing an item cancels any occlusion state. A benign terminal
        # status is sticky: an item seen again after being placed in a basket
        # is still "in the basket" as far as this episode is concerned.
        if not track.status.is_resolved_benign and track.status is not ItemStatus.VISIBLE:
            track.status = ItemStatus.VISIBLE
            self._last_changes.append(
                ItemStatusChange(track.item_id, previous_status, ItemStatus.VISIBLE, timestamp, 0.0)
            )

    def _advance_ladder(self, track: ItemTrack, timestamp: float) -> None:
        """Move an unobserved item one rung down the ladder, if enough time has passed."""
        if track.status.is_resolved_benign:
            return
        if track.disappeared_at is None:
            track.disappeared_at = timestamp
        unobserved = timestamp - track.disappeared_at

        config = self.config
        if unobserved >= config.missing_after_seconds:
            target = ItemStatus.MISSING
        elif unobserved >= config.occluded_after_seconds:
            target = ItemStatus.OCCLUDED
        elif unobserved >= config.possibly_occluded_after_seconds:
            target = ItemStatus.POSSIBLY_OCCLUDED
        else:
            return

        if track.status is target:
            return
        previous = track.status
        track.status = target
        self._last_changes.append(
            ItemStatusChange(track.item_id, previous, target, timestamp, unobserved)
        )
        logger.debug(
            "item status advanced",
            extra={
                "fields": {
                    "camera_id": self.camera_id,
                    "item_id": track.item_id,
                    "from": previous.value,
                    "to": target.value,
                    "unobserved_s": round(unobserved, 2),
                }
            },
        )

    def _create_track(self, detection: ItemDetection, timestamp: float) -> None:
        item_id = self._next_id
        self._next_id += 1
        track = ItemTrack(
            item_id=item_id,
            camera_id=self.camera_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            first_seen=timestamp,
            last_seen=timestamp,
            object_class=detection.object_class,
            status=ItemStatus.VISIBLE,
            source=detection.source,
            trajectory=deque(maxlen=self._trajectory_length),
            hit_count=1,
            last_visible_position=detection.bbox.center,
            last_visible_timestamp=timestamp,
        )
        track.trajectory.append(TrajectoryPoint(timestamp, detection.bbox.center))
        self._tracks[item_id] = track

    def _retire(self, timestamp: float) -> None:
        for item_id, track in list(self._tracks.items()):
            if track.disappeared_at is None:
                continue
            if (timestamp - track.disappeared_at) > self.config.remove_after_seconds:
                self._tracks.pop(item_id, None)


def is_phone(track: ItemTrack) -> bool:
    """Phones get their own predicate because they drive the single most
    important false-positive suppression in the system."""
    return track.object_class is ObjectClass.PHONE


def is_personal_effect(track: ItemTrack) -> bool:
    return track.object_class.is_personal_effect
