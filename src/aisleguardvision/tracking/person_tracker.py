"""ByteTrack person tracking.

Implemented in-repo rather than delegating to Ultralytics' built-in tracker,
for three reasons:

1. Ultralytics' tracker consumes its own ``Results`` objects. Depending on it
   would reintroduce exactly the coupling the backend abstraction exists to
   prevent -- a TensorRT, ONNX, Triton or DeepStream backend could not reuse it.
2. It is frame-indexed. Our cameras run at different and varying frame rates,
   so track lifetimes must be expressed in **seconds**.
3. Tracking behavior is safety-relevant here (an ID switch can attribute one
   shopper's item to another), so it needs to be unit-testable without a GPU
   or model weights.

The algorithm is ByteTrack (Zhang et al., 2022): associate high-confidence
detections first, then recover tracks using the low-confidence detections that
a plain confidence threshold would have discarded. That second pass is what
carries a track through partial occlusion -- a shopper stepping behind a
display, which happens constantly in a store aisle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..core.config import ByteTrackConfig
from ..core.logging import get_logger
from ..core.types import (
    BoundingBox,
    Detection,
    ObjectClass,
    PersonTrack,
    Point,
    TrackState,
    TrajectoryPoint,
)
from .kalman import KalmanBoxFilter, greedy_match, iou_matrix

logger = get_logger(__name__)


@dataclass(slots=True)
class _TrackedPerson:
    """Internal tracker bookkeeping wrapped around the public PersonTrack."""

    track: PersonTrack
    kalman: KalmanBoxFilter
    #: Wall-clock time of the most recent successful update.
    last_update_time: float
    #: Wall-clock time of the most recent predict step.
    last_predict_time: float
    #: Rolling detection confidences, used for the track-quality signal.
    recent_confidences: deque[float] = field(default_factory=lambda: deque(maxlen=30))

    @property
    def track_id(self) -> int:
        return self.track.track_id


class PersonTracker:
    """Multi-object tracker for one camera.

    Not thread-safe: one instance belongs to one camera's analysis pipeline.
    """

    def __init__(
        self,
        camera_id: str,
        config: ByteTrackConfig | None = None,
        trajectory_length: int = 256,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or ByteTrackConfig()
        self._trajectory_length = trajectory_length
        self._active: dict[int, _TrackedPerson] = {}
        self._lost: dict[int, _TrackedPerson] = {}
        self._next_id = 1
        self._last_timestamp: float | None = None
        #: Incremented whenever a track is removed, for metrics/diagnostics.
        self.removed_count = 0

    # -- public API --------------------------------------------------------
    def update(self, detections: list[Detection], timestamp: float) -> list[PersonTrack]:
        """Advance the tracker by one detection frame.

        ``detections`` should already be filtered to people. Returns all
        currently confirmed tracks.
        """
        person_detections = [d for d in detections if d.object_class is ObjectClass.PERSON] or list(
            detections
        )

        dt = self._delta(timestamp)
        self._predict_all(dt, timestamp)

        high, low = self._split_by_confidence(person_detections)

        # -- pass 1: active tracks vs high-confidence detections -----------
        active = list(self._active.values())
        matched, unmatched_tracks, unmatched_high = self._associate(
            active, high, self.config.match_iou_threshold
        )
        for track_index, detection_index in matched:
            self._update_track(active[track_index], high[detection_index], timestamp)

        # -- pass 2: still-unmatched active tracks vs low-confidence dets ---
        # This is the ByteTrack contribution: a shopper half behind a display
        # produces a weak box that a plain threshold would throw away, taking
        # the track (and any item association built on it) with it.
        remaining = [active[i] for i in unmatched_tracks]
        matched_low, unmatched_tracks_2, _ = self._associate(
            remaining, low, self.config.second_match_iou_threshold
        )
        for track_index, detection_index in matched_low:
            self._update_track(remaining[track_index], low[detection_index], timestamp)

        # -- pass 3: revive lost tracks with leftover high-confidence dets --
        leftover_high = [high[i] for i in unmatched_high]
        lost = list(self._lost.values())
        matched_lost, _, unmatched_leftover = self._associate(
            lost, leftover_high, self.config.revive_iou_threshold
        )
        for track_index, detection_index in matched_lost:
            revived = lost[track_index]
            self._update_track(revived, leftover_high[detection_index], timestamp)
            self._lost.pop(revived.track_id, None)
            self._active[revived.track_id] = revived
            logger.debug(
                "track revived after occlusion",
                extra={"fields": {"camera_id": self.camera_id, "track_id": revived.track_id}},
            )

        # -- unmatched tracks age out --------------------------------------
        for track_index in unmatched_tracks_2:
            self._mark_lost(remaining[track_index], timestamp)

        # -- spawn new tracks ----------------------------------------------
        for detection_index in unmatched_leftover:
            detection = leftover_high[detection_index]
            if detection.confidence >= self.config.new_track_threshold:
                self._create_track(detection, timestamp)

        self._retire_stale(timestamp)
        return self.confirmed_tracks()

    def confirmed_tracks(self) -> list[PersonTrack]:
        """Tracks that have survived long enough to be trusted."""
        return [t.track for t in self._active.values() if t.track.state is TrackState.CONFIRMED]

    def all_tracks(self, include_lost: bool = False) -> list[PersonTrack]:
        tracks = [t.track for t in self._active.values()]
        if include_lost:
            tracks.extend(t.track for t in self._lost.values())
        return tracks

    def get(self, track_id: int) -> PersonTrack | None:
        holder = self._active.get(track_id) or self._lost.get(track_id)
        return holder.track if holder else None

    def track_quality(self, track_id: int) -> float:
        """Mean recent detection confidence, a proxy for track reliability.

        Feeds LOW_PERSON_TRACK_CONFIDENCE negative evidence.
        """
        holder = self._active.get(track_id) or self._lost.get(track_id)
        if holder is None or not holder.recent_confidences:
            return 0.0
        return sum(holder.recent_confidences) / len(holder.recent_confidences)

    def reset(self) -> None:
        self._active.clear()
        self._lost.clear()
        self._last_timestamp = None

    @property
    def active_count(self) -> int:
        return len(self._active)

    # -- internals ---------------------------------------------------------
    def _delta(self, timestamp: float) -> float:
        if self._last_timestamp is None:
            dt = 1.0 / 30.0
        else:
            dt = timestamp - self._last_timestamp
            if dt <= 0:
                # A non-monotonic timestamp means a source seek or a clock
                # adjustment. Fall back to a nominal step instead of feeding a
                # negative dt into the filter.
                dt = 1.0 / 30.0
        self._last_timestamp = timestamp
        return dt

    def _predict_all(self, dt: float, timestamp: float) -> None:
        for holder in list(self._active.values()) + list(self._lost.values()):
            step = timestamp - holder.last_predict_time
            holder.kalman.predict(step if step > 0 else dt)
            holder.last_predict_time = timestamp
            holder.track.bbox = holder.kalman.box
            holder.track.velocity = holder.kalman.velocity

    def _split_by_confidence(
        self, detections: list[Detection]
    ) -> tuple[list[Detection], list[Detection]]:
        high = [d for d in detections if d.confidence >= self.config.high_threshold]
        low = [
            d
            for d in detections
            if self.config.low_threshold <= d.confidence < self.config.high_threshold
        ]
        return high, low

    def _associate(
        self, tracks: list[_TrackedPerson], detections: list[Detection], threshold: float
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        track_boxes = [t.track.bbox for t in tracks]
        detection_boxes = [d.bbox for d in detections]
        return greedy_match(iou_matrix(track_boxes, detection_boxes), threshold)

    def _update_track(self, holder: _TrackedPerson, detection: Detection, timestamp: float) -> None:
        holder.kalman.update(detection.bbox)
        track = holder.track
        track.bbox = holder.kalman.box
        track.velocity = holder.kalman.velocity
        track.confidence = detection.confidence
        track.last_seen = timestamp
        track.hit_count += 1
        track.time_since_update = 0
        track.trajectory.append(TrajectoryPoint(timestamp, track.bbox.center))
        holder.last_update_time = timestamp
        holder.recent_confidences.append(detection.confidence)

        if track.state is TrackState.TENTATIVE and track.hit_count >= self.config.min_hits:
            track.state = TrackState.CONFIRMED
            logger.debug(
                "track confirmed",
                extra={"fields": {"camera_id": self.camera_id, "track_id": track.track_id}},
            )
        elif track.state is TrackState.LOST:
            track.state = TrackState.CONFIRMED

    def _create_track(self, detection: Detection, timestamp: float) -> None:
        track_id = self._next_id
        self._next_id += 1
        track = PersonTrack(
            track_id=track_id,
            camera_id=self.camera_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            first_seen=timestamp,
            last_seen=timestamp,
            state=TrackState.TENTATIVE if self.config.min_hits > 1 else TrackState.CONFIRMED,
            trajectory=deque(maxlen=self._trajectory_length),
            hit_count=1,
        )
        track.trajectory.append(TrajectoryPoint(timestamp, detection.bbox.center))
        holder = _TrackedPerson(
            track=track,
            kalman=KalmanBoxFilter(detection.bbox),
            last_update_time=timestamp,
            last_predict_time=timestamp,
        )
        holder.recent_confidences.append(detection.confidence)
        self._active[track_id] = holder

    def _mark_lost(self, holder: _TrackedPerson, timestamp: float) -> None:
        track = holder.track
        track.time_since_update += 1
        if track.state is TrackState.TENTATIVE:
            # A tentative track that missed a frame was probably a detector
            # artifact. Drop it immediately rather than letting noise linger.
            self._active.pop(track.track_id, None)
            self.removed_count += 1
            return
        track.state = TrackState.LOST
        self._active.pop(track.track_id, None)
        self._lost[track.track_id] = holder

    def _retire_stale(self, timestamp: float) -> None:
        max_lost = self.config.max_lost_seconds
        for track_id, holder in list(self._lost.items()):
            if (timestamp - holder.last_update_time) > max_lost:
                holder.track.state = TrackState.REMOVED
                self._lost.pop(track_id, None)
                self.removed_count += 1
                logger.debug(
                    "track removed",
                    extra={
                        "fields": {
                            "camera_id": self.camera_id,
                            "track_id": track_id,
                            "age": round(holder.track.track_age, 2),
                        }
                    },
                )


def attach_pose(track: PersonTrack, pose, timestamp: float) -> None:
    """Attach a pose observation to a track, recording when it was observed.

    Kept as a free function so the pose association module does not need to
    reach into tracker internals.
    """
    track.pose = pose
    track.pose_timestamp = timestamp


def predicted_position(track: PersonTrack, horizon_seconds: float) -> Point:
    """Extrapolate a track centre forward, for gating and visualization."""
    centre = track.bbox.center
    return Point(
        centre.x + track.velocity.x * horizon_seconds,
        centre.y + track.velocity.y * horizon_seconds,
    )


def boxes_of(tracks: list[PersonTrack]) -> list[BoundingBox]:
    return [t.bbox for t in tracks]
