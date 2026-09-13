"""Adaptive frame scheduler.

Decode rate, detection rate and pose rate are three separate budgets:

    camera decode   30 FPS   (whatever the stream delivers)
    YOLO detection  10 FPS   (tracking interpolates between detections)
    pose             2-10 FPS conditionally, per person

That separation is the single most important design decision for multi-camera
scaling. 64 cameras at 30 FPS is 1,920 frames per second; running a detector on
all of them is not a tuning problem, it is an impossible one. Running detection
at 10 FPS and pose only on the shoppers who are actually near merchandise
brings it into range -- and costs nothing in detection quality, because the
Kalman tracker carries identity through the gaps.

Pose is escalated when a person is doing something worth watching:

* they are inside or near a merchandise zone;
* their behavior state has left IDLE;
* their risk score is climbing;
* an item is already associated with one of their hands.

All decisions are **time-based**. A frame-count scheduler behaves differently on
a 10 FPS camera than on a 30 FPS one, and a store has both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.config import SchedulerConfig
from ..core.logging import get_logger
from ..core.types import BehaviorState, PersonTrack, ZoneKind
from ..behavior.zones import ZoneRegistry

logger = get_logger(__name__)

MERCHANDISE_KINDS = (ZoneKind.SHELF, ZoneKind.HIGH_VALUE)


@dataclass(slots=True)
class ScheduleDecision:
    """What to run for one frame."""

    run_detection: bool
    run_pose: bool
    #: People pose should run on, highest priority first. Empty means
    #: "full-frame pose" when ``run_pose`` is set and no tracks are known yet.
    pose_targets: list[PersonTrack] = field(default_factory=list)
    #: Effective pose rate chosen this frame, for logging and the overlay.
    pose_fps: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class PersonPriority:
    """Why a person is (or is not) interesting enough for pose."""

    track_id: int
    score: float
    reason: str


class FrameScheduler:
    """Per-camera rate governor for detection and pose."""

    def __init__(self, config: SchedulerConfig, detection_fps_override: float = 0.0) -> None:
        self.config = config
        self._detection_interval = 1.0 / (detection_fps_override or config.detection_fps)
        #: Next scheduled run time, not the last actual one -- see _due().
        self._next_detection: float | None = None
        self._next_pose: float | None = None
        self._current_pose_fps = config.pose_idle_fps if config.adaptive_pose else config.pose_fps

    # -- decisions ---------------------------------------------------------
    def decide(
        self,
        timestamp: float,
        tracks: list[PersonTrack],
        zones: ZoneRegistry | None = None,
        states: dict[int, BehaviorState] | None = None,
        risks: dict[int, float] | None = None,
        associated_ids: set[int] | None = None,
    ) -> ScheduleDecision:
        """Decide what to run for the frame at ``timestamp``."""
        run_detection, next_detection = self._due(
            self._next_detection, timestamp, self._detection_interval
        )

        pose_fps = self._choose_pose_rate(tracks, zones, states, risks, associated_ids)
        self._current_pose_fps = pose_fps
        pose_interval = 1.0 / max(pose_fps, 1e-6)
        run_pose, next_pose = self._due(self._next_pose, timestamp, pose_interval)

        targets: list[PersonTrack] = []
        reason = ""
        if run_pose:
            if self.config.pose_only_for_relevant_people and tracks:
                priorities = self._prioritize(tracks, zones, states, risks, associated_ids)
                selected = priorities[: self.config.max_pose_targets]
                by_id = {t.track_id: t for t in tracks}
                targets = [by_id[p.track_id] for p in selected if p.score > 0.0]
                if not targets:
                    # Nobody is interesting; skip the pose pass entirely rather
                    # than burning a GPU slot on people walking down the aisle.
                    run_pose = False
                    reason = "no relevant people for pose"
                else:
                    reason = selected[0].reason
            else:
                targets = list(tracks)

        if run_detection:
            self._next_detection = next_detection
        if run_pose:
            self._next_pose = next_pose

        return ScheduleDecision(
            run_detection=run_detection,
            run_pose=run_pose,
            pose_targets=targets,
            pose_fps=pose_fps,
            reason=reason,
        )

    def force_pose_next(self) -> None:
        """Make the next frame eligible for pose regardless of rate.

        Used when something just happened that we want pose evidence for --
        a new track entering a high-value zone, for instance.
        """
        self._next_pose = None

    def reset(self) -> None:
        self._next_detection = None
        self._next_pose = None

    @property
    def detection_fps(self) -> float:
        return 1.0 / self._detection_interval

    @property
    def current_pose_fps(self) -> float:
        return self._current_pose_fps

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _due(next_at: float | None, now: float, interval: float) -> tuple[bool, float]:
        """Decide whether a stage is due, on a drift-free schedule.

        Returns ``(is_due, next_scheduled_time)``.

        The schedule advances by fixed intervals rather than re-anchoring to
        each actual run time. Anchoring on the last actual run makes every
        frame's lateness permanent, so the effective rate drifts steadily below
        the configured one -- at 10 FPS requested the stage ends up running at
        7 or 8, which quietly costs detection coverage nobody asked to give up.

        The epsilon matters for the same reason: frame timestamps are floats,
        and ``3 * (1/30)`` is 0.09999999999999999, which a bare ``>=`` rejects.
        """
        epsilon = interval * 1e-6
        if next_at is None:
            return True, now + interval
        if now + epsilon < next_at:
            # A clock going backwards means a source seek; re-anchor rather
            # than stalling until the old schedule catches up.
            if now < next_at - interval * 2:
                return True, now + interval
            return False, next_at

        advanced = next_at + interval
        if advanced <= now:
            # Fell more than a whole interval behind (a stall or an overloaded
            # GPU). Re-anchor instead of firing a catch-up burst, which would
            # make an overload worse.
            advanced = now + interval
        return True, advanced

    def _choose_pose_rate(
        self,
        tracks: list[PersonTrack],
        zones: ZoneRegistry | None,
        states: dict[int, BehaviorState] | None,
        risks: dict[int, float] | None,
        associated_ids: set[int] | None,
    ) -> float:
        if not self.config.adaptive_pose:
            return self.config.pose_fps
        if not tracks:
            return self.config.pose_idle_fps

        priorities = self._prioritize(tracks, zones, states, risks, associated_ids)
        if priorities and priorities[0].score >= 0.5:
            return self.config.pose_active_fps
        if priorities and priorities[0].score > 0.0:
            return self.config.pose_fps
        return self.config.pose_idle_fps

    def _prioritize(
        self,
        tracks: list[PersonTrack],
        zones: ZoneRegistry | None,
        states: dict[int, BehaviorState] | None,
        risks: dict[int, float] | None,
        associated_ids: set[int] | None,
    ) -> list[PersonPriority]:
        """Rank people by how much pose attention they warrant."""
        states = states or {}
        risks = risks or {}
        associated_ids = associated_ids or set()
        priorities: list[PersonPriority] = []

        for track in tracks:
            score = 0.0
            reason = "not relevant"

            if zones is not None and zones.has_merchandise_zones:
                hit = zones.nearest_zone(
                    track.bbox.center, MERCHANDISE_KINDS, track.body_height
                )
                if hit is not None:
                    if hit.inside:
                        score = max(score, 0.8)
                        reason = f"inside merchandise zone {hit.zone.zone_id}"
                    elif hit.normalized_distance <= self.config.pose_relevance_distance_ratio:
                        score = max(score, 0.5)
                        reason = f"near merchandise zone {hit.zone.zone_id}"
            elif zones is None or not zones.has_merchandise_zones:
                # With no zones configured there is nothing to be near, so
                # everybody gets baseline attention rather than nobody.
                score = max(score, 0.4)
                reason = "no zones configured; baseline pose coverage"

            state = states.get(track.track_id, BehaviorState.IDLE)
            if state is not BehaviorState.IDLE and not state.is_benign_terminal:
                score = max(score, 0.9)
                reason = f"behavior state {state.value}"

            risk = risks.get(track.track_id, 0.0)
            if risk >= 30.0:
                score = max(score, 0.95)
                reason = f"risk {risk:.0f}"

            if track.track_id in associated_ids:
                score = 1.0
                reason = "item associated with a hand"

            priorities.append(PersonPriority(track.track_id, score, reason))

        priorities.sort(key=lambda p: -p.score)
        return priorities


class BatchAccumulator:
    """Groups frames from several cameras into one backend invocation.

    Not used by the single-camera MVP -- it would add latency for no gain --
    but present so the multi-camera GPU worker in ``docs/SCALING.md`` is a
    wiring exercise rather than a redesign. The detector's
    ``detect_batch`` already accepts exactly what this produces.
    """

    def __init__(self, max_batch_size: int, timeout_ms: float) -> None:
        self.max_batch_size = max(1, max_batch_size)
        self.timeout_seconds = max(0.0, timeout_ms / 1000.0)
        self._pending: list[tuple] = []
        self._opened_at: float | None = None

    def add(self, item: tuple, now: float) -> list[tuple] | None:
        """Add one frame. Returns a batch when it is ready to dispatch."""
        if self._opened_at is None:
            self._opened_at = now
        self._pending.append(item)
        if len(self._pending) >= self.max_batch_size:
            return self.flush()
        if self.timeout_seconds > 0 and (now - self._opened_at) >= self.timeout_seconds:
            return self.flush()
        return None

    def flush(self) -> list[tuple]:
        batch = self._pending
        self._pending = []
        self._opened_at = None
        return batch

    @property
    def pending_count(self) -> int:
        return len(self._pending)
