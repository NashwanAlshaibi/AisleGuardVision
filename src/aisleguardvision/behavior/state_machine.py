"""Per-person behavior state machine.

The machine advances along a concealment *sequence* and branches out of it the
moment a benign explanation appears:

    IDLE
      -> SHELF_INTERACTION
      -> ITEM_ASSOCIATED  --+--> ITEM_RETURNED  --> IDLE
                            +--> ITEM_TO_BASKET --> IDLE
                            +--> ITEM_TO_CART   --> IDLE
      -> ITEM_REMOVED_FROM_SHELF
      -> HAND_MOVING_TO_STORAGE
      -> POSSIBLE_CONCEALMENT
      -> ITEM_OCCLUDED
      -> ITEM_MISSING
      -> REVIEW_ALERT

Two invariants:

* **Progress is monotonic within an episode.** A momentarily lost keypoint must
  not knock the sequence back a rung and then re-raise the same evidence,
  which would double-count it.
* **Benign branches are reachable from every sequence state, including
  ITEM_MISSING.** An item that reappears in a basket after two seconds of
  occlusion resolves benignly, however far along the sequence had progressed.

The machine only orders states. It never decides whether an alert is warranted
-- that is the risk engine's job, and reaching ``REVIEW_ALERT`` requires the
score to have crossed the configured threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import BehaviorState, EvidenceType

#: Position of each state along the concealment sequence. Benign terminals and
#: REVIEW_ALERT sit outside the ordering and are handled explicitly.
SEQUENCE_ORDER: dict[BehaviorState, int] = {
    BehaviorState.IDLE: 0,
    BehaviorState.SHELF_INTERACTION: 1,
    BehaviorState.ITEM_ASSOCIATED: 2,
    BehaviorState.ITEM_REMOVED_FROM_SHELF: 3,
    BehaviorState.HAND_MOVING_TO_STORAGE: 4,
    BehaviorState.POSSIBLE_CONCEALMENT: 5,
    BehaviorState.ITEM_OCCLUDED: 6,
    BehaviorState.ITEM_MISSING: 7,
    BehaviorState.REVIEW_ALERT: 8,
}

#: Minimum evidence that unlocks each sequence state. A state is reachable only
#: when every listed observation is active.
STATE_REQUIREMENTS: dict[BehaviorState, tuple[EvidenceType, ...]] = {
    BehaviorState.SHELF_INTERACTION: (EvidenceType.SHELF_INTERACTION,),
    BehaviorState.ITEM_ASSOCIATED: (EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,),
    BehaviorState.ITEM_REMOVED_FROM_SHELF: (
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        EvidenceType.ITEM_REMOVED_FROM_SHELF,
    ),
    BehaviorState.HAND_MOVING_TO_STORAGE: (
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        EvidenceType.HAND_MOVED_TO_STORAGE_REGION,
    ),
    BehaviorState.POSSIBLE_CONCEALMENT: (
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        EvidenceType.HAND_MOVED_TO_STORAGE_REGION,
        EvidenceType.CONCEALMENT_MOTION_PROFILE,
    ),
    BehaviorState.ITEM_OCCLUDED: (
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        EvidenceType.HAND_MOVED_TO_STORAGE_REGION,
        EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
    ),
    BehaviorState.ITEM_MISSING: (
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
        EvidenceType.ITEM_REMAINS_MISSING,
    ),
}

#: Benign evidence that immediately branches the episode out of the sequence.
BENIGN_BRANCHES: dict[EvidenceType, BehaviorState] = {
    EvidenceType.ITEM_RETURNED_TO_SHELF: BehaviorState.ITEM_RETURNED,
    EvidenceType.ITEM_PLACED_IN_BASKET: BehaviorState.ITEM_TO_BASKET,
    EvidenceType.ITEM_PLACED_IN_CART: BehaviorState.ITEM_TO_CART,
}


@dataclass(slots=True)
class StateTransition:
    """One recorded state change, kept for the incident timeline."""

    from_state: BehaviorState
    to_state: BehaviorState
    timestamp: float
    reason: str = ""


class BehaviorStateMachine:
    """Tracks one person's position in the concealment sequence."""

    def __init__(
        self, initial: BehaviorState = BehaviorState.IDLE, history_limit: int = 64
    ) -> None:
        self._state = initial
        self._entered_at: float = 0.0
        self._history: list[StateTransition] = []
        self._history_limit = history_limit

    # -- accessors ---------------------------------------------------------
    @property
    def state(self) -> BehaviorState:
        return self._state

    @property
    def rank(self) -> int:
        return SEQUENCE_ORDER.get(self._state, 0)

    @property
    def history(self) -> list[StateTransition]:
        return list(self._history)

    @property
    def entered_at(self) -> float:
        return self._entered_at

    def time_in_state(self, now: float) -> float:
        return max(0.0, now - self._entered_at)

    # -- mutation ----------------------------------------------------------
    def transition(self, target: BehaviorState, timestamp: float, reason: str = "") -> bool:
        """Move to ``target``. Returns ``True`` if the state actually changed."""
        if target is self._state:
            return False
        transition = StateTransition(self._state, target, timestamp, reason)
        self._state = target
        self._entered_at = timestamp
        self._history.append(transition)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]
        return True

    def reset(self, timestamp: float, reason: str = "episode reset") -> bool:
        return self.transition(BehaviorState.IDLE, timestamp, reason)

    def clear_history(self) -> None:
        self._history.clear()


def derive_state(
    active_evidence: set[EvidenceType],
    current: BehaviorState,
    risk_score: float,
    alert_threshold: float,
) -> tuple[BehaviorState, str]:
    """Compute the state the evidence justifies.

    Returns ``(state, reason)``. Never regresses along the sequence: evidence
    that has already been counted stays counted for the life of the episode,
    so a dropped frame cannot cause the same observation to be scored twice.
    """
    # 1. A benign explanation wins outright, from any sequence state.
    for evidence, branch in BENIGN_BRANCHES.items():
        if evidence in active_evidence:
            return branch, f"benign resolution: {evidence.value}"

    # 2. Highest sequence state whose requirements are all satisfied.
    best = BehaviorState.IDLE
    best_reason = "no qualifying evidence"
    for state, requirements in STATE_REQUIREMENTS.items():
        satisfied = all(requirement in active_evidence for requirement in requirements)
        if satisfied and SEQUENCE_ORDER[state] > SEQUENCE_ORDER[best]:
            best = state
            best_reason = "evidence: " + ", ".join(r.value for r in requirements)

    # 3. The alert state additionally requires the score to cross the
    #    threshold. Sequence position alone never produces an alert.
    if (
        SEQUENCE_ORDER[best] >= SEQUENCE_ORDER[BehaviorState.ITEM_MISSING]
        and risk_score >= alert_threshold
    ):
        return (
            BehaviorState.REVIEW_ALERT,
            f"risk {risk_score:.1f} >= threshold {alert_threshold:.1f} with full sequence",
        )

    # 4. Monotonic progress within the episode.
    if current.is_benign_terminal:
        return current, "benign terminal state held"
    if SEQUENCE_ORDER.get(current, 0) > SEQUENCE_ORDER[best]:
        return current, "holding highest state reached this episode"
    return best, best_reason


@dataclass(slots=True)
class EpisodeSummary:
    """A finished or in-flight interaction episode, for the incident record."""

    started_at: float
    ended_at: float | None = None
    peak_state: BehaviorState = BehaviorState.IDLE
    peak_risk: float = 0.0
    transitions: list[StateTransition] = field(default_factory=list)
    outcome: str = "open"

    def duration(self, now: float) -> float:
        end = self.ended_at if self.ended_at is not None else now
        return max(0.0, end - self.started_at)
