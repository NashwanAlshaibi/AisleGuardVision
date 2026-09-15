"""Behavior state machine: ordering, monotonicity and benign branching."""

from __future__ import annotations

import pytest

from aisleguardvision.behavior.evidence import EvidenceContext, EvidenceLedger
from aisleguardvision.behavior.state_machine import (
    BENIGN_BRANCHES,
    SEQUENCE_ORDER,
    STATE_REQUIREMENTS,
    BehaviorStateMachine,
    derive_state,
)
from aisleguardvision.core.types import BehaviorState, EvidenceEvent, EvidenceType

ALERT_THRESHOLD = 85.0

FULL_SEQUENCE = {
    EvidenceType.SHELF_INTERACTION,
    EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
    EvidenceType.ITEM_REMOVED_FROM_SHELF,
    EvidenceType.HAND_MOVED_TO_STORAGE_REGION,
    EvidenceType.CONCEALMENT_MOTION_PROFILE,
    EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
    EvidenceType.ITEM_REMAINS_MISSING,
}


def derive(active, current=BehaviorState.IDLE, risk=0.0):
    return derive_state(set(active), current, risk, ALERT_THRESHOLD)[0]


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def test_machine_starts_idle():
    machine = BehaviorStateMachine()
    assert machine.state is BehaviorState.IDLE
    assert machine.history == []


def test_transition_records_history():
    machine = BehaviorStateMachine()
    assert machine.transition(BehaviorState.SHELF_INTERACTION, 10.0, "wrist in zone")

    assert machine.state is BehaviorState.SHELF_INTERACTION
    assert len(machine.history) == 1
    assert machine.history[0].from_state is BehaviorState.IDLE
    assert machine.history[0].reason == "wrist in zone"


def test_transition_to_the_same_state_is_a_no_op():
    machine = BehaviorStateMachine()
    machine.transition(BehaviorState.SHELF_INTERACTION, 10.0)
    assert not machine.transition(BehaviorState.SHELF_INTERACTION, 11.0)
    assert len(machine.history) == 1


def test_time_in_state_is_measured_in_seconds():
    machine = BehaviorStateMachine()
    machine.transition(BehaviorState.SHELF_INTERACTION, 100.0)
    assert machine.time_in_state(103.5) == pytest.approx(3.5)


def test_history_is_bounded():
    """A track that lives for an hour must not accumulate unbounded history."""
    machine = BehaviorStateMachine(history_limit=4)
    states = [
        BehaviorState.SHELF_INTERACTION,
        BehaviorState.ITEM_ASSOCIATED,
        BehaviorState.IDLE,
        BehaviorState.SHELF_INTERACTION,
        BehaviorState.ITEM_ASSOCIATED,
        BehaviorState.IDLE,
    ]
    for index, state in enumerate(states):
        machine.transition(state, float(index))
    assert len(machine.history) == 4


# ---------------------------------------------------------------------------
# Evidence-driven derivation
# ---------------------------------------------------------------------------


def test_no_evidence_stays_idle():
    assert derive([]) is BehaviorState.IDLE


def test_shelf_interaction_alone_reaches_only_shelf_interaction():
    assert derive([EvidenceType.SHELF_INTERACTION]) is BehaviorState.SHELF_INTERACTION


def test_association_advances_to_item_associated():
    assert (
        derive([EvidenceType.SHELF_INTERACTION, EvidenceType.STABLE_HAND_ITEM_ASSOCIATION])
        is BehaviorState.ITEM_ASSOCIATED
    )


def test_storage_approach_requires_an_association():
    """Hand-to-waist without a tracked item is a pocket adjustment, and the
    state machine must not advance for it."""
    assert derive([EvidenceType.HAND_MOVED_TO_STORAGE_REGION]) is BehaviorState.IDLE


def test_full_sequence_without_the_threshold_stops_at_item_missing():
    assert derive(FULL_SEQUENCE, risk=50.0) is BehaviorState.ITEM_MISSING


def test_full_sequence_with_the_threshold_reaches_review_alert():
    assert derive(FULL_SEQUENCE, risk=91.0) is BehaviorState.REVIEW_ALERT


def test_review_alert_requires_both_sequence_and_score():
    """Sequence position alone never produces an alert, and neither does a
    score reached without the sequence."""
    assert derive(FULL_SEQUENCE, risk=84.9) is not BehaviorState.REVIEW_ALERT
    assert derive([EvidenceType.SHELF_INTERACTION], risk=99.0) is BehaviorState.SHELF_INTERACTION


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_progress_is_monotonic_within_an_episode():
    """A dropped frame must not knock the sequence back a rung, which would let
    the same evidence be counted twice on the way back up."""
    state = derive(
        [EvidenceType.SHELF_INTERACTION, EvidenceType.STABLE_HAND_ITEM_ASSOCIATION],
        current=BehaviorState.HAND_MOVING_TO_STORAGE,
    )
    assert state is BehaviorState.HAND_MOVING_TO_STORAGE


def test_evidence_expiring_does_not_regress_the_state():
    assert derive([], current=BehaviorState.ITEM_MISSING) is BehaviorState.ITEM_MISSING


# ---------------------------------------------------------------------------
# Benign branching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("evidence_type", "expected"),
    sorted(BENIGN_BRANCHES.items(), key=lambda item: item[0].value),
)
def test_benign_evidence_branches_out_of_the_sequence(evidence_type, expected):
    assert derive([*FULL_SEQUENCE, evidence_type], risk=99.0) is expected


def test_benign_branch_wins_from_the_deepest_sequence_state():
    """An item that reappears in a basket after two seconds of occlusion
    resolves benignly, however far the sequence had progressed."""
    state = derive(
        [*FULL_SEQUENCE, EvidenceType.ITEM_PLACED_IN_BASKET],
        current=BehaviorState.REVIEW_ALERT,
        risk=99.0,
    )
    assert state is BehaviorState.ITEM_TO_BASKET


def test_benign_terminal_state_is_held():
    assert derive([], current=BehaviorState.ITEM_RETURNED) is BehaviorState.ITEM_RETURNED


def test_benign_terminal_states_are_marked():
    for state in (
        BehaviorState.ITEM_RETURNED,
        BehaviorState.ITEM_TO_BASKET,
        BehaviorState.ITEM_TO_CART,
    ):
        assert state.is_benign_terminal
    assert not BehaviorState.REVIEW_ALERT.is_benign_terminal


# ---------------------------------------------------------------------------
# Table invariants
# ---------------------------------------------------------------------------


def test_sequence_order_matches_the_documented_progression():
    expected = [
        BehaviorState.IDLE,
        BehaviorState.SHELF_INTERACTION,
        BehaviorState.ITEM_ASSOCIATED,
        BehaviorState.ITEM_REMOVED_FROM_SHELF,
        BehaviorState.HAND_MOVING_TO_STORAGE,
        BehaviorState.POSSIBLE_CONCEALMENT,
        BehaviorState.ITEM_OCCLUDED,
        BehaviorState.ITEM_MISSING,
        BehaviorState.REVIEW_ALERT,
    ]
    assert sorted(SEQUENCE_ORDER, key=lambda s: SEQUENCE_ORDER[s]) == expected


def test_every_sequence_state_beyond_shelf_requires_an_item_association():
    """The association is what separates a shopper from a suspect. No state
    past SHELF_INTERACTION may be reachable without it."""
    for state, requirements in STATE_REQUIREMENTS.items():
        if SEQUENCE_ORDER[state] > SEQUENCE_ORDER[BehaviorState.SHELF_INTERACTION]:
            assert EvidenceType.STABLE_HAND_ITEM_ASSOCIATION in requirements, state


def test_reset_returns_to_idle():
    machine = BehaviorStateMachine()
    machine.transition(BehaviorState.ITEM_MISSING, 5.0)
    assert machine.reset(6.0)
    assert machine.state is BehaviorState.IDLE


# ---------------------------------------------------------------------------
# Evidence ledger
# ---------------------------------------------------------------------------


def event(evidence_type, timestamp=0.0, ttl=None, item_id=None, confidence=1.0):
    return EvidenceEvent(
        evidence_type=evidence_type,
        timestamp=timestamp,
        description="test",
        confidence=confidence,
        person_id=1,
        item_id=item_id,
        ttl=ttl,
    )


def test_ledger_deduplicates_repeated_observations():
    """A wrist that stays in a zone for 30 frames is one shelf interaction,
    not thirty."""
    ledger = EvidenceLedger()
    assert ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.0)) is True
    assert ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.1)) is False
    assert len(ledger) == 1


def test_ledger_refreshes_the_timestamp_on_re_observation():
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.0, ttl=1.0))
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.9, ttl=1.0))
    assert len(ledger.active(1.5)) == 1, "the refresh should have kept it alive"


def test_ledger_keeps_the_strongest_confidence():
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.0, confidence=0.4))
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.1, confidence=0.9))
    assert ledger.get(EvidenceType.SHELF_INTERACTION).confidence == pytest.approx(0.9)


def test_ledger_separates_observations_about_different_items():
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION, 0.0, item_id=1))
    ledger.add(event(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION, 0.0, item_id=2))
    assert len(ledger) == 2


def test_ledger_expires_volatile_evidence():
    """Volatile observations carry an explicit TTL set by their factory."""
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.0, ttl=2.0))
    assert len(ledger.active(1.0)) == 1
    assert len(ledger.active(5.0)) == 0


def test_ledger_retains_structural_facts_indefinitely():
    """Facts about the episode (an item was taken off this shelf) must not
    expire mid-sequence and take the score down with them."""
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.ITEM_REMOVED_FROM_SHELF, 0.0, ttl=None))
    assert len(ledger.active(10_000.0)) == 1


def test_prune_removes_expired_evidence():
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.0, ttl=1.0))
    expired = ledger.prune(5.0)
    assert len(expired) == 1
    assert len(ledger) == 0


def test_clear_positive_keeps_the_negative_record():
    """On a benign resolution the score must visibly collapse AND the reason
    must remain visible to the reviewer."""
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.SHELF_INTERACTION, 0.0, ttl=4.0))
    ledger.add(event(EvidenceType.ITEM_PLACED_IN_BASKET, 0.0, ttl=None))

    ledger.clear_positive()
    assert not ledger.has(EvidenceType.SHELF_INTERACTION)
    assert ledger.has(EvidenceType.ITEM_PLACED_IN_BASKET)


def test_remove_retracts_a_contradicted_fact():
    ledger = EvidenceLedger()
    ledger.add(event(EvidenceType.ITEM_VISIBLE_IN_HAND, 0.0))
    assert ledger.remove(EvidenceType.ITEM_VISIBLE_IN_HAND) == 1
    assert not ledger.has(EvidenceType.ITEM_VISIBLE_IN_HAND)


def test_evidence_context_is_usable_as_a_factory_input():
    from aisleguardvision.behavior import evidence as ev
    from aisleguardvision.core.types import Hand

    ctx = EvidenceContext(person_id=7, timestamp=12.0, window=4.0, hand=Hand.RIGHT, zone_id="z1")
    built = ev.shelf_interaction(ctx, "Main Shelf", 0.5, 0.9)

    assert built.person_id == 7
    assert built.hand is Hand.RIGHT
    assert "Right wrist" in built.description
    assert "Main Shelf" in built.description
    assert built.ttl == 4.0


def test_negative_evidence_classification_is_consistent():
    negatives = {t for t in EvidenceType if t.is_negative}
    assert EvidenceType.ITEM_PLACED_IN_BASKET in negatives
    assert EvidenceType.NORMAL_PHONE_INTERACTION in negatives
    assert EvidenceType.SHELF_INTERACTION not in negatives
    assert EvidenceType.ITEM_REMAINS_MISSING not in negatives
