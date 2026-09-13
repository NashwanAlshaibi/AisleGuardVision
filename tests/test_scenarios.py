"""End-to-end behavior scenarios.

These are the tests that matter most. Each one asserts an outcome that a real
store depends on, and each maps to a specific way the system could harm a
customer by being wrong.

The false-positive cases are not "nice to have": an alert sends a staff member
to confront a shopper. Every one of these must stay green.
"""

from __future__ import annotations

import pytest

from aisleguardvision.core.config import AppConfig
from aisleguardvision.core.types import BehaviorState, EvidenceType, ThreatLevel
from aisleguardvision.simulation.harness import SimulationRunner, run_scenario
from aisleguardvision.simulation.scenarios import all_scenarios, get_scenario


@pytest.fixture(scope="module")
def runner() -> SimulationRunner:
    return SimulationRunner(AppConfig())


def evidence_types(result) -> set[EvidenceType]:
    """Every evidence type raised at any point in the run."""
    types: set[EvidenceType] = set()
    for outcome in result.outcomes:
        for observation in outcome.observations:
            types.update(e.evidence_type for e in observation.active_evidence)
    return types


# ---------------------------------------------------------------------------
# The required false-negative case
# ---------------------------------------------------------------------------


def test_full_concealment_sequence_reaches_high_risk(runner):
    """Shelf interaction + item association + hand-to-waist + disappearance.

    This is the only scenario that is supposed to alert. If this breaks, the
    product does nothing.
    """
    result = runner.run(get_scenario("POSSIBLE_CONCEALMENT"))

    assert result.alerted, "the full concealment sequence must produce a reviewable incident"
    assert result.peak_risk >= 85.0
    assert result.peak_threat is ThreatLevel.HIGH_RISK
    assert result.peak_state is BehaviorState.REVIEW_ALERT

    raised = evidence_types(result)
    # The whole sequence must be present -- a high score reached some other way
    # would be a scoring bug, not an incident.
    for required in (
        EvidenceType.SHELF_INTERACTION,
        EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        EvidenceType.ITEM_REMOVED_FROM_SHELF,
        EvidenceType.HAND_MOVED_TO_STORAGE_REGION,
        EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
        EvidenceType.ITEM_REMAINS_MISSING,
    ):
        assert required in raised, f"missing required evidence: {required.value}"

    alert = result.alerts[0]
    assert alert.risk.positive_evidence, "an alert must carry its supporting evidence"
    assert "No basket or cart placement" in " ".join(
        e.description for e in alert.active_evidence
    )


def test_concealment_alert_explains_itself(runner):
    """Every alert must record why. An unexplainable alert is unusable."""
    result = runner.run(get_scenario("POSSIBLE_CONCEALMENT"))
    alert = result.alerts[0]

    breakdown = runner.risk_engine.explain(alert.risk)
    assert breakdown["risk_score"] >= 85.0
    assert breakdown["positive_evidence"]
    assert breakdown["breakdown"], "the risk arithmetic must be itemized"

    total = sum(item["points"] for item in breakdown["breakdown"])
    # The score is a clamped sum, so it can be below the raw total but never above.
    assert breakdown["risk_score"] <= max(total, 0.0) + 1e-6


# ---------------------------------------------------------------------------
# False-positive cases -- each maps to a way of wrongly accusing someone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_name",
    [
        "NORMAL_BROWSING",
        "PHONE_INTERACTION",
        "POCKET_ADJUSTMENT",
        "ITEM_PICKUP_RETURN",
        "ITEM_TO_BASKET",
        "ITEM_TO_CART",
        "TEMPORARY_OCCLUSION",
    ],
)
def test_benign_scenarios_never_alert(runner, scenario_name):
    result = runner.run(get_scenario(scenario_name))
    assert not result.alerted, (
        f"{scenario_name} produced an alert (peak risk {result.peak_risk:.1f}). "
        "A false alert sends staff to confront an innocent shopper."
    )
    assert result.peak_risk < 85.0


def test_normal_browsing_stays_low(runner):
    """Browsing a shelf is the most common thing that happens in a store."""
    result = runner.run(get_scenario("NORMAL_BROWSING"))
    assert result.peak_threat is ThreatLevel.LOW
    assert result.peak_risk <= 30.0


def test_pocket_reach_without_merchandise_scores_zero(runner):
    """Reaching into your own pocket is not evidence of anything.

    This is the canonical false positive: the hand-to-waist movement is
    identical to concealment, and only the absence of merchandise separates
    them.
    """
    result = runner.run(get_scenario("POCKET_ADJUSTMENT"))
    assert result.peak_risk == 0.0, "a bare pocket reach must contribute no risk at all"
    assert EvidenceType.NO_MERCHANDISE_INTERACTION in evidence_types(result)
    assert EvidenceType.STABLE_HAND_ITEM_ASSOCIATION not in evidence_types(result)


def test_phone_interaction_is_strongly_suppressed(runner):
    """Phone use reproduces the entire concealment motion sequence.

    Hand to waist, object disappears at the waist, object stays gone. Only
    object identity distinguishes it, which is why the phone class is not
    optional in the detector configuration.
    """
    result = runner.run(get_scenario("PHONE_INTERACTION"))
    raised = evidence_types(result)

    assert result.peak_risk == 0.0
    assert EvidenceType.NORMAL_PHONE_INTERACTION in raised
    # A phone must never be scored as merchandise, however it moves.
    assert EvidenceType.STABLE_HAND_ITEM_ASSOCIATION not in raised
    assert EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE not in raised


def test_item_returned_to_shelf_resolves_benignly(runner):
    """Picking something up and putting it back builds real positive evidence
    and must then retract all of it."""
    result = runner.run(get_scenario("ITEM_PICKUP_RETURN"))
    raised = evidence_types(result)

    assert EvidenceType.STABLE_HAND_ITEM_ASSOCIATION in raised, "the pickup should be detected"
    assert EvidenceType.ITEM_RETURNED_TO_SHELF in raised, "the return should be detected"
    assert not result.alerted

    states = {transition[3] for transition in result.transitions}
    assert BehaviorState.ITEM_RETURNED in states


@pytest.mark.parametrize(
    ("scenario_name", "expected_evidence", "expected_state"),
    [
        ("ITEM_TO_BASKET", EvidenceType.ITEM_PLACED_IN_BASKET, BehaviorState.ITEM_TO_BASKET),
        ("ITEM_TO_CART", EvidenceType.ITEM_PLACED_IN_CART, BehaviorState.ITEM_TO_CART),
    ],
)
def test_container_placement_resolves_benignly(
    runner, scenario_name, expected_evidence, expected_state
):
    result = runner.run(get_scenario(scenario_name))
    assert expected_evidence in evidence_types(result)
    assert expected_state in {transition[3] for transition in result.transitions}
    assert not result.alerted


def test_temporary_occlusion_does_not_alert_immediately(runner):
    """A briefly occluded item is occlusion, not concealment.

    Another shopper walking past, a display blocking the view, or the detector
    losing confidence for a few frames all look identical to an item vanishing.
    """
    result = runner.run(get_scenario("TEMPORARY_OCCLUSION"))
    raised = evidence_types(result)

    assert not result.alerted
    assert EvidenceType.TEMPORARY_OCCLUSION in raised
    # The item reappears, so it must never be scored as remaining missing.
    assert EvidenceType.ITEM_REMAINS_MISSING not in raised


# ---------------------------------------------------------------------------
# The honest-limitation case
# ---------------------------------------------------------------------------


def test_zone_only_mode_cannot_reach_the_alert_threshold(runner):
    """Without a merchandise detector the system must not page a human.

    No COCO class covers general retail merchandise. Shelf geometry plus wrist
    kinematics is real evidence, but it is not sufficient grounds to send
    someone to confront a shopper, so risk is capped below the threshold.
    """
    result = runner.run(get_scenario("CONCEALMENT_WITHOUT_ITEM_DETECTOR"))
    ceiling = runner.config.behavior.zone_only_risk_ceiling

    assert not result.alerted
    assert result.peak_risk <= ceiling
    assert ceiling < runner.config.behavior.alert_threshold


def test_zone_only_ceiling_is_enforced_even_at_a_low_threshold():
    """Lowering the alert threshold must not bypass the zone-only ceiling."""
    config = AppConfig()
    # The config layer refuses a ceiling at or above the threshold, which is
    # the invariant that keeps zone-only mode from ever alerting.
    with pytest.raises(Exception):
        AppConfig.model_validate(
            {"behavior": {"alert_threshold": 50, "zone_only_risk_ceiling": 59}}
        )
    assert config.behavior.zone_only_risk_ceiling < config.behavior.alert_threshold


# ---------------------------------------------------------------------------
# Suite-level guarantees
# ---------------------------------------------------------------------------


def test_every_scenario_matches_its_declared_expectation(runner):
    """Catches a new scenario being added without its expectation being met."""
    failures = []
    for scenario in all_scenarios():
        result = runner.run(scenario)
        if not result.matches_expectation:
            failures.append(
                f"{scenario.name}: expected {scenario.expected}, got {result.outcome} "
                f"(peak risk {result.peak_risk:.1f})"
            )
    assert not failures, "scenario expectations not met:\n  " + "\n  ".join(failures)


def test_scenarios_are_deterministic():
    """The same scenario must produce the same verdict every time.

    Nondeterminism here would mean an incident that cannot be reproduced from
    the recorded clip, which makes review impossible.
    """
    scenario = get_scenario("POSSIBLE_CONCEALMENT")
    first = run_scenario(scenario)
    second = run_scenario(scenario)
    assert first.peak_risk == second.peak_risk
    assert first.outcome == second.outcome
    assert [t[2:] for t in first.transitions] == [t[2:] for t in second.transitions]
