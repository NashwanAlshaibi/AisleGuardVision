"""Risk scoring: weights, normalization, thresholds and explainability."""

from __future__ import annotations

import pytest

from aisleguardvision.behavior.risk import RiskEngine
from aisleguardvision.core.config import AppConfig, BehaviorConfig, RiskConfig
from aisleguardvision.core.types import (
    BehaviorState,
    EvidenceEvent,
    EvidenceType,
    ThreatLevel,
)


def evidence(
    evidence_type: EvidenceType, confidence: float = 1.0, timestamp: float = 100.0
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=evidence_type,
        timestamp=timestamp,
        description=f"test: {evidence_type.value}",
        confidence=confidence,
        person_id=1,
    )


@pytest.fixture
def engine() -> RiskEngine:
    config = AppConfig()
    return RiskEngine(config.risk, config.behavior)


def assess(engine: RiskEngine, events, **kwargs):
    return engine.assess(
        person_id=1,
        camera_id="cam_test",
        timestamp=100.0,
        state=BehaviorState.IDLE,
        evidence=events,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_empty_evidence_scores_zero(engine):
    result = assess(engine, [])
    assert result.risk_score == 0.0
    assert result.threat_level is ThreatLevel.LOW


def test_score_is_the_sum_of_configured_weights(engine):
    result = assess(
        engine,
        [
            evidence(EvidenceType.SHELF_INTERACTION),  # +10
            evidence(EvidenceType.ITEM_REMOVED_FROM_SHELF),  # +10
        ],
    )
    assert result.risk_score == pytest.approx(20.0)
    assert result.raw_score == pytest.approx(20.0)


def test_confidence_scales_the_weight(engine):
    full = assess(engine, [evidence(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION, 1.0)])
    half = assess(engine, [evidence(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION, 0.5)])
    assert half.risk_score == pytest.approx(full.risk_score / 2)


def test_confidence_is_clamped_to_the_unit_interval(engine):
    # A backend returning an out-of-range confidence must not inflate the score.
    absurd = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION, 5.0)])
    assert absurd.risk_score == pytest.approx(10.0)
    negative = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION, -2.0)])
    assert negative.risk_score == pytest.approx(0.0)


def test_score_is_clamped_to_0_100(engine):
    everything = [
        evidence(t)
        for t in EvidenceType
        if not t.is_negative
    ]
    result = assess(engine, everything)
    assert 0.0 <= result.risk_score <= 100.0
    assert result.raw_score >= result.risk_score


def test_negative_evidence_cannot_drive_the_score_below_zero(engine):
    result = assess(
        engine,
        [
            evidence(EvidenceType.SHELF_INTERACTION),  # +10
            evidence(EvidenceType.ITEM_PLACED_IN_BASKET),  # -70
        ],
    )
    assert result.risk_score == 0.0
    assert result.raw_score < 0.0, "the raw total stays negative for auditing"


@pytest.mark.parametrize(
    "negative",
    [
        EvidenceType.ITEM_PLACED_IN_BASKET,
        EvidenceType.ITEM_PLACED_IN_CART,
        EvidenceType.NORMAL_PHONE_INTERACTION,
        EvidenceType.ITEM_RETURNED_TO_SHELF,
    ],
)
def test_a_single_strong_benign_explanation_wipes_out_a_full_sequence(engine, negative):
    """False positives cost more than false negatives: a wrong alert sends
    staff to confront an innocent shopper."""
    sequence = [
        evidence(EvidenceType.SHELF_INTERACTION),
        evidence(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION),
        evidence(EvidenceType.ITEM_REMOVED_FROM_SHELF),
        evidence(EvidenceType.HAND_MOVED_TO_STORAGE_REGION),
    ]
    without = assess(engine, sequence)
    with_benign = assess(engine, [*sequence, evidence(negative)])

    assert without.risk_score > 40.0
    assert with_benign.risk_score < 5.0


def test_absence_evidence_scores_zero(engine):
    """'Nothing benign was observed' documents what was checked; it must never
    add risk on its own."""
    result = assess(
        engine,
        [evidence(EvidenceType.NO_BASKET_PLACEMENT), evidence(EvidenceType.NO_SHELF_RETURN)],
    )
    assert result.risk_score == 0.0
    # ...but it is still recorded for the reviewer.
    assert len(result.contributions) == 2


# ---------------------------------------------------------------------------
# Zone multipliers
# ---------------------------------------------------------------------------


def test_zone_multiplier_scales_positive_evidence(engine):
    plain = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION)])
    boosted = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION)], zone_multiplier=1.5)
    assert boosted.risk_score == pytest.approx(plain.risk_score * 1.5)


def test_zone_multiplier_never_discounts_exculpatory_evidence(engine):
    """Standing in a high-value aisle must not make the evidence in your favour
    count for less."""
    events = [
        evidence(EvidenceType.SHELF_INTERACTION),
        evidence(EvidenceType.ITEM_PLACED_IN_BASKET),
    ]
    plain = assess(engine, events)
    boosted = assess(engine, events, zone_multiplier=2.0)

    negative_plain = min(c.applied_weight for c in plain.contributions)
    negative_boosted = min(c.applied_weight for c in boosted.contributions)
    assert negative_plain == negative_boosted == pytest.approx(-70.0)


# ---------------------------------------------------------------------------
# Threat levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, ThreatLevel.LOW),
        (29.9, ThreatLevel.LOW),
        (30.0, ThreatLevel.ELEVATED),
        (59.9, ThreatLevel.ELEVATED),
        (60.0, ThreatLevel.REVIEW),
        (84.9, ThreatLevel.REVIEW),
        (85.0, ThreatLevel.HIGH_RISK),
        (100.0, ThreatLevel.HIGH_RISK),
    ],
)
def test_threat_level_boundaries(engine, score, expected):
    assert engine.threat_level(score) is expected


def test_threat_thresholds_must_increase():
    with pytest.raises(Exception):
        RiskConfig(elevated_threshold=60, review_threshold=30, high_risk_threshold=85)


# ---------------------------------------------------------------------------
# Alert gating
# ---------------------------------------------------------------------------


def test_should_alert_requires_the_threshold(engine):
    below = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION)])
    assert not engine.should_alert(below)


def test_should_alert_requires_the_evidence_sequence(engine):
    """A score that reaches the threshold without item evidence is a scoring
    bug, not an incident, and must be refused."""
    inflated = [
        evidence(EvidenceType.SHELF_INTERACTION),
        evidence(EvidenceType.HIGH_VALUE_ZONE_INTERACTION),
        evidence(EvidenceType.HAND_MOVED_TO_STORAGE_REGION),
        evidence(EvidenceType.CONCEALMENT_MOTION_PROFILE),
        evidence(EvidenceType.ITEM_REMAINS_MISSING),
    ]
    result = assess(engine, inflated, zone_multiplier=3.0)
    assert result.risk_score >= 85.0, "precondition: the score crosses the threshold"
    assert not engine.should_alert(result), "missing item association must block the alert"


def test_should_alert_accepts_a_complete_sequence(engine):
    complete = [
        evidence(EvidenceType.SHELF_INTERACTION),
        evidence(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION),
        evidence(EvidenceType.ITEM_REMOVED_FROM_SHELF),
        evidence(EvidenceType.HAND_MOVED_TO_STORAGE_REGION),
        evidence(EvidenceType.CONCEALMENT_MOTION_PROFILE),
        evidence(EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE),
        evidence(EvidenceType.ITEM_REMAINS_MISSING),
    ]
    result = assess(engine, complete)
    assert result.risk_score >= 85.0
    assert engine.should_alert(result)


# ---------------------------------------------------------------------------
# Zone-only ceiling
# ---------------------------------------------------------------------------


def test_zone_only_mode_caps_the_score(engine):
    complete = [
        evidence(EvidenceType.SHELF_INTERACTION),
        evidence(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION),
        evidence(EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE),
        evidence(EvidenceType.ITEM_REMAINS_MISSING),
    ]
    uncapped = assess(engine, complete, item_detection_available=True)
    capped = assess(engine, complete, item_detection_available=False)

    ceiling = engine.behavior.zone_only_risk_ceiling
    assert uncapped.risk_score > ceiling
    assert capped.risk_score == pytest.approx(ceiling)
    assert not engine.should_alert(capped)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def test_every_point_is_attributable(engine):
    result = assess(
        engine,
        [
            evidence(EvidenceType.SHELF_INTERACTION, 0.8),
            evidence(EvidenceType.STABLE_HAND_ITEM_ASSOCIATION, 0.6),
        ],
    )
    total = sum(c.applied_weight for c in result.contributions)
    assert result.raw_score == pytest.approx(total, abs=0.01)


def test_positive_and_negative_evidence_are_separated(engine):
    result = assess(
        engine,
        [
            evidence(EvidenceType.SHELF_INTERACTION),
            evidence(EvidenceType.LOW_POSE_CONFIDENCE),
        ],
    )
    assert len(result.positive_evidence) == 1
    assert len(result.negative_evidence) == 1


def test_explain_is_machine_readable(engine):
    result = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION)])
    payload = engine.explain(result)

    assert set(payload) >= {
        "risk_score",
        "threat_level",
        "positive_evidence",
        "negative_evidence",
        "breakdown",
    }
    assert payload["breakdown"][0]["evidence"] == EvidenceType.SHELF_INTERACTION.value


def test_explain_text_is_human_readable(engine):
    result = assess(engine, [evidence(EvidenceType.SHELF_INTERACTION)])
    text = result.explain()
    assert "SHELF_INTERACTION" in text
    assert "risk=" in text


# ---------------------------------------------------------------------------
# Weight configuration invariants
# ---------------------------------------------------------------------------


def test_positive_evidence_cannot_be_given_a_negative_weight():
    """A sign error here would invert the explanation shown to a reviewer."""
    with pytest.raises(Exception):
        RiskConfig(weights={"SHELF_INTERACTION": -10})


def test_negative_evidence_cannot_be_given_a_positive_weight():
    with pytest.raises(Exception):
        RiskConfig(weights={"ITEM_PLACED_IN_BASKET": 50})


def test_unknown_evidence_type_in_config_is_rejected():
    with pytest.raises(Exception):
        RiskConfig(weights={"DEFINITELY_NOT_A_REAL_EVIDENCE_TYPE": 10})


def test_partial_weight_override_keeps_the_other_defaults():
    """An operator must be able to retune one weight without restating all 24."""
    config = RiskConfig.model_validate({"weights": {"SHELF_INTERACTION": 25}})
    assert config.weight_for(EvidenceType.SHELF_INTERACTION) == 25.0
    assert config.weight_for(EvidenceType.ITEM_PLACED_IN_BASKET) == -70.0


def test_every_evidence_type_has_a_weight():
    config = RiskConfig()
    for evidence_type in EvidenceType:
        assert evidence_type in config.weights


def test_no_single_positive_weight_can_trigger_an_alert():
    """The design requires a sequence. If any one observation were worth 85+,
    a single frame-level mistake could page a human."""
    config = AppConfig()
    for evidence_type, weight in config.risk.weights.items():
        if not evidence_type.is_negative:
            assert weight < config.behavior.alert_threshold, (
                f"{evidence_type.value} alone would reach the alert threshold"
            )


def test_behavior_config_rejects_a_ceiling_at_or_above_the_threshold():
    with pytest.raises(Exception):
        BehaviorConfig(alert_threshold=50, zone_only_risk_ceiling=50)
