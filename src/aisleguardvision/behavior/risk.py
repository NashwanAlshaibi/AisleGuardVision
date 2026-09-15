"""Explainable risk scoring.

There is no learned "shoplifting probability" in AisleGuard Vision, and there
should never be one. A model that outputs a single opaque number cannot be
audited, cannot be tuned per store, cannot be explained to the person being
reviewed, and cannot be defended when it is wrong.

Instead the score is arithmetic over evidence:

    raw   = sum(weight(evidence) * confidence(evidence) * zone_multiplier)
    score = clamp(raw, 0, 100)

Every point is attributable to a named observation with a human-readable
justification, and the full breakdown travels with the alert.

Two deliberate asymmetries:

* Negative weights are large relative to positive ones. A basket placement
  (-70) or a phone identification (-70) wipes out a complete positive sequence.
  False positives are far more costly than false negatives here: a wrong alert
  sends a staff member to confront an innocent shopper.
* The zone multiplier applies only to positive contributions. Standing in a
  high-value aisle must never make the exculpatory evidence count for less.
"""

from __future__ import annotations

import numpy as np

from ..core.config import BehaviorConfig, RiskConfig
from ..core.logging import get_logger
from ..core.types import (
    BehaviorState,
    EvidenceEvent,
    EvidenceType,
    RiskAssessment,
    RiskContribution,
    ThreatLevel,
)

logger = get_logger(__name__)

MIN_SCORE = 0.0
MAX_SCORE = 100.0


class RiskEngine:
    """Turns an evidence ledger into an explainable 0-100 score."""

    def __init__(
        self, config: RiskConfig | None = None, behavior: BehaviorConfig | None = None
    ) -> None:
        self.config = config or RiskConfig()
        self.behavior = behavior or BehaviorConfig()

    def assess(
        self,
        *,
        person_id: int,
        camera_id: str,
        timestamp: float,
        state: BehaviorState,
        evidence: list[EvidenceEvent],
        zone_multiplier: float = 1.0,
        item_detection_available: bool = True,
    ) -> RiskAssessment:
        """Score the currently active evidence.

        ``item_detection_available`` is not cosmetic. Without a merchandise
        detector the system has zone geometry and wrist kinematics only, which
        is not sufficient grounds to ask a human to review a shopper. In that
        mode the score is capped below the alert threshold -- see
        ``behavior.zone_only_risk_ceiling``.
        """
        contributions: list[RiskContribution] = []
        raw = 0.0

        for event in evidence:
            base = self.config.weight_for(event.evidence_type)
            if base == 0.0:
                # Zero-weight observations (the "nothing benign was seen"
                # records) are still surfaced to the reviewer, but they must
                # not silently add points.
                contributions.append(
                    RiskContribution(
                        evidence_type=event.evidence_type,
                        description=event.description,
                        base_weight=0.0,
                        confidence=event.confidence,
                        applied_weight=0.0,
                    )
                )
                continue

            confidence = float(np.clip(event.confidence, 0.0, 1.0))
            applied = base * confidence
            if applied > 0:
                applied *= zone_multiplier
            raw += applied
            contributions.append(
                RiskContribution(
                    evidence_type=event.evidence_type,
                    description=event.description,
                    base_weight=base,
                    confidence=confidence,
                    applied_weight=applied,
                )
            )

        score = float(np.clip(raw, MIN_SCORE, MAX_SCORE))

        if not item_detection_available:
            ceiling = self.behavior.zone_only_risk_ceiling
            if score > ceiling:
                logger.debug(
                    "risk capped: no merchandise detector configured",
                    extra={
                        "fields": {
                            "camera_id": camera_id,
                            "person_id": person_id,
                            "uncapped": round(score, 1),
                            "ceiling": ceiling,
                        }
                    },
                )
                score = ceiling

        return RiskAssessment(
            person_id=person_id,
            camera_id=camera_id,
            timestamp=timestamp,
            risk_score=round(score, 2),
            threat_level=self.threat_level(score),
            behavior_state=state,
            contributions=contributions,
            raw_score=round(raw, 2),
        )

    def threat_level(self, score: float) -> ThreatLevel:
        return ThreatLevel.from_score(
            score,
            elevated=self.config.elevated_threshold,
            review=self.config.review_threshold,
            high=self.config.high_risk_threshold,
        )

    def should_alert(self, assessment: RiskAssessment) -> bool:
        """Whether this assessment warrants a reviewable incident.

        Threshold *and* sequence completeness. A score that reaches 85 without
        the item ever having been tracked into a plausible storage region is a
        scoring-configuration bug, not an incident, and is refused here.
        """
        if assessment.risk_score < self.behavior.alert_threshold:
            return False
        required = {
            EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
            EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
        }
        present = {c.evidence_type for c in assessment.contributions}
        missing = required - present
        if missing:
            logger.warning(
                "risk threshold reached without the required evidence sequence; "
                "refusing to raise an alert (check risk weights)",
                extra={
                    "fields": {
                        "camera_id": assessment.camera_id,
                        "person_id": assessment.person_id,
                        "score": assessment.risk_score,
                        "missing": ",".join(sorted(m.value for m in missing)),
                    }
                },
            )
            return False
        return True

    def explain(self, assessment: RiskAssessment) -> dict[str, object]:
        """Alert-payload view of the arithmetic."""
        return {
            "risk_score": assessment.risk_score,
            "raw_score": assessment.raw_score,
            "threat_level": assessment.threat_level.value,
            "behavior_state": assessment.behavior_state.value,
            "positive_evidence": assessment.positive_evidence,
            "negative_evidence": assessment.negative_evidence,
            "breakdown": [
                {
                    "evidence": c.evidence_type.value,
                    "description": c.description,
                    "weight": round(c.base_weight, 2),
                    "confidence": round(c.confidence, 3),
                    "points": round(c.applied_weight, 2),
                }
                for c in sorted(assessment.contributions, key=lambda c: -abs(c.applied_weight))
            ],
        }
