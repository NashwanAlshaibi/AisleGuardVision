"""Event payload models.

Pydantic at this boundary (unlike the hot path, which uses dataclasses):
incident JSON is written to disk, POSTed to a webhook and returned by the API,
so a schema that validates and serializes itself is worth its cost here.

Every payload carries the interpretation notice. An event means *possible
concealment behavior was observed and a human should review the clip*. It is
never a determination that a theft occurred, and a downstream consumer should
not be able to read one of these without seeing that said plainly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.types import (
    BehaviorState,
    EventType,
    RiskAssessment,
    SecurityEvent,
    ThreatLevel,
)

#: Shipped verbatim with every event, on disk and over the wire.
INTERPRETATION_NOTICE = (
    "Possible concealment behavior detected - human review recommended. "
    "This is a behavioral risk signal, not a determination that a theft "
    "occurred. AisleGuard Vision performs no facial, identity or demographic "
    "recognition; person identifiers are temporary per-camera tracking ids."
)


class EvidenceItem(BaseModel):
    """One scored observation as it appears in an incident record."""

    model_config = ConfigDict(extra="forbid")

    evidence: str
    description: str
    weight: float = Field(description="Configured weight for this evidence type")
    confidence: float = Field(ge=0.0, le=1.0)
    points: float = Field(description="weight * confidence, as applied to the score")


class TrackMetadata(BaseModel):
    """Non-identifying context about the tracked person and item."""

    model_config = ConfigDict(extra="allow")

    track_age_seconds: float = 0.0
    track_confidence: float = 0.0
    associated_item_id: int | None = None
    associated_hand: str | None = None
    association_seconds: float = 0.0
    storage_region: str | None = None
    zone_id: str | None = None
    frame_width: int = 0
    frame_height: int = 0
    bbox: list[float] = Field(default_factory=list)


class SecurityEventModel(BaseModel):
    """The canonical incident record.

    Written to ``<event_uuid>.json`` beside the clip and the snapshot, returned
    by ``GET /events/{id}``, and POSTed to the webhook.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    camera_id: str
    person_id: int = Field(description="Temporary per-camera tracking id, not an identity")
    timestamp: float
    timestamp_iso: str
    event_type: EventType = EventType.POSSIBLE_CONCEALMENT
    risk_score: float = Field(ge=0.0, le=100.0)
    threat_level: ThreatLevel
    behavior_state: BehaviorState

    positive_evidence: list[str] = Field(default_factory=list)
    negative_evidence: list[str] = Field(default_factory=list)
    evidence_breakdown: list[EvidenceItem] = Field(default_factory=list)

    snapshot_path: str | None = None
    clip_path: str | None = None
    track_metadata: TrackMetadata = Field(default_factory=TrackMetadata)

    review_status: str = "pending"
    notice: str = INTERPRETATION_NOTICE
    schema_version: int = 1

    @classmethod
    def from_event(cls, event: SecurityEvent) -> SecurityEventModel:
        return cls(
            event_id=event.event_id,
            camera_id=event.camera_id,
            person_id=event.person_id,
            timestamp=event.timestamp,
            timestamp_iso=datetime.fromtimestamp(event.timestamp, UTC).isoformat(),
            event_type=event.event_type,
            risk_score=event.risk_score,
            threat_level=event.threat_level,
            behavior_state=event.behavior_state,
            positive_evidence=event.positive_evidence,
            negative_evidence=event.negative_evidence,
            evidence_breakdown=[
                EvidenceItem(
                    evidence=c.evidence_type.value,
                    description=c.description,
                    weight=round(c.base_weight, 2),
                    confidence=round(c.confidence, 3),
                    points=round(c.applied_weight, 2),
                )
                for c in sorted(event.contributions, key=lambda c: -abs(c.applied_weight))
            ],
            snapshot_path=event.snapshot_path,
            clip_path=event.clip_path,
            track_metadata=TrackMetadata(**event.track_metadata)
            if event.track_metadata
            else TrackMetadata(),
            review_status=event.review_status,
        )

    def to_webhook_payload(self) -> dict[str, Any]:
        """The POST body sent to a webhook provider.

        Media paths are included but the media itself is not: an alert must be
        deliverable in milliseconds over a store's uplink, and a reviewer
        fetches the clip from the API when they open the incident.
        """
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "person_id": self.person_id,
            "timestamp": self.timestamp_iso,
            "risk_score": self.risk_score,
            "threat_level": self.threat_level.value,
            "event_type": self.event_type.value,
            "behavior_state": self.behavior_state.value,
            "positive_evidence": self.positive_evidence,
            "negative_evidence": self.negative_evidence,
            "snapshot_path": self.snapshot_path,
            "clip_path": self.clip_path,
            "notice": self.notice,
        }


def build_event(
    assessment: RiskAssessment,
    *,
    event_type: EventType = EventType.POSSIBLE_CONCEALMENT,
    track_metadata: dict[str, Any] | None = None,
) -> SecurityEvent:
    """Create a :class:`SecurityEvent` from a risk assessment."""
    return SecurityEvent(
        event_id=SecurityEvent.new_id(),
        camera_id=assessment.camera_id,
        person_id=assessment.person_id,
        timestamp=assessment.timestamp,
        risk_score=assessment.risk_score,
        threat_level=assessment.threat_level,
        behavior_state=assessment.behavior_state,
        event_type=event_type,
        positive_evidence=assessment.positive_evidence,
        negative_evidence=assessment.negative_evidence,
        contributions=list(assessment.contributions),
        track_metadata=track_metadata or {},
    )
