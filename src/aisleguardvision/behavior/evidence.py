"""Evidence ledger.

The behavior engine does not compute a score directly. It raises discrete,
human-readable *observations* -- positive ones supporting a concealment
hypothesis and negative ones arguing against it -- into a per-person ledger,
and the risk engine turns the ledger into a number. Keeping those two steps
separate is what makes every alert explainable: the alert payload is literally
the ledger.

Lifetime rules
--------------
Evidence is scoped to an **interaction episode** (one shelf approach through to
its resolution), not to the person's whole time in the store. Within an
episode, volatile observations (a wrist currently in a zone) carry a TTL and
expire; structural facts about the episode (an item was removed from this
shelf) persist until the episode ends.

Re-raising the same observation refreshes it rather than stacking: a wrist that
stays in a shelf zone for 30 frames is one shelf interaction, not thirty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import EvidenceEvent, EvidenceType, Hand, Point, StorageRegion


class EvidenceLedger:
    """Deduplicating, TTL-aware store of evidence for one episode.

    ``EvidenceEvent.ttl`` carries the whole lifetime policy and the ledger
    never rewrites it. ``ttl=None`` means "structural fact about this episode,
    valid until the episode ends" -- an item having been taken off a shelf does
    not stop being true four seconds later, and expiring it would silently
    dismantle a sequence mid-flight. The evidence factories set an explicit TTL
    on the volatile observations instead.
    """

    def __init__(self) -> None:
        self._events: dict[tuple, EvidenceEvent] = {}

    def add(self, event: EvidenceEvent) -> bool:
        """Record an observation.

        Returns ``True`` if this is a newly raised observation and ``False`` if
        it merely refreshed one already present -- the engine uses that to emit
        a log line and a state transition only on the first sighting.
        """
        key = event.dedup_key
        existing = self._events.get(key)
        if existing is None:
            self._events[key] = event
            return True
        # Refresh the timestamp so the observation stays alive, and keep the
        # strongest confidence seen: an association that was briefly weak but
        # is now strong should be scored on its best evidence.
        existing.timestamp = event.timestamp
        existing.confidence = max(existing.confidence, event.confidence)
        existing.description = event.description
        existing.metadata.update(event.metadata)
        existing.ttl = event.ttl
        return False

    def prune(self, now: float) -> list[EvidenceEvent]:
        """Remove expired observations and return them."""
        expired: list[EvidenceEvent] = []
        for key, event in list(self._events.items()):
            if not event.is_active(now):
                expired.append(event)
                del self._events[key]
        return expired

    def active(self, now: float) -> list[EvidenceEvent]:
        """Currently valid observations, oldest first."""
        return sorted(
            (e for e in self._events.values() if e.is_active(now)), key=lambda e: e.timestamp
        )

    def has(self, evidence_type: EvidenceType) -> bool:
        return any(e.evidence_type is evidence_type for e in self._events.values())

    def get(self, evidence_type: EvidenceType) -> EvidenceEvent | None:
        for event in self._events.values():
            if event.evidence_type is evidence_type:
                return event
        return None

    def all_of(self, evidence_type: EvidenceType) -> list[EvidenceEvent]:
        return [e for e in self._events.values() if e.evidence_type is evidence_type]

    def remove(self, evidence_type: EvidenceType) -> int:
        """Retract every instance of an observation. Returns how many were removed.

        Used when a fact is positively contradicted -- e.g. an item believed
        missing reappears in the shopper's hand.
        """
        keys = [k for k, e in self._events.items() if e.evidence_type is evidence_type]
        for key in keys:
            del self._events[key]
        return len(keys)

    def clear_positive(self) -> None:
        """Drop all positive evidence, keeping the negative record.

        Called when an episode is resolved benignly. The negative evidence is
        kept briefly so the score visibly collapses and the reviewer-facing
        explanation says *why* it collapsed.
        """
        for key, event in list(self._events.items()):
            if not event.is_negative:
                del self._events[key]

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events.values())


# ---------------------------------------------------------------------------
# Factories
#
# Centralized so the wording a reviewer sees in an alert is defined in exactly
# one place, and so descriptions always carry the concrete numbers that
# justified the observation.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvidenceContext:
    """Common fields shared by the evidence raised in one update."""

    person_id: int
    timestamp: float
    #: Default TTL for volatile evidence (the temporal window).
    window: float
    hand: Hand | None = None
    item_id: int | None = None
    zone_id: str | None = None
    extra: dict = field(default_factory=dict)


def shelf_interaction(
    ctx: EvidenceContext, zone_name: str, duration: float, confidence: float
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.SHELF_INTERACTION,
        timestamp=ctx.timestamp,
        description=(
            f"{_hand_label(ctx.hand)} interacted with shelf zone '{zone_name}' for {duration:.2f}s"
        ),
        confidence=confidence,
        person_id=ctx.person_id,
        zone_id=ctx.zone_id,
        hand=ctx.hand,
        ttl=ctx.window,
        metadata={"duration_seconds": round(duration, 3), "zone_name": zone_name},
    )


def high_value_zone_interaction(
    ctx: EvidenceContext, zone_name: str, confidence: float
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.HIGH_VALUE_ZONE_INTERACTION,
        timestamp=ctx.timestamp,
        description=f"Interaction occurred in high-value zone '{zone_name}'",
        confidence=confidence,
        person_id=ctx.person_id,
        zone_id=ctx.zone_id,
        hand=ctx.hand,
        ttl=ctx.window,
        metadata={"zone_name": zone_name},
    )


def stable_association(ctx: EvidenceContext, duration: float, confidence: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.STABLE_HAND_ITEM_ASSOCIATION,
        timestamp=ctx.timestamp,
        description=(
            f"{_hand_label(ctx.hand)} associated with item {ctx.item_id} "
            f"for {duration:.2f}s (confidence {confidence:.2f})"
        ),
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,  # structural fact about the episode
        metadata={"duration_seconds": round(duration, 3)},
    )


def item_removed_from_shelf(
    ctx: EvidenceContext, zone_name: str, confidence: float
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.ITEM_REMOVED_FROM_SHELF,
        timestamp=ctx.timestamp,
        description=f"Item {ctx.item_id} moved out of shelf zone '{zone_name}' while held",
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        zone_id=ctx.zone_id,
        hand=ctx.hand,
        ttl=None,
        metadata={"zone_name": zone_name},
    )


def hand_moved_to_storage(
    ctx: EvidenceContext, region: StorageRegion, travel_ratio: float, confidence: float
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.HAND_MOVED_TO_STORAGE_REGION,
        timestamp=ctx.timestamp,
        description=(
            f"{_hand_label(ctx.hand)} moved from the shelf toward the "
            f"{region.value.replace('_', ' ')} region "
            f"({travel_ratio:.2f} body-heights of travel)"
        ),
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,
        metadata={"region": region.value, "travel_ratio": round(travel_ratio, 3)},
    )


def concealment_motion(
    ctx: EvidenceContext, speed_ratio: float, confidence: float
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.CONCEALMENT_MOTION_PROFILE,
        timestamp=ctx.timestamp,
        description=(
            f"{_hand_label(ctx.hand)} showed a downward/inward motion profile "
            f"({speed_ratio:.2f} body-heights/s)"
        ),
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,
        metadata={"speed_ratio": round(speed_ratio, 3)},
    )


def item_disappeared_near_storage(
    ctx: EvidenceContext, region: StorageRegion, position: Point, confidence: float
) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.ITEM_DISAPPEARED_NEAR_STORAGE,
        timestamp=ctx.timestamp,
        description=(
            f"Item {ctx.item_id} became occluded near the "
            f"{region.value.replace('_', ' ')} region "
            f"at ({position.x:.0f}, {position.y:.0f})"
        ),
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,
        metadata={"region": region.value},
    )


def item_remains_missing(ctx: EvidenceContext, duration: float, confidence: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.ITEM_REMAINS_MISSING,
        timestamp=ctx.timestamp,
        description=f"Item {ctx.item_id} remained unobserved for {duration:.2f}s",
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,
        metadata={"missing_seconds": round(duration, 3)},
    )


def no_basket_placement(ctx: EvidenceContext) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.NO_BASKET_PLACEMENT,
        timestamp=ctx.timestamp,
        description="No basket or cart placement was detected during the sequence",
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        ttl=None,
    )


def no_shelf_return(ctx: EvidenceContext) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.NO_SHELF_RETURN,
        timestamp=ctx.timestamp,
        description="No return of the item to a shelf zone was detected",
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        ttl=None,
    )


# -- negative ---------------------------------------------------------------


def item_returned(ctx: EvidenceContext, zone_name: str) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.ITEM_RETURNED_TO_SHELF,
        timestamp=ctx.timestamp,
        description=f"Item {ctx.item_id} was returned to shelf zone '{zone_name}'",
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        zone_id=ctx.zone_id,
        ttl=None,
    )


def item_visible_in_hand(ctx: EvidenceContext, confidence: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.ITEM_VISIBLE_IN_HAND,
        timestamp=ctx.timestamp,
        description=f"Item {ctx.item_id} remains plainly visible in the shopper's hand",
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=ctx.window,
    )


def item_placed_in_container(ctx: EvidenceContext, is_cart: bool, zone_name: str) -> EvidenceEvent:
    evidence_type = (
        EvidenceType.ITEM_PLACED_IN_CART if is_cart else EvidenceType.ITEM_PLACED_IN_BASKET
    )
    container = "cart" if is_cart else "basket"
    return EvidenceEvent(
        evidence_type=evidence_type,
        timestamp=ctx.timestamp,
        description=f"Item {ctx.item_id} was placed in a shopping {container} ('{zone_name}')",
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        zone_id=ctx.zone_id,
        ttl=None,
    )


def phone_interaction(ctx: EvidenceContext, confidence: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.NORMAL_PHONE_INTERACTION,
        timestamp=ctx.timestamp,
        description=(
            f"Object associated with {_hand_label(ctx.hand).lower()} is a mobile phone, "
            "not merchandise"
        ),
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,
    )


def personal_effect(ctx: EvidenceContext, label: str, confidence: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.PERSONAL_EFFECT_INTERACTION,
        timestamp=ctx.timestamp,
        description=f"Associated object is a personal effect ({label}), not merchandise",
        confidence=confidence,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=None,
    )


def temporary_occlusion(ctx: EvidenceContext, reason: str) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.TEMPORARY_OCCLUSION,
        timestamp=ctx.timestamp,
        description=(
            f"Item {ctx.item_id} disappearance is consistent with temporary occlusion: {reason}"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        ttl=ctx.window,
    )


def unstable_item_track(ctx: EvidenceContext, hits: int, age: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.UNSTABLE_ITEM_TRACK,
        timestamp=ctx.timestamp,
        description=(
            f"Item {ctx.item_id} track is unstable ({hits} detections over {age:.2f}s); "
            "disappearance is more likely a tracking artifact"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        ttl=None,
    )


def low_pose_confidence(ctx: EvidenceContext, value: float, threshold: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.LOW_POSE_CONFIDENCE,
        timestamp=ctx.timestamp,
        description=(
            f"Pose confidence for the torso and arms is low ({value:.2f} < {threshold:.2f}); "
            "hand positions are unreliable"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        ttl=ctx.window,
    )


def low_track_confidence(ctx: EvidenceContext, value: float, threshold: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.LOW_PERSON_TRACK_CONFIDENCE,
        timestamp=ctx.timestamp,
        description=(
            f"Person track confidence is low ({value:.2f} < {threshold:.2f}); "
            "identity continuity across frames is uncertain"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        ttl=ctx.window,
    )


def camera_occlusion(ctx: EvidenceContext, reason: str) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.CAMERA_OCCLUSION,
        timestamp=ctx.timestamp,
        description=f"Camera view is partially obstructed: {reason}",
        confidence=1.0,
        person_id=ctx.person_id,
        ttl=ctx.window,
    )


def short_accidental_overlap(ctx: EvidenceContext, duration: float) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.SHORT_ACCIDENTAL_OVERLAP,
        timestamp=ctx.timestamp,
        description=(
            f"Hand/item overlap lasted only {duration:.2f}s, consistent with a hand "
            "passing in front of merchandise"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        item_id=ctx.item_id,
        hand=ctx.hand,
        ttl=ctx.window,
    )


def no_merchandise_interaction(ctx: EvidenceContext) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.NO_MERCHANDISE_INTERACTION,
        timestamp=ctx.timestamp,
        description=(
            "Hand approached a personal storage region with no preceding merchandise "
            "interaction (e.g. reaching into a pocket)"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        hand=ctx.hand,
        ttl=ctx.window,
    )


def item_detection_unavailable(ctx: EvidenceContext) -> EvidenceEvent:
    return EvidenceEvent(
        evidence_type=EvidenceType.ITEM_DETECTION_UNAVAILABLE,
        timestamp=ctx.timestamp,
        description=(
            "No merchandise detector is configured; only zone-based interaction "
            "evidence is available and risk is capped accordingly"
        ),
        confidence=1.0,
        person_id=ctx.person_id,
        ttl=ctx.window,
    )


def _hand_label(hand: Hand | None) -> str:
    if hand is None:
        return "Hand"
    return "Left wrist" if hand is Hand.LEFT else "Right wrist"
