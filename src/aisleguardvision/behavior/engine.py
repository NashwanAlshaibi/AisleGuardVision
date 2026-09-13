"""Temporal behavior engine.

This is where AisleGuard Vision decides what it thinks it is looking at, and
the single most important rule in the codebase applies here:

    **No single frame, and no single observation, produces an alert.**

Picking up merchandise, holding it, putting it back, checking a phone,
reaching into a pocket, adjusting clothing, carrying a bag, loading a basket
and being briefly occluded are all *normal* and each is handled explicitly
below so that none of them alone can escalate risk.

A high-risk assessment requires a mutually supporting sequence:

1. the shopper's wrist interacted with a merchandise zone, for a minimum dwell;
2. a merchandise item became stably associated with that same wrist;
3. the item left the shelf zone while held;
4. that wrist then travelled toward a pose-derived storage region;
5. with a downward/inward motion profile;
6. the item moved with the wrist;
7. the item became occluded near that region;
8. and stayed unobserved past the occlusion timeout;
9. with no basket or cart placement observed;
10. and no return to a shelf observed.

Any benign explanation appearing at any point terminates the episode.

State is per person, per camera, and keyed on the temporary track id. The
engine reasons in **seconds**, never in frames -- cameras in one store run at
different rates and any given camera's effective rate varies with load.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..core.config import AppConfig, BehaviorConfig
from ..core.logging import get_logger
from ..core.types import (
    BehaviorObservation,
    BehaviorState,
    BoundingBox,
    EvidenceType,
    Hand,
    ItemStatus,
    ItemTrack,
    KeypointName,
    PersonTrack,
    Point,
    RiskAssessment,
    ShelfInteraction,
    ShelfZone,
    StorageRegionEstimate,
    ZoneKind,
)
from ..tracking.association import AssociationResult, WristHistoryStore
from ..tracking.item_tracker import ItemTracker
from . import evidence as ev
from .evidence import EvidenceContext, EvidenceLedger
from .geometry import body_axes, estimate_storage_regions, motion_profile, nearest_storage_region
from .risk import RiskEngine
from .state_machine import BehaviorStateMachine, EpisodeSummary, derive_state
from .zones import ZoneRegistry

logger = get_logger(__name__)

MERCHANDISE_ZONE_KINDS = (ZoneKind.SHELF, ZoneKind.HIGH_VALUE)
CONTAINER_ZONE_KINDS = (ZoneKind.BASKET, ZoneKind.CART)


@dataclass(slots=True)
class PersonContext:
    """Everything the engine remembers about one tracked shopper."""

    person_id: int
    camera_id: str
    machine: BehaviorStateMachine
    ledger: EvidenceLedger
    episode_started_at: float
    last_evidence_at: float
    #: (hand, zone_id) -> timestamp the current continuous contact began.
    shelf_contact: dict[tuple[Hand, str], float] = field(default_factory=dict)
    #: (hand, storage region) -> timestamp the wrist arrived and stayed.
    storage_contact: dict[tuple[Hand, str], float] = field(default_factory=dict)
    #: Structured SHELF_INTERACTION records for the incident timeline.
    shelf_interactions: list[ShelfInteraction] = field(default_factory=list)
    #: The merchandise item this episode is about, if any.
    active_item_id: int | None = None
    active_hand: Hand | None = None
    #: item_id -> zone the item was in when the association started.
    item_origin_zone: dict[int, str] = field(default_factory=dict)
    #: Items already resolved benignly this episode; never re-escalated.
    resolved_items: set[int] = field(default_factory=set)
    peak_risk: float = 0.0
    peak_state: BehaviorState = BehaviorState.IDLE
    last_assessment: RiskAssessment | None = None
    storage_regions: list[StorageRegionEstimate] = field(default_factory=list)
    episode: EpisodeSummary | None = None
    #: Set when a benign terminal state is entered, to time the reset.
    benign_since: float | None = None
    #: Highest zone risk multiplier encountered this episode.
    zone_multiplier: float = 1.0

    def start_episode(self, timestamp: float) -> None:
        self.episode_started_at = timestamp
        self.last_evidence_at = timestamp
        self.ledger.clear()
        self.shelf_contact.clear()
        self.storage_contact.clear()
        self.shelf_interactions.clear()
        self.item_origin_zone.clear()
        self.resolved_items.clear()
        self.active_item_id = None
        self.active_hand = None
        self.peak_risk = 0.0
        self.peak_state = BehaviorState.IDLE
        self.benign_since = None
        self.zone_multiplier = 1.0
        self.episode = EpisodeSummary(started_at=timestamp)


class TemporalBehaviorEngine:
    """Per-camera behavior reasoning over tracked people and items."""

    def __init__(
        self,
        camera_id: str,
        config: AppConfig | None = None,
        risk_engine: RiskEngine | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.config = config or AppConfig()
        self.behavior: BehaviorConfig = self.config.behavior
        self.risk_engine = risk_engine or RiskEngine(self.config.risk, self.config.behavior)
        self._contexts: dict[int, PersonContext] = {}
        self._transition_counts: dict[str, int] = defaultdict(int)

    # -- public API --------------------------------------------------------
    def update(
        self,
        *,
        timestamp: float,
        person_tracks: list[PersonTrack],
        item_tracks: list[ItemTrack],
        associations: list[AssociationResult],
        zones: ZoneRegistry,
        wrists: WristHistoryStore,
        item_tracker: ItemTracker | None = None,
        track_quality: dict[int, float] | None = None,
        item_detection_available: bool = True,
        container_boxes: list[BoundingBox] | None = None,
    ) -> list[BehaviorObservation]:
        """Advance the engine by one analysis frame.

        Returns one :class:`BehaviorObservation` per tracked person. The caller
        decides what to do with observations that warrant an alert -- the
        engine never dispatches anything itself.
        """
        items_by_id = {item.item_id: item for item in item_tracks}
        associations_by_person: dict[int, list[AssociationResult]] = defaultdict(list)
        for association in associations:
            associations_by_person[association.person_id].append(association)

        observations: list[BehaviorObservation] = []
        live_ids = {track.track_id for track in person_tracks}

        for track in person_tracks:
            context = self._context_for(track, timestamp)
            observation = self._update_person(
                context=context,
                track=track,
                timestamp=timestamp,
                associations=associations_by_person.get(track.track_id, []),
                items_by_id=items_by_id,
                zones=zones,
                wrists=wrists,
                item_tracker=item_tracker,
                track_quality=(track_quality or {}).get(track.track_id, track.confidence),
                item_detection_available=item_detection_available,
                container_boxes=container_boxes or [],
            )
            observations.append(observation)

        self._retire_contexts(live_ids)
        return observations

    def context(self, person_id: int) -> PersonContext | None:
        return self._contexts.get(person_id)

    def shelf_interactions(self, person_id: int) -> list[ShelfInteraction]:
        context = self._contexts.get(person_id)
        return list(context.shelf_interactions) if context else []

    def state_of(self, person_id: int) -> BehaviorState:
        context = self._contexts.get(person_id)
        return context.machine.state if context else BehaviorState.IDLE

    def states(self) -> dict[int, BehaviorState]:
        """Current behavior state per tracked person (for the scheduler)."""
        return {pid: ctx.machine.state for pid, ctx in self._contexts.items()}

    def risks(self) -> dict[int, float]:
        """Peak episode risk per tracked person (for the scheduler)."""
        return {pid: ctx.peak_risk for pid, ctx in self._contexts.items()}

    @property
    def transition_counts(self) -> dict[str, int]:
        return dict(self._transition_counts)

    def reset(self) -> None:
        self._contexts.clear()

    # -- per-person update -------------------------------------------------
    def _update_person(
        self,
        *,
        context: PersonContext,
        track: PersonTrack,
        timestamp: float,
        associations: list[AssociationResult],
        items_by_id: dict[int, ItemTrack],
        zones: ZoneRegistry,
        wrists: WristHistoryStore,
        item_tracker: ItemTracker | None,
        track_quality: float,
        item_detection_available: bool,
        container_boxes: list[BoundingBox],
    ) -> BehaviorObservation:
        behavior = self.behavior
        previous_state = context.machine.state
        context.ledger.prune(timestamp)

        ctx = EvidenceContext(
            person_id=track.track_id,
            timestamp=timestamp,
            window=behavior.temporal_window_seconds,
        )
        new_evidence: list = []

        def raise_evidence(event) -> None:
            if context.ledger.add(event):
                new_evidence.append(event)
                context.last_evidence_at = timestamp
                logger.debug(
                    "evidence raised",
                    extra={
                        "fields": {
                            "camera_id": self.camera_id,
                            "person_id": track.track_id,
                            "evidence": event.evidence_type.value,
                            "confidence": round(event.confidence, 2),
                        }
                    },
                )
            else:
                context.last_evidence_at = timestamp

        # -- gate 0: the track must be old enough to reason about ----------
        # A track that has existed for a fraction of a second is at least as
        # likely to be a detector artifact as a shopper.
        track_is_mature = track.track_age >= behavior.min_person_track_seconds

        # -- quality signals (always evaluated, even for young tracks) ------
        if track_quality < behavior.low_track_confidence:
            raise_evidence(
                ev.low_track_confidence(ctx, track_quality, behavior.low_track_confidence)
            )
        pose = track.pose if track.has_fresh_pose(
            timestamp, self.config.detection.scheduler.pose_staleness_seconds
        ) else None
        if pose is not None:
            torso_confidence = pose.torso_confidence()
            if torso_confidence < behavior.low_pose_confidence:
                raise_evidence(
                    ev.low_pose_confidence(ctx, torso_confidence, behavior.low_pose_confidence)
                )
        if not item_detection_available:
            raise_evidence(ev.item_detection_unavailable(ctx))

        # -- storage regions ------------------------------------------------
        bag_boxes = [
            item.bbox
            for item in items_by_id.values()
            if item.object_class.is_personal_effect and item.object_class.value != "phone"
        ]
        context.storage_regions = (
            estimate_storage_regions(
                track,
                behavior.storage_regions,
                self.config.detection.inference.keypoint_confidence,
                bag_boxes,
            )
            if pose is not None
            else []
        )

        if track_is_mature and pose is not None:
            self._evaluate_shelf_interaction(context, track, pose, zones, timestamp, ctx, raise_evidence)
            self._evaluate_associations(
                context, track, associations, items_by_id, item_tracker, timestamp, ctx, raise_evidence
            )
            self._evaluate_item_removal(context, items_by_id, zones, timestamp, ctx, raise_evidence)
            self._evaluate_hand_to_storage(
                context, track, pose, wrists, timestamp, ctx, raise_evidence
            )
            self._evaluate_item_disappearance(
                context,
                track,
                items_by_id,
                zones,
                item_tracker,
                container_boxes,
                timestamp,
                ctx,
                raise_evidence,
            )

        # -- score ----------------------------------------------------------
        active = context.ledger.active(timestamp)
        assessment = self.risk_engine.assess(
            person_id=track.track_id,
            camera_id=self.camera_id,
            timestamp=timestamp,
            state=context.machine.state,
            evidence=active,
            zone_multiplier=context.zone_multiplier,
            item_detection_available=item_detection_available,
        )

        # -- state ----------------------------------------------------------
        active_types = {event.evidence_type for event in active}
        target, reason = derive_state(
            active_types, context.machine.state, assessment.risk_score, behavior.alert_threshold
        )
        if context.machine.transition(target, timestamp, reason):
            self._transition_counts[f"{previous_state.value}->{target.value}"] += 1
            logger.info(
                "behavior state transition",
                extra={
                    "fields": {
                        "camera_id": self.camera_id,
                        "person_id": track.track_id,
                        "from": previous_state.value,
                        "to": target.value,
                        "risk": assessment.risk_score,
                        "reason": reason,
                    }
                },
            )
            if context.episode is not None:
                context.episode.transitions = context.machine.history

        assessment.behavior_state = context.machine.state
        context.last_assessment = assessment
        if assessment.risk_score > context.peak_risk:
            context.peak_risk = assessment.risk_score
        if context.machine.state is not BehaviorState.IDLE:
            context.peak_state = (
                context.machine.state
                if context.machine.state is not BehaviorState.IDLE
                else context.peak_state
            )
        if context.episode is not None:
            context.episode.peak_risk = context.peak_risk
            context.episode.peak_state = context.peak_state

        self._maybe_reset_episode(context, timestamp)

        return BehaviorObservation(
            person_id=track.track_id,
            camera_id=self.camera_id,
            timestamp=timestamp,
            state=context.machine.state,
            previous_state=previous_state,
            risk=assessment,
            new_evidence=new_evidence,
            active_evidence=active,
            storage_regions=list(context.storage_regions),
            associated_item_id=context.active_item_id,
            associated_hand=context.active_hand,
        )

    # -- stage evaluators --------------------------------------------------
    def _evaluate_shelf_interaction(
        self, context, track: PersonTrack, pose, zones: ZoneRegistry, timestamp, ctx, raise_evidence
    ) -> None:
        """Stage 4: sustained wrist presence in or near a merchandise zone.

        Requires *continuous* presence for a configurable dwell. A wrist that
        clips a shelf polygon for one frame while the shopper walks past is
        scored as an accidental overlap instead.
        """
        behavior = self.behavior
        body_height = track.body_height
        keypoint_confidence = self.config.detection.inference.keypoint_confidence
        seen_keys: set[tuple[Hand, str]] = set()

        for hand in Hand:
            wrist = pose.wrist(hand, keypoint_confidence)
            if wrist is None:
                continue
            if zones.is_excluded(wrist):
                continue
            hit = zones.nearest_zone(wrist, MERCHANDISE_ZONE_KINDS, body_height)
            if hit is None:
                continue
            in_contact = hit.inside or hit.normalized_distance <= behavior.shelf_approach_distance_ratio
            key = (hand, hit.zone.zone_id)
            if not in_contact:
                continue

            seen_keys.add(key)
            started = context.shelf_contact.get(key)
            if started is None:
                context.shelf_contact[key] = timestamp
                continue

            duration = timestamp - started
            if duration < behavior.min_shelf_interaction_seconds:
                continue

            ctx.hand = hand
            ctx.zone_id = hit.zone.zone_id
            # Confidence tapers with distance: a wrist deep inside the shelf
            # face is stronger evidence than one hovering at the boundary.
            confidence = 1.0 if hit.inside else float(
                np.clip(
                    1.0 - hit.normalized_distance / max(behavior.shelf_approach_distance_ratio, 1e-6),
                    0.2,
                    1.0,
                )
            )
            raise_evidence(ev.shelf_interaction(ctx, hit.zone.name, duration, confidence))
            context.shelf_interactions.append(
                ShelfInteraction(
                    person_id=track.track_id,
                    camera_id=self.camera_id,
                    timestamp=timestamp,
                    hand=hand,
                    zone_id=hit.zone.zone_id,
                    position=wrist,
                    confidence=confidence,
                    associated_item_id=context.active_item_id,
                    duration=duration,
                )
            )
            if hit.zone.kind is ZoneKind.HIGH_VALUE:
                raise_evidence(ev.high_value_zone_interaction(ctx, hit.zone.name, confidence))
            context.zone_multiplier = max(context.zone_multiplier, hit.zone.risk_multiplier)

        # Contacts that ended: short ones are explicitly exculpatory.
        for key in list(context.shelf_contact.keys()):
            if key in seen_keys:
                continue
            started = context.shelf_contact.pop(key)
            duration = timestamp - started
            if 0.0 < duration < behavior.accidental_overlap_seconds:
                ctx.hand, ctx.zone_id = key[0], key[1]
                raise_evidence(ev.short_accidental_overlap(ctx, duration))
        ctx.hand = None
        ctx.zone_id = None

    def _evaluate_associations(
        self,
        context,
        track: PersonTrack,
        associations: list[AssociationResult],
        items_by_id: dict[int, ItemTrack],
        item_tracker: ItemTracker | None,
        timestamp,
        ctx,
        raise_evidence,
    ) -> None:
        """Stage 7: what is this hand actually holding?

        The phone branch here is the single highest-value false-positive
        control in the product. A shopper pulling out a phone, looking at it
        and putting it back in a pocket reproduces the *entire* concealment
        motion sequence; only object identity distinguishes the two.
        """
        for association in associations:
            item = items_by_id.get(association.item_id)
            if item is None:
                continue
            ctx.item_id = item.item_id
            ctx.hand = association.hand

            if item_tracker is not None:
                item_tracker.set_association(
                    item.item_id,
                    track.track_id,
                    association.hand,
                    association.confidence,
                    timestamp,
                )

            if item.object_class.value == "phone":
                raise_evidence(ev.phone_interaction(ctx, association.confidence))
                # Retract any positive evidence that this object generated
                # before it was identified: it was never merchandise.
                if context.active_item_id == item.item_id:
                    context.active_item_id = None
                    context.active_hand = None
                continue

            if item.object_class.is_personal_effect:
                raise_evidence(
                    ev.personal_effect(ctx, item.object_class.value, association.confidence)
                )
                continue

            if item.item_id in context.resolved_items:
                continue

            if not association.is_stable:
                continue

            if not item.is_stable(
                self.config.detection.tracking.item.min_hits_for_stability,
                self.config.detection.tracking.item.min_age_for_stability_seconds,
            ):
                raise_evidence(ev.unstable_item_track(ctx, item.hit_count, item.track_age))
                continue

            raise_evidence(ev.stable_association(ctx, association.duration, association.confidence))
            context.active_item_id = item.item_id
            context.active_hand = association.hand

            if item.status is ItemStatus.VISIBLE:
                raise_evidence(ev.item_visible_in_hand(ctx, association.confidence))

        ctx.item_id = None
        ctx.hand = None

    def _evaluate_item_removal(
        self, context, items_by_id, zones: ZoneRegistry, timestamp, ctx, raise_evidence
    ) -> None:
        """Stage 9 step 3: the held item leaves the shelf zone it came from."""
        item_id = context.active_item_id
        if item_id is None:
            return
        item = items_by_id.get(item_id)
        if item is None or item.status is not ItemStatus.VISIBLE:
            return

        origin = context.item_origin_zone.get(item_id)
        if origin is None:
            zone = zones.zone_containing(item.center, MERCHANDISE_ZONE_KINDS)
            if zone is not None:
                context.item_origin_zone[item_id] = zone.zone_id
            return

        zone = zones.by_id(origin)
        if zone is None:
            return
        still_inside = zones.zone_containing(item.center, MERCHANDISE_ZONE_KINDS)
        if still_inside is not None and still_inside.zone_id == origin:
            return

        ctx.item_id = item_id
        ctx.hand = context.active_hand
        ctx.zone_id = origin
        raise_evidence(
            ev.item_removed_from_shelf(ctx, zone.name, item.association_confidence or 1.0)
        )
        ctx.item_id = None
        ctx.hand = None
        ctx.zone_id = None

    def _evaluate_hand_to_storage(
        self, context, track: PersonTrack, pose, wrists: WristHistoryStore, timestamp, ctx, raise_evidence
    ) -> None:
        """Stages 8-9: is the hand travelling toward a plausible storage region?

        This is the step that most naively-built systems get wrong. A wrist
        near a hip is not evidence of anything on its own -- people reach into
        their pockets constantly. It becomes meaningful only when a tracked
        merchandise item is already associated with that same wrist, and when
        the wrist is *approaching* the region rather than merely near it.
        """
        behavior = self.behavior
        regions = context.storage_regions
        if not regions:
            return

        keypoint_confidence = self.config.detection.inference.keypoint_confidence
        body_height = track.body_height
        axes = body_axes(pose, keypoint_confidence)
        down_axis = axes[0] if axes else None
        left_hip = pose.point_of(KeypointName.LEFT_HIP, keypoint_confidence)
        right_hip = pose.point_of(KeypointName.RIGHT_HIP, keypoint_confidence)
        body_center = (
            Point((left_hip.x + right_hip.x) * 0.5, (left_hip.y + right_hip.y) * 0.5)
            if left_hip and right_hip
            else None
        )

        has_merchandise_context = context.ledger.has(
            EvidenceType.STABLE_HAND_ITEM_ASSOCIATION
        ) or context.ledger.has(EvidenceType.SHELF_INTERACTION)

        hands = [context.active_hand] if context.active_hand is not None else list(Hand)
        seen_contacts: set[tuple[Hand, str]] = set()
        for hand in hands:
            if hand is None:
                continue
            wrist = pose.wrist(hand, keypoint_confidence)
            if wrist is None:
                continue

            nearest = nearest_storage_region(wrist, regions)
            if nearest is None:
                continue
            region, normalized = nearest
            if normalized > behavior.storage_regions.approach_radius_multiplier:
                continue

            # The wrist must ARRIVE AND STAY. A hand crossing the body on its
            # way elsewhere clips the torso region for a few frames; without a
            # dwell requirement that reads as a concealment approach, which is
            # one of the largest false-positive sources in this design.
            contact_key = (hand, region.region.value)
            seen_contacts.add(contact_key)
            arrived_at = context.storage_contact.setdefault(contact_key, timestamp)
            if (timestamp - arrived_at) < behavior.storage_dwell_seconds:
                continue

            path = wrists.trajectory(
                track.track_id, hand, behavior.temporal_window_seconds, timestamp
            )
            if len(path) < 2:
                continue

            # Proximity alone is meaningless -- an arm hanging at the side sits
            # inside the waist region permanently. What matters is that the
            # wrist *approached* the region during the window.
            start = path[0].position
            distance_then = region.center.distance_to(start) / max(body_height, 1e-6)
            distance_now = region.center.distance_to(wrist) / max(body_height, 1e-6)
            travel = distance_then - distance_now
            if travel < behavior.storage_approach_travel_ratio:
                continue

            # A deliberate move to a pocket with no preceding merchandise
            # interaction is an ordinary pocket adjustment. Record that
            # explicitly, so the reason a reviewer is *not* being paged is
            # part of the record rather than an absence of one.
            if not has_merchandise_context:
                ctx.hand = hand
                raise_evidence(ev.no_merchandise_interaction(ctx))
                ctx.hand = None
                continue

            ctx.hand = hand
            ctx.item_id = context.active_item_id
            confidence = float(np.clip(region.confidence, 0.2, 1.0))
            raise_evidence(ev.hand_moved_to_storage(ctx, region.region, travel, confidence))

            # Motion profile is evaluated over a much shorter window: a
            # concealment movement is a sub-second gesture and averaging it
            # over the full temporal window would dilute it away.
            recent = wrists.trajectory(
                track.track_id, hand, behavior.motion_window_seconds, timestamp
            )
            profile = motion_profile(recent, body_height, down_axis, body_center)
            if (
                profile.speed_ratio >= behavior.concealment_motion_speed_ratio
                and profile.is_concealment_like
            ):
                raise_evidence(ev.concealment_motion(ctx, profile.speed_ratio, confidence))
            ctx.hand = None
            ctx.item_id = None

        # A wrist that left a region restarts its dwell clock next time.
        for key in list(context.storage_contact.keys()):
            if key not in seen_contacts:
                del context.storage_contact[key]

    def _evaluate_item_disappearance(
        self,
        context,
        track: PersonTrack,
        items_by_id,
        zones: ZoneRegistry,
        item_tracker: ItemTracker | None,
        container_boxes: list[BoundingBox],
        timestamp,
        ctx,
        raise_evidence,
    ) -> None:
        """Stage 6 + 9 steps 7-10: interpret the item's observation state.

        Benign explanations are checked *first and exhaustively*. Only when the
        item cannot be accounted for by a container, a shelf return or ordinary
        occlusion does its disappearance become positive evidence.
        """
        item_id = context.active_item_id
        if item_id is None:
            return
        item = items_by_id.get(item_id)
        if item is None:
            return

        ctx.item_id = item_id
        ctx.hand = context.active_hand
        behavior = self.behavior

        # -- benign: placed into a basket or cart --------------------------
        position = item.center if item.status is ItemStatus.VISIBLE else item.last_visible_position
        if position is not None:
            container = zones.zone_containing(position, CONTAINER_ZONE_KINDS)
            in_detected_container = any(box.contains_point(position) for box in container_boxes)
            if container is not None or in_detected_container:
                is_cart = container is not None and container.kind is ZoneKind.CART
                name = container.name if container is not None else "detected shopping container"
                ctx.zone_id = container.zone_id if container is not None else None
                self._resolve_benign(
                    context,
                    item,
                    item_tracker,
                    ItemStatus.IN_CART if is_cart else ItemStatus.IN_BASKET,
                    timestamp,
                )
                raise_evidence(ev.item_placed_in_container(ctx, is_cart, name))
                ctx.zone_id = None
                return

        # -- benign: returned to a shelf -----------------------------------
        # "Returned" requires that the item was actually taken off the shelf
        # first; an item merely sitting on its shelf face while a hand hovers
        # over it has not been returned to anything.
        #
        # It further requires the item not be close to a storage region: a
        # shopper standing at the shelf face has their own waist projected
        # onto the shelf polygon in a 2D view, so storage proximity has to win
        # that tie or concealment at the shelf would read as a return.
        if position is not None and context.ledger.has(EvidenceType.ITEM_REMOVED_FROM_SHELF):
            shelf = zones.zone_containing(position, MERCHANDISE_ZONE_KINDS)
            if shelf is not None:
                nearest = nearest_storage_region(position, context.storage_regions)
                near_storage = (
                    nearest is not None
                    and nearest[1] <= behavior.storage_regions.approach_radius_multiplier
                )
                if not near_storage:
                    ctx.zone_id = shelf.zone_id
                    self._resolve_benign(
                        context, item, item_tracker, ItemStatus.RETURNED, timestamp
                    )
                    raise_evidence(ev.item_returned(ctx, shelf.name))
                    ctx.zone_id = None
                    return

        if item.status is ItemStatus.VISIBLE:
            # Still plainly visible: nothing has been concealed.
            return

        # The item is no longer visible. Retract the "visible in hand" fact
        # rather than waiting for it to expire -- it is no longer true, and
        # leaving it in the ledger would suppress a genuine sequence for a
        # full temporal window.
        context.ledger.remove(EvidenceType.ITEM_VISIBLE_IN_HAND)

        if item.status is ItemStatus.POSSIBLY_OCCLUDED:
            # Too early to say anything at all.
            return

        if not item.is_stable(
            self.config.detection.tracking.item.min_hits_for_stability,
            self.config.detection.tracking.item.min_age_for_stability_seconds,
        ):
            raise_evidence(ev.unstable_item_track(ctx, item.hit_count, item.track_age))
            return

        last_position = item.last_visible_position
        if last_position is None:
            return

        nearest = nearest_storage_region(last_position, context.storage_regions)
        if nearest is None or nearest[1] > behavior.storage_regions.approach_radius_multiplier:
            # Vanished somewhere unremarkable: ordinary occlusion.
            raise_evidence(
                ev.temporary_occlusion(ctx, "item disappeared away from any personal storage region")
            )
            return

        region, normalized = nearest
        confidence = float(np.clip(1.0 - normalized / 2.0, 0.3, 1.0)) * float(
            np.clip(region.confidence, 0.3, 1.0)
        )
        raise_evidence(
            ev.item_disappeared_near_storage(ctx, region.region, last_position, confidence)
        )

        if item.status is ItemStatus.MISSING and item.disappeared_at is not None:
            missing_for = timestamp - item.disappeared_at
            raise_evidence(ev.item_remains_missing(ctx, missing_for, 1.0))
            # Document what was checked and not found. Zero-weight, but it is
            # what a reviewer needs to see to trust the alert.
            raise_evidence(ev.no_basket_placement(ctx))
            raise_evidence(ev.no_shelf_return(ctx))

        ctx.item_id = None
        ctx.hand = None

    # -- episode lifecycle -------------------------------------------------
    def _resolve_benign(
        self, context, item: ItemTrack, item_tracker: ItemTracker | None, status, timestamp
    ) -> None:
        if item_tracker is not None:
            item_tracker.resolve(item.item_id, status, timestamp)
        else:
            item.status = status
        context.resolved_items.add(item.item_id)
        # A benign outcome invalidates the positive case built so far. The
        # negative record is kept so the score visibly collapses and the
        # explanation says why.
        context.ledger.clear_positive()
        context.benign_since = timestamp

    def _maybe_reset_episode(self, context: PersonContext, timestamp: float) -> None:
        state = context.machine.state
        if state.is_benign_terminal:
            if (
                context.benign_since is not None
                and (timestamp - context.benign_since) >= self.behavior.benign_reset_seconds
            ):
                logger.debug(
                    "episode resolved benignly, resetting",
                    extra={
                        "fields": {
                            "camera_id": self.camera_id,
                            "person_id": context.person_id,
                            "outcome": state.value,
                        }
                    },
                )
                if context.episode is not None:
                    context.episode.ended_at = timestamp
                    context.episode.outcome = state.value
                context.machine.reset(timestamp, f"benign outcome: {state.value}")
                context.start_episode(timestamp)
            return

        if state is BehaviorState.IDLE:
            return

        if (timestamp - context.last_evidence_at) >= self.behavior.episode_timeout_seconds:
            logger.debug(
                "episode timed out with no new evidence, resetting",
                extra={
                    "fields": {"camera_id": self.camera_id, "person_id": context.person_id}
                },
            )
            if context.episode is not None:
                context.episode.ended_at = timestamp
                context.episode.outcome = "timeout"
            context.machine.reset(timestamp, "episode timeout")
            context.start_episode(timestamp)

    def _context_for(self, track: PersonTrack, timestamp: float) -> PersonContext:
        context = self._contexts.get(track.track_id)
        if context is None:
            context = PersonContext(
                person_id=track.track_id,
                camera_id=self.camera_id,
                machine=BehaviorStateMachine(),
                ledger=EvidenceLedger(default_ttl=self.behavior.temporal_window_seconds),
                episode_started_at=timestamp,
                last_evidence_at=timestamp,
                episode=EpisodeSummary(started_at=timestamp),
            )
            self._contexts[track.track_id] = context
        return context

    def _retire_contexts(self, live_ids: set[int]) -> None:
        for person_id in list(self._contexts.keys()):
            if person_id not in live_ids:
                del self._contexts[person_id]


def zone_names(zones: list[ShelfZone]) -> str:
    return ", ".join(z.name or z.zone_id for z in zones)
