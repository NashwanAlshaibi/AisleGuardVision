"""Synthetic behavior scenarios.

The behavior engine must be developable and testable **without YOLO**. Model
quality and behavior logic are independent problems, and coupling them means
every tuning change needs a GPU, weights and video, and every regression is
ambiguous between "the detector got worse" and "the logic got worse".

These scenarios emit synthetic *detections* -- person boxes, COCO keypoints and
item boxes -- which are then driven through the real tracker, the real
associator and the real behavior engine. Only the neural networks are absent.

Coordinate convention: a 1280x720 frame with the shelf face on the left and the
shopper standing in front of it. The shopper faces the camera, so their right
side appears on the image-left (toward the shelf), which is what makes a
right-handed reach natural.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..core.config import ZoneConfig
from ..core.types import (
    COCO_KEYPOINT_ORDER,
    BoundingBox,
    ItemDetection,
    KeypointName,
    ObjectClass,
    Point,
    ZoneKind,
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# -- canonical scene geometry ------------------------------------------------
# Shelf face sits clear of where the shopper stands, so an arm hanging at the
# side is not permanently "approaching" it.
SHELF_POLYGON = [(100.0, 150.0), (520.0, 150.0), (520.0, 500.0), (100.0, 500.0)]
BASKET_POLYGON = [(820.0, 555.0), (1010.0, 555.0), (1010.0, 690.0), (820.0, 690.0)]

#: The shopper's standing position and size.
PERSON_CX = 700.0
PERSON_TOP = 100.0
PERSON_HEIGHT = 520.0
PERSON_HALF_WIDTH = 95.0

# Derived landmark positions (see _make_keypoints for the body model).
SHOULDER_Y = PERSON_TOP + 0.20 * PERSON_HEIGHT  # 204
HIP_Y = PERSON_TOP + 0.50 * PERSON_HEIGHT  # 360
SHOULDER_DX = 0.13 * PERSON_HEIGHT  # 67.6
HIP_DX = 0.09 * PERSON_HEIGHT  # 46.8

#: Where a right-handed reach into the shelf lands.
SHELF_REACH = Point(450.0, 300.0)
#: Right wrist resting at the shopper's side.
ARM_AT_SIDE = Point(PERSON_CX - HIP_DX - 15.0, 430.0)
#: Centre of the estimated right-waist storage region (matches
#: estimate_storage_regions with the shipped StorageRegionConfig).
RIGHT_WAIST = Point(PERSON_CX - HIP_DX - 0.55 * (2 * SHOULDER_DX) * 0.35, HIP_Y)
#: An item held out to the side, away from any storage region.
HELD_OUT = Point(890.0, 300.0)
#: Inside the basket / cart zone.
CONTAINER_DROP = Point(905.0, 610.0)


@dataclass(slots=True)
class SimPerson:
    """A synthetic person detection with pose."""

    bbox: BoundingBox
    keypoints: np.ndarray  # (17, 3) -> x, y, confidence
    confidence: float = 0.92


@dataclass(slots=True)
class SimFrame:
    """One synthetic analysis frame."""

    timestamp: float
    persons: list[SimPerson] = field(default_factory=list)
    items: list[ItemDetection] = field(default_factory=list)


@dataclass(slots=True)
class Scenario:
    """A named synthetic sequence with its expected outcome."""

    name: str
    description: str
    #: "NO_ALERT" or "ALERT". Asserted by the test suite.
    expected: str
    #: Why this scenario exists, shown by scripts/simulate_behavior.py.
    rationale: str
    frames: list[SimFrame]
    zones: list[ZoneConfig]
    fps: float = 20.0
    #: False when the scenario deliberately runs without a merchandise detector.
    item_detection_available: bool = True


# ---------------------------------------------------------------------------
# Body model
# ---------------------------------------------------------------------------


def _make_keypoints(
    right_wrist: Point,
    left_wrist: Point | None = None,
    confidence: float = 0.9,
    cx: float = PERSON_CX,
) -> np.ndarray:
    """Build a COCO-17 keypoint array for a shopper facing the camera.

    Only the wrists are scenario-controlled; the rest of the skeleton is a
    fixed standing pose, which is all the behavior engine needs (it reasons
    about shoulders, hips and wrists).
    """
    left = left_wrist if left_wrist is not None else Point(cx + HIP_DX + 15.0, 430.0)
    # Elbows are placed halfway between shoulder and wrist -- good enough for
    # visualization and never used in a decision.
    right_shoulder = Point(cx - SHOULDER_DX, SHOULDER_Y)
    left_shoulder = Point(cx + SHOULDER_DX, SHOULDER_Y)

    positions: dict[KeypointName, Point] = {
        KeypointName.NOSE: Point(cx, PERSON_TOP + 40.0),
        KeypointName.LEFT_EYE: Point(cx + 12.0, PERSON_TOP + 32.0),
        KeypointName.RIGHT_EYE: Point(cx - 12.0, PERSON_TOP + 32.0),
        KeypointName.LEFT_EAR: Point(cx + 26.0, PERSON_TOP + 38.0),
        KeypointName.RIGHT_EAR: Point(cx - 26.0, PERSON_TOP + 38.0),
        KeypointName.LEFT_SHOULDER: left_shoulder,
        KeypointName.RIGHT_SHOULDER: right_shoulder,
        KeypointName.LEFT_ELBOW: Point(
            (left_shoulder.x + left.x) * 0.5, (left_shoulder.y + left.y) * 0.5
        ),
        KeypointName.RIGHT_ELBOW: Point(
            (right_shoulder.x + right_wrist.x) * 0.5, (right_shoulder.y + right_wrist.y) * 0.5
        ),
        KeypointName.LEFT_WRIST: left,
        KeypointName.RIGHT_WRIST: right_wrist,
        KeypointName.LEFT_HIP: Point(cx + HIP_DX, HIP_Y),
        KeypointName.RIGHT_HIP: Point(cx - HIP_DX, HIP_Y),
        KeypointName.LEFT_KNEE: Point(cx + HIP_DX, PERSON_TOP + 0.72 * PERSON_HEIGHT),
        KeypointName.RIGHT_KNEE: Point(cx - HIP_DX, PERSON_TOP + 0.72 * PERSON_HEIGHT),
        KeypointName.LEFT_ANKLE: Point(cx + HIP_DX, PERSON_TOP + 0.97 * PERSON_HEIGHT),
        KeypointName.RIGHT_ANKLE: Point(cx - HIP_DX, PERSON_TOP + 0.97 * PERSON_HEIGHT),
    }
    array = np.zeros((len(COCO_KEYPOINT_ORDER), 3), dtype=np.float32)
    for index, name in enumerate(COCO_KEYPOINT_ORDER):
        point = positions[name]
        array[index] = (point.x, point.y, confidence)
    return array


def make_person(
    right_wrist: Point,
    left_wrist: Point | None = None,
    cx: float = PERSON_CX,
    keypoint_confidence: float = 0.9,
    detection_confidence: float = 0.92,
) -> SimPerson:
    """A standing shopper with the given wrist positions."""
    bbox = BoundingBox(
        cx - PERSON_HALF_WIDTH,
        PERSON_TOP,
        cx + PERSON_HALF_WIDTH,
        PERSON_TOP + PERSON_HEIGHT,
    )
    return SimPerson(
        bbox=bbox,
        keypoints=_make_keypoints(right_wrist, left_wrist, keypoint_confidence, cx),
        confidence=detection_confidence,
    )


def make_item(
    centre: Point,
    object_class: ObjectClass = ObjectClass.MERCHANDISE,
    size: float = 44.0,
    confidence: float = 0.78,
    source: str = "simulation",
) -> ItemDetection:
    half = size * 0.5
    return ItemDetection(
        bbox=BoundingBox(centre.x - half, centre.y - half, centre.x + half, centre.y + half),
        confidence=confidence,
        object_class=object_class,
        class_name=object_class.value,
        source=source,
    )


# ---------------------------------------------------------------------------
# Timeline helper
# ---------------------------------------------------------------------------


def lerp(a: Point, b: Point, t: float) -> Point:
    t = max(0.0, min(1.0, t))
    return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)


@dataclass(slots=True)
class Keyframe:
    """A waypoint on a scenario timeline."""

    at: float
    wrist: Point
    #: Item centre, or ``None`` when the item is not detected in this interval.
    item: Point | None = None
    item_class: ObjectClass = ObjectClass.MERCHANDISE


def build_frames(keyframes: list[Keyframe], duration: float, fps: float) -> list[SimFrame]:
    """Interpolate keyframes into per-frame detections.

    Item visibility is *not* interpolated: an item is emitted only when both
    surrounding keyframes declare one. That is how a scenario expresses "the
    detector stops seeing it here" without the interpolation quietly inventing
    detections across the gap.
    """
    frames: list[SimFrame] = []
    step = 1.0 / fps
    count = int(round(duration / step))
    ordered = sorted(keyframes, key=lambda k: k.at)

    for index in range(count + 1):
        timestamp = round(index * step, 6)
        before = ordered[0]
        after = ordered[-1]
        for current, following in zip(ordered[:-1], ordered[1:], strict=True):
            if current.at <= timestamp <= following.at:
                before, after = current, following
                break
        else:
            if timestamp < ordered[0].at:
                before = after = ordered[0]
            else:
                before = after = ordered[-1]

        span = after.at - before.at
        alpha = (timestamp - before.at) / span if span > 1e-9 else 0.0
        wrist = lerp(before.wrist, after.wrist, alpha)

        items: list[ItemDetection] = []
        if before.item is not None and after.item is not None:
            centre = lerp(before.item, after.item, alpha)
            items.append(make_item(centre, before.item_class))
        elif before.item is not None and span <= 1e-9:
            items.append(make_item(before.item, before.item_class))

        frames.append(SimFrame(timestamp=timestamp, persons=[make_person(wrist)], items=items))
    return frames


def _zones(*specs: tuple[str, ZoneKind, list[tuple[float, float]]]) -> list[ZoneConfig]:
    return [
        ZoneConfig(id=zone_id, kind=kind, polygon=polygon, name=zone_id, camera_id="sim_cam")
        for zone_id, kind, polygon in specs
    ]


SHELF_ZONES = _zones(("shelf_001", ZoneKind.SHELF, SHELF_POLYGON))
SHELF_AND_BASKET = _zones(
    ("shelf_001", ZoneKind.SHELF, SHELF_POLYGON),
    ("basket_001", ZoneKind.BASKET, BASKET_POLYGON),
)
SHELF_AND_CART = _zones(
    ("shelf_001", ZoneKind.SHELF, SHELF_POLYGON),
    ("cart_001", ZoneKind.CART, BASKET_POLYGON),
)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def normal_browsing() -> Scenario:
    """Shopper reaches into the shelf, browses, withdraws. No item detected."""
    frames = build_frames(
        [
            Keyframe(0.0, ARM_AT_SIDE),
            Keyframe(1.5, ARM_AT_SIDE),
            Keyframe(2.2, SHELF_REACH),
            Keyframe(4.5, Point(SHELF_REACH.x + 30, SHELF_REACH.y - 20)),
            Keyframe(5.2, ARM_AT_SIDE),
            Keyframe(7.0, ARM_AT_SIDE),
        ],
        duration=7.0,
        fps=20.0,
    )
    return Scenario(
        name="NORMAL_BROWSING",
        description="Shopper browses a shelf without taking anything",
        expected="NO_ALERT",
        rationale=(
            "Shelf interaction on its own is the most common thing that happens in "
            "a store. It must never approach the alert threshold."
        ),
        frames=frames,
        zones=SHELF_ZONES,
    )


def phone_interaction() -> Scenario:
    """Shopper takes out a phone, looks at it, puts it back in a pocket.

    This reproduces the *entire* concealment motion sequence -- hand to waist,
    object disappears at the waist, object stays gone. Only object identity
    distinguishes it from concealment, which is why it is tested explicitly.
    """
    phone_up = Point(PERSON_CX - 20.0, 265.0)
    frames = build_frames(
        [
            Keyframe(0.0, RIGHT_WAIST, RIGHT_WAIST, ObjectClass.PHONE),
            Keyframe(1.5, RIGHT_WAIST, RIGHT_WAIST, ObjectClass.PHONE),
            Keyframe(2.3, phone_up, phone_up, ObjectClass.PHONE),
            Keyframe(5.0, phone_up, phone_up, ObjectClass.PHONE),
            Keyframe(5.8, RIGHT_WAIST, RIGHT_WAIST, ObjectClass.PHONE),
            Keyframe(6.0, RIGHT_WAIST, None),
            Keyframe(9.0, RIGHT_WAIST, None),
        ],
        duration=9.0,
        fps=20.0,
    )
    return Scenario(
        name="PHONE_INTERACTION",
        description="Shopper retrieves a phone from a pocket, uses it, pockets it again",
        expected="NO_ALERT",
        rationale=(
            "Geometrically identical to concealment. Suppressed because the tracked "
            "object is classified as a phone, not merchandise."
        ),
        frames=frames,
        zones=SHELF_ZONES,
    )


def pocket_adjustment() -> Scenario:
    """Shopper reaches into a pocket with no merchandise involved at all."""
    frames = build_frames(
        [
            Keyframe(0.0, Point(PERSON_CX - 40.0, 250.0)),
            Keyframe(1.6, Point(PERSON_CX - 40.0, 250.0)),
            Keyframe(2.4, RIGHT_WAIST),
            Keyframe(4.5, RIGHT_WAIST),
            Keyframe(5.3, Point(PERSON_CX - 40.0, 250.0)),
            Keyframe(7.0, Point(PERSON_CX - 40.0, 250.0)),
        ],
        duration=7.0,
        fps=20.0,
    )
    return Scenario(
        name="POCKET_ADJUSTMENT",
        description="Shopper reaches into their own pocket, no merchandise interaction",
        expected="NO_ALERT",
        rationale=(
            "Hand-to-waist motion with no preceding merchandise interaction. Must "
            "score at or near zero -- this is the canonical false positive."
        ),
        frames=frames,
        zones=SHELF_ZONES,
    )


def item_pickup_return() -> Scenario:
    """Shopper picks an item off the shelf, inspects it, puts it back."""
    inspect = Point(640.0, 280.0)
    frames = build_frames(
        [
            Keyframe(0.0, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(1.8, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(2.5, SHELF_REACH, SHELF_REACH),
            Keyframe(3.3, SHELF_REACH, SHELF_REACH),
            Keyframe(4.1, inspect, inspect),
            Keyframe(6.0, inspect, inspect),
            Keyframe(6.9, SHELF_REACH, SHELF_REACH),
            Keyframe(8.5, SHELF_REACH, SHELF_REACH),
            Keyframe(9.3, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(11.0, ARM_AT_SIDE, SHELF_REACH),
        ],
        duration=11.0,
        fps=20.0,
    )
    return Scenario(
        name="ITEM_PICKUP_RETURN",
        description="Shopper picks up merchandise, inspects it, returns it to the shelf",
        expected="NO_ALERT",
        rationale=(
            "Builds real positive evidence (shelf interaction, stable association, "
            "removal) and then retracts all of it on the return branch."
        ),
        frames=frames,
        zones=SHELF_ZONES,
    )


def _item_to_container(name: str, zones: list[ZoneConfig], container: str) -> Scenario:
    frames = build_frames(
        [
            Keyframe(0.0, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(1.8, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(2.5, SHELF_REACH, SHELF_REACH),
            Keyframe(3.4, SHELF_REACH, SHELF_REACH),
            Keyframe(4.4, CONTAINER_DROP, CONTAINER_DROP),
            Keyframe(5.4, CONTAINER_DROP, CONTAINER_DROP),
            Keyframe(5.6, ARM_AT_SIDE, None),
            Keyframe(9.0, ARM_AT_SIDE, None),
        ],
        duration=9.0,
        fps=20.0,
    )
    return Scenario(
        name=name,
        description=f"Shopper takes merchandise from the shelf and puts it in a {container}",
        expected="NO_ALERT",
        rationale=(
            f"Placement in a shopping {container} is a strong benign resolution and "
            "terminates the episode even though the item then stops being detected."
        ),
        frames=frames,
        zones=zones,
    )


def item_to_basket() -> Scenario:
    return _item_to_container("ITEM_TO_BASKET", SHELF_AND_BASKET, "basket")


def item_to_cart() -> Scenario:
    return _item_to_container("ITEM_TO_CART", SHELF_AND_CART, "cart")


def temporary_occlusion() -> Scenario:
    """Item is held, briefly lost by the detector, then seen again."""
    frames = build_frames(
        [
            Keyframe(0.0, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(1.8, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(2.5, SHELF_REACH, SHELF_REACH),
            Keyframe(3.4, SHELF_REACH, SHELF_REACH),
            Keyframe(4.3, HELD_OUT, HELD_OUT),
            Keyframe(5.0, HELD_OUT, HELD_OUT),
            # Detector loses the item for ~0.9 s (another shopper walks past).
            Keyframe(5.05, HELD_OUT, None),
            Keyframe(5.9, HELD_OUT, None),
            Keyframe(5.95, HELD_OUT, HELD_OUT),
            Keyframe(8.5, HELD_OUT, HELD_OUT),
        ],
        duration=8.5,
        fps=20.0,
    )
    return Scenario(
        name="TEMPORARY_OCCLUSION",
        description="Held merchandise is briefly occluded and then reappears",
        expected="NO_ALERT",
        rationale=(
            "Disappearance away from any storage region is scored as ordinary "
            "occlusion. Reappearance resets the ladder."
        ),
        frames=frames,
        zones=SHELF_ZONES,
    )


def possible_concealment() -> Scenario:
    """The full sequence: shelf -> hand -> waist -> gone -> stays gone."""
    frames = build_frames(
        [
            Keyframe(0.0, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(2.0, ARM_AT_SIDE, SHELF_REACH),
            Keyframe(2.6, SHELF_REACH, SHELF_REACH),
            Keyframe(3.4, SHELF_REACH, SHELF_REACH),
            # Deliberate move from the shelf face to the right waistband.
            Keyframe(4.2, RIGHT_WAIST, RIGHT_WAIST),
            # Item stops being detected at the waist and never returns.
            Keyframe(4.25, RIGHT_WAIST, None),
            Keyframe(9.0, RIGHT_WAIST, None),
        ],
        duration=9.0,
        fps=20.0,
    )
    return Scenario(
        name="POSSIBLE_CONCEALMENT",
        description=(
            "Shopper takes merchandise from the shelf, moves it to the waistband, "
            "and it is not seen again"
        ),
        expected="ALERT",
        rationale=(
            "Every element of the sequence is present and no benign explanation "
            "appears. This is what the system exists to surface for human review."
        ),
        frames=frames,
        zones=SHELF_ZONES,
    )


def concealment_without_item_detector() -> Scenario:
    """The same movement, but with no merchandise detector available.

    Documents the honest limitation: without item-level detection the system
    has geometry and kinematics only, which is not sufficient evidence to page
    a human. Risk is capped below the alert threshold by design.
    """
    scenario = possible_concealment()
    frames = [
        SimFrame(timestamp=frame.timestamp, persons=frame.persons, items=[])
        for frame in scenario.frames
    ]
    return Scenario(
        name="CONCEALMENT_WITHOUT_ITEM_DETECTOR",
        description="Same movement as POSSIBLE_CONCEALMENT but no item detections exist",
        expected="NO_ALERT",
        rationale=(
            "No COCO class covers general retail merchandise. Without a custom "
            "product model the MVP cannot and must not reach HIGH RISK."
        ),
        frames=frames,
        zones=SHELF_ZONES,
        item_detection_available=False,
    )


#: Every scenario, by name.
SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "NORMAL_BROWSING": normal_browsing,
    "PHONE_INTERACTION": phone_interaction,
    "POCKET_ADJUSTMENT": pocket_adjustment,
    "ITEM_PICKUP_RETURN": item_pickup_return,
    "ITEM_TO_BASKET": item_to_basket,
    "ITEM_TO_CART": item_to_cart,
    "TEMPORARY_OCCLUSION": temporary_occlusion,
    "POSSIBLE_CONCEALMENT": possible_concealment,
    "CONCEALMENT_WITHOUT_ITEM_DETECTOR": concealment_without_item_detector,
}


def get_scenario(name: str) -> Scenario:
    key = name.upper()
    if key not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[key]()


def all_scenarios() -> list[Scenario]:
    return [factory() for factory in SCENARIOS.values()]
