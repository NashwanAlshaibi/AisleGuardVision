"""Typed domain models for AisleGuard Vision.

Every stage of the pipeline exchanges the structures defined here. Nothing in
this module imports torch, ultralytics or cv2 -- these types are the firewall
between the inference backend and the behavior/risk logic, which is what lets
the behavior engine be unit-tested and simulated without any model installed.

Design notes
------------
* Hot-path structures are plain ``dataclass`` objects (not pydantic models):
  they are allocated per person per frame and pydantic validation would show up
  in the profile. Pydantic is used for configuration and API boundaries where
  validation matters more than allocation cost.
* All coordinates are pixel coordinates in the source frame of a single camera.
* All times are POSIX timestamps in seconds (``float``), captured at frame
  acquisition. Never assume a fixed frame rate anywhere in the system.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point:
    """A 2D point in pixel coordinates."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        return float(np.hypot(self.x - other.x, self.y - other.y))

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_int_tuple(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned box in ``xyxy`` pixel coordinates.

    The box is normalized on construction so ``x1 <= x2`` and ``y1 <= y2``
    always hold; downstream geometry may rely on that invariant.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x1 > self.x2:
            object.__setattr__(self, "x1", self.x2)
            object.__setattr__(self, "x2", self.x1)
        if self.y1 > self.y2:
            object.__setattr__(self, "y1", self.y2)
            object.__setattr__(self, "y2", self.y1)

    # -- derived quantities ------------------------------------------------
    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Point:
        return Point((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def bottom_center(self) -> Point:
        return Point((self.x1 + self.x2) * 0.5, self.y2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    # -- relations ---------------------------------------------------------
    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2)))

    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)

    @classmethod
    def from_xyxy(cls, values: Any) -> BoundingBox:
        x1, y1, x2, y2 = (float(v) for v in values)
        return cls(x1, y1, x2, y2)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BoundingBox:
        return cls(cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)

    def intersection_area(self, other: BoundingBox) -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def iou(self, other: BoundingBox) -> float:
        inter = self.intersection_area(other)
        if inter <= 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def containment_of(self, other: BoundingBox) -> float:
        """Fraction of ``other`` contained in ``self`` (0..1).

        Preferred over IoU when comparing very different sizes -- e.g. a small
        item box against a large person box, where IoU is tiny even at full
        containment.
        """
        if other.area <= 0.0:
            return 0.0
        return self.intersection_area(other) / other.area

    def contains_point(self, point: Point) -> bool:
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2

    def expanded(self, ratio: float) -> BoundingBox:
        """Return the box scaled about its center by ``1 + ratio``."""
        dw = self.width * ratio * 0.5
        dh = self.height * ratio * 0.5
        return BoundingBox(self.x1 - dw, self.y1 - dh, self.x2 + dw, self.y2 + dh)

    def clipped_to(self, width: int, height: int) -> BoundingBox:
        return BoundingBox(
            max(0.0, min(self.x1, width - 1.0)),
            max(0.0, min(self.y1, height - 1.0)),
            max(0.0, min(self.x2, width - 1.0)),
            max(0.0, min(self.y2, height - 1.0)),
        )


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------


class KeypointName(str, Enum):
    """The 17 COCO keypoints, in canonical model output order."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


#: COCO keypoint order as emitted by YOLO pose models.
COCO_KEYPOINT_ORDER: tuple[KeypointName, ...] = tuple(KeypointName)

#: Skeleton edges used for visualization only.
COCO_SKELETON: tuple[tuple[KeypointName, KeypointName], ...] = (
    (KeypointName.LEFT_SHOULDER, KeypointName.RIGHT_SHOULDER),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_ELBOW),
    (KeypointName.LEFT_ELBOW, KeypointName.LEFT_WRIST),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_ELBOW),
    (KeypointName.RIGHT_ELBOW, KeypointName.RIGHT_WRIST),
    (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_HIP),
    (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.RIGHT_HIP),
    (KeypointName.LEFT_HIP, KeypointName.LEFT_KNEE),
    (KeypointName.LEFT_KNEE, KeypointName.LEFT_ANKLE),
    (KeypointName.RIGHT_HIP, KeypointName.RIGHT_KNEE),
    (KeypointName.RIGHT_KNEE, KeypointName.RIGHT_ANKLE),
    (KeypointName.NOSE, KeypointName.LEFT_EYE),
    (KeypointName.NOSE, KeypointName.RIGHT_EYE),
    (KeypointName.LEFT_EYE, KeypointName.LEFT_EAR),
    (KeypointName.RIGHT_EYE, KeypointName.RIGHT_EAR),
)


class Hand(str, Enum):
    """Which wrist an observation refers to."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def wrist(self) -> KeypointName:
        return KeypointName.LEFT_WRIST if self is Hand.LEFT else KeypointName.RIGHT_WRIST

    @property
    def elbow(self) -> KeypointName:
        return KeypointName.LEFT_ELBOW if self is Hand.LEFT else KeypointName.RIGHT_ELBOW

    @property
    def shoulder(self) -> KeypointName:
        return KeypointName.LEFT_SHOULDER if self is Hand.LEFT else KeypointName.RIGHT_SHOULDER


@dataclass(frozen=True, slots=True)
class Keypoint:
    """A single pose keypoint with its detection confidence."""

    name: KeypointName
    x: float
    y: float
    confidence: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)

    def is_reliable(self, threshold: float) -> bool:
        return self.confidence >= threshold


@dataclass(slots=True)
class PoseObservation:
    """A pose associated (or not yet associated) with a person track."""

    keypoints: dict[KeypointName, Keypoint]
    confidence: float
    bbox: BoundingBox | None = None
    timestamp: float = field(default_factory=time.time)

    def get(self, name: KeypointName) -> Keypoint | None:
        return self.keypoints.get(name)

    def point_of(self, name: KeypointName, min_confidence: float = 0.0) -> Point | None:
        kp = self.keypoints.get(name)
        if kp is None or kp.confidence < min_confidence:
            return None
        return kp.point

    def wrist(self, hand: Hand, min_confidence: float = 0.0) -> Point | None:
        return self.point_of(hand.wrist, min_confidence)

    @property
    def mean_confidence(self) -> float:
        if not self.keypoints:
            return 0.0
        return float(np.mean([kp.confidence for kp in self.keypoints.values()]))

    def torso_confidence(self) -> float:
        """Mean confidence of the keypoints the behavior engine actually relies on.

        Overall pose confidence is dominated by the face keypoints, which are
        often crisp even when the torso and arms -- the parts we reason about --
        are not. This is the number that gates behavior decisions.
        """
        names = (
            KeypointName.LEFT_SHOULDER,
            KeypointName.RIGHT_SHOULDER,
            KeypointName.LEFT_HIP,
            KeypointName.RIGHT_HIP,
            KeypointName.LEFT_WRIST,
            KeypointName.RIGHT_WRIST,
        )
        values = [self.keypoints[n].confidence for n in names if n in self.keypoints]
        if not values:
            return 0.0
        return float(np.mean(values))

    @classmethod
    def from_array(
        cls,
        array: np.ndarray,
        bbox: BoundingBox | None = None,
        timestamp: float | None = None,
    ) -> PoseObservation:
        """Build from a ``(17, 3)`` ``[x, y, confidence]`` array in COCO order."""
        if array.ndim != 2 or array.shape[0] != len(COCO_KEYPOINT_ORDER):
            raise ValueError(f"expected ({len(COCO_KEYPOINT_ORDER)}, 3) keypoint array, got {array.shape}")
        keypoints: dict[KeypointName, Keypoint] = {}
        for name, row in zip(COCO_KEYPOINT_ORDER, array, strict=True):
            conf = float(row[2]) if array.shape[1] > 2 else 1.0
            keypoints[name] = Keypoint(name=name, x=float(row[0]), y=float(row[1]), confidence=conf)
        confidence = float(np.mean(array[:, 2])) if array.shape[1] > 2 else 1.0
        return cls(
            keypoints=keypoints,
            confidence=confidence,
            bbox=bbox,
            timestamp=timestamp if timestamp is not None else time.time(),
        )


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------


class ObjectClass(str, Enum):
    """Semantic roles the pipeline cares about, decoupled from model class ids.

    A backend maps its own label space onto these. COCO gives us PERSON and a
    handful of carryables; everything a retail model would add (merchandise
    SKUs, shelf facings) maps to MERCHANDISE.
    """

    PERSON = "person"
    PHONE = "phone"
    HANDBAG = "handbag"
    BACKPACK = "backpack"
    SUITCASE = "suitcase"
    BOTTLE = "bottle"
    CUP = "cup"
    BOOK = "book"
    SHOPPING_CART = "shopping_cart"
    BASKET = "basket"
    MERCHANDISE = "merchandise"
    UNKNOWN = "unknown"

    @property
    def is_carryable(self) -> bool:
        """True if an instance of this class can plausibly be picked up."""
        return self in _CARRYABLE_CLASSES

    @property
    def is_container(self) -> bool:
        return self in (ObjectClass.SHOPPING_CART, ObjectClass.BASKET)

    @property
    def is_personal_effect(self) -> bool:
        """Items a shopper plausibly brought with them, not store merchandise."""
        return self in (
            ObjectClass.PHONE,
            ObjectClass.HANDBAG,
            ObjectClass.BACKPACK,
            ObjectClass.SUITCASE,
        )


_CARRYABLE_CLASSES = frozenset(
    {
        ObjectClass.PHONE,
        ObjectClass.HANDBAG,
        ObjectClass.BOTTLE,
        ObjectClass.CUP,
        ObjectClass.BOOK,
        ObjectClass.MERCHANDISE,
    }
)


@dataclass(slots=True)
class Detection:
    """A single normalized detection.

    This is the *only* structure a detector backend is allowed to emit. The
    behavior engine never sees an Ultralytics ``Results`` object, an ONNX
    tensor or a Triton response.
    """

    bbox: BoundingBox
    confidence: float
    object_class: ObjectClass = ObjectClass.UNKNOWN
    class_id: int = -1
    class_name: str = ""
    #: Optional pose emitted by a pose model alongside the box.
    pose: PoseObservation | None = None
    #: Free-form backend annotations (e.g. segmentation mask handle, SKU id).
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InferenceResult:
    """Everything one model invocation produced for one frame."""

    camera_id: str
    frame_id: int
    timestamp: float
    detections: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    model_name: str = ""
    #: Dimensions of the image the detections are expressed in.
    frame_width: int = 0
    frame_height: int = 0

    def of_class(self, object_class: ObjectClass) -> list[Detection]:
        return [d for d in self.detections if d.object_class is object_class]


@dataclass(slots=True)
class Frame:
    """One decoded camera frame travelling through the pipeline."""

    camera_id: str
    frame_id: int
    #: Acquisition time. All temporal reasoning is anchored to this value.
    timestamp: float
    image: np.ndarray
    #: Frames the decoder dropped since the previous delivered frame.
    dropped_before: int = 0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------


class TrackState(str, Enum):
    """Lifecycle of a tracker hypothesis."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(slots=True)
class TrajectoryPoint:
    timestamp: float
    position: Point


@dataclass(slots=True)
class PersonTrack:
    """A tracked shopper within a single camera session.

    ``track_id`` is a *temporary computer-vision identifier*. It is scoped to
    one camera and one session, is reused after the track is removed, carries
    no identity information and is never persisted as a person identifier.
    AisleGuard Vision performs no facial, identity, demographic or biometric
    recognition of any kind.
    """

    track_id: int
    camera_id: str
    bbox: BoundingBox
    confidence: float
    first_seen: float
    last_seen: float
    state: TrackState = TrackState.TENTATIVE
    pose: PoseObservation | None = None
    pose_timestamp: float | None = None
    trajectory: deque[TrajectoryPoint] = field(default_factory=lambda: deque(maxlen=256))
    #: Number of tracker updates that matched a detection.
    hit_count: int = 0
    #: Consecutive updates without a matching detection.
    time_since_update: int = 0
    #: Smoothed centre velocity in pixels/second.
    velocity: Point = field(default_factory=lambda: Point(0.0, 0.0))

    @property
    def center(self) -> Point:
        return self.bbox.center

    @property
    def track_age(self) -> float:
        """Seconds since the track was first observed."""
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def body_height(self) -> float:
        """Reference length for normalizing distances.

        Prefer the shoulder-to-hip span when pose is reliable (robust to the
        box growing when arms extend); fall back to the box height.
        """
        if self.pose is not None:
            torso = _torso_span(self.pose)
            if torso is not None and torso > 1.0:
                # Shoulder->hip is roughly 30% of standing height.
                return torso / 0.3
        return max(1.0, self.bbox.height)

    @property
    def shoulder_width(self) -> float:
        if self.pose is not None:
            left = self.pose.point_of(KeypointName.LEFT_SHOULDER)
            right = self.pose.point_of(KeypointName.RIGHT_SHOULDER)
            if left is not None and right is not None:
                width = left.distance_to(right)
                if width > 1.0:
                    return width
        return max(1.0, self.bbox.width * 0.5)

    def has_fresh_pose(self, now: float, max_age: float) -> bool:
        return (
            self.pose is not None
            and self.pose_timestamp is not None
            and (now - self.pose_timestamp) <= max_age
        )

    def recent_trajectory(self, seconds: float, now: float | None = None) -> list[TrajectoryPoint]:
        reference = now if now is not None else self.last_seen
        cutoff = reference - seconds
        return [p for p in self.trajectory if p.timestamp >= cutoff]


def _torso_span(pose: PoseObservation) -> float | None:
    """Mean shoulder-to-hip distance, or ``None`` if keypoints are missing."""
    spans: list[float] = []
    for shoulder, hip in (
        (KeypointName.LEFT_SHOULDER, KeypointName.LEFT_HIP),
        (KeypointName.RIGHT_SHOULDER, KeypointName.RIGHT_HIP),
    ):
        s = pose.point_of(shoulder)
        h = pose.point_of(hip)
        if s is not None and h is not None:
            spans.append(s.distance_to(h))
    if not spans:
        return None
    return float(np.mean(spans))


class ItemStatus(str, Enum):
    """Observation state of a merchandise candidate.

    Disappearance is a *ladder*, never a conclusion. An item must spend a
    configurable amount of time in OCCLUDED before it is allowed to become
    MISSING, and MISSING on its own is still not an alert.
    """

    VISIBLE = "VISIBLE"
    POSSIBLY_OCCLUDED = "POSSIBLY_OCCLUDED"
    OCCLUDED = "OCCLUDED"
    MISSING = "MISSING"
    RETURNED = "RETURNED"
    IN_BASKET = "IN_BASKET"
    IN_CART = "IN_CART"
    UNKNOWN = "UNKNOWN"

    @property
    def is_resolved_benign(self) -> bool:
        """True for outcomes that explain the item without concealment."""
        return self in (ItemStatus.RETURNED, ItemStatus.IN_BASKET, ItemStatus.IN_CART)


@dataclass(slots=True)
class ItemDetection:
    """A merchandise-candidate detection, before temporal tracking."""

    bbox: BoundingBox
    confidence: float
    object_class: ObjectClass = ObjectClass.MERCHANDISE
    class_name: str = ""
    #: Where this candidate came from, e.g. "coco_proxy", "custom_retail_v1".
    source: str = "unknown"
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ItemTrack:
    """A merchandise candidate tracked over time."""

    item_id: int
    camera_id: str
    bbox: BoundingBox
    confidence: float
    first_seen: float
    last_seen: float
    object_class: ObjectClass = ObjectClass.MERCHANDISE
    status: ItemStatus = ItemStatus.VISIBLE
    source: str = "unknown"
    trajectory: deque[TrajectoryPoint] = field(default_factory=lambda: deque(maxlen=256))
    #: Person track this item is currently believed to be held by.
    associated_person_id: int | None = None
    associated_hand: Hand | None = None
    association_confidence: float = 0.0
    #: When the current association was first established (for dwell checks).
    association_since: float | None = None
    #: Last position observed while the item was VISIBLE.
    last_visible_position: Point | None = None
    last_visible_timestamp: float | None = None
    #: Set when the item stopped being observed, used for the status ladder.
    disappeared_at: float | None = None
    hit_count: int = 0

    @property
    def center(self) -> Point:
        return self.bbox.center

    @property
    def track_age(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def association_duration(self) -> float:
        if self.association_since is None:
            return 0.0
        return max(0.0, self.last_seen - self.association_since)

    def is_stable(self, min_hits: int, min_age_seconds: float) -> bool:
        """Whether this track is trustworthy enough to reason about.

        An unstable item track is treated as *negative* evidence: a flickering
        detection is a far more likely explanation for a disappearance than
        concealment.
        """
        return self.hit_count >= min_hits and self.track_age >= min_age_seconds


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------


class ZoneKind(str, Enum):
    """Semantic role of a configured region of interest."""

    SHELF = "shelf"
    HIGH_VALUE = "high_value"
    CHECKOUT = "checkout"
    BASKET = "basket"
    CART = "cart"
    ENTRANCE = "entrance"
    EXCLUSION = "exclusion"

    @property
    def is_merchandise_source(self) -> bool:
        return self in (ZoneKind.SHELF, ZoneKind.HIGH_VALUE)

    @property
    def is_container(self) -> bool:
        return self in (ZoneKind.BASKET, ZoneKind.CART)


@dataclass(slots=True)
class ShelfZone:
    """A configured polygonal region of interest on one camera.

    Named ``ShelfZone`` for continuity with the product spec; ``kind``
    generalizes it to checkout, basket, cart, entrance and exclusion regions.
    """

    zone_id: str
    camera_id: str
    polygon: np.ndarray  # (N, 2) float32
    kind: ZoneKind = ZoneKind.SHELF
    name: str = ""
    #: Multiplies risk contributions originating in this zone (high-value aisles).
    risk_multiplier: float = 1.0
    enabled: bool = True
    #: Cached derived geometry, filled by the zone registry.
    _bounds: BoundingBox | None = field(default=None, repr=False, compare=False)

    @property
    def bounds(self) -> BoundingBox:
        if self._bounds is None:
            xs = self.polygon[:, 0]
            ys = self.polygon[:, 1]
            self._bounds = BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return self._bounds


# ---------------------------------------------------------------------------
# Behavior, evidence and risk
# ---------------------------------------------------------------------------


class BehaviorState(str, Enum):
    """States of the per-person temporal behavior machine.

    Reaching ``REVIEW_ALERT`` means *possible concealment behavior was observed
    and a human should look at the clip*. It is never a determination that a
    theft occurred.
    """

    IDLE = "IDLE"
    SHELF_INTERACTION = "SHELF_INTERACTION"
    ITEM_ASSOCIATED = "ITEM_ASSOCIATED"
    ITEM_REMOVED_FROM_SHELF = "ITEM_REMOVED_FROM_SHELF"
    HAND_MOVING_TO_STORAGE = "HAND_MOVING_TO_STORAGE"
    POSSIBLE_CONCEALMENT = "POSSIBLE_CONCEALMENT"
    ITEM_OCCLUDED = "ITEM_OCCLUDED"
    ITEM_MISSING = "ITEM_MISSING"
    REVIEW_ALERT = "REVIEW_ALERT"
    # Benign terminal branches -- these end the episode and reset to IDLE.
    ITEM_RETURNED = "ITEM_RETURNED"
    ITEM_TO_BASKET = "ITEM_TO_BASKET"
    ITEM_TO_CART = "ITEM_TO_CART"

    @property
    def is_benign_terminal(self) -> bool:
        return self in (
            BehaviorState.ITEM_RETURNED,
            BehaviorState.ITEM_TO_BASKET,
            BehaviorState.ITEM_TO_CART,
        )


class StorageRegion(str, Enum):
    """Plausible concealment destinations estimated from pose geometry.

    These are *geometric priors only*. A wrist near one of these regions is
    weak evidence on its own -- people put their hands in their pockets
    constantly. It only matters in combination with a tracked item.
    """

    LEFT_WAIST = "left_waist"
    RIGHT_WAIST = "right_waist"
    FRONT_WAIST = "front_waist"
    TORSO = "torso"
    JACKET_AREA = "jacket_area"
    BAG_AREA = "bag_area"
    BACKPACK_AREA = "backpack_area"


@dataclass(slots=True)
class StorageRegionEstimate:
    """A body-relative region with a radius scaled to the subject's size."""

    region: StorageRegion
    center: Point
    radius: float
    confidence: float

    def contains(self, point: Point) -> bool:
        return self.center.distance_to(point) <= self.radius

    def normalized_distance(self, point: Point) -> float:
        """Distance to the region centre expressed in radii (0 == centre)."""
        if self.radius <= 0:
            return float("inf")
        return self.center.distance_to(point) / self.radius


class EvidenceType(str, Enum):
    """Observations the risk engine scores.

    Positive types support a concealment hypothesis; NEGATIVE_* types argue
    against it. False-positive reduction is why the negative set exists and is
    weighted far more heavily than any single positive.
    """

    # -- positive ----------------------------------------------------------
    SHELF_INTERACTION = "SHELF_INTERACTION"
    HIGH_VALUE_ZONE_INTERACTION = "HIGH_VALUE_ZONE_INTERACTION"
    STABLE_HAND_ITEM_ASSOCIATION = "STABLE_HAND_ITEM_ASSOCIATION"
    ITEM_REMOVED_FROM_SHELF = "ITEM_REMOVED_FROM_SHELF"
    HAND_MOVED_TO_STORAGE_REGION = "HAND_MOVED_TO_STORAGE_REGION"
    CONCEALMENT_MOTION_PROFILE = "CONCEALMENT_MOTION_PROFILE"
    ITEM_DISAPPEARED_NEAR_STORAGE = "ITEM_DISAPPEARED_NEAR_STORAGE"
    ITEM_REMAINS_MISSING = "ITEM_REMAINS_MISSING"
    NO_BASKET_PLACEMENT = "NO_BASKET_PLACEMENT"
    NO_SHELF_RETURN = "NO_SHELF_RETURN"

    # -- negative ----------------------------------------------------------
    ITEM_RETURNED_TO_SHELF = "ITEM_RETURNED_TO_SHELF"
    ITEM_VISIBLE_IN_HAND = "ITEM_VISIBLE_IN_HAND"
    ITEM_PLACED_IN_BASKET = "ITEM_PLACED_IN_BASKET"
    ITEM_PLACED_IN_CART = "ITEM_PLACED_IN_CART"
    NORMAL_PHONE_INTERACTION = "NORMAL_PHONE_INTERACTION"
    PERSONAL_EFFECT_INTERACTION = "PERSONAL_EFFECT_INTERACTION"
    TEMPORARY_OCCLUSION = "TEMPORARY_OCCLUSION"
    UNSTABLE_ITEM_TRACK = "UNSTABLE_ITEM_TRACK"
    LOW_POSE_CONFIDENCE = "LOW_POSE_CONFIDENCE"
    LOW_PERSON_TRACK_CONFIDENCE = "LOW_PERSON_TRACK_CONFIDENCE"
    CAMERA_OCCLUSION = "CAMERA_OCCLUSION"
    SHORT_ACCIDENTAL_OVERLAP = "SHORT_ACCIDENTAL_OVERLAP"
    NO_MERCHANDISE_INTERACTION = "NO_MERCHANDISE_INTERACTION"
    ITEM_DETECTION_UNAVAILABLE = "ITEM_DETECTION_UNAVAILABLE"

    @property
    def is_negative(self) -> bool:
        return self in _NEGATIVE_EVIDENCE


_NEGATIVE_EVIDENCE = frozenset(
    {
        EvidenceType.ITEM_RETURNED_TO_SHELF,
        EvidenceType.ITEM_VISIBLE_IN_HAND,
        EvidenceType.ITEM_PLACED_IN_BASKET,
        EvidenceType.ITEM_PLACED_IN_CART,
        EvidenceType.NORMAL_PHONE_INTERACTION,
        EvidenceType.PERSONAL_EFFECT_INTERACTION,
        EvidenceType.TEMPORARY_OCCLUSION,
        EvidenceType.UNSTABLE_ITEM_TRACK,
        EvidenceType.LOW_POSE_CONFIDENCE,
        EvidenceType.LOW_PERSON_TRACK_CONFIDENCE,
        EvidenceType.CAMERA_OCCLUSION,
        EvidenceType.SHORT_ACCIDENTAL_OVERLAP,
        EvidenceType.NO_MERCHANDISE_INTERACTION,
        EvidenceType.ITEM_DETECTION_UNAVAILABLE,
    }
)


@dataclass(slots=True)
class EvidenceEvent:
    """One scored observation, with the human-readable reason it was raised."""

    evidence_type: EvidenceType
    timestamp: float
    #: Human-readable justification; surfaced verbatim in the alert payload.
    description: str
    #: Scales the configured weight (0..1). Typically an association or pose
    #: confidence, so weak observations contribute proportionally less.
    confidence: float = 1.0
    person_id: int | None = None
    item_id: int | None = None
    zone_id: str | None = None
    hand: Hand | None = None
    #: Seconds this observation stays in the ledger. ``None`` means it persists
    #: for the lifetime of the interaction episode.
    ttl: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_negative(self) -> bool:
        return self.evidence_type.is_negative

    @property
    def dedup_key(self) -> tuple[Any, ...]:
        """Evidence identity: re-raising the same observation refreshes it."""
        return (self.evidence_type, self.item_id, self.zone_id, self.hand)

    def is_active(self, now: float) -> bool:
        if self.ttl is None:
            return True
        return (now - self.timestamp) <= self.ttl


@dataclass(slots=True)
class ShelfInteraction:
    """Structured SHELF_INTERACTION evidence record (spec Stage 4)."""

    person_id: int
    camera_id: str
    timestamp: float
    hand: Hand
    zone_id: str
    position: Point
    confidence: float
    associated_item_id: int | None = None
    #: How long the wrist has continuously satisfied the interaction criteria.
    duration: float = 0.0


class ThreatLevel(str, Enum):
    LOW = "LOW"
    ELEVATED = "ELEVATED"
    REVIEW = "REVIEW"
    HIGH_RISK = "HIGH_RISK"

    @classmethod
    def from_score(
        cls,
        score: float,
        elevated: float = 30.0,
        review: float = 60.0,
        high: float = 85.0,
    ) -> ThreatLevel:
        if score >= high:
            return cls.HIGH_RISK
        if score >= review:
            return cls.REVIEW
        if score >= elevated:
            return cls.ELEVATED
        return cls.LOW


@dataclass(slots=True)
class RiskContribution:
    """One line of the risk arithmetic, kept for explainability."""

    evidence_type: EvidenceType
    description: str
    base_weight: float
    confidence: float
    applied_weight: float


@dataclass(slots=True)
class RiskAssessment:
    """The explainable output of the risk engine.

    ``risk_score`` is a clamped sum of configured evidence weights, not a
    neural network output. Every point is attributable to a contribution.
    """

    person_id: int
    camera_id: str
    timestamp: float
    risk_score: float
    threat_level: ThreatLevel
    behavior_state: BehaviorState
    contributions: list[RiskContribution] = field(default_factory=list)
    raw_score: float = 0.0

    @property
    def positive_evidence(self) -> list[str]:
        return [c.description for c in self.contributions if c.applied_weight > 0]

    @property
    def negative_evidence(self) -> list[str]:
        return [c.description for c in self.contributions if c.applied_weight < 0]

    def explain(self) -> str:
        lines = [
            f"risk={self.risk_score:.1f} ({self.threat_level.value}) state={self.behavior_state.value}"
        ]
        lines.extend(
            f"  {c.applied_weight:+7.2f}  {c.evidence_type.value}: {c.description}"
            for c in sorted(self.contributions, key=lambda c: -abs(c.applied_weight))
        )
        return "\n".join(lines)


@dataclass(slots=True)
class BehaviorObservation:
    """Per-person output of one behavior-engine update."""

    person_id: int
    camera_id: str
    timestamp: float
    state: BehaviorState
    previous_state: BehaviorState
    risk: RiskAssessment
    #: Evidence raised during *this* update only.
    new_evidence: list[EvidenceEvent] = field(default_factory=list)
    #: All evidence currently active in the episode ledger.
    active_evidence: list[EvidenceEvent] = field(default_factory=list)
    storage_regions: list[StorageRegionEstimate] = field(default_factory=list)
    associated_item_id: int | None = None
    associated_hand: Hand | None = None

    @property
    def state_changed(self) -> bool:
        return self.state is not self.previous_state


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    POSSIBLE_CONCEALMENT = "possible_concealment"
    ELEVATED_BEHAVIOR = "elevated_behavior"
    CAMERA_OFFLINE = "camera_offline"
    CAMERA_RECOVERED = "camera_recovered"


@dataclass(slots=True)
class SecurityEvent:
    """A recorded incident awaiting human review.

    An event means: *possible concealment behavior detected -- human review
    recommended*. It is explicitly not a determination that a theft occurred,
    and must never be presented or acted on as one.
    """

    event_id: str
    camera_id: str
    person_id: int
    timestamp: float
    risk_score: float
    threat_level: ThreatLevel
    behavior_state: BehaviorState
    event_type: EventType = EventType.POSSIBLE_CONCEALMENT
    positive_evidence: list[str] = field(default_factory=list)
    negative_evidence: list[str] = field(default_factory=list)
    contributions: list[RiskContribution] = field(default_factory=list)
    snapshot_path: str | None = None
    clip_path: str | None = None
    metadata_path: str | None = None
    track_metadata: dict[str, Any] = field(default_factory=dict)
    #: Human review outcome, filled in by the dashboard later.
    review_status: str = "pending"

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())


class CameraState(str, Enum):
    """Connection lifecycle of a camera worker."""

    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    DISABLED = "disabled"
    FAILED = "failed"

    @property
    def is_healthy(self) -> bool:
        return self is CameraState.STREAMING


@dataclass(slots=True)
class CameraStatus:
    """Health snapshot for one camera, surfaced by the API and the overlay."""

    camera_id: str
    name: str
    state: CameraState = CameraState.IDLE
    enabled: bool = True
    #: Sanitized source (credentials stripped). Safe to log and to return over the API.
    source_label: str = ""
    last_frame_timestamp: float | None = None
    frames_received: int = 0
    frames_dropped: int = 0
    decode_fps: float = 0.0
    inference_fps: float = 0.0
    pose_fps: float = 0.0
    reconnect_count: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    resolution: tuple[int, int] | None = None
    active_tracks: int = 0

    def is_stalled(self, now: float, timeout: float) -> bool:
        if self.state is not CameraState.STREAMING:
            return False
        if self.last_frame_timestamp is None:
            return False
        return (now - self.last_frame_timestamp) > timeout
