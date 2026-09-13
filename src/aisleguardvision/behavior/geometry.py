"""Geometric primitives for zones, storage regions and motion analysis.

Two rules govern everything here:

1. **NumPy, not Python loops.** Point-in-polygon runs against every tracked
   wrist on every frame for every zone; it is vectorized over points.
2. **No fixed pixel thresholds.** Every distance that feeds a decision is
   normalized by the subject's apparent size (body height or shoulder width),
   so the same configuration works for a shopper near the camera and one at
   the far end of the aisle.

This module has no dependency on cv2, torch or the inference layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import StorageRegionConfig
from ..core.types import (
    BoundingBox,
    Hand,
    KeypointName,
    PersonTrack,
    Point,
    PoseObservation,
    StorageRegion,
    StorageRegionEstimate,
    TrajectoryPoint,
)

EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


def as_polygon(points: object) -> np.ndarray:
    """Coerce a vertex sequence into an ``(N, 2)`` float32 array."""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"polygon must be an (N, 2) array of vertices, got shape {array.shape}")
    if array.shape[0] < 3:
        raise ValueError("polygon needs at least 3 vertices")
    return array


def polygon_bounds(polygon: np.ndarray) -> BoundingBox:
    return BoundingBox(
        float(polygon[:, 0].min()),
        float(polygon[:, 1].min()),
        float(polygon[:, 0].max()),
        float(polygon[:, 1].max()),
    )


def polygon_area(polygon: np.ndarray) -> float:
    """Absolute area via the shoelace formula."""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_centroid(polygon: np.ndarray) -> Point:
    """Area-weighted centroid, falling back to the vertex mean for degenerate
    (zero-area) polygons."""
    x = polygon[:, 0].astype(np.float64)
    y = polygon[:, 1].astype(np.float64)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() * 0.5
    if abs(area) < EPSILON:
        return Point(float(x.mean()), float(y.mean()))
    cx = float(((x + x_next) * cross).sum() / (6.0 * area))
    cy = float(((y + y_next) * cross).sum() / (6.0 * area))
    return Point(cx, cy)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized crossing-number test.

    Parameters
    ----------
    points:
        ``(M, 2)`` array of query points.
    polygon:
        ``(N, 2)`` array of vertices, implicitly closed.

    Returns
    -------
    ``(M,)`` boolean array. Points exactly on an edge are not guaranteed to
    be classified consistently, which is irrelevant at pixel scale.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.size == 0:
        return np.zeros((0,), dtype=bool)
    poly = np.asarray(polygon, dtype=np.float64)

    x = pts[:, 0][:, None]  # (M, 1)
    y = pts[:, 1][:, None]

    x1 = poly[:, 0][None, :]  # (1, N)
    y1 = poly[:, 1][None, :]
    x2 = np.roll(poly[:, 0], -1)[None, :]
    y2 = np.roll(poly[:, 1], -1)[None, :]

    # Edge straddles the horizontal ray from the point.
    straddles = (y1 > y) != (y2 > y)
    # X coordinate where the edge crosses the ray.
    denominator = np.where(np.abs(y2 - y1) < EPSILON, EPSILON, y2 - y1)
    x_intersect = x1 + (y - y1) * (x2 - x1) / denominator
    crossings = straddles & (x < x_intersect)
    return (crossings.sum(axis=1) % 2).astype(bool)


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    """Single-point convenience wrapper around :func:`points_in_polygon`."""
    return bool(points_in_polygon(np.array([[point.x, point.y]]), polygon)[0])


def distance_points_to_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Minimum Euclidean distance from each point to the polygon boundary.

    Returns ``0.0`` for points inside the polygon, so callers can treat the
    result as "how far outside am I".
    """
    pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if pts.size == 0:
        return np.zeros((0,), dtype=np.float64)
    poly = np.asarray(polygon, dtype=np.float64)

    a = poly  # (N, 2) segment starts
    b = np.roll(poly, -1, axis=0)  # (N, 2) segment ends
    ab = b - a  # (N, 2)
    ab_len_sq = np.maximum((ab**2).sum(axis=1), EPSILON)  # (N,)

    ap = pts[:, None, :] - a[None, :, :]  # (M, N, 2)
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / ab_len_sq[None, :], 0.0, 1.0)  # (M, N)
    projection = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (M, N, 2)
    distances = np.linalg.norm(pts[:, None, :] - projection, axis=2)  # (M, N)
    boundary_distance = distances.min(axis=1)

    inside = points_in_polygon(pts, poly)
    return np.where(inside, 0.0, boundary_distance)


def distance_point_to_polygon(point: Point, polygon: np.ndarray) -> float:
    return float(distance_points_to_polygon(np.array([[point.x, point.y]]), polygon)[0])


def box_polygon_overlap_ratio(box: BoundingBox, polygon: np.ndarray, samples: int = 8) -> float:
    """Approximate fraction of ``box`` lying inside ``polygon``.

    Uses a ``samples x samples`` regular grid rather than exact polygon
    clipping: the exact answer costs a Sutherland-Hodgman clip per box per
    zone per frame and buys nothing at this scale, whereas a 64-point sample
    is a single vectorized call.
    """
    if box.area <= 0:
        return 0.0
    xs = np.linspace(box.x1, box.x2, samples)
    ys = np.linspace(box.y1, box.y2, samples)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return float(points_in_polygon(points, polygon).mean())


# ---------------------------------------------------------------------------
# Body-relative normalization
# ---------------------------------------------------------------------------


def normalized_distance(a: Point, b: Point, reference_length: float) -> float:
    """Distance between two points expressed in body-height units."""
    if reference_length <= EPSILON:
        return float("inf")
    return a.distance_to(b) / reference_length


def body_axes(
    pose: PoseObservation, min_confidence: float = 0.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return unit vectors ``(down_axis, right_axis)`` in image coordinates.

    ``down_axis`` points from the shoulder line toward the hip line -- i.e.
    "down the body" even when the subject is leaning or the camera is tilted.
    ``right_axis`` points from the subject's left shoulder to their right
    shoulder. Returns ``None`` when the required keypoints are unavailable.
    """
    left_shoulder = pose.point_of(KeypointName.LEFT_SHOULDER, min_confidence)
    right_shoulder = pose.point_of(KeypointName.RIGHT_SHOULDER, min_confidence)
    left_hip = pose.point_of(KeypointName.LEFT_HIP, min_confidence)
    right_hip = pose.point_of(KeypointName.RIGHT_HIP, min_confidence)
    if None in (left_shoulder, right_shoulder, left_hip, right_hip):
        return None
    assert left_shoulder and right_shoulder and left_hip and right_hip  # for type checkers

    shoulder_center = np.array(
        [(left_shoulder.x + right_shoulder.x) * 0.5, (left_shoulder.y + right_shoulder.y) * 0.5]
    )
    hip_center = np.array([(left_hip.x + right_hip.x) * 0.5, (left_hip.y + right_hip.y) * 0.5])
    down = hip_center - shoulder_center
    down_norm = np.linalg.norm(down)
    if down_norm < 1.0:
        # Degenerate (subject nearly edge-on or bad keypoints): fall back to
        # image-down, which is correct for a typical ceiling-mounted camera.
        down = np.array([0.0, 1.0])
    else:
        down = down / down_norm

    right = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
    right_norm = np.linalg.norm(right)
    right = right / right_norm if right_norm >= 1.0 else np.array([1.0, 0.0])
    return down, right


# ---------------------------------------------------------------------------
# Storage regions
# ---------------------------------------------------------------------------


def estimate_storage_regions(
    track: PersonTrack,
    config: StorageRegionConfig,
    keypoint_confidence: float = 0.3,
    bag_boxes: list[BoundingBox] | None = None,
) -> list[StorageRegionEstimate]:
    """Estimate plausible concealment destinations from pose geometry.

    These are geometric priors, nothing more. A wrist inside one of these
    regions is *not* evidence of concealment: people put their hands in their
    pockets, adjust clothing and reach for phones constantly. The regions only
    become meaningful in combination with a tracked merchandise item that
    moved with that same wrist -- see :mod:`aisleguardvision.behavior.engine`.

    All radii and offsets scale with the subject's apparent body size, so a
    person at 3 m and a person at 12 m get proportionally sized regions.
    """
    pose = track.pose
    if pose is None:
        return []

    axes = body_axes(pose, keypoint_confidence)
    if axes is None:
        return []
    down, right = axes

    left_hip = pose.point_of(KeypointName.LEFT_HIP, keypoint_confidence)
    right_hip = pose.point_of(KeypointName.RIGHT_HIP, keypoint_confidence)
    left_shoulder = pose.point_of(KeypointName.LEFT_SHOULDER, keypoint_confidence)
    right_shoulder = pose.point_of(KeypointName.RIGHT_SHOULDER, keypoint_confidence)
    if None in (left_hip, right_hip, left_shoulder, right_shoulder):
        return []
    assert left_hip and right_hip and left_shoulder and right_shoulder

    body_height = track.body_height
    shoulder_width = track.shoulder_width
    waist_radius = body_height * config.waist_radius_ratio
    torso_radius = body_height * config.torso_radius_ratio
    lateral = shoulder_width * config.waist_lateral_offset_ratio

    hip_center = Point((left_hip.x + right_hip.x) * 0.5, (left_hip.y + right_hip.y) * 0.5)
    shoulder_center = Point(
        (left_shoulder.x + right_shoulder.x) * 0.5, (left_shoulder.y + right_shoulder.y) * 0.5
    )

    # Confidence of the estimate is the weakest keypoint it depends on.
    anchor_confidence = min(
        pose.keypoints[KeypointName.LEFT_HIP].confidence,
        pose.keypoints[KeypointName.RIGHT_HIP].confidence,
        pose.keypoints[KeypointName.LEFT_SHOULDER].confidence,
        pose.keypoints[KeypointName.RIGHT_SHOULDER].confidence,
    )

    def offset(origin: Point, along_right: float = 0.0, along_down: float = 0.0) -> Point:
        return Point(
            origin.x + right[0] * along_right + down[0] * along_down,
            origin.y + right[1] * along_right + down[1] * along_down,
        )

    regions = [
        # Side pockets / waistband, offset outward from each hip.
        StorageRegionEstimate(
            region=StorageRegion.LEFT_WAIST,
            center=offset(left_hip, along_right=-lateral * 0.35),
            radius=waist_radius,
            confidence=anchor_confidence,
        ),
        StorageRegionEstimate(
            region=StorageRegion.RIGHT_WAIST,
            center=offset(right_hip, along_right=lateral * 0.35),
            radius=waist_radius,
            confidence=anchor_confidence,
        ),
        # Front waistband / front pockets.
        StorageRegionEstimate(
            region=StorageRegion.FRONT_WAIST,
            center=hip_center,
            radius=waist_radius * 1.1,
            confidence=anchor_confidence,
        ),
        # Torso / under-shirt, midway between shoulders and hips.
        StorageRegionEstimate(
            region=StorageRegion.TORSO,
            center=Point(
                (shoulder_center.x + hip_center.x) * 0.5, (shoulder_center.y + hip_center.y) * 0.5
            ),
            radius=torso_radius,
            confidence=anchor_confidence,
        ),
        # Inner jacket, above the torso midpoint.
        StorageRegionEstimate(
            region=StorageRegion.JACKET_AREA,
            center=offset(
                Point(
                    (shoulder_center.x + hip_center.x) * 0.5,
                    (shoulder_center.y + hip_center.y) * 0.5,
                ),
                along_down=-body_height * config.jacket_offset_ratio,
            ),
            radius=torso_radius,
            confidence=anchor_confidence * 0.8,
        ),
    ]

    # Bag / backpack regions are only emitted when an actual bag was detected.
    # Inventing a "bag area" for every shopper would manufacture evidence.
    for box in bag_boxes or []:
        centre = box.center
        containment = track.bbox.expanded(0.4).containment_of(box)
        if containment < 0.3:
            continue
        region = (
            StorageRegion.BACKPACK_AREA
            if centre.distance_to(shoulder_center) < centre.distance_to(hip_center)
            else StorageRegion.BAG_AREA
        )
        regions.append(
            StorageRegionEstimate(
                region=region,
                center=centre,
                radius=max(box.width, box.height) * 0.6,
                confidence=anchor_confidence,
            )
        )

    return regions


def nearest_storage_region(
    point: Point, regions: list[StorageRegionEstimate]
) -> tuple[StorageRegionEstimate, float] | None:
    """Return the closest region and the distance in radii, or ``None``."""
    if not regions:
        return None
    best: tuple[StorageRegionEstimate, float] | None = None
    for region in regions:
        distance = region.normalized_distance(point)
        if best is None or distance < best[1]:
            best = (region, distance)
    return best


# ---------------------------------------------------------------------------
# Motion analysis
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MotionProfile:
    """Summary of a wrist's recent movement, in body-relative units."""

    #: Speed in body heights per second.
    speed_ratio: float
    #: Velocity direction as a unit vector in image coordinates.
    direction: tuple[float, float]
    #: Component of travel along the body's "down" axis, in body heights.
    downward_travel: float
    #: Component of travel toward the body midline, in body heights.
    inward_travel: float
    #: Net displacement over the window, in body heights.
    displacement_ratio: float
    #: Seconds actually covered by the samples used.
    duration: float
    #: Number of trajectory samples the profile was computed from.
    samples: int

    @property
    def is_concealment_like(self) -> bool:
        """Downward and/or inward movement, as opposed to outward/upward.

        Deliberately permissive: this is one input to the engine, never a
        decision on its own.
        """
        return self.downward_travel > 0 or self.inward_travel > 0


def motion_profile(
    trajectory: list[TrajectoryPoint],
    reference_length: float,
    down_axis: np.ndarray | None = None,
    body_center: Point | None = None,
) -> MotionProfile:
    """Summarize a sequence of timestamped positions.

    Velocity is estimated from the endpoints of the window rather than by
    differencing adjacent samples, which is far less sensitive to per-frame
    keypoint jitter.
    """
    empty = MotionProfile(0.0, (0.0, 0.0), 0.0, 0.0, 0.0, 0.0, len(trajectory))
    if len(trajectory) < 2 or reference_length <= EPSILON:
        return empty

    first, last = trajectory[0], trajectory[-1]
    duration = last.timestamp - first.timestamp
    if duration <= EPSILON:
        return empty

    dx = last.position.x - first.position.x
    dy = last.position.y - first.position.y
    displacement = float(np.hypot(dx, dy))
    if displacement < EPSILON:
        return MotionProfile(0.0, (0.0, 0.0), 0.0, 0.0, 0.0, duration, len(trajectory))

    direction = (dx / displacement, dy / displacement)
    displacement_ratio = displacement / reference_length
    speed_ratio = displacement_ratio / duration

    axis = down_axis if down_axis is not None else np.array([0.0, 1.0])
    downward = float(np.dot(np.array([dx, dy]), axis)) / reference_length

    inward = 0.0
    if body_center is not None:
        to_center_start = np.array(
            [body_center.x - first.position.x, body_center.y - first.position.y]
        )
        norm = np.linalg.norm(to_center_start)
        if norm > EPSILON:
            inward = float(np.dot(np.array([dx, dy]), to_center_start / norm)) / reference_length

    return MotionProfile(
        speed_ratio=speed_ratio,
        direction=direction,
        downward_travel=downward,
        inward_travel=inward,
        displacement_ratio=displacement_ratio,
        duration=duration,
        samples=len(trajectory),
    )


def trajectory_similarity(
    a: list[TrajectoryPoint],
    b: list[TrajectoryPoint],
    reference_length: float,
    tolerance_seconds: float = 0.15,
) -> float:
    """Correlation between two trajectories, in ``[0, 1]``.

    Used to decide whether an item is *moving with* a wrist rather than merely
    being near it -- a box sitting on a shelf that a hand happens to pass in
    front of scores low here, which is the whole point.

    Samples are paired by nearest timestamp within ``tolerance_seconds``;
    per-frame displacement vectors are then compared by cosine similarity,
    weighted by magnitude so that stationary noise does not dominate.
    """
    if len(a) < 2 or len(b) < 2 or reference_length <= EPSILON:
        return 0.0

    b_times = np.array([p.timestamp for p in b])
    pairs: list[tuple[TrajectoryPoint, TrajectoryPoint]] = []
    for point in a:
        index = int(np.argmin(np.abs(b_times - point.timestamp)))
        if abs(b_times[index] - point.timestamp) <= tolerance_seconds:
            pairs.append((point, b[index]))
    if len(pairs) < 2:
        return 0.0

    weighted_sum = 0.0
    weight_total = 0.0
    for (a0, b0), (a1, b1) in zip(pairs[:-1], pairs[1:], strict=True):
        va = np.array([a1.position.x - a0.position.x, a1.position.y - a0.position.y])
        vb = np.array([b1.position.x - b0.position.x, b1.position.y - b0.position.y])
        mag_a = float(np.linalg.norm(va))
        mag_b = float(np.linalg.norm(vb))
        motion_scale = max(mag_a, mag_b) / reference_length
        if motion_scale < 0.01:
            # Both effectively stationary: uninformative, skip rather than
            # scoring it as perfect agreement.
            continue
        cosine = (
            float(np.dot(va, vb) / (mag_a * mag_b)) if mag_a > EPSILON and mag_b > EPSILON else 0.0
        )
        # Penalize differing speeds as well as differing directions.
        speed_agreement = (
            min(mag_a, mag_b) / max(mag_a, mag_b) if max(mag_a, mag_b) > EPSILON else 0.0
        )
        score = max(0.0, cosine) * (0.5 + 0.5 * speed_agreement)
        weighted_sum += score * motion_scale
        weight_total += motion_scale

    if weight_total <= EPSILON:
        return 0.0
    return float(np.clip(weighted_sum / weight_total, 0.0, 1.0))


def hand_box(wrist: Point, reference_length: float, size_ratio: float = 0.12) -> BoundingBox:
    """A square box centred on the wrist, sized relative to the body.

    Gives the hand/item associator an IoU channel; a bare keypoint has no
    extent to intersect with.
    """
    half = max(2.0, reference_length * size_ratio * 0.5)
    return BoundingBox(wrist.x - half, wrist.y - half, wrist.x + half, wrist.y + half)


def wrist_points(pose: PoseObservation, min_confidence: float) -> dict[Hand, Point]:
    """Reliable wrist positions, keyed by hand."""
    result: dict[Hand, Point] = {}
    for hand in Hand:
        point = pose.wrist(hand, min_confidence)
        if point is not None:
            result[hand] = point
    return result
