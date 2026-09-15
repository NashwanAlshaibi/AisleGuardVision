"""Geometry: polygons, storage regions and motion analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aisleguardvision.behavior.geometry import (
    body_axes,
    box_polygon_overlap_ratio,
    distance_point_to_polygon,
    distance_points_to_polygon,
    estimate_storage_regions,
    hand_box,
    motion_profile,
    nearest_storage_region,
    normalized_distance,
    point_in_polygon,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
    trajectory_similarity,
    wrist_points,
)
from aisleguardvision.core.config import StorageRegionConfig
from aisleguardvision.core.types import (
    BoundingBox,
    Hand,
    PersonTrack,
    Point,
    PoseObservation,
    StorageRegion,
    TrajectoryPoint,
)
from aisleguardvision.simulation.scenarios import make_person

KEYPOINT_CONFIDENCE = 0.3


# ---------------------------------------------------------------------------
# Polygons
# ---------------------------------------------------------------------------


def test_point_in_polygon_basics(square_polygon):
    assert point_in_polygon(Point(50, 50), square_polygon)
    assert not point_in_polygon(Point(150, 50), square_polygon)
    assert not point_in_polygon(Point(50, -10), square_polygon)


def test_points_in_polygon_is_vectorized(square_polygon):
    points = np.array([[50, 50], [150, 50], [10, 90], [-5, 50], [99, 99]])
    result = points_in_polygon(points, square_polygon)
    assert result.tolist() == [True, False, True, False, True]
    assert result.dtype == bool


def test_points_in_polygon_handles_empty_input(square_polygon):
    assert points_in_polygon(np.zeros((0, 2)), square_polygon).shape == (0,)


def test_concave_polygon_is_handled():
    """An L-shaped shelf face is a real configuration; a convex-only test would
    wrongly accept points in the notch."""
    l_shape = np.array(
        [[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]], dtype=np.float32
    )
    assert point_in_polygon(Point(20, 20), l_shape)
    assert point_in_polygon(Point(20, 80), l_shape)
    assert not point_in_polygon(Point(80, 80), l_shape), "the notch must be outside"


def test_polygon_area_and_centroid(square_polygon):
    assert polygon_area(square_polygon) == pytest.approx(10000.0)
    centroid = polygon_centroid(square_polygon)
    assert centroid.x == pytest.approx(50.0)
    assert centroid.y == pytest.approx(50.0)


def test_polygon_area_is_winding_independent(square_polygon):
    reversed_polygon = square_polygon[::-1].copy()
    assert polygon_area(reversed_polygon) == pytest.approx(polygon_area(square_polygon))


def test_distance_to_polygon_is_zero_inside(square_polygon):
    assert distance_point_to_polygon(Point(50, 50), square_polygon) == 0.0


def test_distance_to_polygon_outside(square_polygon):
    assert distance_point_to_polygon(Point(130, 50), square_polygon) == pytest.approx(30.0)
    # Diagonal from a corner.
    assert distance_point_to_polygon(Point(-3, -4), square_polygon) == pytest.approx(5.0)


def test_distance_points_to_polygon_vectorized(square_polygon):
    distances = distance_points_to_polygon(
        np.array([[50, 50], [130, 50], [50, 120]]), square_polygon
    )
    assert distances.tolist() == pytest.approx([0.0, 30.0, 20.0])


def test_box_polygon_overlap_ratio(square_polygon):
    inside = BoundingBox(10, 10, 90, 90)
    assert box_polygon_overlap_ratio(inside, square_polygon) == pytest.approx(1.0)

    outside = BoundingBox(200, 200, 280, 280)
    assert box_polygon_overlap_ratio(outside, square_polygon) == 0.0

    half = BoundingBox(50, 0, 150, 100)
    assert 0.3 < box_polygon_overlap_ratio(half, square_polygon) < 0.7


def test_normalized_distance_scales_with_body_size():
    a, b = Point(0, 0), Point(0, 100)
    assert normalized_distance(a, b, 500.0) == pytest.approx(0.2)
    # The same pixel gap is twice as significant for a subject half the size.
    assert normalized_distance(a, b, 250.0) == pytest.approx(0.4)
    assert normalized_distance(a, b, 0.0) == float("inf")


# ---------------------------------------------------------------------------
# Body geometry
# ---------------------------------------------------------------------------


def make_track(right_wrist: Point, left_wrist: Point | None = None) -> PersonTrack:
    person = make_person(right_wrist, left_wrist)
    track = PersonTrack(
        track_id=1,
        camera_id="test",
        bbox=person.bbox,
        confidence=person.confidence,
        first_seen=0.0,
        last_seen=1.0,
    )
    track.pose = PoseObservation.from_array(person.keypoints, person.bbox, 1.0)
    track.pose_timestamp = 1.0
    return track


def test_body_axes_point_down_and_right():
    track = make_track(Point(600, 400))
    axes = body_axes(track.pose, KEYPOINT_CONFIDENCE)
    assert axes is not None
    down, right = axes

    assert down[1] > 0.9, "the down axis should point down the image for an upright subject"
    assert abs(np.linalg.norm(down) - 1.0) < 1e-6
    assert abs(np.linalg.norm(right) - 1.0) < 1e-6
    # The subject faces the camera, so their right side is on the image-left.
    assert right[0] < 0


def test_body_axes_returns_none_without_keypoints():
    pose = PoseObservation(keypoints={}, confidence=0.0)
    assert body_axes(pose, KEYPOINT_CONFIDENCE) is None


def test_body_height_uses_torso_span_when_pose_is_available():
    track = make_track(Point(600, 400))
    # The simulated body model places shoulders and hips 0.3 body heights apart.
    assert track.body_height == pytest.approx(520.0, rel=0.02)


def test_storage_regions_are_estimated_from_pose():
    track = make_track(Point(600, 400))
    regions = estimate_storage_regions(track, StorageRegionConfig(), KEYPOINT_CONFIDENCE)
    found = {r.region for r in regions}

    assert StorageRegion.LEFT_WAIST in found
    assert StorageRegion.RIGHT_WAIST in found
    assert StorageRegion.FRONT_WAIST in found
    assert StorageRegion.TORSO in found
    # No bag was detected, so no bag region may be invented.
    assert StorageRegion.BAG_AREA not in found
    assert StorageRegion.BACKPACK_AREA not in found


def test_storage_region_radii_scale_with_body_size():
    """A fixed pixel radius would make the system behave completely differently
    for a shopper near the camera and one at the end of the aisle."""
    config = StorageRegionConfig()
    track = make_track(Point(600, 400))
    regions = estimate_storage_regions(track, config, KEYPOINT_CONFIDENCE)
    waist = next(r for r in regions if r.region is StorageRegion.RIGHT_WAIST)
    assert waist.radius == pytest.approx(track.body_height * config.waist_radius_ratio, rel=0.05)


def test_storage_regions_require_pose():
    track = PersonTrack(
        track_id=1,
        camera_id="test",
        bbox=BoundingBox(0, 0, 100, 300),
        confidence=0.9,
        first_seen=0.0,
        last_seen=1.0,
    )
    assert estimate_storage_regions(track, StorageRegionConfig(), KEYPOINT_CONFIDENCE) == []


def test_bag_region_only_appears_when_a_bag_is_detected():
    track = make_track(Point(600, 400))
    bag = BoundingBox(660, 330, 760, 430)  # overlapping the subject
    regions = estimate_storage_regions(track, StorageRegionConfig(), KEYPOINT_CONFIDENCE, [bag])
    assert {StorageRegion.BAG_AREA, StorageRegion.BACKPACK_AREA} & {r.region for r in regions}


def test_nearest_storage_region_uses_normalized_distance():
    track = make_track(Point(600, 400))
    regions = estimate_storage_regions(track, StorageRegionConfig(), KEYPOINT_CONFIDENCE)
    waist = next(r for r in regions if r.region is StorageRegion.RIGHT_WAIST)

    nearest = nearest_storage_region(waist.center, regions)
    assert nearest is not None
    assert nearest[1] == pytest.approx(0.0, abs=1e-6), "the region centre is zero radii away"

    far = nearest_storage_region(Point(5000, 5000), regions)
    assert far is not None and far[1] > 10


def test_nearest_storage_region_with_no_regions():
    assert nearest_storage_region(Point(0, 0), []) is None


def test_hand_box_scales_with_body():
    box = hand_box(Point(100, 100), reference_length=500.0, size_ratio=0.12)
    assert box.width == pytest.approx(60.0)
    assert box.center.x == pytest.approx(100.0)
    # Never degenerate, even for a tiny subject far from the camera.
    assert hand_box(Point(0, 0), reference_length=1.0).width >= 4.0


def test_wrist_points_respects_confidence():
    track = make_track(Point(600, 400))
    assert set(wrist_points(track.pose, 0.3)) == {Hand.LEFT, Hand.RIGHT}
    assert wrist_points(track.pose, 0.99) == {}


# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------


def path(points: list[tuple[float, float, float]]) -> list[TrajectoryPoint]:
    return [TrajectoryPoint(t, Point(x, y)) for t, x, y in points]


def test_motion_profile_measures_speed_in_body_heights():
    trajectory = path([(0.0, 0.0, 0.0), (1.0, 0.0, 100.0)])
    profile = motion_profile(trajectory, reference_length=200.0, down_axis=np.array([0.0, 1.0]))

    assert profile.displacement_ratio == pytest.approx(0.5)
    assert profile.speed_ratio == pytest.approx(0.5)
    assert profile.downward_travel == pytest.approx(0.5)
    assert profile.is_concealment_like


def test_motion_profile_detects_upward_movement_as_not_concealment_like():
    trajectory = path([(0.0, 0.0, 100.0), (1.0, 0.0, 0.0)])
    profile = motion_profile(trajectory, reference_length=200.0, down_axis=np.array([0.0, 1.0]))
    assert profile.downward_travel < 0
    assert not profile.is_concealment_like


def test_motion_profile_measures_inward_travel():
    """Movement toward the body midline, which is what concealment looks like."""
    trajectory = path([(0.0, 0.0, 0.0), (1.0, 100.0, 0.0)])
    profile = motion_profile(
        trajectory,
        reference_length=200.0,
        down_axis=np.array([0.0, 1.0]),
        body_center=Point(200.0, 0.0),
    )
    assert profile.inward_travel == pytest.approx(0.5)


def test_motion_profile_handles_degenerate_input():
    assert motion_profile([], 100.0).speed_ratio == 0.0
    assert motion_profile(path([(0.0, 0.0, 0.0)]), 100.0).speed_ratio == 0.0
    # Zero elapsed time must not divide by zero.
    assert motion_profile(path([(1.0, 0, 0), (1.0, 50, 50)]), 100.0).speed_ratio == 0.0
    # Stationary.
    assert motion_profile(path([(0.0, 5, 5), (1.0, 5, 5)]), 100.0).speed_ratio == 0.0


def test_trajectory_similarity_is_high_for_items_moving_with_a_wrist():
    """This is what separates 'holding an item' from 'a hand passing in front
    of one on a shelf'."""
    wrist = path([(0.0, 0, 0), (0.1, 10, 10), (0.2, 20, 20), (0.3, 30, 30)])
    item = path([(0.0, 3, 3), (0.1, 13, 13), (0.2, 23, 23), (0.3, 33, 33)])
    assert trajectory_similarity(wrist, item, reference_length=200.0) > 0.85


def test_trajectory_similarity_is_low_for_a_stationary_item():
    wrist = path([(0.0, 0, 0), (0.1, 20, 0), (0.2, 40, 0), (0.3, 60, 0)])
    shelved = path([(0.0, 100, 0), (0.1, 100, 0), (0.2, 100, 0), (0.3, 100, 0)])
    assert trajectory_similarity(wrist, shelved, reference_length=200.0) < 0.3


def test_trajectory_similarity_is_low_for_opposing_motion():
    wrist = path([(0.0, 0, 0), (0.1, 20, 0), (0.2, 40, 0)])
    opposite = path([(0.0, 100, 0), (0.1, 80, 0), (0.2, 60, 0)])
    assert trajectory_similarity(wrist, opposite, reference_length=200.0) < 0.1


def test_trajectory_similarity_needs_overlapping_timestamps():
    a = path([(0.0, 0, 0), (0.1, 10, 10)])
    b = path([(90.0, 0, 0), (90.1, 10, 10)])
    assert trajectory_similarity(a, b, reference_length=200.0) == 0.0


def test_trajectory_similarity_handles_short_input():
    assert trajectory_similarity([], [], 100.0) == 0.0
    assert trajectory_similarity(path([(0.0, 0, 0)]), path([(0.0, 0, 0)]), 100.0) == 0.0


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


def test_bounding_box_normalizes_inverted_coordinates():
    box = BoundingBox(100, 100, 10, 10)
    assert (box.x1, box.y1, box.x2, box.y2) == (10, 10, 100, 100)


def test_bounding_box_iou_and_containment():
    big = BoundingBox(0, 0, 100, 100)
    small = BoundingBox(40, 40, 60, 60)

    assert big.iou(small) == pytest.approx(400 / 10000)
    # Containment is the right measure when sizes differ wildly: a merchandise
    # item inside a person box has negligible IoU but full containment.
    assert big.containment_of(small) == pytest.approx(1.0)
    assert big.iou(BoundingBox(200, 200, 300, 300)) == 0.0


def test_bounding_box_expanded_and_clipped():
    box = BoundingBox(50, 50, 150, 150)
    expanded = box.expanded(0.2)
    assert expanded.width == pytest.approx(120.0)
    assert expanded.center.x == pytest.approx(box.center.x)

    clipped = BoundingBox(-20, -20, 500, 500).clipped_to(200, 100)
    assert clipped.x1 == 0 and clipped.y1 == 0
    assert clipped.x2 == 199 and clipped.y2 == 99


def test_point_distance():
    assert Point(0, 0).distance_to(Point(3, 4)) == pytest.approx(5.0)
    assert math.isclose(Point(1, 1).distance_to(Point(1, 1)), 0.0)
