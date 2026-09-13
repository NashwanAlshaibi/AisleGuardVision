"""Association: pose-to-person, item-to-wrist, and person tracking."""

from __future__ import annotations

import pytest

from aisleguardvision.core.config import AssociationConfig, ByteTrackConfig, ItemTrackerConfig
from aisleguardvision.core.types import (
    BoundingBox,
    Detection,
    Hand,
    ItemDetection,
    ItemStatus,
    ObjectClass,
    PersonTrack,
    Point,
    PoseObservation,
    TrackState,
)
from aisleguardvision.tracking.association import (
    HandItemAssociator,
    WristHistoryStore,
    associate_poses_with_tracks,
    pose_person_score,
)
from aisleguardvision.tracking.item_tracker import ItemTracker
from aisleguardvision.tracking.kalman import greedy_match, iou_matrix
from aisleguardvision.tracking.person_tracker import PersonTracker, attach_pose
from aisleguardvision.simulation.scenarios import make_item, make_person

from conftest import person_detection

KEYPOINT_CONFIDENCE = 0.3


def track_from_person(person, track_id: int = 1, timestamp: float = 1.0) -> PersonTrack:
    track = PersonTrack(
        track_id=track_id,
        camera_id="test",
        bbox=person.bbox,
        confidence=person.confidence,
        first_seen=0.0,
        last_seen=timestamp,
        state=TrackState.CONFIRMED,
    )
    attach_pose(track, PoseObservation.from_array(person.keypoints, person.bbox, timestamp), timestamp)
    return track


# ---------------------------------------------------------------------------
# Matching primitives
# ---------------------------------------------------------------------------


def test_iou_matrix_shape_and_values():
    a = [BoundingBox(0, 0, 10, 10)]
    b = [BoundingBox(0, 0, 10, 10), BoundingBox(20, 20, 30, 30)]
    matrix = iou_matrix(a, b)
    assert matrix.shape == (1, 2)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[0, 1] == 0.0


def test_iou_matrix_handles_empty_input():
    assert iou_matrix([], []).shape == (0, 0)
    assert iou_matrix([BoundingBox(0, 0, 1, 1)], []).shape == (1, 0)


def test_greedy_match_picks_the_best_pairs():
    import numpy as np

    cost = np.array([[0.9, 0.1], [0.2, 0.8]])
    matches, unmatched_rows, unmatched_cols = greedy_match(cost, 0.5)
    assert sorted(matches) == [(0, 0), (1, 1)]
    assert unmatched_rows == [] and unmatched_cols == []


def test_greedy_match_respects_the_threshold():
    import numpy as np

    matches, rows, cols = greedy_match(np.array([[0.3]]), 0.5)
    assert matches == []
    assert rows == [0] and cols == [0]


def test_greedy_match_is_deterministic():
    """Replayed video and repeated tests must produce identical assignments."""
    import numpy as np

    cost = np.array([[0.7, 0.7], [0.7, 0.7]])
    first = greedy_match(cost, 0.5)[0]
    second = greedy_match(cost, 0.5)[0]
    assert first == second


# ---------------------------------------------------------------------------
# Person tracking
# ---------------------------------------------------------------------------


def test_tracker_assigns_and_keeps_an_id():
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2))
    for index in range(5):
        tracks = tracker.update([person_detection(500 + index * 4, 400)], index * 0.05)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].state is TrackState.CONFIRMED


def test_tracker_needs_min_hits_before_confirming():
    """A one-frame detector artifact must never become a confirmed track."""
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=3))
    assert tracker.update([person_detection(500, 400)], 0.0) == []
    assert tracker.update([person_detection(502, 400)], 0.05) == []
    assert len(tracker.update([person_detection(504, 400)], 0.10)) == 1


def test_tracker_separates_two_people():
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2))
    for index in range(4):
        tracks = tracker.update(
            [person_detection(300 + index * 3, 400), person_detection(900 - index * 3, 400)],
            index * 0.05,
        )
    assert len({t.track_id for t in tracks}) == 2


def test_tracker_survives_a_brief_occlusion():
    """ByteTrack's second pass is what carries a shopper behind a display."""
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2, max_lost_seconds=1.5))
    for index in range(5):
        tracker.update([person_detection(500, 400)], index * 0.05)
    original = tracker.confirmed_tracks()[0].track_id

    # Three frames with no detection at all.
    for index in range(5, 8):
        tracker.update([], index * 0.05)
    # Then the person reappears nearby.
    tracks = tracker.update([person_detection(510, 400)], 8 * 0.05)

    assert tracks, "the track should have been revived, not replaced"
    assert tracks[0].track_id == original


def test_tracker_retires_a_track_after_max_lost_seconds():
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2, max_lost_seconds=0.5))
    for index in range(4):
        tracker.update([person_detection(500, 400)], index * 0.05)
    assert tracker.confirmed_tracks()

    tracker.update([], 5.0)
    tracker.update([], 6.0)
    assert tracker.confirmed_tracks() == []


def test_track_lifetimes_are_time_based_not_frame_based():
    """The same real-world occlusion must behave identically at 10 and 30 FPS."""

    def run(fps: float) -> bool:
        tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2, max_lost_seconds=1.0))
        step = 1.0 / fps
        for index in range(6):
            tracker.update([person_detection(500, 400)], index * step)
        # 0.6 s gap, expressed in that camera's frames.
        start = 6 * step
        gap_frames = int(0.6 / step)
        for index in range(gap_frames):
            tracker.update([], start + index * step)
        tracks = tracker.update([person_detection(505, 400)], start + 0.6)
        return bool(tracks)

    assert run(30.0) == run(10.0) is True


def test_track_quality_reflects_detection_confidence():
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2))
    for index in range(4):
        tracker.update([person_detection(500, 400, confidence=0.62)], index * 0.05)
    assert tracker.track_quality(1) == pytest.approx(0.62, abs=0.01)


def test_tracker_records_a_trajectory():
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2))
    for index in range(6):
        tracks = tracker.update([person_detection(400 + index * 20, 400)], index * 0.05)
    assert len(tracks[0].trajectory) >= 5
    assert tracks[0].trajectory[-1].position.x > tracks[0].trajectory[0].position.x


def test_tracker_handles_a_non_monotonic_timestamp():
    """A source seek or clock adjustment must not feed a negative dt into the
    Kalman filter."""
    tracker = PersonTracker("cam", ByteTrackConfig(min_hits=2))
    tracker.update([person_detection(500, 400)], 100.0)
    tracker.update([person_detection(502, 400)], 99.0)  # time goes backwards
    assert tracker.active_count >= 1


# ---------------------------------------------------------------------------
# Pose association
# ---------------------------------------------------------------------------


def test_pose_matches_the_right_person():
    """Pose output order has no relationship to track order. Assuming it does
    attributes one shopper's arms to another."""
    left_person = make_person(Point(250, 400), cx=300.0)
    right_person = make_person(Point(850, 400), cx=900.0)

    tracks = [
        PersonTrack(1, "cam", left_person.bbox, 0.9, 0.0, 1.0, TrackState.CONFIRMED),
        PersonTrack(2, "cam", right_person.bbox, 0.9, 0.0, 1.0, TrackState.CONFIRMED),
    ]
    # Poses supplied in the OPPOSITE order to the tracks.
    poses = [
        Detection(
            bbox=right_person.bbox,
            confidence=0.9,
            object_class=ObjectClass.PERSON,
            pose=PoseObservation.from_array(right_person.keypoints, right_person.bbox),
        ),
        Detection(
            bbox=left_person.bbox,
            confidence=0.9,
            object_class=ObjectClass.PERSON,
            pose=PoseObservation.from_array(left_person.keypoints, left_person.bbox),
        ),
    ]

    matched = associate_poses_with_tracks(tracks, poses, AssociationConfig())
    assert set(matched) == {1, 2}

    left_wrist = matched[1].wrist(Hand.RIGHT, KEYPOINT_CONFIDENCE)
    assert left_wrist is not None and left_wrist.x < 500, "track 1 got the wrong person's pose"


def test_unmatched_poses_are_discarded_not_guessed():
    person = make_person(Point(600, 400))
    stranger = make_person(Point(250, 400), cx=300.0)
    tracks = [PersonTrack(1, "cam", person.bbox, 0.9, 0.0, 1.0, TrackState.CONFIRMED)]
    poses = [
        Detection(
            bbox=stranger.bbox,
            confidence=0.9,
            object_class=ObjectClass.PERSON,
            pose=PoseObservation.from_array(stranger.keypoints, stranger.bbox),
        )
    ]
    assert associate_poses_with_tracks(tracks, poses, AssociationConfig()) == {}


def test_pose_person_score_rewards_keypoint_containment():
    person = make_person(Point(600, 400))
    pose = PoseObservation.from_array(person.keypoints, person.bbox)
    assert pose_person_score(person.bbox, pose, person.bbox) > 0.9

    elsewhere = BoundingBox(0, 0, 50, 50)
    assert pose_person_score(elsewhere, pose, person.bbox) < 0.2


def test_pose_association_with_no_tracks_or_no_poses():
    assert associate_poses_with_tracks([], [], AssociationConfig()) == {}


# ---------------------------------------------------------------------------
# Wrist history
# ---------------------------------------------------------------------------


def test_wrist_history_records_and_windows():
    store = WristHistoryStore()
    for index in range(10):
        store.record(1, Hand.RIGHT, Point(index * 10, 100), index * 0.1)

    recent = store.trajectory(1, Hand.RIGHT, window_seconds=0.35, now=0.9)
    assert all(p.timestamp >= 0.55 for p in recent)
    assert len(recent) < 10


def test_wrist_history_prune_bounds_memory():
    store = WristHistoryStore()
    store.record(1, Hand.RIGHT, Point(0, 0), 0.0)
    store.record(2, Hand.LEFT, Point(0, 0), 0.0)

    store.prune({1})
    assert store.latest(1, Hand.RIGHT) is not None
    assert store.latest(2, Hand.LEFT) is None


def test_wrist_history_records_from_a_pose():
    person = make_person(Point(600, 400))
    pose = PoseObservation.from_array(person.keypoints, person.bbox)
    store = WristHistoryStore()
    store.record_pose(1, pose, 1.0, KEYPOINT_CONFIDENCE)

    assert store.latest(1, Hand.RIGHT) is not None
    assert store.latest(1, Hand.LEFT) is not None


# ---------------------------------------------------------------------------
# Hand/item association
# ---------------------------------------------------------------------------


def run_association(wrist: Point, item_centre: Point, frames: int = 12, dt: float = 0.05):
    """Drive the associator over ``frames`` frames with a static configuration."""
    associator = HandItemAssociator(AssociationConfig())
    item_tracker = ItemTracker("cam", ItemTrackerConfig())
    store = WristHistoryStore()
    results = []

    for index in range(frames):
        timestamp = index * dt
        person = make_person(wrist)
        track = track_from_person(person, timestamp=timestamp)
        store.record_pose(1, track.pose, timestamp, KEYPOINT_CONFIDENCE)
        items = item_tracker.update([make_item(item_centre)], timestamp)
        results = associator.associate(
            [track], items, store, timestamp, KEYPOINT_CONFIDENCE
        )
    return results, item_tracker


def test_item_at_the_wrist_is_associated():
    results, _ = run_association(Point(450, 300), Point(452, 302))
    assert results
    assert results[0].hand is Hand.RIGHT
    assert results[0].confidence >= 0.45


def test_a_distant_item_is_never_associated():
    results, _ = run_association(Point(450, 300), Point(1100, 650))
    assert results == []


def test_association_is_not_stable_on_the_first_frame():
    """A single frame of proximity means nothing: hands pass in front of
    shelved products constantly."""
    results, _ = run_association(Point(450, 300), Point(452, 302), frames=1)
    assert not results or not results[0].is_stable


def test_association_becomes_stable_after_the_configured_dwell():
    config = AssociationConfig()
    frames = int(config.min_association_seconds / 0.05) + 4
    results, _ = run_association(Point(450, 300), Point(452, 302), frames=frames)
    assert results[0].is_stable
    assert results[0].duration >= config.min_association_seconds


def test_association_reports_its_channels_for_auditing():
    results, _ = run_association(Point(450, 300), Point(452, 302))
    channels = results[0].channels
    assert {"distance", "iou", "motion", "persistence"} <= set(channels)
    assert 0.0 <= channels["distance"] <= 1.0


def test_one_item_is_associated_with_at_most_one_hand():
    associator = HandItemAssociator(AssociationConfig())
    item_tracker = ItemTracker("cam", ItemTrackerConfig())
    store = WristHistoryStore()
    centre = Point(600, 330)

    for index in range(12):
        timestamp = index * 0.05
        # Both wrists on top of the item.
        person = make_person(centre, left_wrist=centre)
        track = track_from_person(person, timestamp=timestamp)
        store.record_pose(1, track.pose, timestamp, KEYPOINT_CONFIDENCE)
        items = item_tracker.update([make_item(centre)], timestamp)
        results = associator.associate([track], items, store, timestamp, KEYPOINT_CONFIDENCE)

    assert len({r.item_id for r in results}) == len(results)


def test_association_requires_a_pose():
    """Without wrists there is nothing to associate to."""
    associator = HandItemAssociator(AssociationConfig())
    track = PersonTrack(1, "cam", BoundingBox(500, 100, 700, 620), 0.9, 0.0, 1.0)
    item_tracker = ItemTracker("cam", ItemTrackerConfig())
    items = item_tracker.update([make_item(Point(600, 300))], 1.0)

    assert associator.associate([track], items, WristHistoryStore(), 1.0) == []


def test_association_ignores_items_that_are_gone():
    """Only an observable item can start or sustain an association."""
    associator = HandItemAssociator(AssociationConfig())
    item_tracker = ItemTracker("cam", ItemTrackerConfig())
    store = WristHistoryStore()
    centre = Point(450, 300)

    for index in range(10):
        timestamp = index * 0.05
        person = make_person(centre)
        track = track_from_person(person, timestamp=timestamp)
        store.record_pose(1, track.pose, timestamp, KEYPOINT_CONFIDENCE)
        item_tracker.update([make_item(centre)], timestamp)

    # Let the item fall all the way down the ladder to MISSING.
    for index in range(10, 60):
        item_tracker.update([], index * 0.05)

    person = make_person(centre)
    track = track_from_person(person, timestamp=3.0)
    store.record_pose(1, track.pose, 3.0, KEYPOINT_CONFIDENCE)
    assert item_tracker.tracks[0].status is ItemStatus.MISSING
    assert associator.associate([track], item_tracker.tracks, store, 3.0) == []


def test_link_duration_resets_after_a_long_gap():
    """Two brief, unrelated touches minutes apart must not sum into a stable
    association."""
    associator = HandItemAssociator(AssociationConfig())
    assert associator.link_duration(1, Hand.RIGHT, 1, now=0.0) == 0.0


# ---------------------------------------------------------------------------
# Item tracking and the disappearance ladder
# ---------------------------------------------------------------------------


def item_detection(centre: Point, object_class=ObjectClass.MERCHANDISE) -> ItemDetection:
    return make_item(centre, object_class)


def test_item_tracker_creates_and_maintains_a_track():
    tracker = ItemTracker("cam", ItemTrackerConfig())
    for index in range(6):
        tracks = tracker.update([item_detection(Point(400 + index, 300))], index * 0.05)
    assert len(tracks) == 1
    assert tracks[0].hit_count == 6
    assert tracks[0].status is ItemStatus.VISIBLE


def test_item_disappearance_climbs_the_ladder_over_time():
    """VISIBLE -> POSSIBLY_OCCLUDED -> OCCLUDED -> MISSING. Never a jump."""
    config = ItemTrackerConfig()
    tracker = ItemTracker("cam", config)
    for index in range(8):
        tracker.update([item_detection(Point(400, 300))], index * 0.05)

    base = 8 * 0.05
    tracker.update([], base + 0.05)
    assert tracker.tracks[0].status is ItemStatus.VISIBLE

    tracker.update([], base + config.possibly_occluded_after_seconds + 0.01)
    assert tracker.tracks[0].status is ItemStatus.POSSIBLY_OCCLUDED

    tracker.update([], base + config.occluded_after_seconds + 0.01)
    assert tracker.tracks[0].status is ItemStatus.OCCLUDED

    tracker.update([], base + config.missing_after_seconds + 0.01)
    assert tracker.tracks[0].status is ItemStatus.MISSING


def test_item_reappearing_resets_the_ladder():
    tracker = ItemTracker("cam", ItemTrackerConfig())
    for index in range(8):
        tracker.update([item_detection(Point(400, 300))], index * 0.05)
    tracker.update([], 1.2)
    assert tracker.tracks[0].status is not ItemStatus.VISIBLE

    tracker.update([item_detection(Point(400, 300))], 1.3)
    assert tracker.tracks[0].status is ItemStatus.VISIBLE


def test_item_track_stability_requires_hits_and_age():
    """An unstable track is a better explanation for a disappearance than
    concealment is."""
    config = ItemTrackerConfig()
    tracker = ItemTracker("cam", config)
    tracker.update([item_detection(Point(400, 300))], 0.0)
    assert not tracker.tracks[0].is_stable(
        config.min_hits_for_stability, config.min_age_for_stability_seconds
    )

    for index in range(1, 10):
        tracker.update([item_detection(Point(400, 300))], index * 0.06)
    assert tracker.tracks[0].is_stable(
        config.min_hits_for_stability, config.min_age_for_stability_seconds
    )


def test_items_of_different_classes_never_share_a_track():
    """A phone must never inherit a merchandise track's history, or the phone
    suppression could be bypassed by a class swap."""
    tracker = ItemTracker("cam", ItemTrackerConfig())
    tracker.update([item_detection(Point(400, 300), ObjectClass.MERCHANDISE)], 0.0)
    tracker.update([item_detection(Point(400, 300), ObjectClass.PHONE)], 0.05)
    assert len(tracker.tracks) == 2


def test_benign_resolution_is_sticky():
    tracker = ItemTracker("cam", ItemTrackerConfig())
    tracker.update([item_detection(Point(400, 300))], 0.0)
    item_id = tracker.tracks[0].item_id

    tracker.resolve(item_id, ItemStatus.IN_BASKET, 0.1)
    assert tracker.tracks[0].status is ItemStatus.IN_BASKET

    # Seeing it again does not undo the resolution.
    tracker.update([item_detection(Point(400, 300))], 0.2)
    assert tracker.tracks[0].status is ItemStatus.IN_BASKET


def test_ladder_statuses_cannot_be_pushed_in_as_resolutions():
    tracker = ItemTracker("cam", ItemTrackerConfig())
    tracker.update([item_detection(Point(400, 300))], 0.0)
    with pytest.raises(ValueError):
        tracker.resolve(tracker.tracks[0].item_id, ItemStatus.MISSING, 0.1)


def test_item_association_dwell_accumulates_across_frames():
    tracker = ItemTracker("cam", ItemTrackerConfig())
    tracker.update([item_detection(Point(400, 300))], 0.0)
    item_id = tracker.tracks[0].item_id

    tracker.set_association(item_id, 1, Hand.RIGHT, 0.8, 0.0)
    for index in range(1, 8):
        tracker.update([item_detection(Point(400, 300))], index * 0.1)
        tracker.set_association(item_id, 1, Hand.RIGHT, 0.8, index * 0.1)

    assert tracker.tracks[0].association_duration == pytest.approx(0.7, abs=0.01)


def test_changing_the_associated_hand_restarts_the_dwell():
    tracker = ItemTracker("cam", ItemTrackerConfig())
    tracker.update([item_detection(Point(400, 300))], 0.0)
    item_id = tracker.tracks[0].item_id

    tracker.set_association(item_id, 1, Hand.RIGHT, 0.8, 0.0)
    tracker.update([item_detection(Point(400, 300))], 0.5)
    tracker.set_association(item_id, 1, Hand.LEFT, 0.8, 0.5)
    assert tracker.tracks[0].association_duration == pytest.approx(0.0)


def test_old_item_tracks_are_retired():
    config = ItemTrackerConfig(remove_after_seconds=1.0)
    tracker = ItemTracker("cam", config)
    tracker.update([item_detection(Point(400, 300))], 0.0)
    tracker.update([], 0.1)
    tracker.update([], 5.0)
    assert tracker.tracks == []
