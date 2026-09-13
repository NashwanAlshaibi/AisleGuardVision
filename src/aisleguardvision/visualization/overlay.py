"""OpenCV visualization.

Kept strictly separate from analysis: nothing in this module influences a
decision, and the entire pipeline runs with it never imported (``--headless``).
That separation is what lets a production edge node avoid paying for drawing it
will never display.

What is drawn:

* person boxes with temporary track ids
* pose skeletons and wrists
* configured shelf, high-value, checkout, basket, cart and exclusion zones
* item tracks and their observation status
* estimated storage regions
* per-person behavior state, risk score and threat level
* camera / detection / pose FPS, inference latency, queue depth, drops
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..core.types import (
    BehaviorObservation,
    BehaviorState,
    BoundingBox,
    COCO_SKELETON,
    Hand,
    ItemStatus,
    ItemTrack,
    KeypointName,
    PersonTrack,
    Point,
    ShelfZone,
    StorageRegionEstimate,
    ThreatLevel,
    ZoneKind,
)

# BGR palette.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
GREEN = (80, 220, 100)
YELLOW = (60, 200, 250)
ORANGE = (40, 140, 255)
RED = (60, 60, 245)
BLUE = (235, 180, 70)
PURPLE = (220, 120, 190)
CYAN = (230, 230, 90)

THREAT_COLORS: dict[ThreatLevel, tuple[int, int, int]] = {
    ThreatLevel.LOW: GREEN,
    ThreatLevel.ELEVATED: YELLOW,
    ThreatLevel.REVIEW: ORANGE,
    ThreatLevel.HIGH_RISK: RED,
}

ZONE_COLORS: dict[ZoneKind, tuple[int, int, int]] = {
    ZoneKind.SHELF: BLUE,
    ZoneKind.HIGH_VALUE: PURPLE,
    ZoneKind.CHECKOUT: CYAN,
    ZoneKind.BASKET: GREEN,
    ZoneKind.CART: GREEN,
    ZoneKind.ENTRANCE: GREY,
    ZoneKind.EXCLUSION: (90, 90, 90),
}

ITEM_STATUS_COLORS: dict[ItemStatus, tuple[int, int, int]] = {
    ItemStatus.VISIBLE: GREEN,
    ItemStatus.POSSIBLY_OCCLUDED: YELLOW,
    ItemStatus.OCCLUDED: ORANGE,
    ItemStatus.MISSING: RED,
    ItemStatus.RETURNED: GREEN,
    ItemStatus.IN_BASKET: GREEN,
    ItemStatus.IN_CART: GREEN,
    ItemStatus.UNKNOWN: GREY,
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass(slots=True)
class OverlayStats:
    """Numbers shown in the statistics panel."""

    camera_id: str = ""
    video_fps: float = 0.0
    detection_fps: float = 0.0
    pose_fps: float = 0.0
    inference_latency_ms: float = 0.0
    pose_latency_ms: float = 0.0
    tracks: int = 0
    item_tracks: int = 0
    queue_depth: int = 0
    dropped: int = 0
    device: str = ""
    alerts: int = 0
    #: Shown when no merchandise detector is available, so the operator is
    #: never left wondering why risk plateaus.
    mode_note: str = ""


@dataclass(slots=True)
class OverlayOptions:
    """What to draw. Everything defaults on for the MVP display."""

    draw_zones: bool = True
    draw_boxes: bool = True
    draw_skeleton: bool = True
    draw_wrists: bool = True
    draw_items: bool = True
    draw_storage_regions: bool = True
    draw_trajectory: bool = True
    draw_stats: bool = True
    draw_evidence: bool = True
    #: Keypoints below this confidence are not drawn, so the display reflects
    #: what the engine actually used.
    keypoint_confidence: float = 0.3
    max_evidence_lines: int = 6


class OverlayRenderer:
    """Draws the analysis state onto a frame."""

    def __init__(self, options: OverlayOptions | None = None) -> None:
        self.options = options or OverlayOptions()

    def render(
        self,
        image: np.ndarray,
        *,
        zones: list[ShelfZone] | None = None,
        tracks: list[PersonTrack] | None = None,
        items: list[ItemTrack] | None = None,
        observations: dict[int, BehaviorObservation] | None = None,
        stats: OverlayStats | None = None,
        copy: bool = False,
    ) -> np.ndarray:
        """Return the annotated frame.

        ``copy=False`` draws in place, which avoids a full-frame allocation per
        displayed frame.
        """
        canvas = image.copy() if copy else image
        options = self.options
        observations = observations or {}

        if options.draw_zones and zones:
            self._draw_zones(canvas, zones)
        if options.draw_items and items:
            for item in items:
                self._draw_item(canvas, item)
        for track in tracks or []:
            observation = observations.get(track.track_id)
            self._draw_person(canvas, track, observation)
        if options.draw_stats and stats is not None:
            self._draw_stats(canvas, stats)
        if options.draw_evidence:
            self._draw_evidence(canvas, observations)
        return canvas

    # -- zones -------------------------------------------------------------
    def _draw_zones(self, image: np.ndarray, zones: list[ShelfZone]) -> None:
        overlay = image.copy()
        for zone in zones:
            if not zone.enabled:
                continue
            color = ZONE_COLORS.get(zone.kind, BLUE)
            polygon = zone.polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [polygon], color)
            cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
            anchor = zone.polygon.astype(int).min(axis=0)
            _label(
                image,
                f"{zone.kind.value}: {zone.name or zone.zone_id}",
                (int(anchor[0]) + 4, int(anchor[1]) + 18),
                color,
            )
        # One blend for all zones rather than one per zone.
        cv2.addWeighted(overlay, 0.15, image, 0.85, 0, image)

    # -- people ------------------------------------------------------------
    def _draw_person(
        self, image: np.ndarray, track: PersonTrack, observation: BehaviorObservation | None
    ) -> None:
        options = self.options
        threat = observation.risk.threat_level if observation else ThreatLevel.LOW
        color = THREAT_COLORS.get(threat, GREEN)
        state = observation.state if observation else BehaviorState.IDLE
        risk = observation.risk.risk_score if observation else 0.0

        if options.draw_boxes:
            x1, y1, x2, y2 = track.bbox.as_int_xyxy()
            thickness = 3 if threat in (ThreatLevel.REVIEW, ThreatLevel.HIGH_RISK) else 2
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
            lines = [f"ID: {track.track_id}", f"STATE: {state.value}", f"RISK: {risk:.0f}%"]
            _label_block(image, lines, (x1, max(0, y1 - 6)), color)

        if options.draw_trajectory and len(track.trajectory) > 1:
            points = np.array(
                [[p.position.x, p.position.y] for p in list(track.trajectory)[-40:]],
                dtype=np.int32,
            )
            cv2.polylines(image, [points.reshape((-1, 1, 2))], False, color, 1, cv2.LINE_AA)

        pose = track.pose
        if pose is None:
            return

        if options.draw_skeleton:
            for start, end in COCO_SKELETON:
                a = pose.point_of(start, options.keypoint_confidence)
                b = pose.point_of(end, options.keypoint_confidence)
                if a is None or b is None:
                    continue
                cv2.line(image, a.as_int_tuple(), b.as_int_tuple(), WHITE, 2, cv2.LINE_AA)
            for keypoint in pose.keypoints.values():
                if keypoint.confidence < options.keypoint_confidence:
                    continue
                cv2.circle(image, keypoint.point.as_int_tuple(), 3, CYAN, -1, cv2.LINE_AA)

        if options.draw_wrists:
            for hand in Hand:
                wrist = pose.wrist(hand, options.keypoint_confidence)
                if wrist is None:
                    continue
                highlight = observation is not None and observation.associated_hand is hand
                cv2.circle(
                    image,
                    wrist.as_int_tuple(),
                    10 if highlight else 7,
                    ORANGE if highlight else YELLOW,
                    2,
                    cv2.LINE_AA,
                )
                _label(
                    image,
                    "L" if hand is Hand.LEFT else "R",
                    (int(wrist.x) + 12, int(wrist.y) - 8),
                    YELLOW,
                )

        if options.draw_storage_regions and observation is not None:
            self._draw_storage_regions(image, observation.storage_regions)

    @staticmethod
    def _draw_storage_regions(
        image: np.ndarray, regions: list[StorageRegionEstimate]
    ) -> None:
        for region in regions:
            cv2.circle(
                image,
                region.center.as_int_tuple(),
                max(3, int(region.radius)),
                PURPLE,
                1,
                cv2.LINE_AA,
            )

    # -- items -------------------------------------------------------------
    @staticmethod
    def _draw_item(image: np.ndarray, item: ItemTrack) -> None:
        color = ITEM_STATUS_COLORS.get(item.status, GREY)
        x1, y1, x2, y2 = item.bbox.as_int_xyxy()
        # Missing/occluded items are drawn dashed at their last known position,
        # so a reviewer can see where the system lost sight of them.
        if item.status in (ItemStatus.OCCLUDED, ItemStatus.MISSING):
            _dashed_rectangle(image, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"item {item.item_id} {item.status.value}"
        if item.associated_person_id is not None:
            label += f" -> ID {item.associated_person_id}"
        _label(image, label, (x1, max(12, y1 - 6)), color)

    # -- panels ------------------------------------------------------------
    @staticmethod
    def _draw_stats(image: np.ndarray, stats: OverlayStats) -> None:
        lines = [
            "AisleGuard Vision",
            f"CAMERA: {stats.camera_id}",
            "",
            f"VIDEO:     {stats.video_fps:5.1f} FPS",
            f"DETECTION: {stats.detection_fps:5.1f} FPS",
            f"POSE:      {stats.pose_fps:5.1f} FPS",
            f"INFERENCE: {stats.inference_latency_ms:5.0f} ms",
            f"POSE LAT:  {stats.pose_latency_ms:5.0f} ms",
            f"TRACKS:    {stats.tracks}",
            f"ITEMS:     {stats.item_tracks}",
            f"QUEUE:     {stats.queue_depth}",
            f"DROPPED:   {stats.dropped}",
            f"ALERTS:    {stats.alerts}",
        ]
        if stats.device:
            lines.append(f"DEVICE:    {stats.device}")
        if stats.mode_note:
            lines.extend(["", stats.mode_note])

        width = 300
        height = 22 * len(lines) + 16
        panel = image[8 : 8 + height, 8 : 8 + width]
        if panel.size:
            cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0, panel)
        cv2.rectangle(image, (8, 8), (8 + width, 8 + height), GREY, 1, cv2.LINE_AA)

        y = 30
        for index, line in enumerate(lines):
            color = WHITE if index > 1 else GREEN
            if line.startswith("Zone-only"):
                color = YELLOW
            cv2.putText(image, line, (18, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 22

    def _draw_evidence(
        self, image: np.ndarray, observations: dict[int, BehaviorObservation]
    ) -> None:
        """Show the reasoning for the highest-risk person currently in frame.

        The explanation is the product. If an operator cannot see why the score
        is what it is, they cannot tune the system or trust it.
        """
        if not observations:
            return
        top = max(observations.values(), key=lambda o: o.risk.risk_score)
        if top.risk.risk_score <= 0:
            return

        lines = [f"TRACK {top.person_id}  {top.risk.threat_level.value}  {top.risk.risk_score:.0f}"]
        contributions = sorted(
            (c for c in top.risk.contributions if c.applied_weight != 0),
            key=lambda c: -abs(c.applied_weight),
        )[: self.options.max_evidence_lines]
        lines.extend(f"{c.applied_weight:+5.0f} {c.description[:52]}" for c in contributions)

        height = 20 * len(lines) + 14
        width = 470
        origin_y = image.shape[0] - height - 10
        panel = image[origin_y : origin_y + height, 10 : 10 + width]
        if panel.size:
            cv2.addWeighted(panel, 0.25, np.zeros_like(panel), 0.75, 0, panel)
        cv2.rectangle(
            image,
            (10, origin_y),
            (10 + width, origin_y + height),
            THREAT_COLORS.get(top.risk.threat_level, GREY),
            1,
            cv2.LINE_AA,
        )
        y = origin_y + 20
        for index, line in enumerate(lines):
            color = THREAT_COLORS.get(top.risk.threat_level, WHITE) if index == 0 else WHITE
            if index > 0 and line.startswith("-"):
                color = GREEN  # negative (exculpatory) evidence
            cv2.putText(image, line, (18, y), FONT, 0.42, color, 1, cv2.LINE_AA)
            y += 20


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def _label(
    image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    """Text with a filled backing box, so it stays readable over any footage."""
    (width, height), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    x, y = origin
    cv2.rectangle(image, (x, y - height - 4), (x + width + 6, y + 4), BLACK, -1)
    cv2.putText(image, text, (x + 3, y), FONT, 0.45, color, 1, cv2.LINE_AA)


def _label_block(
    image: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int]
) -> None:
    x, y = origin
    # Grow upward from the anchor so the block never covers the subject.
    start = y - 20 * len(lines)
    for index, line in enumerate(lines):
        _label(image, line, (x, max(14, start + 20 * (index + 1))), color)


def _dashed_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash: int = 8,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_incident_banner(image: np.ndarray, text: str) -> np.ndarray:
    """Stamp an incident banner across the top of a saved snapshot/clip frame.

    The wording is fixed on purpose: a saved image can end up in front of
    someone who never read the documentation, and it must not read as an
    accusation.
    """
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 44), BLACK, -1)
    cv2.putText(image, text, (12, 20), FONT, 0.55, RED, 2, cv2.LINE_AA)
    cv2.putText(
        image,
        "Possible concealment behavior - human review recommended (not a theft determination)",
        (12, 37),
        FONT,
        0.40,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    return image


def crop_person(image: np.ndarray, bbox: BoundingBox, padding: float = 0.2) -> np.ndarray:
    """Extract a padded crop of a tracked person, for the incident snapshot."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox.expanded(padding).clipped_to(width, height).as_int_xyxy()
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2].copy()


def point_to_int(point: Point) -> tuple[int, int]:
    return point.as_int_tuple()


def keypoint_names() -> list[str]:
    return [name.value for name in KeypointName]
