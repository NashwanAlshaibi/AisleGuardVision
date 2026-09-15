"""Pose estimation.

Two modes:

**Crop mode** (default). Pose runs only on the people the scheduler selected,
by cropping their boxes with padding, batching the crops and mapping keypoints
back into full-frame coordinates. This is what makes *conditional* pose real:
at 64 cameras, running a pose model over every full frame is not affordable,
but running it on the two shoppers currently at a shelf face is.

**Full-frame mode**. One pass over the whole frame. Simpler, and preferable
when most people in view are relevant anyway.

In both modes the resulting poses are matched to tracks **geometrically**
(see :func:`associate_poses_with_tracks`), never by output order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..core.config import DetectionConfig
from ..core.logging import get_logger
from ..core.metrics import MetricNames, MetricsRegistry, get_metrics
from ..core.types import BoundingBox, Detection, Keypoint, PersonTrack, PoseObservation
from .backend import DetectorBackend

logger = get_logger(__name__)

#: Padding added around a person box before cropping, as a fraction of its
#: size. Extended arms routinely fall outside the detector's box, and a wrist
#: clipped off the crop is precisely the keypoint this system depends on.
CROP_PADDING = 0.25


@dataclass(slots=True)
class PoseResult:
    """Poses produced for one frame, with timing."""

    camera_id: str
    timestamp: float
    detections: list[Detection]
    latency_ms: float = 0.0
    #: Number of people pose actually ran on this frame.
    target_count: int = 0


class PoseEstimator:
    """Runs a pose backend over selected people."""

    def __init__(
        self,
        backend: DetectorBackend,
        config: DetectionConfig,
        metrics: MetricsRegistry | None = None,
        crop_mode: bool = True,
    ) -> None:
        self.backend = backend
        self.config = config
        self.metrics = metrics or get_metrics()
        self.crop_mode = crop_mode

    def estimate(
        self,
        image: np.ndarray,
        camera_id: str,
        timestamp: float,
        targets: list[PersonTrack] | None = None,
    ) -> PoseResult:
        """Estimate poses, optionally restricted to ``targets``."""
        started = time.perf_counter()

        if self.crop_mode and targets:
            detections = self._estimate_crops(image, targets)
            target_count = len(targets)
        else:
            detections = self.backend.infer([image])[0]
            target_count = len(detections)

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.metrics.observe(MetricNames.POSE_LATENCY_MS, latency_ms, camera_id=camera_id)
        if detections:
            self.metrics.tick(MetricNames.POSE_FPS, camera_id=camera_id)

        return PoseResult(
            camera_id=camera_id,
            timestamp=timestamp,
            detections=detections,
            latency_ms=latency_ms,
            target_count=target_count,
        )

    # -- crop mode ---------------------------------------------------------
    def _estimate_crops(self, image: np.ndarray, targets: list[PersonTrack]) -> list[Detection]:
        height, width = image.shape[:2]
        crops: list[np.ndarray] = []
        origins: list[tuple[float, float]] = []

        for track in targets:
            box = track.bbox.expanded(CROP_PADDING).clipped_to(width, height)
            x1, y1, x2, y2 = box.as_int_xyxy()
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            crops.append(image[y1:y2, x1:x2])
            origins.append((float(x1), float(y1)))

        if not crops:
            return []

        # One batched invocation for every selected person.
        batched = self.backend.infer(crops)

        detections: list[Detection] = []
        for (offset_x, offset_y), crop_detections in zip(origins, batched, strict=True):
            for detection in crop_detections:
                detections.append(_shift_detection(detection, offset_x, offset_y))
        return detections

    def warmup(self) -> None:
        self.backend.warmup(
            width=self.config.inference.image_size, height=self.config.inference.image_size
        )

    def close(self) -> None:
        self.backend.close()

    @property
    def is_operational(self) -> bool:
        return self.backend.capabilities.name != "null"


def _shift_detection(detection: Detection, offset_x: float, offset_y: float) -> Detection:
    """Translate a crop-space detection back into full-frame coordinates."""
    bbox = BoundingBox(
        detection.bbox.x1 + offset_x,
        detection.bbox.y1 + offset_y,
        detection.bbox.x2 + offset_x,
        detection.bbox.y2 + offset_y,
    )
    pose = detection.pose
    if pose is not None:
        shifted = {
            name: Keypoint(
                name=keypoint.name,
                x=keypoint.x + offset_x,
                y=keypoint.y + offset_y,
                confidence=keypoint.confidence,
            )
            for name, keypoint in pose.keypoints.items()
        }
        pose = PoseObservation(
            keypoints=shifted,
            confidence=pose.confidence,
            bbox=bbox,
            timestamp=pose.timestamp,
        )
    return Detection(
        bbox=bbox,
        confidence=detection.confidence,
        object_class=detection.object_class,
        class_id=detection.class_id,
        class_name=detection.class_name,
        pose=pose,
        attributes=detection.attributes,
    )


def best_pose_for_track(track: PersonTrack, poses: list[PoseObservation]) -> PoseObservation | None:
    """Pick the pose whose keypoints best fall inside a track's box.

    A small helper for diagnostics and tests; the pipeline uses
    :func:`~aisleguardvision.tracking.association.associate_poses_with_tracks`,
    which solves the assignment globally instead of greedily per track.
    """
    if not poses:
        return None
    expanded = track.bbox.expanded(0.15)
    best: tuple[float, PoseObservation] | None = None
    for pose in poses:
        points = np.array([(kp.x, kp.y) for kp in pose.keypoints.values() if kp.confidence >= 0.2])
        if points.size == 0:
            continue
        inside = (
            (points[:, 0] >= expanded.x1)
            & (points[:, 0] <= expanded.x2)
            & (points[:, 1] >= expanded.y1)
            & (points[:, 1] <= expanded.y2)
        )
        score = float(inside.mean())
        if best is None or score > best[0]:
            best = (score, pose)
    return best[1] if best and best[0] >= 0.5 else None
