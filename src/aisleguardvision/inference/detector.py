"""Person and object detection.

A thin, typed layer over a :class:`DetectorBackend` that splits one model
invocation into the three things the pipeline needs from it:

* **people** -- fed to ByteTrack;
* **carryable objects** -- phones, bags, bottles, cups, books; the phone class
  in particular drives the most important false-positive suppression there is;
* **containers** -- carts and baskets, where COCO provides them.

Everything is expressed in :class:`Detection`; no framework types escape.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..core.config import DetectionConfig
from ..core.logging import get_logger
from ..core.metrics import MetricNames, MetricsRegistry, get_metrics
from ..core.types import Detection, InferenceResult, ObjectClass
from .backend import DetectorBackend

logger = get_logger(__name__)

CONTAINER_CLASSES = frozenset({ObjectClass.SHOPPING_CART, ObjectClass.BASKET})


@dataclass(slots=True)
class DetectionBundle:
    """One frame's detections, partitioned by role."""

    camera_id: str
    frame_id: int
    timestamp: float
    people: list[Detection] = field(default_factory=list)
    objects: list[Detection] = field(default_factory=list)
    containers: list[Detection] = field(default_factory=list)
    latency_ms: float = 0.0
    frame_width: int = 0
    frame_height: int = 0

    @property
    def all_detections(self) -> list[Detection]:
        return [*self.people, *self.objects, *self.containers]

    def to_inference_result(self, model_name: str = "") -> InferenceResult:
        return InferenceResult(
            camera_id=self.camera_id,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            detections=self.all_detections,
            latency_ms=self.latency_ms,
            model_name=model_name,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
        )


class PersonDetector:
    """Detects people and relevant objects on one or more frames."""

    def __init__(
        self,
        backend: DetectorBackend,
        config: DetectionConfig,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self.backend = backend
        self.config = config
        self.metrics = metrics or get_metrics()

    def detect(
        self,
        image: np.ndarray,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> DetectionBundle:
        """Single-frame convenience wrapper around :meth:`detect_batch`."""
        return self.detect_batch([(image, camera_id, frame_id, timestamp)])[0]

    def detect_batch(
        self, requests: list[tuple[np.ndarray, str, int, float]]
    ) -> list[DetectionBundle]:
        """Detect over a batch of ``(image, camera_id, frame_id, timestamp)``.

        The batch may span cameras: that is exactly the multi-camera GPU worker
        case, and it costs one backend invocation instead of N.
        """
        if not requests:
            return []

        images = [request[0] for request in requests]
        started = time.perf_counter()
        results = self.backend.infer(images)
        latency_ms = (time.perf_counter() - started) * 1000.0
        per_image_latency = latency_ms / max(1, len(requests))

        bundles: list[DetectionBundle] = []
        for (image, camera_id, frame_id, timestamp), detections in zip(
            requests, results, strict=True
        ):
            bundle = self._partition(detections, camera_id, frame_id, timestamp, image)
            bundle.latency_ms = per_image_latency
            bundles.append(bundle)

            self.metrics.tick(MetricNames.INFERENCE_FPS, camera_id=camera_id)
            self.metrics.observe(
                MetricNames.INFERENCE_LATENCY_MS, per_image_latency, camera_id=camera_id
            )
        return bundles

    def _partition(
        self,
        detections: list[Detection],
        camera_id: str,
        frame_id: int,
        timestamp: float,
        image: np.ndarray,
    ) -> DetectionBundle:
        inference = self.config.inference
        bundle = DetectionBundle(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
            frame_width=int(image.shape[1]),
            frame_height=int(image.shape[0]),
        )
        for detection in detections:
            object_class = detection.object_class
            if object_class is ObjectClass.PERSON:
                if detection.confidence >= inference.person_confidence:
                    bundle.people.append(detection)
            elif object_class in CONTAINER_CLASSES:
                if detection.confidence >= inference.object_confidence:
                    bundle.containers.append(detection)
            elif object_class is not ObjectClass.UNKNOWN:
                if detection.confidence >= inference.object_confidence:
                    bundle.objects.append(detection)
        return bundle

    def warmup(self) -> None:
        self.backend.warmup(width=self.config.inference.image_size, height=self.config.inference.image_size)

    def close(self) -> None:
        self.backend.close()

    @property
    def is_operational(self) -> bool:
        """False when running on the null backend (no weights available)."""
        return self.backend.capabilities.name != "null"
