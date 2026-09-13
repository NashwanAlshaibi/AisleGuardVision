"""Ultralytics YOLO11 backend.

The only module in the codebase permitted to import ``ultralytics`` or
``torch``. Everything it produces is normalized into :class:`Detection` before
it crosses the module boundary, which is what allows a TensorRT, ONNX, Triton
or DeepStream backend to be dropped in later without any change to tracking,
behavior or risk logic.

Importing this module requires torch and ultralytics to be installed. It is
imported lazily from :func:`~aisleguardvision.inference.backend.create_detector_backend`,
so a machine without them can still run the behavior engine, the simulator,
the tests and the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import InferenceConfig, ModelsConfig
from ..core.logging import get_logger
from ..core.types import COCO_KEYPOINT_ORDER, BoundingBox, Detection, ObjectClass, PoseObservation
from .backend import (
    BackendCapabilities,
    BackendError,
    DetectorBackend,
    PoseBackend,
    map_class_name,
    register_backend,
)
from .device import DeviceInfo

logger = get_logger(__name__)

try:  # pragma: no cover - import guard, exercised only by environment
    import torch
    import ultralytics as ultralytics_module
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the Ultralytics backend requires torch and ultralytics: "
        "pip install -r requirements-inference.txt"
    ) from exc


class UltralyticsBackend(DetectorBackend):
    """YOLO11 object detection through Ultralytics."""

    def __init__(
        self,
        models: ModelsConfig,
        inference: InferenceConfig,
        device: DeviceInfo | None = None,
        pose: bool = False,
    ) -> None:
        super().__init__(models, inference, device)
        self._weights = models.pose if pose else models.detector
        self._is_pose = pose
        self._model: Any = None
        self._class_names: dict[int, str] = {}
        self._keep: set[str] = {name.lower() for name in inference.keep_classes}
        self._precision_kwargs: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return
        try:
            self._model = YOLO(self._weights)
            self._model.to(self.device.device)
        except Exception as exc:
            raise BackendError(f"failed to load YOLO weights {self._weights!r}: {exc}") from exc

        names = getattr(self._model, "names", {}) or {}
        self._class_names = {int(k): str(v) for k, v in names.items()}
        self._precision_kwargs = _precision_kwargs(self._use_fp16)
        self._loaded = True
        logger.info(
            "YOLO model loaded",
            extra={
                "fields": {
                    "weights": self._weights,
                    "task": "pose" if self._is_pose else "detect",
                    "device": self.device.device,
                    "fp16": self._use_fp16,
                    "classes": len(self._class_names),
                }
            },
        )

    def close(self) -> None:
        self._model = None
        self._loaded = False
        if self.device.is_cuda:
            try:
                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - best effort
                pass

    @property
    def _use_fp16(self) -> bool:
        return bool(self.models.fp16 and self.device.supports_fp16)

    # -- inference ---------------------------------------------------------
    def _infer_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run one batched prediction.

        ``torch.inference_mode()`` (not ``no_grad``) because it additionally
        disables autograd's version counter bookkeeping, which is measurable on
        small models running at high frame rates.
        """
        confidence = (
            self.inference.pose_confidence if self._is_pose else self.inference.object_confidence
        )
        with torch.inference_mode():
            results = self._model.predict(
                images,
                imgsz=self.inference.image_size,
                conf=confidence,
                iou=self.inference.nms_iou,
                max_det=self.inference.max_detections,
                device=self.device.device,
                verbose=False,
                **self._precision_kwargs,
            )
        return [self._convert(result) for result in results]

    def _convert(self, result: Any) -> list[Detection]:
        """Normalize one Ultralytics ``Results`` object into our types.

        This function is the entire coupling surface to Ultralytics. Keeping it
        small and in one place is deliberate.
        """
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)

        keypoints = None
        pose_data = getattr(result, "keypoints", None)
        if pose_data is not None and getattr(pose_data, "data", None) is not None:
            keypoints = pose_data.data.detach().cpu().numpy()  # (N, 17, 3)

        for index in range(len(xyxy)):
            class_id = int(class_ids[index])
            class_name = self._class_names.get(class_id, str(class_id))
            if self._keep and class_name.lower() not in self._keep and not self._is_pose:
                continue

            object_class = ObjectClass.PERSON if self._is_pose else map_class_name(class_name)
            if object_class is ObjectClass.UNKNOWN and not self._is_pose:
                continue

            bbox = BoundingBox.from_xyxy(xyxy[index])
            pose = None
            if keypoints is not None and index < len(keypoints):
                raw = keypoints[index]
                if raw.shape[0] == len(COCO_KEYPOINT_ORDER):
                    if raw.shape[1] == 2:
                        # Some exports omit the confidence column; treat those
                        # keypoints as fully confident rather than dropping the
                        # pose, and let the behavior engine's own confidence
                        # gates do the filtering.
                        raw = np.concatenate([raw, np.ones((raw.shape[0], 1), raw.dtype)], axis=1)
                    pose = PoseObservation.from_array(raw, bbox)

            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=float(confidences[index]),
                    object_class=object_class,
                    class_id=class_id,
                    class_name=class_name,
                    pose=pose,
                )
            )
        return detections

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            # Ultralytics accepts a list of images and batches them internally.
            supports_batching=True,
            max_batch_size=max(1, self.inference.max_batch_size),
            supports_pose=self._is_pose,
            supports_fp16=self.device.supports_fp16,
            name="ultralytics-pose" if self._is_pose else "ultralytics",
        )


class UltralyticsPoseBackend(UltralyticsBackend, PoseBackend):
    """YOLO11-Pose. Emits detections with ``pose`` populated."""

    def __init__(
        self,
        models: ModelsConfig,
        inference: InferenceConfig,
        device: DeviceInfo | None = None,
        pose: bool = True,
    ) -> None:
        super().__init__(models, inference, device, pose=True)


def _precision_kwargs(use_fp16: bool) -> dict[str, Any]:
    """Half-precision arguments for the installed Ultralytics version.

    Ultralytics renamed ``half`` to ``quantize`` in 8.4 and deprecated the old
    name. Rather than pinning a version or emitting a deprecation warning on
    every frame, the argument is resolved once from the package's own default
    config and nothing precision-related is passed at all when FP32 is in use.
    """
    if not use_fp16:
        return {}
    try:
        import yaml

        defaults_path = Path(ultralytics_module.__file__).parent / "cfg" / "default.yaml"
        keys = yaml.safe_load(defaults_path.read_text(encoding="utf-8")) or {}
    except Exception:  # pragma: no cover - depends on the installed layout
        keys = {}

    if "quantize" in keys:
        return {"quantize": "fp16"}
    if "half" in keys:
        return {"half": True}
    logger.warning(
        "could not determine the half-precision argument for this Ultralytics "
        "version; running in FP32"
    )
    return {}


def _factory(
    models: ModelsConfig,
    inference: InferenceConfig,
    device: DeviceInfo | None = None,
    pose: bool = False,
) -> DetectorBackend:
    return (
        UltralyticsPoseBackend(models, inference, device)
        if pose
        else UltralyticsBackend(models, inference, device, pose=False)
    )


register_backend("ultralytics", _factory)
