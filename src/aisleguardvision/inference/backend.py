"""Detector backend abstraction.

The contract that keeps the rest of the system portable:

* a backend receives a **list** of BGR images and returns a **list** of
  per-image :class:`~aisleguardvision.core.types.Detection` lists;
* it never returns a framework-native object;
* it maps its own label space onto :class:`ObjectClass`.

The list-in/list-out shape is the whole point of designing it now rather than
later. The single-camera MVP passes a list of one. A multi-camera GPU worker
passes eight frames from eight cameras and gets one batched GPU invocation --
with no signature change, no change to the tracker, and no change to the
behavior engine.

Planned implementations beyond ``UltralyticsBackend``:

``TensorRTBackend``   Jetson / dGPU, FP16 and INT8 engines
``ONNXBackend``       portable CPU and non-NVIDIA accelerators
``TritonBackend``     centralized multi-GPU inference server
``DeepStreamBackend`` NVDEC-to-GPU-tensor with no host round trip

Each becomes a new subclass registered here. Nothing in ``behavior/``,
``tracking/`` or ``events/`` changes when one is added; that is the test of
whether this abstraction is doing its job.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.config import InferenceConfig, ModelsConfig
from ..core.logging import get_logger
from ..core.types import Detection, ObjectClass
from .device import DeviceInfo, select_device

logger = get_logger(__name__)


#: COCO label -> internal semantic class. Anything unlisted is dropped.
#: Note what is deliberately absent: there is no COCO class for general retail
#: merchandise. See ProductDetector for how that gap is handled honestly.
COCO_CLASS_MAP: dict[str, ObjectClass] = {
    "person": ObjectClass.PERSON,
    "cell phone": ObjectClass.PHONE,
    "handbag": ObjectClass.HANDBAG,
    "backpack": ObjectClass.BACKPACK,
    "suitcase": ObjectClass.SUITCASE,
    "bottle": ObjectClass.BOTTLE,
    "cup": ObjectClass.CUP,
    "book": ObjectClass.BOOK,
}


def map_class_name(name: str) -> ObjectClass:
    return COCO_CLASS_MAP.get(name.strip().lower(), ObjectClass.UNKNOWN)


class BackendError(RuntimeError):
    """Raised when a backend cannot be created or a model cannot be loaded."""


@dataclass(slots=True)
class BackendCapabilities:
    """What a backend can do, for scheduler and startup decisions."""

    supports_batching: bool = False
    max_batch_size: int = 1
    supports_pose: bool = False
    supports_fp16: bool = False
    name: str = ""


@dataclass(slots=True)
class BackendStats:
    """Rolling backend counters, surfaced through /metrics."""

    invocations: int = 0
    images: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    errors: int = 0

    @property
    def mean_latency_ms(self) -> float:
        return self.total_latency_ms / self.invocations if self.invocations else 0.0

    def record(self, latency_ms: float, image_count: int) -> None:
        self.invocations += 1
        self.images += image_count
        self.total_latency_ms += latency_ms
        self.last_latency_ms = latency_ms


class DetectorBackend(ABC):
    """Base class for all object-detection backends.

    Subclasses implement :meth:`_infer_batch`. The public :meth:`infer` adds
    timing, error isolation and the "never let a model failure take down a
    camera" guarantee.
    """

    def __init__(
        self,
        models: ModelsConfig,
        inference: InferenceConfig,
        device: DeviceInfo | None = None,
    ) -> None:
        self.models = models
        self.inference = inference
        self.device = device or select_device(models.device, models.fp16)
        self.stats = BackendStats()
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights and prepare the backend. Must be idempotent."""

    def warmup(self, width: int = 640, height: int = 384, iterations: int = 2) -> None:
        """Run throwaway inferences so the first real frame is not slow.

        On CUDA the first invocation pays kernel autotuning and memory-pool
        setup, which can be hundreds of milliseconds -- long enough to look
        like a stall on a live stream.
        """
        if not self._loaded:
            self.load()
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(1, iterations)):
            try:
                self.infer([blank])
            except Exception as exc:  # pragma: no cover - backend specific
                logger.warning("warmup inference failed", extra={"fields": {"error": str(exc)}})
                return
        logger.info(
            "backend warmed up",
            extra={"fields": {"backend": self.capabilities.name, "device": self.device.device}},
        )

    def close(self) -> None:
        """Release resources. Safe to call more than once."""
        self._loaded = False

    # -- inference ---------------------------------------------------------
    def infer(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Run detection over a batch of images.

        Returns one detection list per input image, in the same order. On a
        backend error an empty result is returned for every image and the
        error is counted -- a model failure must degrade the camera, not crash
        the process.
        """
        if not images:
            return []
        if not self._loaded:
            self.load()

        started = time.perf_counter()
        try:
            results = self._infer_batch(images)
        except Exception as exc:
            self.stats.errors += 1
            logger.error(
                "detector backend inference failed",
                extra={
                    "fields": {
                        "backend": self.capabilities.name,
                        "batch": len(images),
                        "error": str(exc),
                    }
                },
                exc_info=True,
            )
            return [[] for _ in images]

        latency_ms = (time.perf_counter() - started) * 1000.0
        self.stats.record(latency_ms, len(images))

        if len(results) != len(images):
            logger.error(
                "backend returned a mismatched result count; padding",
                extra={"fields": {"expected": len(images), "got": len(results)}},
            )
            results = list(results) + [[] for _ in range(len(images) - len(results))]
        return results[: len(images)]

    @abstractmethod
    def _infer_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Backend-specific batched inference."""

    # -- introspection -----------------------------------------------------
    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Static description of what this backend supports."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __enter__(self) -> DetectorBackend:
        self.load()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class PoseBackend(DetectorBackend):
    """Backend that additionally populates ``Detection.pose``.

    Kept as a separate type so that the scheduler can assert at startup that a
    pose model is actually available, rather than discovering mid-stream that
    every ``Detection.pose`` is ``None``.
    """

    @abstractmethod
    def _infer_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        """Must return detections whose ``pose`` field is populated."""


class NullBackend(DetectorBackend):
    """Backend that detects nothing.

    Not a placeholder for missing work: it is how the pipeline stays runnable
    when no model weights are present (CI, the simulator, a config check on a
    laptop). It reports zero capability rather than pretending, so the behavior
    engine correctly enters zone-only mode and caps risk.
    """

    def load(self) -> None:
        self._loaded = True

    def _infer_batch(self, images: list[np.ndarray]) -> list[list[Detection]]:
        return [[] for _ in images]

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            supports_batching=True,
            max_batch_size=64,
            supports_pose=False,
            supports_fp16=False,
            name="null",
        )


#: Backend key -> factory. Populated by the concrete modules on import.
_REGISTRY: dict[str, Any] = {"null": NullBackend}


def register_backend(name: str, factory: Any) -> None:
    """Register a backend implementation under a config key."""
    _REGISTRY[name.strip().lower()] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_detector_backend(
    models: ModelsConfig,
    inference: InferenceConfig,
    *,
    pose: bool = False,
    device: DeviceInfo | None = None,
) -> DetectorBackend:
    """Build the configured backend.

    Falls back to :class:`NullBackend` -- loudly -- when the requested backend
    is unavailable. A store camera that runs with degraded capability and says
    so is strictly better than one that fails to start.
    """
    key = (models.backend or "ultralytics").strip().lower()

    if key == "ultralytics":
        # Imported lazily and registered on first use, so that merely importing
        # this module does not require torch.
        try:
            from . import ultralytics_backend  # noqa: F401
        except Exception as exc:
            logger.error(
                "ultralytics backend unavailable; falling back to the null backend. "
                "Install the 'inference' extra to enable live detection.",
                extra={"fields": {"error": str(exc)}},
            )
            return NullBackend(models, inference, device)

    factory = _REGISTRY.get(key)
    if factory is None:
        logger.error(
            "unknown detector backend; falling back to the null backend",
            extra={"fields": {"requested": key, "available": ",".join(available_backends())}},
        )
        return NullBackend(models, inference, device)

    try:
        return factory(models, inference, device=device, pose=pose)
    except TypeError:
        # Backends that do not offer pose accept the narrower signature.
        return factory(models, inference, device=device)
    except BackendError as exc:
        logger.error(
            "detector backend failed to initialize; falling back to the null backend",
            extra={"fields": {"backend": key, "error": str(exc)}},
        )
        return NullBackend(models, inference, device)


def filter_detections(
    detections: list[Detection],
    keep: set[ObjectClass] | None = None,
    min_confidence: float = 0.0,
) -> list[Detection]:
    """Post-filter a detection list by class and confidence."""
    return [
        d
        for d in detections
        if d.confidence >= min_confidence and (keep is None or d.object_class in keep)
    ]
