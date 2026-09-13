"""Merchandise ("product") detection.

A HONEST STATEMENT OF A HARD LIMITATION
=======================================

An ordinary COCO-trained YOLO model provides **no generic retail-product
class**. COCO has 80 categories -- person, car, dog, banana, laptop -- and not
one of them covers:

* a wig or a hair bundle
* a jar or tube of gel
* a cosmetics package
* a hair-care product
* a hair accessory
* a general beauty-supply item

There is no threshold, no prompt and no post-processing that makes
``yolo11n.pt`` recognize a wig. Any system claiming otherwise is either using a
custom model or is not doing what it says.

How AisleGuard Vision handles that
----------------------------------

1. **State the limitation.** This module exists and is named for the capability
   the product needs, so the gap is visible rather than papered over.
2. **Implement the interface anyway** (:class:`ProductDetector`), so the rest
   of the pipeline is built against the capability it will eventually have.
3. **Ship a functional fallback.** :class:`CocoProxyProductDetector` surfaces
   the handful of carryable COCO classes that *do* correspond to real
   merchandise in some store categories (bottle, cup, book), and
   :class:`ZoneOnlyProductDetector` yields nothing at all, leaving the system
   to reason from shelf geometry.
4. **Keep the rest of the architecture operational.** With no item detections,
   tracking, pose, zone interaction, the state machine, recording and alerting
   all still run -- and the behavior engine correctly refuses to reach the
   alert threshold (``behavior.zone_only_risk_ceiling``).
5. **Document the plug point.** :class:`CustomModelProductDetector` is a real,
   working implementation that takes any YOLO-format retail model and maps its
   classes to :class:`ObjectClass.MERCHANDISE`. Training that model is the work
   that remains; wiring it in is a config change.

Nothing in this module ever invents a detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..core.config import DetectionConfig, ModelsConfig
from ..core.logging import get_logger
from ..core.types import BoundingBox, Detection, ItemDetection, ObjectClass
from .backend import DetectorBackend, create_detector_backend
from .device import DeviceInfo

logger = get_logger(__name__)

#: COCO classes that behave like merchandise candidates in *some* store
#: categories (a grocery or convenience aisle). Explicitly NOT a claim to
#: general retail product detection.
COCO_MERCHANDISE_PROXY: dict[ObjectClass, str] = {
    ObjectClass.BOTTLE: "bottle",
    ObjectClass.CUP: "cup",
    ObjectClass.BOOK: "book",
}


@dataclass(slots=True)
class ProductDetectorInfo:
    """What a product detector can actually do, for logging and the API."""

    name: str
    #: True only for a detector that can find general retail merchandise.
    provides_merchandise_detection: bool
    description: str
    class_count: int = 0


class ProductDetector(ABC):
    """Interface for merchandise-candidate detection.

    Implementations return :class:`ItemDetection` -- deliberately a different
    type from :class:`Detection`, so that "this is a merchandise candidate"
    cannot be confused with "this is some COCO object" anywhere downstream.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[ItemDetection]:
        """Return merchandise candidates for one frame."""

    @abstractmethod
    def info(self) -> ProductDetectorInfo:
        """Describe this detector's real capability."""

    @property
    def provides_merchandise_detection(self) -> bool:
        """Whether item-level merchandise evidence is available.

        The behavior engine consults this. When it is ``False`` the system runs
        in zone-only mode and risk is capped below the alert threshold.
        """
        return self.info().provides_merchandise_detection

    def from_detections(
        self, detections: list[Detection], source: str
    ) -> list[ItemDetection]:
        """Convert already-computed object detections into item candidates.

        Lets a proxy detector reuse the main detector's output instead of
        running a second model over the same frame.
        """
        items: list[ItemDetection] = []
        for detection in detections:
            if detection.object_class not in COCO_MERCHANDISE_PROXY:
                continue
            items.append(
                ItemDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    object_class=detection.object_class,
                    class_name=detection.class_name,
                    source=source,
                )
            )
        return items

    def close(self) -> None:
        """Release any model resources."""


class ZoneOnlyProductDetector(ProductDetector):
    """Detects nothing, and says so.

    This is the correct default for a beauty-supply or general-merchandise
    store with no custom model: the system reasons from shelf zones and wrist
    kinematics, and the risk ceiling keeps it from ever paging a human on that
    evidence alone.
    """

    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[ItemDetection]:
        return []

    def info(self) -> ProductDetectorInfo:
        return ProductDetectorInfo(
            name="zone_only",
            provides_merchandise_detection=False,
            description=(
                "No merchandise detector configured. Shelf-zone interaction evidence "
                "only; risk is capped below the alert threshold by design."
            ),
        )


class CocoProxyProductDetector(ProductDetector):
    """Treats a few carryable COCO classes as merchandise candidates.

    Useful in a grocery or convenience aisle where bottles and cups genuinely
    are the merchandise. Reports ``provides_merchandise_detection=True`` because
    for those categories the evidence is real -- but it covers a tiny slice of
    what a store sells, and it is not a substitute for a trained retail model.
    """

    def __init__(self, detector_backend: DetectorBackend | None = None) -> None:
        self._backend = detector_backend

    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[ItemDetection]:
        if self._backend is None:
            return []
        detections = self._backend.infer([image])[0]
        return self.from_detections(detections, source="coco_proxy")

    def info(self) -> ProductDetectorInfo:
        return ProductDetectorInfo(
            name="coco_proxy",
            provides_merchandise_detection=True,
            description=(
                "COCO carryable classes (bottle, cup, book) used as merchandise "
                "proxies. Covers a narrow slice of retail inventory only."
            ),
            class_count=len(COCO_MERCHANDISE_PROXY),
        )


class CustomModelProductDetector(ProductDetector):
    """A trained retail-merchandise model.

    This is the real plug point. Point ``models.product`` at any YOLO-format
    weights file trained on store inventory and every class it emits is mapped
    to :class:`ObjectClass.MERCHANDISE`, keeping the SKU or category label in
    ``class_name`` for the incident record.

    Nothing else in the pipeline changes: the item tracker, the hand/item
    associator, the behavior engine and the risk weights already expect
    item-level evidence and simply start receiving it.
    """

    def __init__(
        self,
        models: ModelsConfig,
        config: DetectionConfig,
        device: DeviceInfo | None = None,
    ) -> None:
        if not models.product:
            raise ValueError("CustomModelProductDetector requires models.product to be set")
        product_models = models.model_copy(update={"detector": models.product})
        self._backend = create_detector_backend(product_models, config.inference, device=device)
        self._weights = models.product
        self._min_confidence = config.inference.object_confidence

    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[ItemDetection]:
        detections = self._backend.infer([image])[0]
        items: list[ItemDetection] = []
        for detection in detections:
            if detection.confidence < self._min_confidence:
                continue
            items.append(
                ItemDetection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    object_class=ObjectClass.MERCHANDISE,
                    class_name=detection.class_name,
                    source="custom_retail_model",
                    attributes={"model": self._weights, "model_class_id": detection.class_id},
                )
            )
        return items

    def info(self) -> ProductDetectorInfo:
        operational = self._backend.capabilities.name != "null"
        return ProductDetectorInfo(
            name="custom_retail_model",
            provides_merchandise_detection=operational,
            description=(
                f"Custom retail merchandise model ({self._weights})"
                if operational
                else f"Custom retail model {self._weights} failed to load; no item evidence"
            ),
        )

    def close(self) -> None:
        self._backend.close()


class StaticProductDetector(ProductDetector):
    """Returns a fixed set of candidates. For tests and the simulator only."""

    def __init__(self, items: list[ItemDetection] | None = None) -> None:
        self.items = items or []

    def detect(self, image: np.ndarray, camera_id: str, timestamp: float) -> list[ItemDetection]:
        return list(self.items)

    def info(self) -> ProductDetectorInfo:
        return ProductDetectorInfo(
            name="static",
            provides_merchandise_detection=True,
            description="Scripted merchandise candidates (testing only)",
        )


def create_product_detector(
    models: ModelsConfig,
    config: DetectionConfig,
    detector_backend: DetectorBackend | None = None,
    device: DeviceInfo | None = None,
) -> ProductDetector:
    """Build the best product detector the configuration supports.

    Order of preference:

    1. a custom retail model, when ``models.product`` is set;
    2. the COCO proxy, reusing the main detector's output;
    3. zone-only.

    Each choice is logged at startup, including what it cannot do, so an
    operator is never left guessing why an alert did or did not fire.
    """
    if models.product:
        try:
            detector = CustomModelProductDetector(models, config, device)
            logger.info(
                "merchandise detector loaded",
                extra={"fields": {"detector": "custom_retail_model", "weights": models.product}},
            )
            return detector
        except Exception as exc:
            logger.error(
                "custom merchandise model failed to load; falling back",
                extra={"fields": {"weights": models.product, "error": str(exc)}},
            )

    if detector_backend is not None and detector_backend.capabilities.name != "null":
        detector = CocoProxyProductDetector(detector_backend)
        logger.warning(
            "no custom merchandise model configured: using COCO carryable classes as a "
            "proxy. This covers bottles, cups and books only -- it does NOT detect "
            "general retail merchandise (wigs, hair products, cosmetics, accessories). "
            "See docs/ARCHITECTURE.md for how to plug in a trained retail model.",
            extra={"fields": {"detector": "coco_proxy"}},
        )
        return detector

    logger.warning(
        "no merchandise detection available: running in zone-only mode. Shelf-zone "
        "interaction will be scored, but risk is capped below the alert threshold "
        "because zone geometry alone is not sufficient grounds for human review.",
        extra={"fields": {"detector": "zone_only"}},
    )
    return ZoneOnlyProductDetector()


def item_from_box(
    box: BoundingBox, confidence: float = 1.0, source: str = "manual"
) -> ItemDetection:
    """Build a merchandise candidate from a raw box (tools and tests)."""
    return ItemDetection(
        bbox=box,
        confidence=confidence,
        object_class=ObjectClass.MERCHANDISE,
        class_name="merchandise",
        source=source,
    )
