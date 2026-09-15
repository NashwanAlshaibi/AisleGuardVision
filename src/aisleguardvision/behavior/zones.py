"""Retail interaction zones.

An ordinary COCO-trained YOLO model has **no generic retail-product class**.
It cannot recognize a wig, a hair bundle, a jar of gel, a cosmetics package or
any other beauty-supply item. Pretending otherwise would be the single most
damaging thing this codebase could do.

So the MVP grounds merchandise interaction in *configured geometry* instead:
an operator marks the shelf faces, the high-value sections, the checkout, the
basket/cart staging areas and any exclusion regions, and the behavior engine
reasons about wrists entering and leaving those polygons. A future custom
merchandise detector (see :mod:`aisleguardvision.inference.product_detector`)
adds item-level evidence on top without changing any of this.

Zones are per camera and defined in image pixel coordinates, so re-aiming a
camera invalidates its zones. ``scripts/test_stream.py --zones`` renders the
configured polygons so calibration can be checked visually.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import ZoneConfig
from ..core.logging import get_logger
from ..core.types import BoundingBox, Point, ShelfZone, ZoneKind
from .geometry import (
    as_polygon,
    box_polygon_overlap_ratio,
    distance_points_to_polygon,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class ZoneHit:
    """Result of testing one point against one zone."""

    zone: ShelfZone
    inside: bool
    #: Distance to the zone boundary in pixels (0 when inside).
    distance: float
    #: Distance normalized by the subject's body height, when one was supplied.
    normalized_distance: float


class ZoneRegistry:
    """Zones for a single camera, with vectorized queries.

    Polygon vertices are stacked once at construction so that querying N
    wrists against M zones is M vectorized calls rather than N*M Python
    iterations.
    """

    def __init__(self, camera_id: str, zones: list[ShelfZone] | None = None) -> None:
        self.camera_id = camera_id
        self._zones: list[ShelfZone] = []
        for zone in zones or []:
            self.add(zone)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, camera_id: str, configs: list[ZoneConfig]) -> ZoneRegistry:
        registry = cls(camera_id)
        for config in configs:
            if not config.enabled:
                logger.debug("zone disabled, skipping", extra={"fields": {"zone_id": config.id}})
                continue
            try:
                polygon = as_polygon(config.polygon)
            except ValueError as exc:
                logger.error(
                    "invalid zone polygon, zone ignored",
                    extra={"fields": {"zone_id": config.id, "error": str(exc)}},
                )
                continue
            if polygon_area(polygon) < 1.0:
                logger.error(
                    "degenerate zone polygon (zero area), zone ignored",
                    extra={"fields": {"zone_id": config.id}},
                )
                continue
            registry.add(
                ShelfZone(
                    zone_id=config.id,
                    camera_id=config.camera_id or camera_id,
                    polygon=polygon,
                    kind=config.kind,
                    name=config.name or config.id,
                    risk_multiplier=config.risk_multiplier,
                    enabled=config.enabled,
                )
            )
        logger.info(
            "zones loaded",
            extra={
                "fields": {
                    "camera_id": camera_id,
                    "zone_count": len(registry._zones),
                    "kinds": ",".join(sorted({z.kind.value for z in registry._zones})) or "none",
                }
            },
        )
        return registry

    def add(self, zone: ShelfZone) -> None:
        if any(existing.zone_id == zone.zone_id for existing in self._zones):
            raise ValueError(f"duplicate zone id {zone.zone_id!r} on camera {self.camera_id!r}")
        self._zones.append(zone)

    # -- accessors ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._zones)

    def __iter__(self):
        return iter(self._zones)

    @property
    def zones(self) -> list[ShelfZone]:
        return list(self._zones)

    def by_id(self, zone_id: str) -> ShelfZone | None:
        for zone in self._zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def of_kind(self, *kinds: ZoneKind) -> list[ShelfZone]:
        wanted = set(kinds)
        return [z for z in self._zones if z.enabled and z.kind in wanted]

    @property
    def merchandise_zones(self) -> list[ShelfZone]:
        return [z for z in self._zones if z.enabled and z.kind.is_merchandise_source]

    @property
    def container_zones(self) -> list[ShelfZone]:
        return [z for z in self._zones if z.enabled and z.kind.is_container]

    @property
    def exclusion_zones(self) -> list[ShelfZone]:
        return [z for z in self._zones if z.enabled and z.kind is ZoneKind.EXCLUSION]

    @property
    def has_merchandise_zones(self) -> bool:
        return bool(self.merchandise_zones)

    # -- queries -----------------------------------------------------------
    def is_excluded(self, point: Point) -> bool:
        """True if the point falls in any exclusion region.

        Exclusion zones mask out areas that would otherwise generate noise:
        staff-only doorways, a mirror, a display screen showing a video loop,
        or a region of the frame where another store's aisle is visible.
        """
        return any(
            points_in_polygon(np.array([[point.x, point.y]]), zone.polygon)[0]
            for zone in self.exclusion_zones
        )

    def box_is_excluded(self, box: BoundingBox, min_overlap: float = 0.6) -> bool:
        return any(
            box_polygon_overlap_ratio(box, zone.polygon) >= min_overlap
            for zone in self.exclusion_zones
        )

    def query_point(
        self,
        point: Point,
        kinds: tuple[ZoneKind, ...] | None = None,
        reference_length: float | None = None,
    ) -> list[ZoneHit]:
        """Test one point against the zones, nearest first.

        ``reference_length`` (typically the subject's body height) turns the
        pixel distance into a scale-invariant ratio, which is what the
        behavior engine thresholds on.
        """
        candidates = [z for z in self._zones if z.enabled and (kinds is None or z.kind in kinds)]
        if not candidates:
            return []
        query = np.array([[point.x, point.y]], dtype=np.float64)
        hits: list[ZoneHit] = []
        for zone in candidates:
            distance = float(distance_points_to_polygon(query, zone.polygon)[0])
            inside = distance <= 0.0
            normalized = (
                distance / reference_length
                if reference_length and reference_length > 0
                else float("inf")
            )
            hits.append(
                ZoneHit(zone=zone, inside=inside, distance=distance, normalized_distance=normalized)
            )
        hits.sort(key=lambda h: (not h.inside, h.distance))
        return hits

    def zone_containing(
        self, point: Point, kinds: tuple[ZoneKind, ...] | None = None
    ) -> ShelfZone | None:
        for hit in self.query_point(point, kinds):
            if hit.inside:
                return hit.zone
        return None

    def nearest_zone(
        self,
        point: Point,
        kinds: tuple[ZoneKind, ...] | None = None,
        reference_length: float | None = None,
    ) -> ZoneHit | None:
        hits = self.query_point(point, kinds, reference_length)
        return hits[0] if hits else None

    def box_overlaps(self, box: BoundingBox, zone: ShelfZone, min_ratio: float = 0.2) -> bool:
        return box_polygon_overlap_ratio(box, zone.polygon) >= min_ratio

    def zones_overlapping_box(
        self, box: BoundingBox, kinds: tuple[ZoneKind, ...] | None = None, min_ratio: float = 0.2
    ) -> list[tuple[ShelfZone, float]]:
        """Zones the box meaningfully overlaps, highest overlap first."""
        results: list[tuple[ShelfZone, float]] = []
        for zone in self._zones:
            if not zone.enabled or (kinds is not None and zone.kind not in kinds):
                continue
            # Cheap AABB reject before the sampling test.
            if box.intersection_area(zone.bounds) <= 0:
                continue
            ratio = box_polygon_overlap_ratio(box, zone.polygon)
            if ratio >= min_ratio:
                results.append((zone, ratio))
        results.sort(key=lambda item: -item[1])
        return results

    def centroid_of(self, zone_id: str) -> Point | None:
        zone = self.by_id(zone_id)
        return polygon_centroid(zone.polygon) if zone else None

    def describe(self) -> str:
        if not self._zones:
            return f"camera {self.camera_id}: no zones configured (zone-driven evidence disabled)"
        parts = [f"{z.zone_id}[{z.kind.value}]" for z in self._zones]
        return f"camera {self.camera_id}: {len(parts)} zones -> {', '.join(parts)}"
