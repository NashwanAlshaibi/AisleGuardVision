"""Shared pytest fixtures.

These tests run with NO GPU, NO model weights and NO torch installed. That is
deliberate: the behavior engine, tracking, risk scoring and event pipeline are
the parts where a bug means a false accusation, and they must be verifiable in
CI on any machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aisleguardvision.behavior.zones import ZoneRegistry  # noqa: E402
from aisleguardvision.core.config import AppConfig  # noqa: E402
from aisleguardvision.core.logging import configure_logging  # noqa: E402
from aisleguardvision.core.types import (  # noqa: E402
    BoundingBox,
    Detection,
    ObjectClass,
    Point,
)
from aisleguardvision.simulation.scenarios import SHELF_ZONES  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _quiet_logging():
    """Keep test output readable; the engine logs at INFO on every transition."""
    configure_logging(level="CRITICAL", fmt="text", force=True)


@pytest.fixture
def config() -> AppConfig:
    """Default configuration - the values actually shipped in config/."""
    return AppConfig()


@pytest.fixture
def square_polygon():
    import numpy as np

    return np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]], dtype=np.float32)


@pytest.fixture
def shelf_registry() -> ZoneRegistry:
    return ZoneRegistry.from_config("sim_cam", SHELF_ZONES)


def person_detection(
    x: float, y: float, width: float = 190.0, height: float = 520.0, confidence: float = 0.9
) -> Detection:
    """A person detection centred on ``(x, y)``."""
    return Detection(
        bbox=BoundingBox(x - width / 2, y - height / 2, x + width / 2, y + height / 2),
        confidence=confidence,
        object_class=ObjectClass.PERSON,
        class_name="person",
    )


def box_at(centre: Point, size: float = 40.0) -> BoundingBox:
    half = size / 2
    return BoundingBox(centre.x - half, centre.y - half, centre.x + half, centre.y + half)
