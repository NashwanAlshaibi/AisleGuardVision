"""Constant-velocity Kalman filter for box tracking.

State vector: ``[cx, cy, w, h, vx, vy]``

* centre position and box size are observed directly,
* centre velocity is estimated,
* box size follows a random walk (people change apparent size slowly; modelling
  size velocity mostly amplifies detector jitter).

**dt is a parameter of every predict step, not a constant.** Store cameras run
anywhere from 10 to 30 FPS and a single camera's effective rate changes when
the network is congested or the scheduler drops frames. A tracker with a baked
in frame interval silently mis-predicts under exactly the conditions where
tracking matters most.

Process and measurement noise scale with box height, which makes the filter
behave identically for a shopper near the camera and one far away.
"""

from __future__ import annotations

import numpy as np

from ..core.types import BoundingBox, Point

STATE_DIM = 6
MEASUREMENT_DIM = 4


class KalmanBoxFilter:
    """Kalman filter tracking one axis-aligned box."""

    __slots__ = ("x", "P", "_std_position", "_std_velocity", "_std_measurement")

    def __init__(
        self,
        box: BoundingBox,
        std_position: float = 0.05,
        std_velocity: float = 0.00625,
        std_measurement: float = 0.05,
    ) -> None:
        centre = box.center
        self.x = np.array(
            [centre.x, centre.y, box.width, box.height, 0.0, 0.0], dtype=np.float64
        )
        self._std_position = std_position
        self._std_velocity = std_velocity
        self._std_measurement = std_measurement

        height = max(box.height, 1.0)
        std = np.array(
            [
                2 * std_position * height,
                2 * std_position * height,
                2 * std_position * height,
                2 * std_position * height,
                10 * std_velocity * height,
                10 * std_velocity * height,
            ]
        )
        self.P = np.diag(np.square(std))

    # -- model matrices ----------------------------------------------------
    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        F = np.eye(STATE_DIM)
        F[0, 4] = dt
        F[1, 5] = dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        height = max(self.x[3], 1.0)
        std = np.array(
            [
                self._std_position * height,
                self._std_position * height,
                self._std_position * height,
                self._std_position * height,
                self._std_velocity * height,
                self._std_velocity * height,
            ]
        )
        # Scale with elapsed time: a longer gap means more accumulated
        # uncertainty, which is what lets a track survive a dropped frame
        # without the gate snapping shut.
        return np.diag(np.square(std)) * max(dt, 1e-3) / (1.0 / 30.0)

    def _measurement_noise(self) -> np.ndarray:
        height = max(self.x[3], 1.0)
        std = np.full(MEASUREMENT_DIM, self._std_measurement * height)
        return np.diag(np.square(std))

    # -- filter steps ------------------------------------------------------
    def predict(self, dt: float) -> BoundingBox:
        """Advance the state by ``dt`` seconds and return the predicted box."""
        dt = max(0.0, float(dt))
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        # Width/height must stay positive even under a long unobserved gap.
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        return self.box

    def update(self, box: BoundingBox) -> None:
        """Correct the state with a measured box."""
        centre = box.center
        z = np.array([centre.x, centre.y, box.width, box.height], dtype=np.float64)
        H = np.zeros((MEASUREMENT_DIM, STATE_DIM))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0

        y = z - H @ self.x
        S = H @ self.P @ H.T + self._measurement_noise()
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:  # pragma: no cover - numerically degenerate
            return
        self.x = self.x + K @ y
        identity = np.eye(STATE_DIM)
        # Joseph form would be more numerically stable, but with a 6x6 state
        # and well-conditioned R the simple form is adequate and cheaper.
        self.P = (identity - K @ H) @ self.P
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)

    # -- accessors ---------------------------------------------------------
    @property
    def box(self) -> BoundingBox:
        cx, cy, w, h = self.x[:4]
        return BoundingBox.from_cxcywh(float(cx), float(cy), float(w), float(h))

    @property
    def velocity(self) -> Point:
        """Centre velocity in pixels per second."""
        return Point(float(self.x[4]), float(self.x[5]))


def iou_matrix(a_boxes: list[BoundingBox], b_boxes: list[BoundingBox]) -> np.ndarray:
    """Vectorized pairwise IoU. Shape ``(len(a), len(b))``."""
    if not a_boxes or not b_boxes:
        return np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float64)

    a = np.array([b.as_xyxy() for b in a_boxes], dtype=np.float64)
    b = np.array([b.as_xyxy() for b in b_boxes], dtype=np.float64)

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def greedy_match(
    cost: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy maximum-score bipartite matching.

    ``cost`` is a *similarity* matrix (higher is better); pairs below
    ``threshold`` are never matched.

    Greedy rather than Hungarian on purpose: it avoids a SciPy dependency
    (which matters for Jetson wheels), is O(n log n) in the number of
    candidate pairs, and on IoU matrices for pedestrian tracking the two
    agree on the overwhelming majority of assignments. If a deployment ever
    shows this to be the limiting factor, swapping in
    ``scipy.optimize.linear_sum_assignment`` here is a local change.

    Returns ``(matches, unmatched_rows, unmatched_cols)``.
    """
    rows, cols = cost.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))

    candidates = np.argwhere(cost >= threshold)
    if candidates.size == 0:
        return [], list(range(rows)), list(range(cols))

    scores = cost[candidates[:, 0], candidates[:, 1]]
    # Sort by score descending; ties broken by index for determinism, which
    # keeps tests and replayed video reproducible.
    order = np.lexsort((candidates[:, 1], candidates[:, 0], -scores))

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []
    for index in order:
        r, c = int(candidates[index, 0]), int(candidates[index, 1])
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        matches.append((r, c))

    unmatched_rows = [r for r in range(rows) if r not in used_rows]
    unmatched_cols = [c for c in range(cols) if c not in used_cols]
    return matches, unmatched_rows, unmatched_cols
