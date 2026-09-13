"""Metrics registry.

A small, dependency-free, thread-safe metrics layer with a Prometheus text
exposition renderer. Deliberately not `prometheus_client`: the MVP must not
require a metrics server, and every instrument here is cheap enough to call
from the frame loop (a lock acquisition and a float add).

Instruments
-----------
``Counter``    monotonically increasing totals (frames, alerts, reconnects)
``Gauge``      instantaneous values (queue depth, active tracks)
``RateMeter``  events/second over a sliding window (all the *_fps metrics)
``Histogram``  bounded-sample latency distributions (inference_latency_ms)

Labels are modelled as a flat tuple appended to the metric name, which keeps
lookup to a single dict access on the hot path.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Counter:
    """Monotonically increasing total."""

    name: str
    description: str = ""
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0


@dataclass(slots=True)
class Gauge:
    """Instantaneous value that can go up and down."""

    name: str
    description: str = ""
    _value: float = 0.0

    def set(self, value: float) -> None:
        self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value


class RateMeter:
    """Events per second over a sliding wall-clock window.

    Used for every ``*_fps`` metric. Timestamps outside the window are evicted
    lazily on read, so a stalled camera correctly decays to 0 FPS instead of
    reporting its last healthy rate forever.
    """

    __slots__ = ("name", "description", "_window", "_events", "_lock", "_total")

    def __init__(self, name: str, window_seconds: float = 5.0, description: str = "") -> None:
        self.name = name
        self.description = description
        self._window = window_seconds
        self._events: deque[float] = deque()
        self._lock = threading.Lock()
        self._total = 0

    def tick(self, count: int = 1, now: float | None = None) -> None:
        timestamp = now if now is not None else time.monotonic()
        with self._lock:
            for _ in range(count):
                self._events.append(timestamp)
            self._total += count
            self._evict(timestamp)

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        events = self._events
        while events and events[0] < cutoff:
            events.popleft()

    @property
    def value(self) -> float:
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            if not self._events:
                return 0.0
            span = now - self._events[0]
            if span <= 1e-6:
                return 0.0
            return len(self._events) / span

    @property
    def total(self) -> int:
        return self._total


class Histogram:
    """Bounded-sample distribution, reporting mean and percentiles.

    Keeps the last N samples rather than fixed buckets: for latency tuning,
    "what did p95 look like over the last 512 inferences" is the question that
    actually gets asked, and it costs one deque append per observation.
    """

    __slots__ = ("name", "description", "_samples", "_lock", "_count", "_sum")

    def __init__(self, name: str, size: int = 512, description: str = "") -> None:
        self.name = name
        self.description = description
        self._samples: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()
        self._count = 0
        self._sum = 0.0

    def observe(self, value: float) -> None:
        with self._lock:
            self._samples.append(float(value))
            self._count += 1
            self._sum += float(value)

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            samples = np.fromiter(self._samples, dtype=np.float64) if self._samples else None
            count, total = self._count, self._sum
        if samples is None or samples.size == 0:
            return {"count": count, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        return {
            "count": float(count),
            "mean": float(samples.mean()),
            "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)),
            "max": float(samples.max()),
            "sum": total,
        }

    @property
    def value(self) -> float:
        """Mean of retained samples, for compact display."""
        with self._lock:
            if not self._samples:
                return 0.0
            return float(sum(self._samples) / len(self._samples))


def _key(name: str, labels: dict[str, str] | None) -> str:
    if not labels:
        return name
    rendered = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{rendered}}}"


class MetricsRegistry:
    """Process-wide registry.

    Instrument lookup is ``get_or_create`` so call sites do not need to
    pre-declare anything; the first camera worker to start registers its own
    per-camera instruments.
    """

    def __init__(self, rate_window_seconds: float = 5.0, histogram_size: int = 512) -> None:
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._rates: dict[str, RateMeter] = {}
        self._histograms: dict[str, Histogram] = {}
        self._lock = threading.Lock()
        self._rate_window = rate_window_seconds
        self._histogram_size = histogram_size

    # -- instrument accessors ---------------------------------------------
    def counter(self, name: str, labels: dict[str, str] | None = None) -> Counter:
        key = _key(name, labels)
        with self._lock:
            instrument = self._counters.get(key)
            if instrument is None:
                instrument = Counter(key)
                self._counters[key] = instrument
            return instrument

    def gauge(self, name: str, labels: dict[str, str] | None = None) -> Gauge:
        key = _key(name, labels)
        with self._lock:
            instrument = self._gauges.get(key)
            if instrument is None:
                instrument = Gauge(key)
                self._gauges[key] = instrument
            return instrument

    def rate(self, name: str, labels: dict[str, str] | None = None) -> RateMeter:
        key = _key(name, labels)
        with self._lock:
            instrument = self._rates.get(key)
            if instrument is None:
                instrument = RateMeter(key, self._rate_window)
                self._rates[key] = instrument
            return instrument

    def histogram(self, name: str, labels: dict[str, str] | None = None) -> Histogram:
        key = _key(name, labels)
        with self._lock:
            instrument = self._histograms.get(key)
            if instrument is None:
                instrument = Histogram(key, self._histogram_size)
                self._histograms[key] = instrument
            return instrument

    # -- convenience shorthands -------------------------------------------
    def inc(self, name: str, amount: float = 1.0, **labels: str) -> None:
        self.counter(name, labels or None).inc(amount)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        self.gauge(name, labels or None).set(value)

    def tick(self, name: str, count: int = 1, **labels: str) -> None:
        self.rate(name, labels or None).tick(count)

    def observe(self, name: str, value: float, **labels: str) -> None:
        self.histogram(name, labels or None).observe(value)

    # -- export ------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Full metrics snapshot, as returned by ``GET /metrics``."""
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            rates = dict(self._rates)
            histograms = dict(self._histograms)
        return {
            "counters": {k: v.value for k, v in counters.items()},
            "gauges": {k: v.value for k, v in gauges.items()},
            "rates": {k: round(v.value, 2) for k, v in rates.items()},
            "histograms": {k: v.snapshot() for k, v in histograms.items()},
        }

    def to_prometheus(self) -> str:
        """Render in Prometheus text exposition format.

        Present so that wiring a real scrape endpoint later is a routing
        change, not a metrics rewrite.
        """
        lines: list[str] = []
        snap = self.snapshot()
        for name, value in snap["counters"].items():
            base, labels = _split_key(name)
            lines.append(f"# TYPE {base} counter")
            lines.append(f"{base}{labels} {value}")
        for section in ("gauges", "rates"):
            for name, value in snap[section].items():
                base, labels = _split_key(name)
                lines.append(f"# TYPE {base} gauge")
                lines.append(f"{base}{labels} {value}")
        for name, stats in snap["histograms"].items():
            base, labels = _split_key(name)
            lines.append(f"# TYPE {base} summary")
            for quantile in ("p50", "p95", "p99"):
                q = quantile[1:]
                q_value = f"0.{q}" if len(q) == 2 else f"0.{q}"
                inner = labels[1:-1] + "," if labels else ""
                lines.append(f'{base}{{{inner}quantile="{q_value}"}} {stats[quantile]}')
            lines.append(f"{base}_count{labels} {stats['count']}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._rates.clear()
            self._histograms.clear()


def _split_key(key: str) -> tuple[str, str]:
    if "{" not in key:
        return key, ""
    base, rest = key.split("{", 1)
    labels = "{" + rest
    # Prometheus requires quoted label values.
    inner = labels[1:-1]
    quoted = ",".join(
        f'{part.split("=", 1)[0]}="{part.split("=", 1)[1]}"'
        for part in inner.split(",")
        if "=" in part
    )
    return base, "{" + quoted + "}"


#: Canonical metric names. Referenced from workers, the API and the overlay so
#: a rename is a single-site change.
class MetricNames:
    CAMERA_FPS = "camera_fps"
    DECODE_FPS = "decode_fps"
    INFERENCE_FPS = "inference_fps"
    POSE_FPS = "pose_fps"
    PIPELINE_FPS = "pipeline_fps"
    INFERENCE_LATENCY_MS = "inference_latency_ms"
    POSE_LATENCY_MS = "pose_latency_ms"
    PIPELINE_LATENCY_MS = "pipeline_latency_ms"
    END_TO_END_LATENCY_MS = "end_to_end_latency_ms"
    QUEUE_DEPTH = "queue_depth"
    FRAMES_RECEIVED = "frames_received"
    FRAMES_PROCESSED = "frames_processed"
    FRAMES_DROPPED = "frames_dropped"
    ACTIVE_TRACKS = "active_tracks"
    ACTIVE_ITEM_TRACKS = "active_item_tracks"
    EVENTS_CREATED = "events_created"
    ALERTS_GENERATED = "alerts_generated"
    ALERTS_SUPPRESSED_COOLDOWN = "alerts_suppressed_cooldown"
    CAMERA_RECONNECTS = "camera_reconnects"
    CAMERA_FAILURES = "camera_failures"
    WEBHOOK_FAILURES = "webhook_failures"
    WEBHOOK_DELIVERIES = "webhook_deliveries"
    CLIP_WRITE_ERRORS = "clip_write_errors"
    CLIPS_WRITTEN = "clips_written"
    BEHAVIOR_TRANSITIONS = "behavior_transitions"


_GLOBAL_REGISTRY: MetricsRegistry | None = None
_GLOBAL_LOCK = threading.Lock()


def get_metrics() -> MetricsRegistry:
    """Return the process-wide registry, creating it on first use.

    A module-level singleton is justified here: metrics are cross-cutting and
    threading one through every constructor would be noise. Components still
    accept an explicit registry for testability, defaulting to this one.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = MetricsRegistry()
    return _GLOBAL_REGISTRY


def configure_metrics(rate_window_seconds: float, histogram_size: int) -> MetricsRegistry:
    """Replace the global registry with one built from configuration."""
    global _GLOBAL_REGISTRY
    with _GLOBAL_LOCK:
        _GLOBAL_REGISTRY = MetricsRegistry(rate_window_seconds, histogram_size)
        return _GLOBAL_REGISTRY
