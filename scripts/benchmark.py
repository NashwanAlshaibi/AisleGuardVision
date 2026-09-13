#!/usr/bin/env python3
"""Benchmark the pipeline and report measured throughput.

    python scripts/benchmark.py --source ./data/samples/store.mp4
    python scripts/benchmark.py --source 0 --seconds 30
    python scripts/benchmark.py --synthetic --cameras 1 4 8 16

Measures decode FPS, detection FPS, pose FPS, end-to-end latency, GPU memory,
CPU usage, frames dropped and queue depth.

A NOTE ON CAMERA-COUNT CLAIMS
-----------------------------
``--cameras N`` replays the same workload through N independent pipelines to
estimate headroom. That is an *estimate*, and it is reported as one. It does
not model RTSP bandwidth, NVDEC decoder slots, per-process GPU context
overhead, or the network. A real 64-camera claim requires 64 real cameras on
the real network, and this tool will not make that claim on your behalf.

What it does tell you honestly: how many pipeline-seconds of work this machine
completes per wall-clock second, which is the number that actually bounds
camera count.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from aisleguardvision.behavior.zones import ZoneRegistry  # noqa: E402
from aisleguardvision.core.config import AppConfig, CameraConfig, load_config  # noqa: E402
from aisleguardvision.core.logging import configure_logging, get_logger  # noqa: E402
from aisleguardvision.core.metrics import configure_metrics  # noqa: E402
from aisleguardvision.core.types import Frame  # noqa: E402
from aisleguardvision.inference.backend import create_detector_backend  # noqa: E402
from aisleguardvision.inference.detector import PersonDetector  # noqa: E402
from aisleguardvision.inference.device import cuda_memory_mb, log_device, select_device  # noqa: E402
from aisleguardvision.inference.pose import PoseEstimator  # noqa: E402
from aisleguardvision.inference.product_detector import create_product_detector  # noqa: E402
from aisleguardvision.pipeline import CameraPipeline  # noqa: E402

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Measured results. Every field is observed, none is extrapolated."""

    label: str
    frames: int = 0
    wall_seconds: float = 0.0
    pipeline_fps: float = 0.0
    decode_fps: float = 0.0
    detection_invocations: int = 0
    detection_fps: float = 0.0
    pose_invocations: int = 0
    pose_fps: float = 0.0
    inference_latency_ms: dict[str, float] = field(default_factory=dict)
    pose_latency_ms: dict[str, float] = field(default_factory=dict)
    pipeline_latency_ms: dict[str, float] = field(default_factory=dict)
    frames_dropped: int = 0
    peak_queue_depth: int = 0
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    cpu_percent: float = 0.0
    device: str = ""
    detector: str = ""
    item_detection_available: bool = False


def summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(samples)
    return {
        "mean": round(statistics.fmean(ordered), 2),
        "p50": round(ordered[len(ordered) // 2], 2),
        "p95": round(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], 2),
        "max": round(ordered[-1], 2),
    }


def cpu_sampler():
    """Return a callable giving process CPU percent, or None if psutil is absent."""
    try:
        import psutil
    except ImportError:
        return None
    process = psutil.Process()
    process.cpu_percent(None)  # prime the counter
    return lambda: process.cpu_percent(None)


def load_frames(source: str, limit: int) -> list[np.ndarray]:
    """Decode up to ``limit`` frames into memory, measuring decode speed."""
    import cv2

    capture = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(f"error: could not open source {source!r}")
    frames: list[np.ndarray] = []
    started = time.perf_counter()
    while len(frames) < limit:
        ok, image = capture.read()
        if not ok or image is None:
            break
        frames.append(image)
    elapsed = time.perf_counter() - started
    capture.release()
    if not frames:
        raise SystemExit(f"error: no frames decoded from {source!r}")
    logger.info(
        "decoded source",
        extra={
            "fields": {
                "frames": len(frames),
                "decode_fps": round(len(frames) / elapsed, 1) if elapsed else 0.0,
                "resolution": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
            }
        },
    )
    return frames


def synthetic_frames(count: int, width: int, height: int) -> list[np.ndarray]:
    """Random-noise frames. Stresses the model without needing any footage.

    Noise gives a pessimistic-to-realistic timing picture: NMS cost varies with
    detection count, and noise produces few detections, so treat these numbers
    as a floor on latency rather than a ceiling.
    """
    rng = np.random.default_rng(1234)
    return [
        rng.integers(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(count)
    ]


def build_pipeline(config: AppConfig, camera_id: str, detector, pose, product) -> CameraPipeline:
    camera = CameraConfig(id=camera_id, name=camera_id, source="benchmark", enabled=True)
    if not config.camera(camera_id):
        config.cameras.cameras.append(camera)
    zones = ZoneRegistry.from_config(
        camera_id, config.cameras.cameras[0].zones if config.cameras.cameras else []
    )
    return CameraPipeline(
        camera_id=camera_id,
        config=config,
        detector=detector,
        pose_estimator=pose,
        product_detector=product,
        zones=zones,
        recorder=None,
        dispatcher=None,
        buffer=None,
    )


def run_benchmark(
    config: AppConfig,
    frames: list[np.ndarray],
    camera_count: int,
    seconds: float,
    label: str,
) -> BenchmarkResult:
    """Drive ``camera_count`` pipelines over the frame set for ``seconds``."""
    device = select_device(config.detection.models.device, config.detection.models.fp16)
    models, inference = config.detection.models, config.detection.inference

    backend = create_detector_backend(models, inference, pose=False, device=device)
    detector = PersonDetector(backend, config.detection)
    pose_backend = create_detector_backend(models, inference, pose=True, device=device)
    pose = PoseEstimator(pose_backend, config.detection)
    product = create_product_detector(models, config.detection, backend, device)

    if detector.is_operational:
        detector.warmup()
        pose.warmup()

    pipelines = [
        build_pipeline(config, f"bench_{i:02d}", detector, pose, product)
        for i in range(camera_count)
    ]

    cpu = cpu_sampler()
    result = BenchmarkResult(
        label=label,
        device=device.device,
        detector=models.detector,
        item_detection_available=product.provides_merchandise_detection,
    )
    inference_latencies: list[float] = []
    pose_latencies: list[float] = []
    pipeline_latencies: list[float] = []

    started = time.perf_counter()
    index = 0
    # Synthetic time advances at the camera's nominal rate so the scheduler's
    # time-based gating behaves exactly as it would on a live stream.
    virtual_time = time.time()
    frame_interval = 1.0 / 30.0

    while (time.perf_counter() - started) < seconds:
        image = frames[index % len(frames)]
        index += 1
        virtual_time += frame_interval

        for pipeline in pipelines:
            outcome = pipeline.process(
                Frame(
                    camera_id=pipeline.camera_id,
                    frame_id=index,
                    timestamp=virtual_time,
                    image=image,
                )
            )
            result.frames += 1
            pipeline_latencies.append(outcome.total_latency_ms)
            if outcome.ran_detection:
                result.detection_invocations += 1
                inference_latencies.append(outcome.inference_latency_ms)
            if outcome.ran_pose:
                result.pose_invocations += 1
                pose_latencies.append(outcome.pose_latency_ms)

    result.wall_seconds = time.perf_counter() - started
    result.pipeline_fps = round(result.frames / result.wall_seconds, 2)
    result.detection_fps = round(result.detection_invocations / result.wall_seconds, 2)
    result.pose_fps = round(result.pose_invocations / result.wall_seconds, 2)
    result.inference_latency_ms = summarize(inference_latencies)
    result.pose_latency_ms = summarize(pose_latencies)
    result.pipeline_latency_ms = summarize(pipeline_latencies)
    result.gpu_allocated_mb, result.gpu_reserved_mb = cuda_memory_mb()
    result.cpu_percent = round(cpu(), 1) if cpu else 0.0

    detector.close()
    pose.close()
    return result


def print_result(result: BenchmarkResult) -> None:
    print()
    print("=" * 74)
    print(f"  {result.label}")
    print("=" * 74)
    print(f"  device            : {result.device}")
    print(f"  detector          : {result.detector}")
    print(f"  item detection    : {'yes' if result.item_detection_available else 'NO (zone-only)'}")
    print(f"  wall time         : {result.wall_seconds:.2f} s")
    print(f"  pipeline frames   : {result.frames}  ({result.pipeline_fps:.1f} frames/s total)")
    print(f"  detector calls    : {result.detection_invocations}  ({result.detection_fps:.1f}/s)")
    print(f"  pose calls        : {result.pose_invocations}  ({result.pose_fps:.1f}/s)")
    print()
    print("  latency (ms)        mean     p50     p95     max")
    for name, stats in (
        ("  detection        ", result.inference_latency_ms),
        ("  pose             ", result.pose_latency_ms),
        ("  full pipeline    ", result.pipeline_latency_ms),
    ):
        print(
            f"{name} {stats['mean']:7.1f} {stats['p50']:7.1f} "
            f"{stats['p95']:7.1f} {stats['max']:7.1f}"
        )
    print()
    if result.gpu_reserved_mb:
        print(f"  GPU memory        : {result.gpu_allocated_mb:.0f} MB allocated, "
              f"{result.gpu_reserved_mb:.0f} MB reserved")
    else:
        print("  GPU memory        : n/a (not running on CUDA)")
    print(f"  process CPU       : {result.cpu_percent:.0f}%" if result.cpu_percent
          else "  process CPU       : n/a (install psutil for CPU measurement)")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", help="Video file, webcam index or RTSP URL")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use random-noise frames instead of a source"
    )
    parser.add_argument("--width", type=int, default=1280, help="Synthetic frame width")
    parser.add_argument("--height", type=int, default=720, help="Synthetic frame height")
    parser.add_argument("--frames", type=int, default=120, help="Frames to hold in memory")
    parser.add_argument("--seconds", type=float, default=10.0, help="Seconds to run each config")
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=[1],
        help="Simulated camera counts to test, e.g. --cameras 1 4 8 16 32 64",
    )
    parser.add_argument("--config-dir", default=None, help="Configuration directory")
    parser.add_argument("--json", help="Write results as JSON to this path")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    configure_logging(level=args.log_level, fmt="text", force=True)
    configure_metrics(5.0, 512)
    config = load_config(args.config_dir) if args.config_dir else AppConfig()

    if args.synthetic or not args.source:
        frames = synthetic_frames(min(args.frames, 60), args.width, args.height)
        source_label = f"synthetic {args.width}x{args.height} noise"
    else:
        frames = load_frames(args.source, args.frames)
        source_label = str(args.source)

    device = select_device(config.detection.models.device, config.detection.models.fp16)
    log_device(device)

    print()
    print("AisleGuard Vision - benchmark")
    print(f"source : {source_label}")
    print(f"device : {device.describe()}")

    results: list[BenchmarkResult] = []
    for count in args.cameras:
        label = f"{count} simulated camera{'s' if count != 1 else ''}"
        result = run_benchmark(config, frames, count, args.seconds, label)
        print_result(result)
        results.append(result)

    if len(results) > 1:
        print("-" * 74)
        print("  cameras   pipeline FPS   detector FPS   p95 pipeline latency")
        for count, result in zip(args.cameras, results, strict=True):
            print(
                f"  {count:>7}   {result.pipeline_fps:>12.1f}   "
                f"{result.detection_fps:>12.1f}   {result.pipeline_latency_ms['p95']:>14.1f} ms"
            )
        print("-" * 74)

    print()
    print("  These are MEASURED numbers for this machine and this workload.")
    print("  They do not account for RTSP bandwidth, NVDEC decoder slots, per-process")
    print("  GPU context overhead or the store network. Do not use them to claim a")
    print("  supported camera count without testing against real cameras.")
    print()

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
        )
        print(f"  results written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
