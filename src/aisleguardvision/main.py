"""AisleGuard Vision command-line entry point.

    python -m aisleguardvision.main --source 0 --display
    python -m aisleguardvision.main --source ./data/samples/store.mp4 --display
    python -m aisleguardvision.main --source "rtsp://..." --display
    python -m aisleguardvision.main --headless --api

With no ``--source``, every enabled camera in ``config/cameras.yaml`` is run.
``--source`` overrides that with a single ad-hoc camera, which is the fastest
path to seeing the system work on a laptop.

This module owns the CLI, the display window and process lifecycle. All
analysis lives in :class:`~aisleguardvision.pipeline.CameraPipeline`, so the
same pipeline runs identically headless on an edge node.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

from . import __version__
from .api.state import RuntimeState, set_state
from .camera.manager import CameraManager
from .camera.worker import CameraDecodeWorker
from .core.config import AppConfig, CameraConfig, ConfigError, load_config
from .core.logging import configure_logging, get_logger
from .core.metrics import MetricNames, configure_metrics
from .core.types import SecurityEvent
from .events.dispatcher import AlertDispatcher, set_default_dispatcher
from .events.recorder import IncidentRecorder, IncidentStore
from .inference.backend import create_detector_backend
from .inference.detector import PersonDetector
from .inference.device import log_device, select_device
from .inference.pose import PoseEstimator
from .inference.product_detector import create_product_detector
from .pipeline import CameraPipeline, PipelineResult
from .utils.sanitize import sanitize_source

logger = get_logger(__name__)

BANNER = r"""
    _    _    _         ____                     _
   / \  (_)__| | ___   / ___|_   _  __ _ _ __ __| |
  / _ \ | / __| |/ _ \ | |  _| | | |/ _` | '__/ _` |
 / ___ \| \__ \ |  __/ | |_| | |_| | (_| | | | (_| |
/_/   \_\_|___/_|\___|  \____|\__,_|\__,_|_|  \__,_|   V I S I O N

Behavioral risk signals for human review. Not a theft determination.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aisleguardvision",
        description="AisleGuard Vision - retail loss-prevention behavioral analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m aisleguardvision.main --source 0 --display\n"
            "  python -m aisleguardvision.main --source ./data/samples/store.mp4 --display\n"
            '  python -m aisleguardvision.main --source "rtsp://user:pass@host/stream" --headless\n'
            "  python -m aisleguardvision.main --headless --api\n"
        ),
    )
    source = parser.add_argument_group("source")
    source.add_argument(
        "--source",
        help="Webcam index (0), video file path, or RTSP/HTTP URL. "
        "Overrides config/cameras.yaml with a single ad-hoc camera.",
    )
    source.add_argument(
        "--camera-id", default="cam_cli", help="Id to use for an ad-hoc --source camera"
    )
    source.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Run only this configured camera id (repeatable)",
    )
    source.add_argument(
        "--loop", action="store_true", help="Loop a file source instead of stopping at the end"
    )

    display = parser.add_argument_group("display")
    display.add_argument("--display", action="store_true", help="Show the annotated video window")
    display.add_argument(
        "--headless",
        action="store_true",
        help="Never open a window (default when --display is absent)",
    )
    display.add_argument(
        "--no-zones-overlay", action="store_true", help="Do not draw configured zones"
    )
    display.add_argument("--record-output", help="Write the annotated video to this path (mp4)")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--config-dir", default=None, help="Configuration directory")
    runtime.add_argument(
        "--device", default=None, help="auto | cuda | cuda:0 | mps | cpu (overrides config)"
    )
    runtime.add_argument("--detector", default=None, help="Detector weights (overrides config)")
    runtime.add_argument("--pose-model", default=None, help="Pose weights (overrides config)")
    runtime.add_argument(
        "--product-model", default=None, help="Custom retail merchandise model weights"
    )
    runtime.add_argument(
        "--no-pose", action="store_true", help="Disable pose estimation (detection only)"
    )
    runtime.add_argument(
        "--detection-fps", type=float, default=None, help="Override the detection rate"
    )
    runtime.add_argument(
        "--alert-threshold", type=float, default=None, help="Override the alert threshold (0-100)"
    )
    runtime.add_argument("--api", action="store_true", help="Start the FastAPI service")
    runtime.add_argument("--api-port", type=int, default=None, help="API port")
    runtime.add_argument(
        "--no-record", action="store_true", help="Disable incident clip and snapshot recording"
    )
    runtime.add_argument(
        "--max-seconds", type=float, default=0.0, help="Exit after this long (0 = run forever)"
    )

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("--log-level", default=None, help="DEBUG | INFO | WARNING | ERROR")
    logging_group.add_argument("--log-format", default=None, choices=["text", "json"])
    logging_group.add_argument("--log-file", default=None, help="Also write logs to this file")
    logging_group.add_argument(
        "--version", action="version", version=f"AisleGuard Vision {__version__}"
    )
    return parser


class Application:
    """Owns process lifecycle: cameras, pipelines, recording, alerting, API."""

    def __init__(self, config: AppConfig, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.display = bool(args.display and not args.headless)
        self._stop = threading.Event()
        self._pipelines: dict[str, CameraPipeline] = {}
        self._writers: dict[str, object] = {}
        self.events_seen: list[SecurityEvent] = []

        self.metrics = configure_metrics(
            config.metrics.rate_window_seconds, config.metrics.histogram_size
        )
        self.device = select_device(config.detection.models.device, config.detection.models.fp16)
        log_device(self.device)

        self.camera_manager = CameraManager(config, self.metrics)
        self.recorder: IncidentRecorder | None = None
        if config.recording.enabled and not args.no_record:
            self.recorder = IncidentRecorder(config.recording, config.storage, self.metrics)
        self.dispatcher = AlertDispatcher(config.alerts, metrics=self.metrics)
        set_default_dispatcher(self.dispatcher)

        self.state = RuntimeState(
            config=config,
            camera_manager=self.camera_manager,
            incident_store=IncidentStore(config.storage),
            metrics=self.metrics,
            device_label=f"{self.device.device} ({self.device.name})",
        )
        set_state(self.state)

        self._detector: PersonDetector | None = None
        self._pose: PoseEstimator | None = None
        self._overlay = None

    # -- setup -------------------------------------------------------------
    def build_models(self) -> None:
        models = self.config.detection.models
        inference = self.config.detection.inference

        backend = create_detector_backend(models, inference, pose=False, device=self.device)
        self._detector = PersonDetector(backend, self.config.detection, self.metrics)

        if not self.args.no_pose:
            pose_backend = create_detector_backend(models, inference, pose=True, device=self.device)
            self._pose = PoseEstimator(pose_backend, self.config.detection, self.metrics)

        if models.warmup and self._detector.is_operational:
            self._detector.warmup()
            if self._pose is not None and self._pose.is_operational:
                self._pose.warmup()

        if not self._detector.is_operational:
            logger.warning(
                "running without a detection model: no people will be detected. "
                "Install the inference extra (pip install -r requirements-inference.txt) "
                "and ensure the weights are downloadable."
            )

    def build_pipelines(self, cameras: list[CameraConfig]) -> None:
        assert self._detector is not None
        product_detector = create_product_detector(
            self.config.detection.models,
            self.config.detection,
            self._detector.backend,
            self.device,
        )
        self.state.item_detection_available = product_detector.provides_merchandise_detection

        for camera in cameras:
            worker = self.camera_manager.worker(camera.id)
            if worker is None:
                continue
            self._pipelines[camera.id] = CameraPipeline(
                camera_id=camera.id,
                config=self.config,
                detector=self._detector,
                pose_estimator=self._pose,
                product_detector=product_detector,
                zones=self.camera_manager.zones(camera.id),
                recorder=self.recorder,
                dispatcher=self.dispatcher,
                buffer=self.camera_manager.buffer(camera.id),
                metrics=self.metrics,
            )

    # -- run ---------------------------------------------------------------
    def run(self, cameras: list[CameraConfig]) -> int:
        if not cameras:
            logger.error(
                "no cameras to run. Provide --source, or enable a camera in "
                "config/cameras.yaml and export its ${CAM_*_RTSP} variable."
            )
            return 2

        self.build_models()
        self.build_pipelines(cameras)

        if self.recorder is not None:
            self.recorder.start()
        self.dispatcher.start()
        if self.args.api:
            from .api.server import start_api_thread

            start_api_thread(self.state)

        self.camera_manager.start_all()

        if self.display:
            self._init_display()

        logger.info(
            "AisleGuard Vision running",
            extra={
                "fields": {
                    "cameras": ",".join(c.id for c in cameras),
                    "display": self.display,
                    "alert_threshold": self.config.behavior.alert_threshold,
                    "recording": self.recorder is not None,
                }
            },
        )

        try:
            self._loop(cameras)
        except KeyboardInterrupt:  # pragma: no cover - interactive
            logger.info("interrupted by user")
        finally:
            self.shutdown()
        return 0

    def _init_display(self) -> None:
        from .visualization.overlay import OverlayOptions, OverlayRenderer

        options = OverlayOptions(draw_zones=not self.args.no_zones_overlay)
        self._overlay = OverlayRenderer(options)
        if self.recorder is not None:
            # Reuse the same renderer for incident media so a saved clip looks
            # like what the operator saw live.
            self.recorder.annotator = self._annotate_for_incident

    def _annotate_for_incident(self, image, event: SecurityEvent):
        from .visualization.overlay import draw_incident_banner

        return draw_incident_banner(
            image, f"{event.threat_level.value}  risk {event.risk_score:.0f}  {event.camera_id}"
        )

    def _loop(self, cameras: list[CameraConfig]) -> None:
        """Main analysis loop.

        Single-threaded across cameras on purpose for the MVP: it keeps the
        control flow obvious and, at one to a handful of cameras, the GPU is the
        bottleneck rather than the loop. Fan-out to GPU worker processes is the
        documented next step (``docs/SCALING.md``) and needs no change here --
        each worker runs this same loop over its own camera subset.
        """
        deadline = time.time() + self.args.max_seconds if self.args.max_seconds > 0 else None
        last_stats = time.time()
        stats_interval = self.config.logging.stats_interval_seconds

        while not self._stop.is_set():
            if deadline is not None and time.time() >= deadline:
                logger.info("--max-seconds reached; stopping")
                break

            processed_any = False
            for camera in cameras:
                worker = self.camera_manager.worker(camera.id)
                pipeline = self._pipelines.get(camera.id)
                if worker is None or pipeline is None or not worker.is_enabled:
                    continue

                # Latest-wins: if we fell behind, skip to the newest frame
                # rather than working through a backlog of stale ones.
                frame = worker.queue.get_latest(timeout=0.05)
                if frame is None:
                    continue

                processed_any = True
                result = pipeline.process(frame)
                self.events_seen.extend(result.events)
                self._publish(camera.id, worker, pipeline, result)

                if self.display and not self._render(camera.id, result, worker, pipeline):
                    self._stop.set()
                    break

            if not processed_any:
                # Nothing to do; yield rather than spin.
                self._stop.wait(0.005)

            if time.time() - last_stats >= stats_interval:
                self._log_stats()
                last_stats = time.time()

            if self._all_sources_exhausted(cameras):
                logger.info("all file sources reached their end; stopping")
                break

    def _all_sources_exhausted(self, cameras: list[CameraConfig]) -> bool:
        """True once every file-backed camera has stopped producing frames."""
        from .camera.stream import is_live_source, resolve_source

        relevant = [c for c in cameras if not is_live_source(resolve_source(c.source))]
        if not relevant or self.args.loop:
            return False
        for camera in relevant:
            worker = self.camera_manager.worker(camera.id)
            if worker is None:
                continue
            status = worker.status()
            if status.state.is_healthy or worker.queue.depth > 0:
                return False
            if status.frames_received == 0:
                return False
        return True

    def _publish(
        self,
        camera_id: str,
        worker: CameraDecodeWorker,
        pipeline: CameraPipeline,
        result: PipelineResult,
    ) -> None:
        self.state.publish_camera_stats(
            camera_id,
            inference_fps=self.metrics.rate(
                MetricNames.INFERENCE_FPS, {"camera_id": camera_id}
            ).value,
            pose_fps=self.metrics.rate(MetricNames.POSE_FPS, {"camera_id": camera_id}).value,
            active_tracks=len(result.tracks),
            queue_depth=worker.queue.depth,
        )

    def _render(
        self,
        camera_id: str,
        result: PipelineResult,
        worker: CameraDecodeWorker,
        pipeline: CameraPipeline,
    ) -> bool:
        """Draw and show one frame. Returns False when the user asked to quit."""
        import cv2

        from .visualization.overlay import OverlayStats

        assert self._overlay is not None
        stats = OverlayStats(
            camera_id=camera_id,
            video_fps=self.metrics.rate(MetricNames.DECODE_FPS, {"camera_id": camera_id}).value,
            detection_fps=self.metrics.rate(
                MetricNames.INFERENCE_FPS, {"camera_id": camera_id}
            ).value,
            pose_fps=self.metrics.rate(MetricNames.POSE_FPS, {"camera_id": camera_id}).value,
            inference_latency_ms=pipeline.last_inference_latency_ms,
            pose_latency_ms=pipeline.last_pose_latency_ms,
            tracks=len(result.tracks),
            item_tracks=len(result.items),
            queue_depth=worker.queue.depth,
            dropped=worker.queue.dropped,
            device=self.device.device,
            alerts=pipeline.alerts_generated,
            mode_note=(
                ""
                if pipeline.item_detection_available
                else "Zone-only mode: no merchandise detector; risk capped"
            ),
        )
        canvas = self._overlay.render(
            result.frame.image,
            zones=pipeline.zones.zones,
            tracks=result.tracks,
            items=result.items,
            observations=result.observations,
            stats=stats,
        )

        if self.args.record_output:
            self._write_output(camera_id, canvas)

        try:
            cv2.imshow(f"AisleGuard Vision - {camera_id}", canvas)
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as exc:
            # A headless OpenCV build has no GUI support. Say so clearly and
            # keep processing rather than crashing.
            logger.error(
                "display unavailable (headless OpenCV build?); continuing without a window. "
                "Install opencv-python for GUI support: pip install -r requirements-display.txt",
                extra={"fields": {"error": str(exc)}},
            )
            self.display = False
            return True
        return key not in (ord("q"), 27)

    def _write_output(self, camera_id: str, canvas) -> None:
        import cv2

        writer = self._writers.get(camera_id)
        if writer is None:
            height, width = canvas.shape[:2]
            path = Path(self.args.record_output)
            if len(self._pipelines) > 1:
                path = path.with_name(f"{path.stem}_{camera_id}{path.suffix}")
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*self.config.recording.codec),
                self.config.recording.clip_fps,
                (width, height),
            )
            self._writers[camera_id] = writer
            logger.info("writing annotated output", extra={"fields": {"path": str(path)}})
        writer.write(canvas)

    def _log_stats(self) -> None:
        for camera_id, pipeline in self._pipelines.items():
            worker = self.camera_manager.worker(camera_id)
            if worker is None:
                continue
            status = worker.status()
            logger.info(
                "pipeline statistics",
                extra={
                    "fields": {
                        "camera_id": camera_id,
                        "state": status.state.value,
                        "decode_fps": round(status.decode_fps, 1),
                        "detection_fps": round(
                            self.metrics.rate(
                                MetricNames.INFERENCE_FPS, {"camera_id": camera_id}
                            ).value,
                            1,
                        ),
                        "pose_fps": round(
                            self.metrics.rate(MetricNames.POSE_FPS, {"camera_id": camera_id}).value,
                            1,
                        ),
                        "inference_ms": round(pipeline.last_inference_latency_ms, 1),
                        "tracks": pipeline.person_tracker.active_count,
                        "queue": worker.queue.depth,
                        "dropped": worker.queue.dropped,
                        "reconnects": status.reconnect_count,
                        "alerts": pipeline.alerts_generated,
                    }
                },
            )

    # -- shutdown ----------------------------------------------------------
    def request_stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        logger.info("shutting down")
        self.camera_manager.stop_all()
        if self.recorder is not None:
            self.recorder.stop()
        self.dispatcher.stop()
        set_default_dispatcher(None)

        for writer in self._writers.values():
            try:
                writer.release()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                pass

        if self._detector is not None:
            self._detector.close()
        if self._pose is not None:
            self._pose.close()

        if self.display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:  # pragma: no cover
                pass

        logger.info(
            "shutdown complete",
            extra={"fields": {"alerts_generated": len(self.events_seen)}},
        )


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI overrides on top of the loaded configuration."""
    models = config.detection.models
    if args.device:
        models.device = args.device
    if args.detector:
        models.detector = args.detector
    if args.pose_model:
        models.pose = args.pose_model
    if args.product_model:
        models.product = args.product_model
    if args.detection_fps:
        config.detection.scheduler.detection_fps = args.detection_fps
    if args.alert_threshold is not None:
        config.behavior.alert_threshold = args.alert_threshold
        config.risk.high_risk_threshold = args.alert_threshold
    if args.api_port:
        config.api.port = args.api_port
    if args.api:
        config.api.enabled = True
    if args.no_record:
        config.recording.enabled = False
    return config


def resolve_cameras(config: AppConfig, args: argparse.Namespace) -> list[CameraConfig]:
    """Decide which cameras to run."""
    if args.source:
        camera = CameraConfig(
            id=args.camera_id,
            name=f"CLI source ({sanitize_source(args.source)})",
            source=args.source,
            enabled=True,
            loop_file_source=args.loop,
            # Play recorded footage at its native rate when it is being watched.
            pace_file_source=bool(args.display and not args.headless),
            # Any zones configured for the first camera are reused, so
            # `--source file.mp4` can exercise a real zone setup.
            zones=config.cameras.cameras[0].zones if config.cameras.cameras else [],
        )
        return [camera]

    cameras = config.enabled_cameras()
    if args.cameras:
        wanted = set(args.cameras)
        cameras = [c for c in cameras if c.id in wanted]
        missing = wanted - {c.id for c in cameras}
        if missing:
            logger.warning(
                "requested cameras are not enabled or not configured",
                extra={"fields": {"missing": ",".join(sorted(missing))}},
            )

    skipped = [c.id for c in config.cameras.cameras if c.enabled and not c.is_resolvable()]
    if skipped:
        logger.warning(
            "skipping cameras whose source environment variable is unset",
            extra={"fields": {"cameras": ",".join(skipped)}},
        )
    return cameras


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config_dir)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(
        level=args.log_level or config.logging.level,
        fmt=args.log_format or config.logging.format,
        log_file=args.log_file or config.logging.file,
        force=True,
    )
    config = apply_overrides(config, args)

    if not args.headless:
        print(BANNER)

    cameras = resolve_cameras(config, args)
    if args.source:
        # An ad-hoc camera is not in the loaded config; register it so that
        # per-camera lookups (zones, scheduling overrides) resolve.
        config.cameras.cameras.append(cameras[0])

    app = Application(config, args)

    def _signal_handler(signum, _frame) -> None:
        logger.info("signal received; stopping", extra={"fields": {"signal": signum}})
        app.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass

    for camera in cameras:
        if app.camera_manager.worker(camera.id) is None:
            app.camera_manager.add_camera(camera)

    return app.run([c for c in cameras if app.camera_manager.worker(c.id) is not None])


if __name__ == "__main__":
    raise SystemExit(main())
