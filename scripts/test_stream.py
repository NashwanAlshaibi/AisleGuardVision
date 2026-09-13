#!/usr/bin/env python3
"""Check a camera source and calibrate its zones. No models required.

Use this before anything else when adding a camera: it verifies that the source
opens, reports the resolution and real frame rate, and draws the configured
zone polygons so you can see whether they land where you think they do.

    python scripts/test_stream.py --source 0
    python scripts/test_stream.py --source ./data/samples/store.mp4 --zones
    python scripts/test_stream.py --source "$CAM_001_RTSP" --camera-id cam_001 --zones
    python scripts/test_stream.py --source 0 --pick-zone     # click out a polygon

``--pick-zone`` prints YAML you can paste straight into config/cameras.yaml.

Credentials are never printed: the source is sanitized before display.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aisleguardvision.behavior.zones import ZoneRegistry  # noqa: E402
from aisleguardvision.camera.stream import VideoStream  # noqa: E402
from aisleguardvision.core.config import CameraConfig, load_config  # noqa: E402
from aisleguardvision.core.logging import configure_logging  # noqa: E402
from aisleguardvision.utils.sanitize import sanitize_source  # noqa: E402
from aisleguardvision.visualization.overlay import ZONE_COLORS  # noqa: E402

_clicked: list[tuple[int, int]] = []


def _on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked.append((x, y))
        print(f"  vertex {len(_clicked)}: [{x}, {y}]")
    elif event == cv2.EVENT_RBUTTONDOWN and _clicked:
        removed = _clicked.pop()
        print(f"  removed vertex {removed}")


def draw_zones(image: np.ndarray, registry: ZoneRegistry) -> None:
    overlay = image.copy()
    for zone in registry:
        color = ZONE_COLORS.get(zone.kind, (235, 180, 70))
        polygon = zone.polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)
        anchor = zone.polygon.astype(int).min(axis=0)
        cv2.putText(
            image,
            f"{zone.kind.value}: {zone.zone_id}",
            (int(anchor[0]) + 6, int(anchor[1]) + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)


def emit_zone_yaml(points: list[tuple[int, int]], zone_id: str, kind: str) -> None:
    if len(points) < 3:
        print("\n  need at least 3 vertices to define a zone\n")
        return
    print()
    print("  Paste into config/cameras.yaml under the camera's `zones:` list:")
    print()
    print(f"      - id: {zone_id}")
    print(f"        kind: {kind}")
    print(f"        name: {zone_id}")
    print("        polygon:")
    for x, y in points:
        print(f"          - [{x}, {y}]")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True, help="Webcam index, file path or RTSP URL")
    parser.add_argument("--camera-id", default="cam_test", help="Camera id (for zone lookup)")
    parser.add_argument("--config-dir", default=None, help="Configuration directory")
    parser.add_argument("--zones", action="store_true", help="Draw this camera's configured zones")
    parser.add_argument("--pick-zone", action="store_true", help="Click out a new zone polygon")
    parser.add_argument("--zone-kind", default="shelf", help="Kind for --pick-zone output")
    parser.add_argument("--zone-id", default="shelf_new", help="Id for --pick-zone output")
    parser.add_argument("--seconds", type=float, default=20.0, help="How long to run")
    parser.add_argument("--headless", action="store_true", help="Do not open a window")
    parser.add_argument("--save-frame", help="Write one frame to this path and exit")
    args = parser.parse_args(argv)

    configure_logging(level="INFO", fmt="text", force=True)

    registry = ZoneRegistry(args.camera_id)
    if args.zones or args.pick_zone:
        config = load_config(args.config_dir)
        camera = config.camera(args.camera_id)
        if camera is not None:
            registry = ZoneRegistry.from_config(args.camera_id, camera.zones)
            print(f"  {registry.describe()}")
        else:
            print(f"  no camera {args.camera_id!r} in config; no zones to draw")

    camera_config = CameraConfig(id=args.camera_id, source=args.source, enabled=True)
    stream = VideoStream(camera_config)
    if not stream.open():
        print(f"error: could not open {sanitize_source(args.source)}", file=sys.stderr)
        return 1

    info = stream.info
    print()
    print(f"  source     : {sanitize_source(args.source)}")
    print(f"  resolution : {info.width}x{info.height}")
    print(f"  source fps : {info.fps if info.fps else 'unknown (reported nothing usable)'}")
    print(f"  backend    : {info.backend or 'default'}")
    print(f"  live       : {stream.is_live}")
    print()

    window = f"AisleGuard stream test - {args.camera_id}"
    interactive = not args.headless and not args.save_frame
    if interactive:
        try:
            cv2.namedWindow(window)
            if args.pick_zone:
                cv2.setMouseCallback(window, _on_mouse)
                print("  left-click to add a vertex, right-click to undo, 'w' to write YAML, "
                      "'q' to quit")
        except cv2.error:
            print("  display unavailable (headless OpenCV build); continuing without a window")
            interactive = False

    deadline = time.time() + args.seconds
    frames = 0
    started = time.time()
    try:
        while time.time() < deadline:
            ok, image = stream.read()
            if not ok or image is None:
                if not stream.is_live:
                    print("  end of file")
                    break
                if stream.should_retry() and not stream.reconnect():
                    continue
                continue
            frames += 1

            if args.save_frame:
                cv2.imwrite(args.save_frame, image)
                print(f"  wrote {args.save_frame}")
                break

            if not interactive:
                continue

            canvas = image.copy()
            if args.zones:
                draw_zones(canvas, registry)
            for index, point in enumerate(_clicked):
                cv2.circle(canvas, point, 5, (60, 60, 245), -1)
                if index:
                    cv2.line(canvas, _clicked[index - 1], point, (60, 60, 245), 2)
            if len(_clicked) > 2:
                cv2.line(canvas, _clicked[-1], _clicked[0], (60, 60, 245), 1)

            measured = frames / max(1e-6, time.time() - started)
            cv2.putText(
                canvas,
                f"{image.shape[1]}x{image.shape[0]}  {measured:.1f} FPS  frames={frames}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("w") and args.pick_zone:
                emit_zone_yaml(_clicked, args.zone_id, args.zone_kind)
    finally:
        stream.close()
        if interactive:
            cv2.destroyAllWindows()

    elapsed = time.time() - started
    print()
    print(f"  frames read : {frames}")
    print(f"  measured fps: {frames / elapsed:.2f}" if elapsed > 0 else "")
    print(f"  reconnects  : {stream.reconnect_count}")
    print()
    if args.pick_zone and _clicked:
        emit_zone_yaml(_clicked, args.zone_id, args.zone_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
