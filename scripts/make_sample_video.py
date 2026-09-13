#!/usr/bin/env python3
"""Render a simulation scenario to an MP4, for end-to-end testing.

The repository ships no video: sample security footage is large, and real store
footage of real shoppers is not something to commit to a git repository.

This renders a *synthetic* scene from the same scenarios the behavior tests
use, producing a file that exercises the full decode -> detect -> track ->
behave path. The figures are schematic stick figures, so a real detector will
find little or nothing in them -- that is expected. Use this to verify plumbing
(decode, buffering, overlay, recording, shutdown), and real footage to verify
detection quality.

    python scripts/make_sample_video.py
    python scripts/make_sample_video.py --scenario PHONE_INTERACTION --fps 25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aisleguardvision.core.types import COCO_SKELETON, COCO_KEYPOINT_ORDER  # noqa: E402
from aisleguardvision.simulation.scenarios import (  # noqa: E402
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SCENARIOS,
    SimFrame,
    get_scenario,
)

FLOOR_COLOR = (48, 44, 42)
WALL_COLOR = (78, 72, 68)
SHELF_COLOR = (120, 110, 100)
PERSON_COLOR = (210, 200, 190)
ITEM_COLOR = (70, 160, 240)


def render_frame(frame: SimFrame, zones) -> np.ndarray:
    """Draw one synthetic scene frame."""
    image = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), WALL_COLOR, dtype=np.uint8)
    cv2.rectangle(image, (0, 520), (FRAME_WIDTH, FRAME_HEIGHT), FLOOR_COLOR, -1)

    for zone in zones:
        polygon = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(image, [polygon], SHELF_COLOR)
        cv2.polylines(image, [polygon], True, (150, 140, 130), 2)
        anchor = np.array(zone.polygon, dtype=int).min(axis=0)
        cv2.putText(
            image,
            zone.id,
            (int(anchor[0]) + 8, int(anchor[1]) + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    index = {name: i for i, name in enumerate(COCO_KEYPOINT_ORDER)}
    for person in frame.persons:
        x1, y1, x2, y2 = person.bbox.as_int_xyxy()
        cv2.rectangle(image, (x1, y1), (x2, y2), (90, 90, 90), 1)
        points = person.keypoints
        for start, end in COCO_SKELETON:
            a = points[index[start]]
            b = points[index[end]]
            cv2.line(
                image,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                PERSON_COLOR,
                6,
                cv2.LINE_AA,
            )
        for row in points:
            cv2.circle(image, (int(row[0]), int(row[1])), 5, PERSON_COLOR, -1, cv2.LINE_AA)
        head = points[index[COCO_KEYPOINT_ORDER[0]]]
        cv2.circle(image, (int(head[0]), int(head[1])), 26, PERSON_COLOR, -1, cv2.LINE_AA)

    for item in frame.items:
        ix1, iy1, ix2, iy2 = item.bbox.as_int_xyxy()
        cv2.rectangle(image, (ix1, iy1), (ix2, iy2), ITEM_COLOR, -1)
        cv2.rectangle(image, (ix1, iy1), (ix2, iy2), (250, 250, 250), 2)

    cv2.putText(
        image,
        "SYNTHETIC TEST FOOTAGE - not real store video",
        (12, FRAME_HEIGHT - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--scenario", default="POSSIBLE_CONCEALMENT", choices=sorted(SCENARIOS), help="Scenario"
    )
    parser.add_argument("--output", default="data/samples/store.mp4", help="Output path")
    parser.add_argument("--fps", type=float, default=20.0, help="Output frame rate")
    parser.add_argument("--codec", default="mp4v", help="FourCC codec")
    parser.add_argument(
        "--all", action="store_true", help="Render every scenario into data/samples/"
    )
    args = parser.parse_args(argv)

    targets = sorted(SCENARIOS) if args.all else [args.scenario]
    for name in targets:
        scenario = get_scenario(name)
        output = (
            Path("data/samples") / f"{name.lower()}.mp4"
            if args.all
            else Path(args.output)
        )
        output.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*args.codec),
            args.fps,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        if not writer.isOpened():
            print(f"error: could not open a video writer for {output}", file=sys.stderr)
            return 1
        for frame in scenario.frames:
            writer.write(render_frame(frame, scenario.zones))
        writer.release()
        print(f"wrote {output}  ({len(scenario.frames)} frames @ {args.fps:g} FPS)  [{name}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
