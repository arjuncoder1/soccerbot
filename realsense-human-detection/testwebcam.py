#!/usr/bin/env python3
"""Test HumanDetector on a laptop webcam (no RealSense / no depth).

Opens the default camera, runs one-frame person detection each tick, draws
boxes + confidence. Press q to quit.

Usage:

    .venv/bin/python realsense-human-detection/testwebcam.py
    .venv/bin/python realsense-human-detection/testwebcam.py --camera 1 --conf 0.4
"""

from __future__ import annotations

import argparse
import time

import cv2

from realsense_human_detection import HumanDetector
from realsense_human_detection.detect import DEFAULT_MODEL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Webcam smoke-test for HumanDetector.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (default 0).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path or hub id.")
    p.add_argument("--conf", type=float, default=0.5, help="Min detection confidence.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: auto).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    detector = HumanDetector(
        args.model,
        conf_threshold=args.conf,
        device=args.device,
    )
    print(f"webcam={args.camera} device={detector.device} — press q to quit")

    prev = time.time()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame; quitting.")
                break

            people = detector.detect(frame)
            annotated = detector.annotate(frame, people)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}  people: {len(people)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            for p in people:
                x1, y1, x2, y2 = p.bbox
                print(
                    f"person conf={p.confidence:.2f} bbox=({x1},{y1},{x2},{y2}) center={p.center}",
                    flush=True,
                )

            cv2.imshow("testwebcam — HumanDetector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
