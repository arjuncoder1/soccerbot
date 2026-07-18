"""Live RealSense + YOLO loop (optional hardware CLI)."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from realsense_human_detection.detect import DEFAULT_MODEL, HumanDetector


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live RealSense human detection + avoidance guidance.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="YOLO weights path or hub id.")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--distance", type=float, default=1.0, help="Avoidance range in meters.")
    p.add_argument("--no-window", action="store_true", help="Headless: print only.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        import pyrealsense2 as rs
    except ImportError as e:
        raise SystemExit(
            "pyrealsense2 not found — no prebuilt wheel for this platform.\n"
            "Run the build pyrealsense script:\n"
            "  .venv/bin/python realsense-human-detection/scripts/build_librealsense.py"
        ) from e

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()

    detector = HumanDetector(
        args.model,
        conf_threshold=args.conf,
        distance_threshold_m=args.distance,
    )
    print(f"YOLO device={detector.device} fp16={detector.use_half}")

    if not args.no_window:
        cv2.namedWindow("RealSense - Human Detection", cv2.WINDOW_AUTOSIZE)

    prev_time = time.time()
    fps = 0.0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            # Convert RealSense z16 (mm) to meters for the portable API.
            depth_m = depth_raw.astype(np.float32) * depth_frame.get_units()

            detections = detector.detect(
                color_image,
                depth_m=depth_m,
                fx=intr.fx,
                ppx=intr.ppx,
            )

            for det in detections:
                if det.instruction:
                    print(det.instruction)

            if args.no_window:
                continue

            annotated = detector.annotate(color_image, detections)
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imshow("RealSense - Human Detection", np.hstack((annotated, depth_colormap)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
