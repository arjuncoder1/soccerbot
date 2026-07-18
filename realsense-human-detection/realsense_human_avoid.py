"""
Intel RealSense D435 + YOLO human detection with analog directional guidance.

When a detected person is within DISTANCE_THRESHOLD meters, the script computes
the person's horizontal angle offset from the camera's optical center (using the
color stream's intrinsics) and turns that into a continuous, analog-style
directional instruction (exact degrees + a clock-face position + a graded
"how hard to turn" word), instead of a flat left/right binary.

Requirements:
    pip install pyrealsense2 opencv-python numpy ultralytics
    # Install a CUDA-enabled torch build for your RTX 3050 Ti (CUDA 12.1 example):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Model:
    Uses yolov8n.pt (COCO). Class 0 = "person". The .pt file auto-downloads
    on first run via ultralytics if not already present. Runs on CUDA with
    FP16 inference automatically when a GPU is available, falling back to
    CPU/FP32 otherwise.
"""

import time

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DISTANCE_THRESHOLD_M = 1.0      # trigger avoidance guidance at/below this range
FRAME_W, FRAME_H = 640, 480
FPS = 30
MODEL_PATH = "yolov8n.pt"
PERSON_CLASS_ID = 0             # COCO class id for "person"
CONF_THRESHOLD = 0.5
DEPTH_PATCH = 5                 # half-size (px) of the sampling patch around bbox center


def angle_to_clock(angle_deg: float) -> str:
    """Map a signed angle (0 = straight ahead, + = right, - = left) to a clock
    position, the way pilots/drivers say "target at 2 o'clock". 30 deg/hour,
    with fractional hours kept (e.g. 1:30) for finer, more analog resolution
    than a plain left/right label."""
    hour = (angle_deg / 30.0) + 12.0
    hour %= 12.0
    if hour == 0:
        hour = 12.0
    h = int(hour)
    m = int(round((hour - h) * 60))
    if m == 60:
        m = 0
        h = 12 if h == 11 else h + 1
    if h == 0:
        h = 12
    return f"{h}:{m:02d}"


def direction_instruction(angle_deg: float) -> str:
    """Turn a signed horizontal angle into graded, continuous guidance rather
    than a binary left/right decision."""
    abs_angle = abs(angle_deg)
    clock = angle_to_clock(angle_deg)

    if abs_angle < 3:
        return f"Person dead ahead ({clock}, {angle_deg:+.1f}deg) -> back straight up"

    move_toward = "left" if angle_deg > 0 else "right"  # step away from the person

    if abs_angle < 10:
        magnitude = "slightly"
    elif abs_angle < 25:
        magnitude = "moderately"
    elif abs_angle < 45:
        magnitude = "sharply"
    else:
        magnitude = "fully"

    return f"Person at {clock} ({angle_deg:+.1f}deg) -> move {magnitude} {move_toward}"


def median_depth_m(depth_frame: rs.depth_frame, cx: int, cy: int, patch: int = DEPTH_PATCH) -> float:
    """Median distance (meters) over a small patch around (cx, cy), skipping
    zero/invalid depth samples. More robust than a single-pixel read."""
    samples = []
    for dy in range(-patch, patch + 1):
        for dx in range(-patch, patch + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < FRAME_W and 0 <= y < FRAME_H:
                d = depth_frame.get_distance(x, y)
                if d > 0:
                    samples.append(d)
    if not samples:
        return 0.0
    return float(np.median(samples))


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FPS)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    focal_x = intr.fx
    principal_x = intr.ppx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"  # FP16 inference only makes sense on GPU
    print(f"Running YOLO on device={device}, fp16={use_half}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_PATH)
    model.to(device)

    prev_time = time.time()
    fps = 0.0

    cv2.namedWindow("RealSense D435 - Human Detection", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            results = model(color_image, device=device, half=use_half, verbose=False)[0]

            active_instruction = None

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if cls_id != PERSON_CLASS_ID or conf < CONF_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                distance = median_depth_m(depth_frame, cx, cy)
                if distance <= 0:
                    continue

                color = (0, 255, 0)
                label = f"person {distance:.2f}m"

                if distance <= DISTANCE_THRESHOLD_M:
                    angle_deg = np.degrees(np.arctan2(cx - principal_x, focal_x))
                    instruction = direction_instruction(angle_deg)
                    color = (0, 0, 255)
                    label = f"{distance:.2f}m | {instruction}"
                    print(instruction)
                    if active_instruction is None or distance < active_instruction[1]:
                        active_instruction = (instruction, distance, cx, cy)

                cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                cv2.circle(color_image, (cx, cy), 4, color, -1)
                cv2.putText(color_image, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center_x = FRAME_W // 2
            cv2.line(color_image, (center_x, 0), (center_x, FRAME_H), (255, 255, 0), 1)

            # FPS counter
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Alert banner + pointer arrow for the nearest in-range person
            if active_instruction is not None:
                instruction, distance, cx, cy = active_instruction
                cv2.rectangle(color_image, (0, FRAME_H - 30), (FRAME_W, FRAME_H), (0, 0, 255), -1)
                cv2.putText(color_image, instruction, (8, FRAME_H - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                arrow_dir = -1 if "left" in instruction else 1
                arrow_y = 40
                cv2.arrowedLine(color_image, (center_x, arrow_y),
                                 (center_x + arrow_dir * 60, arrow_y),
                                 (0, 0, 255), 4, tipLength=0.4)

            # Depth view side-by-side with the color/detection view
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            preview = np.hstack((color_image, depth_colormap))

            cv2.imshow("RealSense D435 - Human Detection", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
