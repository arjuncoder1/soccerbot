# realsense-human-detection

Portable package: give it one RGB frame (optional depth + intrinsics) and get
person bounding boxes, distance, bearing, and avoidance guidance.

## Install

From the repo root (workspace member):

```bash
./realsense-human-detection/install.sh
```

If the pyrealsense2 wheel is missing for your platform:

```bash
.venv/bin/python realsense-human-detection/scripts/build_librealsense.py
```

## One-frame API (no camera required)

```python
from realsense_human_detection import HumanDetector

detector = HumanDetector()  # loads YOLO once
detections = detector.detect(color_bgr)  # list[PersonDetection]

# With aligned depth (meters) + color intrinsics:
detections = detector.detect(color_bgr, depth_m=depth_m, fx=fx, ppx=ppx)
for d in detections:
    print(d.bbox, d.confidence, d.distance_m, d.angle_deg, d.instruction)
```

`PersonDetection` fields: `bbox` `(x1,y1,x2,y2)`, `confidence`, `center`,
`distance_m`, `angle_deg`, `instruction`, `in_range`.

## Live RealSense loop

```bash
uv run realsense-human-detect
# or
.venv/bin/python realsense-human-detection/main.py
```
