"""Teleimager (G1 head-camera) + YOLO human detector.

Same public API as ``realsense-human-detection/realsense_human_avoid.py``'s
``HumanDetector`` so ``avoid.py`` can drop this in unchanged, but the frame
source is the G1's onboard head camera served by Unitree's teleimager over
ZMQ (already running on the robot).

We subscribe directly to the teleimager RGB ZMQ port and use the full
1280x720 frame -- the sibling ``TeleimagerFrontCamera`` in local-vla-inference
crops to just the left eye for ACT inference, which throws away half the
field of view we need for people avoidance.

Distance is estimated from the person's bounding-box height using the
pinhole model:

    dist_m = (person_height_m * focal_px) / bbox_height_px

``focal_px`` defaults to the RealSense-reported color intrinsics fy
(~906 px at 1280x720) which we fetch once at connect time from the
teleimager config server on port 60000. Override with ``focal_px=`` if
needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REALSENSE_DIR = REPO_ROOT / "realsense-human-detection"

logger = logging.getLogger(__name__)

DEFAULT_TELEIMAGER_HOST = "192.168.123.164"
TELEIMAGER_CONFIG_PORT = 60000
TELEIMAGER_HEAD_PORT = 55555
PERSON_CLASS_ID = 0  # COCO
CONF_THRESHOLD = 0.5
DEFAULT_FOCAL_PX = 906.0  # RealSense fy at 1280x720; overridden by config if reachable
DEFAULT_PERSON_HEIGHT_M = 1.7
DEFAULT_MODEL_PATH = str(REALSENSE_DIR / "yolov8n.pt")


@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]
    distance_m: float
    confidence: float


@dataclass
class DetectionSnapshot:
    color_image: np.ndarray  # HxWx3 RGB (full 1280x720)
    detections: list[PersonDetection] = field(default_factory=list)


class HumanDetector:
    """Teleimager RGB + YOLO human detector.

    API-compatible with the RealSense HumanDetector used by ``avoid.py``:
    supports context-manager usage and ``poll_nearest_person(within_m=...)``.
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_TELEIMAGER_HOST,
        model_path: str = DEFAULT_MODEL_PATH,
        device: str | None = None,
        use_half: bool | None = None,
        conf_threshold: float = CONF_THRESHOLD,
        focal_px: float | None = None,
        person_height_m: float = DEFAULT_PERSON_HEIGHT_M,
        recv_timeout_s: float = 5.0,
    ) -> None:
        import torch  # deferred so import failure surfaces at construction, not module load
        self.host = host
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_half = self.device == "cuda" if use_half is None else use_half
        self.conf_threshold = conf_threshold
        self.focal_px_override = focal_px
        self.focal_px = focal_px if focal_px is not None else DEFAULT_FOCAL_PX
        self.person_height_m = person_height_m
        self.recv_timeout_s = recv_timeout_s

        self._ctx = None
        self._sub = None
        self._model = None

    # -- lifecycle ----------------------------------------------------------

    def _fetch_config(self) -> dict | None:
        """Ask teleimager on :60000 for the head-camera config (port, intrinsics)."""
        import zmq

        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{self.host}:{TELEIMAGER_CONFIG_PORT}")
        try:
            s.send(b"GET_DATA")
            if s.poll(int(self.recv_timeout_s * 1000)):
                return s.recv_json()
            logger.warning(
                "teleimager config request to %s:%d timed out; using defaults",
                self.host, TELEIMAGER_CONFIG_PORT,
            )
            return None
        finally:
            s.close(0)

    def open(self) -> "HumanDetector":
        if self._sub is not None:
            return self
        import zmq
        from ultralytics import YOLO

        head_port = TELEIMAGER_HEAD_PORT
        config = self._fetch_config()
        if config is not None:
            head = config.get("head_camera", {})
            if not head.get("enable_zmq", True):
                raise RuntimeError(
                    "teleimager head camera has enable_zmq: false -- enable it in "
                    "cam_config_server.yaml on the robot."
                )
            head_port = int(head.get("zmq_port", TELEIMAGER_HEAD_PORT))
            if self.focal_px_override is None:
                intr = head.get("color_intrinsics") or head.get("intrinsics")
                if intr:
                    # Use fy (vertical focal length) since we measure bbox HEIGHT.
                    self.focal_px = float(intr.get("fy", self.focal_px))
            logger.info(
                "teleimager head camera: port=%d shape=%s binocular=%s focal_px=%.1f (fy)",
                head_port,
                head.get("image_shape"),
                head.get("binocular"),
                self.focal_px,
            )

        self._ctx = zmq.Context()
        sub = self._ctx.socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.CONFLATE, 1)  # always the newest frame
        sub.setsockopt(zmq.RCVTIMEO, int(self.recv_timeout_s * 1000))
        sub.connect(f"tcp://{self.host}:{head_port}")
        self._sub = sub

        # Fail fast if the image stream isn't live.
        frame = self._read_frame()
        if frame is None:
            raise RuntimeError(
                f"No frame from teleimager tcp://{self.host}:{head_port} "
                f"within {self.recv_timeout_s}s"
            )
        logger.info("Teleimager RGB stream connected: frame shape=%s", frame.shape)

        model = YOLO(self.model_path)
        model.to(self.device)
        logger.info(
            "YOLO loaded (%s) on device=%s fp16=%s",
            self.model_path, self.device, self.use_half,
        )
        self._model = model
        return self

    def close(self) -> None:
        if self._sub is not None:
            try:
                self._sub.close(0)
            except Exception:  # noqa: BLE001 -- teardown best-effort
                pass
            self._sub = None
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:  # noqa: BLE001
                pass
            self._ctx = None
        self._model = None

    def __enter__(self) -> "HumanDetector":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- polling ------------------------------------------------------------

    def _read_frame(self) -> Optional[np.ndarray]:
        """Grab one JPEG payload off the ZMQ SUB socket and decode to RGB."""
        import zmq

        if self._sub is None:
            raise RuntimeError("teleimager subscriber not open")
        try:
            data = self._sub.recv()
        except zmq.Again:
            return None
        buf = np.frombuffer(data, dtype=np.uint8)
        start = 0
        if len(buf) > 2 and not (buf[0] == 0xFF and buf[1] == 0xD8):
            marker = bytes(data).find(b"\xff\xd8")
            if marker < 0:
                logger.warning("teleimager frame is not a JPEG (%d bytes)", len(data))
                return None
            start = marker
        bgr = cv2.imdecode(buf[start:], cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("failed to decode teleimager JPEG (%d bytes)", len(data))
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _estimate_distance_m(self, bbox_h_px: int) -> float:
        if bbox_h_px <= 0:
            return 0.0
        return (self.person_height_m * self.focal_px) / float(bbox_h_px)

    def poll_snapshot(self) -> Optional[DetectionSnapshot]:
        if self._sub is None or self._model is None:
            raise RuntimeError("HumanDetector.poll_snapshot() before open()")
        frame_rgb = self._read_frame()
        if frame_rgb is None:
            return None

        results = self._model(
            frame_rgb, device=self.device, half=self.use_half, verbose=False,
        )[0]
        detections: list[PersonDetection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != PERSON_CLASS_ID or conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            distance = self._estimate_distance_m(y2 - y1)
            if distance <= 0:
                continue
            detections.append(PersonDetection(
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                distance_m=distance,
                confidence=conf,
            ))
        return DetectionSnapshot(color_image=frame_rgb, detections=detections)

    def poll_nearest_person(self, within_m: float | None = None) -> Optional[float]:
        snap = self.poll_snapshot()
        if snap is None or not snap.detections:
            return None
        nearest = min(d.distance_m for d in snap.detections)
        if within_m is not None and nearest > within_m:
            return None
        return nearest

    def iter_nearest_person(self, within_m: float | None = None) -> Iterator[Optional[float]]:
        while True:
            yield self.poll_nearest_person(within_m=within_m)


# ---------------------------------------------------------------------------
# Standalone smoke test.
#   python human_detector_teleimager.py --host 192.168.123.164 --secs 10
#   python human_detector_teleimager.py --host 192.168.123.164 --secs 5 --save-annotated logs/det.jpg
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse
    import logging as _logging
    import time

    p = argparse.ArgumentParser(description="Teleimager + YOLO human detector smoke test.")
    p.add_argument("--host", default=DEFAULT_TELEIMAGER_HOST, help="Robot IP running teleimager.")
    p.add_argument("--secs", type=float, default=10.0)
    p.add_argument("--within-m", type=float, default=None, help="Only print people closer than this.")
    p.add_argument("--focal-px", type=float, default=None,
                   help="Override focal_px (fetched from teleimager config otherwise).")
    p.add_argument(
        "--save-annotated",
        default=None,
        metavar="PATH",
        help="On each poll save frame with bboxes/distances drawn to PATH (overwritten).",
    )
    args = p.parse_args()

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.monotonic()
    with HumanDetector(host=args.host, focal_px=args.focal_px) as det:
        while time.monotonic() - t0 < args.secs:
            snap = det.poll_snapshot()
            if snap is None:
                continue
            if not snap.detections:
                logger.info("no person")
            else:
                for d in snap.detections:
                    if args.within_m is not None and d.distance_m > args.within_m:
                        continue
                    logger.info(
                        "person at ~%.2f m (bbox=%dx%d, conf=%.2f)",
                        d.distance_m,
                        d.bbox[2] - d.bbox[0],
                        d.bbox[3] - d.bbox[1],
                        d.confidence,
                    )
            if args.save_annotated:
                vis = cv2.cvtColor(snap.color_image, cv2.COLOR_RGB2BGR)
                for d in snap.detections:
                    x1, y1, x2, y2 = d.bbox
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f"{d.distance_m:.2f}m", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                cv2.imwrite(args.save_annotated, vis)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
