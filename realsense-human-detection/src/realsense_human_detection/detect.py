"""One-frame human detector: RGB(+optional depth) → boxes + guidance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from realsense_human_detection.depth import median_depth_m_from_array
from realsense_human_detection.guidance import direction_instruction

PERSON_CLASS_ID = 0  # COCO
DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONF = 0.5
DEFAULT_DISTANCE_THRESHOLD_M = 1.0
DEFAULT_DEPTH_PATCH = 5


@dataclass(frozen=True)
class PersonDetection:
    """One person found in a single frame."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    center: tuple[int, int]  # cx, cy
    distance_m: float | None
    angle_deg: float | None
    instruction: str | None
    in_range: bool

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.bbox


class HumanDetector:
    """Load YOLO once; call ``detect`` per frame.

    Depth and camera intrinsics are optional. Without them you still get
    bounding boxes and confidence; with them you also get distance, bearing,
    and avoidance instruction for people within ``distance_threshold_m``.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        conf_threshold: float = DEFAULT_CONF,
        distance_threshold_m: float = DEFAULT_DISTANCE_THRESHOLD_M,
        depth_patch: int = DEFAULT_DEPTH_PATCH,
        device: str | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_half = device == "cuda"
        self.conf_threshold = conf_threshold
        self.distance_threshold_m = distance_threshold_m
        self.depth_patch = depth_patch

        self._model = YOLO(str(model_path))
        self._model.to(device)

    def detect(
        self,
        color_bgr: np.ndarray,
        *,
        depth_m: np.ndarray | None = None,
        fx: float | None = None,
        ppx: float | None = None,
    ) -> list[PersonDetection]:
        """Detect humans in one BGR frame.

        Parameters
        ----------
        color_bgr:
            HxWx3 uint8 BGR image (OpenCV / RealSense color).
        depth_m:
            Optional HxW float meters aligned to color. Invalid pixels = 0.
        fx, ppx:
            Color-camera intrinsics for horizontal bearing. If omitted with
            depth present, angle/instruction are left None.

        Returns
        -------
        list[PersonDetection]
            Sorted nearest-first when distance is known, else by confidence.
        """
        if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
            raise ValueError(f"color_bgr must be HxWx3, got {color_bgr.shape}")

        results = self._model(
            color_bgr,
            device=self.device,
            half=self.use_half,
            verbose=False,
        )[0]

        people: list[PersonDetection] = []
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return people

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != PERSON_CLASS_ID or conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            distance: float | None = None
            angle: float | None = None
            instruction: str | None = None
            in_range = False

            if depth_m is not None:
                d = median_depth_m_from_array(
                    depth_m, cx, cy, patch=self.depth_patch
                )
                if d > 0:
                    distance = d
                    if d <= self.distance_threshold_m and fx is not None and ppx is not None:
                        angle = float(np.degrees(np.arctan2(cx - ppx, fx)))
                        instruction = direction_instruction(angle)
                        in_range = True

            people.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(cx, cy),
                    distance_m=distance,
                    angle_deg=angle,
                    instruction=instruction,
                    in_range=in_range,
                )
            )

        people.sort(
            key=lambda p: (
                p.distance_m if p.distance_m is not None else float("inf"),
                -p.confidence,
            )
        )
        return people

    def nearest_in_range(self, detections: list[PersonDetection]) -> PersonDetection | None:
        """Nearest person that triggered avoidance guidance, or None."""
        in_range = [d for d in detections if d.in_range]
        return in_range[0] if in_range else None

    def annotate(
        self,
        color_bgr: np.ndarray,
        detections: list[PersonDetection],
    ) -> np.ndarray:
        """Draw boxes/labels on a copy of the frame (for debug UIs)."""
        import cv2

        out = color_bgr.copy()
        h, w = out.shape[:2]
        cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 0, 255) if det.in_range else (0, 255, 0)
            if det.distance_m is not None and det.instruction:
                label = f"{det.distance_m:.2f}m | {det.instruction}"
            elif det.distance_m is not None:
                label = f"person {det.distance_m:.2f}m"
            else:
                label = f"person {det.confidence:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.circle(out, det.center, 4, color, -1)
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        nearest = self.nearest_in_range(detections)
        if nearest is not None and nearest.instruction:
            cv2.rectangle(out, (0, h - 30), (w, h), (0, 0, 255), -1)
            cv2.putText(
                out,
                nearest.instruction,
                (8, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
        return out


def detect_humans(
    color_bgr: np.ndarray,
    *,
    depth_m: np.ndarray | None = None,
    fx: float | None = None,
    ppx: float | None = None,
    model_path: str | Path = DEFAULT_MODEL,
    **kwargs: Any,
) -> list[PersonDetection]:
    """Convenience: construct a detector and run one frame (loads model each call)."""
    return HumanDetector(model_path, **kwargs).detect(
        color_bgr, depth_m=depth_m, fx=fx, ppx=ppx
    )
