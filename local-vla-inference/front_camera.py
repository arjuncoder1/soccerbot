"""Unitree front camera via the robot's existing VideoClient service.

Uses ``unitree_sdk2py.go2.video.VideoClient`` (works on G1 for many setups).
No ImageServer / OpenCV / extra processes on the robot.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class UnitreeFrontCamera:
    """Grab JPEG frames from the Unitree onboard front camera over DDS/RPC."""

    def __init__(self, timeout_s: float = 3.0) -> None:
        self.timeout_s = timeout_s
        self._client = None

    def connect(self) -> None:
        from unitree_sdk2py.go2.video.video_client import VideoClient

        client = VideoClient()
        client.SetTimeout(self.timeout_s)
        client.Init()
        # Probe once so we fail fast if the robot video service is down.
        code, _data = client.GetImageSample()
        if code != 0:
            raise RuntimeError(
                f"Unitree VideoClient.GetImageSample failed (code={code}). "
                "Robot front-camera service must already be running (factory Unitree stack)."
            )
        self._client = client
        logger.info("Connected to Unitree front camera (VideoClient)")

    def read(self) -> np.ndarray:
        if self._client is None:
            raise RuntimeError("Front camera not connected")
        code, data = self._client.GetImageSample()
        if code != 0 or not data:
            raise RuntimeError(f"GetImageSample failed (code={code})")
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode front camera JPEG")
        # LeRobot / policy path expects RGB.
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_resized(self, height: int, width: int) -> np.ndarray:
        frame = self.read()
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        return frame

    def disconnect(self) -> None:
        self._client = None
