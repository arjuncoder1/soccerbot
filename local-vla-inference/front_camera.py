"""Front-camera sources for G1D ACT inference.

The G1's front camera is a RealSense D435i wired by USB to an onboard
computer. Unitree's stock stack does NOT expose it over DDS (``videohub`` /
``VideoClient`` is a Go2 service — GetImageSample fails with 3102 on G1).

Sources, selected via ``--camera``:

- ``zmq://HOST:PORT`` — Unitree teleop ``image_server`` already running on the
  robot (same one used for data collection). ZMQ SUB of JPEG frames.
- ``opencv:N``        — camera attached to THIS machine, OpenCV index N.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ZMQFrontCamera:
    """Subscribe to Unitree's teleop image_server (ZMQ PUB of JPEG frames)."""

    def __init__(self, host: str, port: int, timeout_s: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._socket = None
        self._context = None

    def connect(self) -> None:
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.CONFLATE, 1)  # always take the newest frame
        self._socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        self._socket.connect(f"tcp://{self.host}:{self.port}")
        # Fail fast if the image_server isn't reachable.
        self.read()
        logger.info("Connected to image_server at tcp://%s:%d", self.host, self.port)

    def read(self) -> np.ndarray:
        import zmq

        if self._socket is None:
            raise RuntimeError("ZMQ camera not connected")
        try:
            data = self._socket.recv()
        except zmq.Again as e:
            raise RuntimeError(
                f"No frame from image_server tcp://{self.host}:{self.port} "
                f"within {self.timeout_s}s. Is Unitree's teleop image_server running on the robot?"
            ) from e
        # image_server sends raw JPEG bytes (optionally with a small header; JPEG starts at FFD8).
        buf = np.frombuffer(data, dtype=np.uint8)
        start = 0
        if len(buf) > 2 and not (buf[0] == 0xFF and buf[1] == 0xD8):
            marker = bytes(data).find(b"\xff\xd8")
            if marker < 0:
                raise RuntimeError("image_server frame is not a JPEG")
            start = marker
        bgr = cv2.imdecode(buf[start:], cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("Failed to decode image_server JPEG")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_resized(self, height: int, width: int) -> np.ndarray:
        return _resize(self.read(), height, width)

    def disconnect(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
        if self._context is not None:
            self._context.term()
        self._socket = None
        self._context = None


class OpenCVFrontCamera:
    """Camera plugged into THIS machine (e.g. RealSense RGB as a UVC device)."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._cap = None

    def connect(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open OpenCV camera index {self.index} on this machine. "
                "List devices with `ls /dev/video*` and try other indices."
            )
        self._cap = cap
        logger.info("Opened local OpenCV camera index %d", self.index)

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("OpenCV camera not connected")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"Failed to read frame from OpenCV camera {self.index}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def read_resized(self, height: int, width: int) -> np.ndarray:
        return _resize(self.read(), height, width)

    def disconnect(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None


def _resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def make_front_camera(spec: str):
    """Parse a --camera spec: ``zmq://HOST:PORT`` or ``opencv:N``."""
    if spec.startswith("zmq://"):
        hostport = spec[len("zmq://") :]
        host, _, port = hostport.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError(f"Bad zmq camera spec: {spec} (expected zmq://HOST:PORT)")
        return ZMQFrontCamera(host, int(port))
    if spec.startswith("opencv:"):
        idx = spec[len("opencv:") :]
        if not idx.lstrip("-").isdigit():
            raise ValueError(f"Bad opencv camera spec: {spec} (expected opencv:N)")
        return OpenCVFrontCamera(int(idx))
    raise ValueError(f"Unknown camera spec: {spec} (use zmq://HOST:PORT or opencv:N)")
