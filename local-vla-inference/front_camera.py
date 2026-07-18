"""Front-camera sources for G1D ACT inference.

The G1's front camera is wired by USB to the onboard dev PC and served by
Unitree's **teleimager** (github.com/unitreerobotics/teleimager), which is
already running on the robot. It publishes raw JPEG frames over ZMQ PUB
(head camera default port 55555) and serves its camera config over ZMQ
REQ/REP on port 60000.

Sources, selected via ``--camera``:

- ``teleimager://HOST``  — query config on :60000, subscribe to the head
  camera stream, crop the left eye if binocular. (default)
- ``zmq://HOST:PORT``    — subscribe to an explicit teleimager ZMQ port.
- ``opencv:N``           — camera attached to THIS machine, OpenCV index N.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TELEIMAGER_CONFIG_PORT = 60000
TELEIMAGER_HEAD_PORT = 55555


class TeleimagerFrontCamera:
    """Head camera from Unitree's teleimager server (config on :60000)."""

    def __init__(self, host: str, timeout_s: float = 5.0) -> None:
        self.host = host
        self.timeout_s = timeout_s
        self.binocular = False
        self._zmq_cam: ZMQFrontCamera | None = None

    def _fetch_config(self) -> dict | None:
        import zmq

        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.REQ)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(f"tcp://{self.host}:{TELEIMAGER_CONFIG_PORT}")
        try:
            s.send(b"GET_DATA")
            if s.poll(int(self.timeout_s * 1000)):
                return s.recv_json()
            logger.warning(
                "teleimager config request to %s:%d timed out; assuming head port %d",
                self.host, TELEIMAGER_CONFIG_PORT, TELEIMAGER_HEAD_PORT,
            )
            return None
        finally:
            s.close(0)

    def connect(self) -> None:
        port = TELEIMAGER_HEAD_PORT
        config = self._fetch_config()
        if config is not None:
            head = config["head_camera"]
            if not head.get("enable_zmq", True):
                raise RuntimeError(
                    "teleimager head camera has enable_zmq: false — enable it in "
                    "cam_config_server.yaml on the robot."
                )
            port = int(head["zmq_port"])
            self.binocular = bool(head.get("binocular", False))
            logger.info(
                "teleimager head camera: port=%d shape=%s binocular=%s",
                port, head.get("image_shape"), self.binocular,
            )
        self._zmq_cam = ZMQFrontCamera(self.host, port, timeout_s=self.timeout_s)
        self._zmq_cam.connect()

    def read(self) -> np.ndarray:
        frame = self._zmq_cam.read()
        if self.binocular:
            frame = frame[:, : frame.shape[1] // 2]  # left eye only
        return frame

    def read_resized(self, height: int, width: int) -> np.ndarray:
        return _resize(self.read(), height, width)

    def disconnect(self) -> None:
        if self._zmq_cam is not None:
            self._zmq_cam.disconnect()
        self._zmq_cam = None


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
    """Parse a --camera spec: ``teleimager://HOST``, ``zmq://HOST:PORT``, or ``opencv:N``."""
    if spec.startswith("teleimager://"):
        host = spec[len("teleimager://") :]
        if not host:
            raise ValueError(f"Bad teleimager camera spec: {spec} (expected teleimager://HOST)")
        return TeleimagerFrontCamera(host)
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
