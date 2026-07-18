"""Run ACT inference on Unitree G1D arms only (no hands / fingers).

Policy: ``myx160/unitree_lerobot_act_g1d_16d_001`` (16-D). We command the
14 arm joints and zero-pad / ignore the last 2 dims.

Camera: **one remote Unitree front cam over ZMQ** (ImageServer on the robot).
That frame is copied into all 4 policy image inputs.

On the robot (separate terminal), start the image server, e.g.:

    python -m lerobot.robots.unitree_g1.run_g1_server --camera --camera-device 4

Then on the client:

    export CYCLONEDDS_HOME=$HOME/cyclonedds/install
    ./local-vla-inference/run.sh --robot-ip 192.168.123.164
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import torch

from embodiment_g1d_16d import (
    ARM_JOINTS,
    CAMERA_KEYS,
    DEFAULT_POLICY_ID,
    IMAGE_SHAPE,
    UNUSED_PAD,
    dataset_features,
)

logger = logging.getLogger(__name__)

# Physical remote cam key on the robot (must match ImageServer publish name).
FRONT_CAMERA_KEY = "front"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACT inference for G1D arms-only — remote Unitree front ZMQ camera."
    )
    p.add_argument("--policy", default=DEFAULT_POLICY_ID, help="Hub repo id or local checkpoint path.")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available).")
    p.add_argument("--robot-ip", default="192.168.123.164", help="G1 robot IP (DDS).")
    p.add_argument(
        "--camera-host",
        default=None,
        help="ZMQ image server host (default: same as --robot-ip).",
    )
    p.add_argument("--camera-port", type=int, default=5555, help="ZMQ image server port.")
    p.add_argument(
        "--camera-name",
        default="head_camera",
        help="Camera name inside the ZMQ JSON payload (Unitree default: head_camera).",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run (0 = forever).")
    p.add_argument(
        "--simulation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LeRobot UnitreeG1 simulation mode.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load policy and print one fake forward pass; do not connect hardware.",
    )
    return p.parse_args(argv)


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resize_front_frame(frame: np.ndarray) -> np.ndarray:
    import cv2

    h, w, _ = IMAGE_SHAPE
    if frame.shape[0] != h or frame.shape[1] != w:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return frame


def build_robot(args: argparse.Namespace):
    from lerobot.cameras.zmq import ZMQCameraConfig
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config

    h, w, _ = IMAGE_SHAPE
    cam_host = args.camera_host or args.robot_ip
    cameras = {
        FRONT_CAMERA_KEY: ZMQCameraConfig(
            server_address=cam_host,
            port=args.camera_port,
            camera_name=args.camera_name,
            width=w,
            height=h,
            fps=int(args.fps),
            timeout_ms=10_000,
            warmup_s=5,
        )
    }
    cfg = UnitreeG1Config(
        id="g1d_act",
        is_simulation=args.simulation,
        robot_ip=args.robot_ip,
        cameras=cameras,
        controller=None,
        gravity_compensation=False,
    )
    logger.info(
        "Remote front cam: zmq://%s:%s name=%s → policy keys %s",
        cam_host,
        args.camera_port,
        args.camera_name,
        CAMERA_KEYS,
    )
    return UnitreeG1(cfg)


def pack_observation(robot_obs: dict[str, Any]) -> dict[str, Any]:
    """Map arms + remote front cam; replicate into all policy image keys."""
    out: dict[str, Any] = {}
    for joint in ARM_JOINTS:
        key = f"{joint}.q"
        if key not in robot_obs:
            raise KeyError(f"Missing arm joint in robot observation: {key}")
        out[key] = float(robot_obs[key])
    for pad in UNUSED_PAD:
        out[pad] = 0.0

    if FRONT_CAMERA_KEY not in robot_obs:
        raise KeyError(f"Missing remote front camera frame ({FRONT_CAMERA_KEY}).")
    front = resize_front_frame(np.asarray(robot_obs[FRONT_CAMERA_KEY]))
    for cam in CAMERA_KEYS:
        out[cam] = front
    return out


def send_arm_action(robot, action: dict[str, float]) -> None:
    arm_only = {f"{j}.q": action[f"{j}.q"] for j in ARM_JOINTS}
    robot.send_action(arm_only)


def dry_run(policy, preprocess, postprocess, device: torch.device) -> None:
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    features = dataset_features()
    fake_obs = {f"{j}.q": 0.0 for j in ARM_JOINTS}
    for pad in UNUSED_PAD:
        fake_obs[pad] = 0.0
    blank = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    for cam in CAMERA_KEYS:
        fake_obs[cam] = blank

    frame = build_inference_frame(observation=fake_obs, ds_features=features, device=device)
    batch = preprocess(frame)
    action = policy.select_action(batch)
    action = postprocess(action)
    robot_action = make_robot_action(action, features)
    arm_keys = [f"{j}.q" for j in ARM_JOINTS]
    logger.info("Dry-run OK. Arm dims=%d; remote front cam → %s", len(arm_keys), CAMERA_KEYS)


def run(args: argparse.Namespace) -> None:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    device = resolve_device(args.device)
    logger.info("Loading ACT policy %s on %s", args.policy, device)

    policy = ACTPolicy.from_pretrained(args.policy)
    policy.to(device)
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy,
    )
    features = dataset_features()

    if args.dry_run:
        dry_run(policy, preprocess, postprocess, device)
        return

    robot = build_robot(args)
    cam_host = args.camera_host or args.robot_ip
    try:
        robot.connect()
    except Exception:
        logger.error(
            "Failed to connect. Is the robot ImageServer running?\n"
            "  On the G1:  python -m lerobot.robots.unitree_g1.run_g1_server "
            "--camera --camera-device 4 --camera-port %s\n"
            "  Then:       ./local-vla-inference/run.sh --robot-ip %s --camera-host %s",
            args.camera_port,
            args.robot_ip,
            cam_host,
        )
        raise

    dt = 1.0 / args.fps
    t0 = time.time()
    step = 0
    logger.info("ACT loop @ %.1f Hz (Ctrl+C to stop)", args.fps)

    try:
        while True:
            loop_start = time.perf_counter()
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                logger.info("Duration reached (%.1fs); stopping", args.duration)
                break

            obs = pack_observation(robot.get_observation())
            frame = build_inference_frame(observation=obs, ds_features=features, device=device)
            batch = preprocess(frame)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            robot_action = make_robot_action(action, features)
            send_arm_action(robot, robot_action)

            step += 1
            if step % int(args.fps) == 0:
                logger.info("step=%d elapsed=%.1fs", step, time.time() - t0)

            sleep = dt - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        robot.disconnect()
        logger.info("Disconnected")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
