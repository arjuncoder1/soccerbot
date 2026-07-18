"""Run ACT inference on Unitree G1D arms only (no hands / fingers).

Policy: ``myx160/unitree_lerobot_act_g1d_16d_001`` (16-D). We command the
14 arm joints and zero-pad / ignore the last 2 dims.

Example (on the robot machine, after ``./install.sh``):

    export CYCLONEDDS_HOME=$HOME/cyclonedds/install
    uv run --package local-vla-inference python local-vla-inference/main.py \\
        --robot-ip 192.168.123.164
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ACT inference for G1D arms-only (16-D policy, no hands).")
    p.add_argument("--policy", default=DEFAULT_POLICY_ID, help="Hub repo id or local checkpoint path.")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available).")
    p.add_argument("--robot-ip", default="192.168.123.164", help="G1 DDS / camera host IP.")
    p.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run (0 = forever).")
    p.add_argument(
        "--simulation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use LeRobot UnitreeG1 simulation mode.",
    )
    p.add_argument(
        "--camera-port",
        type=int,
        default=5555,
        help="ZMQ image server port on the robot (shared by all cameras).",
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


def build_robot(args: argparse.Namespace):
    from lerobot.cameras.zmq import ZMQCameraConfig
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config

    h, w, _ = IMAGE_SHAPE
    cameras = {
        cam: ZMQCameraConfig(
            server_address=args.robot_ip,
            port=args.camera_port,
            camera_name=cam,
            width=w,
            height=h,
            fps=int(args.fps),
        )
        for cam in CAMERA_KEYS
    }
    cfg = UnitreeG1Config(
        id="g1d_act",
        is_simulation=args.simulation,
        robot_ip=args.robot_ip,
        cameras=cameras,
        controller=None,
        gravity_compensation=False,
    )
    return UnitreeG1(cfg)


def pack_observation(robot_obs: dict[str, Any]) -> dict[str, Any]:
    """Map UnitreeG1 arm + camera obs; pad unused dims with 0 (no hands)."""
    out: dict[str, Any] = {}
    for joint in ARM_JOINTS:
        key = f"{joint}.q"
        if key not in robot_obs:
            raise KeyError(f"Missing arm joint in robot observation: {key}")
        out[key] = float(robot_obs[key])
    for pad in UNUSED_PAD:
        out[pad] = 0.0
    for cam in CAMERA_KEYS:
        if cam not in robot_obs:
            raise KeyError(f"Missing camera frame: {cam}")
        out[cam] = robot_obs[cam]
    return out


def send_arm_action(robot, action: dict[str, float]) -> None:
    arm_only = {f"{j}.q": action[f"{j}.q"] for j in ARM_JOINTS}
    robot.send_action(arm_only)


def dry_run(policy, preprocess, postprocess, device: torch.device) -> None:
    import numpy as np

    from lerobot.policies.utils import build_inference_frame, make_robot_action

    features = dataset_features()
    fake_obs = {f"{j}.q": 0.0 for j in ARM_JOINTS}
    for pad in UNUSED_PAD:
        fake_obs[pad] = 0.0
    for cam in CAMERA_KEYS:
        fake_obs[cam] = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    frame = build_inference_frame(observation=fake_obs, ds_features=features, device=device)
    batch = preprocess(frame)
    action = policy.select_action(batch)
    action = postprocess(action)
    robot_action = make_robot_action(action, features)
    arm_keys = [f"{j}.q" for j in ARM_JOINTS]
    logger.info("Dry-run OK. Arm action dims=%d (hands ignored)", len(arm_keys))
    logger.info("Sample arm action: %s", {k: round(robot_action[k], 4) for k in arm_keys[:4]})


def run(args: argparse.Namespace) -> None:
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    device = resolve_device(args.device)
    logger.info("Loading ACT policy %s on %s (arms only)", args.policy, device)

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
    robot.connect()

    dt = 1.0 / args.fps
    t0 = time.time()
    step = 0
    logger.info("Starting ACT arm control loop at %.1f Hz (Ctrl+C to stop)", args.fps)

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
