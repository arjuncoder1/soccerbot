"""Run ACT inference on Unitree G1D arms only (no hands / fingers).

Policy: ``myx160/unitree_lerobot_act_g1d_16d_001`` (16-D). We command the
14 arm joints via the official ``rt/arm_sdk`` DDS topic and zero-pad /
ignore the last 2 dims.

Everything runs on this machine:
  - state:   subscribe ``rt/lowstate`` (DDS)
  - arms:    publish ``rt/arm_sdk`` (DDS, weight joint 29)
  - camera:  Unitree ``teleimager`` already running on the robot
             (``--camera teleimager://192.168.123.164``, head cam on :55555)

Usage:

    export CYCLONEDDS_HOME=$HOME/cyclonedds/install
    ./local-vla-inference/run.sh --iface eth0
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
from front_camera import make_front_camera
from g1_arms import G1Arms

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACT inference for G1D arms-only via direct DDS (rt/arm_sdk)."
    )
    p.add_argument("--policy", default=DEFAULT_POLICY_ID, help="Hub repo id or local checkpoint path.")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available).")
    p.add_argument(
        "--iface",
        default=None,
        help="Network interface connected to the robot (e.g. eth0, enp2s0). "
        "Omit to use the DDS default.",
    )
    p.add_argument(
        "--camera",
        default="teleimager://192.168.123.164",
        help="Front camera source: 'teleimager://HOST' (Unitree teleimager on the robot; "
        "auto-detects head-cam port/binocular via :60000), 'zmq://HOST:PORT', "
        "or 'opencv:N' (camera on this machine).",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run (0 = forever).")
    p.add_argument("--kp", type=float, default=60.0, help="Arm joint position gain.")
    p.add_argument("--kd", type=float, default=1.5, help="Arm joint damping gain.")
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


def pack_observation(arm_obs: dict[str, float], front_rgb: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for joint in ARM_JOINTS:
        key = f"{joint}.q"
        if key not in arm_obs:
            raise KeyError(f"Missing arm joint in observation: {key}")
        out[key] = arm_obs[key]
    for pad in UNUSED_PAD:
        out[pad] = 0.0
    for cam in CAMERA_KEYS:
        out[cam] = front_rgb
    return out


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
    logger.info("Dry-run OK. Arm action dims=%d", len(arm_keys))
    logger.info("Sample arm action: %s", {k: round(robot_action[k], 4) for k in arm_keys[:4]})


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

    # One DDS init per process; shared by arms + camera clients.
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    arms = G1Arms(kp=args.kp, kd=args.kd)
    front = make_front_camera(args.camera)

    arms.connect()
    front.connect()

    # Engage arm_sdk smoothly at the current pose before the policy takes over.
    arms.hold_current_pose(ramp_s=2.0)

    h, w, _ = IMAGE_SHAPE
    dt = 1.0 / args.fps
    t0 = time.time()
    step = 0
    logger.info("ACT loop @ %.1f Hz via rt/arm_sdk (Ctrl+C to stop)", args.fps)

    try:
        while True:
            loop_start = time.perf_counter()
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                logger.info("Duration reached (%.1fs); stopping", args.duration)
                break

            front_rgb = front.read_resized(h, w)
            obs = pack_observation(arms.get_arm_positions(), front_rgb)
            frame = build_inference_frame(observation=obs, ds_features=features, device=device)
            batch = preprocess(frame)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            robot_action = make_robot_action(action, features)
            arms.send_arm_positions(robot_action)

            step += 1
            if step % int(args.fps) == 0:
                logger.info("step=%d elapsed=%.1fs", step, time.time() - t0)

            sleep = dt - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        front.disconnect()
        arms.disconnect()  # ramps arm_sdk weight back to 0
        logger.info("Disconnected")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
