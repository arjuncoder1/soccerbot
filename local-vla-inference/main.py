"""Run ACT inference on Unitree G1D arms only (no hands / fingers).

Policy: ``myx160/unitree_lerobot_act_g1d_16d_001`` (16-D).

Camera: Unitree **onboard front cam** via existing ``VideoClient`` (no extra
software / ImageServer / OpenCV capture on the robot). That frame is copied
into all 4 policy image inputs.

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
from front_camera import UnitreeFrontCamera

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACT inference for G1D arms-only + Unitree front VideoClient."
    )
    p.add_argument("--policy", default=DEFAULT_POLICY_ID, help="Hub repo id or local checkpoint path.")
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available).")
    p.add_argument("--robot-ip", default="192.168.123.164", help="G1 robot IP for DDS bridge.")
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


def build_robot(args: argparse.Namespace):
    from lerobot.robots.unitree_g1 import UnitreeG1, UnitreeG1Config

    # No cameras on the LeRobot robot config — front cam is VideoClient only.
    cfg = UnitreeG1Config(
        id="g1d_act",
        is_simulation=args.simulation,
        robot_ip=args.robot_ip,
        cameras={},
        controller=None,
        gravity_compensation=False,
    )
    return UnitreeG1(cfg)


def pack_observation(robot_obs: dict[str, Any], front_rgb: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for joint in ARM_JOINTS:
        key = f"{joint}.q"
        if key not in robot_obs:
            raise KeyError(f"Missing arm joint in robot observation: {key}")
        out[key] = float(robot_obs[key])
    for pad in UNUSED_PAD:
        out[pad] = 0.0
    for cam in CAMERA_KEYS:
        out[cam] = front_rgb
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
    _ = make_robot_action(action, features)
    logger.info("Dry-run OK (arms only; front VideoClient not used).")


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

    h, w, _ = IMAGE_SHAPE
    robot = build_robot(args)
    front = UnitreeFrontCamera()

    robot.connect()
    # DDS is up after robot.connect(); VideoClient uses the robot's existing service.
    try:
        front.connect()
    except Exception:
        logger.error(
            "Could not read Unitree front camera via VideoClient. "
            "Nothing was started on the robot — this uses the stock camera service only."
        )
        robot.disconnect()
        raise

    dt = 1.0 / args.fps
    t0 = time.time()
    step = 0
    logger.info("ACT loop @ %.1f Hz | Unitree front VideoClient → %s", args.fps, CAMERA_KEYS)

    try:
        while True:
            loop_start = time.perf_counter()
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                logger.info("Duration reached (%.1fs); stopping", args.duration)
                break

            front_rgb = front.read_resized(h, w)
            obs = pack_observation(robot.get_observation(), front_rgb)
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
        front.disconnect()
        robot.disconnect()
        logger.info("Disconnected")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
