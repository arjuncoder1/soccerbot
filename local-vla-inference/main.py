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
import csv
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
        default="zmq://192.168.123.164:55555",
        help="Front camera source: 'zmq://HOST:PORT' (teleimager head-cam stream, verified on "
        ":55555), 'teleimager://HOST' (auto-detect port/binocular via :60000), "
        "or 'opencv:N' (camera on this machine).",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run (0 = forever).")
    p.add_argument("--kp", type=float, default=60.0, help="Arm joint position gain.")
    p.add_argument("--kd", type=float, default=1.5, help="Arm joint damping gain.")
    p.add_argument(
        "--clamp",
        type=float,
        default=0.01,
        metavar="RAD",
        help="Slew limit: max radians any arm joint may move per control step toward the "
        "policy target. Default 0.01 (~0.3 rad/s at 30 fps = super slow). "
        "Use --clamp 0 to disable.",
    )
    p.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="CSV log of every step (measured, policy target, emitted command per joint). "
        "Default: act_log_<timestamp>.csv in the current directory.",
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
    arm_keys = [f"{j}.q" for j in ARM_JOINTS]
    # Commanded pose starts at the measured pose; each step it creeps toward
    # the policy target by at most --clamp rad, so motion stays slow no matter
    # what the policy outputs.
    cmd_q = dict(arms.get_arm_positions())

    log_path = args.log or time.strftime("act_log_%Y%m%d_%H%M%S.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    snapshot_keys = sorted(arms.get_full_snapshot())
    # Run config up top so a log file is self-describing (pandas: skiprows=1).
    log_file.write(f"# args: {vars(args)}\n")
    log_writer.writerow(
        ["t", "step", "cam_ms", "policy_ms", "loop_ms", "clamp_hits", "max_target_gap"]
        + [f"target_{j}" for j in ARM_JOINTS]
        + [f"cmd_{j}" for j in ARM_JOINTS]
        + snapshot_keys
    )
    logger.info(
        "ACT loop @ %.1f Hz via rt/arm_sdk, clamp=%.3f rad/step, log=%s (Ctrl+C to FREEZE and stop)",
        args.fps,
        args.clamp,
        log_path,
    )

    frozen = False
    try:
        while True:
            loop_start = time.perf_counter()
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                logger.info("Duration reached (%.1fs); stopping", args.duration)
                break

            cam_start = time.perf_counter()
            front_rgb = front.read_resized(h, w)
            cam_ms = (time.perf_counter() - cam_start) * 1000

            snapshot = arms.get_full_snapshot()
            measured = {k: snapshot[k] for k in arm_keys}

            policy_start = time.perf_counter()
            obs = pack_observation(measured, front_rgb)
            frame = build_inference_frame(observation=obs, ds_features=features, device=device)
            batch = preprocess(frame)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            robot_action = make_robot_action(action, features)
            policy_ms = (time.perf_counter() - policy_start) * 1000

            clamp_hits = 0
            if args.clamp and args.clamp > 0:
                for key in arm_keys:
                    delta = float(robot_action[key]) - cmd_q[key]
                    if abs(delta) > args.clamp:
                        clamp_hits += 1
                    cmd_q[key] += float(np.clip(delta, -args.clamp, args.clamp))
            else:
                for key in arm_keys:
                    cmd_q[key] = float(robot_action[key])
            arms.send_arm_positions(cmd_q)

            gaps = [abs(float(robot_action[k]) - measured[k]) for k in arm_keys]
            max_gap = max(gaps)
            loop_ms = (time.perf_counter() - loop_start) * 1000
            log_writer.writerow(
                [
                    round(time.time() - t0, 4),
                    step,
                    round(cam_ms, 1),
                    round(policy_ms, 1),
                    round(loop_ms, 1),
                    clamp_hits,
                    round(max_gap, 5),
                ]
                + [round(float(robot_action[k]), 5) for k in arm_keys]
                + [round(cmd_q[k], 5) for k in arm_keys]
                + [round(snapshot[k], 5) for k in snapshot_keys]
            )

            step += 1
            if step % int(args.fps) == 0:
                worst = arm_keys[int(np.argmax(gaps))]
                leg_dq = max(
                    abs(v) for k, v in snapshot.items() if ".dq" in k and ("Hip" in k or "Knee" in k or "Ankle" in k)
                )
                logger.info(
                    "step=%d elapsed=%.1fs | target gap max=%.3f rad (%s) clamped=%d/14 | "
                    "leg max|dq|=%.2f rad/s | cam=%.0fms policy=%.0fms",
                    step,
                    time.time() - t0,
                    max_gap,
                    worst.removeprefix("k").removesuffix(".q"),
                    clamp_hits,
                    leg_dq,
                    cam_ms,
                    policy_ms,
                )

            sleep = dt - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        # Cut actions and freeze: hold the last commanded pose, keep arm_sdk engaged.
        logger.info("Ctrl+C — freezing arms at last commanded pose")
        arms.freeze(cmd_q)
        frozen = True
    finally:
        log_file.close()
        logger.info("Step log written to %s (%d steps)", log_path, step)
        front.disconnect()
        if not frozen:
            arms.disconnect()  # normal exit: ramp arm_sdk weight back to 0
        logger.info("Done (frozen=%s)", frozen)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
