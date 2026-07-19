"""Replay a JSON arm trajectory recorded by ``record_arms.py``.

Engages ``rt/arm_sdk`` with a safe hold ramp, streams recorded joint targets
on wall-clock timestamps, then releases the arms on exit / Ctrl+C.

Usage:

    ./local-vla-inference/run.sh replay_arms.py --iface enp5s0 arms.json
    ./local-vla-inference/run.sh replay_arms.py --iface enp5s0 arms.json --speed 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from g1_arms import ARM_JOINT_INDEX, G1Arms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a G1 arm JSON recording via rt/arm_sdk.")
    p.add_argument("recording", help="JSON file from record_arms.py.")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("--kp", type=float, default=60.0, help="Arm joint position gain.")
    p.add_argument("--kd", type=float, default=1.5, help="Arm joint damping gain.")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default 1.0).")
    p.add_argument("--ramp", type=float, default=2.0, help="Seconds to ramp arm_sdk weight 0→1 before play.")
    p.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the trajectory until Ctrl+C.",
    )
    return p.parse_args()


def load_recording(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    frames = data.get("frames")
    if not frames:
        sys.exit(f"No frames in {path}")
    joint_keys = data.get("joint_keys") or [f"{n}.q" for n in ARM_JOINT_INDEX]
    for i, fr in enumerate(frames):
        if "q" not in fr or "t" not in fr:
            sys.exit(f"Frame {i} missing 't' or 'q'")
        for k in joint_keys:
            if k not in fr["q"]:
                sys.exit(f"Frame {i} missing joint {k}")
    data["joint_keys"] = joint_keys
    return data


def play_once(arms: G1Arms, frames: list[dict], speed: float) -> None:
    t0 = time.time()
    for fr in frames:
        target_wall = t0 + float(fr["t"]) / speed
        delay = target_wall - time.time()
        if delay > 0:
            time.sleep(delay)
        arms.send_arm_positions(fr["q"], weight=1.0)


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        sys.exit("--speed must be > 0")

    data = load_recording(args.recording)
    frames = data["frames"]
    print(
        f"Loaded {len(frames)} frames, {data.get('duration_s', frames[-1]['t']):.2f}s @ "
        f"record fps={data.get('fps', '?')} → play speed={args.speed}x",
        flush=True,
    )

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    arms = G1Arms(kp=args.kp, kd=args.kd)
    arms.connect(state_only=False)

    try:
        print(f"Engaging arm_sdk (ramp {args.ramp}s)...", flush=True)
        arms.hold_current_pose(ramp_s=args.ramp)
        # Blend toward first frame before timed playback.
        first = frames[0]["q"]
        blend_s = min(1.0, args.ramp)
        steps = max(1, int(blend_s / 0.02))
        start = arms.get_arm_positions()
        for i in range(steps):
            a = (i + 1) / steps
            blended = {k: (1.0 - a) * start[k] + a * first[k] for k in first}
            arms.send_arm_positions(blended, weight=1.0)
            time.sleep(0.02)
        print("Playing...", flush=True)
        while True:
            play_once(arms, frames, args.speed)
            if not args.loop:
                break
            print("Looping...", flush=True)
        print("Done.", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.", flush=True)
    finally:
        arms.disconnect()


if __name__ == "__main__":
    main()
