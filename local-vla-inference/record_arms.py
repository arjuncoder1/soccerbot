"""Record G1 arm joint positions to JSON until Ctrl+C (read-only).

Subscribes ``rt/lowstate`` via ``G1Arms`` (state_only) and writes a trajectory
of the 14 arm ``.q`` values for later replay with ``replay_arms.py``.

Usage:

    ./local-vla-inference/run.sh record_arms.py --iface enp5s0 -o arms.json
    ./local-vla-inference/run.sh record_arms.py --iface enp5s0 -o arms.json --fps 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from g1_arms import ARM_JOINT_INDEX, G1Arms


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record G1 arm joint positions to JSON (Ctrl+C to stop).")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("-o", "--output", default="arms_recording.json", help="Output JSON path.")
    p.add_argument("--fps", type=float, default=30.0, help="Sample rate in Hz (default 30).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        sys.exit("--fps must be > 0")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    arms = G1Arms()
    arms.connect(state_only=True)

    joint_keys = [f"{name}.q" for name in ARM_JOINT_INDEX]
    frames: list[dict] = []
    dt = 1.0 / args.fps
    t0 = time.time()
    print(f"Recording {len(joint_keys)} arm joints @ {args.fps} Hz → {args.output}", flush=True)
    print("Ctrl+C to stop and save.", flush=True)

    try:
        while True:
            loop_start = time.time()
            q = arms.get_arm_positions()
            frames.append({"t": round(loop_start - t0, 4), "q": {k: round(q[k], 6) for k in joint_keys}})
            if len(frames) % int(max(1, args.fps)) == 0:
                print(f"  {len(frames)} frames ({frames[-1]['t']:.1f}s)", flush=True)
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        arms.disconnect()

    payload = {
        "fps": args.fps,
        "joint_keys": joint_keys,
        "n_frames": len(frames),
        "duration_s": frames[-1]["t"] if frames else 0.0,
        "frames": frames,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(frames)} frames ({payload['duration_s']:.2f}s) → {args.output}")


if __name__ == "__main__":
    main()
