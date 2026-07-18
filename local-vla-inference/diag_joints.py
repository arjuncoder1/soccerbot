"""Log G1 arm joint positions (read-only, publishes nothing).

Same DDS infra as main.py: subscribes ``rt/lowstate`` and prints the 14 arm
joints. Safe to run any time — no commands are sent to the robot.

Usage:

    ./local-vla-inference/run.sh diag_joints.py --iface enp5s0
    ./local-vla-inference/run.sh diag_joints.py --iface enp5s0 --csv joints.csv --fps 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time

from g1_arms import ARM_JOINT_INDEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only G1 arm joint position logger (rt/lowstate).")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("--fps", type=float, default=2.0, help="Print/log rate in Hz (default 2).")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = until Ctrl+C).")
    p.add_argument("--csv", default=None, help="Optional CSV output path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    lock = threading.Lock()
    latest: dict = {"msg": None}
    first = threading.Event()

    def on_state(msg) -> None:
        with lock:
            latest["msg"] = msg
        first.set()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("Waiting for rt/lowstate...", flush=True)
    if not first.wait(timeout=10.0):
        sys.exit("No rt/lowstate within 10s — check --iface and robot network.")

    names = list(ARM_JOINT_INDEX)
    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t", *names])

    header = " ".join(f"{n.removeprefix('k'):>18}" for n in names)
    dt = 1.0 / args.fps
    t0 = time.time()
    row_count = 0
    try:
        while True:
            if args.duration > 0 and time.time() - t0 >= args.duration:
                break
            with lock:
                msg = latest["msg"]
            q = [float(msg.motor_state[idx].q) for idx in ARM_JOINT_INDEX.values()]
            if row_count % 20 == 0:
                print(header)
            print(" ".join(f"{v:>18.4f}" for v in q), flush=True)
            if writer:
                writer.writerow([round(time.time() - t0, 4), *[round(v, 6) for v in q]])
            row_count += 1
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        if csv_file:
            csv_file.close()
            print(f"Wrote {row_count} rows to {args.csv}")


if __name__ == "__main__":
    main()
