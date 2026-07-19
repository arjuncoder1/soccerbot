"""Log G1 arm + torso joint positions and say which are MOVING (read-only).

Same DDS infra as main.py: subscribes ``rt/lowstate``. Instead of a wall of
noisy numbers, each line reports joints whose position changed more than
``--threshold`` radians since the last sample:

    [11:42:01] MOVING  LeftElbow(+0.152)  WaistYaw(-0.081)
    [11:42:02] still

Use ``--raw`` for the full numeric table. ``--csv`` always logs raw values.
``--torso-only`` skips the 14 arm joints.

Usage:

    ./local-vla-inference/run.sh diag_joints.py --iface enp5s0
    ./local-vla-inference/run.sh diag_joints.py --iface enp5s0 --torso-only
    ./local-vla-inference/run.sh diag_joints.py --iface enp5s0 --csv joints.csv --fps 30
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time

from g1_arms import ARM_JOINT_INDEX, TORSO_JOINT_INDEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only G1 arm+torso joint motion logger (rt/lowstate).")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("--fps", type=float, default=2.0, help="Print/log rate in Hz (default 2).")
    p.add_argument("--duration", type=float, default=0.0, help="Seconds to run (0 = until Ctrl+C).")
    p.add_argument("--csv", default=None, help="Optional CSV output path (raw positions).")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Radians of change since last sample to count as moving (default 0.01).",
    )
    p.add_argument("--raw", action="store_true", help="Print the full numeric table instead of motion summary.")
    p.add_argument("--torso-only", action="store_true", help="Log only waist yaw/roll/pitch (skip arms).")
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

    joints = dict(TORSO_JOINT_INDEX) if args.torso_only else {**TORSO_JOINT_INDEX, **ARM_JOINT_INDEX}
    names = list(joints)
    short_names = [n.removeprefix("k") for n in names]
    writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["t", *names])

    header = " ".join(f"{n:>18}" for n in short_names)
    dt = 1.0 / args.fps
    t0 = time.time()
    row_count = 0
    prev_q: list[float] | None = None
    try:
        while True:
            if args.duration > 0 and time.time() - t0 >= args.duration:
                break
            with lock:
                msg = latest["msg"]
            q = [float(msg.motor_state[idx].q) for idx in joints.values()]

            if args.raw:
                if row_count % 20 == 0:
                    print(header)
                print(" ".join(f"{v:>18.4f}" for v in q), flush=True)
            else:
                stamp = time.strftime("%H:%M:%S")
                if prev_q is None:
                    print(f"[{stamp}] baseline captured (threshold {args.threshold} rad)", flush=True)
                else:
                    moving = [
                        (short_names[i], q[i] - prev_q[i])
                        for i in range(len(q))
                        if abs(q[i] - prev_q[i]) >= args.threshold
                    ]
                    if moving:
                        detail = "  ".join(f"{n}({d:+.3f})" for n, d in moving)
                        print(f"[{stamp}] MOVING  {detail}", flush=True)
                    else:
                        print(f"[{stamp}] still", flush=True)
                prev_q = q

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
