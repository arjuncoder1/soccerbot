"""Show current G1 control state: motion-switcher mode, loco FSM, lowstate snapshot.

Read-only. Loops continuously (default 1 Hz) so you can watch the state change
live, e.g. while standing the robot up. Interprets whether ACT inference
(rt/arm_sdk) will work.

Usage:

    ./local-vla-inference/run.sh diag_state.py --iface enp5s0
    ./local-vla-inference/run.sh diag_state.py --iface enp5s0 --once
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from g1_arms import ARM_JOINT_INDEX

# LocoClient FSM ids (from unitree g1_loco examples; firmware-dependent).
FSM_NAMES = {
    0: "zero torque",
    1: "damp",
    2: "squat",
    3: "sit",
    4: "stand (locked)",
    200: "start / balance stand",
    500: "advanced (main operation)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1 control-state diagnosis (read-only).")
    p.add_argument("--iface", default=None, help="Network interface on the robot's network (e.g. enp5s0).")
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between updates (default 1).")
    p.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    p.add_argument("--joints", action="store_true", help="Also print all 14 arm joints every update.")
    return p.parse_args()


def verdict(fsm_id: int | None, name: str) -> str:
    if fsm_id == 1:
        return "DAMP — stand up (L1+A then L1+Up) before inference"
    if not name:
        return "DEBUG mode (no ai service) — re-enable ai_sport before inference"
    return "OK to run inference"


def main() -> None:
    args = parse_args()

    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    msc = MotionSwitcherClient()
    msc.SetTimeout(3.0)
    msc.Init()

    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()

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

    try:
        while True:
            status, result = msc.CheckMode()
            name = (result or {}).get("name", "") if status == 0 else "?"
            code, fsm_id = loco.GetFsmId()
            if code != 0:
                fsm_id = None
            fsm_desc = FSM_NAMES.get(fsm_id, "unknown")

            with lock:
                msg = latest["msg"]

            stamp = time.strftime("%H:%M:%S")
            print(
                f"[{stamp}] switcher={name or 'NONE(debug)'} "
                f"fsm={fsm_id}({fsm_desc}) "
                f"mode_machine={msg.mode_machine} mode_pr={msg.mode_pr} "
                f"| {verdict(fsm_id, name)}",
                flush=True,
            )
            if args.joints:
                for jname, idx in ARM_JOINT_INDEX.items():
                    m = msg.motor_state[idx]
                    print(
                        f"  {jname.removeprefix('k'):<20} q={m.q:+.4f}  dq={m.dq:+.4f}  "
                        f"tau={m.tau_est:+.4f}  temp={m.temperature}"
                    )

            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
