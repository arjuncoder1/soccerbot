"""Show current G1 control state: motion-switcher mode, loco FSM, lowstate snapshot.

Read-only. Interprets whether the robot is in damp / standing / debug so you
know if ACT inference (rt/arm_sdk) will work.

Usage:

    ./local-vla-inference/run.sh diag_state.py --iface enp5s0
"""

from __future__ import annotations

import argparse
import sys
import threading

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
    return p.parse_args()


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

    # --- motion switcher (which high-level service owns the robot) ---
    msc = MotionSwitcherClient()
    msc.SetTimeout(3.0)
    msc.Init()
    status, result = msc.CheckMode()
    name = (result or {}).get("name", "")
    print(f"motion_switcher: status={status} result={result}")
    if status != 0:
        print("  !! CheckMode failed — DDS reachable? correct --iface?")
    elif name:
        print(f"  -> high-level service '{name}' is ACTIVE (normal for arm_sdk use)")
    else:
        print("  -> NO high-level service (debug mode) — ai_sport is off; arm_sdk balance won't run")

    # --- loco FSM (damp / stand / ...) ---
    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()
    code, fsm_id = loco.GetFsmId()
    fsm_desc = FSM_NAMES.get(fsm_id, "unknown")
    print(f"loco fsm: code={code} id={fsm_id} ({fsm_desc})")

    # --- lowstate snapshot ---
    lock = threading.Lock()
    latest: dict = {"msg": None}
    first = threading.Event()

    def on_state(msg) -> None:
        with lock:
            latest["msg"] = msg
        first.set()

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)
    if not first.wait(timeout=10.0):
        sys.exit("No rt/lowstate within 10s — check --iface and robot network.")

    with lock:
        msg = latest["msg"]
    print(f"lowstate: mode_machine={msg.mode_machine} mode_pr={msg.mode_pr}")
    print("arm joints (q):")
    for jname, idx in ARM_JOINT_INDEX.items():
        m = msg.motor_state[idx]
        print(f"  {jname.removeprefix('k'):<20} q={m.q:+.4f}  dq={m.dq:+.4f}  tau={m.tau_est:+.4f}  temp={m.temperature}")

    # --- verdict ---
    print()
    if fsm_id == 1:
        print("VERDICT: robot is in DAMP. Stand it up (L1+A then L1+Up, or LocoClient.StandUp()) before inference.")
    elif not name:
        print("VERDICT: debug mode (no ai service). Reboot or re-enable ai_sport before inference.")
    else:
        print("VERDICT: high-level active and not in damp — OK to run inference.")


if __name__ == "__main__":
    main()
