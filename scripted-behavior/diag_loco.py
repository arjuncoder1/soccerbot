"""Read the G1 loco FSM/balance state and (optionally) try a small yaw command
with return codes logged. Diagnostic-only.

Usage:
    cd ~/soccerbot/scripted-behavior

    # Just read state:
    python diag_loco.py --iface enp5s0

    # Try to switch into walking-enabled mode (Start / FSM 500) and then
    # command a tiny yaw for 2 s, logging Move return codes:
    python diag_loco.py --iface enp5s0 --start --spin 2.0

FSM ids of interest (from unitree_sdk2py.g1.loco.g1_loco_client):
    0   ZeroTorque
    1   Damp
    3   Sit
    500 Start  <-- required before Move() will accept a velocity
    702 Lie2StandUp
    706 Squat2StandUp / StandUp2Squat
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from dds import ensure_dds

logger = logging.getLogger("scripted_behavior.diag_loco")


def _cli() -> int:
    p = argparse.ArgumentParser(description="G1 LocoClient FSM diagnostics.")
    p.add_argument("--iface", default=None, help="DDS interface (e.g. enp5s0).")
    p.add_argument("--start", action="store_true", help="Call loco.Start() (SetFsmId(500)).")
    p.add_argument("--spin", type=float, default=0.0, metavar="SECS",
                   help="After --start, spin at 0.3 rad/s for SECS seconds, logging Move return codes.")
    p.add_argument("--rate", type=float, default=0.3, help="Yaw rate for --spin (rad/s).")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ensure_dds(args.iface)
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()

    code_id, fsm_id = loco.GetFsmId()
    logger.info("current FSM id=%s (rpc=%s)", fsm_id, code_id)

    if args.start:
        logger.info("Calling loco.Start()  # SetFsmId(500)")
        rc = loco.Start()
        logger.info("Start() -> %s", rc)
        time.sleep(1.0)
        code_id, fsm_id = loco.GetFsmId()
        logger.info("post-Start FSM id=%s (rpc=%s)", fsm_id, code_id)

    if args.spin > 0.0:
        dt = 0.05  # 20 Hz
        t0 = time.monotonic()
        i = 0
        try:
            while time.monotonic() - t0 < args.spin:
                rc = loco.Move(0.0, 0.0, args.rate)
                if i % 10 == 0:
                    logger.info("Move(0,0,%.2f) -> rc=%s  t=%.2fs", args.rate,
                                rc, time.monotonic() - t0)
                i += 1
                time.sleep(dt)
        finally:
            rc = loco.StopMove()
            logger.info("StopMove() -> rc=%s (%d Move calls)", rc, i)

    return 0


if __name__ == "__main__":
    sys.exit(_cli())
