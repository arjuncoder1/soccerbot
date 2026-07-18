"""Stage 2: rotate the G1 base ~180 deg in place, holding the arms wherever
the pickup policy left them.

Approach:
  * ``LocoClient.Move(0, 0, TURN_YAW_RATE_RPS)`` drives the base yaw. Move()
    with the default ``continous_move=False`` decays if not re-sent, so we
    call it every ``1/TURN_CONTROL_HZ`` seconds.
  * Meanwhile a background thread re-publishes the last measured arm pose
    over ``rt/arm_sdk`` at the same rate, so the arms (and whatever the
    hands were doing at the end of pickup) do not drop. arm_sdk overlays
    only the arm joints; the locomotion controller keeps authority over
    legs+waist.
  * Yaw progress is integrated from ``imu_state.rpy[2]`` using a
    shortest-arc diff so wrap at +-pi is handled. Stop when we're within
    ``TURN_TOLERANCE_RAD`` of ``TURN_TARGET_RAD`` or hit ``TURN_180_MAX_S``.

arm_sdk is left engaged on exit so later stages can keep controlling
the arms without a re-engage jump.
"""

from __future__ import annotations

import logging
import math
import sys as _sys
import threading
import time

from config import REPO_ROOT, OrchestratorConfig
from dds import ensure_dds

logger = logging.getLogger("scripted_behavior.turn_180")

# Yaw rate command sent to LocoClient.Move. Positive = CCW viewed from above.
# 0.6 rad/s * ~5.2 s ~= pi rad; slow enough that the balancer is comfortable.
TURN_YAW_RATE_RPS = 0.6
# Target rotation magnitude (radians).
TURN_TARGET_RAD = math.pi
# Accept the turn as "done" once within this tolerance of TURN_TARGET_RAD.
TURN_TOLERANCE_RAD = math.radians(5.0)
# Hard safety timeout: bail (StopMove) after this many seconds regardless of
# what the IMU reports. Well above 2 * (pi / TURN_YAW_RATE_RPS).
TURN_180_MAX_S = 15.0
# How often we re-issue Move() AND re-publish the arm hold.
TURN_CONTROL_HZ = 20.0
# After StopMove(), keep holding arms while the feet settle before returning.
TURN_STOP_SETTLE_S = 1.0


def _shortest_angle_delta(a: float, b: float) -> float:
    """Signed shortest-arc delta ``a - b`` in (-pi, pi]."""
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _import_g1_arms():
    """Import ``G1Arms`` from the sibling package without polluting sys.path."""
    g1_arms_dir = str(REPO_ROOT / "local-vla-inference")
    _sys.path.insert(0, g1_arms_dir)
    try:
        from g1_arms import G1Arms  # type: ignore[import-not-found]
    finally:
        _sys.path.pop(0)
    return G1Arms


def turn_180_degrees(cfg: OrchestratorConfig) -> None:
    G1Arms = _import_g1_arms()
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    ensure_dds(cfg.iface)

    arms = G1Arms(kp=60.0, kd=1.5)
    arms.connect()
    hold_pose = dict(arms.get_arm_positions())
    # Anchor the arms before the base starts moving so any drift from the
    # pickup handoff is squashed first.
    arms.hold_current_pose(ramp_s=0.5)

    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()

    stop_arm_hold = threading.Event()
    dt = 1.0 / TURN_CONTROL_HZ

    def _arm_holder() -> None:
        while not stop_arm_hold.is_set():
            try:
                arms.send_arm_positions(hold_pose, weight=1.0)
            except Exception:  # noqa: BLE001 -- keep looping; final send in finally
                logger.exception("arm hold publish failed (continuing)")
            time.sleep(dt)

    holder = threading.Thread(target=_arm_holder, name="turn180-arm-hold", daemon=True)
    holder.start()

    start_yaw = float(arms.get_full_snapshot()["imu.yaw"])
    prev_yaw = start_yaw
    accumulated = 0.0
    t0 = time.monotonic()
    logger.info(
        "Starting turn: start_yaw=%.3f rad (%.1f deg), target=%.1f deg, rate=%.2f rad/s",
        start_yaw,
        math.degrees(start_yaw),
        math.degrees(TURN_TARGET_RAD),
        TURN_YAW_RATE_RPS,
    )

    reached = False
    try:
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= TURN_180_MAX_S:
                logger.warning(
                    "Turn timeout after %.2fs (accumulated %.1f deg / target %.1f deg)",
                    elapsed,
                    math.degrees(accumulated),
                    math.degrees(TURN_TARGET_RAD),
                )
                break

            loco.Move(0.0, 0.0, TURN_YAW_RATE_RPS)

            cur_yaw = float(arms.get_full_snapshot()["imu.yaw"])
            step = _shortest_angle_delta(cur_yaw, prev_yaw)
            accumulated += abs(step)
            prev_yaw = cur_yaw

            if (
                abs(accumulated - TURN_TARGET_RAD) <= TURN_TOLERANCE_RAD
                or accumulated >= TURN_TARGET_RAD
            ):
                reached = True
                logger.info(
                    "Turn reached target: %.1f deg in %.2fs",
                    math.degrees(accumulated),
                    elapsed,
                )
                break

            time.sleep(dt)
    finally:
        try:
            loco.StopMove()
            logger.info("LocoClient.StopMove() sent")
        except Exception:  # noqa: BLE001 -- log and continue teardown
            logger.exception("LocoClient.StopMove() failed")

        settle_end = time.monotonic() + TURN_STOP_SETTLE_S
        while time.monotonic() < settle_end:
            try:
                arms.send_arm_positions(hold_pose, weight=1.0)
            except Exception:  # noqa: BLE001
                logger.exception("arm hold publish failed during settle")
            time.sleep(dt)

        stop_arm_hold.set()
        holder.join(timeout=1.0)

    if not reached:
        raise RuntimeError(
            f"turn_180_degrees: timed out; only rotated {math.degrees(accumulated):.1f} deg "
            f"of {math.degrees(TURN_TARGET_RAD):.1f} deg"
        )


# ---------------------------------------------------------------------------
# Standalone entry point: run just this stage.
#   python3 turn_180.py --iface eth0
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse
    import logging as _logging

    p = argparse.ArgumentParser(description="Standalone 180-deg turn test.")
    p.add_argument("--iface", default=None, help="DDS network interface (e.g. eth0).")
    args = p.parse_args()

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = OrchestratorConfig(iface=args.iface)
    try:
        turn_180_degrees(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
