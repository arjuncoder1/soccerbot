"""Sidestep primitive: shuffle N steps left or right, holding the arms.

Mirror of ``turn_180.py`` for lateral motion. Key differences:

  * The G1's ``rt/lowstate`` does not include odometry, and integrating IMU
    accel drifts far too fast to close the loop on lateral distance. So
    this primitive is **open-loop on time**: each step is
    ``SIDESTEP_STEP_S`` seconds of ``LocoClient.Move(0, +/- vy, 0)`` at
    ``SIDESTEP_VY_MPS``. Nominal per-step distance is the product; tune
    both constants on-robot.
  * A brief ``StopMove`` + ``SIDESTEP_PAUSE_S`` pause between steps lets
    the balancer settle so consecutive steps don't compound momentum.
  * Same arm-hold pattern as ``turn_180``: snapshot arm pose, re-publish
    it over ``rt/arm_sdk`` at ``SIDESTEP_CONTROL_HZ`` in a background
    thread, leave arm_sdk engaged on exit.
"""

from __future__ import annotations

import logging
import sys as _sys
import threading
import time

from config import REPO_ROOT, OrchestratorConfig
from dds import ensure_dds

logger = logging.getLogger("scripted_behavior.sidestep")

# Lateral velocity command magnitude (m/s). Positive vy on the G1 loco frame
# is to the robot's *left*; we flip the sign for "right".
SIDESTEP_VY_MPS = 0.25
# Seconds of continuous Move() per "step".
SIDESTEP_STEP_S = 0.6
# Pause between consecutive steps (StopMove + settle) so momentum doesn't
# accumulate across steps and the balancer catches up.
SIDESTEP_PAUSE_S = 0.4
# Re-issue Move() / arm hold at this rate. Move() decays if not re-sent.
SIDESTEP_CONTROL_HZ = 20.0
# After the last step, keep holding the arms and let feet settle.
SIDESTEP_END_SETTLE_S = 0.6


def _import_g1_arms():
    """Import ``G1Arms`` from the sibling package without polluting sys.path."""
    g1_arms_dir = str(REPO_ROOT / "local-vla-inference")
    _sys.path.insert(0, g1_arms_dir)
    try:
        from g1_arms import G1Arms  # type: ignore[import-not-found]
    finally:
        _sys.path.pop(0)
    return G1Arms


def sidestep(direction: str, steps: int, cfg: OrchestratorConfig) -> None:
    """Shuffle ``steps`` steps ``direction`` (``"left"`` | ``"right"``).

    Open-loop on time; nominal distance per step is
    ``SIDESTEP_VY_MPS * SIDESTEP_STEP_S``.
    """
    if direction not in ("left", "right"):
        raise ValueError(f"direction must be 'left' or 'right', got {direction!r}")
    if steps <= 0:
        return

    vy_signed = SIDESTEP_VY_MPS if direction == "left" else -SIDESTEP_VY_MPS

    G1Arms = _import_g1_arms()
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    ensure_dds(cfg.iface)

    arms = G1Arms(kp=60.0, kd=1.5)
    arms.connect()
    hold_pose = dict(arms.get_arm_positions())
    arms.hold_current_pose(ramp_s=0.5)

    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()

    stop_arm_hold = threading.Event()
    dt = 1.0 / SIDESTEP_CONTROL_HZ

    def _arm_holder() -> None:
        while not stop_arm_hold.is_set():
            try:
                arms.send_arm_positions(hold_pose, weight=1.0)
            except Exception:  # noqa: BLE001 -- keep looping; final send in finally
                logger.exception("arm hold publish failed (continuing)")
            time.sleep(dt)

    holder = threading.Thread(target=_arm_holder, name="sidestep-arm-hold", daemon=True)
    holder.start()

    logger.info(
        "Sidestepping %d step(s) to the %s: vy=%+.2f m/s, %.2fs per step, "
        "%.2fs pause between",
        steps,
        direction,
        vy_signed,
        SIDESTEP_STEP_S,
        SIDESTEP_PAUSE_S,
    )

    try:
        for i in range(1, steps + 1):
            step_end = time.monotonic() + SIDESTEP_STEP_S
            while time.monotonic() < step_end:
                loco.Move(0.0, vy_signed, 0.0)
                time.sleep(dt)

            # Between steps: stop lateral motion, hold pose, let feet settle.
            try:
                loco.StopMove()
            except Exception:  # noqa: BLE001
                logger.exception("StopMove between steps failed")

            if i < steps:
                pause_end = time.monotonic() + SIDESTEP_PAUSE_S
                while time.monotonic() < pause_end:
                    time.sleep(dt)

            logger.info("Sidestep %d/%d complete (%s)", i, steps, direction)
    finally:
        try:
            loco.StopMove()
            logger.info("LocoClient.StopMove() sent (sidestep teardown)")
        except Exception:  # noqa: BLE001
            logger.exception("Final LocoClient.StopMove() failed")

        settle_end = time.monotonic() + SIDESTEP_END_SETTLE_S
        while time.monotonic() < settle_end:
            try:
                arms.send_arm_positions(hold_pose, weight=1.0)
            except Exception:  # noqa: BLE001
                logger.exception("arm hold publish failed during settle")
            time.sleep(dt)

        stop_arm_hold.set()
        holder.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Standalone entry point: run just this primitive.
#   python3 sidestep.py left  --steps 2 --iface eth0
#   python3 sidestep.py right --steps 3 --iface eth0
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse
    import logging as _logging

    p = argparse.ArgumentParser(description="Standalone sidestep primitive test.")
    p.add_argument("direction", choices=("left", "right"))
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--iface", default=None, help="DDS network interface (e.g. eth0).")
    args = p.parse_args()

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = OrchestratorConfig(iface=args.iface)
    try:
        sidestep(args.direction, args.steps, cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
