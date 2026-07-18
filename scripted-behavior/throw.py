"""Stage 4: hardcoded ball-throw trajectory.

TODO(scripted-behavior): record a single throw demo (windup -> forward
swing -> release) as 14-D arm ``qpos`` frames at 30 Hz -- e.g. by
running ``local-vla-inference/diag_state.py`` while a human backdrives
the arms through the motion -- then save the frames as
``scripted-behavior/trajectories/throw.json`` and stream them out over
``rt/arm_sdk`` using ``G1Arms.send_arm_positions`` from
``local-vla-inference/g1_arms.py``.

The 6-second hardcoded-throw timeout mentioned in AGENTS.md lives here
(``THROW_MAX_S``); refuse to run past it no matter what the trajectory
says.
"""

from __future__ import annotations

import logging

from config import OrchestratorConfig

logger = logging.getLogger("scripted_behavior.throw")

THROW_MAX_S = 6.0
THROW_CONTROL_HZ = 30.0


def throw_ball(cfg: OrchestratorConfig) -> None:
    raise NotImplementedError(
        "throw_ball: hardcoded throw trajectory not recorded yet"
    )


# ---------------------------------------------------------------------------
# Standalone entry point: run just this stage.
#   python3 throw.py --iface eth0            (currently NotImplementedError)
# ---------------------------------------------------------------------------


def _cli() -> int:
    import argparse
    import logging as _logging

    p = argparse.ArgumentParser(description="Standalone throw stage test.")
    p.add_argument("--iface", default=None, help="DDS network interface (e.g. eth0).")
    args = p.parse_args()

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = OrchestratorConfig(iface=args.iface)
    try:
        throw_ball(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
