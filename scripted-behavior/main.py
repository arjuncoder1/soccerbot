"""High-level orchestrator for the G1 soccer ball pickup demo.

Pipeline (each stage lives in its own module; any raise aborts the demo):

    1. PICKUP     -- ``pickup.run_pickup_policy``   (subprocess to VLA, or replay)
    2. TURN_180   -- ``turn_180.turn_180_degrees``  (LocoClient yaw + arm hold)
    3. AVOID      -- ``avoid.avoid_humans``         (shuffle + realsense)
    4. THROW      -- ``throw.throw_ball``           (hardcoded replay; TODO)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from avoid import avoid_humans
from config import OrchestratorConfig, PickupBackend
from pickup import run_pickup_policy
from throw import throw_ball
from turn_180 import turn_180_degrees

logger = logging.getLogger("scripted_behavior.orchestrator")


def run_demo(cfg: OrchestratorConfig) -> None:
    logger.info("=== Stage 1/4: PICKUP (%s) ===", cfg.backend.value)
    run_pickup_policy(cfg)

    logger.info("=== Stage 2/4: TURN 180 ===")
    turn_180_degrees(cfg)

    logger.info("=== Stage 3/4: AVOID (shuffle until clear) ===")
    avoid_humans(cfg)

    logger.info("=== Stage 4/4: THROW (hardcoded) ===")
    throw_ball(cfg)

    logger.info("Demo complete")


def parse_args(argv: list[str] | None = None) -> OrchestratorConfig:
    p = argparse.ArgumentParser(
        description="G1 soccer-ball pickup demo orchestrator "
        "(pickup -> turn 180 -> shuffle-avoid -> throw)."
    )
    p.add_argument(
        "--backend",
        type=PickupBackend,
        choices=list(PickupBackend),
        default=PickupBackend.LOCAL,
        help="Which pickup to run: local (ACT), remote (pi0.5), or replay.",
    )
    p.add_argument(
        "--iface",
        default=None,
        help="Network interface to the robot (passed to local-vla-inference).",
    )
    p.add_argument(
        "--pickup-duration",
        type=float,
        default=30.0,
        help="Seconds to run the pickup policy before advancing to stage 2.",
    )
    p.add_argument(
        "--remote-server",
        default=None,
        help="HOST:PORT of the remote pi0.5 policy server (backend=remote).",
    )
    p.add_argument(
        "pickup_extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the pickup launcher after '--'.",
    )
    args = p.parse_args(argv)

    extra = args.pickup_extra or []
    if extra and extra[0] == "--":
        extra = extra[1:]

    return OrchestratorConfig(
        backend=args.backend,
        iface=args.iface,
        pickup_duration_s=args.pickup_duration,
        pickup_extra_args=extra,
        remote_server=args.remote_server,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args(argv)
    t0 = time.time()
    try:
        run_demo(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user after %.1fs", time.time() - t0)
        return 130
    except NotImplementedError as exc:
        logger.error("Blocked on unimplemented stage: %s", exc)
        return 2
    except Exception:  # noqa: BLE001 -- top-level guard for a live demo
        logger.exception("Demo failed after %.1fs", time.time() - t0)
        return 1
    logger.info("Demo finished in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
