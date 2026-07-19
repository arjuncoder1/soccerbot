"""Stage 1: run the learned ACT pickup policy in-process (or replay a trajectory).

Delegates to ``local-vla-inference`` via import (no subprocess) so slew
clamping, Ctrl+C graceful reset, and Rerun telemetry stay in one process.

A third backend, ``replay``, streams a pre-recorded arm-qpos trajectory from
``trajectories/`` over ``rt/arm_sdk``.
"""

from __future__ import annotations

import logging
import sys

from config import REPO_ROOT, OrchestratorConfig, PickupBackend

logger = logging.getLogger("scripted_behavior.pickup")

LOCAL_VLA_DIR = REPO_ROOT / "local-vla-inference"
REPLAY_TRAJECTORY = REPO_ROOT / "scripted-behavior" / "trajectories" / "pickup_ep148_prod2.json"
DEFAULT_POLICY = "ajkoder/g1-pickup-ball-act"
DEFAULT_CLAMP = 0.002
DEFAULT_CAMERA = "zmq://192.168.123.164:55555"


def run_pickup_policy(cfg: OrchestratorConfig) -> None:
    if cfg.backend is PickupBackend.REPLAY:
        if not REPLAY_TRAJECTORY.exists():
            raise FileNotFoundError(f"pickup replay trajectory missing: {REPLAY_TRAJECTORY}")
        from arm_replay import replay_arm_trajectory

        logger.info("Replaying pickup trajectory: %s", REPLAY_TRAJECTORY)
        replay_arm_trajectory(REPLAY_TRAJECTORY, iface=cfg.iface)
        logger.info("Pickup replay finished")
        return

    if cfg.backend is PickupBackend.REMOTE:
        raise NotImplementedError(
            "remote backend is not imported in-process yet; use soccerbot --backend local|replay "
            "or the remote-vla-inference client directly"
        )

    if cfg.backend is not PickupBackend.LOCAL:
        raise AssertionError(f"unknown backend: {cfg.backend}")

    if str(LOCAL_VLA_DIR) not in sys.path:
        sys.path.insert(0, str(LOCAL_VLA_DIR))
    import main as local_vla  # type: ignore[import-not-found]

    # Optional extras: --policy / --clamp / --camera forwarded after '--'.
    policy = DEFAULT_POLICY
    clamp = DEFAULT_CLAMP
    camera = DEFAULT_CAMERA
    layout = "14d"
    extra = list(cfg.pickup_extra_args)
    # Tiny argv parse for common flags without another ArgumentParser.
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok.startswith("--policy="):
            policy = tok.split("=", 1)[1]
        elif tok == "--policy" and i + 1 < len(extra):
            i += 1
            policy = extra[i]
        elif tok.startswith("--clamp="):
            clamp = float(tok.split("=", 1)[1])
        elif tok == "--clamp" and i + 1 < len(extra):
            i += 1
            clamp = float(extra[i])
        elif tok.startswith("--camera="):
            camera = tok.split("=", 1)[1]
        elif tok == "--camera" and i + 1 < len(extra):
            i += 1
            camera = extra[i]
        elif tok.startswith("--layout="):
            layout = tok.split("=", 1)[1]
        elif tok == "--layout" and i + 1 < len(extra):
            i += 1
            layout = extra[i]
        i += 1

    args = local_vla.build_args(
        layout=layout,
        policy=policy,
        iface=cfg.iface,
        camera=camera,
        clamp=clamp,
        duration=cfg.pickup_duration_s,
        leave_arms_engaged=True,
        rerun=True,
    )
    logger.info(
        "Starting in-process ACT pickup: policy=%s clamp=%.3f camera=%s",
        policy,
        clamp,
        camera,
    )
    local_vla.run(args)
    logger.info("Pickup policy finished cleanly")


def _cli() -> int:
    import argparse
    import logging as _logging

    p = argparse.ArgumentParser(description="Standalone pickup-policy launcher test.")
    p.add_argument(
        "--backend",
        type=PickupBackend,
        choices=list(PickupBackend),
        default=PickupBackend.LOCAL,
    )
    p.add_argument("--iface", default=None)
    p.add_argument("--pickup-duration", type=float, default=15.0)
    p.add_argument("--remote-server", default=None)
    p.add_argument("pickup_extra", nargs=argparse.REMAINDER)
    args = p.parse_args()

    extra = args.pickup_extra or []
    if extra and extra[0] == "--":
        extra = extra[1:]

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = OrchestratorConfig(
        backend=args.backend,
        iface=args.iface,
        pickup_duration_s=args.pickup_duration,
        remote_server=args.remote_server,
        pickup_extra_args=extra,
    )
    try:
        run_pickup_policy(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
