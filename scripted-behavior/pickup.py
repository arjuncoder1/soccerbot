"""Stage 1: run the learned VLA pickup policy in a subprocess.

Delegates to ``local-vla-inference/run.sh`` or
``remote-vla-inference/run_client.sh`` so we don't drag torch / lerobot
into the orchestrator process. Ctrl+C in the orchestrator propagates
to the child.

A third backend, ``replay``, streams the pre-recorded arm-qpos
trajectory at ``trajectories/pickup_ep10.json`` directly over
``rt/arm_sdk`` -- no learned policy, no camera, no torch.
"""

from __future__ import annotations

import logging
import subprocess

from config import REPO_ROOT, OrchestratorConfig, PickupBackend

logger = logging.getLogger("scripted_behavior.pickup")

LOCAL_VLA_RUN = REPO_ROOT / "local-vla-inference" / "run.sh"
REMOTE_VLA_RUN = REPO_ROOT / "remote-vla-inference" / "run_client.sh"
REPLAY_TRAJECTORY = REPO_ROOT / "scripted-behavior" / "trajectories" / "pickup_ep10.json"


def run_pickup_policy(cfg: OrchestratorConfig) -> None:
    if cfg.backend is PickupBackend.REPLAY:
        if not REPLAY_TRAJECTORY.exists():
            raise FileNotFoundError(f"pickup replay trajectory missing: {REPLAY_TRAJECTORY}")
        from arm_replay import replay_arm_trajectory

        logger.info("Replaying pickup trajectory: %s", REPLAY_TRAJECTORY)
        replay_arm_trajectory(REPLAY_TRAJECTORY, iface=cfg.iface)
        logger.info("Pickup replay finished")
        return

    if cfg.backend is PickupBackend.LOCAL:
        script = LOCAL_VLA_RUN
        cmd = [str(script), f"--duration={cfg.pickup_duration_s}"]
        if cfg.iface:
            cmd.append(f"--iface={cfg.iface}")
    elif cfg.backend is PickupBackend.REMOTE:
        script = REMOTE_VLA_RUN
        if not cfg.remote_server:
            raise ValueError(
                "remote backend requires --remote-server HOST:PORT "
                "(policy server started via remote-vla-inference/run_server.sh)"
            )
        cmd = [str(script), f"--server_address={cfg.remote_server}"]
        # remote client currently reads iface via env var, not CLI.
    else:  # pragma: no cover -- exhaustive
        raise AssertionError(f"unknown backend: {cfg.backend}")

    cmd.extend(cfg.pickup_extra_args)

    if not script.exists():
        raise FileNotFoundError(f"pickup launcher missing: {script}")

    logger.info("Starting pickup policy: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"pickup policy exited non-zero ({result.returncode}); aborting demo"
        )
    logger.info("Pickup policy finished cleanly")


# ---------------------------------------------------------------------------
# Standalone entry point: run just Stage 1.
#   python3 pickup.py --backend local  --iface eth0 --pickup-duration 15
#   python3 pickup.py --backend remote --remote-server modal.host:50051
#   python3 pickup.py --backend replay --iface eth0
# ---------------------------------------------------------------------------


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
    import sys
    sys.exit(_cli())
