"""Stage 1: ACT pickup via in-process ``local-vla-inference`` (or trajectory replay)."""

from __future__ import annotations

import logging

from soccerbot.config import OrchestratorConfig, PickupBackend
from soccerbot.deps import ensure_logic_imports, import_local_vla_main

logger = logging.getLogger("soccerbot.pickup")


def run_pickup(cfg: OrchestratorConfig) -> None:
    ensure_logic_imports()

    if cfg.backend is PickupBackend.REPLAY:
        from arm_replay import replay_arm_trajectory

        path = cfg.replay_trajectory
        if not path.exists():
            raise FileNotFoundError(f"pickup replay trajectory missing: {path}")
        logger.info("Replaying pickup trajectory: %s (slew=%.3f)", path, cfg.replay_slew_clamp)
        replay_arm_trajectory(
            path,
            iface=cfg.iface,
            slew_clamp=cfg.replay_slew_clamp,
        )
        logger.info("Pickup replay finished")
        return

    if cfg.backend is PickupBackend.REMOTE:
        raise NotImplementedError(
            "remote pickup is not wired through soccerbot yet; use --backend local|replay"
        )

    if cfg.backend is not PickupBackend.LOCAL:
        raise AssertionError(f"unknown backend: {cfg.backend}")

    # Load by unique module name so scripted-behavior/main.py cannot shadow it.
    local_vla = import_local_vla_main()

    args = local_vla.build_args(
        layout=cfg.layout,
        policy=cfg.policy,
        iface=cfg.iface,
        camera=cfg.camera,
        clamp=cfg.clamp,
        duration=cfg.pickup_duration_s,
        fps=cfg.fps,
        device=cfg.device,
        leave_arms_engaged=True,
    )
    logger.info(
        "Starting in-process ACT pickup: policy=%s layout=%s clamp=%.3f camera=%s duration=%.1fs",
        cfg.policy,
        cfg.layout,
        cfg.clamp,
        cfg.camera,
        cfg.pickup_duration_s,
    )
    local_vla.run(args)
    logger.info("ACT pickup finished cleanly")
