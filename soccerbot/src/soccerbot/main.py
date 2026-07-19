"""Soccerbot core orchestrator.

Depends on workspace logic packages (not published to PyPI):

  - ``local-vla-inference`` — ACT pickup (imported in-process, never subprocess)
  - ``scripted-behavior``   — turn / avoid / throw + trajectory replay

Pipeline:

  1. PICKUP   — ACT (``ajkoder/g1-pickup-ball-act``, clamp 0.002) or JSON replay
  2. TURN_180 — LocoClient yaw while holding arms
  3. AVOID    — teleimager YOLO + sidestep shuffle
  4. THROW    — relative push (slew-clamped)

Ctrl+C runs a graceful reset (StopMove + release arm_sdk). Keep
``./killswitch.sh`` open in another terminal for headed Damp / ZeroTorque.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from soccerbot.config import (
    DEFAULT_CAMERA,
    DEFAULT_CLAMP_RAD,
    DEFAULT_POLICY,
    OrchestratorConfig,
    PickupBackend,
)
from soccerbot.deps import ensure_logic_imports, import_telemetry
from soccerbot.pickup import run_pickup
from soccerbot.safety import graceful_reset

logger = logging.getLogger("soccerbot.orchestrator")


def run_demo(cfg: OrchestratorConfig) -> None:
    ensure_logic_imports()

    # Scripted stage modules (flat imports from scripted-behavior/).
    from avoid import avoid_humans
    from config import OrchestratorConfig as ScriptedConfig
    from config import PickupBackend as ScriptedBackend
    from throw import throw_ball
    from turn_180 import turn_180_degrees

    # One shared Rerun session for the whole demo (pickup + turn/avoid/throw).
    # Passing it into ACT avoids a second rr.init / shutdown_rerun that used to
    # wipe --record-path FileSinks. Side-channel only: cfg.rerun=False is a no-op.
    telemetry_mod = import_telemetry()
    telemetry = telemetry_mod.Telemetry(
        enabled=cfg.rerun,
        session_name="soccerbot_demo",
        record_path=cfg.record_path,
        display=cfg.display,
    )
    telemetry.start()

    # Bridge soccerbot config → scripted-behavior config (shared field names).
    scripted = ScriptedConfig(
        backend=ScriptedBackend(cfg.backend.value),
        iface=cfg.iface,
        pickup_duration_s=cfg.pickup_duration_s,
        teleimager_host=cfg.teleimager_host,
        remote_server=cfg.remote_server,
        telemetry=telemetry,
    )

    try:
        telemetry.log_stage("STAGE 1/4: PICKUP (%s)" % cfg.backend.value)
        logger.info("=== Stage 1/4: PICKUP (%s) ===", cfg.backend.value)
        run_pickup(cfg, telemetry=telemetry)

        telemetry.log_stage("STAGE 2/4: TURN 180")
        logger.info("=== Stage 2/4: TURN 180 ===")
        turn_180_degrees(scripted)

        telemetry.log_stage("STAGE 3/4: AVOID (shuffle until clear)")
        logger.info("=== Stage 3/4: AVOID (shuffle until clear) ===")
        avoid_humans(scripted)

        telemetry.log_stage("STAGE 4/4: THROW")
        logger.info("=== Stage 4/4: THROW ===")
        throw_ball(scripted, slew_clamp=cfg.throw_slew_clamp)

        telemetry.log_stage("Demo complete")
    finally:
        telemetry.stop()

    logger.info("Demo complete")


def parse_args(argv: list[str] | None = None) -> OrchestratorConfig:
    p = argparse.ArgumentParser(
        description="Soccerbot orchestrator: ACT pickup → turn → avoid → throw.",
    )
    p.add_argument(
        "--backend",
        type=PickupBackend,
        choices=list(PickupBackend),
        default=PickupBackend.LOCAL,
        help="Pickup source: local ACT (default), replay trajectory, or remote.",
    )
    p.add_argument("--iface", default=None, help="DDS NIC (e.g. enp5s0).")
    p.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help="Teleimager / camera spec (default: working zmq://192.168.123.164:55555).",
    )
    p.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        help="ACT Hub id or local pretrained_model dir.",
    )
    p.add_argument("--layout", choices=("14d", "16d"), default="14d")
    p.add_argument(
        "--clamp",
        type=float,
        default=DEFAULT_CLAMP_RAD,
        help="ACT slew clamp rad/step (default 0.002).",
    )
    p.add_argument("--pickup-duration", type=float, default=30.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default=None)
    p.add_argument("--teleimager-host", default="192.168.123.164")
    p.add_argument("--remote-server", default=None)
    p.add_argument(
        "--replay-trajectory",
        default=None,
        help="JSON path for --backend replay (default: pickup_ep148_prod2.json).",
    )
    p.add_argument("--no-rerun", action="store_true", help="Disable Rerun visualization.")
    p.add_argument(
        "--record-path",
        default=None,
        metavar="PATH",
        help="Write the full demo Rerun stream (pickup + turn/avoid/throw) to this .rrd "
        "file (e.g. logs/demo.rrd), for offline analysis with query_demo.py.",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="With --record-path: write to disk only, don't spawn a live viewer window "
        "(e.g. running headless on the robot itself).",
    )
    p.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Print resolved config and exit (no robot).",
    )
    args = p.parse_args(argv)

    from pathlib import Path

    from soccerbot.config import DEFAULT_REPLAY_TRAJECTORY

    cfg = OrchestratorConfig(
        backend=args.backend,
        iface=args.iface,
        camera=args.camera,
        policy=args.policy,
        layout=args.layout,
        clamp=args.clamp,
        pickup_duration_s=args.pickup_duration,
        fps=args.fps,
        device=args.device,
        rerun=not args.no_rerun,
        record_path=args.record_path,
        display=not args.no_display,
        teleimager_host=args.teleimager_host,
        remote_server=args.remote_server,
        replay_trajectory=(
            Path(args.replay_trajectory) if args.replay_trajectory else DEFAULT_REPLAY_TRAJECTORY
        ),
    )
    if args.dry_run_config:
        print(cfg)
        raise SystemExit(0)
    return cfg


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = parse_args(argv)
    logger.info(
        "Soccerbot starting: backend=%s policy=%s clamp=%.3f camera=%s iface=%s rerun=%s",
        cfg.backend.value,
        cfg.policy,
        cfg.clamp,
        cfg.camera,
        cfg.iface,
        cfg.rerun,
    )
    logger.info("Tip: keep ./killswitch.sh --iface %s open in another terminal", cfg.iface or "<iface>")

    t0 = time.time()
    try:
        run_demo(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted after %.1fs — graceful reset", time.time() - t0)
        try:
            graceful_reset(iface=cfg.iface)
        except Exception:  # noqa: BLE001
            logger.exception("graceful_reset failed")
        return 130
    except NotImplementedError as exc:
        logger.error("Blocked on unimplemented stage: %s", exc)
        return 2
    except Exception:  # noqa: BLE001
        logger.exception("Demo failed after %.1fs — attempting graceful reset", time.time() - t0)
        try:
            graceful_reset(iface=cfg.iface)
        except Exception:  # noqa: BLE001
            logger.exception("graceful_reset failed")
        return 1

    logger.info("Demo finished in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
