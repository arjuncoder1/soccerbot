"""High-level orchestrator for the G1 soccer ball pickup demo.

Pipeline (each stage lives in its own module; any raise aborts the demo):

    1. PICKUP     -- ``pickup.run_pickup_policy``   (subprocess to VLA, or replay)
    2. TURN_180   -- ``turn_180.turn_180_degrees``  (LocoClient yaw + arm hold)
    3. THROW      -- replay of recorded trajectory, or teach mode
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

from config import REPO_ROOT, OrchestratorConfig, PickupBackend
from pickup import run_pickup_policy
from sidestep import sidestep as do_sidestep
from turn_180 import turn_180_degrees

logger = logging.getLogger("scripted_behavior.orchestrator")

THROW_TRAJECTORY = Path(__file__).resolve().parent / "trajectories" / "throw_recorded.json"


def _import_g1_arms():
    g1_dir = str(REPO_ROOT / "local-vla-inference")
    if g1_dir not in sys.path:
        sys.path.insert(0, g1_dir)
    from g1_arms import G1Arms
    return G1Arms


def run_demo(cfg: OrchestratorConfig) -> None:
    from dds import ensure_dds
    ensure_dds(cfg.iface)

    G1Arms = _import_g1_arms()
    bridge = G1Arms(kp=60.0, kd=1.5)
    bridge.connect(state_only=False)

    bridge_stop = threading.Event()
    bridge_pose = [None]

    def _bridge_loop():
        while not bridge_stop.is_set():
            p = bridge_pose[0]
            if p is not None:
                try:
                    bridge.send_arm_positions(p, weight=1.0)
                except Exception:
                    pass
            time.sleep(0.05)

    bridge_thread = threading.Thread(target=_bridge_loop, daemon=True)
    bridge_thread.start()

    try:
        logger.info("=== Stage 1/4: PICKUP (%s) ===", cfg.backend.value)
        run_pickup_policy(cfg)

        bridge_pose[0] = bridge.get_arm_positions()
        bridge.send_arm_positions(bridge_pose[0], weight=1.0)

        logger.info("=== Stage 2/4: TURN 180 ===")
        bridge.lock_joint(12, q=0.0, kp=0.0, kd=2.0)
        turn_180_degrees(cfg)
        bridge.unlock_joint(12)

        bridge_pose[0] = bridge.get_arm_positions()
        bridge.send_arm_positions(bridge_pose[0], weight=1.0)

        logger.info("=== Stage 3/4: SIDESTEP (8L 8R) ===")
        do_sidestep("left", 8, cfg, arms=bridge)
        bridge_pose[0] = bridge.get_arm_positions()
        bridge.send_arm_positions(bridge_pose[0], weight=1.0)
        do_sidestep("right", 8, cfg, arms=bridge)
        bridge_pose[0] = bridge.get_arm_positions()
        bridge.send_arm_positions(bridge_pose[0], weight=1.0)

        if cfg.mode == "teach":
            logger.info("=== TEACH MODE ===")
            bridge_stop.set()
            bridge_thread.join(timeout=1.0)
            _record_throw(cfg)
        else:
            logger.info("=== Stage 4/4: THROW (replay) ===")
            _replay_throw(cfg, bridge, bridge_stop, bridge_thread)

        logger.info("Demo complete")
    finally:
        bridge_stop.set()
        bridge_thread.join(timeout=1.0)
        _walk_reset(cfg)


def _replay_throw(cfg, bridge, bridge_stop, bridge_thread):
    from arm_replay import replay_arm_trajectory
    if not THROW_TRAJECTORY.exists():
        logger.error("No throw trajectory at %s", THROW_TRAJECTORY)
        return
    logger.info("Replaying throw trajectory: %s", THROW_TRAJECTORY.name)
    bridge_stop.set()
    bridge_thread.join(timeout=1.0)
    replay_arm_trajectory(
        THROW_TRAJECTORY,
        iface=cfg.iface,
        instant_engage=True,
        arms=bridge,
    )
    logger.info("Throw replay finished")


def _record_throw(cfg):
    G1Arms = _import_g1_arms()
    from dds import ensure_dds
    ensure_dds(cfg.iface)

    arms = G1Arms(kp=60.0, kd=1.5)
    arms.connect(state_only=False)
    hold_pose = arms.get_arm_positions()
    arms.send_arm_positions(hold_pose, weight=1.0)

    JOINTS_ORDER = [
        "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw",
        "kLeftElbow", "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
        "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw",
        "kRightElbow", "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    ]

    stop_hold = threading.Event()
    loosen = threading.Event()

    def _hold_loop():
        while not stop_hold.is_set():
            try:
                if loosen.is_set():
                    cur = arms.get_arm_positions()
                    arms.send_arm_positions(cur, weight=1.0)
                else:
                    arms.send_arm_positions(hold_pose, weight=1.0)
            except Exception:
                pass
            time.sleep(0.05)

    hold_thread = threading.Thread(target=_hold_loop, daemon=True)
    hold_thread.start()

    input("\n>>> Press ENTER to LOOSEN arms and start recording...\n")
    arms.kp = 0.0
    arms.kd = 0.5
    loosen.set()
    logger.info("Arms loosened (kp=0, kd=0.5) -- guide them now. Press ENTER to stop recording.")

    stop_event = threading.Event()
    frames = []

    def _record():
        while not stop_event.is_set():
            t0 = time.time()
            pos = arms.get_arm_positions()
            frame = [float(pos.get(f"{j}.q", 0.0)) for j in JOINTS_ORDER]
            frames.append(frame)
            elapsed = time.time() - t0
            time.sleep(max(0.0, 1.0/30.0 - elapsed))

    rec_thread = threading.Thread(target=_record, daemon=True)
    rec_thread.start()
    input("")
    stop_event.set()
    rec_thread.join(timeout=2.0)
    stop_hold.set()
    hold_thread.join(timeout=1.0)

    logger.info("Recorded %d frames (%.1fs at 30Hz)", len(frames), len(frames)/30.0)

    import json
    out_path = Path(__file__).resolve().parent / "trajectories" / "throw_recorded.json"
    traj = {
        "name": "throw_recorded",
        "source": "manual_guidance",
        "fps": 30,
        "kind": "arm_qpos_14d",
        "joints_order": JOINTS_ORDER,
        "num_frames": len(frames),
        "frames": frames,
    }
    with open(out_path, "w") as f:
        json.dump(traj, f, indent=1)
    logger.info("Saved trajectory to %s", out_path)


def _walk_reset(cfg):
    try:
        from dds import ensure_dds
        ensure_dds(cfg.iface)
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        loco = LocoClient()
        loco.SetTimeout(3.0)
        loco.Init()
        loco.StopMove()
        time.sleep(0.2)
        loco.Start()
        logger.info("Walk state restored (FSM 500)")
    except Exception:
        logger.exception("walk reset failed -- use walk_reset.py manually")


def parse_args(argv: list[str] | None = None) -> OrchestratorConfig:
    p = argparse.ArgumentParser(
        description="G1 soccer-ball pickup demo orchestrator "
        "(pickup -> turn 180 -> throw)."
    )
    p.add_argument(
        "--backend",
        type=PickupBackend,
        choices=list(PickupBackend),
        default=PickupBackend.REPLAY,
        help="Which pickup to run: local (ACT), remote (pi0.5), or replay.",
    )
    p.add_argument(
        "--mode",
        choices=["replay", "teach"],
        default="replay",
        help="replay: throw from recorded data. teach: loosen arms to record new throw.",
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
        mode=args.mode,
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
    except Exception:
        logger.exception("Demo failed after %.1fs", time.time() - t0)
        return 1
    logger.info("Demo finished in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
