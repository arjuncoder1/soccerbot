"""Run ACT inference on Unitree G1 arms only (no hands / fingers).

Layouts:
  - ``16d`` — Hub ACT ``myx160/unitree_lerobot_act_g1d_16d_001`` (14 arms + 2 pad)
  - ``14d`` — cleaned G1 dataset / local ACT (``left_arm_*`` / ``right_arm_*``, ``color_0``)

Everything runs on this machine:
  - state:   subscribe ``rt/lowstate`` (DDS)
  - arms:    publish ``rt/arm_sdk`` (DDS, weight joint 29)
  - camera:  Unitree ``teleimager`` already running on the robot
             (``--camera teleimager://192.168.123.164``, head cam on :55555)

Usage:

    export CYCLONEDDS_HOME=$HOME/cyclonedds/install
    ./local-vla-inference/run.sh --iface enp5s0 --layout 14d \\
      --policy ~/soccerbot/model/pretrained_model
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from front_camera import make_front_camera
from g1_arms import G1Arms

logger = logging.getLogger(__name__)

_LOCAL_POLICY_MARKERS = ("/", "./", "../", "~")
_REQUIRED_POLICY_FILES = ("config.json", "model.safetensors")


def load_layout(name: str) -> ModuleType:
    if name == "14d":
        import embodiment_g1_14d as layout
    elif name == "16d":
        import embodiment_g1d_16d as layout
    else:
        raise ValueError(f"Unknown layout {name!r}; use 14d or 16d")
    return layout


def resolve_policy_ref(policy: str) -> str:
    """Return a Hub repo id, or an absolute local checkpoint directory.

    Absolute/relative filesystem paths must exist as a LeRobot ``pretrained_model``
    dir. Otherwise ``from_pretrained`` falls through to the Hub and raises
    ``HFValidationError`` on paths like ``/home/.../pretrained_model``.
    """
    expanded = Path(policy).expanduser()
    looks_local = (
        policy.startswith(_LOCAL_POLICY_MARKERS)
        or expanded.exists()
        or policy.count("/") > 1  # Hub ids are at most namespace/name
    )
    if not looks_local:
        return policy

    path = expanded.resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local policy path is not a directory: {path}\n"
            "Point --policy at the LeRobot checkpoint folder that contains "
            "config.json and model.safetensors (usually .../pretrained_model)."
        )
    missing = [name for name in _REQUIRED_POLICY_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete checkpoint at {path}; missing: {', '.join(missing)}\n"
            f"Contents: {sorted(p.name for p in path.iterdir())}"
        )
    return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ACT inference for G1 arms-only via direct DDS (rt/arm_sdk)."
    )
    p.add_argument(
        "--layout",
        choices=("14d", "16d"),
        default="16d",
        help="Observation/action layout. Use 14d for cleaned G1 ACT checkpoints.",
    )
    p.add_argument(
        "--policy",
        default=None,
        help="Hub repo id or local checkpoint path. Default depends on --layout.",
    )
    p.add_argument("--device", default=None, help="cuda / cpu / mps (default: cuda if available).")
    p.add_argument(
        "--iface",
        default=None,
        help="Network interface connected to the robot (e.g. eth0, enp5s0). "
        "Omit to use the DDS default.",
    )
    p.add_argument(
        "--camera",
        default="zmq://192.168.123.164:55555",
        help="Front camera source: 'zmq://HOST:PORT' (teleimager head-cam stream, verified on "
        ":55555), 'teleimager://HOST' (auto-detect port/binocular via :60000), "
        "or 'opencv:N' (camera on this machine).",
    )
    p.add_argument("--fps", type=float, default=30.0, help="Control loop rate.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to run (0 = forever).")
    p.add_argument("--kp", type=float, default=60.0, help="Arm joint position gain.")
    p.add_argument("--kd", type=float, default=1.5, help="Arm joint damping gain.")
    p.add_argument(
        "--clamp",
        type=float,
        default=0.01,
        metavar="RAD",
        help="Slew limit: max radians any arm joint may move per control step toward the "
        "policy target. Default 0.01 (~0.3 rad/s at 30 fps = super slow). "
        "Use --clamp 0 to disable. Soccerbot pickup uses 0.01.",
    )
    p.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="CSV log of every step (measured, policy target, emitted command per joint). "
        "Default: act_log_<timestamp>.csv in the current directory.",
    )
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Spawn a Rerun viewer and stream teleimager RGB + arm target/cmd/measured.",
    )
    p.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun even if the caller defaulted it on.",
    )
    p.add_argument(
        "--record-path",
        default=None,
        metavar="PATH",
        help="Also/instead write this ACT session to an .rrd file (FileSink). "
        "Use with --no-display for headless robot PCs (e.g. Waldo).",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="With --record-path: write to disk only; do not spawn a live viewer.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load policy and print one fake forward pass; do not connect hardware.",
    )
    p.add_argument(
        "--image-no-motors",
        action="store_true",
        help="Read one real camera frame + arm angles (rt/lowstate), run policy, print the "
        "predicted action chunk; never publish rt/arm_sdk / never command motors.",
    )
    p.add_argument(
        "--leave-arms-engaged",
        action="store_true",
        help="On clean exit, leave arm_sdk engaged at the last pose so a following "
        "scripted stage can take over without a re-engage jerk. Ctrl+C still does a "
        "graceful reset (StopMove + release).",
    )
    return p.parse_args(argv)


def build_args(
    *,
    layout: str = "14d",
    policy: str = "ajkoder/g1-pickup-ball-act",
    iface: str | None = None,
    camera: str = "zmq://192.168.123.164:55555",
    clamp: float = 0.01,
    duration: float = 30.0,
    fps: float = 30.0,
    device: str | None = None,
    rerun: bool = True,
    record_path: str | None = None,
    display: bool = True,
    leave_arms_engaged: bool = True,
    dry_run: bool = False,
    image_no_motors: bool = False,
    log: str | None = None,
    kp: float = 60.0,
    kd: float = 1.5,
) -> argparse.Namespace:
    """Programmatic Namespace for in-process callers (soccerbot orchestrator)."""
    return argparse.Namespace(
        layout=layout,
        policy=policy,
        iface=iface,
        camera=camera,
        clamp=clamp,
        duration=duration,
        fps=fps,
        device=device,
        rerun=rerun,
        no_rerun=not rerun,
        record_path=record_path,
        no_display=not display,
        dry_run=dry_run,
        image_no_motors=image_no_motors,
        leave_arms_engaged=leave_arms_engaged,
        log=log,
        kp=kp,
        kd=kd,
    )


def resolve_device(name: str | None):
    import torch

    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pack_observation_16d(layout: ModuleType, arm_obs: dict[str, float], front_rgb: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for joint in layout.ARM_JOINTS:
        key = f"{joint}.q"
        if key not in arm_obs:
            raise KeyError(f"Missing arm joint in observation: {key}")
        out[key] = arm_obs[key]
    for pad in layout.UNUSED_PAD:
        out[pad] = 0.0
    for cam in layout.CAMERA_KEYS:
        out[cam] = front_rgb
    return out


def action_to_dds(layout: ModuleType, robot_action: dict[str, float]) -> dict[str, float]:
    """Map policy action keys to DDS ``<joint>.q`` for G1Arms."""
    if hasattr(layout, "to_dds_action"):
        return layout.to_dds_action(robot_action)
    return {f"{j}.q": float(robot_action[f"{j}.q"]) for j in layout.ARM_JOINTS}


def dry_run(layout: ModuleType, policy, preprocess, postprocess, device) -> None:
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    features = layout.dataset_features()
    if hasattr(layout, "pack_observation"):
        blank = np.zeros(layout.IMAGE_SHAPE, dtype=np.uint8)
        fake_measured = {f"{j}.q": 0.0 for j in layout.ARM_JOINTS}
        fake_obs = layout.pack_observation(fake_measured, blank)
    else:
        fake_obs = pack_observation_16d(layout, {f"{j}.q": 0.0 for j in layout.ARM_JOINTS}, np.zeros(layout.IMAGE_SHAPE, dtype=np.uint8))

    frame = build_inference_frame(observation=fake_obs, ds_features=features, device=device)
    batch = preprocess(frame)
    action = policy.select_action(batch)
    action = postprocess(action)
    robot_action = make_robot_action(action, features)
    dds_action = action_to_dds(layout, robot_action)
    arm_keys = [f"{j}.q" for j in layout.ARM_JOINTS]
    logger.info("Dry-run OK. Arm action dims=%d", len(arm_keys))
    logger.info("Sample arm action: %s", {k: round(dds_action[k], 4) for k in arm_keys[:4]})


def _format_joint_row(dds_action: dict[str, float], arm_keys: list[str]) -> str:
    return " ".join(f"{k.removesuffix('.q')}={dds_action[k]:+.4f}" for k in arm_keys)


def print_action_trajectory(
    layout: ModuleType,
    policy,
    preprocess,
    postprocess,
    features: dict,
    observation: dict[str, Any],
    device,
    measured: dict[str, float] | None = None,
) -> None:
    """Run one ACT chunk prediction and print joint angles in policy units (radians)."""
    import torch
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    arm_keys = [f"{j}.q" for j in layout.ARM_JOINTS]
    if measured is not None:
        logger.info("Measured arm q (rad): %s", _format_joint_row(measured, arm_keys))

    frame = build_inference_frame(observation=observation, ds_features=features, device=device)
    batch = preprocess(frame)
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch)  # (B, T, action_dim)

    # Postprocess each horizon step the same way the live loop unnormalizes actions.
    n_steps = int(chunk.shape[1])
    logger.info(
        "Predicted action chunk: %d steps × %d arm joints (radians, policy units). "
        "No motor commands will be sent.",
        n_steps,
        len(arm_keys),
    )
    for t in range(n_steps):
        action_t = postprocess(chunk[:, t])
        robot_action = make_robot_action(action_t, features)
        dds_action = action_to_dds(layout, robot_action)
        print(f"step {t:03d}: {_format_joint_row(dds_action, arm_keys)}", flush=True)


def image_no_motors(
    args: argparse.Namespace,
    layout: ModuleType,
    policy,
    preprocess,
    postprocess,
    features: dict,
    pack_obs,
    device,
) -> None:
    """Real camera + read-only lowstate → print predicted trajectory. Never write motors."""
    from dds_init import ensure_dds

    ensure_dds(args.iface)

    arms = G1Arms(kp=args.kp, kd=args.kd)
    front = make_front_camera(args.camera)
    # state_only: subscribe rt/lowstate only — arm_sdk publisher is never created.
    arms.connect(state_only=True)
    front.connect()

    try:
        h, w, _ = layout.IMAGE_SHAPE
        front_rgb = front.read_resized(h, w)
        measured = arms.get_arm_positions()
        obs = pack_obs(measured, front_rgb)
        logger.info(
            "image-no-motors: got camera frame shape=%s and %d arm joints; running policy",
            front_rgb.shape,
            len(measured),
        )
        print_action_trajectory(
            layout, policy, preprocess, postprocess, features, obs, device, measured=measured
        )
    finally:
        front.disconnect()
        arms.disconnect()  # state_only: no release / no arm_sdk writes
    logger.info("image-no-motors done (no motors commanded)")


def _graceful_interrupt(arms: G1Arms, iface: str | None, front) -> None:
    """Ctrl+C: stop loco velocity and hand arms back to the balancer."""
    from dds_init import ensure_dds

    logger.warning("Ctrl+C — graceful reset (StopMove + release arm_sdk)")
    try:
        ensure_dds(iface)
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

        loco = LocoClient()
        loco.SetTimeout(3.0)
        loco.Init()
        loco.StopMove()
        logger.info("LocoClient.StopMove() sent")
    except Exception as exc:  # noqa: BLE001
        logger.warning("StopMove during interrupt failed: %s", exc)
    try:
        arms.release()
        arms.detach()
    except Exception as exc:  # noqa: BLE001
        logger.warning("arm release during interrupt failed: %s", exc)
    try:
        front.disconnect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("camera disconnect during interrupt failed: %s", exc)


def run(args: argparse.Namespace, telemetry: Any | None = None) -> None:
    """Run ACT pickup.

    If ``telemetry`` is passed (soccerbot orchestrator), reuse that session and
    do **not** start/stop Rerun here — a second ``rr.init`` / ``shutdown_rerun``
    was wiping ``--record-path`` FileSinks and killing stages 2–4 recording.
    """
    import torch
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.utils import build_inference_frame, make_robot_action

    from telemetry import Telemetry

    if args.dry_run and args.image_no_motors:
        raise SystemExit("Use either --dry-run or --image-no-motors, not both.")

    layout = load_layout(args.layout)
    default_policy = getattr(layout, "DEFAULT_POLICY_ID", None)
    policy_arg = args.policy or default_policy
    if not policy_arg:
        raise SystemExit("--policy is required for --layout 14d (no default Hub id).")

    device = resolve_device(args.device)
    policy_ref = resolve_policy_ref(policy_arg)
    logger.info("Loading ACT policy %s on %s (layout=%s)", policy_ref, device, args.layout)

    policy = ACTPolicy.from_pretrained(policy_ref)
    policy.to(device)
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        pretrained_path=policy_ref,
    )
    features = layout.dataset_features()
    pack_obs = (
        layout.pack_observation
        if hasattr(layout, "pack_observation")
        else lambda arm_obs, rgb: pack_observation_16d(layout, arm_obs, rgb)
    )

    if args.dry_run:
        dry_run(layout, policy, preprocess, postprocess, device)
        return

    if args.image_no_motors:
        image_no_motors(args, layout, policy, preprocess, postprocess, features, pack_obs, device)
        return

    # One DDS init per process; shared by arms + camera clients.
    from dds_init import ensure_dds

    ensure_dds(args.iface)

    arms = G1Arms(kp=args.kp, kd=args.kd)
    front = make_front_camera(args.camera)

    arms.connect()
    front.connect()

    # Engage arm_sdk smoothly at the current pose before the policy takes over.
    arms.hold_current_pose(ramp_s=2.0)

    # Own a Telemetry session only when the orchestrator did not hand us one.
    owns_telemetry = telemetry is None
    if owns_telemetry:
        rerun_on = bool(getattr(args, "rerun", False)) and not bool(
            getattr(args, "no_rerun", False)
        )
        telemetry = Telemetry(
            enabled=rerun_on,
            session_name="soccerbot-act",
            record_path=getattr(args, "record_path", None),
            display=not bool(getattr(args, "no_display", False)),
        )
        telemetry.start()

    h, w, _ = layout.IMAGE_SHAPE
    dt = 1.0 / args.fps
    t0 = time.time()
    step = 0
    arm_keys = [f"{j}.q" for j in layout.ARM_JOINTS]
    # Commanded pose starts at the measured pose; each step it creeps toward
    # the policy target by at most --clamp rad, so motion stays slow no matter
    # what the policy outputs.
    cmd_q = dict(arms.get_arm_positions())

    log_path = args.log or time.strftime("act_log_%Y%m%d_%H%M%S.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    snapshot_keys = sorted(arms.get_full_snapshot())
    # Run config up top so a log file is self-describing (pandas: skiprows=1).
    log_file.write(f"# args: {vars(args)}\n")
    log_writer.writerow(
        ["t", "step", "cam_ms", "policy_ms", "loop_ms", "clamp_hits", "max_target_gap"]
        + [f"target_{j}" for j in layout.ARM_JOINTS]
        + [f"cmd_{j}" for j in layout.ARM_JOINTS]
        + snapshot_keys
    )
    logger.info(
        "ACT loop @ %.1f Hz via rt/arm_sdk, layout=%s, clamp=%.3f rad/step, log=%s "
        "(Ctrl+C = graceful reset: StopMove + release arm_sdk)",
        args.fps,
        args.layout,
        args.clamp,
        log_path,
    )

    interrupted = False
    leave_engaged = bool(getattr(args, "leave_arms_engaged", False))
    try:
        while True:
            loop_start = time.perf_counter()
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                logger.info("Duration reached (%.1fs); stopping", args.duration)
                break

            cam_start = time.perf_counter()
            front_rgb = front.read_resized(h, w)
            cam_ms = (time.perf_counter() - cam_start) * 1000

            snapshot = arms.get_full_snapshot()
            measured = {k: snapshot[k] for k in arm_keys}

            policy_start = time.perf_counter()
            obs = pack_obs(measured, front_rgb)
            frame = build_inference_frame(observation=obs, ds_features=features, device=device)
            batch = preprocess(frame)
            with torch.inference_mode():
                action = policy.select_action(batch)
            action = postprocess(action)
            robot_action = make_robot_action(action, features)
            dds_action = action_to_dds(layout, robot_action)
            policy_ms = (time.perf_counter() - policy_start) * 1000

            clamp_hits = 0
            if args.clamp and args.clamp > 0:
                for key in arm_keys:
                    delta = float(dds_action[key]) - cmd_q[key]
                    if abs(delta) > args.clamp:
                        clamp_hits += 1
                    cmd_q[key] += float(np.clip(delta, -args.clamp, args.clamp))
            else:
                for key in arm_keys:
                    cmd_q[key] = float(dds_action[key])
            arms.send_arm_positions(cmd_q)

            gaps = [abs(float(dds_action[k]) - measured[k]) for k in arm_keys]
            max_gap = max(gaps)
            loop_ms = (time.perf_counter() - loop_start) * 1000
            elapsed = time.time() - t0
            log_writer.writerow(
                [
                    round(elapsed, 4),
                    step,
                    round(cam_ms, 1),
                    round(policy_ms, 1),
                    round(loop_ms, 1),
                    clamp_hits,
                    round(max_gap, 5),
                ]
                + [round(float(dds_action[k]), 5) for k in arm_keys]
                + [round(cmd_q[k], 5) for k in arm_keys]
                + [round(snapshot[k], 5) for k in snapshot_keys]
            )

            telemetry.log_step(
                step=step,
                elapsed_s=elapsed,
                rgb=front_rgb,
                measured=measured,
                target=dds_action,
                commanded=cmd_q,
                extras={
                    "clamp_hits": float(clamp_hits),
                    "max_target_gap": float(max_gap),
                    "cam_ms": float(cam_ms),
                    "policy_ms": float(policy_ms),
                    "imu_pitch": float(snapshot.get("imu.pitch", 0.0)),
                    "imu_roll": float(snapshot.get("imu.roll", 0.0)),
                },
                stage="pickup",
            )

            step += 1
            if step % int(args.fps) == 0:
                worst = arm_keys[int(np.argmax(gaps))]
                leg_dq = max(
                    abs(v) for k, v in snapshot.items() if ".dq" in k and ("Hip" in k or "Knee" in k or "Ankle" in k)
                )
                logger.info(
                    "step=%d elapsed=%.1fs | target gap max=%.3f rad (%s) clamped=%d/14 | "
                    "leg max|dq|=%.2f rad/s | cam=%.0fms policy=%.0fms",
                    step,
                    elapsed,
                    max_gap,
                    worst.removeprefix("k").removesuffix(".q"),
                    clamp_hits,
                    leg_dq,
                    cam_ms,
                    policy_ms,
                )

            sleep = dt - (time.perf_counter() - loop_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        interrupted = True
        _graceful_interrupt(arms, args.iface, front)
    finally:
        log_file.close()
        logger.info("Step log written to %s (%d steps)", log_path, step)
        if owns_telemetry:
            telemetry.stop()
        if not interrupted:
            try:
                front.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("camera disconnect failed: %s", exc)
            if leave_engaged:
                # Hold last pose, then drop local DDS handles so the next
                # in-process stage can own rt/arm_sdk without a second publisher.
                try:
                    arms.freeze(cmd_q)
                    arms.detach()
                    logger.info(
                        "Clean exit: arm_sdk left engaged; local publisher detached for next stage"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("leave-engaged freeze/detach failed: %s", exc)
            else:
                arms.disconnect()
        logger.info("Done (interrupted=%s leave_engaged=%s)", interrupted, leave_engaged)
        if interrupted:
            raise


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
