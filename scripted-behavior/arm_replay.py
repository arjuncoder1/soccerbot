"""Generic 14-D arm-qpos trajectory replay over ``rt/arm_sdk``.

Trajectories live in ``scripted-behavior/trajectories/*.json`` with the
schema written by ``scripts/extract_pickup_trajectory.py`` (or produced
manually):

    {
      "name": "pickup_ep10",
      "fps": 30,
      "kind": "arm_qpos_14d",
      "joints_order": [ 14 joint names in G1_29 order ],
      "num_frames": N,
      "frames": [[q0..q13], [q0..q13], ...]
    }

Replay strategy:
  * Snapshot the current arm pose.
  * Ramp arm_sdk from 0 -> 1 while linearly interpolating the current
    pose to ``frames[0]`` over ``REPLAY_RAMP_S`` seconds, so the arms
    do not jump.
  * Stream ``frames`` at ``fps`` (paced with ``time.sleep`` against a
    monotonic clock).
  * Per-step slew-rate limit ``REPLAY_SLEW_CLAMP`` rad/frame protects
    against garbage frames; if a frame asks for more than that per joint,
    the command is clipped and a warning is logged.
  * Hard timeout ``max_duration_s`` (default = trajectory length + 2 s).
  * arm_sdk is left engaged on exit so the next stage inherits the
    arms without a re-engage jump.
"""

from __future__ import annotations

import json
import logging
import sys as _sys
import time
from pathlib import Path

from config import REPO_ROOT

logger = logging.getLogger("scripted_behavior.arm_replay")

# Smooth blend from current measured pose to frames[0] over this many seconds.
REPLAY_RAMP_S = 2.0
# Max radians any single joint may move per replay frame. At 30 fps this is
# 0.05 rad * 30 = 1.5 rad/s which is already quite fast; safety net only.
REPLAY_SLEW_CLAMP = 0.05


def _import_g1_arms():
    g1_arms_dir = str(REPO_ROOT / "local-vla-inference")
    _sys.path.insert(0, g1_arms_dir)
    try:
        from g1_arms import G1Arms  # type: ignore[import-not-found]
    finally:
        _sys.path.pop(0)
    return G1Arms


def load_trajectory(path: str | Path) -> dict:
    with open(path) as f:
        traj = json.load(f)
    if traj.get("kind") != "arm_qpos_14d":
        raise ValueError(f"unsupported trajectory kind: {traj.get('kind')!r}")
    if len(traj["joints_order"]) != 14:
        raise ValueError(
            f"expected 14 joint names, got {len(traj['joints_order'])}"
        )
    if any(len(f) != 14 for f in traj["frames"]):
        raise ValueError("all frames must have exactly 14 dims")
    return traj


def replay_arm_trajectory(
    trajectory_path: str | Path,
    *,
    iface: str | None,
    max_duration_s: float | None = None,
    slew_clamp: float = REPLAY_SLEW_CLAMP,
    speed_factor: float = 1.0,
    instant_engage: bool = False,
    arms=None,
) -> None:
    """Play back a JSON arm-qpos trajectory over ``rt/arm_sdk``.

    Blocks until the trajectory finishes or ``max_duration_s`` elapses.
    Leaves arm_sdk engaged.
    """
    traj = load_trajectory(trajectory_path)
    joints = traj["joints_order"]
    frames = traj["frames"]
    fps = float(traj.get("fps", 30.0)) * speed_factor
    dt = 1.0 / fps
    n = len(frames)
    nominal_s = n * dt
    if max_duration_s is None:
        max_duration_s = nominal_s + 2.0

    logger.info(
        "Replaying %s: %d frames @ %.1f Hz (~%.1fs), max=%.1fs, slew=%.3f rad/frame",
        traj.get("name", trajectory_path),
        n,
        fps,
        nominal_s,
        max_duration_s,
        slew_clamp,
    )

    if arms is None:
        G1Arms = _import_g1_arms()
        from dds import ensure_dds
        ensure_dds(iface)
        arms = G1Arms(kp=60.0, kd=1.5)
        arms.connect()

    def _pack(frame_qs: list[float]) -> dict[str, float]:
        return {f"{name}.q": float(q) for name, q in zip(joints, frame_qs)}

    # Snapshot current arm pose and blend into frames[0] over REPLAY_RAMP_S.
    current = arms.get_arm_positions()
    start_pose = [current.get(f"{name}.q", 0.0) for name in joints]
    target0 = frames[0]

    ramp_steps = max(1, int(REPLAY_RAMP_S * fps))
    for i in range(ramp_steps):
        alpha = (i + 1) / ramp_steps
        blended = [
            start_pose[j] + alpha * (target0[j] - start_pose[j])
            for j in range(14)
        ]
        w = 1.0 if instant_engage else alpha
        arms.send_arm_positions(_pack(blended), weight=w)
        time.sleep(dt)

    # Actual replay.
    t0 = time.monotonic()
    last_cmd = list(target0)
    clamp_hits = 0
    try:
        for k, frame in enumerate(frames):
            elapsed = time.monotonic() - t0
            if elapsed >= max_duration_s:
                logger.warning(
                    "Replay hard timeout at frame %d/%d (%.2fs)",
                    k,
                    n,
                    elapsed,
                )
                break

            clamped = []
            for j in range(14):
                delta = frame[j] - last_cmd[j]
                if slew_clamp > 0 and abs(delta) > slew_clamp:
                    clamp_hits += 1
                    delta = slew_clamp if delta > 0 else -slew_clamp
                clamped.append(last_cmd[j] + delta)
            last_cmd = clamped

            arms.send_arm_positions(_pack(clamped), weight=1.0)

            # Sleep to next scheduled tick, not just dt, so cumulative drift
            # doesn't stretch the replay.
            next_tick = t0 + (k + 1) * dt
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        logger.info(
            "Replay finished (%d frames sent, %d clamp hits)",
            k + 1 if n else 0,
            clamp_hits,
        )
        # Hold the final pose for a moment so the next stage starts stable.
        settle_end = time.monotonic() + 0.3
        while time.monotonic() < settle_end:
            arms.send_arm_positions(_pack(last_cmd), weight=1.0)
            time.sleep(dt)
