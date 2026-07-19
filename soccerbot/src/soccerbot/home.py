"""Move G1 arms to a home pose (slew-clamped).

Unitree's arm SDK treats home as all arm joints at 0 rad
(``ctrl_dual_arm_go_home``). Override with a JSON file if needed:

    {
      "kind": "arm_qpos_14d",
      "joints_order": ["kLeftShoulderPitch", ...],
      "q": [0, 0, ...]
    }

or a flat ``{"kLeftShoulderPitch.q": 0.0, ...}`` map.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from soccerbot.deps import LOCAL_VLA_DIR, REPO_ROOT, _ensure_front

logger = logging.getLogger(__name__)

# Same joint order as local-vla-inference G1Arms / 14-D layout.
ARM_JOINTS: tuple[str, ...] = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)

# Unitree arm SDK home = zeros.
DEFAULT_HOME_Q: dict[str, float] = {f"{name}.q": 0.0 for name in ARM_JOINTS}

DEFAULT_HOME_JSON = REPO_ROOT / "scripted-behavior" / "home_pose.json"
# Match ACT's cautious rate: home/reset is recovery, not a performance move.
# At 50 Hz this is ~0.1 rad/s per joint — slow on purpose after jerky resets.
HOME_SLEW_CLAMP = 0.002  # rad/step
HOME_CONTROL_DT = 0.02
HOME_MAX_DURATION_S = 90.0  # allow full travel under the tight slew cap


def load_home_pose(path: Path | None = None) -> dict[str, float]:
    """Load home pose from JSON, or return the Unitree zero-pose default."""
    candidate = path or DEFAULT_HOME_JSON
    if candidate is None or not Path(candidate).is_file():
        return dict(DEFAULT_HOME_Q)

    with open(candidate) as f:
        data = json.load(f)

    if isinstance(data, dict) and "q" in data and "joints_order" in data:
        joints = data["joints_order"]
        qs = data["q"]
        if len(joints) != 14 or len(qs) != 14:
            raise ValueError(f"home pose must be 14-D, got {len(joints)}/{len(qs)}")
        return {f"{name}.q": float(q) for name, q in zip(joints, qs)}

    if isinstance(data, dict) and all(isinstance(v, (int, float)) for v in data.values()):
        out = dict(DEFAULT_HOME_Q)
        for key, value in data.items():
            k = key if key.endswith(".q") else f"{key}.q"
            if k in out:
                out[k] = float(value)
        return out

    raise ValueError(f"Unrecognized home pose schema in {candidate}")


def go_home(
    *,
    iface: str | None = None,
    pose: dict[str, float] | None = None,
    pose_path: Path | None = None,
    slew_clamp: float = HOME_SLEW_CLAMP,
    duration_s: float | None = None,
    release_after: bool = False,
) -> None:
    """Slew-limit interpolate arms from current pose to home.

    Engages ``arm_sdk`` if needed. By default leaves it engaged at home so a
    following stage can take over; pass ``release_after=True`` to hand back.
    """
    from soccerbot.safety import _ensure_dds

    _ensure_dds(iface)
    _ensure_front(LOCAL_VLA_DIR)
    from g1_arms import G1Arms

    from g1_arms import ARM_JOINT_LIMITS

    target = pose if pose is not None else load_home_pose(pose_path)
    missing = [k for k in DEFAULT_HOME_Q if k not in target]
    if missing:
        raise ValueError(f"home pose missing joints: {missing}")
    # Refuse out-of-limit custom poses up front (send_arm_positions would clamp
    # them anyway, but a bad home_pose.json should fail loudly, not silently).
    for name, (lo, hi) in ARM_JOINT_LIMITS.items():
        q = float(target[f"{name}.q"])
        if not (lo <= q <= hi):
            raise ValueError(
                f"home pose joint {name}={q:.3f} outside URDF limits [{lo:.3f}, {hi:.3f}]"
            )

    arms = G1Arms(kp=60.0, kd=1.5)
    arms.connect()
    try:
        start = arms.get_arm_positions()
        # Engage without the 0->1 weight ramp: if arm_sdk is already engaged
        # (e.g. mid-demo), ramping from 0 hands the arms back to the balancer
        # for ~0.5s and causes a visible jerk (see HANDOVER §5). A single
        # weight=1 publish is a no-op when engaged and an instant
        # engage-at-current-pose otherwise.
        arms.send_arm_positions(start, weight=1.0)

        # Estimate duration from max joint delta if not provided.
        # Must not truncate below steps_needed or the slew clamp never reaches home.
        if duration_s is None:
            max_delta = max(abs(target[k] - start.get(k, 0.0)) for k in DEFAULT_HOME_Q)
            steps_needed = max(1, int(max_delta / max(slew_clamp, 1e-6)) + 1)
            duration_s = max(2.0, steps_needed * HOME_CONTROL_DT)
            if duration_s > HOME_MAX_DURATION_S:
                logger.warning(
                    "Home travel needs %.1fs at slew=%.3f; capping at %.1fs",
                    duration_s,
                    slew_clamp,
                    HOME_MAX_DURATION_S,
                )
                duration_s = HOME_MAX_DURATION_S

        logger.info(
            "Going home over %.1fs (slew=%.3f rad/step, release_after=%s)",
            duration_s,
            slew_clamp,
            release_after,
        )
        _interpolate_clamped(arms, start, target, duration_s, slew_clamp)

        # Hold home briefly so it settles.
        settle_end = time.monotonic() + 0.4
        while time.monotonic() < settle_end:
            arms.send_arm_positions(target, weight=1.0)
            time.sleep(HOME_CONTROL_DT)
        logger.info("Home pose reached")
    finally:
        if release_after:
            arms.disconnect()  # release + detach
        else:
            arms.detach()  # keep arm_sdk engaged at home for next owner


def _interpolate_clamped(
    arms: Any,
    start: dict[str, float],
    target: dict[str, float],
    duration_s: float,
    slew_clamp: float,
) -> None:
    keys = list(DEFAULT_HOME_Q)
    cmd = {k: float(start.get(k, 0.0)) for k in keys}
    num_steps = max(1, int(duration_s / HOME_CONTROL_DT))
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()
        alpha = step / num_steps
        desired = {
            k: float(start.get(k, 0.0)) * (1.0 - alpha) + float(target[k]) * alpha for k in keys
        }
        if slew_clamp > 0:
            for k in keys:
                delta = desired[k] - cmd[k]
                if abs(delta) > slew_clamp:
                    delta = slew_clamp if delta > 0 else -slew_clamp
                cmd[k] = cmd[k] + delta
        else:
            cmd = desired
        arms.send_arm_positions(cmd, weight=1.0)
        sleep = HOME_CONTROL_DT - (time.perf_counter() - t0)
        if sleep > 0:
            time.sleep(sleep)
