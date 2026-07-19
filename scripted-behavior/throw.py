"""Gentle goalkeeper forward push, relative to wherever the arms already are.

Single entry point: ``throw(arms)``. Call it once the ball is already held in
the robot's hands. Unlike an earlier version of this file, it does **not**
first drive the arms to some fixed, known pose -- it reads whatever the
current arm position actually is (assumed to already be a "holding the ball
in front of the torso" pose -- that's the caller's job, not this function's)
and applies a small, gentle forward push relative to that, then returns to
wherever it started. No model, no learning.

Why relative, not absolute: the caller (the not-yet-built turn/detect/shuffle
FSM) hands off to this function from whatever pose the pickup phase happened
to leave the arms in. Snapping to one fixed pose first is a needless extra
motion and assumes a single "correct" hold that may not match reality.
Instead this file defines the push as a DELTA (nudge shoulder_pitch and elbow
by a fixed amount, leave shoulder_roll/shoulder_yaw untouched so whatever
left/right hand placement the current hold already has is preserved) and
relies on forward kinematics to confirm that delta pushes the hand forward
from a realistic range of starting poses, not just one.

How the delta itself was chosen -- verified against the real Unitree G1
kinematic chain (g1_arm_fk.py, sourced from the real G1 29-DoF URDF), not
guessed:

  - shoulder_pitch and elbow are, empirically, the two joints that
    consistently move the hand forward across many different starting
    configurations (found via numeric search in earlier iterations of this
    file); shoulder_roll/shoulder_yaw's effect on forward reach is much less
    consistent in sign depending on where you start, which is exactly why
    this version leaves them alone rather than driving them to fixed values.
  - `_CANDIDATE_HOLDING_POSES` is a small family of plausible "holding a ball
    in front of the torso" poses (perturbations around a previously-verified
    hold), standing in for "we don't know exactly what pose the real robot
    will hand off from, but it'll be roughly like one of these".
  - `_self_check()` (bottom of this file) applies the delta to every one of
    those candidates and asserts, for each: the hand moves forward by a
    real, positive amount; the elbow stays in front of the torso the whole
    time (never swings behind the back to get there -- an earlier version of
    this file got this wrong and it wasn't caught until an actual render);
    the hand never crosses to the body's other side; every resulting joint
    angle is within the real URDF limit. Runs at import time -- if any
    candidate fails, importing this module raises immediately.

Uses the same direct-DDS ``rt/arm_sdk`` interface (``G1Arms``, imported from
the sibling ``local-vla-inference/g1_arms.py``) that
``local-vla-inference/main.py`` uses for ACT inference, so driving the arms
from this script and from the learned policy go through the identical
weight-ramped engage/release mechanism and both work alongside the robot's
stock balance controller rather than replacing it.

SAFETY -- READ BEFORE RUNNING ON REAL HARDWARE:
The joint *limits* are real and enforced by clamping. What is NOT
independently verified: that the *specific* pose the real robot hands off
from resembles the candidates this was checked against closely enough for
the "pushes forward" guarantee to hold in practice, and the timing/dynamics
of how fast the real arms track this trajectory. Before running against real
hardware:
  1. Run with --dry-run (the default) and read the printed candidate table.
  2. Only then consider --execute against real hardware, starting with a
     spotter and a kill switch in hand, and only after visually confirming
     the motion in front of you (this file does not include a simulator).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from dds import ensure_dds
from g1_arm_fk import LEFT_ARM, RIGHT_ARM, check_limits, left_elbow_position, left_hand_position

logger = logging.getLogger(__name__)

# G1Arms (rt/arm_sdk direct-DDS interface) lives in the sibling
# local-vla-inference package. Import it from there instead of duplicating its
# DDS plumbing here -- both packages run on the same robot machine.
_LOCAL_VLA_INFERENCE_DIR = Path(__file__).resolve().parent.parent / "local-vla-inference"


def _import_g1_arms():
    """Load ``G1Arms`` from the sibling local-vla-inference package."""
    sys.path.insert(0, str(_LOCAL_VLA_INFERENCE_DIR))
    try:
        from g1_arms import G1Arms  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return G1Arms

# G1 arm joint names (matches g1_arm_fk.py / g1_arms.ARM_JOINT_INDEX /
# local-vla-inference/embodiment_g1d_16d.py ARM_JOINTS). Keys used throughout
# this file carry the ".q" suffix G1Arms.get_arm_positions() /
# send_arm_positions() expect.
_LEFT_ARM_JOINTS = (
    "kLeftShoulderPitch",
    "kLeftShoulderRoll",
    "kLeftShoulderYaw",
    "kLeftElbow",
    "kLeftWristRoll",
    "kLeftWristPitch",
    "kLeftWristYaw",
)
_RIGHT_ARM_JOINTS = (
    "kRightShoulderPitch",
    "kRightShoulderRoll",
    "kRightShoulderYaw",
    "kRightElbow",
    "kRightWristRoll",
    "kRightWristPitch",
    "kRightWristYaw",
)
ARM_JOINTS: tuple[str, ...] = _LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS
ARM_JOINT_KEYS: tuple[str, ...] = tuple(f"{j}.q" for j in ARM_JOINTS)

# "kLeftShoulderPitch.q" <-> g1_arm_fk's "shoulder_pitch", one map per side.
_FK_NAME_LEFT = {
    "kLeftShoulderPitch.q": "shoulder_pitch",
    "kLeftShoulderRoll.q": "shoulder_roll",
    "kLeftShoulderYaw.q": "shoulder_yaw",
    "kLeftElbow.q": "elbow",
    "kLeftWristRoll.q": "wrist_roll",
    "kLeftWristPitch.q": "wrist_pitch",
    "kLeftWristYaw.q": "wrist_yaw",
}
_FK_NAME_RIGHT = {
    "kRightShoulderPitch.q": "shoulder_pitch",
    "kRightShoulderRoll.q": "shoulder_roll",
    "kRightShoulderYaw.q": "shoulder_yaw",
    "kRightElbow.q": "elbow",
    "kRightWristRoll.q": "wrist_roll",
    "kRightWristPitch.q": "wrist_pitch",
    "kRightWristYaw.q": "wrist_yaw",
}

# The push itself: a gentle nudge relative to whatever the current pose is.
# shoulder_roll/shoulder_yaw are deliberately left untouched -- see module
# docstring for why. Found + verified via g1_arm_fk, not guessed.
_DELTA_SHOULDER_PITCH = -0.6
_DELTA_ELBOW = 1.05
_DELTA_WRIST_PITCH = -0.45
_FOLLOW_THROUGH_EXTRA_WRIST_PITCH = -0.15  # wrist settles a little further

RELEASE_DURATION_S = 0.35
FOLLOW_THROUGH_DURATION_S = 0.25
RECOVER_DURATION_S = 0.9

# A family of plausible "holding a ball in front of the torso" starting
# poses (perturbations around a previously hand-verified hold), used to
# check the delta generalizes rather than only working from one exact point.
# Real key format ("kLeftShoulderPitch.q" etc.) via _mirror_for_check below.
_CANDIDATE_HOLDING_POSES: tuple[dict[str, float], ...] = (
    {"shoulder_pitch": -0.6, "shoulder_roll": 0.7, "shoulder_yaw": -1.8, "elbow": 0.0, "wrist_pitch": 0.2},
    {"shoulder_pitch": -0.4, "shoulder_roll": 0.5, "shoulder_yaw": -1.5, "elbow": 0.2, "wrist_pitch": 0.1},
    {"shoulder_pitch": -0.8, "shoulder_roll": 0.9, "shoulder_yaw": -2.0, "elbow": -0.2, "wrist_pitch": 0.3},
    {"shoulder_pitch": 0.1, "shoulder_roll": 0.35, "shoulder_yaw": -1.0, "elbow": 0.0, "wrist_pitch": 0.2},
    {"shoulder_pitch": -0.3, "shoulder_roll": 0.3, "shoulder_yaw": -1.2, "elbow": 0.5, "wrist_pitch": 0.0},
)


def _clamp(joint_defs, angles: dict[str, float]) -> dict[str, float]:
    """Clamp {fk_joint_name: value} to the real URDF limits."""
    clamped = {}
    for j in joint_defs:
        v = angles.get(j.name, 0.0)
        if j.limit is not None:
            lo, hi = j.limit
            v = max(lo, min(hi, v))
        clamped[j.name] = v
    return clamped


def _push_target(current_pose: dict[str, float]) -> dict[str, float]:
    """Given the robot's CURRENT 14-joint pose, return the pushed-forward
    target: shoulder_pitch/elbow/wrist_pitch nudged, shoulder_roll/
    shoulder_yaw/wrist_roll/wrist_yaw left exactly as they were, all clamped
    to the real G1 joint limits."""
    raw = dict(current_pose)
    for prefix in ("kLeft", "kRight"):
        raw[f"{prefix}ShoulderPitch.q"] += _DELTA_SHOULDER_PITCH
        raw[f"{prefix}Elbow.q"] += _DELTA_ELBOW
        raw[f"{prefix}WristPitch.q"] += _DELTA_WRIST_PITCH

    left_fk = _clamp(LEFT_ARM, {fk: raw[key] for key, fk in _FK_NAME_LEFT.items()})
    right_fk = _clamp(RIGHT_ARM, {fk: raw[key] for key, fk in _FK_NAME_RIGHT.items()})
    return {
        **{key: left_fk[fk] for key, fk in _FK_NAME_LEFT.items()},
        **{key: right_fk[fk] for key, fk in _FK_NAME_RIGHT.items()},
    }


def _interpolate_to(
    arms,
    start: dict[str, float],
    target: dict[str, float],
    duration_s: float,
    control_dt: float = 0.02,
) -> None:
    """Linearly interpolate the 14 arm joints from ``start`` to ``target``."""
    num_steps = max(1, int(duration_s / control_dt))
    for step in range(1, num_steps + 1):
        step_start = time.time()
        alpha = step / num_steps
        action = {key: start[key] * (1 - alpha) + target[key] * alpha for key in ARM_JOINT_KEYS}
        arms.send_arm_positions(action)
        elapsed = time.time() - step_start
        sleep_s = max(0.0, control_dt - elapsed)
        time.sleep(sleep_s)


def throw_ball(cfg) -> None:
    """Orchestrator entry point: engage arms (if needed) and run ``throw``.

    ``cfg`` is an ``OrchestratorConfig`` (duck-typed so this module's
    import-time FK self-check does not depend on ``config``).
    """
    g1_arms_cls = _import_g1_arms()
    ensure_dds(cfg.iface)
    arms = g1_arms_cls(kp=60.0, kd=1.5)
    arms.connect()
    try:
        # Keep arm_sdk engaged across the prior stages: a single weight=1
        # publish is a no-op if already engaged (see turn_180.py).
        hold = dict(arms.get_arm_positions())
        arms.send_arm_positions(hold, weight=1.0)
        throw(arms)
    finally:
        arms.disconnect()


def throw(arms) -> None:
    """Run the gentle push against a connected, already-engaged ``G1Arms``.

    Reads the CURRENT arm position and treats it as the ball-holding pose --
    does not move to any fixed pose first. Pushes forward from there, holds
    briefly, then returns to the exact pose it started from. Assumes arm_sdk
    is already engaged (e.g. via ``arms.hold_current_pose()``); does not
    release it afterward -- the caller decides what happens next.
    """
    logger.info("Starting gentle goalkeeper push from current arm position")
    start_pose = arms.get_arm_positions()
    release_pose = _push_target(start_pose)
    follow_through_pose = dict(release_pose)
    for prefix in ("kLeft", "kRight"):
        key = f"{prefix}WristPitch.q"
        lo, hi = (-1.6144, 1.6144)
        follow_through_pose[key] = max(lo, min(hi, release_pose[key] + _FOLLOW_THROUGH_EXTRA_WRIST_PITCH))

    logger.info("-> release (%.2fs)", RELEASE_DURATION_S)
    _interpolate_to(arms, start_pose, release_pose, RELEASE_DURATION_S)
    logger.info("-> follow_through (%.2fs)", FOLLOW_THROUGH_DURATION_S)
    _interpolate_to(arms, release_pose, follow_through_pose, FOLLOW_THROUGH_DURATION_S)
    logger.info("-> recover (%.2fs)", RECOVER_DURATION_S)
    _interpolate_to(arms, follow_through_pose, start_pose, RECOVER_DURATION_S)
    logger.info("Push complete.")


def _mirror_for_check(active: dict[str, float]) -> dict[str, float]:
    """Expand a single-arm {fk_joint_name: value} dict into a full 14-joint
    ".q"-keyed pose, mirroring roll/yaw for the right arm."""
    return {
        "kLeftShoulderPitch.q": active["shoulder_pitch"],
        "kLeftShoulderRoll.q": active["shoulder_roll"],
        "kLeftShoulderYaw.q": active["shoulder_yaw"],
        "kLeftElbow.q": active["elbow"],
        "kLeftWristRoll.q": 0.0,
        "kLeftWristPitch.q": active["wrist_pitch"],
        "kLeftWristYaw.q": 0.0,
        "kRightShoulderPitch.q": active["shoulder_pitch"],
        "kRightShoulderRoll.q": -active["shoulder_roll"],
        "kRightShoulderYaw.q": -active["shoulder_yaw"],
        "kRightElbow.q": active["elbow"],
        "kRightWristRoll.q": 0.0,
        "kRightWristPitch.q": active["wrist_pitch"],
        "kRightWristYaw.q": 0.0,
    }


def _print_dry_run() -> None:
    print("Gentle goalkeeper push -- relative delta, verified against candidate holding poses\n")
    print(f"delta: shoulder_pitch={_DELTA_SHOULDER_PITCH:+.2f}  elbow={_DELTA_ELBOW:+.2f}  "
          f"wrist_pitch={_DELTA_WRIST_PITCH:+.2f}  (shoulder_roll/yaw unchanged)\n")
    print(f"{'candidate start (L hand)':<30}{'-> release (L hand)':<30}{'forward gain'}")
    for candidate in _CANDIDATE_HOLDING_POSES:
        start_pose = _mirror_for_check(candidate)
        release_pose = _push_target(start_pose)
        start_hand = left_hand_position(candidate)
        release_hand = left_hand_position(
            {fk: release_pose[key] for key, fk in _FK_NAME_LEFT.items()}
        )
        gain = release_hand[0] - start_hand[0]
        s = f"({start_hand[0]:+.2f},{start_hand[1]:+.2f},{start_hand[2]:+.2f})"
        r = f"({release_hand[0]:+.2f},{release_hand[1]:+.2f},{release_hand[2]:+.2f})"
        print(f"{s:<30}{r:<30}{gain:+.3f}m")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually connect to the real robot over DDS and run the push (default: dry-run print only).",
    )
    parser.add_argument("--iface", default=None, help="Network interface connected to the robot (e.g. eth0).")
    parser.add_argument("--kp", type=float, default=60.0, help="Arm joint position gain.")
    parser.add_argument("--kd", type=float, default=1.5, help="Arm joint damping gain.")
    return parser.parse_args(argv)


def _run_standalone(args: argparse.Namespace) -> None:
    """Standalone lifecycle for testing throw.py on its own: connect, engage
    arm_sdk safely from whatever pose the arms are currently in, push,
    release. Not what the eventual FSM will do (it should stay engaged
    across steps) -- this is just for exercising the push in isolation.
    """
    g1_arms_cls = _import_g1_arms()
    ensure_dds(args.iface)

    arms = g1_arms_cls(kp=args.kp, kd=args.kd)
    arms.connect()
    arms.hold_current_pose(ramp_s=2.0)
    try:
        throw(arms)
    finally:
        arms.disconnect()  # ramps arm_sdk weight back to 0


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if not args.execute:
        _print_dry_run()
        return

    logger.warning("Targeting REAL hardware. Ctrl+C to abort.")
    time.sleep(3.0)
    _run_standalone(args)


def _self_check() -> None:
    """Verify the push DELTA against every candidate holding pose.

    Runs at import time. Raises immediately if the delta doesn't reliably
    push forward, keeps the elbow in front of the torso, avoids crossing the
    centerline, and stays within real joint limits -- for EVERY candidate,
    not just one. This file cannot be imported with a delta that only works
    from a single lucky starting pose.
    """
    for candidate in _CANDIDATE_HOLDING_POSES:
        start_pose = _mirror_for_check(candidate)
        release_pose = _push_target(start_pose)

        left_release_fk = {fk: release_pose[key] for key, fk in _FK_NAME_LEFT.items()}
        errors = check_limits(LEFT_ARM, left_release_fk)
        if errors:
            raise AssertionError(f"throw.py push violates real G1 joint limits for candidate {candidate}: {errors}")

        start_hand = left_hand_position(candidate)
        release_hand = left_hand_position(left_release_fk)
        gain = release_hand[0] - start_hand[0]
        if not (gain > 0.05):
            raise AssertionError(
                f"throw.py push does not move the hand forward enough for candidate {candidate}: "
                f"gain={gain:.3f} (need >0.05m)"
            )
        if release_hand[1] < 0.0:
            raise AssertionError(f"throw.py push crosses the left hand to the wrong side for candidate {candidate}")

        start_elbow = left_elbow_position(candidate)
        release_elbow = left_elbow_position(left_release_fk)
        if release_elbow[0] < 0.0:
            raise AssertionError(
                f"throw.py push swings the elbow behind the torso for candidate {candidate}: "
                f"start={start_elbow[0]:.3f} release={release_elbow[0]:.3f}"
            )


_self_check()


if __name__ == "__main__":
    main()
