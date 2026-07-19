"""Pure-Python forward kinematics for the Unitree G1 arm, from the real URDF.

Source: unitreerobotics/unitree_ros, robots/g1_description/g1_29dof.urdf
(fetched via `gh api repos/unitreerobotics/unitree_ros/contents/...`). Every
origin xyz / axis / limit below is copied directly from that file's <joint>
tags -- not assumed, not guessed. No dependencies (pure `math`), so this can
run anywhere: on the robot, in CI, or standalone.

This exists so that scripted-behavior/throw.py's trajectory can be verified
numerically (does the hand actually move forward, does it stay within the
real joint limits, is the path monotonic) instead of eyeballed. See
`throw.py`'s module docstring for how it's used.

Frame: ROS/URDF convention, X=forward, Y=left, Z=up, relative to torso_link.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

IDENTITY: Mat3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def mat_mul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def mat_vec(a: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def rot_x(theta: float) -> Mat3:
    c, s = math.cos(theta), math.sin(theta)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def rot_y(theta: float) -> Mat3:
    c, s = math.cos(theta), math.sin(theta)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def rot_z(theta: float) -> Mat3:
    c, s = math.cos(theta), math.sin(theta)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


_AXIS_ROT = {"x": rot_x, "y": rot_y, "z": rot_z}


@dataclass(frozen=True)
class JointDef:
    name: str
    origin: Vec3
    axis: str  # 'x' | 'y' | 'z' -- all G1 arm joints are pure axis-aligned revolutes
    pre_roll: float = 0.0  # see module note below
    limit: tuple[float, float] | None = None


# --- Real G1 29-DoF URDF joint data ---
# origin <rpy> for shoulder_pitch/shoulder_roll is (roll, ~1e-4, ~1e-4) -- the
# pitch/yaw components are noise-level (<0.02 deg) and dropped; only the roll
# component is kept (`pre_roll`). All other arm joints have rpy=(0,0,0).
LEFT_ARM: list[JointDef] = [
    JointDef("shoulder_pitch", (0.0039563, 0.10022, 0.23778), "y", pre_roll=0.27931, limit=(-3.0892, 2.6704)),
    JointDef("shoulder_roll", (0, 0.038, -0.013831), "x", pre_roll=-0.27925, limit=(-1.5882, 2.2515)),
    JointDef("shoulder_yaw", (0, 0.00624, -0.1032), "z", limit=(-2.618, 2.618)),
    JointDef("elbow", (0.015783, 0, -0.080518), "y", limit=(-1.0472, 2.0944)),
    JointDef("wrist_roll", (0.100, 0.00188791, -0.010), "x", limit=(-1.9722, 1.9722)),
    JointDef("wrist_pitch", (0.038, 0, 0), "y", limit=(-1.6144, 1.6144)),
    JointDef("wrist_yaw", (0.046, 0, 0), "z", limit=(-1.6144, 1.6144)),
]
RIGHT_ARM: list[JointDef] = [
    JointDef("shoulder_pitch", (0.0039563, -0.10021, 0.23778), "y", pre_roll=-0.27931, limit=(-3.0892, 2.6704)),
    JointDef("shoulder_roll", (0, -0.038, -0.013831), "x", pre_roll=0.27925, limit=(-2.2515, 1.5882)),
    JointDef("shoulder_yaw", (0, -0.00624, -0.1032), "z", limit=(-2.618, 2.618)),
    JointDef("elbow", (0.015783, 0, -0.080518), "y", limit=(-1.0472, 2.0944)),
    JointDef("wrist_roll", (0.100, -0.00188791, -0.010), "x", limit=(-1.9722, 1.9722)),
    JointDef("wrist_pitch", (0.038, 0, 0), "y", limit=(-1.6144, 1.6144)),
    JointDef("wrist_yaw", (0.046, 0, 0), "z", limit=(-1.6144, 1.6144)),
]
# left_hand_palm_joint / right_hand_palm_joint (fixed), relative to wrist_yaw_link.
HAND_OFFSET_LEFT: Vec3 = (0.0415, 0.003, 0)
HAND_OFFSET_RIGHT: Vec3 = (0.0415, -0.003, 0)


def arm_fk(joints: list[JointDef], angles: dict[str, float], hand_offset: Vec3) -> Vec3:
    """Hand position relative to torso_link, given {joint_name: angle}."""
    pos: Vec3 = (0.0, 0.0, 0.0)
    orient: Mat3 = IDENTITY
    for j in joints:
        pos = vec_add(pos, mat_vec(orient, j.origin))
        if j.pre_roll:
            orient = mat_mul(orient, rot_x(j.pre_roll))
        angle = angles.get(j.name, 0.0)
        orient = mat_mul(orient, _AXIS_ROT[j.axis](angle))
    return vec_add(pos, mat_vec(orient, hand_offset))


def left_hand_position(angles: dict[str, float]) -> Vec3:
    return arm_fk(LEFT_ARM, angles, HAND_OFFSET_LEFT)


def right_hand_position(angles: dict[str, float]) -> Vec3:
    return arm_fk(RIGHT_ARM, angles, HAND_OFFSET_RIGHT)


def left_elbow_position(angles: dict[str, float]) -> Vec3:
    """Elbow joint position -- only depends on shoulder_pitch/roll/yaw, not
    the elbow angle itself. Useful for catching a pose where the hand ends up
    somewhere reasonable but the elbow swings behind the torso to get there
    (looks like the arm is bent backward, even though the hand path is fine)."""
    return arm_fk(LEFT_ARM[:3], angles, (0.0, 0.0, 0.0))


def right_elbow_position(angles: dict[str, float]) -> Vec3:
    return arm_fk(RIGHT_ARM[:3], angles, (0.0, 0.0, 0.0))


def check_limits(joints: list[JointDef], angles: dict[str, float]) -> list[str]:
    """Return a list of human-readable violations; empty if all within real limits."""
    errors = []
    for j in joints:
        if j.limit is None:
            continue
        v = angles.get(j.name, 0.0)
        lo, hi = j.limit
        if not (lo <= v <= hi):
            errors.append(f"{j.name}={v} outside [{lo}, {hi}]")
    return errors
