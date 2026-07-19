"""G1 29-DoF stick-figure FK from real URDF joint origins (g1_29dof.urdf).

Returns polylines in pelvis frame (X forward, Y left, Z up) for Rerun viz.
Read-only geometry — never commands motors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]
IDENTITY: Mat3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _mmul(a: Mat3, b: Mat3) -> Mat3:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _mv(a: Mat3, v: Vec3) -> Vec3:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))  # type: ignore[return-value]


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _rx(t: float) -> Mat3:
    c, s = math.cos(t), math.sin(t)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def _ry(t: float) -> Mat3:
    c, s = math.cos(t), math.sin(t)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def _rz(t: float) -> Mat3:
    c, s = math.cos(t), math.sin(t)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


_AXIS = {"x": _rx, "y": _ry, "z": _rz}


@dataclass(frozen=True)
class J:
    dds: str  # e.g. kLeftHipPitch (no .q)
    origin: Vec3
    axis: str
    pre_y: float = 0.0  # fixed URDF rpy pitch before revolute
    pre_x: float = 0.0


# Pelvis → feet / waist (from unitree g1_29dof.urdf).
LEFT_LEG = [
    J("kLeftHipPitch", (0.0, 0.064452, -0.1027), "y"),
    J("kLeftHipRoll", (0.0, 0.052, -0.030465), "x", pre_y=-0.1749),
    J("kLeftHipYaw", (0.025001, 0.0, -0.12412), "z"),
    J("kLeftKnee", (-0.078273, 0.0021489, -0.17734), "y", pre_y=0.1749),
    J("kLeftAnklePitch", (0.0, 0.0, -0.30001), "y"),
    J("kLeftAnkleRoll", (0.0, 0.0, -0.017558), "x"),
]
RIGHT_LEG = [
    J("kRightHipPitch", (0.0, -0.064452, -0.1027), "y"),
    J("kRightHipRoll", (0.0, -0.052, -0.030465), "x", pre_y=-0.1749),
    J("kRightHipYaw", (0.025001, 0.0, -0.12412), "z"),
    J("kRightKnee", (-0.078273, -0.0021489, -0.17734), "y", pre_y=0.1749),
    J("kRightAnklePitch", (0.0, 0.0, -0.30001), "y"),
    J("kRightAnkleRoll", (0.0, 0.0, -0.017558), "x"),
]
WAIST = [
    J("kWaistYaw", (0.0, 0.0, 0.0), "z"),
    J("kWaistRoll", (-0.0039635, 0.0, 0.035), "x"),
    J("kWaistPitch", (0.0, 0.0, 0.019), "y"),
]
# Arms relative to torso_link (same numbers as g1_arm_fk.py).
LEFT_ARM = [
    J("kLeftShoulderPitch", (0.0039563, 0.10022, 0.23778), "y", pre_x=0.27931),
    J("kLeftShoulderRoll", (0.0, 0.038, -0.013831), "x", pre_x=-0.27925),
    J("kLeftShoulderYaw", (0.0, 0.00624, -0.1032), "z"),
    J("kLeftElbow", (0.015783, 0.0, -0.080518), "y"),
    J("kLeftWristRoll", (0.100, 0.00188791, -0.010), "x"),
    J("kLeftWristPitch", (0.038, 0.0, 0.0), "y"),
    J("kLeftWristYaw", (0.046, 0.0, 0.0), "z"),
]
RIGHT_ARM = [
    J("kRightShoulderPitch", (0.0039563, -0.10021, 0.23778), "y", pre_x=-0.27931),
    J("kRightShoulderRoll", (0.0, -0.038, -0.013831), "x", pre_x=0.27925),
    J("kRightShoulderYaw", (0.0, -0.00624, -0.1032), "z"),
    J("kRightElbow", (0.015783, 0.0, -0.080518), "y"),
    J("kRightWristRoll", (0.100, -0.00188791, -0.010), "x"),
    J("kRightWristPitch", (0.038, 0.0, 0.0), "y"),
    J("kRightWristYaw", (0.046, 0.0, 0.0), "z"),
]


def _walk(chain: list[J], q: dict[str, float], *, start_pos: Vec3, start_R: Mat3) -> tuple[list[Vec3], Vec3, Mat3]:
    pos, R = start_pos, start_R
    pts = [pos]
    for j in chain:
        pos = _add(pos, _mv(R, j.origin))
        if j.pre_y:
            R = _mmul(R, _ry(j.pre_y))
        if j.pre_x:
            R = _mmul(R, _rx(j.pre_x))
        R = _mmul(R, _AXIS[j.axis](float(q.get(j.dds, 0.0))))
        pts.append(pos)
    return pts, pos, R


def skeleton_from_snapshot(snap: dict[str, float]) -> dict[str, list[Vec3]]:
    """``snap`` keys like ``kLeftHipPitch.q`` from G1Arms.get_full_snapshot()."""
    q = {k.removesuffix(".q"): float(v) for k, v in snap.items() if k.endswith(".q")}
    origin = (0.0, 0.0, 0.0)
    I = IDENTITY
    left_leg, _, _ = _walk(LEFT_LEG, q, start_pos=origin, start_R=I)
    right_leg, _, _ = _walk(RIGHT_LEG, q, start_pos=origin, start_R=I)
    waist, torso_pos, torso_R = _walk(WAIST, q, start_pos=origin, start_R=I)
    left_arm, _, _ = _walk(LEFT_ARM, q, start_pos=torso_pos, start_R=torso_R)
    right_arm, _, _ = _walk(RIGHT_ARM, q, start_pos=torso_pos, start_R=torso_R)
    # Spine stub: pelvis → torso tip for readability.
    spine = [origin, torso_pos]
    return {
        "left_leg": left_leg,
        "right_leg": right_leg,
        "waist": waist,
        "spine": spine,
        "left_arm": left_arm,
        "right_arm": right_arm,
    }
