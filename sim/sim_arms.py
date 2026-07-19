"""G1Arms-compatible wrapper using Isaac Sim articulation API.

Drop-in replacement for ``local-vla-inference/g1_arms.py:G1Arms``.
Commands are applied to the Isaac Sim articulation instead of real DDS.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger("sim.sim_arms")

# Same 14 arm joint names as the real G1Arms (indices 15-28 in G1_29 convention).
LEFT_ARM_JOINTS = [
    "LeftShoulderPitch",
    "LeftShoulderRoll",
    "LeftShoulderYaw",
    "LeftElbow",
    "LeftWristRoll",
    "LeftWristPitch",
    "LeftWristYaw",
]
RIGHT_ARM_JOINTS = [
    "RightShoulderPitch",
    "RightShoulderRoll",
    "RightShoulderYaw",
    "RightElbow",
    "RightWristRoll",
    "RightWristPitch",
    "RightWristYaw",
]
ARM_JOINT_NAMES = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS


class SimG1Arms:
    """Isaac Sim backend matching the G1Arms interface."""

    def __init__(self, kp: float = 60.0, kd: float = 1.5, state_timeout_s: float = 10.0):
        self._kp = kp
        self._kd = kd
        self._state_timeout_s = state_timeout_s
        self._robot = None
        self._arm_dof_indices: list[int] = []
        self._connected = False
        # Simulated IMU state (no real IMU in sim — report zeros).
        self._imu = {"imu.yaw": 0.0, "imu.pitch": 0.0, "imu.roll": 0.0}

    def connect(self, state_only: bool = False) -> None:
        from sim.scene import build_scene

        # Lazy scene creation — reuse if already built.
        if self._robot is None:
            _, self._robot, _ = build_scene()
        # Build name-to-index mapping for the arm DOFs.
        dof_names = list(self._robot.dof_names)
        self._arm_dof_indices = []
        for name in ARM_JOINT_NAMES:
            # Isaac Sim may use slightly different naming; try common variants.
            idx = _find_dof_index(dof_names, name)
            self._arm_dof_indices.append(idx)
        self._connected = True
        logger.info("SimG1Arms connected (%d arm DOFs mapped)", len(self._arm_dof_indices))

    def get_arm_positions(self) -> dict[str, float]:
        positions = self._robot.get_joint_positions()
        return {
            f"{name}.q": float(positions[idx])
            for name, idx in zip(ARM_JOINT_NAMES, self._arm_dof_indices)
        }

    def get_full_snapshot(self) -> dict[str, float]:
        snapshot = self.get_arm_positions()
        snapshot.update(self._imu)
        return snapshot

    def send_arm_positions(self, action: dict[str, float], weight: float = 1.0) -> None:
        if weight <= 0.0:
            return
        targets = self._robot.get_joint_positions().copy()
        for name, idx in zip(ARM_JOINT_NAMES, self._arm_dof_indices):
            key = f"{name}.q"
            if key in action:
                targets[idx] = action[key]
        self._robot.set_joint_positions(targets)
        # Step the sim so changes take effect.
        self._robot._world.step(render=True)

    def hold_current_pose(self, ramp_s: float = 2.0, control_dt: float = 0.02) -> None:
        logger.info("SimG1Arms: hold_current_pose (no-op in sim)")

    def freeze(self, hold: dict | None = None) -> None:
        if hold is not None:
            self.send_arm_positions(hold)

    def release(self, ramp_s: float = 1.0, control_dt: float = 0.02) -> None:
        logger.info("SimG1Arms: release (no-op in sim)")

    def disconnect(self) -> None:
        self._connected = False
        logger.info("SimG1Arms: disconnected")


def _find_dof_index(dof_names: list[str], target: str) -> int:
    """Find DOF index by name, trying common Isaac Sim naming conventions."""
    # Exact match first.
    if target in dof_names:
        return dof_names.index(target)
    # Try lowercase, underscore variants.
    target_lower = target.lower()
    for i, name in enumerate(dof_names):
        if name.lower() == target_lower:
            return i
        # Isaac Sim often uses e.g. "left_shoulder_pitch" instead of "LeftShoulderPitch".
        normalised = name.lower().replace("_", "")
        if normalised == target_lower.replace("_", ""):
            return i
    # Fallback: log warning and use positional guess.
    logger.warning("DOF %r not found in %s — using index 0 as fallback", target, dof_names[:5])
    return 0
