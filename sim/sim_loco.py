"""LocoClient-compatible wrapper using Isaac Sim base teleportation.

Drop-in replacement for ``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient``.
Moves the robot base kinematically (no dynamics) at the requested velocity.
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger("sim.sim_loco")

# Kinematic integration timestep — how often the caller is expected to call Move.
_DT = 0.02  # 50 Hz control loop assumed


class LocoClient:
    """Isaac Sim kinematic locomotion matching the Unitree LocoClient API."""

    def __init__(self):
        self._robot = None
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        self._yaw = 0.0  # accumulated yaw (rad)
        self._pos = np.array([0.0, 0.0, 0.0])
        self._last_move_time: float | None = None

    def Init(self) -> None:
        logger.info("SimLocoClient: Init")

    def SetTimeout(self, t: float) -> None:
        pass  # No-op in sim.

    def Move(self, vx: float, vy: float, yaw_rate: float) -> None:
        now = time.monotonic()
        if self._last_move_time is not None:
            dt = min(now - self._last_move_time, 0.1)  # cap to avoid jumps
        else:
            dt = _DT
        self._last_move_time = now

        # Integrate yaw.
        self._yaw += yaw_rate * dt
        # Integrate position in the robot's local frame, rotated to world.
        cos_y = np.cos(self._yaw)
        sin_y = np.sin(self._yaw)
        dx_world = vx * cos_y - vy * sin_y
        dy_world = vx * sin_y + vy * cos_y
        self._pos[0] += dx_world * dt
        self._pos[1] += dy_world * dt

        self._vx = vx
        self._vy = vy
        self._yaw_rate = yaw_rate

        # Apply to Isaac Sim robot prim if available.
        if self._robot is not None:
            from isaacsim.core.utils.rotations import euler_angles_to_quat

            quat = euler_angles_to_quat(np.array([0.0, 0.0, self._yaw]))
            self._robot.set_world_pose(position=self._pos, orientation=quat)

    def StopMove(self) -> None:
        self._vx = 0.0
        self._vy = 0.0
        self._yaw_rate = 0.0
        logger.info("SimLocoClient: StopMove at pos=(%.2f, %.2f), yaw=%.1f°",
                    self._pos[0], self._pos[1], np.degrees(self._yaw))

    def bind_robot(self, robot) -> None:
        """Sim-only: attach the Isaac articulation so Move() teleports it."""
        self._robot = robot
        pos = robot.get_world_pose()[0]
        self._pos = np.array(pos, dtype=float)
