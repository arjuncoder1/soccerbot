"""Direct DDS arm control for Unitree G1 via the official ``rt/arm_sdk`` topic.

Mirrors ``example/g1/high_level/g1_arm7_sdk_dds_example.py`` from
unitree_sdk2_python: subscribe ``rt/lowstate`` for joint state, publish
``rt/arm_sdk`` LowCmd with joint 29 (kNotUsedJoint) as the enable weight.
Works alongside the robot's stock balance controller — nothing extra runs on
the robot.
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

# G1 29-DoF joint indices (arms only).
ARM_JOINT_INDEX: dict[str, int] = {
    "kLeftShoulderPitch": 15,
    "kLeftShoulderRoll": 16,
    "kLeftShoulderYaw": 17,
    "kLeftElbow": 18,
    "kLeftWristRoll": 19,
    "kLeftWristPitch": 20,
    "kLeftWristYaw": 21,
    "kRightShoulderPitch": 22,
    "kRightShoulderRoll": 23,
    "kRightShoulderYaw": 24,
    "kRightElbow": 25,
    "kRightWristRoll": 26,
    "kRightWristPitch": 27,
    "kRightWristYaw": 28,
}

WEIGHT_JOINT = 29  # kNotUsedJoint: q = 1 enables arm_sdk, 0 releases it

# Legs + waist yaw (read-only here; never commanded). Used for diagnosis
# logging to see the balance controller react (e.g. sidestepping).
LEG_JOINT_INDEX: dict[str, int] = {
    "kLeftHipPitch": 0,
    "kLeftHipRoll": 1,
    "kLeftHipYaw": 2,
    "kLeftKnee": 3,
    "kLeftAnklePitch": 4,
    "kLeftAnkleRoll": 5,
    "kRightHipPitch": 6,
    "kRightHipRoll": 7,
    "kRightHipYaw": 8,
    "kRightKnee": 9,
    "kRightAnklePitch": 10,
    "kRightAnkleRoll": 11,
    "kWaistYaw": 12,
}

# Full torso (waist) — yaw is also in LEG_JOINT_INDEX for balance logs.
TORSO_JOINT_INDEX: dict[str, int] = {
    "kWaistYaw": 12,
    "kWaistRoll": 13,
    "kWaistPitch": 14,
}


class G1Arms:
    """Arms-only G1 interface over DDS (rt/lowstate in, rt/arm_sdk out)."""

    def __init__(self, kp: float = 60.0, kd: float = 1.5, state_timeout_s: float = 10.0) -> None:
        self.kp = kp
        self.kd = kd
        self.state_timeout_s = state_timeout_s

        self._lock = threading.Lock()
        self._low_state = None
        self._first_state = threading.Event()
        self._publisher = None
        self._subscriber = None
        self._cmd = None
        self._crc = None
        self._state_only = False
        # Optional {joint_index: {'q','kp','kd'}} that ``send_arm_positions``
        # will always overwrite on top of the arm keys. Callers opt in via
        # ``lock_joint(...)``; empty by default so existing callers (e.g. the
        # ACT/pi0.5 policy paths) see identical behaviour.
        self._locked_joints: dict[int, dict[str, float]] = {}

    def connect(self, *, state_only: bool = False) -> None:
        """Requires ChannelFactoryInitialize() to have been called already.

        ``state_only=True`` only subscribes to ``rt/lowstate`` (no arm_sdk
        publish) — use for diagnosis / joint logging.
        """
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._state_only = state_only
        if not state_only:
            self._cmd = unitree_hg_msg_dds__LowCmd_()
            self._crc = CRC()
            self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._publisher.Init()

        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_low_state, 10)

        if not self._first_state.wait(timeout=self.state_timeout_s):
            raise TimeoutError(
                f"No rt/lowstate within {self.state_timeout_s}s. "
                "Check the DDS network interface (--iface) and that this machine "
                "is on the robot's network (192.168.123.x)."
            )
        if state_only:
            logger.info("G1 state connected (rt/lowstate OK, no arm_sdk)")
        else:
            logger.info("G1 arms connected (rt/lowstate OK, publishing rt/arm_sdk)")

    def _on_low_state(self, msg) -> None:
        with self._lock:
            self._low_state = msg
        self._first_state.set()

    def get_arm_positions(self) -> dict[str, float]:
        """Return {joint_name.q: position} for the 14 arm joints."""
        with self._lock:
            state = self._low_state
        if state is None:
            raise RuntimeError("No lowstate received yet")
        return {f"{name}.q": float(state.motor_state[idx].q) for name, idx in ARM_JOINT_INDEX.items()}

    def get_full_snapshot(self) -> dict[str, float]:
        """Everything useful from lowstate for diagnosis logging: arm q/dq/tau,
        leg+waist q/dq/tau (to see the balance controller react), IMU rpy and
        angular velocity, and mode_machine.
        """
        with self._lock:
            state = self._low_state
        if state is None:
            raise RuntimeError("No lowstate received yet")
        snap: dict[str, float] = {"mode_machine": float(state.mode_machine)}
        for name, idx in ARM_JOINT_INDEX.items():
            m = state.motor_state[idx]
            snap[f"{name}.q"] = float(m.q)
            snap[f"{name}.dq"] = float(m.dq)
            snap[f"{name}.tau"] = float(m.tau_est)
        for name, idx in LEG_JOINT_INDEX.items():
            m = state.motor_state[idx]
            snap[f"{name}.q"] = float(m.q)
            snap[f"{name}.dq"] = float(m.dq)
            snap[f"{name}.tau"] = float(m.tau_est)
        imu = state.imu_state
        snap["imu.roll"] = float(imu.rpy[0])
        snap["imu.pitch"] = float(imu.rpy[1])
        snap["imu.yaw"] = float(imu.rpy[2])
        snap["imu.gyro_x"] = float(imu.gyroscope[0])
        snap["imu.gyro_y"] = float(imu.gyroscope[1])
        snap["imu.gyro_z"] = float(imu.gyroscope[2])
        return snap

    def send_arm_positions(self, action: dict[str, float], weight: float = 1.0) -> None:
        """Publish arm joint targets. ``action`` keys are '<joint_name>.q'."""
        if self._publisher is None:
            raise RuntimeError("arm_sdk publisher not available (connected state_only)")
        cmd = self._cmd
        cmd.motor_cmd[WEIGHT_JOINT].q = float(np.clip(weight, 0.0, 1.0))
        for name, idx in ARM_JOINT_INDEX.items():
            key = f"{name}.q"
            if key not in action:
                continue
            cmd.motor_cmd[idx].q = float(action[key])
            cmd.motor_cmd[idx].dq = 0.0
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].kp = self.kp
            cmd.motor_cmd[idx].kd = self.kd
        # arm_sdk controls waist joints (12-14) in addition to arm joints (15-28)
        # per the official g1_arm7_sdk_dds_example. Always hold them at the current
        # measured position with full stiffness so they don't go limp when arm_sdk
        # takes over. _locked_joints overrides below if the caller pinned a specific
        # waist target.
        with self._lock:
            state = self._low_state
        if state is not None:
            for name, idx in TORSO_JOINT_INDEX.items():
                if idx not in self._locked_joints:
                    cmd.motor_cmd[idx].q = float(state.motor_state[idx].q)
                    cmd.motor_cmd[idx].dq = 0.0
                    cmd.motor_cmd[idx].tau = 0.0
                    cmd.motor_cmd[idx].kp = self.kp
                    cmd.motor_cmd[idx].kd = self.kd
        # Extra locked joints override the measured hold above.
        for idx, spec in self._locked_joints.items():
            cmd.motor_cmd[idx].q = float(spec["q"])
            cmd.motor_cmd[idx].dq = 0.0
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].kp = float(spec["kp"])
            cmd.motor_cmd[idx].kd = float(spec["kd"])
        cmd.crc = self._crc.Crc(cmd)
        self._publisher.Write(cmd)

    def lock_joint(self, index: int, q: float, kp: float, kd: float) -> None:
        """Register an extra joint that every ``send_arm_positions`` will
        also command. Used e.g. to pin waist yaw during arm-only replays so
        the (compromised) balancer cannot twist it around. Off by default.
        """
        self._locked_joints[int(index)] = {"q": float(q), "kp": float(kp), "kd": float(kd)}
        logger.info(
            "G1Arms.lock_joint(idx=%d, q=%.4f, kp=%.1f, kd=%.1f) engaged",
            index,
            q,
            kp,
            kd,
        )

    def unlock_joint(self, index: int) -> None:
        self._locked_joints.pop(int(index), None)

    def hold_current_pose(self, ramp_s: float = 2.0, control_dt: float = 0.02) -> None:
        """Ramp arm_sdk weight 0→1 while holding the current pose (safe engage)."""
        steps = max(1, int(ramp_s / control_dt))
        current = self.get_arm_positions()
        for i in range(steps):
            self.send_arm_positions(current, weight=(i + 1) / steps)
            time.sleep(control_dt)
        logger.info("arm_sdk engaged (weight=1.0), holding current pose")

    def freeze(self, hold: dict[str, float] | None = None) -> None:
        """Hold a pose and stop: send one last command with weight 1 and leave
        arm_sdk engaged. The controller keeps the last command, so the arms
        stay stiff exactly where they are (no handback to the stock controller).
        """
        if self._publisher is None:
            return
        try:
            self.send_arm_positions(hold or self.get_arm_positions(), weight=1.0)
            logger.info("arms FROZEN (arm_sdk stays engaged at last pose)")
        except Exception as e:
            logger.warning("Failed to freeze arms: %s", e)

    def release(self, ramp_s: float = 1.0, control_dt: float = 0.02) -> None:
        """Ramp weight back to 0 so the stock controller regains the arms."""
        if self._publisher is None:
            return
        try:
            current = self.get_arm_positions()
            steps = max(1, int(ramp_s / control_dt))
            for i in range(steps):
                self.send_arm_positions(current, weight=1.0 - (i + 1) / steps)
                time.sleep(control_dt)
            logger.info("arm_sdk released (weight=0)")
        except Exception as e:
            logger.warning("Failed to release arm_sdk cleanly: %s", e)

    def disconnect(self) -> None:
        if not self._state_only:
            self.release()
        self._publisher = None
        self._subscriber = None
