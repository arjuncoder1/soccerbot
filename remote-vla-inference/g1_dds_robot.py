"""Direct-DDS arms-only Robot adapter for the LeRobot async ``RobotClient``.

This mirrors exactly how ``local-vla-inference`` talks to the G1 and fixes the
"Timed out waiting for robot state" hang the remote client hit:

  - LeRobot's stock ``UnitreeG1`` (``is_simulation=false``) uses a ZMQ **socket
    bridge** (``unitree_sdk2_socket.py``) that connects to
    ``tcp://<robot_ip>:<ports>`` and REQUIRES ``run_g1_server.py`` running on
    the robot's Orin to bridge DDS<->ZMQ. Nothing runs there, so no
    ``rt/lowstate`` ever arrives -> timeout.

  - The local inference never uses that bridge. It talks **direct DDS via the
    real Unitree SDK**: ``ChannelFactoryInitialize(0, iface)``, subscribe
    ``rt/lowstate`` for state, publish ``rt/arm_sdk`` for arm targets. This runs
    alongside the robot's stock balance controller (legs untouched) and needs
    nothing extra on the robot.

This adapter wraps the validated local ``G1Arms`` (rt/arm_sdk) and
``make_front_camera`` (teleimager) behind the LeRobot ``Robot`` interface so the
async ``RobotClient`` can drive π0.5 over the exact same transport as local.

Observation (matches the smoke/checkpoint feature layout):
  - 29 joint positions ``<joint_name>.q`` in ``G1_29_JointIndex`` order
    (concatenated by LeRobot into a 29-D ``observation.state``).
  - one RGB image ``global_view`` (H, W, 3), resized to 480x640.

Action: the client emits 14 arm ``.q`` targets; we publish them via arm_sdk.
Legs / waist / hands are never commanded (stock balancer keeps the robot up).
"""

import logging
import os
import sys
import threading

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.robots.unitree_g1.config_unitree_g1 import UnitreeG1Config
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex

logger = logging.getLogger("remote_vla_g1_dds")

# Reuse the validated local DDS arm controller + camera sources verbatim.
_LOCAL_DIR = os.environ.get(
    "LOCAL_VLA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local-vla-inference"),
)
if _LOCAL_DIR not in sys.path:
    sys.path.insert(0, _LOCAL_DIR)

from front_camera import make_front_camera  # noqa: E402
from g1_arms import G1Arms  # noqa: E402

STATE_JOINTS = tuple(j.name for j in G1_29_JointIndex)  # 29, index order
ARM_JOINTS = tuple(j.name for j in G1_29_JointArmIndex)  # 14
IMAGE_KEY = "global_view"
IMAGE_H, IMAGE_W = 480, 640


class G1ArmsDDSRobot(Robot):
    """Arms-only G1 over direct DDS (rt/lowstate in, rt/arm_sdk out).

    Presented to LeRobot's ``RobotClient`` as a ``unitree_g1`` robot so
    ``map_robot_keys_to_lerobot_features`` and feature routing work unchanged.
    """

    config_class = UnitreeG1Config
    name = "unitree_g1"

    def __init__(
        self,
        config: UnitreeG1Config,
        *,
        iface: str | None = None,
        camera_spec: str | None = None,
        kp: float = 60.0,
        kd: float = 1.5,
        state_timeout_s: float = 10.0,
        engage_ramp_s: float = 2.0,
    ):
        super().__init__(config)
        self.config = config
        self.iface = iface or None
        # Default to the same head-cam stream the local inference validated.
        self.camera_spec = camera_spec or f"zmq://{config.robot_ip}:55555"
        self._engage_ramp_s = float(engage_ramp_s)

        self._arms = G1Arms(kp=kp, kd=kd, state_timeout_s=state_timeout_s)
        self._camera = make_front_camera(self.camera_spec)
        self._connected = False
        # LocoClient (stock locomotion service) used to StopMove() on shutdown so
        # the robot stops acting but stays standing on its balancer.
        self._loco = None
        # When frozen, disconnect() keeps arm_sdk engaged at the last pose.
        self._frozen = False
        # When stopped (Ctrl+C stop-and-stand), arms are already released.
        self._stopped = False

    # ---- feature descriptors (usable disconnected) -----------------------
    @property
    def observation_features(self) -> dict:
        feats: dict = {f"{name}.q": float for name in STATE_JOINTS}
        feats[IMAGE_KEY] = (IMAGE_H, IMAGE_W, 3)
        return feats

    @property
    def action_features(self) -> dict:
        return {f"{name}.q": float for name in ARM_JOINTS}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ---- lifecycle -------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        # One DDS init per process, exactly like local main.py.
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        if self.iface:
            logger.info("DDS ChannelFactoryInitialize(0, %r)", self.iface)
            ChannelFactoryInitialize(0, self.iface)
        else:
            logger.info("DDS ChannelFactoryInitialize(0) [default interface]")
            ChannelFactoryInitialize(0)

        logger.info("Connecting G1 arms via rt/arm_sdk (kp=%.1f kd=%.1f)...", self._arms.kp, self._arms.kd)
        self._arms.connect()

        logger.info("Connecting front camera: %s", self.camera_spec)
        self._camera.connect()

        # LocoClient talks to the stock locomotion service (same one the local
        # diag uses). Best-effort: if the motion service isn't available we can
        # still release the arms on shutdown.
        try:
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

            loco = LocoClient()
            loco.SetTimeout(3.0)
            loco.Init()
            self._loco = loco
            logger.info("LocoClient ready (StopMove on shutdown; balancer keeps standing).")
        except Exception as e:  # noqa: BLE001
            self._loco = None
            logger.warning("LocoClient init failed (%s); Ctrl+C will still release the arms.", e)

        # Safe engage: ramp arm_sdk weight 0->1 while holding the measured pose
        # so the arms do not jump when the policy takes over.
        self._arms.hold_current_pose(ramp_s=self._engage_ramp_s)
        self._connected = True
        logger.info("G1ArmsDDSRobot connected (arms overlay live, balancer untouched).")

    def _raw_lowstate(self):
        # G1Arms owns the single rt/lowstate subscriber; read its latest frame.
        with self._arms._lock:  # noqa: SLF001
            return self._arms._low_state  # noqa: SLF001

    def get_observation(self) -> dict:
        if not self._connected:
            raise RuntimeError("G1ArmsDDSRobot.get_observation() before connect()")
        state = self._raw_lowstate()
        if state is None:
            raise RuntimeError("No rt/lowstate received yet")
        obs: dict = {
            f"{joint.name}.q": float(state.motor_state[joint.value].q) for joint in G1_29_JointIndex
        }
        frame = self._camera.read_resized(IMAGE_H, IMAGE_W)
        obs[IMAGE_KEY] = np.ascontiguousarray(frame)
        return obs

    def send_action(self, action: dict) -> dict:
        if not self._connected:
            raise RuntimeError("G1ArmsDDSRobot.send_action() before connect()")
        # ``action`` carries arm ``<joint>.q`` targets (already safety-filtered by
        # the client). Publish via arm_sdk at full weight; ignore any non-arm keys.
        self._arms.send_arm_positions(action, weight=1.0)
        return action

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            if self._frozen:
                # Hold stiff: arm_sdk stays engaged at the last pose. Do NOT release.
                logger.info("Disconnect while FROZEN: keeping arm_sdk engaged (no balancer handback).")
            elif self._stopped:
                # stop_and_stand() already released the arms; nothing more to do.
                logger.info("Disconnect after stop-and-stand: arms already released, robot standing.")
            else:
                self._arms.disconnect()  # ramps arm_sdk weight -> 0 (handback to balancer)
        finally:
            try:
                self._camera.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.warning("Camera disconnect failed: %s", e)
            self._connected = False
            logger.info("G1ArmsDDSRobot disconnected (frozen=%s stopped=%s).", self._frozen, self._stopped)

    def stop_and_stand(self) -> None:
        """Ctrl+C stop: stop everything the client is doing but leave the robot
        standing on its stock balancer.

        - ``LocoClient.StopMove()``: zero locomotion velocity (stop any walking);
          the balancer keeps the robot upright (does NOT go limp).
        - Release the ``arm_sdk`` overlay (ramp weight -> 0) so the arms stop and
          hand back to the balancer's own hold.
        """
        if self._loco is not None:
            try:
                self._loco.StopMove()
                logger.info("LocoClient.StopMove() sent — robot stops moving, stays standing.")
            except Exception as e:  # noqa: BLE001
                logger.warning("StopMove failed: %s", e)
        try:
            self._arms.release()  # ramp arm_sdk weight -> 0 (arms back to balancer)
            logger.info("arm_sdk released — arms handed back to the balancer.")
        except Exception as e:  # noqa: BLE001
            logger.warning("arm release failed: %s", e)
        self._stopped = True

    def freeze(self, hold: dict | None = None) -> None:
        """Hold the arms stiff at the last pose (arm_sdk stays engaged). Kept for
        callers that want a stiff hold instead of the default stop-and-stand."""
        self._arms.freeze(hold)
        self._frozen = True
