"""G1 arms-only RobotClient for Modal remote π0.5 inference.

Default checkpoint: sudoping01/pi05_g1_boxmove_v2 (LeRobot ``pi05``).
Action is 18-D (Unitree upper-body style). This client applies the first 14
dims as L/R arm joints and masks grippers / legs / waist / remote.

Safety / logging (adapted from ``local-vla-inference``):
  - Slew-rate limit: every control step each arm joint may move at most
    ``ARM_SLEW_CLAMP`` rad toward the policy target (default 0.01 rad/step,
    ~0.3 rad/s @ 30 fps). Set ``ARM_SLEW_CLAMP=0`` to disable. This is the
    primary safety net: no matter what the policy emits, motion stays slow.
  - CSV step log (``LOG_CSV``): per-step target/commanded/measured per joint,
    plus clamp hits and max target gap. Empty ``LOG_CSV`` disables.
  - Stop on Ctrl+C: release the arm_sdk overlay and ``LocoClient.StopMove()`` so
    the robot stops acting but stays standing on its stock balancer.

  ./remote-vla-inference/run_client.sh --server_address=HOST:PORT
"""

import csv
import logging
import math
import os
import threading
import time
from typing import Any

import draccus
import torch

from lerobot.async_inference import robot_client as _robot_client_mod
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.robots.unitree_g1.g1_utils import (
    G1_29_JointArmIndex,
    G1_29_JointIndex,
)
from lerobot.utils.import_utils import register_third_party_plugins

logger = logging.getLogger("remote_vla_g1_client")

DEFAULT_POLICY = "sudoping01/pi05_g1_boxmove_v2"
PI05_G1_ACTION_DIM = 18
ARM_DIM = 14

ARM_JOINT_NAMES = tuple(j.name for j in G1_29_JointArmIndex)
LEG_WAIST_NAMES = tuple(
    j.name
    for j in G1_29_JointIndex
    if j.value < G1_29_JointArmIndex.kLeftShoulderPitch.value
)


def arms_only_action(action: dict[str, float]) -> dict[str, float]:
    """Keep only G1 arm ``*.q`` keys; drop legs, waist, hands, remote."""
    filtered: dict[str, float] = {}
    for key, value in action.items():
        if not key.endswith(".q"):
            continue
        joint = key[: -len(".q")]
        if joint in LEG_WAIST_NAMES or joint not in ARM_JOINT_NAMES:
            continue
        filtered[key] = value
    return filtered


class ArmsOnlyRobotClient(RobotClient):
    def __init__(
        self,
        config: RobotClientConfig,
        *,
        log_joints_every: int = 10,
        arm_relative: bool = False,
        slew_clamp: float = 0.01,
        log_csv: str | None = None,
    ):
        super().__init__(config)
        self._arm_keys = [f"{name}.q" for name in ARM_JOINT_NAMES]
        self._arm_relative = arm_relative
        self._log_every = max(1, log_joints_every)
        self._step = 0
        self._warned = False
        self._last_cmd: dict[str, float] | None = None

        # Safety: slew-rate limit toward policy target (rad per control step).
        self._slew_clamp = max(0.0, float(slew_clamp))
        # Commanded pose that creeps toward the target; seeded lazily from obs.
        self._cmd_q: dict[str, float] | None = None

        # CSV step log.
        self._csv_path = log_csv or None
        self._csv_file = None
        self._csv_writer = None
        self._t0 = time.time()
        if self._csv_path:
            self._open_csv()

    # ---- CSV logging -----------------------------------------------------
    def _open_csv(self) -> None:
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["t", "step", "clamp_hits", "max_target_gap"]
            + [f"target_{n}" for n in ARM_JOINT_NAMES]
            + [f"cmd_{n}" for n in ARM_JOINT_NAMES]
            + [f"meas_{n}" for n in ARM_JOINT_NAMES]
        )
        logger.info("Per-step CSV log: %s", self._csv_path)

    def _log_csv(
        self,
        target: dict[str, float],
        cmd: dict[str, float],
        measured: dict[str, float],
        clamp_hits: int,
        max_gap: float,
    ) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.writerow(
            [round(time.time() - self._t0, 4), self._step, clamp_hits, round(max_gap, 5)]
            + [round(float(target.get(k, float("nan"))), 5) for k in self._arm_keys]
            + [round(float(cmd.get(k, float("nan"))), 5) for k in self._arm_keys]
            + [round(float(measured.get(k, float("nan"))), 5) for k in self._arm_keys]
        )

    def close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            logger.info("CSV log closed: %s", self._csv_path)

    # ---- observation helpers --------------------------------------------
    def _hold_arm_action_from_obs(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {key: float(obs[key]) for key in self._arm_keys if key in obs}

    def _format_arm_vec(self, values: dict[str, float]) -> str:
        left = [values.get(f"{n}.q", float("nan")) for n in ARM_JOINT_NAMES[:7]]
        right = [values.get(f"{n}.q", float("nan")) for n in ARM_JOINT_NAMES[7:]]

        def fmt(xs: list[float]) -> str:
            return "[" + ", ".join(f"{x:+.3f}" for x in xs) + "]"

        return f"L{fmt(left)} R{fmt(right)}"

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        flat = action_tensor.detach().cpu().reshape(-1)
        n = flat.numel()

        if n < ARM_DIM:
            if not self._warned:
                logger.warning("Policy action dim=%d < arm DoF=%d; holding arms.", n, ARM_DIM)
                self._warned = True
            return arms_only_action(self._hold_arm_action_from_obs())

        if n != PI05_G1_ACTION_DIM and not self._warned:
            logger.info(
                "Policy action dim=%d (expected %d). Using first %d as arms; masking rest.",
                n,
                PI05_G1_ACTION_DIM,
                ARM_DIM,
            )
            self._warned = True

        arm_vec = flat[:ARM_DIM]
        current = self._hold_arm_action_from_obs()
        raw: dict[str, float] = {}
        for i, key in enumerate(self._arm_keys):
            val = float(arm_vec[i].item())
            if self._arm_relative:
                val = current.get(key, 0.0) + val
            raw[key] = float(max(-math.pi, min(math.pi, val)))
        return arms_only_action(raw)

    # ---- safety gate -----------------------------------------------------
    def apply_safety(self, target: dict[str, float]) -> dict[str, float]:
        """Slew-limit the policy target toward the last commanded pose.

        Returns the arms-only command actually sent, and logs it. Seeds the
        commanded pose from the measured pose on the first call so the arms
        start from where they are (no jump).
        """
        target = arms_only_action(target)
        measured = self._hold_arm_action_from_obs()

        if self._cmd_q is None:
            self._cmd_q = dict(measured) if measured else dict(target)

        clamp_hits = 0
        cmd: dict[str, float] = {}
        for key in self._arm_keys:
            tgt = float(target.get(key, self._cmd_q.get(key, measured.get(key, 0.0))))
            prev = float(self._cmd_q.get(key, measured.get(key, tgt)))
            if self._slew_clamp > 0.0:
                delta = tgt - prev
                if abs(delta) > self._slew_clamp:
                    clamp_hits += 1
                    delta = max(-self._slew_clamp, min(self._slew_clamp, delta))
                cmd[key] = prev + delta
            else:
                cmd[key] = tgt
        self._cmd_q = dict(cmd)

        gaps = [
            abs(float(target.get(k, measured.get(k, 0.0))) - measured.get(k, 0.0))
            for k in self._arm_keys
        ]
        max_gap = max(gaps) if gaps else 0.0

        self._step += 1
        self._log_csv(target, cmd, measured, clamp_hits, max_gap)
        if self._step % self._log_every == 0:
            delta_str = ""
            if self._last_cmd is not None:
                diffs = [abs(cmd.get(k, 0.0) - self._last_cmd.get(k, 0.0)) for k in self._arm_keys]
                delta_str = f" | cmdΔmax={max(diffs):.4f}"
            logger.info(
                "step=%d ARM cmd %s | meas %s | gapmax=%.3f clamped=%d/%d%s",
                self._step,
                self._format_arm_vec(cmd),
                self._format_arm_vec(measured),
                max_gap,
                clamp_hits,
                ARM_DIM,
                delta_str,
            )
        self._last_cmd = dict(cmd)
        return cmd

    def freeze_arms(self) -> None:
        """Re-command the last commanded pose so the arms hold instead of going
        limp when we stop (best-effort, mirrors local-vla freeze)."""
        if self._cmd_q is None:
            return
        try:
            for _ in range(3):
                self.robot.send_action(dict(self._cmd_q))
                time.sleep(0.02)
            logger.info("Arms frozen at last commanded pose (%s).", self._format_arm_vec(self._cmd_q))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to freeze arms: %s", e)


def _install_direct_dds_robot(robot_cfg) -> None:
    """Make ``RobotClient`` build a direct-DDS arms robot (rt/arm_sdk) instead of
    LeRobot's ZMQ socket bridge, matching how ``local-vla-inference`` connects.

    The stock ``UnitreeG1`` non-sim path needs ``run_g1_server.py`` running on the
    robot to bridge DDS<->ZMQ; without it there is no ``rt/lowstate`` and connect
    times out. Direct DDS talks to the robot's stock stack over the network
    interface, publishes arm targets via ``rt/arm_sdk`` (balancer untouched), and
    needs nothing extra on the robot.
    """
    from g1_dds_robot import G1ArmsDDSRobot

    iface = os.environ.get("G1_IFACE", "").strip() or None
    camera = os.environ.get("G1_CAMERA", "").strip() or None
    kp = float(os.environ.get("G1_ARM_KP", "60.0"))
    kd = float(os.environ.get("G1_ARM_KD", "1.5"))

    original_make = _robot_client_mod.make_robot_from_config

    def make_robot_from_config(cfg):
        if type(cfg).__name__ == "UnitreeG1Config" and not getattr(cfg, "is_simulation", True):
            logger.info(
                "Real robot: using direct-DDS arm_sdk transport (iface=%s, camera=%s).",
                iface or "<dds-default>",
                camera or f"zmq://{cfg.robot_ip}:55555",
            )
            return G1ArmsDDSRobot(cfg, iface=iface, camera_spec=camera, kp=kp, kd=kd)
        return original_make(cfg)

    _robot_client_mod.make_robot_from_config = make_robot_from_config


def _install_send_action_filter(client: ArmsOnlyRobotClient) -> None:
    robot = client.robot
    original = robot.send_action

    def send_action_arms_only(action: dict[str, Any]):
        cmd = client.apply_safety(dict(action))
        return original(cmd)

    robot.send_action = send_action_arms_only  # type: ignore[method-assign]


@draccus.wrap()
def main(cfg: RobotClientConfig):
    logging.basicConfig(level=logging.INFO)
    register_third_party_plugins()

    robot_cfg = cfg.robot
    if type(robot_cfg).__name__ != "UnitreeG1Config":
        raise SystemExit("Use --robot.type=unitree_g1")

    use_direct_dds = (not getattr(robot_cfg, "is_simulation", True)) and os.environ.get(
        "G1_DIRECT_DDS", "1"
    ) != "0"

    if use_direct_dds:
        # Arms overlay via rt/arm_sdk: legs stay on the robot's stock balancer,
        # so no LeRobot leg controller is used (and none can drive the legs).
        robot_cfg.controller = None
        _install_direct_dds_robot(robot_cfg)
    elif getattr(robot_cfg, "controller", None) is None:
        logger.warning("Enabling GrootLocomotionController so the VLA cannot drive legs.")
        robot_cfg.controller = "GrootLocomotionController"

    log_every = max(1, int(os.environ.get("LOG_JOINTS_EVERY", "5")))
    arm_relative = os.environ.get("ARM_ACTIONS_RELATIVE", "0") in ("1", "true", "True")
    slew_clamp = float(os.environ.get("ARM_SLEW_CLAMP", "0.01"))
    log_csv = os.environ.get("LOG_CSV", "").strip() or None

    logger.info(
        "Remote VLA client → %s | %s | %s | task=%r | log_every=%d | slew=%.3f rad/step | csv=%s",
        cfg.server_address,
        cfg.policy_type,
        cfg.pretrained_name_or_path,
        cfg.task,
        log_every,
        slew_clamp,
        log_csv or "off",
    )
    if slew_clamp <= 0.0:
        logger.warning("Slew-rate limit DISABLED (ARM_SLEW_CLAMP=0): arms follow raw policy targets!")

    client = ArmsOnlyRobotClient(
        cfg,
        log_joints_every=log_every,
        arm_relative=arm_relative,
        slew_clamp=slew_clamp,
        log_csv=log_csv,
    )
    _install_send_action_filter(client)

    if client.start():
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        try:
            client.control_loop(cfg.task)
        except KeyboardInterrupt:
            logger.info("Ctrl+C — stopping the robot: release arms + StopMove; balancer keeps it standing.")
            # Stop everything the client is doing but leave the robot standing on
            # its stock balancer (StopMove + release arm_sdk overlay).
            stop_and_stand = getattr(client.robot, "stop_and_stand", None)
            if callable(stop_and_stand):
                stop_and_stand()
            else:
                client.freeze_arms()
            client.stop()
            receiver.join(timeout=5.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
        finally:
            client.close_csv()


if __name__ == "__main__":
    main()
