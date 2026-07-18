"""G1 arms-only RobotClient for Modal remote π0.5 inference.

Default checkpoint: sudoping01/pi05_g1_boxmove_v2 (LeRobot ``pi05``).
Action is 18-D (Unitree upper-body style). This client applies the first 14
dims as L/R arm joints and masks grippers / legs / waist / remote.

  ./remote-vla-inference/run_client.sh --server_address=HOST:PORT
"""

import logging
import math
import os
import threading
from typing import Any

import draccus
import torch

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
    def __init__(self, config: RobotClientConfig, *, log_joints_every: int = 10, arm_relative: bool = False):
        super().__init__(config)
        self._arm_keys = [f"{name}.q" for name in ARM_JOINT_NAMES]
        self._arm_relative = arm_relative
        self._log_every = max(1, log_joints_every)
        self._step = 0
        self._warned = False
        self._last_cmd: dict[str, float] | None = None

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

    def log_arm_joints(self, commanded: dict[str, float]) -> None:
        self._step += 1
        if self._step % self._log_every != 0:
            return
        measured = self._hold_arm_action_from_obs()
        delta = ""
        if self._last_cmd is not None:
            diffs = [abs(commanded.get(k, 0.0) - self._last_cmd.get(k, 0.0)) for k in self._arm_keys]
            delta = f" | cmdΔmax={max(diffs):.4f}"
        self._last_cmd = dict(commanded)
        logger.info(
            "step=%d ARM cmd %s | meas %s%s",
            self._step,
            self._format_arm_vec(commanded),
            self._format_arm_vec(measured),
            delta,
        )


def _install_send_action_filter(client: ArmsOnlyRobotClient) -> None:
    robot = client.robot
    original = robot.send_action

    def send_action_arms_only(action: dict[str, Any]):
        filtered = arms_only_action(dict(action))
        client.log_arm_joints(filtered)
        return original(filtered)

    robot.send_action = send_action_arms_only  # type: ignore[method-assign]


@draccus.wrap()
def main(cfg: RobotClientConfig):
    logging.basicConfig(level=logging.INFO)
    register_third_party_plugins()

    robot_cfg = cfg.robot
    if type(robot_cfg).__name__ != "UnitreeG1Config":
        raise SystemExit("Use --robot.type=unitree_g1")

    if getattr(robot_cfg, "controller", None) is None:
        logger.warning("Enabling GrootLocomotionController so the VLA cannot drive legs.")
        robot_cfg.controller = "GrootLocomotionController"

    log_every = max(1, int(os.environ.get("LOG_JOINTS_EVERY", "5")))
    arm_relative = os.environ.get("ARM_ACTIONS_RELATIVE", "0") in ("1", "true", "True")

    logger.info(
        "Remote VLA client → %s | %s | %s | task=%r | log_joints_every=%d",
        cfg.server_address,
        cfg.policy_type,
        cfg.pretrained_name_or_path,
        cfg.task,
        log_every,
    )

    client = ArmsOnlyRobotClient(cfg, log_joints_every=log_every, arm_relative=arm_relative)
    _install_send_action_filter(client)

    if client.start():
        receiver = threading.Thread(target=client.receive_actions, daemon=True)
        receiver.start()
        try:
            client.control_loop(cfg.task)
        except KeyboardInterrupt:
            logger.info("Stopping...")
            client.stop()
            receiver.join(timeout=5.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)


if __name__ == "__main__":
    main()
