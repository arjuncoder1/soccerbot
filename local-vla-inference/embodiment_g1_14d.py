"""G1 arms-only 14-D layout matching ``ajkoder/g1_prod_n1_cleaned``.

For policies (ACT or GR00T) trained on the cleaned dataset: 14-D
state/action named ``left_arm_*`` / ``right_arm_*`` (see
``training/embodiment_g1.py``) and a single front camera ``color_0``.
"""

from __future__ import annotations

import numpy as np

# DDS joint names (G1Arms keys, ``<name>.q``), in dataset order.
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

_ARM_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

# Dataset feature names, same joint order as ARM_JOINTS.
FEATURE_NAMES: tuple[str, ...] = tuple(f"left_arm_{s}" for s in _ARM_SUFFIXES) + tuple(
    f"right_arm_{s}" for s in _ARM_SUFFIXES
)

STATE_ACTION_DIM = 14
STATE_ACTION_NAMES = FEATURE_NAMES
assert len(ARM_JOINTS) == len(FEATURE_NAMES) == STATE_ACTION_DIM

CAMERA_KEY = "color_0"
IMAGE_SHAPE = (720, 1280, 3)  # H, W, C — cleaned dataset records 720p


def dataset_features() -> dict:
    """LeRobot feature dict for ``build_inference_frame`` / ``make_robot_action``."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_ACTION_DIM,),
            "names": list(STATE_ACTION_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_ACTION_DIM,),
            "names": list(STATE_ACTION_NAMES),
        },
        f"observation.images.{CAMERA_KEY}": {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        },
    }


def pack_observation(arm_obs: dict[str, float], front_rgb: np.ndarray) -> dict:
    """Map DDS joint readings + front camera into dataset-named raw observation."""
    out: dict = {}
    for dds, feat in zip(ARM_JOINTS, FEATURE_NAMES):
        key = f"{dds}.q"
        if key not in arm_obs:
            raise KeyError(f"Missing arm joint in observation: {key}")
        out[feat] = arm_obs[key]
    out[CAMERA_KEY] = front_rgb
    return out


def to_dds_action(robot_action: dict[str, float]) -> dict[str, float]:
    """Convert dataset-named action back to DDS ``<joint>.q`` keys for G1Arms."""
    return {f"{dds}.q": float(robot_action[feat]) for dds, feat in zip(ARM_JOINTS, FEATURE_NAMES)}
