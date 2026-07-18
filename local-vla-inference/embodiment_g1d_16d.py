"""G1D ACT layout for ``myx160/unitree_lerobot_act_g1d_16d_001``.

Policy is 16-D. We only drive the 14 arm joints; the last 2 dims are unused
padding (no Dex1 / finger hands) — zeros in state, ignored in action.
"""

from __future__ import annotations

# Unitree G1 arm joint names (LeRobot / lowcmd key prefix before ``.q``).
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

# Policy expects 16-D; dims 14–15 are unused (no hands).
UNUSED_PAD: tuple[str, ...] = (
    "pad_0",
    "pad_1",
)

STATE_ACTION_DIM = 16
ARM_DIM = len(ARM_JOINTS)
STATE_ACTION_NAMES: tuple[str, ...] = tuple(f"{j}.q" for j in ARM_JOINTS) + UNUSED_PAD
assert len(STATE_ACTION_NAMES) == STATE_ACTION_DIM
assert ARM_DIM == 14

CAMERA_KEYS: tuple[str, ...] = (
    "cam_left_high",
    "cam_right_high",
    "cam_left_wrist",
    "cam_right_wrist",
)

IMAGE_SHAPE = (480, 640, 3)  # H, W, C — policy expects 3x480x640 after preprocess

DEFAULT_POLICY_ID = "myx160/unitree_lerobot_act_g1d_16d_001"


def dataset_features() -> dict:
    """LeRobot feature dict for ``build_inference_frame`` / ``make_robot_action``."""
    features: dict = {
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
    }
    for cam in CAMERA_KEYS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": IMAGE_SHAPE,
            "names": ["height", "width", "channels"],
        }
    return features
