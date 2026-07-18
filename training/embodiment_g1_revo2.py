"""G1 + BrainCo Revo 2 bimanual layout for GR00T fine-tuning.

26-D state/action (no waist/legs). Hand order matches BrainCo Revo 2 SDK
``set_finger_positions``: Thumb, ThumbAux, Index, Middle, Ring, Pinky.
Left and right hands use the same convention.
"""

from __future__ import annotations

import json
from pathlib import Path

# Unitree G1 arm DoF (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw).
_ARM_SUFFIXES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)

# BrainCo Revo 2 active motors (SDK order). Passiveives are not action targets.
_HAND_SUFFIXES = (
    "thumb_flex",  # FINGERID_THUMB
    "thumb_aux",  # FINGERID_THUMB_AUX (abduction/adduction)
    "index",
    "middle",
    "ring",
    "pinky",
)

EMBODIMENT_TAG = "new_embodiment"
BASE_MODEL_PATH = "nvidia/GR00T-N1.7-3B"
STATE_ACTION_DIM = 26

# Flat LeRobot feature names: left arm, right arm, left hand, right hand.
STATE_ACTION_NAMES: tuple[str, ...] = (
    tuple(f"left_arm_{s}" for s in _ARM_SUFFIXES)
    + tuple(f"right_arm_{s}" for s in _ARM_SUFFIXES)
    + tuple(f"left_hand_{s}" for s in _HAND_SUFFIXES)
    + tuple(f"right_hand_{s}" for s in _HAND_SUFFIXES)
)

assert len(STATE_ACTION_NAMES) == STATE_ACTION_DIM


def validate_dataset_layout(dataset_root: Path) -> None:
    """Fail fast if a local LeRobot dataset does not match the 26-D layout."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}; expected a LeRobot dataset.")

    info = json.loads(info_path.read_text())
    features = info.get("features") or {}
    errors: list[str] = []

    for key in ("observation.state", "action"):
        feat = features.get(key)
        if not isinstance(feat, dict):
            errors.append(f"missing features.{key}")
            continue
        shape = feat.get("shape")
        if list(shape) != [STATE_ACTION_DIM]:
            errors.append(f"{key} shape={shape!r}, expected [{STATE_ACTION_DIM}]")
        names = feat.get("names")
        if names is not None and list(names) != list(STATE_ACTION_NAMES):
            errors.append(
                f"{key} names do not match G1+Revo2 layout "
                f"(got {len(names) if names else 0} names, expected {STATE_ACTION_DIM})"
            )

    if errors:
        raise ValueError(
            "Dataset does not match G1+Revo2 embodiment:\n  - "
            + "\n  - ".join(errors)
            + "\nSee training/embodiment_g1_revo2.py for STATE_ACTION_NAMES."
        )
