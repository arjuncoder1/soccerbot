"""Unitree G1 dual-arm layout for GR00T fine-tuning.

14-D state/action (arms only; no hands, waist, or legs).
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

EMBODIMENT_TAG = "new_embodiment"
BASE_MODEL_PATH = "nvidia/GR00T-N1.7-3B"

# π0.5 pretrained aliases for --pi-base (Hub ids).
PI05_BASE_ALIASES: dict[str, str] = {
    "original": "lerobot/pi05_base",
    # Same checkpoint remote-vla-inference serves (G1 box-move finetune).
    "g1-boxmove": "sudoping01/pi05_g1_boxmove_v2",
}
DEFAULT_PI05_BASE = "g1-boxmove"
# Back-compat alias for the remote-vla default checkpoint.
PI05_PRETRAINED_PATH = PI05_BASE_ALIASES[DEFAULT_PI05_BASE]

STATE_ACTION_DIM = 14

# Flat LeRobot feature names: left arm, right arm.
STATE_ACTION_NAMES: tuple[str, ...] = tuple(f"left_arm_{s}" for s in _ARM_SUFFIXES) + tuple(
    f"right_arm_{s}" for s in _ARM_SUFFIXES
)

assert len(STATE_ACTION_NAMES) == STATE_ACTION_DIM


def validate_dataset_layout(dataset_root: Path) -> None:
    """Fail fast if a local LeRobot dataset does not match the 14-D layout."""
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
                f"{key} names do not match G1 arms-only layout "
                f"(got {len(names) if names else 0} names, expected {STATE_ACTION_DIM})"
            )

    if errors:
        raise ValueError(
            "Dataset does not match G1 arms-only embodiment:\n  - "
            + "\n  - ".join(errors)
            + "\nSee training/embodiment_g1.py for STATE_ACTION_NAMES."
        )
