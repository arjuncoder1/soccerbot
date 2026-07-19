"""Orchestrator config for the soccerbot demo."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from soccerbot.deps import REPO_ROOT

# Working teleimager head stream (single ZMQ JPEG port — see local-vla-inference).
DEFAULT_CAMERA = "zmq://192.168.123.164:55555"
DEFAULT_POLICY = "ajkoder/g1-pickup-ball-act"
# Was 0.002 (~3.4°/s) which chronically undershot table reaches; 0.01 matches
# the standalone local-vla-inference default (~17°/s at 30 Hz) — still slew-limited.
DEFAULT_CLAMP_RAD = 0.01
DEFAULT_REPLAY_TRAJECTORY = (
    REPO_ROOT / "scripted-behavior" / "trajectories" / "pickup_ep148_prod2.json"
)


class PickupBackend(str, enum.Enum):
    LOCAL = "local"  # in-process ACT via local-vla-inference
    REPLAY = "replay"  # JSON arm trajectory in scripted-behavior/trajectories/
    REMOTE = "remote"  # optional remote pi0.5 server


@dataclass
class OrchestratorConfig:
    backend: PickupBackend = PickupBackend.LOCAL
    iface: str | None = None
    camera: str = DEFAULT_CAMERA
    policy: str = DEFAULT_POLICY
    layout: str = "14d"
    clamp: float = DEFAULT_CLAMP_RAD
    pickup_duration_s: float = 30.0
    fps: float = 30.0
    device: str | None = None
    rerun: bool = True
    record_path: str | None = None  # --record-path: also/instead write stages 2-4 to an .rrd
    display: bool = True  # --no-display: with record_path, skip spawning a live viewer window
    teleimager_host: str = "192.168.123.164"
    remote_server: str | None = None
    replay_trajectory: Path = DEFAULT_REPLAY_TRAJECTORY
    # Scripted-stage slew clamps (rad/frame).
    replay_slew_clamp: float = 0.05
    throw_slew_clamp: float = 0.02
    pickup_extra_args: list[str] = field(default_factory=list)
