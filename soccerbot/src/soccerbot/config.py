"""Orchestrator config for the soccerbot demo."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

from soccerbot.deps import REPO_ROOT

# Working teleimager head stream (single ZMQ JPEG port — see local-vla-inference).
DEFAULT_CAMERA = "zmq://192.168.123.164:55555"
DEFAULT_POLICY = "ajkoder/g1-pickup-ball-act"
DEFAULT_CLAMP_RAD = 0.002
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
    pickup_duration_s: float = 45.0
    fps: float = 30.0
    device: str | None = None
    teleimager_host: str = "192.168.123.164"
    remote_server: str | None = None
    replay_trajectory: Path = DEFAULT_REPLAY_TRAJECTORY
    # Scripted-stage slew clamps (rad/frame).
    # Replay: tighter than a raw demo spike, still enough for normal recordings.
    replay_slew_clamp: float = 0.01
    # Throw stays loose on purpose: needs ~1.7 rad/s; 0.06 @50Hz = 3 rad/s
    # only catches garbage targets (see throw.THROW_SLEW_CLAMP).
    throw_slew_clamp: float = 0.06
    pickup_extra_args: list[str] = field(default_factory=list)
