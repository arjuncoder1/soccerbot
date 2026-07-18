"""Shared config for the scripted-behavior orchestrator.

Kept in its own tiny module so stage modules can import it without
pulling in ``main`` (which owns the CLI and logging setup).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class PickupBackend(str, enum.Enum):
    LOCAL = "local"
    REMOTE = "remote"
    REPLAY = "replay"


@dataclass
class OrchestratorConfig:
    backend: PickupBackend = PickupBackend.LOCAL
    iface: str | None = None  # network iface passed through to the VLA client
    pickup_duration_s: float = 30.0
    pickup_extra_args: list[str] = field(default_factory=list)
    remote_server: str | None = None  # "HOST:PORT" for remote backend
