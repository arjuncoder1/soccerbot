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
    pickup_duration_s: float = 20.0
    pickup_extra_args: list[str] = field(default_factory=list)
    remote_server: str | None = None  # "HOST:PORT" for remote backend
    mode: str = "replay"
    teleimager_host: str = "192.168.123.164"  # robot IP hosting teleimager ZMQ image_server
    # Shared local-vla-inference.telemetry.Telemetry instance, set once by
    # soccerbot.main.run_demo() and read by turn_180.py/avoid.py/throw.py.
    # Untyped (Any) to avoid this low-level config module depending on the
    # sibling local-vla-inference package. None (or a disabled instance) is a
    # no-op everywhere it's used -- Rerun stays fully optional.
    telemetry: object | None = None
