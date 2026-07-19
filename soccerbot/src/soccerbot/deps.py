"""Import helpers for workspace logic packages (not published to PyPI).

``local-vla-inference`` and ``scripted-behavior`` are flat virtual workspace
members. Soccerbot is the core orchestrator and loads them by putting their
directories on ``sys.path`` so we can ``import main`` / ``import arm_replay``
in-process — no subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

def _discover_repo_root() -> Path:
    """Find the workspace root that contains the logic packages."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3],  # soccerbot/src/soccerbot/deps.py → repo
        here.parents[2],
        Path.cwd(),
        *here.parents,
    ]
    for candidate in candidates:
        if (candidate / "local-vla-inference").is_dir() and (
            candidate / "scripted-behavior"
        ).is_dir():
            return candidate
    return here.parents[3]


REPO_ROOT = _discover_repo_root()

_LOGIC_DIRS = (
    REPO_ROOT / "local-vla-inference",
    REPO_ROOT / "scripted-behavior",
    REPO_ROOT / "realsense-human-detection",
)


def ensure_logic_imports() -> Path:
    """Prepend logic-package dirs to ``sys.path``. Idempotent."""
    for path in _LOGIC_DIRS:
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)
    return REPO_ROOT
