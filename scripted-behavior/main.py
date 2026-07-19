"""Thin wrapper — the core orchestrator now lives in ``soccerbot``.

Prefer:

    python -m soccerbot --iface enp5s0
    ./run_soccerbot.sh --iface enp5s0

This module keeps the old ``scripted-behavior/main.py`` entry point working
by forwarding to ``soccerbot.main``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SOCCERBOT_SRC = _REPO_ROOT / "soccerbot" / "src"
if str(_SOCCERBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_SOCCERBOT_SRC))

from soccerbot.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
