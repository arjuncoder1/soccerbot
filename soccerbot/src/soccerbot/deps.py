"""Import helpers for workspace logic packages (not published to PyPI).

``local-vla-inference`` and ``scripted-behavior`` both expose a top-level
``main`` module. Never ``import main`` after putting both on ``sys.path``.
Load ACT via :func:`import_local_vla_main` (unique module name).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


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

LOCAL_VLA_DIR = REPO_ROOT / "local-vla-inference"
SCRIPTED_DIR = REPO_ROOT / "scripted-behavior"
REALSENSE_DIR = REPO_ROOT / "realsense-human-detection"


def _ensure_front(directory: Path) -> None:
    """Put ``directory`` at the front of ``sys.path`` (idempotent move-to-front)."""
    text = str(directory)
    if not directory.is_dir():
        return
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)


def ensure_logic_imports() -> Path:
    """Put scripted-behavior (and optional realsense) on ``sys.path``.

    Does **not** load ``local-vla-inference/main.py`` as ``main``. Use
    :func:`import_local_vla_main` for the ACT runner.
    """
    _ensure_front(REALSENSE_DIR)
    _ensure_front(SCRIPTED_DIR)
    return REPO_ROOT


def import_local_vla_main() -> ModuleType:
    """Load ``local-vla-inference/main.py`` as ``local_vla_inference_main``."""
    module_name = "local_vla_inference_main"
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Sibling imports (g1_arms, front_camera, dds_init, …) need this dir on path.
    _ensure_front(LOCAL_VLA_DIR)

    path = LOCAL_VLA_DIR / "main.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local-vla-inference main from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def import_scripted(name: str) -> ModuleType:
    """Import a scripted-behavior module by name (``avoid``, ``throw``, …)."""
    ensure_logic_imports()
    return importlib.import_module(name)
