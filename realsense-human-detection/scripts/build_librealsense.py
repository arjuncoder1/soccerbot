#!/usr/bin/env python3
"""Clone/build librealsense with pyrealsense2 bindings for the soccerbot workspace.

Use this when `uv sync` can't install pyrealsense2 (no prebuilt wheel for the
platform, e.g. Jetson / aarch64 or macOS). Builds against the repo-root
``.venv`` python (the workspace convention — see AGENTS.md) and installs the
bindings directly into that venv's site-packages, so no PYTHONPATH hacks are
needed. Also writes a minimal dist-info so pip / importlib.metadata see
pyrealsense2 at the built version.

  ./realsense-human-detection/install.sh                     # tries wheel first
  .venv/bin/python realsense-human-detection/scripts/build_librealsense.py
  rm -rf realsense-human-detection/tmp/build                 # safe after install

Env overrides: VENV_DIR (default <repo>/.venv), LIBREALSENSE_GIT.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]  # realsense-human-detection/
REPO_ROOT = PKG_ROOT.parent
VENV_DIR = Path(os.environ.get("VENV_DIR", REPO_ROOT / ".venv"))
WS = PKG_ROOT / "tmp" / "build"
SRC = WS / "src" / "librealsense"
BUILD = WS / "build" / "librealsense"

LIBREALSENSE_GIT = os.environ.get(
    "LIBREALSENSE_GIT", "https://github.com/realsenseai/librealsense.git"
)


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _git() -> str:
    for candidate in ("/usr/bin/git", shutil.which("git")):
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError("git not found")


def _python() -> Path:
    """Repo-root .venv python (workspace convention), else the running one."""
    venv_py = VENV_DIR / "bin" / "python"
    if venv_py.is_file():
        return venv_py.resolve()
    print(f"warning: {venv_py} not found; falling back to {sys.executable}", flush=True)
    print("         run `uv sync --all-packages` from the repo root first", flush=True)
    return Path(sys.executable).resolve()


def _already_working(python: Path) -> bool:
    probe = "import pyrealsense2 as rs; rs.context(); print(getattr(rs, '__version__', '?'))"
    result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"pyrealsense2 {result.stdout.strip()} already importable; nothing to build", flush=True)
        return True
    return False


def _clone() -> None:
    if SRC.is_dir():
        print(f"exists: {SRC}", flush=True)
        return
    SRC.parent.mkdir(parents=True, exist_ok=True)
    _run([_git(), "clone", "--depth", "1", LIBREALSENSE_GIT, str(SRC)])


def _librealsense_version() -> str:
    cmake = (SRC / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(
        r"project\s*\(\s*librealsense2\b[^)]*VERSION\s+(\d+\.\d+\.\d+)",
        cmake,
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        return match.group(1)

    rs_h = SRC / "include" / "librealsense2" / "rs.h"
    if not rs_h.is_file():
        rs_h = SRC / "common" / "include" / "librealsense2" / "rs.h"
    text = rs_h.read_text(encoding="utf-8")
    major = re.search(r"#define\s+RS2_API_MAJOR_VERSION\s+(\d+)", text)
    minor = re.search(r"#define\s+RS2_API_MINOR_VERSION\s+(\d+)", text)
    patch = re.search(r"#define\s+RS2_API_PATCH_VERSION\s+(\d+)", text)
    if not (major and minor and patch):
        raise RuntimeError(f"could not parse librealsense version from {SRC}")
    return f"{major.group(1)}.{minor.group(1)}.{patch.group(1)}"


def _site_packages(python: Path) -> Path:
    out = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        text=True,
    ).strip()
    path = Path(out)
    if not path.is_dir():
        raise RuntimeError(f"site-packages not found: {path}")
    return path


def _inject_metadata(site: Path, version: str) -> None:
    dist_info = site / f"pyrealsense2-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(
        "\n".join(
            [
                "Metadata-Version: 2.1",
                "Name: pyrealsense2",
                f"Version: {version}",
                "Summary: Intel RealSense SDK (built from source by build_librealsense.py)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {dist_info / 'METADATA'}", flush=True)


def _verify(python: Path) -> None:
    _run(
        [
            str(python),
            "-c",
            "import pyrealsense2 as rs; ctx = rs.context(); "
            "print('ok: pyrealsense2', getattr(rs, '__version__', '?'), "
            "'| devices:', len(ctx.query_devices()))",
        ]
    )


def main() -> int:
    python = _python()
    site = _site_packages(python)
    print(f"python: {python}", flush=True)
    print(f"site-packages: {site}", flush=True)

    if _already_working(python):
        return 0

    _clone()
    version = _librealsense_version()
    print(f"librealsense version: {version}", flush=True)

    BUILD.mkdir(parents=True, exist_ok=True)
    cmake_args = [
        "cmake",
        str(SRC),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_PYTHON_BINDINGS:bool=true",
        # Drop the bindings straight into the workspace venv.
        f"-DPYTHON_INSTALL_DIR={site}",
        f"-DPYTHON_EXECUTABLE={python}",
        f"-DPython_EXECUTABLE={python}",
        f"-DPython3_EXECUTABLE={python}",
        "-DBUILD_EXAMPLES:bool=false",
        "-DBUILD_GRAPHICAL_EXAMPLES:bool=false",
    ]
    if sys.platform == "darwin":
        # macOS has no native UVC backend in the same way; RSUSB is the usual path.
        cmake_args.append("-DFORCE_RSUSB_BACKEND=ON")

    jobs = os.cpu_count() or 4
    _run(cmake_args, cwd=BUILD)
    _run(["make", f"-j{jobs}"], cwd=BUILD)
    # Writes libs to /usr/local and python bindings to the venv site-packages.
    _run(["sudo", "make", "install"], cwd=BUILD)

    _inject_metadata(site, version)
    _verify(python)
    print("done: librealsense installed, pyrealsense2 in venv site-packages", flush=True)
    print(f"next: {VENV_DIR}/bin/python {PKG_ROOT}/realsense_human_avoid.py", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
