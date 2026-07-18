#!/usr/bin/env bash
# Build CycloneDDS (user-local), then uv sync the repo-root .venv with Python 3.12.
#
# No sudo / system Python packages required by default:
#   - uv-managed CPython 3.12 already includes Python.h
#   - CycloneDDS C lib installs under $HOME/cyclonedds/install
#
# You DO need a C compiler + cmake already on PATH (or set ALLOW_APT=1).
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
ALLOW_APT="${ALLOW_APT:-0}"

CYCLONE_SRC="${CYCLONE_SRC:-$HOME/cyclonedds}"
CYCLONE_PREFIX="${CYCLONE_PREFIX:-$CYCLONE_SRC/install}"
CYCLONE_BRANCH="${CYCLONE_BRANCH:-releases/0.10.x}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing required command: $1" >&2
    if [[ "$1" == "cmake" || "$1" == "g++" || "$1" == "gcc" ]]; then
      echo "hint: install a compiler toolchain, or re-run with ALLOW_APT=1 ./install.sh" >&2
    fi
    exit 1
  }
}

maybe_apt_build_tools() {
  [[ "$ALLOW_APT" == "1" ]] || return 0
  command -v apt-get >/dev/null 2>&1 || return 0
  echo "ALLOW_APT=1: installing cmake + build-essential via apt..."
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cmake build-essential git pkg-config
}

# Point the cyclonedds Python extension at uv-managed headers (no python3.12-dev).
export_python_build_env() {
  local py="$1"
  local include
  include="$("$py" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
  if [[ ! -f "$include/Python.h" ]]; then
    echo "error: Python.h not found under $include" >&2
    echo "hint: uv python install ${PYTHON_VERSION}" >&2
    exit 1
  fi
  export CPATH="${include}${CPATH:+:$CPATH}"
  export C_INCLUDE_PATH="${include}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
  export CPPFLAGS="-I${include} ${CPPFLAGS:-}"
  # Some setuptools builds also honor this.
  export PYTHON_INCLUDE_DIR="$include"
  echo "Using Python headers (no sudo): $include/Python.h"
}

need_cmd uv
need_cmd git
maybe_apt_build_tools
need_cmd cmake
need_cmd g++

if [[ ! -f "$CYCLONE_PREFIX/lib/libddsc.so" && ! -f "$CYCLONE_PREFIX/lib/libddsc.dylib" ]]; then
  echo "Building CycloneDDS ($CYCLONE_BRANCH) → $CYCLONE_PREFIX (user home, not system)"
  if [[ ! -d "$CYCLONE_SRC/.git" ]]; then
    git clone --depth 1 -b "$CYCLONE_BRANCH" \
      https://github.com/eclipse-cyclonedds/cyclonedds.git "$CYCLONE_SRC"
  fi
  cmake -S "$CYCLONE_SRC" -B "$CYCLONE_SRC/build" \
    -DCMAKE_INSTALL_PREFIX="$CYCLONE_PREFIX"
  cmake --build "$CYCLONE_SRC/build" --target install
else
  echo "Using existing CycloneDDS at $CYCLONE_PREFIX"
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export CMAKE_PREFIX_PATH="${CYCLONE_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
echo "Syncing workspace at $REPO_ROOT → $VENV_DIR (Python ${PYTHON_VERSION})"
cd "$REPO_ROOT"

uv python install "$PYTHON_VERSION"
PYTHON_BIN="$(uv python find "$PYTHON_VERSION")"
export_python_build_env "$PYTHON_BIN"

uv sync -p "$PYTHON_BIN" --all-packages --reinstall-package cyclonedds

echo
echo "Done. Root venv: $VENV_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
echo "  export LD_LIBRARY_PATH=$CYCLONE_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
echo "  ./local-vla-inference/run.sh --dry-run"
echo
echo "No system Python packages were required. Sync with: uv sync -p 3.12 --all-packages"
