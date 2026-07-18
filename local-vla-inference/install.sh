#!/usr/bin/env bash
# Robot-machine install for local-vla-inference.
#
# unitree_sdk2py pins cyclonedds==0.10.2, which does NOT work on Python 3.13
# (ImportError: undefined symbol: _Py_IsFinalizing). This script always uses
# Python 3.12 in a dedicated venv under local-vla-inference/.venv.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PKG_DIR/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

CYCLONE_SRC="${CYCLONE_SRC:-$HOME/cyclonedds}"
CYCLONE_PREFIX="${CYCLONE_PREFIX:-$CYCLONE_SRC/install}"
CYCLONE_BRANCH="${CYCLONE_BRANCH:-releases/0.10.x}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing required command: $1" >&2
    exit 1
  }
}

ensure_build_deps() {
  if command -v cmake >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    return
  fi
  if command -v apt-get >/dev/null 2>&1; then
    echo "Installing cmake + build-essential via apt..."
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y cmake build-essential git
  elif command -v brew >/dev/null 2>&1; then
    echo "Installing cmake via Homebrew..."
    brew install cmake
  else
    echo "error: need cmake and a C++ compiler (install cmake / build-essential)" >&2
    exit 1
  fi
}

ensure_build_deps
need_cmd git
need_cmd cmake
need_cmd uv

if [[ ! -f "$CYCLONE_PREFIX/lib/libddsc.so" && ! -f "$CYCLONE_PREFIX/lib/libddsc.dylib" ]]; then
  echo "Building CycloneDDS ($CYCLONE_BRANCH) → $CYCLONE_PREFIX"
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
echo "Syncing local-vla-inference with Python ${PYTHON_VERSION} → $VENV_DIR"
cd "$PKG_DIR"
uv python install "$PYTHON_VERSION"
# Standalone project (not in the 3.13 workspace): creates/uses PKG_DIR/.venv
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync -p "$PYTHON_VERSION" --project "$PKG_DIR"

echo
echo "Done. On the robot:"
echo "  source $VENV_DIR/bin/activate"
echo "  export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
echo "  export LD_LIBRARY_PATH=$CYCLONE_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
echo "  python $PKG_DIR/main.py --dry-run"
echo
echo "Do NOT use the repo-root .venv (Python 3.13) — cyclonedds 0.10.2 breaks there."
