#!/usr/bin/env bash
# Build CycloneDDS, then uv sync the repo-root .venv with Python 3.12.
#
# unitree_sdk2py → cyclonedds==0.10.2 does not work on Python 3.13
# (undefined symbol: _Py_IsFinalizing). Always sync root with -p 3.12.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
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
echo "Syncing workspace at $REPO_ROOT → $VENV_DIR (Python ${PYTHON_VERSION})"
cd "$REPO_ROOT"
uv python install "$PYTHON_VERSION"
# Recreate root .venv on 3.12 so cyclonedds links against the right CPython.
uv sync -p "$PYTHON_VERSION" --all-packages --reinstall-package cyclonedds

echo
echo "Done. Root venv: $VENV_DIR"
echo "  source $VENV_DIR/bin/activate"
echo "  export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
echo "  export LD_LIBRARY_PATH=$CYCLONE_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
echo "  ./local-vla-inference/run.sh --dry-run"
echo
echo "Always sync with: uv sync -p 3.12 --all-packages"
echo "(Python 3.13 breaks unitree_sdk2py / cyclonedds.)"
