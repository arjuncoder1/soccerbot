#!/usr/bin/env bash
# Install local-vla-inference deps, including unitree_sdk2py → cyclonedds.
# cyclonedds 0.10.2 must build against a local CycloneDDS C install.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

echo "CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
echo "Syncing workspace from $REPO_ROOT (python 3.13) ..."
cd "$REPO_ROOT"
uv sync -p 3.13 --all-packages

echo "Done. Activate with: source $REPO_ROOT/.venv/bin/activate"
echo "Keep CYCLONEDDS_HOME set when importing unitree_sdk2py:"
echo "  export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
