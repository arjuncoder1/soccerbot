#!/usr/bin/env bash
# Robot install: CycloneDDS + root .venv on Python 3.12 (required for unitree_sdk2py).
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

CYCLONE_SRC="${CYCLONE_SRC:-$HOME/cyclonedds}"
CYCLONE_PREFIX="${CYCLONE_PREFIX:-$CYCLONE_SRC/install}"
CYCLONE_BRANCH="${CYCLONE_BRANCH:-releases/0.10.x}"

log() { echo "==> $*"; }
die() { echo "error: $*" >&2; exit 1; }

need_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

install_apt_deps() {
  need_cmd apt-get
  need_cmd sudo
  log "Installing apt packages (cmake, gcc, python${PYTHON_VERSION}-dev)..."
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    build-essential \
    git \
    pkg-config \
    "python${PYTHON_VERSION}" \
    "python${PYTHON_VERSION}-dev" \
    "python${PYTHON_VERSION}-venv"
}

find_python_h() {
  local py="$1"
  local include
  include="$("$py" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
  if [[ -f "$include/Python.h" ]]; then
    echo "$include"
    return
  fi
  # Common Debian/Ubuntu locations
  for cand in \
    "/usr/include/python${PYTHON_VERSION}" \
    "/usr/include/python${PYTHON_VERSION}m" \
    "/usr/local/include/python${PYTHON_VERSION}"
  do
    if [[ -f "$cand/Python.h" ]]; then
      echo "$cand"
      return
    fi
  done
  die "Python.h not found (tried $include). Is python${PYTHON_VERSION}-dev installed?"
}

build_cyclonedds_c() {
  if [[ -f "$CYCLONE_PREFIX/lib/libddsc.so" || -f "$CYCLONE_PREFIX/lib/libddsc.dylib" ]]; then
    log "CycloneDDS C lib already at $CYCLONE_PREFIX"
    return
  fi
  log "Building CycloneDDS C library → $CYCLONE_PREFIX"
  if [[ ! -d "$CYCLONE_SRC/.git" ]]; then
    git clone --depth 1 -b "$CYCLONE_BRANCH" \
      https://github.com/eclipse-cyclonedds/cyclonedds.git "$CYCLONE_SRC"
  fi
  cmake -S "$CYCLONE_SRC" -B "$CYCLONE_SRC/build" \
    -DCMAKE_INSTALL_PREFIX="$CYCLONE_PREFIX"
  cmake --build "$CYCLONE_SRC/build" --target install -j"$(nproc 2>/dev/null || echo 2)"
}

need_cmd uv
install_apt_deps
need_cmd git
need_cmd cmake
need_cmd g++
need_cmd "python${PYTHON_VERSION}"

PYTHON_BIN="$(command -v "python${PYTHON_VERSION}")"
INCLUDE_DIR="$(find_python_h "$PYTHON_BIN")"
log "Python: $PYTHON_BIN"
log "Python.h: $INCLUDE_DIR/Python.h"

build_cyclonedds_c

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export CMAKE_PREFIX_PATH="${CYCLONE_PREFIX}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="${INCLUDE_DIR}${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="${INCLUDE_DIR}${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CFLAGS="-I${INCLUDE_DIR} ${CFLAGS:-}"
export CPPFLAGS="-I${INCLUDE_DIR} ${CPPFLAGS:-}"
export CXXFLAGS="-I${INCLUDE_DIR} ${CXXFLAGS:-}"

log "CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
log "Creating root venv with system Python ${PYTHON_VERSION}"
cd "$REPO_ROOT"
rm -rf "$VENV_DIR"
uv venv "$VENV_DIR" -p "$PYTHON_BIN"

log "Building/installing cyclonedds==0.10.2 into venv first"
# Force a non-isolated build so CPATH/CFLAGS reach the extension compile.
UV_NO_BUILD_ISOLATION=1 \
  uv pip install \
    --python "$VENV_DIR/bin/python" \
    --no-binary cyclonedds \
    "cyclonedds==0.10.2"

log "Syncing full workspace into $VENV_DIR"
uv sync -p "$VENV_DIR/bin/python" --all-packages

log "Verifying imports"
"$VENV_DIR/bin/python" - <<'PY'
import cyclonedds
import unitree_sdk2py
print("ok:", cyclonedds.__file__)
PY

echo
echo "Done."
echo "  source $VENV_DIR/bin/activate"
echo "  export CYCLONEDDS_HOME=$CYCLONE_PREFIX"
echo "  export LD_LIBRARY_PATH=$CYCLONE_PREFIX/lib:\${LD_LIBRARY_PATH:-}"
echo "  ./local-vla-inference/run.sh --dry-run"
