#!/usr/bin/env bash
# Run ACT inference using the repo-root .venv (Python 3.12 + cyclonedds).
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CYCLONE_PREFIX="${CYCLONEDDS_HOME:-${CYCLONE_PREFIX:-$HOME/cyclonedds/install}}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: missing $VENV_DIR — run ./local-vla-inference/install.sh (or uv sync -p 3.12 --all-packages)" >&2
  exit 1
fi

PY_VER="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" == "3.13" || "$PY_VER" == "3.14" ]]; then
  echo "error: root .venv is Python $PY_VER; cyclonedds 0.10.2 needs 3.12" >&2
  echo "fix: cd $REPO_ROOT && uv sync -p 3.12 --all-packages" >&2
  exit 1
fi

if [[ ! -d "$CYCLONE_PREFIX" ]]; then
  echo "error: CycloneDDS not found at $CYCLONE_PREFIX — run ./local-vla-inference/install.sh first" >&2
  exit 1
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV_DIR/bin:$PATH"

# Optional first arg: another script in this package (e.g. diag_state.py).
SCRIPT="main.py"
if [[ $# -gt 0 && "$1" == *.py && -f "$PKG_DIR/$1" ]]; then
  SCRIPT="$1"
  shift
fi

exec python "$PKG_DIR/$SCRIPT" "$@"
