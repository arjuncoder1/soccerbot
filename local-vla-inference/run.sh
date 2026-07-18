#!/usr/bin/env bash
# Run ACT inference with the Python 3.12 robot venv (not the repo-root 3.13 .venv).
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PKG_DIR/.venv}"
CYCLONE_PREFIX="${CYCLONEDDS_HOME:-${CYCLONE_PREFIX:-$HOME/cyclonedds/install}}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: missing $VENV_DIR — run ./local-vla-inference/install.sh first" >&2
  exit 1
fi

if [[ ! -d "$CYCLONE_PREFIX" ]]; then
  echo "error: CycloneDDS not found at $CYCLONE_PREFIX — run ./local-vla-inference/install.sh first" >&2
  exit 1
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV_DIR/bin:$PATH"

exec python "$PKG_DIR/main.py" "$@"
