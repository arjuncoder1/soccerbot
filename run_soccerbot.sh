#!/usr/bin/env bash
# Run the soccerbot orchestrator with the robot venv + CycloneDDS env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CYCLONE_PREFIX="${CYCLONEDDS_HOME:-${CYCLONE_PREFIX:-$HOME/cyclonedds/install}}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: missing $VENV_DIR — run ./install.sh first" >&2
  exit 1
fi

PY_VER="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VER" == "3.13" || "$PY_VER" == "3.14" ]]; then
  echo "error: root .venv is Python $PY_VER; cyclonedds 0.10.2 needs 3.12" >&2
  echo "fix: ./install.sh" >&2
  exit 1
fi

if [[ ! -d "$CYCLONE_PREFIX" ]]; then
  echo "error: CycloneDDS not found at $CYCLONE_PREFIX — run ./install.sh" >&2
  exit 1
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/soccerbot/src${PYTHONPATH:+:$PYTHONPATH}"

# Relative --record-path (e.g. logs/demo.rrd) must resolve from the repo root.
cd "$REPO_ROOT"

echo "tip: keep ./killswitch.sh running in another terminal" >&2

if ! "$VENV_DIR/bin/python" -c "import rerun" >/dev/null 2>&1; then
  echo "error: rerun-sdk missing in $VENV_DIR — .rrd recording will not work" >&2
  echo "fix:  cd $REPO_ROOT && uv sync -p 3.12 --all-packages" >&2
  echo "check: .venv/bin/python -c 'import rerun; print(rerun.__version__)'" >&2
  # Still allow non-record runs; soccerbot raises if --record-path is set.
fi

if command -v soccerbot >/dev/null 2>&1; then
  exec soccerbot "$@"
fi
exec python -m soccerbot "$@"
