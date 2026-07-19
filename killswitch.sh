#!/usr/bin/env bash
# Headed G1 killswitch panel (Stop / Damp / ZeroTorque / Start).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CYCLONE_PREFIX="${CYCLONEDDS_HOME:-${CYCLONE_PREFIX:-$HOME/cyclonedds/install}}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: missing $VENV_DIR — run ./install.sh first" >&2
  exit 1
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/soccerbot/src${PYTHONPATH:+:$PYTHONPATH}"

# Prefer console script if installed; else module path.
if command -v soccerbot-killswitch >/dev/null 2>&1; then
  exec soccerbot-killswitch "$@"
fi
exec python -m soccerbot.killswitch "$@"
