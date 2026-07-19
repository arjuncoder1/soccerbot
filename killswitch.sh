#!/usr/bin/env bash
# G1 killswitch — CLI by default; pass --gui for the Tk panel.
#
#   ./killswitch.sh --iface enp5s0              # interactive CLI
#   ./killswitch.sh --iface enp5s0 stop         # one-shot StopMove
#   ./killswitch.sh --iface enp5s0 home         # arms → home pose
#   ./killswitch.sh --gui --iface enp5s0        # Tk GUI (+ GO HOME button)
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

if command -v soccerbot-killswitch >/dev/null 2>&1; then
  exec soccerbot-killswitch "$@"
fi
exec python -m soccerbot.killswitch "$@"
