#!/usr/bin/env bash
# Root install for the soccerbot workspace on the robot machine.
# Wraps local-vla-inference/install.sh (Python 3.12 + CycloneDDS + uv sync).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

log() { echo "==> $*"; }

log "Soccerbot workspace install (delegating CycloneDDS + 3.12 venv to local-vla-inference)"
"$REPO_ROOT/local-vla-inference/install.sh"

# Headed killswitch needs Tk
if command -v apt-get >/dev/null 2>&1; then
  log "Ensuring python3-tk for headed killswitch"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk >/dev/null || \
    echo "warning: could not install python3-tk (killswitch GUI needs it)" >&2
fi

# Optional: RealSense python bindings (avoid stage / local depth detector).
if [[ -x "$REPO_ROOT/realsense-human-detection/install.sh" ]]; then
  log "Installing realsense-human-detection extras"
  "$REPO_ROOT/realsense-human-detection/install.sh" || \
    echo "warning: realsense install failed (teleimager-based avoid still works)" >&2
fi

# Ensure soccerbot is an editable install (hatchling) after sync.
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  log "Re-syncing workspace so soccerbot entry points are available"
  export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}"
  export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  uv sync -p "$VENV_DIR/bin/python" --all-packages
fi

echo
echo "Done. Next:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo "  export CYCLONEDDS_HOME=\${CYCLONEDDS_HOME:-\$HOME/cyclonedds/install}"
echo "  export LD_LIBRARY_PATH=\$CYCLONEDDS_HOME/lib:\${LD_LIBRARY_PATH:-}"
echo "  ./diagnose.sh --iface enp5s0"
echo "  ./killswitch.sh --iface enp5s0        # CLI killswitch (stop/damp/zero/home)"
echo "  ./killswitch.sh --gui --iface enp5s0  # Tk GUI killswitch"
echo "  ./run_soccerbot.sh --iface enp5s0"
