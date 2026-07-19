#!/usr/bin/env bash
# Set up pyrealsense2 for realsense-human-detection in the repo-root .venv.
#
# Tries the prebuilt wheel first (declared in pyproject.toml, linux-only).
# On platforms with no wheel (e.g. Jetson / aarch64), pip install can't help —
# librealsense must be built from source with its python bindings.
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

log() { echo "==> $*"; }
die() { echo "error: $*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || die "missing command: uv"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  log "No venv at $VENV_DIR — creating one (Python ${PYTHON_VERSION})"
  (cd "$REPO_ROOT" && uv venv "$VENV_DIR" --python "python${PYTHON_VERSION}")
fi

log "Syncing workspace (installs realsense-human-detection deps incl. pyrealsense2 wheel if available)"
(cd "$REPO_ROOT" && uv sync -p "$VENV_DIR/bin/python" --all-packages)

log "Verifying pyrealsense2"
if "$VENV_DIR/bin/python" - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
print("ok: pyrealsense2", getattr(rs, "__version__", "?"), "| devices:", len(ctx.query_devices()))
PY
then
  echo
  echo "Done."
  echo "  # one-frame API:"
  echo "  $VENV_DIR/bin/python -c 'from realsense_human_detection import HumanDetector'"
  echo "  # live camera:"
  echo "  $VENV_DIR/bin/python $PKG_DIR/main.py"
else
  echo
  echo "pyrealsense2 not found — no prebuilt wheel for this platform." >&2
  echo "Run the build pyrealsense script to build librealsense + python bindings from source:" >&2
  echo "  $VENV_DIR/bin/python $PKG_DIR/scripts/build_librealsense.py" >&2
  exit 1
fi
