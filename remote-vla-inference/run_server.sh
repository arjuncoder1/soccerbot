#!/usr/bin/env bash
# Start LeRobot PolicyServer on Modal (GPU + public TCP tunnel).
# Prints --server_address=HOST:PORT for the G1 client.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-A100}"
HOURS="${HOURS:-2}"
FPS="${FPS:-30}"
PORT="${PORT:-8080}"

echo "Starting remote VLA PolicyServer on Modal (gpu=${GPU}, hours=${HOURS})..."
echo "Default policy the client will request: sudoping01/pi05_g1_boxmove_v2"
echo

# Use the workspace venv's modal directly to avoid optional-extra resolver fights.
MODAL_BIN="${ROOT}/.venv/bin/modal"
if [[ ! -x "$MODAL_BIN" ]]; then
  MODAL_BIN="uv run --directory remote-vla-inference --with modal modal"
fi

exec $MODAL_BIN run remote-vla-inference/policy_server.py \
  --gpu "$GPU" \
  --hours "$HOURS" \
  --fps "$FPS" \
  --port "$PORT" \
  "$@"
