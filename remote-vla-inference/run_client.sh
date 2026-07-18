#!/usr/bin/env bash
# Connect Unitree G1 (arms only) to a Modal PolicyServer running π0.5.
#
# Usage:
#   ./remote-vla-inference/run_client.sh --server_address=HOST:PORT
#   ./remote-vla-inference/run_client.sh --server_address=HOST:PORT --robot.is_simulation=false
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POLICY="${POLICY:-sudoping01/pi05_g1_boxmove_v2}"
POLICY_TYPE="${POLICY_TYPE:-pi05}"
TASK="${TASK:-move the blue box}"

# Client uses Python 3.12 (unitree/cyclonedds are happier than on 3.13).
PY="${PY:-3.12}"

# Ensure client deps (async + pi + unitree) once; skip if already synced.
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  echo "Syncing remote-vla-inference[client] extras on Python ${PY} (set SKIP_SYNC=1 to skip)..."
  uv sync --package remote-vla-inference --extra client -p "$PY"
fi

LOG_JOINTS_EVERY="${LOG_JOINTS_EVERY:-5}"

echo "Policy: ${POLICY_TYPE} @ ${POLICY}"
echo "Task:   ${TASK}"
echo "Python: ${PY}"
echo "Joint log every ${LOG_JOINTS_EVERY} steps"
echo "Extra args: $*"
echo

export LOG_JOINTS_EVERY

exec uv run --package remote-vla-inference --extra client -p "$PY" \
  python remote-vla-inference/g1_client.py \
  --policy_type="$POLICY_TYPE" \
  --pretrained_name_or_path="$POLICY" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --task="$TASK" \
  --robot.type=unitree_g1 \
  --robot.controller=GrootLocomotionController \
  "$@"
