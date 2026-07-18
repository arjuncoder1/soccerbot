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
TASK="${TASK:-pick up the red ball}"

# Client uses Python 3.12 (unitree/cyclonedds are happier than on 3.13).
PY="${PY:-3.12}"

# Ensure client deps (async + pi + unitree) once; skip if already synced.
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
  echo "Syncing remote-vla-inference[client] extras on Python ${PY} (set SKIP_SYNC=1 to skip)..."
  uv sync --package remote-vla-inference --extra client -p "$PY"
fi

LOG_JOINTS_EVERY="${LOG_JOINTS_EVERY:-5}"
# Safety: max radians any arm joint may move per control step toward the policy
# target. 0.01 rad/step @ 30 fps ~= 0.3 rad/s (very slow). Set 0 to disable.
ARM_SLEW_CLAMP="${ARM_SLEW_CLAMP:-0.01}"
# Per-step CSV log (target/cmd/measured per joint). Empty string disables.
LOG_CSV="${LOG_CSV:-remote_vla_log_$(date +%Y%m%d_%H%M%S).csv}"

# Real-robot transport (matches local-vla-inference): direct DDS via the real
# Unitree SDK. Publishes arm targets on rt/arm_sdk (balancer untouched); nothing
# runs on the robot. Only used with --robot.is_simulation=false.
#   IFACE  : network interface on the robot's LAN. Defaults to enp5s0 (the wired
#            NIC this machine uses for the robot, same as the local run's
#            --iface). Empty => DDS default interface.
#   CAMERA : front-cam source. Empty => zmq://<robot_ip>:55555 (teleimager head,
#            the source the local run used).
IFACE="${IFACE:-enp5s0}"
CAMERA="${CAMERA:-}"

# The robot LAN NIC must be up and on 192.168.123.x, or DDS sees no rt/lowstate
# (same requirement as the local run). Warn early instead of a 10s timeout.
if [[ -n "$IFACE" ]] && ! ip -br link show "$IFACE" 2>/dev/null | grep -q "UP"; then
  echo "WARNING: interface '$IFACE' is not UP. Connect the wired link to the robot" >&2
  echo "         (or set IFACE=<other-nic>). Current interfaces:" >&2
  ip -br addr >&2
  echo >&2
fi
# The real Unitree SDK needs the locally built CycloneDDS (same as local run.sh).
CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds/install}"

echo "Policy: ${POLICY_TYPE} @ ${POLICY}"
echo "Task:   ${TASK}"
echo "Python: ${PY}"
echo "Joint log every ${LOG_JOINTS_EVERY} steps | slew clamp=${ARM_SLEW_CLAMP} rad/step | csv=${LOG_CSV:-off}"
echo "DDS: iface=${IFACE:-<default>} camera=${CAMERA:-zmq://<robot_ip>:55555} cyclonedds=${CYCLONEDDS_HOME}"
echo "Extra args: $*"
echo

if [[ -d "$CYCLONEDDS_HOME" ]]; then
  export CYCLONEDDS_HOME
  export LD_LIBRARY_PATH="${CYCLONEDDS_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export LOG_JOINTS_EVERY ARM_SLEW_CLAMP LOG_CSV
export G1_IFACE="$IFACE" G1_CAMERA="$CAMERA"

exec uv run --package remote-vla-inference --extra client -p "$PY" \
  python remote-vla-inference/g1_client.py \
  --policy_type="$POLICY_TYPE" \
  --pretrained_name_or_path="$POLICY" \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --task="$TASK" \
  --robot.type=unitree_g1 \
  "$@"
