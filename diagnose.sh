#!/usr/bin/env bash
# Check that the robot workstation is ready to run soccerbot.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CYCLONE_PREFIX="${CYCLONEDDS_HOME:-${CYCLONE_PREFIX:-$HOME/cyclonedds/install}}"
IFACE=""
STRICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface) IFACE="${2:-}"; shift 2 ;;
    --iface=*) IFACE="${1#*=}"; shift ;;
    --strict) STRICT=1; shift ;;
    -h|--help)
      echo "Usage: ./diagnose.sh [--iface enp5s0] [--strict]"
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

pass=0
fail=0
warn=0

ok()   { echo "  [OK]   $*"; pass=$((pass + 1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail + 1)); }
soft() { echo "  [WARN] $*"; warn=$((warn + 1)); }

echo "==> Soccerbot diagnose"
echo "    repo: $REPO_ROOT (cwd=$(pwd))"

# --- toolchain ---
if command -v uv >/dev/null 2>&1; then ok "uv: $(command -v uv)"; else bad "uv not on PATH"; fi
if [[ -x "$VENV_DIR/bin/python" ]]; then
  PY_VER="$("$VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  ok "venv python $PY_VER at $VENV_DIR"
  if [[ "$PY_VER" == "3.13" || "$PY_VER" == "3.14" ]]; then
    bad "venv is Python $PY_VER; robot needs 3.12 for cyclonedds 0.10.2 (run ./install.sh)"
  fi
else
  bad "missing $VENV_DIR — run ./install.sh"
fi

if [[ -f "$CYCLONE_PREFIX/lib/libddsc.so" || -f "$CYCLONE_PREFIX/lib/libddsc.dylib" ]]; then
  ok "CycloneDDS at $CYCLONE_PREFIX"
else
  bad "CycloneDDS missing at $CYCLONE_PREFIX — run ./install.sh"
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$VENV_DIR/bin:$PATH"

# --- python imports ---
if [[ -x "$VENV_DIR/bin/python" ]]; then
  if "$VENV_DIR/bin/python" - <<'PY'
import cyclonedds
import unitree_sdk2py
from unitree_sdk2py.utils.crc import CRC
CRC()
print("cyclonedds+unitree ok")
PY
  then ok "cyclonedds + unitree_sdk2py import"
  else bad "cyclonedds / unitree_sdk2py import failed"
  fi

  if "$VENV_DIR/bin/python" - <<'PY'
import torch, zmq, cv2
print("torch", torch.__version__)
PY
  then ok "torch + zmq + cv2"
  else bad "torch/zmq/cv2 import failed"
  fi

  if "$VENV_DIR/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("soccerbot/src").resolve()))
import soccerbot
from soccerbot.config import DEFAULT_POLICY, DEFAULT_CLAMP_RAD, DEFAULT_CAMERA
assert DEFAULT_POLICY == "ajkoder/g1-pickup-ball-act"
assert DEFAULT_CLAMP_RAD == 0.01
assert "55555" in DEFAULT_CAMERA
print("soccerbot ok", soccerbot.__version__)
PY
  then ok "soccerbot package (policy/clamp/camera defaults)"
  else soft "soccerbot import via src path failed (uv sync may still be needed)"
  fi

  if "$VENV_DIR/bin/python" - <<'PY'
import rerun  # noqa: F401
print("rerun ok")
PY
  then ok "rerun-sdk (viz)"
  else soft "rerun-sdk missing — ACT will run without live viz (lerobot[viz])"
  fi

  if "$VENV_DIR/bin/python" - <<'PY'
import tkinter  # noqa: F401
print("tkinter ok")
PY
  then ok "tkinter (headed killswitch)"
  else soft "tkinter missing — install python3-tk for ./killswitch.sh"
  fi

  # local-vla defaults (avoid importing cv2/torch-heavy main unless deps present)
  if "$VENV_DIR/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("local-vla-inference").resolve()))
import embodiment_g1_14d as layout
assert layout.DEFAULT_POLICY_ID == "ajkoder/g1-pickup-ball-act"
# Prefer full API check when vision/torch stack is installed.
try:
    import main as local_vla
    args = local_vla.build_args()
    assert args.policy == "ajkoder/g1-pickup-ball-act"
    assert args.clamp == 0.01
    assert args.camera.endswith(":55555")
    print("local-vla build_args ok")
except ModuleNotFoundError as exc:
    print("local-vla layout defaults ok; full main import skipped:", exc)
PY
  then ok "local-vla-inference defaults / API"
  else bad "local-vla-inference defaults failed"
  fi

  if "$VENV_DIR/bin/python" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripted-behavior").resolve()))
import arm_replay, throw, turn_180, avoid
from throw import throw_ball
print("scripted-behavior stages ok")
PY
  then ok "scripted-behavior stages importable"
  else bad "scripted-behavior import failed"
  fi
fi

# --- network (optional) ---
if [[ -n "$IFACE" ]]; then
  if ip link show "$IFACE" >/dev/null 2>&1; then
    ok "iface $IFACE exists"
  else
    bad "iface $IFACE not found"
  fi
  if ping -c 1 -W 1 192.168.123.161 >/dev/null 2>&1; then
    ok "ping G1 192.168.123.161"
  else
    soft "cannot ping 192.168.123.161 (robot off / wrong network?)"
  fi
  if ping -c 1 -W 1 192.168.123.164 >/dev/null 2>&1; then
    ok "ping teleimager host 192.168.123.164"
  else
    soft "cannot ping 192.168.123.164 (teleimager host)"
  fi
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    if "$REPO_ROOT/local-vla-inference/run.sh" diag_state.py --iface "$IFACE" --once >/tmp/soccerbot_diag_state.txt 2>&1; then
      ok "diag_state.py --once (see /tmp/soccerbot_diag_state.txt)"
    else
      soft "diag_state.py failed (robot not ready?) — /tmp/soccerbot_diag_state.txt"
    fi
  fi
else
  soft "pass --iface enp5s0 to also check NIC / robot / teleimager reachability"
fi

echo
echo "Summary: $pass ok, $warn warn, $fail fail"
if [[ "$fail" -gt 0 ]]; then
  exit 1
fi
if [[ "$STRICT" -eq 1 && "$warn" -gt 0 ]]; then
  exit 1
fi
exit 0
