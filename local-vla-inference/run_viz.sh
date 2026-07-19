#!/usr/bin/env bash
# Read-only teleimager → Rerun gRPC server on 0.0.0.0:9876 (no policy / arm_sdk).
# Usage: ./local-vla-inference/run_viz.sh [TELEIMAGER_HOST] [TELEIMAGER_PORT] [GRPC_PORT]
# Connect from another machine:  rerun --connect rerun+http://WALDO_IP:9876/proxy
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
HOST="${1:-192.168.123.164}"
PORT="${2:-55555}"
GRPC_PORT="${3:-9876}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: missing $VENV_DIR" >&2
  exit 1
fi

if ! "$VENV_DIR/bin/python" -c "import rerun" 2>/dev/null; then
  echo "error: rerun not installed. Run: cd $REPO_ROOT && uv sync -p 3.12 --all-packages" >&2
  exit 1
fi

cd "$PKG_DIR"
exec "$VENV_DIR/bin/python" -c "
import socket
import time
import numpy as np
import rerun as rr
from front_camera import ZMQFrontCamera

rr.init('teleimager_viz')
uri = rr.serve_grpc(grpc_port=${GRPC_PORT})
# Advertise LAN bind so remote viewers can connect (serve listens on all ifaces).
ip = socket.gethostbyname(socket.gethostname())
print('Rerun gRPC listening on 0.0.0.0:${GRPC_PORT}', flush=True)
print('connect: rerun --connect rerun+http://%s:${GRPC_PORT}/proxy' % ip, flush=True)
print('or:      rerun --connect', uri, flush=True)

cam = ZMQFrontCamera('${HOST}', ${PORT})
cam.connect()
print('streaming zmq://${HOST}:${PORT} -> Rerun (Ctrl+C quit)', flush=True)
t0 = time.time()
n = 0
try:
    while True:
        rgb = cam.read()
        rr.set_time('step', sequence=n)
        rr.set_time('time', timestamp=time.time() - t0)
        rr.log('teleimager/rgb', rr.Image(np.asarray(rgb)).compress(jpeg_quality=75))
        n += 1
except KeyboardInterrupt:
    pass
finally:
    cam.disconnect()
"
