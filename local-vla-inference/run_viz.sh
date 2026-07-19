#!/usr/bin/env bash
# Read-only G1 viz → Rerun gRPC on 0.0.0.0:9876
#   - 29-DoF stick figure (moves with rt/lowstate) — NEVER publishes arm_sdk
#   - teleimager RGB
#   - optional RealSense depth (--depth)
#
# Usage: ./local-vla-inference/run_viz.sh [--iface IFACE] [--depth]
# Connect: rerun --connect rerun+http://WALDO_IP:9876/proxy
set -euo pipefail

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$PKG_DIR/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
CYCLONE_PREFIX="${CYCLONEDDS_HOME:-${CYCLONE_PREFIX:-$HOME/cyclonedds/install}}"

HOST="192.168.123.164"
PORT=55555
GRPC_PORT=9876
IFACE=""
WANT_DEPTH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iface) IFACE="$2"; shift 2 ;;
    --depth) WANT_DEPTH=1; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --grpc-port) GRPC_PORT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "error: missing $VENV_DIR" >&2
  exit 1
fi
if ! "$VENV_DIR/bin/python" -c "import rerun" 2>/dev/null; then
  echo "error: rerun not installed — uv sync -p 3.12 --all-packages" >&2
  exit 1
fi
if [[ ! -d "$CYCLONE_PREFIX" ]]; then
  echo "error: CycloneDDS missing at $CYCLONE_PREFIX" >&2
  exit 1
fi

export CYCLONEDDS_HOME="$CYCLONE_PREFIX"
export LD_LIBRARY_PATH="${CYCLONE_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$PKG_DIR"
exec "$VENV_DIR/bin/python" - "$HOST" "$PORT" "$GRPC_PORT" "$IFACE" "$WANT_DEPTH" <<'PY'
import socket
import sys
import time

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dds_init import ensure_dds
from front_camera import ZMQFrontCamera
from g1_29_fk import skeleton_from_snapshot
from g1_arms import G1Arms

host, port, grpc_port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
iface = sys.argv[4] or None
want_depth = sys.argv[5] == "1"

COLORS = {
    "left_leg": (80, 180, 255),
    "right_leg": (80, 180, 255),
    "waist": (220, 220, 80),
    "spine": (220, 220, 80),
    "left_arm": (100, 136, 234),
    "right_arm": (224, 96, 58),
}

rr.init("g1_29dof_viz")
uri = rr.serve_grpc(grpc_port=grpc_port)
ip = socket.gethostbyname(socket.gethostname())
print(f"Rerun gRPC on 0.0.0.0:{grpc_port}", flush=True)
print(f"connect: rerun --connect rerun+http://{ip}:{grpc_port}/proxy", flush=True)
print(f"uri: {uri}", flush=True)

rr.send_blueprint(
    rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="G1 29-DoF", origin="g1", contents="g1/**"),
            rrb.Vertical(
                rrb.Spatial2DView(name="teleimager RGB", origin="teleimager"),
                rrb.Spatial2DView(name="depth", origin="realsense"),
                rrb.TimeSeriesView(name="arm joints", origin="joints/q", contents="$origin/**"),
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
    ),
    make_active=True,
    make_default=True,
)

ensure_dds(iface)
arms = G1Arms()
arms.connect(state_only=True)
print("joints: rt/lowstate (state_only, no motors) -> 29-DoF skeleton", flush=True)

cam = ZMQFrontCamera(host, port)
cam.connect()
print(f"rgb: zmq://{host}:{port}", flush=True)

from g1_arms import ARM_JOINT_INDEX

joint_names = list(ARM_JOINT_INDEX.keys())
for name in joint_names:
    rr.log(f"joints/q/{name}", rr.SeriesLines(names=name, widths=1.5), static=True)

depth_pipe = None
if want_depth:
    try:
        import pyrealsense2 as rs

        depth_pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        depth_pipe.start(cfg)
        print("depth: RealSense z16", flush=True)
    except Exception as exc:
        print(f"depth: unavailable ({exc})", flush=True)
        depth_pipe = None

t0 = time.time()
n = 0
try:
    while True:
        rgb = cam.read()
        rr.set_time("step", sequence=n)
        rr.set_time("time", timestamp=time.time() - t0)
        rr.log("teleimager/rgb", rr.Image(np.asarray(rgb)).compress(jpeg_quality=75))

        snap = arms.get_full_snapshot()
        skel = skeleton_from_snapshot(snap)
        all_pts = []
        for chain, pts in skel.items():
            color = COLORS.get(chain, (200, 200, 200))
            arr = np.asarray(pts, dtype=np.float32)
            rr.log(f"g1/{chain}/bones", rr.LineStrips3D([arr], colors=[color], radii=0.012))
            rr.log(f"g1/{chain}/joints", rr.Points3D(arr, colors=[color], radii=0.025))
            all_pts.append(arr)

        # Single overlay of every joint marker (29-DoF feel).
        if all_pts:
            stacked = np.vstack(all_pts)
            rr.log("g1/all_joints", rr.Points3D(stacked, colors=[(255, 255, 255)], radii=0.02))

        for name in joint_names:
            key = f"{name}.q"
            if key in snap:
                rr.log(f"joints/q/{name}", rr.Scalars(float(snap[key])))

        if depth_pipe is not None:
            try:
                frames = depth_pipe.wait_for_frames(timeout_ms=100)
                d = frames.get_depth_frame()
                if d:
                    depth = np.asanyarray(d.get_data())
                    rr.log("realsense/depth", rr.DepthImage(depth, meter=0.001))
            except Exception:
                pass

        n += 1
except KeyboardInterrupt:
    pass
finally:
    cam.disconnect()
    arms.disconnect()
    if depth_pipe is not None:
        try:
            depth_pipe.stop()
        except Exception:
            pass
PY
