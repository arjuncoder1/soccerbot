# local-vla-inference

ACT rollout on Unitree G1D **arms only** using
[`myx160/unitree_lerobot_act_g1d_16d_001`](https://huggingface.co/myx160/unitree_lerobot_act_g1d_16d_001).

Everything talks **directly to the robot's stock services over DDS** —
**nothing extra runs on the robot** (no `run_g1_server.py`, no ZMQ bridge,
no ImageServer):

- **State**: subscribe `rt/lowstate`
- **Arms**: publish `rt/arm_sdk` (official arm SDK topic; joint 29 is the
  enable weight, ramped 0→1 on start and back to 0 on exit so the stock
  controller hands over / takes back the arms smoothly)
- **Camera**: the G1's RealSense front cam is USB-wired to an onboard PC and
  is NOT exposed over DDS (`VideoClient`/`videohub` is Go2-only → error 3102).
  We read it from the Unitree teleop `image_server` that is already running on
  the robot for data collection: `--camera zmq://192.168.123.164:5555`. If a
  camera is plugged into this machine instead, use `--camera opencv:0`. That
  one frame is copied into all 4 policy image inputs.

## Install (client / workstation)

```bash
./local-vla-inference/install.sh   # Python 3.12 root .venv + cyclonedds
```

## Run

```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
./local-vla-inference/run.sh --iface eth0 --camera zmq://192.168.123.164:5555
./local-vla-inference/run.sh --dry-run
```
