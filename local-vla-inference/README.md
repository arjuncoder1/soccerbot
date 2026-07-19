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
- **Camera**: the front cam is served by Unitree's
  [teleimager](https://github.com/unitreerobotics/teleimager) already running
  on the robot's dev PC. Default `--camera teleimager://192.168.123.164`
  queries its config service (:60000) for the head-cam ZMQ port (default
  55555) and binocular flag, subscribes to the JPEG stream, and crops the
  left eye if binocular. Overrides: `zmq://HOST:PORT` or `opencv:N`. That one
  frame is copied into all 4 policy image inputs.

## Install (client / workstation)

```bash
./local-vla-inference/install.sh   # Python 3.12 root .venv + cyclonedds
```

## Run

```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
./local-vla-inference/run.sh --iface eth0
./local-vla-inference/run.sh --dry-run
# Real camera + arm angles, print predicted joint trajectory; never commands motors
./local-vla-inference/run.sh --image-no-motors --iface eth0 --layout 14d \
  --policy /path/to/pretrained_model
```

## Diagnosis (read-only, sends nothing to the robot)

```bash
# control state: motion-switcher mode, loco FSM (damp/stand), lowstate snapshot
./local-vla-inference/run.sh diag_state.py --iface eth0

# live arm joint positions; optional CSV logging
./local-vla-inference/run.sh diag_joints.py --iface eth0
./local-vla-inference/run.sh diag_joints.py --iface eth0 --csv joints.csv --fps 30

# record 14 arm joints to JSON (Ctrl+C to stop), then replay via arm_sdk
./local-vla-inference/run.sh record_arms.py --iface eth0 -o arms.json
./local-vla-inference/run.sh replay_arms.py --iface eth0 arms.json
```
