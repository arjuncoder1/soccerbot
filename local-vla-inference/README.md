# local-vla-inference

ACT rollout on Unitree G1D **arms only** using
[`myx160/unitree_lerobot_act_g1d_16d_001`](https://huggingface.co/myx160/unitree_lerobot_act_g1d_16d_001).

## Camera

Uses the robot’s **stock Unitree front camera** through `VideoClient` (DDS/RPC).
**Nothing extra is started on the robot** — no ImageServer, no OpenCV capture
process, no new services.

That one frame is copied into all 4 policy image inputs.

## Install (client / workstation)

```bash
./local-vla-inference/install.sh   # Python 3.12 root .venv + cyclonedds
```

## Run

```bash
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
./local-vla-inference/run.sh --robot-ip 192.168.123.164
./local-vla-inference/run.sh --dry-run
```
