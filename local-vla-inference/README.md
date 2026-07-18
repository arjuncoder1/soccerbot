# local-vla-inference

ACT rollout on Unitree G1D **arms only** using
[`myx160/unitree_lerobot_act_g1d_16d_001`](https://huggingface.co/myx160/unitree_lerobot_act_g1d_16d_001).

Uses the **repo-root** `.venv`. Sync with **Python 3.12** — `cyclonedds==0.10.2`
breaks on 3.13.

## Camera

One **remote** Unitree front cam over ZMQ (`head_camera`). That frame is copied
into all 4 policy image inputs.

**On the G1**, start the image server:

```bash
python -m lerobot.robots.unitree_g1.run_g1_server --camera --camera-device 4 --camera-port 5555
```

**On the client** (or same machine using the robot IP):

```bash
./local-vla-inference/run.sh --robot-ip 192.168.123.164
# optional overrides:
#   --camera-host 192.168.123.164 --camera-port 5555 --camera-name head_camera
```

## Install

```bash
./local-vla-inference/install.sh
```

## Dry-run (no hardware)

```bash
./local-vla-inference/run.sh --dry-run
```
