# local-vla-inference

ACT rollout on Unitree G1D **arms only** using
[`myx160/unitree_lerobot_act_g1d_16d_001`](https://huggingface.co/myx160/unitree_lerobot_act_g1d_16d_001).

Uses the **repo-root** `.venv`. Sync with **Python 3.12** — `cyclonedds==0.10.2`
(from `unitree_sdk2py`) breaks on 3.13 (`_Py_IsFinalizing`).

## Install (robot)

```bash
./local-vla-inference/install.sh
# or:
#   export CYCLONEDDS_HOME=$HOME/cyclonedds/install   # after building CycloneDDS
#   uv sync -p 3.12 --all-packages
```

## Run

Uses the **Unitree front camera only** (OpenCV). That frame is copied into all
4 image inputs the ACT checkpoint expects.

```bash
./local-vla-inference/run.sh --dry-run

# front cam is often /dev/video0 or /dev/video4 on G1
./local-vla-inference/run.sh --front-camera 0
./local-vla-inference/run.sh --front-camera 4
```
