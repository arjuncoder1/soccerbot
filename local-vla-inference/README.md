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

```bash
./local-vla-inference/run.sh --dry-run
./local-vla-inference/run.sh --robot-ip 192.168.123.164
```
