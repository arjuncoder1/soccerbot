# local-vla-inference

ACT rollout on Unitree G1D **arms only** using
[`myx160/unitree_lerobot_act_g1d_16d_001`](https://huggingface.co/myx160/unitree_lerobot_act_g1d_16d_001).

## Why a separate Python 3.12 venv?

`unitree_sdk2py` pins `cyclonedds==0.10.2`. That extension **does not work on
Python 3.13** (`undefined symbol: _Py_IsFinalizing`). The rest of the soccerbot
workspace stays on 3.13; this package installs into `local-vla-inference/.venv`
with 3.12 via `install.sh`.

## Install (robot machine)

```bash
./local-vla-inference/install.sh
source local-vla-inference/.venv/bin/activate
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}
```

## Run

```bash
# smoke test (no robot)
python local-vla-inference/main.py --dry-run

# real robot
python local-vla-inference/main.py --robot-ip 192.168.123.164
```
