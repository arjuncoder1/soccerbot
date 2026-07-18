# How to run

Quick reference for launching the scripted-behavior orchestrator and each stage
individually. All commands assume the repo has been synced with `uv sync` (see
[`AGENTS.md`](AGENTS.md)) and are run from the repo root unless noted.

The live commands need:

- A Unitree G1 reachable over DDS on interface `--iface eth0` (or whatever your
  robot NIC is named).
- For the `avoid` stage: an Intel RealSense camera plugged in and the
  `realsense-human-detection/` package importable.
- For `backend=local`: the `local-vla-inference/` package importable + an ACT
  checkpoint on disk.
- For `backend=remote`: a reachable pi0.5 policy server (`HOST:PORT`).
- For `backend=replay`: `scripted-behavior/trajectories/pickup_ep10.json`
  (already committed).

> All stage scripts must be launched from inside `scripted-behavior/` because
> they import sibling modules by name (`from config import ...`).

---

## Full orchestrator (all four stages, back to back)

```bash
cd scripted-behavior

# Local ACT policy for pickup (default backend).
python3 main.py --iface eth0 --pickup-duration 30

# Remote pi0.5 policy server for pickup.
python3 main.py --backend remote --iface eth0 --remote-server 192.168.1.42:8000

# Replay the recorded episode-10 pickup trajectory (no learned policy).
python3 main.py --backend replay --iface eth0

# Forward extra flags to the pickup launcher after `--`.
python3 main.py --iface eth0 -- --extra-flag-for-vla foo
```

Pipeline order (each raise aborts the demo):

1. `pickup`   — ACT / pi0.5 / trajectory replay
2. `turn_180` — LocoClient yaw rotate with IMU-integrated stop
3. `avoid`    — 2-left / 2-right shuffle until RealSense reports clear frames
4. `throw`    — hardcoded arm replay (currently `NotImplementedError`)

Exit codes: `0` success, `1` uncaught error, `2` unimplemented stage,
`130` Ctrl-C.

---

## Individual stages

Each of these can be run in isolation to smoke-test one primitive.

### Pickup only

```bash
cd scripted-behavior

python3 pickup.py --backend local  --iface eth0 --pickup-duration 15
python3 pickup.py --backend remote --iface eth0 --remote-server HOST:PORT
python3 pickup.py --backend replay --iface eth0
```

### Turn 180 degrees only

```bash
cd scripted-behavior
python3 turn_180.py --iface eth0
```

Rotates in place at `TURN_YAW_RATE_RPS` until `|Δyaw| ≥ π ± TURN_TOLERANCE_RAD`
or `TURN_180_MAX_S` seconds elapse; holds arms via a background thread.

### Sidestep primitive only

```bash
cd scripted-behavior
python3 sidestep.py left  --steps 2 --iface eth0
python3 sidestep.py right --steps 3 --iface eth0
```

Open-loop lateral shuffle: `SIDESTEP_STEP_S` seconds of `±SIDESTEP_VY_MPS`
followed by `SIDESTEP_PAUSE_S` of stand, repeated `--steps` times.

### Avoid stage only (shuffle + RealSense loop)

```bash
cd scripted-behavior

# Full stage: shuffle 2L / 2R until N consecutive clear frames.
python3 avoid.py --iface eth0

# Detector smoke test only (no robot motion) — polls RealSense for N seconds.
python3 avoid.py --detect-only 10
```

### Throw stage only

```bash
cd scripted-behavior
python3 throw.py --iface eth0   # currently raises NotImplementedError
```

Trajectory hasn't been recorded yet; wire it through `arm_replay.py` once you
have a JSON in `scripted-behavior/trajectories/`.

---

## Ancillary helpers

### RealSense human detector (standalone demo, no robot)

```bash
cd realsense-human-detection
python3 realsense_human_avoid.py
```

Streams from the camera and prints per-frame `PersonDetection` snapshots. Same
`HumanDetector` class the `avoid` stage uses.

### Arm-trajectory replay (library)

`scripted-behavior/arm_replay.py` is a library used by `pickup.py --backend
replay` and (eventually) `throw.py`. It has no CLI of its own; call
`replay_arm_trajectory(path, iface=...)`.

---

## Tuning knobs (module-level constants)

Edit these in-file if the behavior needs adjustment on-robot; no restart of
anything else is needed.

| File | Constant | Default | Purpose |
| --- | --- | --- | --- |
| `turn_180.py` | `TURN_YAW_RATE_RPS`   | `0.6`  | Yaw command magnitude |
| `turn_180.py` | `TURN_TOLERANCE_RAD`  | `5°`   | Stop-band around ±π |
| `turn_180.py` | `TURN_180_MAX_S`      | `15.0` | Hard timeout |
| `sidestep.py` | `SIDESTEP_VY_MPS`     | `0.25` | Lateral speed |
| `sidestep.py` | `SIDESTEP_STEP_S`     | `0.6`  | Move duration per step |
| `sidestep.py` | `SIDESTEP_PAUSE_S`    | `0.4`  | Stand duration between steps |
| `avoid.py`    | `AVOID_CLEAR_CONFIRM_POLLS` | `2` | Consecutive clear frames required |
| `arm_replay.py` | `REPLAY_RAMP_S`     | `2.0`  | Blend-in duration to first frame |
| `arm_replay.py` | `REPLAY_SLEW_CLAMP` | `0.05` | Max Δq per tick (rad) |
