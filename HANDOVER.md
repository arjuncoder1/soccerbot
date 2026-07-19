# Handover — G1 Soccer Pickup Demo

Last updated: 2026-07-18, commit `098585a`.

Covers the 4-stage pickup demo (pickup → turn 180 → shuffle-avoid → throw),
what individual scripts do, exact commands to run each of them, and the
outstanding torso-tilt/waist investigation.

---

## 1. One-time setup per shell

```bash
source ~/soccerbot/.venv/bin/activate
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"
cd ~/soccerbot/scripted-behavior
```

- Robot on `enp5s0` at `192.168.123.161` (DDS).
- Teleimager (head RealSense + YOLO input) served from `192.168.123.164`:
  - RGB (JPEG) ZMQ SUB `:55555`
  - Depth ZMQ SUB `:55556` (unused for now)
  - Config REQ/REP `:60000`
- Killswitch pendant: `L2+B` damp, `L2+A` zero-torque, `L2+Y` stand,
  `START` engage balancer.

---

## 2. Full pipeline

```bash
python main.py --backend replay --iface enp5s0
```

Stages, in order:

| # | Module                                                          | What it does |
|---|-----------------------------------------------------------------|--------------|
| 1 | [pickup.py](scripted-behavior/pickup.py)                        | Replays `trajectories/pickup_ep148_prod2.json` (14-D arm qpos, 450 frames @ 30 fps ≈ 15 s) over `rt/arm_sdk`. |
| 2 | [turn_180.py](scripted-behavior/turn_180.py)                    | `LocoClient.Move(0,0, -0.6)` for `π/0.6 ≈ 5.24 s` (CW yaw) while holding the final pickup arm pose. |
| 3 | [avoid.py](scripted-behavior/avoid.py)                          | Polls YOLO on teleimager RGB. If any person within `AVOID_CLEAR_DISTANCE_M = 4.0`, run `sidestep left 2 → right 2`. Up to 8 cycles. |
| 4 | [throw.py](scripted-behavior/throw.py)                          | ⚠️ Not implemented — raises `NotImplementedError` until `trajectories/throw.json` exists. |

Currently the pipeline aborts at stage 4 with exit code 2 until throw.json is
recorded.

---

## 3. Individual stage commands

Run these from `~/soccerbot/scripted-behavior/`.

### 3.1 PICKUP

```bash
python pickup.py --iface enp5s0
```

Options (defined in [pickup.py](scripted-behavior/pickup.py) / [config.py](scripted-behavior/config.py)):
- `--backend replay` (default) → JSON replay
- `--backend local`  → subprocess to `local-vla-inference` ACT policy
- `--backend remote --remote-server HOST:PORT` → pi0.5 policy server

### 3.2 TURN 180

```bash
python turn_180.py --iface enp5s0
```

- `TURN_YAW_RATE_RPS = -0.6` (negative = CW when viewed from above).
- Uses **no-ramp** `arms.send_arm_positions(hold_pose, weight=1.0)` before the
  yaw command. See §5 for why this matters.

### 3.3 SIDESTEP (used by avoid)

```bash
python sidestep.py left 2 --iface enp5s0
python sidestep.py right 2 --iface enp5s0
```

Same no-ramp arm-hold patch as turn_180. Known caveat: engaging arm_sdk
degrades the whole-body balancer → visible torso wobble during the shuffle.

### 3.4 AVOID

```bash
# Full FSM (detect + sidestep):
python avoid.py --iface enp5s0 --teleimager-host 192.168.123.164

# Detect-only (safe, no motion, prints one line per poll):
python avoid.py --detect-only 20 --teleimager-host 192.168.123.164
```

Per-poll log lines you'll see:
- `poll N: no person (streak 1/2)` — YOLO didn't detect anything.
- `poll N: 1 person(s) but nearest 6.42 m > 4.0 m gate` — over threshold.
- `poll N: person at 2.10 m -- BLOCKED` — triggers sidestep.

### 3.5 THROW

Not implemented. To fill in:
1. Record arm qpos while manually posing the throw (backdrive + `diag_log.py`
   or extract from a LeRobot episode).
2. Save to `trajectories/throw.json` matching the pickup format
   (`frames: [{qpos: [...14 floats...]}, ...]`, `fps: 30`).
3. Point [throw.py](scripted-behavior/throw.py) at it (same pattern as `pickup.py`).

---

## 4. Human detector standalones

### 4.1 Live YOLO on teleimager, print + annotated image

```bash
python human_detector_teleimager.py \
    --host 192.168.123.164 \
    --secs 30 \
    --save-annotated logs/teleimager_annotated.jpg
```

`logs/teleimager_annotated.jpg` is overwritten every frame with green bboxes +
distance labels. Open it in VS Code / any auto-refreshing viewer.

Useful flags:
- `--within-m 4.0` — only print people closer than 4 m
- `--focal-px 906` — override intrinsic if config server unreachable
- `--secs 0.1` — one-shot

### 4.2 Grab a single raw frame

```bash
python -c "
import zmq, cv2, numpy as np
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.SUBSCRIBE, b''); sub.setsockopt(zmq.CONFLATE, 1); sub.setsockopt(zmq.RCVTIMEO, 3000)
sub.connect('tcp://192.168.123.164:55555')
buf = np.frombuffer(sub.recv(), dtype=np.uint8)
img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
cv2.imwrite('logs/teleimager_raw.jpg', img)
print('shape=', img.shape)
"
```

### 4.3 Teleimager config dump

```bash
python -c "
import zmq
ctx = zmq.Context()
s = ctx.socket(zmq.REQ); s.connect('tcp://192.168.123.164:60000')
s.send(b'GET_DATA')
print(s.recv_json())
"
```

Gives you `head_camera.color_intrinsics`, `image_shape`, `binocular`, etc.

---

## 5. Torso tilt / waist investigation summary

**Symptom.** During pickup replay the robot's torso pitches forward ~10°+
during the transition from pickup end → turn start. Visual "tilt" persists
even after fixes.

### What we tried

| Attempt | Approach | Result |
|---|---|---|
| A | **Waist joint lock** — added `lock_joint` hook in [g1_arms.py](local-vla-inference/g1_arms.py) that pins `kWaistYaw=12` to captured value; enabled in `arm_replay.py`. | `kWaistYaw` swing dropped 45° → 1.8°, but user-visible tilt unchanged. Reverted from `arm_replay.py`. Hook still available in `g1_arms.py`. |
| B | **No-ramp arm_sdk hold** at transition. Replaced 0→1 weight ramp with a single `send_arm_positions(hold_pose, weight=1.0)` publish before the yaw command. | ✅ Fixed the jerk. `pickup_then_turn_v3.csv` analysis: peak arm velocity 800°/s → 30°/s, torso pitch during transition 10.6° → 0.3°. Applied to `turn_180.py` + `sidestep.py`. |
| C | **Slower replay (15 fps variant)** — `trajectories/pickup_ep148_prod2_15fps.json`. | User reverted; kept the 30fps `pickup_ep148_prod2.json` as active. 15fps file left on disk pending removal. |
| D | **Waist SKU probe** — attempted a standalone `motor_state` dump to detect if joints 13/14 (waist roll/pitch) physically exist on this SKU. | Failed with `NO LOWSTATE` (likely `ChannelFactoryInitialize` env issue). Working path is `ensure_dds` in [diag_log.py](scripted-behavior/diag_log.py). Not resolved. |

### Root-cause hypothesis (not yet confirmed)

The remaining visible tilt is **the whole-body balancer squatting/leaning to
compensate for the extended arm mass** during pickup — not a control bug.
Waist locking fixes joint drift; it doesn't remove the balancer's response
to the arms' CoM shift.

Related discovery: engaging `arm_sdk` (weight=1) **downgrades the whole-body
balancer**. Waist yaw swings freely with arm_sdk on, freezes with it off.
This also causes the sidestep torso wobble.

### Next things to try

1. Publish a **very short** arm_sdk hold (< 100 ms) at transition, then
   release to `weight=0` before the yaw so balancer regains authority.
2. Record the pickup with the robot **holding heavier CoM offset** or record a
   trajectory that pre-leans, so the "tilt" is baked in as intentional.
3. Log `imu_state.rpy` around the transition to distinguish balancer
   compensation from control-error tilt.

### Files touched during the investigation

- [scripted-behavior/arm_replay.py](scripted-behavior/arm_replay.py) — waist lock removed
- [scripted-behavior/turn_180.py](scripted-behavior/turn_180.py) — no-ramp patch + yaw reversed
- [scripted-behavior/sidestep.py](scripted-behavior/sidestep.py) — no-ramp patch
- [local-vla-inference/g1_arms.py](local-vla-inference/g1_arms.py) — optional `lock_joint` hook (unused)

### Log evidence

| File | Purpose |
|---|---|
| `logs/pickup_then_turn.csv` | Baseline (with ramp) — transition jerk visible |
| `logs/pickup_waist_locked.csv` | Waist lock ON — yaw swing eliminated, tilt still present |
| `logs/pickup_then_turn_v3.csv` | No-ramp patch validated — 0.3° transition pitch |
| `logs/pickup_then_turn_v4.csv` | Reproducibility run of v3 |

Not committed (session artifacts). Analyze with pandas:

```bash
python -c "
import pandas as pd
df = pd.read_csv('logs/pickup_then_turn_v3.csv')
print(df[['t','waist_yaw_q','rpy_pitch_deg']].describe())
"
```

---

## 6. Diagnostic scripts

### 6.1 Full LowState dump (100 Hz)

```bash
python diag_log.py --iface enp5s0 --duration 40 --hz 100 --out logs/mytest.csv
```

Columns: `t, motor_q[0..28], motor_dq[...], motor_tau[...], rpy_roll/pitch/yaw_deg, gyro_x/y/z, accel_x/y/z, waist_yaw_q, ...`.

### 6.2 LocoClient state / FSM probe

```bash
python diag_loco.py --iface enp5s0
```

Prints `GetFsmId`, `GetFsmMode`, `GetGaitType`, `GetSpeedMode`, etc.

---

## 7. Known IPs, ports, joint indices

| Thing | Value |
|---|---|
| Workstation NIC | `enp5s0` `192.168.123.222/24` |
| G1 DDS | `192.168.123.161` |
| Teleimager host | `192.168.123.164` |
| Teleimager RGB (ZMQ SUB) | `:55555` (JPEG, 1280×720 binocular) |
| Teleimager depth (ZMQ SUB) | `:55556` (unused) |
| Teleimager config (REQ/REP) | `:60000` (`GET_DATA` → JSON) |
| Head RealSense fy (color) | 906.65 px @ 1280×720 |
| Legs | joints 0–11 |
| kWaistYaw | 12 |
| Arms | 15–28 |
| Weight joint | 29 |
| DDS topics | `rt/arm_sdk` (publish), `rt/lowstate` (subscribe) |

---

## 8. Environment quick-check

```bash
# Ping robot DDS
ping -c 3 192.168.123.161

# Ping teleimager
ping -c 3 192.168.123.164

# Confirm venv + deps
python -c "import cyclonedds, unitree_sdk2py, ultralytics, torch, zmq, cv2; print('all imports ok')"

# LocoClient smoke (won't move):
python diag_loco.py --iface enp5s0
```

---

## 9. Uncommitted artifacts (as of `098585a`)

Not part of the checkpoint — left on disk for reference or later cleanup:

- `act_20episodes/` — training data
- `act_log_20260718_192407.csv` — ACT training log
- `local-vla-inference/head_frame.jpg` — one-shot camera capture
- `scripted-behavior/home_pose.json` — origin unclear
- `scripted-behavior/logs/` — diagnostic CSVs + teleimager JPGs
- `scripted-behavior/trajectories/pickup_ep148_prod2_15fps.json` — reverted 15fps experiment
- `how_to_run.md` — older run notes, superseded by this doc

Delete or commit at your discretion.
