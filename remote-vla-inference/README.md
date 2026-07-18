# Remote VLA inference (Modal + G1)

LeRobot **async** remote inference: GPU policy on Modal, control loop on the robot PC.

| Piece | Role |
|---|---|
| `policy_server.py` | Modal gRPC `PolicyServer` |
| `g1_client.py` | G1 `RobotClient` — **arms/shoulders only** |
| `run_server.sh` / `run_client.sh` | Launch helpers |

**Default policy:** [`sudoping01/pi05_g1_boxmove_v2`](https://huggingface.co/sudoping01/pi05_g1_boxmove_v2) (LeRobot `pi05`, G1 box-move, 18-D action).

## Working configuration (checkpoint — verified 2026-07-18)

End-to-end verified: client `Ready → SendPolicyInstructions → SendObservations → GetActions`
returns **50 actions × 18-D** in ~1.3s cold / ~0.3s warm on A100.

**Environment**

| Thing | Value |
|---|---|
| Client env | workspace `.venv` at `/home/waldo/soccerbot/.venv`, **Python 3.12** |
| Backup DDS env | conda `remote-vla-g1` (Python 3.12.13) |
| Key deps | `grpcio 1.73.1`, `torch 2.11.0+cu128`, `lerobot 0.6.1`, `modal 1.5.2`, `numpy 2.2.6` |
| CycloneDDS | prebuilt C lib at `$HOME/cyclonedds/install` → export `CYCLONEDDS_HOME` before client |
| Modal image | `debian_slim(python_version="3.12")`, installs `lerobot[async,pi]` |
| Modal server | `gpu=A100`, `fps=30`, `port=8080`, `--hours 2` |
| Tokenizer | ungated mirror `leo009/paligemma-3b-pt-224` (google repo is gated) |
| **Critical fix** | server forces `config.compile_model=False` at load — checkpoint ships `compile_model=True, mode=max-autotune`, which stalls + SIGSEGVs TorchInductor on this image |
| **Reload cache** | `SendPolicyInstructions` skips the ~200 s rebuild when `(policy_type, path, device)` matches the resident policy — client Ctrl+C → restart cycles complete in ~0.5 s |

**Env facts that keep biting (do not "fix" again):**
- `grpcio`/`protobuf` are **base** deps of `remote-vla-inference` (not just the `client`
  extra). They only otherwise arrive via `lerobot[async]`, so any `uv` run without
  `--extra client` used to prune `grpcio` → `ModuleNotFoundError: No module named 'grpc'`.
- Always launch the client via `uv run --package remote-vla-inference --extra client -p 3.12`
  (what `run_client.sh` does) so the env is synced correctly.
- The Modal `Heartbeat ... Deadline exceeded` warnings are a benign laptop↔Modal
  connectivity quirk (appear even in trivial functions); **not** an inference problem.

**Reproduce the working run:**
```bash
cd /home/waldo/soccerbot
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
./remote-vla-inference/run_server.sh                       # prints --server_address=HOST:PORT
./remote-vla-inference/run_client.sh --server_address=HOST:PORT --task="move the blue box"
```

## Quick start

### 1. Modal auth (once)
```bash
modal token set --token-id ak-... --token-secret as-...
# or: modal token new
```

### 2. Start the server (terminal A)
```bash
./remote-vla-inference/run_server.sh
# optional: GPU=A100 HOURS=2 ./remote-vla-inference/run_server.sh
```
Copy the printed address, e.g. `--server_address=rXXX.modal.host:NNNNN`.

### 3. Start the G1 client (terminal B)
```bash
# Sim first
./remote-vla-inference/run_client.sh --server_address=HOST:PORT

# Real robot (direct DDS, exactly how local-vla-inference connects)
cd /home/waldo/soccerbot
./remote-vla-inference/run_client.sh \
  --server_address=HOST:PORT \
  --robot.is_simulation=false \
  --robot.robot_ip=192.168.123.164
```

**Real-robot transport (matches `local-vla-inference`).** For `--robot.is_simulation=false`
the client bypasses LeRobot's ZMQ socket bridge (which needs `run_g1_server.py` running
on the robot) and drives the G1 over **direct DDS via the real Unitree SDK**
(`g1_dds_robot.py`): subscribe `rt/lowstate`, publish `rt/arm_sdk`. This runs alongside
the robot's stock balancer — **nothing extra runs on the robot**.

| Env var | Default | Meaning |
|---|---|---|
| `IFACE` | `enp5s0` | Robot-LAN network interface (the local run's `--iface`). Must be **up** and on `192.168.123.x`. |
| `CAMERA` | `zmq://<robot_ip>:55555` | Front-cam source (teleimager head stream), same as local. |
| `G1_ARM_KP` / `G1_ARM_KD` | `60` / `1.5` | arm_sdk joint gains (match local). |
| `G1_DIRECT_DDS` | `1` | `0` falls back to LeRobot's ZMQ bridge (`GrootLocomotionController`, needs the robot server). |

Arms only: first **14** action dims → L/R arm joints. Grippers / legs / waist / remote from the policy are **masked**. Legs stay on the robot's **stock balancer** (no LeRobot leg controller).

Default task prompt is **`pick up the red ball`** (override with `TASK=... ` or `--task=...`).

### Iterating: Ctrl+C safety and policy caching

- **Ctrl+C in the client terminal kills only the client.** The server (terminal A)
  keeps running with the policy resident on the GPU. The client's shutdown handler
  runs `stop_and_stand()` first, so the robot stops and stays standing.
- **Ctrl+C in the server terminal kills the Modal container.** The tunnel
  address dies; you'll get a new `HOST:PORT` the next time you start it.
- **Client re-runs are fast.** `SendPolicyInstructions` on the server caches by
  `(policy_type, pretrained_name_or_path, device)` — the first client after the
  server starts pays the full ~200 s model load, every subsequent client reconnect
  on the same checkpoint returns in ~0.5 s (`Reusing cached policy ...` in the
  server log). Change the checkpoint/device and it rebuilds.
- Typical iteration: leave terminal A alone, Ctrl+C and restart the command in
  terminal B as needed. Restart terminal A only if you change server code, hit
  the `--hours` limit, or want a fresh Modal container.

### Safety / logging (adapted from `local-vla-inference`)

The client gates every action through `apply_safety()` before it reaches the robot:

| Env var | Default | Meaning |
|---|---|---|
| `ARM_SLEW_CLAMP` | `0.01` | Max rad each arm joint may move per control step toward the policy target. `0.01` @ 30 fps ≈ 0.3 rad/s (very slow). Set `0` to disable (**arms then follow raw policy targets**). |
| `LOG_CSV` | `remote_vla_log_<ts>.csv` | Per-step CSV: `target/cmd/meas` per joint + `clamp_hits`, `max_target_gap`. Empty string disables. |
| `LOG_JOINTS_EVERY` | `5` | Console arm log cadence (cmd vs meas, gap, clamp count). |
| `ARM_ACTIONS_RELATIVE` | `0` | Treat policy output as delta from current pose. |

- **Slew limit** is the primary safety net: the commanded pose creeps toward the
  target ≤ `ARM_SLEW_CLAMP` rad/step, seeded from the measured pose (no startup jump).
- **Ctrl+C = stop and stay standing** (`stop_and_stand()`): `LocoClient.StopMove()`
  stops any locomotion (balancer keeps the robot **upright**, does not go limp) and the
  `arm_sdk` overlay is released (weight → 0) so the arms hand back to the balancer.
  A stiff-hold `robot.freeze()` is still available if you prefer arms locked in place.
- On the real robot this client reuses the *validated* `local-vla-inference` DDS stack
  (`G1Arms` on `rt/arm_sdk` + teleimager camera) behind a LeRobot `Robot` adapter
  (`g1_dds_robot.py`), so the transport is identical to the local run.

## Known issues / gotchas

1. **Feature / camera layout** — Checkpoint expects `observation.images.global_view` + 29-D state. The direct-DDS adapter (`g1_dds_robot.py`) provides exactly this: 29 joint `*.q` in `G1_29_JointIndex` order + a single `global_view` frame from the teleimager head cam (resized to 480×640). If your camera source differs, set `CAMERA=zmq://HOST:PORT` / `teleimager://HOST`.

2. **Action layout guess** — We assume 18-D = **14 arm + extras (grippers/…)**. If Unitree’s dataset ordered joints differently, arms can move wrong. Verify against the dataset feature names before real-robot deploy.

3. **EE mismatch** — Trained with Dex-style grippers / unitree_lerobot layout, **not BrainCo fingers**. Hands are masked on purpose; grasp behavior will not transfer.

4. **Domain gap** — Box-move demos ≠ your lab. Expect poor zero-shot task success; good for pipe smoke-test only until you finetune on your data.

5. **Network latency** — Modal RTT can be tens–hundreds of ms. Async chunking helps (`actions_per_chunk=50`), but the queue can underrun; lower `fps` or raise chunk size if arms stutter.

6. **VRAM / cold start** — π0.5 needs a large GPU (A100+). First handshake downloads weights from HF (minutes). Image rebuild on code change also takes time.

7. **Robot LAN** — Real G1 needs the robot NIC up (`enp5s0`) and DDS reachability to `192.168.123.164`. If the interface is down, only sim works.

8. **Insecure gRPC** — Tunnel is raw TCP (`grpc.insecure_channel`). Fine for a short experiment; do not expose on untrusted networks long-term.

9. **Safety** — Always keep loco for balance, start with low gains / sim, e-stop ready. Masking legs does not make arm motions safe.

10. **Modal tunnel lifetime** — Address dies when `run_server.sh` exits or `--hours` elapses. Restart server and update `--server_address`.

## Swap policy
```bash
POLICY=other/pi05-g1-ckpt POLICY_TYPE=pi05 \
  ./remote-vla-inference/run_client.sh --server_address=HOST:PORT
```
