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

# Real robot (robot LAN up, e.g. enp5s0)
./remote-vla-inference/run_client.sh \
  --server_address=HOST:PORT \
  --robot.is_simulation=false \
  --robot.robot_ip=192.168.123.164
```

Arms only: first **14** action dims → L/R arm joints. Grippers / legs / waist / remote from the policy are **masked**. Legs stay on `GrootLocomotionController`.

## Known issues / gotchas

1. **Feature / camera mismatch** — Checkpoint expects `observation.images.global_view` + 29-D state. Your WaldOS cams may use different keys (`color_0`, head stereo, …). Without a `--rename_map` / matching camera config, the server fails at preprocess or sees blank inputs.

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
