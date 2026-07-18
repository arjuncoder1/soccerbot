# Handoff — Remote VLA inference on Unitree G1 (Modal π0.5)

Context and working state for the next agent. Last updated **2026-07-18**.

## Goal

Run **LeRobot async remote policy inference** for a **Unitree G1** (arms/shoulders
only; legs on the stock balancer; no hands) using a **π0.5** checkpoint in LeRobot
format, with the GPU policy on **Modal** and the control loop on the robot PC.

- Default policy: `sudoping01/pi05_g1_boxmove_v2` (LeRobot `pi05`, 18-D action).
- Repo root: `/home/waldo/soccerbot` (not a git repo at root; `thirdparty/lerobot` is vendored).
- Key dir: `remote-vla-inference/`.

## Current status (what works)

- **Modal server**: `policy_server.py` serves the π0.5 checkpoint on a single **A100**.
  Verified end-to-end: returns 50 actions × 18-D in ~1.3s cold / ~0.3s warm.
- **Real-robot client**: `g1_client.py` + `g1_dds_robot.py` connect to the G1 over
  **direct DDS** (real Unitree SDK) exactly like `local-vla-inference` — subscribe
  `rt/lowstate`, publish `rt/arm_sdk`. Balancer untouched, nothing extra runs on the robot.
- Network verified earlier: `enp5s0` up on `192.168.123.222/24`, robot `192.168.123.164` pingable.

## How to run

Terminal A — server:
```bash
cd /home/waldo/soccerbot
export CYCLONEDDS_HOME=$HOME/cyclonedds/install
./remote-vla-inference/run_server.sh          # prints --server_address=HOST:PORT
```

Terminal B — client (real robot):
```bash
cd /home/waldo/soccerbot
./remote-vla-inference/run_client.sh \
  --server_address=HOST:PORT \
  --robot.is_simulation=false \
  --robot.robot_ip=192.168.123.164
```

Sim smoke test: same but drop the last two flags.

Useful env vars (see `run_client.sh` / README):
`IFACE=enp5s0` (robot NIC), `CAMERA=zmq://192.168.123.164:55555` (teleimager head),
`G1_ARM_KP=60`/`G1_ARM_KD=1.5`, `ARM_SLEW_CLAMP=0.01` (safety), `LOG_CSV=...`,
`G1_DIRECT_DDS=1` (set `0` to use LeRobot's ZMQ bridge instead), `TASK="pick up the red ball"`.

## Architecture / key decisions

- **Why direct DDS (not LeRobot's stock non-sim path):** LeRobot's `UnitreeG1`
  (`is_simulation=false`) uses a **ZMQ socket bridge** (`unitree_sdk2_socket.py`) that
  requires `run_g1_server.py` running on the robot's Orin. Nothing runs there, so
  `rt/lowstate` never arrives → `TimeoutError: Timed out waiting for robot state`.
  `g1_dds_robot.py` (`G1ArmsDDSRobot`) bypasses that and reuses the validated local
  `G1Arms` (`rt/arm_sdk`) + `make_front_camera` (teleimager) behind a LeRobot `Robot`
  adapter, injected into `RobotClient` via a `make_robot_from_config` monkeypatch in
  `g1_client.py`.
- **Observation layout** (matches checkpoint): 29 joint `*.q` in `G1_29_JointIndex`
  order → `observation.state`; one `global_view` frame (teleimager head, resized
  480×640) → `observation.images.global_view`.
- **Action:** policy emits 18-D; client uses the first **14** as L/R arm joints, masks
  the rest (grippers/legs/waist/remote).
- **Safety:** every action passes through `apply_safety()` — slew-rate clamp
  (`ARM_SLEW_CLAMP` rad/step, default 0.01 ≈ 0.3 rad/s @30fps), per-step CSV log.
- **Ctrl+C = stop and stay standing** (`stop_and_stand()`): `LocoClient.StopMove()` stops
  locomotion (balancer keeps robot upright, does NOT go limp) + release `arm_sdk`
  (weight→0) so arms hand back to the balancer. `robot.freeze()` (stiff hold) also exists.
  The user explicitly wants stay-standing, not damp. `LocoClient.Damp()`/`ZeroTorque()`
  are available if a true limp-stop is ever requested (only safe on a hoist/stand).

## Critical fixes already made (do NOT re-break)

- **torch.compile off on server:** the checkpoint ships `compile_model=True,
  mode=max-autotune`; on the Modal image (torch 2.11) TorchInductor stalls for minutes
  then SIGSEGVs. `policy_server.py` forces `config.compile_model=False` at load.
- **Gated tokenizer:** `google/paligemma-3b-pt-224` is gated (401). Server uses ungated
  mirror `leo009/paligemma-3b-pt-224`.
- **`grpcio`/`protobuf` are BASE deps** of `remote-vla-inference` (not just the `client`
  extra), else `uv` prunes them → `ModuleNotFoundError: No module named 'grpc'`.
- **Python 3.12** for the client stack (unitree/cyclonedds unhappy on 3.13).
- **CycloneDDS**: real SDK needs the prebuilt C lib at `$HOME/cyclonedds/install`;
  `run_client.sh` exports `CYCLONEDDS_HOME` + `LD_LIBRARY_PATH`.

## Files

| File | Role |
|---|---|
| `policy_server.py` | Modal gRPC PolicyServer (A100, torch.compile disabled, tokenizer mirror) |
| `g1_client.py` | `ArmsOnlyRobotClient` — masking, slew safety, CSV log, Ctrl+C stop, direct-DDS injection |
| `g1_dds_robot.py` | `G1ArmsDDSRobot` — LeRobot Robot adapter over direct DDS (rt/arm_sdk) + LocoClient |
| `run_server.sh` / `run_client.sh` | Launch helpers |
| `README.md` | Full docs (working config, safety, gotchas) |

Local reference (validated): `local-vla-inference/` — `main.py`, `g1_arms.py`,
`front_camera.py`, `diag_state.py`.

## Live resources (may expire)

- Last Modal server address: `r434.modal.host:46777` (dies when `run_server.sh` exits or
  `--hours` elapses — re-run the server and use the newly printed address).
- Check with: `modal app list` (look for `remote-vla…`, state `ephemeral`, 1 task).

## Open items / caveats

- Domain gap: box-move demos ≠ this lab; expect poor zero-shot success (pipe smoke-test only).
- EE mismatch: trained with Dex-style grippers, not BrainCo fingers; hands masked.
- Verify 18-D action joint ordering against the dataset before trusting arm directions.
- gRPC tunnel is insecure (raw TCP) — fine for short experiments only.
