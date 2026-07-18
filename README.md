# Soccerbot

A Unitree G1 humanoid that picks a soccer ball up off a table and then acts like a
goalkeeper: turns around, watches for a person, and throws the ball straight ahead
once no one is around.

This file is the canonical project overview — architecture, current state, how the
pieces fit together, and what's still TODO. It's written for AI coding agents picking
up this repo cold as much as for humans. For Cursor-Cloud-specific dev-environment
setup (uv sync flags, CPU-only sim notes, lint/test commands) see [`AGENTS.md`](AGENTS.md)
— that file is the "how do I get a shell working here" doc; this one is the "what is
this project and why" doc. Don't duplicate between them: environment mechanics go in
`AGENTS.md`, product/architecture decisions go here.

## The task, precisely

Two phases, run back to back on the real robot:

1. **Learned pickup (VLA policy).** The robot sees a ball on a table of fixed height
   and picks it up. This is the only part that's learned from demonstrations.
2. **Scripted goalkeeper sequence (hardcoded, no learning involved).** Once the pickup
   finishes (a fixed number of timesteps after phase 1 completes):
   - Turn 180°.
   - Watch for a person using the depth-sense camera.
   - While a person is detected, shuffle side to side (like a goalkeeper guarding a
     goal).
   - Once no person has been detected for 6 continuous seconds, throw the ball
     straight ahead. The throw motion itself is a fixed, hardcoded trajectory — it is
     not learned or planned at runtime.

Phase 1 and phase 2 are deliberately separate systems (see [Repo layout](#repo-layout)
below) — phase 2 has no VLA/model dependency at all.

## Robot & sensor setup

- **Robot:** Unitree G1 humanoid.
- **Learned action space: arms only, 14-D** (`training/embodiment_g1.py`) — 7 DOF
  per arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) × left/right. No
  waist, no legs, no hand/finger joints in the learned policy.
- **Hands:** the robot physically has BrainCo Revo 2 hands, but **there is no learned
  grasping**. The hands are not part of the GR00T action space. Picking the ball up
  is achieved through arm positioning alone (the hand is held in a fixed pose); the
  earlier prototype that gave the hands 6 active DOF each (26-D total embodiment,
  see `training/embodiment_g1_revo2.py` in git history) was deliberately reverted —
  see commit `75d3330`.
- **Camera: single RGB-D camera, dual purpose.** The same physical camera feeds RGB
  frames to the GR00T policy during pickup *and* provides the depth stream used for
  human detection during the scripted phase. There is no separate dedicated
  depth-sense sensor. (LeRobot has existing RealSense driver support —
  `thirdparty/lerobot/src/lerobot/cameras/realsense/` — and dataset-level depth-map
  handling, which is the natural fit here, though the exact camera model/driver
  wiring is not yet chosen.)

## Architecture

```
                     ┌─────────────────────────┐
  100 real demo      │   training/ (Modal)      │
  episodes  ───────▶ │   fine-tune GR00T N1.7   │──▶ checkpoint (HF Hub / Modal volume)
  (TODO, see below)  │   on G1 arms-only 14-D   │
                     └─────────────────────────┘
                                                            │
                                                            ▼
                     ┌─────────────────────────┐    ┌──────────────────────────┐
  robot sensors ───▶ │   local-vla-inference/   │───▶│  scripted-behavior/      │
  (RGB-D camera,     │   loads checkpoint, runs │    │  hardcoded FSM: turn,    │
  arm state)         │   GR00T on real G1 arms  │    │  detect, shuffle, throw  │
                     │   for the pickup phase   │    │  (no model, no learning) │
                     └─────────────────────────┘    └──────────────────────────┘
```

The hard boundary between `local-vla-inference/` and `scripted-behavior/` is intentional
and was an explicit design decision (not just a naming convenience): the pickup
policy and the goalkeeper sequence must stay decoupled so the hardcoded phase never
depends on the model, and vice versa.

## Repo layout

This is a `uv` workspace (see root `pyproject.toml`, `[tool.uv.workspace] members`).

| Member | Status | Purpose |
|---|---|---|
| `training/` | **Implemented, tested, validated end-to-end on Modal** | CLI that submits LeRobot/GR00T/MolmoAct2 fine-tuning jobs to Modal GPUs. See [Training pipeline](#training-pipeline-training) below. |
| `thirdparty/lerobot/` | Vendored (Hugging Face LeRobot) | Essentially all real ML functionality (datasets, policies, training loop, GR00T/MolmoAct2 implementations) lives here. See `thirdparty/lerobot/AGENTS.md` for its own architecture notes. Treat as upstream — don't hand-edit unless necessary, and if you do, note it clearly since it complicates future upstream syncs. |
| `local-vla-inference/` | **Stub only** (`main.py` just prints hello), plus a working `install.sh` | Where the real-time control loop belongs: load the fine-tuned GR00T checkpoint, run it against live camera/arm-state input, drive the G1's arms via `unitree_sdk2py` during the pickup phase. The control loop itself isn't built yet, but `install.sh` already handles the tricky part of its dependency chain — see [Local inference setup](#local-inference-setup-local-vla-inference) below. (Renamed from `vla-inference/`.) |
| `scripted-behavior/` | **Stub only** (`main.py` just prints hello), newly scaffolded | The hardcoded post-pickup FSM: turn 180°, depth-based human detection, side-to-side shuffle, 6s-timeout-triggered throw. Deliberately kept out of `local-vla-inference/` — no model dependency, pure scripted robot control. Depends on `unitree_sdk2py`; will likely need a depth-frame human-detection method (simple depth-based presence/proximity check vs. a person-detection model — not yet decided). |
| `soccerbot/` | **Empty stub**, unclear future role | `src/soccerbot/main.py` and `README.md` are both empty. Original root orchestrator idea from the very first commit ("create core orchestrator"); nothing built on it since. Don't assume it has a settled purpose — check with the user before building on it. |
| `datasets/lerobot_dummy_1episode.zip` | Synthetic fixture, **not real robot data** | A 1-episode, 60-frame LeRobot v2.1-format dataset with a 6-DOF SO-100 arm layout (task: "pick up the red cube"). Used only to smoke-test the training pipeline (see below) — it has no relationship to the G1's action space and should never be mistaken for real task data. |

## Training pipeline (`training/`)

`training/main.py` is a Modal launcher for `lerobot-train`. It:

- Bakes the vendored `thirdparty/lerobot` source into a Modal container image and
  installs it with the extras needed per-policy (`dataset,training,groot,molmoact2`).
- Resolves `--dataset` as either a local directory (uploads it to a Modal Volume) or
  a Hugging Face Hub id.
- Normalizes friendly GPU names (`h200`, `a100-80gb`, `b100`, `H100:2`, ...) into
  Modal GPU shortcodes.
- Translates CLI flags into `lerobot-train` args, forwards `HF_TOKEN`/`WANDB_*` from
  the local environment as Modal secrets, and runs training on a GPU worker.

Supported `--policy` values include `groot` and `molmoact2` (the two that matter for
this project) plus LeRobot's other stock policies (`act`, `diffusion`, `smolvla`,
`pi0`, etc. — see `KNOWN_POLICIES` in `training/main.py`).

### GR00T (primary path for this project)

```bash
modal run training/main.py --dataset ./g1_demos --gpu B200 --policy groot --steps 5000
```

When `--policy groot` is used, `training/main.py` automatically:

- Points at `nvidia/GR00T-N1.7-3B` (`training/embodiment_g1.py: BASE_MODEL_PATH`)
  with `--policy.embodiment_tag=new_embodiment`.
- Freezes the VLM backbone and vision tower (`tune_llm=false`, `tune_visual=false`,
  `tune_top_llm_layers=0`); trains the DiT action head, projector, and VLM layernorms
  (`tune_projector=true`, `tune_diffusion_model=true`, `tune_vlln=true`). No LoRA.
- Uses bf16 (`use_bf16=true`) and relative actions (`use_relative_actions=true`).
- **Validates the local dataset's `meta/info.json` against the 14-D arms-only layout
  before submitting** (`validate_dataset_layout` in `embodiment_g1.py`) — fails fast
  locally rather than burning GPU time on a mismatched dataset. This only runs for
  local dataset paths, not Hub ids.

GR00T defaults from LeRobot's own config (`thirdparty/lerobot/src/lerobot/policies/groot/configuration_groot.py`):
`chunk_size=40`, `n_action_steps=40`, `max_state_dim=132`/`max_action_dim=132`
(padded — our 14-D state/action fits well inside), default `batch_size=32`.

`--onlyactionexpertft` is accepted for `groot` as a no-op (GR00T already trains
action-expert-only by default; the flag exists so the same CLI surface works for
both policies below).

### MolmoAct2 (secondary/experimental path)

MolmoAct2 support was built and **validated end-to-end** in an earlier session: a
LoRA-VLM fine-tune (500 steps, batch 8, H100) was run against the dummy SO-100
dataset above and pushed to `AmoghShrivastava1/soccerbot-molmoact2-dummy-smoketest`
on Hugging Face. That run proved the Modal → LeRobot → MolmoAct2 training pipeline
works, but **it is not G1-relevant** — wrong dataset, wrong embodiment, wrong task,
purely a pipeline smoke test. Default mode is LoRA-VLM fine-tuning
(`--policy.enable_lora_vlm=true`); pass `--onlyactionexpertft` to instead freeze the
VLM and train only the action expert (requires continuous action mode).

GR00T is the intended production path for this project; MolmoAct2 is kept around as
an alternative worth revisiting if GR00T doesn't pan out.

### Known gotchas (hard-won, worth reading before re-debugging them)

These came up fine-tuning on Modal from a **Windows** machine — may not all apply
from Linux/Mac, but the first two are Modal-specific and apply everywhere:

1. **`modal.Image.add_local_dir` must be the last build step, unless `copy=True`.**
   Our image needs to `uv pip install -e` the locally-mounted LeRobot source *after*
   adding it, so `add_local_dir(..., copy=True)` is required — otherwise Modal raises
   `InvalidError` at image-build time.
2. **LeRobot dataset format v3.0 is required; older exports are v2.1.** If you hit
   `BackwardCompatibilityError`, convert with LeRobot's own
   `python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id=<id> --root=<path> --push-to-hub=false`
   (the CLI's `--push-to-hub` defaults to `true` with no way to fully suppress the
   tag/delete calls on the Hub side if you don't pass `false` and don't have write
   access — pass it explicitly).
3. **Policies using `QUANTILES` normalization (GR00T, MolmoAct2) need `q01`/`q99`
   dataset stats.** Older/hand-built datasets often don't have them. Fix with
   `lerobot.scripts.augment_dataset_quantile_stats` — but note **its CLI
   unconditionally calls `dataset.push_to_hub()` with no opt-out flag**. If you don't
   want a Hub push (e.g. no token, or a purely local dataset), call
   `compute_quantile_stats_for_dataset` + `write_stats` directly instead of using the
   script's `main()`.
4. **On Windows specifically:** `pathlib.Path("/outputs")` / `Path("/data")` resolve
   to `WindowsPath` locally, so remote-side path strings built on the submitting
   machine come out backslash-separated and break on the Linux Modal worker — use
   `PurePosixPath` for anything that becomes a remote path string (already fixed in
   `training/main.py`). Also, **Git Bash/MSYS silently rewrites POSIX-looking CLI
   args** (e.g. `--dataset.root=/data/...`) into Windows paths before your process
   ever sees them — prefix commands with `MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*"`
   to stop it. And Modal's CLI prints Unicode (✓) that breaks on Windows' default
   `cp1252` console encoding when output is redirected — set
   `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.
5. **`modal volume get` has hit permission/directory-download bugs on Windows** when
   pulling a directory back to local disk. Workaround used successfully: don't
   download at all — point `--dataset.root` (or any file op) directly at the Modal
   Volume path from a one-off `modal run` script that does the work *inside* the
   container (e.g. pushing a checkpoint straight to the Hub from within the
   container instead of downloading it locally first).

## Data collection

**Not started.** Plan is ~100 real-world episodes of the G1 picking a ball up off a
table of fixed height, in LeRobot dataset format (v3.0). No collection tooling,
protocol, or storage location has been decided yet — this is the next concrete
blocker before a real GR00T fine-tune can happen. Do not confuse this with
`datasets/lerobot_dummy_1episode.zip`, which is synthetic SO-100-arm fixture data
used only for pipeline testing (see table above).

Open questions once collection starts: teleop method, camera calibration/mounting,
episode length/task-string convention, and where the resulting dataset lives (local
disk uploaded per-run via `training/main.py --dataset ./path`, or pushed to the Hub
first).

## Current status / TODO

- [x] Modal + LeRobot training pipeline (`training/`) — built, tested, validated
      end-to-end (both GR00T CLI wiring and a real MolmoAct2 training run).
- [x] G1 arms-only 14-D embodiment definition + dataset-layout validation
      (`training/embodiment_g1.py`).
- [ ] Collect ~100 real pickup episodes.
- [ ] Fine-tune GR00T on real G1 data (currently only wired, never run against real
      data).
- [ ] Build `local-vla-inference/`: real-time control loop running the fine-tuned
      GR00T checkpoint against live sensors, driving the G1 via `unitree_sdk2py`.
- [ ] Build `scripted-behavior/`: turn-180 → depth-based human detection → shuffle →
      6s-timeout throw FSM. Human-detection method not yet chosen.
- [ ] Design and hardcode the actual throw trajectory.
- [ ] Decide camera hardware/driver (RealSense is the natural fit given existing
      LeRobot support, not confirmed).
- [ ] Decide what, if anything, `soccerbot/` is for.

## Setup

See [`AGENTS.md`](AGENTS.md) for environment mechanics (uv sync flags, Python
version pinning, CPU-only sim caveats, lint/test commands). In short:

```bash
uv sync --locked --all-packages --extra dev --extra test --extra pusht -p 3.13
```

### Local inference setup (`local-vla-inference/`)

`unitree_sdk2py` depends on CycloneDDS, which needs a local C build —
`local-vla-inference/install.sh` handles this: clones and builds CycloneDDS
(`releases/0.10.x` branch) to `~/cyclonedds/install` (override with
`CYCLONE_SRC`/`CYCLONE_PREFIX`/`CYCLONE_BRANCH` env vars), then runs
`uv sync -p 3.13 --all-packages`. `CYCLONEDDS_HOME` must stay set whenever
importing `unitree_sdk2py`. This is a real build step (needs `cmake`), separate
from — and unrelated to — the Modal-side training setup below.

For Modal training runs you'll additionally need `modal setup` (or
`modal token set --token-id ... --token-secret ...`) to authenticate, and optionally
`HF_TOKEN` in the environment if pushing trained policies to the Hub.
