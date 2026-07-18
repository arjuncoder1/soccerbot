# Soccerbot workspace

See [`README.md`](README.md) for the project overview, architecture, and current
status — this file is environment/dev-workflow mechanics only.

A `uv` workspace with five members (see root `pyproject.toml`):

- `soccerbot/` — currently an empty stub (`src/soccerbot/main.py` is empty); no settled purpose.
- `training/` — Modal launcher for LeRobot policy training (`training/main.py`); depends on workspace `lerobot` + `modal`.
- `thirdparty/lerobot/` — vendored Hugging Face LeRobot library; this is where essentially all real functionality lives. See `thirdparty/lerobot/AGENTS.md` for its architecture, lint/test/build commands, and per-module notes.
- `local-vla-inference/` — ACT inference on the real G1 (arms only). Workspace member; uses the **repo-root** `.venv`. On the robot, sync with **Python 3.12** (`uv sync -p 3.12 --all-packages` or `./local-vla-inference/install.sh`) — `unitree_sdk2py` → `cyclonedds==0.10.2` breaks on 3.13 (`_Py_IsFinalizing`).
- `scripted-behavior/` — stub; hardcoded (non-learned) post-pickup FSM (turn, human-detect, shuffle, throw). Deliberately kept separate from `local-vla-inference/`.

## Cursor Cloud specific instructions

- Package manager is `uv` (installed at `/usr/local/bin/uv`). The startup update script runs `uv sync --locked --all-packages --extra dev --extra test --extra pusht -p 3.13`, which creates `.venv/` with all workspace members editable plus lerobot's dev/test tools and the `pusht` sim env. Run commands via `uv run <cmd>` or `.venv/bin/<cmd>`.
- Python: workspace allows `requires-python >=3.12`. Cloud/dev can use `-p 3.13`. **Robot / unitree**: sync root with `-p 3.12` (cyclonedds 0.10.2 is broken on 3.13). Do not use system `python3`; use the root `.venv`.
- This is a CPU-only VM (no GPU). Pass `--policy.device=cpu` to `lerobot-train`/`lerobot-eval`. Simulator eval renders headlessly via `opencv-python-headless` + ffmpeg (already installed).
- Sim env extras: `pusht` installs cleanly. The `aloha` extra does NOT install here — it pulls `dm-control` → `labmaze`, which has no cp313 wheel and tries to build with `bazel` (absent). Use `pusht` for E2E train/eval smoke tests, or install `bazel` if `aloha` is truly needed.
- Policy extras: some policies need their own extra even for a tiny run, e.g. `diffusion` needs `--extra diffusion` (diffusers). `tdmpc` and `act` are core (no extra). A quick CPU E2E that works out of the box: `uv run lerobot-train --policy.type=tdmpc --policy.device=cpu --env.type=pusht --env.episode_length=5 --dataset.repo_id=lerobot/pusht_image --dataset.episodes="[0]" --batch_size=2 --steps=2 --eval.n_episodes=1 --eval.batch_size=1 --wandb.enable=false --policy.push_to_hub=false --output_dir=tests/outputs/tdmpc_hello/` (downloads a public HF dataset; needs network).
- Lint: `uv`'s dev extra installs a newer `ruff` (0.15.x) than the version pinned in `thirdparty/lerobot/.pre-commit-config.yaml` (0.14.1), so `ruff check` reports a few extra findings (e.g. `UP042`) that CI does not. For authoritative lint/format matching CI, use the pinned versions via `pre-commit` from within `thirdparty/lerobot/`. `ruff format --check src` is clean.
- Tests: run pytest from `thirdparty/lerobot/` (e.g. `uv run pytest tests/optim tests/processor -q`). Many tests skip without hardware, network, or LFS artifacts (this checkout has no `tests/artifacts/` and no LFS files). E2E training/eval targets live in `thirdparty/lerobot/Makefile`.

## Learned User Preferences

- Prefer rebasing PR branches onto `main` rather than merging `main` into them.
- Keep Git LFS out of this repo; do not commit or push LFS-tracked artifacts (including LeRobot `tests/artifacts`).
- Use the workspace root `.venv` / `uv run` for Python and Modal — not system `python3` and not a nested `training/.venv`.
- Prefer simple, hackathon-stable training wiring over speculative analysis or extra validation passes unless explicitly requested.

## Learned Workspace Facts

- Target robot/task is Unitree G1 soccer ball pickup; primary policy path is NVIDIA GR00T N1.7 (`nvidia/GR00T-N1.7-3B`) via `training/main.py` on Modal.
- Current G1 embodiment is arms-only 14-D (left/right arm × 7 joints; no hands, waist, or legs) in `training/embodiment_g1.py` with embodiment tag `new_embodiment`.
- Groot fine-tune strategy: freeze the VLM backbone; train DiT action head + projector; no LoRA.
- `training` depends on the local workspace package `lerobot` from `thirdparty/lerobot` (`lerobot = { workspace = true }`); sync from the repo root.
- Root is a `uv` workspace (`training`, `thirdparty/lerobot`, `soccerbot`, `local-vla-inference`, `scripted-behavior`) with a shared `.venv` at the repo root. On the robot that venv must be Python 3.12 for `unitree_sdk2py`.
- No grasping is learned: the G1 has BrainCo Revo 2 hands, but they are not part of the 14-D action space; the ball is picked up via arm positioning with the hand held in a fixed pose.
- Camera is a single RGB-D unit doing double duty: RGB feeds GR00T during pickup, depth feeds the post-pickup human-detection FSM.
- Post-pickup behavior (turn 180°, depth-based human detection, shuffle, 6s-timeout hardcoded throw) belongs in `scripted-behavior/`, not `local-vla-inference/` — no model dependency by design.
- 100-episode real-world pickup dataset collection has not started; `datasets/lerobot_dummy_1episode.zip` is unrelated synthetic SO-100 fixture data used only for training-pipeline smoke tests.
