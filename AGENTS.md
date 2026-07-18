# Soccerbot workspace

A `uv` workspace with three members (see root `pyproject.toml`):

- `soccerbot/` — currently an empty stub (`src/soccerbot/main.py` is empty).
- `training/` — depends on `lerobot[dataset,training]` + `modal`; the training entry point (`training/main.py`) is a stub.
- `thirdparty/lerobot/` — vendored Hugging Face LeRobot library; this is where essentially all real functionality lives. See `thirdparty/lerobot/AGENTS.md` for its architecture, lint/test/build commands, and per-module notes.

## Cursor Cloud specific instructions

- Package manager is `uv` (installed at `/usr/local/bin/uv`). The startup update script runs `uv sync --locked --all-packages --extra dev --extra test --extra pusht -p 3.13`, which creates `.venv/` with all workspace members editable plus lerobot's dev/test tools and the `pusht` sim env. Run commands via `uv run <cmd>` or `.venv/bin/<cmd>`.
- Python: the workspace root pins `requires-python >=3.13`. Pin sync to `-p 3.13` — if `uv` picks 3.14, torch/other wheels are unavailable for cp314. Do not rely on system `python3` (3.12); use the `.venv`.
- This is a CPU-only VM (no GPU). Pass `--policy.device=cpu` to `lerobot-train`/`lerobot-eval`. Simulator eval renders headlessly via `opencv-python-headless` + ffmpeg (already installed).
- Sim env extras: `pusht` installs cleanly. The `aloha` extra does NOT install here — it pulls `dm-control` → `labmaze`, which has no cp313 wheel and tries to build with `bazel` (absent). Use `pusht` for E2E train/eval smoke tests, or install `bazel` if `aloha` is truly needed.
- Policy extras: some policies need their own extra even for a tiny run, e.g. `diffusion` needs `--extra diffusion` (diffusers). `tdmpc` and `act` are core (no extra). A quick CPU E2E that works out of the box: `uv run lerobot-train --policy.type=tdmpc --policy.device=cpu --env.type=pusht --env.episode_length=5 --dataset.repo_id=lerobot/pusht_image --dataset.episodes="[0]" --batch_size=2 --steps=2 --eval.n_episodes=1 --eval.batch_size=1 --wandb.enable=false --policy.push_to_hub=false --output_dir=tests/outputs/tdmpc_hello/` (downloads a public HF dataset; needs network).
- Lint: `uv`'s dev extra installs a newer `ruff` (0.15.x) than the version pinned in `thirdparty/lerobot/.pre-commit-config.yaml` (0.14.1), so `ruff check` reports a few extra findings (e.g. `UP042`) that CI does not. For authoritative lint/format matching CI, use the pinned versions via `pre-commit` from within `thirdparty/lerobot/`. `ruff format --check src` is clean.
- Tests: run pytest from `thirdparty/lerobot/` (e.g. `uv run pytest tests/optim tests/processor -q`). Many tests skip without hardware, network, or LFS artifacts (this checkout has no `tests/artifacts/` and no LFS files). E2E training/eval targets live in `thirdparty/lerobot/Makefile`.
