# data-pipeline

Rerun hackathon **Track 3** entry: "Best end-to-end port of a non-SO-101 robot" —
reproduce the SO-101 reference pipeline (teleop → Rerun recordings → local
catalog → LeRobot v3 export → replay/deploy) for the Unitree G1.

## Data source

The **real** teleoperated episodes the deployed ACT checkpoint
[`ajkoder/g1-pickup-ball-act`](https://huggingface.co/ajkoder/g1-pickup-ball-act)
was trained on:
[`ajkoder/g1_final_cleaned`](https://huggingface.co/datasets/ajkoder/g1_final_cleaned)
— 121 real G1 pickup episodes, 74,371 frames, 14-D arm state/action, single
`color_0` camera @ 720×1280 (confirmed against that dataset's
`meta/info.json`; the schema `local-vla-inference/embodiment_g1_14d.py`
already targets). `ingest_episodes.py` reads that data out of LeRobot format
and re-emits it as Rerun recordings, so every later stage (catalog, export)
runs on genuine G1 data, not synthetic placeholders.

## Pipeline

### 1. Setup / calibrate
Not duplicated here — already covered by the root `install.sh`,
`diagnose.sh`, `killswitch.sh`, and `scripted-behavior/diag_*.py`. G1 joints
report absolute encoder positions over DDS, so there's no SO-101-style
"match middle pose, sweep joint range" calibration step needed.

### 2. Collect → Rerun recordings (`ingest_episodes.py`)

```bash
python data-pipeline/ingest_episodes.py --episodes 0-2
python data-pipeline/ingest_episodes.py --episodes 0-2 --display   # + live viewer while converting
```

Downloads the requested episodes of `ajkoder/g1_final_cleaned` from the Hub
(needs network + Hub auth if the dataset is private) and writes one `.rrd`
per episode to `recordings/ajkoder__g1_final_cleaned/episode_NNNN.rrd`, using
the same `local-vla-inference/telemetry.Telemetry` class the live robot
pipeline uses — camera frames, state/action scalars (DDS-space joint names),
and a 3D arm skeleton (`g1_arm_fk.py`), all indexed on the dataset's own
`frame_index`/`timestamp`, not wall-clock. `--display` also streams each
episode into a live Rerun viewer as it converts (each episode reconnects to
the same window, shown as its own selectable recording).

### 3. Catalog / query / curate (`catalog.py`), and viewing everything at once (`view_recordings.py`)

```bash
python data-pipeline/catalog.py recordings/ajkoder__g1_final_cleaned
python data-pipeline/catalog.py recordings/ajkoder__g1_final_cleaned --task-contains ball

# Open every episode in the directory together in one Rerun viewer:
python data-pipeline/view_recordings.py recordings/ajkoder__g1_final_cleaned
```

`catalog.py` uses Rerun's **dataframe Query API** (`rr.dataframe
.load_recording` → `.view()` → `.select()`) to scan every `episode_*.rrd` in
a directory and build a catalog table (episode, task, frame count, duration)
— no separate database. "Tags" are whatever's logged into the recording
itself (currently the task string); curation = filtering the catalog by
that. `view_recordings.py` prints that same catalog table, then shells out to
the `rerun` CLI with every matched `.rrd` as an argument — it natively loads
multiple files as separate, switchable recordings in one viewer window, so
you can browse the whole ingested batch without opening files one at a time.
No `rerun`/CLI installed? Drag any individual `.rrd` into
[app.rerun.io](https://app.rerun.io) instead.

### 4. Export to LeRobot v3 (`export_lerobot.py`)

```bash
python data-pipeline/export_lerobot.py recordings/ajkoder__g1_final_cleaned \
    --out-repo-id local/g1_from_rerun --root ./exported_dataset
```

Reads selected episode recordings back out via the Query API and rebuilds a
fresh `LeRobotDataset` (`add_frame`/`save_episode`/`finalize`) — the
Rerun → LeRobot v3 half of the round trip. **State/action only, no video** —
see the module docstring for why re-exporting camera video wasn't attempted
(the image-column extraction path needs verification against the actually
installed rerun-sdk version that this environment couldn't run).

### 5. Replay / deploy
Already built, not duplicated: `local-vla-inference/replay_arms.py` /
`scripted-behavior/arm_replay.py` (trajectory replay) and
`local-vla-inference/main.py` (this exact `ajkoder/g1-pickup-ball-act`
checkpoint, deployed and running). A `.rrd`-native replay path would read
`episode/action/*` back out via the same Query API pattern as
`export_lerobot.py` and feed it to `G1Arms.send_arm_positions()` — not built
here since JSON replay already exists and works; extend `arm_replay.py` to
accept an `.rrd` source instead of `.json` if that's wanted later.

## Verification status

**Not run end to end in this environment** — no `torch`/`lerobot[dataset]`/
`rerun-sdk`/network dataset download available in this sandbox. Every API
call here (`LeRobotDataset` constructor/`add_frame`/`save_episode`/`create`,
`rr.dataframe.load_recording`/`.view()`/`.select()`) was checked against the
real vendored `thirdparty/lerobot` source and Rerun's documented dataframe
API, not guessed — but this needs a real run (`uv sync --extra dataset
--extra viz` in this package, then the three scripts in order) before
demoing to judges.
