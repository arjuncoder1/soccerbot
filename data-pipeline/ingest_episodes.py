"""Track 3 stage 2 ("collect teleoperated episodes into Rerun recordings"),
using REAL recorded G1 data instead of a live teleop session.

The SO-101 reference pipeline logs teleop sessions directly to .rrd as they
happen. This repo has no G1 teleop rig (see scripted-behavior/README's "Not
yet built" -- there's no leader/follower arm for a full-size humanoid the way
SO-101 has cheap paired servos), so a live capture of this stage isn't
possible here. What IS available is the real thing that stage exists to
produce: the actual teleoperated episodes the deployed ACT checkpoint
(``ajkoder/g1-pickup-ball-act``) was trained on, already sitting in LeRobot
v3 format on the Hub as ``ajkoder/g1_final_cleaned`` (121 episodes, 74371
frames, single ``color_0`` camera @ 720x1280, 14-D arm state/action -- see
that dataset's meta/info.json; the exact schema
``local-vla-inference/embodiment_g1_14d.py`` already targets).

This script is the honest substitute for live capture: read those real
episodes back out of LeRobot format and re-emit them as Rerun .rrd
recordings, one per episode, so the rest of the pipeline (catalog.py,
export_lerobot.py) operates on genuine G1 data end to end, not synthetic
placeholders.

Usage:
    python data-pipeline/ingest_episodes.py --episodes 0-2
    python data-pipeline/ingest_episodes.py --episodes 0,5,12 --out-dir recordings
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_VLA_DIR = _REPO_ROOT / "local-vla-inference"

DEFAULT_REPO_ID = "ajkoder/g1_final_cleaned"


def _import_local_vla_module(name: str):
    """Reach into the sibling local-vla-inference package for a flat module
    (embodiment_g1_14d.py / telemetry.py) -- same sys.path-reach pattern
    scripted-behavior's stage modules already use to pull in g1_arms."""
    path = str(_LOCAL_VLA_DIR)
    sys.path.insert(0, path)
    try:
        return __import__(name)
    finally:
        sys.path.remove(path)


def parse_episode_spec(spec: str) -> list[int]:
    """'0,1,2' / '0-5' / '0-2,7,10-12' -> sorted list of episode indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="LeRobot v3 dataset repo id on the Hub.")
    p.add_argument("--episodes", default="0-2", help="Episode indices to ingest, e.g. '0-2' or '0,3,7'.")
    p.add_argument("--out-dir", default="recordings", help="Directory to write <repo_id>/episode_<i>.rrd into.")
    p.add_argument("--root", default=None, help="Local LeRobot dataset cache dir (skip Hub download if present).")
    p.add_argument(
        "--display",
        action="store_true",
        help="Also spawn a live Rerun viewer and stream each episode into it as it's "
        "converted, instead of only writing the .rrd file. Episodes keep reconnecting to "
        "the same viewer window (rr.spawn() no-ops if one's already listening), each shown "
        "as its own selectable recording.",
    )
    return p.parse_args(argv)


def _resolve_task(dataset, episode_index: int, default: str = "pick up the ball") -> str:
    """Best-effort task-string lookup: dataset.meta.tasks is a DataFrame indexed
    by task string with a task_index column (see
    thirdparty/lerobot/src/lerobot/datasets/dataset_metadata.py)."""
    try:
        row0 = next(i for i in range(len(dataset)) if int(dataset[i]["episode_index"]) == episode_index)
        task_idx = int(dataset[row0]["task_index"])
        tasks_df = dataset.meta.tasks
        matches = tasks_df.index[tasks_df["task_index"] == task_idx]
        if len(matches):
            return str(matches[0])
    except Exception:  # noqa: BLE001 -- best-effort only, never block ingestion
        logger.debug("Could not resolve task string for episode %d", episode_index, exc_info=True)
    return default


def _frame_to_dds(layout, frame: dict, feature_key: str) -> dict[str, float]:
    values = frame[feature_key]
    values = values.numpy() if hasattr(values, "numpy") else np.asarray(values)
    named = {name: float(values[j]) for j, name in enumerate(layout.FEATURE_NAMES)}
    return layout.to_dds_action(named)


def _frame_image(frame: dict, camera_key: str) -> np.ndarray | None:
    img = frame.get(f"observation.images.{camera_key}")
    if img is None:
        return None
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    # LeRobot decodes video frames as CHW float in [0, 1]; Rerun wants HWC uint8.
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr


def ingest_episode(
    dataset, layout, telemetry_mod, episode_index: int, task: str, out_path: Path, display: bool = False
) -> int:
    """Write one dataset episode's frames to a Rerun recording at out_path.
    Returns the number of frames written."""
    tm = telemetry_mod.Telemetry(
        enabled=True,
        session_name=f"g1_dataset_episode_{episode_index}",
        record_path=str(out_path),
        display=display,
    )
    tm.start()
    n_frames = 0
    try:
        import rerun as rr

        tm.log_stage(f"episode {episode_index}: {task}")
        rr.log(
            "info",
            rr.TextDocument(
                f"# Episode {episode_index}\n\n"
                f"**Source:** `{dataset.repo_id}` (real recorded G1 teleop data -- "
                f"the dataset `ajkoder/g1-pickup-ball-act` was trained on)\n\n"
                f"**Task:** {task}\n\n"
                "Ingested by `data-pipeline/ingest_episodes.py`, not a live capture.",
                media_type=rr.MediaType.MARKDOWN,
            ),
            static=True,
        )

        for row_idx in range(len(dataset)):
            frame = dataset[row_idx]
            if int(frame["episode_index"]) != episode_index:
                continue
            frame_idx = int(frame["frame_index"])
            timestamp = float(frame["timestamp"])
            tm.set_time(step=frame_idx, seconds=timestamp)

            dds_state = _frame_to_dds(layout, frame, "observation.state")
            dds_action = _frame_to_dds(layout, frame, "action")
            tm.log_scalars("episode/state", {k.removesuffix(".q"): v for k, v in dds_state.items()})
            tm.log_scalars("episode/action", {k.removesuffix(".q"): v for k, v in dds_action.items()})
            tm.log_arm_skeleton(dds_state)

            rgb = _frame_image(frame, layout.CAMERA_KEY)
            if rgb is not None:
                tm.log_image("episode/camera/rgb", rgb)

            n_frames += 1
    finally:
        tm.stop()
    return n_frames


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    episode_indices = parse_episode_spec(args.episodes)
    if not episode_indices:
        logger.error("No episode indices parsed from --episodes=%r", args.episodes)
        return 1

    layout = _import_local_vla_module("embodiment_g1_14d")
    telemetry_mod = _import_local_vla_module("telemetry")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info("Loading %s (episodes=%s)...", args.repo_id, episode_indices)
    dataset = LeRobotDataset(args.repo_id, root=args.root, episodes=episode_indices)
    logger.info("Loaded %d frame(s) across %d episode(s)", len(dataset), len(episode_indices))

    out_dir = Path(args.out_dir) / args.repo_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in episode_indices:
        task = _resolve_task(dataset, ep)
        out_path = out_dir / f"episode_{ep:04d}.rrd"
        n = ingest_episode(dataset, layout, telemetry_mod, ep, task, out_path, display=args.display)
        if n == 0:
            logger.warning("Episode %d: 0 frames written (index not in loaded set?)", ep)
        else:
            logger.info("Episode %d ('%s'): %d frames -> %s", ep, task, n, out_path)

    logger.info("Done. Wrote %d episode(s) to %s", len(episode_indices), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
