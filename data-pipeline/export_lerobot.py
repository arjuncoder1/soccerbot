"""Track 3 stage 4: export selected (catalog-curated) episode recordings back
to a fresh local LeRobot v3 dataset.

Reads each episode_*.rrd via Rerun's dataframe Query API (rr.dataframe
.load_recording -> .view() -> .select()) and re-assembles it frame-by-frame
into a new ``LeRobotDataset`` using the same 14-D G1 layout
(local-vla-inference/embodiment_g1_14d.py) the source data already used --
this closes the loop (LeRobot v3 -> Rerun -> LeRobot v3) so the whole
pipeline is demonstrated end to end on real data, even though this run
started from an existing LeRobot dataset (ingest_episodes.py) rather than a
live teleop session. A real teleop-based recorder would produce the same
.rrd shape and plug into this exporter unchanged.

Only re-exports the STATE/ACTION scalars logged by ingest_episodes.py's
Telemetry.log_scalars() calls -- NOT the camera video, since reconstructing
video frames from Rerun-logged images via the dataframe API needs more
verification than this script attempts (image columns come back as
encoded/raw blobs, not decoded arrays, and re-encoding to video adds a real
dependency chain). Exported episodes are state/action-only LeRobot datasets;
extend `_read_episode_frames()` to add images once that path is verified
against the real installed rerun-sdk version.

Usage:
    python data-pipeline/export_lerobot.py recordings/ajkoder__g1_final_cleaned \\
        --out-repo-id local/g1_from_rerun --root ./exported_dataset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_VLA_DIR = _REPO_ROOT / "local-vla-inference"


def _import_layout():
    path = str(_LOCAL_VLA_DIR)
    sys.path.insert(0, path)
    try:
        import embodiment_g1_14d as layout
    finally:
        sys.path.remove(path)
    return layout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings_dir", help="Directory of episode_*.rrd files (see catalog.py to select which).")
    p.add_argument("--out-repo-id", default="local/g1_from_rerun", help="repo_id for the new LeRobot dataset.")
    p.add_argument("--root", default=None, help="Local directory to write the new dataset into.")
    p.add_argument("--episodes", default=None, help="Optional subset, e.g. '0,1,2' matching episode_%%04d.rrd.")
    p.add_argument("--fps", type=int, default=30, help="Dataset fps (must match the source recordings).")
    return p.parse_args(argv)


def _find_column(df: pd.DataFrame, entity_path: str) -> str | None:
    matches = [c for c in df.columns if c == entity_path or c.startswith(entity_path + ":")]
    return matches[0] if matches else None


def _text_field(text: str, label: str) -> str | None:
    if not text or label not in text:
        return None
    tail = text.split(label, 1)[1].lstrip()
    return tail.split("\n", 1)[0].strip()


def _read_episode_frames(rrd_path: Path, layout) -> tuple[str, list[dict]]:
    """Query one episode .rrd back into a list of {feature_name: value} frame
    dicts (dataset-space names, e.g. 'left_arm_shoulder_pitch'), in frame order,
    plus the episode's task string."""
    import rerun as rr

    recording = rr.dataframe.load_recording(str(rrd_path))
    view = recording.view(index="log_time", contents="episode/**")
    df = view.select().read_all().to_pandas()

    info_view = recording.view(index="log_time", contents="info/**")
    info_df = info_view.select().read_all().to_pandas()
    info_col = _find_column(info_df, "info")
    task = "unknown"
    if info_col is not None:
        series = info_df[info_col].dropna()
        if not series.empty:
            v = series.iloc[-1]
            text = v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v
            task = _text_field(text, "**Task:**") or task

    # DDS-space column names (episode/state/kLeftShoulderPitch, etc.) -> dataset FEATURE_NAMES.
    dds_short_to_feature = dict(zip((j.removesuffix(".q") for j in layout.ARM_JOINTS), layout.FEATURE_NAMES))
    state_cols = {short: _find_column(df, f"episode/state/{short}") for short in dds_short_to_feature}
    action_cols = {short: _find_column(df, f"episode/action/{short}") for short in dds_short_to_feature}
    if not any(state_cols.values()) or not any(action_cols.values()):
        raise ValueError(f"{rrd_path}: no episode/state or episode/action columns found")

    frames: list[dict] = []
    for _, row in df.iterrows():
        state_vals = {}
        action_vals = {}
        ok = True
        for short, feat in dds_short_to_feature.items():
            scol, acol = state_cols[short], action_cols[short]
            sv = row.get(scol) if scol else None
            av = row.get(acol) if acol else None
            if sv is None or av is None or pd.isna(sv) or pd.isna(av):
                ok = False
                break
            state_vals[feat] = float(sv)
            action_vals[feat] = float(av)
        if ok:
            frames.append({"state": state_vals, "action": action_vals})
    return task, frames


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.is_dir():
        logger.error("Not a directory: %s", recordings_dir)
        return 1

    rrd_paths = sorted(recordings_dir.glob("*.rrd"))
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        rrd_paths = [p for p in rrd_paths if any(f"episode_{i:04d}" in p.stem for i in wanted)]
    if not rrd_paths:
        logger.error("No matching episode_*.rrd files in %s", recordings_dir)
        return 1

    layout = _import_layout()

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (layout.STATE_ACTION_DIM,), "names": list(layout.FEATURE_NAMES)},
        "action": {"dtype": "float32", "shape": (layout.STATE_ACTION_DIM,), "names": list(layout.FEATURE_NAMES)},
    }
    out_dataset = LeRobotDataset.create(
        repo_id=args.out_repo_id,
        fps=args.fps,
        features=features,
        root=args.root,
        robot_type="g1",
        use_videos=False,  # camera re-export not attempted here -- see module docstring
    )

    total_frames = 0
    for rrd_path in rrd_paths:
        task, frames = _read_episode_frames(rrd_path, layout)
        if not frames:
            logger.warning("%s: 0 usable frames, skipping", rrd_path)
            continue
        for fr in frames:
            out_dataset.add_frame(
                {
                    "observation.state": np.array([fr["state"][n] for n in layout.FEATURE_NAMES], dtype=np.float32),
                    "action": np.array([fr["action"][n] for n in layout.FEATURE_NAMES], dtype=np.float32),
                    "task": task,
                }
            )
        out_dataset.save_episode()
        logger.info("%s ('%s'): exported %d frames", rrd_path.name, task, len(frames))
        total_frames += len(frames)

    out_dataset.finalize()
    logger.info(
        "Done. Exported %d episode(s), %d total frames -> repo_id=%s root=%s",
        len(rrd_paths),
        total_frames,
        args.out_repo_id,
        args.root or "(default HF_LEROBOT_HOME)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
