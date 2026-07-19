"""Track 3 stage 3: query/inspect/tag/curate recorded episodes -- the "local
catalog" step, built directly on Rerun's dataframe Query API (no separate
database; each episode's own .rrd is queried directly, same technique
already proven in scripted-behavior/query_demo.py).

Scans a directory of episode .rrd files (as written by ingest_episodes.py or
a future live recorder), queries each one's `info` TextDocument + frame
timeline, and prints a catalog table: episode, task, source dataset, frame
count, duration. Supports filtering by task substring, a stand-in for the
reference pipeline's tag-based curation (tags are logged INTO each recording
as static entities, not tracked in an external index file, so an episode's
.rrd stays self-describing).

Usage:
    python data-pipeline/catalog.py recordings/ajkoder__g1_final_cleaned
    python data-pipeline/catalog.py recordings/ajkoder__g1_final_cleaned --task-contains ball
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings_dir", help="Directory containing episode_*.rrd files.")
    p.add_argument("--task-contains", default=None, help="Only list episodes whose task string contains this.")
    p.add_argument("--csv-out", default=None, help="Optional path to write the catalog table as CSV.")
    return p.parse_args(argv)


def _load(rrd_path: Path, contents: str) -> pd.DataFrame:
    import rerun as rr

    recording = rr.dataframe.load_recording(str(rrd_path))
    view = recording.view(index="log_time", contents=contents)
    return view.select().read_all().to_pandas()


def _find_column(df: pd.DataFrame, entity_path: str) -> str | None:
    matches = [c for c in df.columns if c == entity_path or c.startswith(entity_path + ":")]
    return matches[0] if matches else None


def _text_document(df: pd.DataFrame, entity_path: str) -> str | None:
    col = _find_column(df, entity_path)
    if col is None:
        return None
    series = df[col].dropna()
    if series.empty:
        return None
    val = series.iloc[-1]
    return val[0] if hasattr(val, "__len__") and not isinstance(val, str) else val


def inspect_episode(rrd_path: Path) -> dict:
    """Query one episode .rrd and return a catalog row."""
    row: dict = {"path": str(rrd_path), "episode": rrd_path.stem}

    info_df = _load(rrd_path, "info/**")
    info_text = _text_document(info_df, "info") or ""
    row["task"] = _parse_field(info_text, "**Task:**")
    row["source"] = _parse_field(info_text, "**Source:**")

    stage_df = _load(rrd_path, "stage/**")
    stage_col = _find_column(stage_df, "stage")
    row["stage_marker"] = None
    if stage_col is not None:
        vals = stage_df[stage_col].dropna()
        if not vals.empty:
            v = vals.iloc[0]
            row["stage_marker"] = v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v

    ep_df = _load(rrd_path, "episode/**")
    action_col = _find_column(ep_df, "episode/action/kLeftShoulderPitch")
    n_frames = 0
    duration_s = 0.0
    if action_col is not None:
        times = ep_df.loc[ep_df[action_col].notna(), "log_time"]
        n_frames = len(times)
        if n_frames > 1:
            duration_s = (times.max() - times.min()).total_seconds()
    row["n_frames"] = n_frames
    row["duration_s"] = round(duration_s, 2)
    return row


def _parse_field(markdown: str, label: str) -> str | None:
    if label not in markdown:
        return None
    tail = markdown.split(label, 1)[1].lstrip()
    return tail.split("\n", 1)[0].strip()


def build_catalog(recordings_dir: Path) -> pd.DataFrame:
    rows = []
    for rrd_path in sorted(recordings_dir.glob("*.rrd")):
        try:
            rows.append(inspect_episode(rrd_path))
        except Exception as exc:  # noqa: BLE001 -- one bad recording shouldn't kill the catalog scan
            rows.append({"path": str(rrd_path), "episode": rrd_path.stem, "error": str(exc)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.is_dir():
        print(f"Not a directory: {recordings_dir}", file=sys.stderr)
        return 1

    df = build_catalog(recordings_dir)
    if df.empty:
        print(f"No .rrd files found in {recordings_dir}", file=sys.stderr)
        return 1

    if args.task_contains:
        df = df[df.get("task", pd.Series(dtype=str)).fillna("").str.contains(args.task_contains, case=False)]

    print(f"=== Episode catalog: {recordings_dir} ===")
    cols = [c for c in ("episode", "task", "n_frames", "duration_s", "source") if c in df.columns]
    print(df[cols].to_string(index=False))

    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"\nWrote catalog to {args.csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
