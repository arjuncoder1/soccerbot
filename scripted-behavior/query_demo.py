"""Query a recorded ``soccerbot_demo`` run with Rerun's dataframe Query API --
never opens the Viewer, pulls the recording back out as pandas DataFrames and
answers concrete questions about the run:

  1. How long did each stage (pickup / turn_180 / avoid / throw) take?
     (from the "stage" TextLog markers ``soccerbot.main.run_demo`` logs at
     each transition -- one shared ``soccerbot_demo`` recording when run via
     ``python -m soccerbot --record-path ...``.)
  2. How accurate/stable was the turn_180 stage?
     (final ``turn/accumulated_deg`` vs. the 180 deg target -- the EXACT
     metric ``turn_180.py``'s own stopping condition uses, not a re-derived
     approximation -- plus peak |roll|/|pitch| deviation from the IMU)
  3. How many people did the avoid stage see, and how close?
     (from ``avoid/detect/n_people`` / ``avoid/detect/nearest_m``)

To produce a recording to query, run the orchestrator with ``--record-path``
(headless: add ``--no-display`` too), e.g.:
    python -m soccerbot --backend replay --iface enp5s0 --record-path logs/demo.rrd
    python -m soccerbot --backend replay --iface enp5s0 --record-path logs/demo.rrd --no-display

Uses ``rerun.dataframe``: ``load_recording(path) -> Recording``,
``Recording.view(index=, contents=) -> RecordingView``,
``RecordingView.select() -> pyarrow.RecordBatchReader``. Documented Python
dataframe/query API for the rerun-sdk range this project pins (see
``local-vla-inference/pyproject.toml``'s ``lerobot[viz]`` -> rerun-sdk).

Usage:
    python scripted-behavior/query_demo.py logs/demo.rrd
"""

from __future__ import annotations

import argparse
import math
import sys

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rrd_path", help="Path to a recorded soccerbot_demo .rrd file.")
    return p.parse_args(argv)


def _load(rrd_path: str, contents: str) -> pd.DataFrame:
    import rerun as rr

    recording = rr.dataframe.load_recording(rrd_path)
    view = recording.view(index="log_time", contents=contents)
    return view.select().read_all().to_pandas()


def _find_column(df: pd.DataFrame, entity_path: str) -> str | None:
    """Column names are ``<entity_path>:<component>`` -- match on the entity-path
    prefix rather than an exact component-name suffix (the one part of the
    naming convention stable across dataframe API versions)."""
    matches = [c for c in df.columns if c == entity_path or c.startswith(entity_path + ":")]
    return matches[0] if matches else None


def stage_timing(rrd_path: str) -> pd.DataFrame:
    """One row per stage-transition marker, with how long until the next one."""
    df = _load(rrd_path, "stage/**")
    col = _find_column(df, "stage")
    if col is None:
        return pd.DataFrame()
    events = df[["log_time", col]].dropna().rename(columns={col: "text"}).reset_index(drop=True)
    # TextLog values come back as a length-1 list/array per row; unwrap to a plain string.
    events["text"] = events["text"].apply(lambda v: v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v)
    events = events.sort_values("log_time").reset_index(drop=True)
    events["duration_s"] = events["log_time"].diff().shift(-1).dt.total_seconds()
    return events


def turn_accuracy(rrd_path: str) -> dict:
    df = _load(rrd_path, "turn/**")
    col = _find_column(df, "turn/accumulated_deg")
    if col is None or df[col].dropna().empty:
        return {}
    series = df[col].dropna()
    final_deg = float(series.iloc[-1])
    result = {
        "final_deg": final_deg,
        "target_deg": 180.0,
        "error_deg": final_deg - 180.0,
        "samples": len(series),
    }
    for axis in ("roll", "pitch", "yaw"):
        acol = _find_column(df, f"turn/imu/{axis}")
        if acol is None:
            continue
        aseries = df[acol].dropna()
        if not aseries.empty:
            result[f"peak_{axis}_deg"] = math.degrees(float(aseries.abs().max()))
    return result


def avoid_summary(rrd_path: str) -> dict:
    df = _load(rrd_path, "avoid/**")
    n_col = _find_column(df, "avoid/detect/n_people")
    dist_col = _find_column(df, "avoid/detect/nearest_m")
    if n_col is None:
        return {}
    n_series = df[n_col].dropna()
    result: dict = {
        "polls": len(n_series),
        "max_people_seen": int(n_series.max()) if len(n_series) else 0,
    }
    if dist_col is not None:
        dist_series = df[dist_col].dropna()
        if not dist_series.empty:
            result["closest_approach_m"] = float(dist_series.min())
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ok = True

    print(f"=== Stage timing: {args.rrd_path} ===")
    timing = stage_timing(args.rrd_path)
    if timing.empty:
        print(
            "No 'stage' markers found -- was this recorded via "
            "`python -m soccerbot ... --record-path`?",
            file=sys.stderr,
        )
        ok = False
    else:
        print(timing[["log_time", "text", "duration_s"]].to_string(index=False))

    print("\n=== Turn-180 accuracy ===")
    accuracy = turn_accuracy(args.rrd_path)
    if not accuracy:
        print("No turn/accumulated_deg data found (only logged during the turn_180 stage).", file=sys.stderr)
        ok = False
    else:
        print(
            f"final={accuracy['final_deg']:.1f} deg  target={accuracy['target_deg']:.1f} deg  "
            f"error={accuracy['error_deg']:+.1f} deg  (n={accuracy['samples']} samples)"
        )
        for axis in ("roll", "pitch", "yaw"):
            key = f"peak_{axis}_deg"
            if key in accuracy:
                print(f"  peak |{axis}| deviation: {accuracy[key]:.2f} deg")

    print("\n=== Avoid-stage detections ===")
    avoid = avoid_summary(args.rrd_path)
    if not avoid:
        print("No avoid/detect/* data found (only logged during the avoid stage).", file=sys.stderr)
        ok = False
    else:
        line = f"polls={avoid['polls']}  max_people_seen={avoid['max_people_seen']}"
        if "closest_approach_m" in avoid:
            line += f"  closest_approach={avoid['closest_approach_m']:.2f}m"
        print(line)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
