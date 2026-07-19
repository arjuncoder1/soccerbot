"""View every recorded episode in a directory at once, in one Rerun viewer.

The `rerun` CLI (installed alongside rerun-sdk) natively accepts multiple
file paths and loads each as its own recording in the same viewer window,
switchable from the recording list in the top-left -- this just globs a
directory and shells out to it, plus prints the catalog.py summary table
first so you know what you're about to open.

Usage:
    python data-pipeline/view_recordings.py recordings/ajkoder__g1_final_cleaned
    python data-pipeline/view_recordings.py recordings/ajkoder__g1_final_cleaned --episodes 0,2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from catalog import build_catalog


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("recordings_dir", help="Directory containing episode_*.rrd files.")
    p.add_argument("--episodes", default=None, help="Optional subset, e.g. '0,1,2' matching episode_%%04d.rrd.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recordings_dir = Path(args.recordings_dir)
    if not recordings_dir.is_dir():
        print(f"Not a directory: {recordings_dir}", file=sys.stderr)
        return 1

    rrd_paths = sorted(recordings_dir.glob("*.rrd"))
    if args.episodes:
        wanted = {int(x) for x in args.episodes.split(",")}
        rrd_paths = [p for p in rrd_paths if any(f"episode_{i:04d}" in p.stem for i in wanted)]
    if not rrd_paths:
        print(f"No matching episode_*.rrd files in {recordings_dir}", file=sys.stderr)
        return 1

    df = build_catalog(recordings_dir)
    cols = [c for c in ("episode", "task", "n_frames", "duration_s", "source") if c in df.columns]
    print(f"=== Opening {len(rrd_paths)} recording(s) from {recordings_dir} ===")
    print(df[cols].to_string(index=False))
    print()

    rerun_bin = shutil.which("rerun")
    if rerun_bin is None:
        print(
            "`rerun` CLI not found on PATH (comes with rerun-sdk -- "
            "`uv sync` in data-pipeline/ first). Falling back to printing paths:",
            file=sys.stderr,
        )
        for p in rrd_paths:
            print(p)
        print("\nOpen manually with `rerun <path>` per file, or drag them into https://app.rerun.io", file=sys.stderr)
        return 1

    cmd = [rerun_bin, *[str(p) for p in rrd_paths]]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
