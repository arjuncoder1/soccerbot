"""Log a full G1 lowstate snapshot to CSV. State-only (does NOT command).

Subscribes to ``rt/lowstate`` via the validated ``G1Arms`` adapter and writes
one row per sample (default 50 Hz) with:

  * ``mode_machine``  -- current robot mode (balancer state indicator)
  * ``imu.{roll,pitch,yaw,gyro_{x,y,z}}``
  * ``kLeft*/kRight*  q, dq, tau`` for all 14 arm joints
  * ``kLeft*/kRight*  q, dq, tau`` for all 12 leg joints
  * ``kWaistYaw       q, dq, tau``
  * ``t`` -- monotonic seconds since first sample.

Usage:

    cd ~/soccerbot/scripted-behavior
    python diag_log.py --iface enp5s0 --duration 5 --hz 50 \
        --out logs/before_replay.csv

Run once before the pickup replay ("baseline") and once during / after to see
what changed. Diff two CSVs with pandas / a spreadsheet.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

from config import REPO_ROOT
from dds import ensure_dds

logger = logging.getLogger("scripted_behavior.diag_log")


def _import_g1_arms():
    sys.path.insert(0, str(REPO_ROOT / "local-vla-inference"))
    try:
        from g1_arms import G1Arms  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return G1Arms


def log_snapshots(out_path: Path, iface: str | None, duration_s: float, hz: float) -> None:
    G1Arms = _import_g1_arms()
    ensure_dds(iface)

    arms = G1Arms(kp=0.0, kd=0.0)  # gains irrelevant, state-only
    arms.connect(state_only=True)

    # First snapshot -> determine column order.
    first = arms.get_full_snapshot()
    fieldnames = ["t"] + list(first.keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Logging %d fields @ %.1f Hz for %.1fs -> %s",
        len(fieldnames) - 1, hz, duration_s, out_path,
    )

    dt = 1.0 / hz
    t0 = time.monotonic()
    deadline = t0 + duration_s
    n = 0

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        # Write the first snapshot we already have.
        row0 = {"t": 0.0, **first}
        w.writerow(row0)
        n += 1
        next_t = t0 + dt

        while time.monotonic() < deadline:
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            snap = arms.get_full_snapshot()
            row = {"t": time.monotonic() - t0, **snap}
            w.writerow(row)
            n += 1
            next_t += dt

    logger.info("Wrote %d rows -> %s", n, out_path)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Log a full G1 lowstate snapshot to CSV.")
    p.add_argument("--iface", default=None, help="DDS interface (e.g. enp5s0).")
    p.add_argument("--duration", type=float, default=5.0, help="Seconds to record.")
    p.add_argument("--hz", type=float, default=50.0, help="Sample rate.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("logs") / f"lowstate_{int(time.time())}.csv",
        help="CSV output path (default: logs/lowstate_<epoch>.csv).",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        log_snapshots(args.out, args.iface, args.duration, args.hz)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
