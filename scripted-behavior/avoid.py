"""Stage 3: shuffle-avoid FSM.

Repeatedly sidestep 2 left / 2 right until the head-camera human detector
reports no person within ``AVOID_CLEAR_DISTANCE_M``. Bails out after
``AVOID_MAX_CYCLES`` cycles so the demo never hangs on a stationary
spectator.

The lateral motion primitive lives in ``sidestep.py``; the human detector
core lives in ``human_detector_teleimager.py`` (``HumanDetector`` class,
teleimager head-camera + YOLO, distance estimated from bbox height).
"""

from __future__ import annotations

import logging
import time

from config import OrchestratorConfig
from sidestep import sidestep

logger = logging.getLogger("scripted_behavior.avoid")

# Trigger zone: person closer than this counts as "in the way".
AVOID_CLEAR_DISTANCE_M = 4.0
# Give-up limit; throw stage runs regardless.
AVOID_MAX_CYCLES = 8
# How many detector polls to accept as "clear" before we believe it.
AVOID_CLEAR_CONFIRM_POLLS = 2
# Max seconds we wait per human-detect check before assuming clear.
AVOID_CHECK_TIMEOUT_S = 2.0


def _import_human_detector():
    """Import ``HumanDetector`` from the teleimager module."""
    from human_detector_teleimager import HumanDetector  # type: ignore[import-not-found]
    return HumanDetector


def _confirm_clear(detector, distance_m: float) -> bool:
    """Return True iff we see ``AVOID_CLEAR_CONFIRM_POLLS`` consecutive
    frames with no person within ``distance_m``, or the check times out
    (treated as clear rather than block the demo forever)."""
    deadline = time.monotonic() + AVOID_CHECK_TIMEOUT_S
    clear_streak = 0
    polls = 0
    while time.monotonic() < deadline:
        snap = detector.poll_snapshot()
        if snap is None:
            continue
        polls += 1
        if not snap.detections:
            clear_streak += 1
            logger.info("poll %d: no person (streak %d/%d)",
                        polls, clear_streak, AVOID_CLEAR_CONFIRM_POLLS)
            if clear_streak >= AVOID_CLEAR_CONFIRM_POLLS:
                return True
            continue
        nearest = min(d.distance_m for d in snap.detections)
        if nearest > distance_m:
            clear_streak += 1
            logger.info(
                "poll %d: %d person(s) but nearest %.2f m > %.1f m gate (streak %d/%d)",
                polls, len(snap.detections), nearest, distance_m,
                clear_streak, AVOID_CLEAR_CONFIRM_POLLS,
            )
            if clear_streak >= AVOID_CLEAR_CONFIRM_POLLS:
                return True
            continue
        logger.info("poll %d: person at %.2f m -- BLOCKED", polls, nearest)
        return False
    logger.warning(
        "Detector timeout after %.1fs with no positive detections -- treating as clear",
        AVOID_CHECK_TIMEOUT_S,
    )
    return True


def avoid_humans(cfg: OrchestratorConfig) -> None:
    HumanDetector = _import_human_detector()
    with HumanDetector(host=cfg.teleimager_host) as detector:
        for cycle in range(1, AVOID_MAX_CYCLES + 1):
            if _confirm_clear(detector, AVOID_CLEAR_DISTANCE_M):
                logger.info("Avoid stage clear on cycle %d", cycle)
                return
            logger.info(
                "Avoid cycle %d/%d: person still within %.1f m -- shuffling",
                cycle,
                AVOID_MAX_CYCLES,
                AVOID_CLEAR_DISTANCE_M,
            )
            sidestep("left", 2, cfg)
            sidestep("right", 2, cfg)
    logger.warning(
        "Avoid stage giving up after %d cycles; proceeding to throw anyway",
        AVOID_MAX_CYCLES,
    )


# ---------------------------------------------------------------------------
# Standalone entry point.
#   python3 avoid.py --iface eth0                    (full FSM)
#   python3 avoid.py --detect-only 10 --iface eth0   (poll detector, no motion)
# ---------------------------------------------------------------------------


def _detect_only(cfg: OrchestratorConfig, seconds: float) -> int:
    HumanDetector = _import_human_detector()
    t0 = time.monotonic()
    with HumanDetector(host=cfg.teleimager_host) as detector:
        while time.monotonic() - t0 < seconds:
            snap = detector.poll_snapshot()
            if snap is None:
                continue
            if not snap.detections:
                logger.info("no person in frame")
                continue
            nearest = min(d.distance_m for d in snap.detections)
            trigger = " [TRIGGER]" if nearest <= AVOID_CLEAR_DISTANCE_M else ""
            logger.info(
                "%d person(s), nearest %.2f m (avoid threshold %.1f m)%s",
                len(snap.detections), nearest, AVOID_CLEAR_DISTANCE_M, trigger,
            )
    return 0


def _cli() -> int:
    import argparse
    import logging as _logging

    p = argparse.ArgumentParser(description="Standalone shuffle-avoid stage test.")
    p.add_argument("--iface", default=None, help="DDS network interface (e.g. eth0).")
    p.add_argument(
        "--teleimager-host",
        default="192.168.123.164",
        help="Robot IP running teleimager (default: %(default)s).",
    )
    p.add_argument(
        "--detect-only",
        type=float,
        metavar="SECS",
        default=None,
        help="Only poll the teleimager human detector for SECS seconds; no motion.",
    )
    args = p.parse_args()

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = OrchestratorConfig(iface=args.iface, teleimager_host=args.teleimager_host)
    try:
        if args.detect_only is not None:
            return _detect_only(cfg, args.detect_only)
        avoid_humans(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
