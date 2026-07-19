"""Mock HumanDetector for Isaac Sim — always reports clear after a delay.

Drop-in replacement for ``realsense-human-detection/realsense_human_avoid.py:HumanDetector``.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger("sim.sim_detect")

# How many seconds the mock detector reports a person before clearing.
MOCK_PERSON_DURATION_S = 2.0


class HumanDetector:
    """Mock detector that simulates a person clearing the zone."""

    def __init__(self, person_duration_s: float = MOCK_PERSON_DURATION_S):
        self._person_duration_s = person_duration_s
        self._start_time: float | None = None

    def __enter__(self):
        self._start_time = time.monotonic()
        logger.info("MockHumanDetector: opened (person clears after %.1fs)", self._person_duration_s)
        return self

    def __exit__(self, *exc):
        logger.info("MockHumanDetector: closed")
        return False

    def poll_nearest_person(self, within_m: float = 2.0) -> float | None:
        """Return distance to nearest person, or None if clear.

        Simulates a person at 0.5 m for the configured duration, then None.
        """
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        if elapsed < self._person_duration_s:
            return 0.5  # simulated person at 0.5 m
        return None
