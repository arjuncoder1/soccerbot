"""Analog directional guidance from a horizontal bearing to a person."""

from __future__ import annotations


def angle_to_clock(angle_deg: float) -> str:
    """Map signed angle (0 = ahead, + = right, - = left) to a clock face."""
    hour = (angle_deg / 30.0) + 12.0
    hour %= 12.0
    if hour == 0:
        hour = 12.0
    h = int(hour)
    m = int(round((hour - h) * 60))
    if m == 60:
        m = 0
        h = 12 if h == 11 else h + 1
    if h == 0:
        h = 12
    return f"{h}:{m:02d}"


def direction_instruction(angle_deg: float) -> str:
    """Graded continuous guidance instead of a binary left/right label."""
    abs_angle = abs(angle_deg)
    clock = angle_to_clock(angle_deg)

    if abs_angle < 3:
        return f"Person dead ahead ({clock}, {angle_deg:+.1f}deg) -> back straight up"

    move_toward = "left" if angle_deg > 0 else "right"

    if abs_angle < 10:
        magnitude = "slightly"
    elif abs_angle < 25:
        magnitude = "moderately"
    elif abs_angle < 45:
        magnitude = "sharply"
    else:
        magnitude = "fully"

    return f"Person at {clock} ({angle_deg:+.1f}deg) -> move {magnitude} {move_toward}"
