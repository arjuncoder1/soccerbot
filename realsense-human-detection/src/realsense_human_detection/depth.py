"""Depth helpers used by one-frame detection."""

from __future__ import annotations

import numpy as np


def median_depth_m_from_array(
    depth_m: np.ndarray,
    cx: int,
    cy: int,
    *,
    patch: int = 5,
) -> float:
    """Median depth (meters) in a patch around (cx, cy); 0 if no valid samples.

    ``depth_m`` is HxW float meters (invalid / missing pixels should be 0).
    """
    h, w = depth_m.shape[:2]
    y0 = max(0, cy - patch)
    y1 = min(h, cy + patch + 1)
    x0 = max(0, cx - patch)
    x1 = min(w, cx + patch + 1)
    patch_vals = depth_m[y0:y1, x0:x1]
    samples = patch_vals[patch_vals > 0]
    if samples.size == 0:
        return 0.0
    return float(np.median(samples))


def median_depth_m_from_rs(
    depth_frame,
    cx: int,
    cy: int,
    *,
    frame_w: int,
    frame_h: int,
    patch: int = 5,
) -> float:
    """Median depth from a pyrealsense2 depth_frame (get_distance API)."""
    samples: list[float] = []
    for dy in range(-patch, patch + 1):
        for dx in range(-patch, patch + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < frame_w and 0 <= y < frame_h:
                d = float(depth_frame.get_distance(x, y))
                if d > 0:
                    samples.append(d)
    if not samples:
        return 0.0
    return float(np.median(samples))
