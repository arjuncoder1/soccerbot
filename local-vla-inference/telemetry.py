"""Rerun telemetry helpers for live ACT / scripted G1 control.

Logs teleimager RGB (the existing single-port JPEG stream), optional depth
when a caller already has it (e.g. local RealSense HumanDetector — teleimager
itself publishes color JPEGs only), measured/commanded arm joints, policy
targets, clamp hits, and timing. Safe no-op when Rerun is unavailable or
disabled.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Telemetry:
    """Thin wrapper around the Rerun SDK (optional)."""

    def __init__(self, enabled: bool = True, session_name: str = "soccerbot") -> None:
        self.enabled = enabled
        self.session_name = session_name
        self._rr = None
        self._step = 0

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            from lerobot.utils.visualization_utils import init_rerun

            init_rerun(session_name=self.session_name)
            import rerun as rr

            self._rr = rr
            logger.info("Rerun telemetry started (session=%s)", self.session_name)
        except Exception as exc:  # noqa: BLE001 -- viz is optional on the robot
            logger.warning("Rerun unavailable (%s); continuing without visualization", exc)
            self.enabled = False
            self._rr = None

    def stop(self) -> None:
        if not self.enabled or self._rr is None:
            return
        try:
            from lerobot.utils.visualization_utils import shutdown_rerun

            shutdown_rerun()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Rerun shutdown failed: %s", exc)
        self._rr = None

    def set_time(self, step: int | None = None, seconds: float | None = None) -> None:
        if self._rr is None:
            return
        if step is not None:
            self._step = int(step)
            self._rr.set_time("step", sequence=self._step)
        if seconds is not None:
            self._rr.set_time("time", timestamp=float(seconds))

    def log_image(self, path: str, image: np.ndarray, *, compress: bool = True) -> None:
        if self._rr is None or image is None:
            return
        arr = np.asarray(image)
        if arr.ndim == 2:
            # Depth / mono: log as depth image when uint16, else grayscale.
            if arr.dtype == np.uint16:
                self._rr.log(path, self._rr.DepthImage(arr))
            else:
                self._rr.log(path, self._rr.Image(arr))
            return
        if arr.ndim == 3 and arr.shape[-1] == 3:
            if compress:
                try:
                    self._rr.log(path, self._rr.Image(arr).compress(jpeg_quality=75))
                    return
                except Exception:  # noqa: BLE001
                    pass
            self._rr.log(path, self._rr.Image(arr))
            return
        logger.debug("Skipping unsupported image shape %s at %s", arr.shape, path)

    def log_scalars(self, path: str, values: dict[str, float]) -> None:
        if self._rr is None:
            return
        for key, value in values.items():
            try:
                self._rr.log(f"{path}/{key}", self._rr.Scalars(float(value)))
            except Exception:  # noqa: BLE001
                continue

    def log_step(
        self,
        *,
        step: int,
        elapsed_s: float,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
        measured: dict[str, float] | None = None,
        target: dict[str, float] | None = None,
        commanded: dict[str, float] | None = None,
        extras: dict[str, float] | None = None,
        stage: str = "pickup",
    ) -> None:
        if self._rr is None:
            return
        self.set_time(step=step, seconds=elapsed_s)
        if rgb is not None:
            self.log_image(f"{stage}/camera/rgb", rgb)
        if depth is not None:
            self.log_image(f"{stage}/camera/depth", depth, compress=False)
        if measured:
            self.log_scalars(f"{stage}/arm/measured", _strip_q_suffix(measured))
        if target:
            self.log_scalars(f"{stage}/arm/target", _strip_q_suffix(target))
        if commanded:
            self.log_scalars(f"{stage}/arm/commanded", _strip_q_suffix(commanded))
        if extras:
            self.log_scalars(f"{stage}/stats", extras)

    def log_detection(
        self,
        *,
        step: int,
        elapsed_s: float,
        rgb: np.ndarray | None,
        nearest_m: float | None,
        n_people: int,
        stage: str = "avoid",
    ) -> None:
        if self._rr is None:
            return
        self.set_time(step=step, seconds=elapsed_s)
        if rgb is not None:
            self.log_image(f"{stage}/camera/rgb", rgb)
        extras: dict[str, float] = {"n_people": float(n_people)}
        if nearest_m is not None:
            extras["nearest_m"] = float(nearest_m)
        self.log_scalars(f"{stage}/detect", extras)


def _strip_q_suffix(joints: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in joints.items():
        name = key.removesuffix(".q")
        out[name] = float(value)
    return out


def apply_slew_clamp(
    cmd_q: dict[str, float],
    target_q: dict[str, float],
    keys: list[str],
    clamp_rad: float,
) -> tuple[dict[str, float], int]:
    """Move ``cmd_q`` toward ``target_q`` by at most ``clamp_rad`` per joint.

    Returns the updated command dict and the number of joints that hit the clamp.
    """
    if clamp_rad <= 0:
        updated = {k: float(target_q[k]) for k in keys}
        return updated, 0

    hits = 0
    updated = dict(cmd_q)
    for key in keys:
        delta = float(target_q[key]) - float(updated[key])
        if abs(delta) > clamp_rad:
            hits += 1
        updated[key] = float(updated[key] + float(np.clip(delta, -clamp_rad, clamp_rad)))
    return updated, hits


def namespace_to_observation(
    measured: dict[str, float],
    rgb: np.ndarray | None,
    depth: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a LeRobot-style observation dict for ``log_rerun_data`` fallbacks."""
    obs: dict[str, Any] = dict(measured)
    if rgb is not None:
        obs["images.front"] = rgb
    if depth is not None:
        obs["images.depth"] = depth
    return obs
