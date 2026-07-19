"""Rerun telemetry helpers for live ACT / scripted G1 control.

Logs teleimager RGB (the existing single-port JPEG stream), optional depth
when a caller already has it (e.g. local RealSense HumanDetector — teleimager
itself publishes color JPEGs only), measured/commanded arm joints, policy
targets, clamp hits, and timing. Safe no-op when Rerun is unavailable or
disabled.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# g1_arm_fk.py (real-URDF forward kinematics, used by scripted-behavior/throw.py
# to verify its own motion) lives in the sibling scripted-behavior package.
# Reused here purely for joint *geometry* -- log_arm_skeleton never touches
# anything throw.py-specific.
_SCRIPTED_BEHAVIOR_DIR = str(Path(__file__).resolve().parent.parent / "scripted-behavior")

# G1Arms pose keys ("kLeftShoulderPitch.q", ...) -> g1_arm_fk's short per-joint
# names (same short name for both LEFT_ARM/RIGHT_ARM).
_JOINT_SUFFIX_TO_FK_NAME = {
    "ShoulderPitch": "shoulder_pitch",
    "ShoulderRoll": "shoulder_roll",
    "ShoulderYaw": "shoulder_yaw",
    "Elbow": "elbow",
    "WristRoll": "wrist_roll",
    "WristPitch": "wrist_pitch",
    "WristYaw": "wrist_yaw",
}

_INFO_MARKDOWN = """\
# Soccerbot demo

**Hero view (left):** live 3D arm skeleton, forward-kinematics from
`scripted-behavior/g1_arm_fk.py` (real G1 29-DoF URDF geometry).

**Top right:** camera + detection views, one per stage.
**Bottom right:** arm joints (measured/target/commanded) and stage log.

Offline analysis (no Viewer) of a recorded run: `scripted-behavior/query_demo.py`
uses Rerun's dataframe Query API instead.
"""


def _import_g1_arm_fk():
    sys.path.insert(0, _SCRIPTED_BEHAVIOR_DIR)
    try:
        import g1_arm_fk  # type: ignore[import-not-found]
    finally:
        sys.path.remove(_SCRIPTED_BEHAVIOR_DIR)
    return g1_arm_fk


def _to_fk_angles(pose: dict[str, float], side: str) -> dict[str, float]:
    prefix = "kLeft" if side == "left" else "kRight"
    out: dict[str, float] = {}
    for suffix, fk_name in _JOINT_SUFFIX_TO_FK_NAME.items():
        key = f"{prefix}{suffix}.q"
        if key in pose:
            out[fk_name] = pose[key]
    return out


def _arm_chain_points(fk, joints: list, angles: dict[str, float], hand_offset) -> list:
    """Same math as ``g1_arm_fk.arm_fk``, but returns every intermediate joint
    position (not just the final hand position) -- for drawing a skeleton, not
    just checking where the hand ends up. Uses only g1_arm_fk's public helpers."""
    axis_fn = {"x": fk.rot_x, "y": fk.rot_y, "z": fk.rot_z}
    pos = (0.0, 0.0, 0.0)
    orient = fk.IDENTITY
    points = [pos]
    for j in joints:
        pos = fk.vec_add(pos, fk.mat_vec(orient, j.origin))
        if j.pre_roll:
            orient = fk.mat_mul(orient, fk.rot_x(j.pre_roll))
        angle = angles.get(j.name, 0.0)
        orient = fk.mat_mul(orient, axis_fn[j.axis](angle))
        points.append(pos)
    pos = fk.vec_add(pos, fk.mat_vec(orient, hand_offset))
    points.append(pos)
    return points


class Telemetry:
    """Thin wrapper around the Rerun SDK (optional)."""

    def __init__(
        self,
        enabled: bool = True,
        session_name: str = "soccerbot",
        record_path: str | None = None,
        display: bool = True,
    ) -> None:
        self.enabled = enabled
        self.session_name = session_name
        # record_path: also/instead write an .rrd file for query_demo.py.
        # display=False + record_path set: headless, file-only (no viewer window).
        self.record_path = record_path
        self.display = display
        self._rr = None
        self._step = 0

    def start(self) -> None:
        if not self.enabled:
            return
        try:
            import rerun as rr

            if self.record_path:
                # Multi-sink API (rr.set_sinks, Rerun >=0.24) instead of
                # lerobot's init_rerun helper, which only supports one sink at
                # a time (spawn XOR connect XOR save) -- lets --record-path
                # combine with the live viewer instead of replacing it.
                rr.init(self.session_name)
                sinks: list = [rr.FileSink(self.record_path)]
                if self.display:
                    rr.spawn()
                    sinks.insert(0, rr.GrpcSink())
                rr.set_sinks(*sinks)
            else:
                from lerobot.utils.visualization_utils import init_rerun

                init_rerun(session_name=self.session_name)

            self._rr = rr
            logger.info(
                "Rerun telemetry started (session=%s, record_path=%s, display=%s)",
                self.session_name,
                self.record_path,
                self.display,
            )
            self._send_blueprint()
        except Exception as exc:  # noqa: BLE001 -- viz is optional on the robot
            logger.warning("Rerun unavailable (%s); continuing without visualization", exc)
            self.enabled = False
            self._rr = None

    def _send_blueprint(self) -> None:
        """Purpose-built panel layout (3D skeleton hero view + cameras + arm
        joints + stage log), replacing Rerun's flat default auto-layout. Best
        effort: falls back to the default layout on any API mismatch rather
        than blocking telemetry startup."""
        if self._rr is None:
            return
        try:
            import rerun.blueprint as rrb

            rr = self._rr
            hero_3d = rrb.Spatial3DView(name="G1 arm skeleton (g1_arm_fk.py)", origin="g1", contents=["g1/**"])
            cameras = rrb.Grid(
                rrb.Spatial2DView(name="pickup camera", origin="pickup/camera"),
                rrb.Spatial2DView(name="avoid camera", origin="avoid/camera"),
                name="Cameras",
            )
            joints = rrb.TimeSeriesView(name="Arm joints (rad)", contents=["pickup/arm/**", "turn/**"])
            stage_log = rrb.TextLogView(name="Stage log", origin="stage")
            info = rrb.TextDocumentView(name="About this run", origin="info")

            blueprint = rrb.Blueprint(
                rrb.Horizontal(
                    hero_3d,
                    rrb.Vertical(cameras, joints, stage_log, info, row_shares=[2, 2, 1, 1]),
                    column_shares=[3, 2],
                ),
                rrb.BlueprintPanel(state="collapsed"),
                rrb.SelectionPanel(state="collapsed"),
                rrb.TimePanel(state="expanded"),
            )
            rr.send_blueprint(blueprint, make_active=True, make_default=True)
            rr.log("info", rr.TextDocument(_INFO_MARKDOWN, media_type=rr.MediaType.MARKDOWN), static=True)
        except Exception:
            logger.warning("Custom Rerun blueprint failed to build; using the default layout.", exc_info=True)

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

    def log_stage(self, name: str) -> None:
        """Log a stage-transition marker (shows up in the Stage log panel)."""
        if self._rr is None:
            return
        try:
            self._rr.log("stage", self._rr.TextLog(name))
        except Exception:  # noqa: BLE001
            logger.debug("log_stage failed for %r", name, exc_info=True)

    def log_arm_skeleton(self, pose: dict[str, float]) -> None:
        """Log both arms as a 3D skeleton (forward kinematics from
        ``g1_arm_fk.py``, real G1 URDF geometry) under ``g1/*``. ``pose`` is a
        flat ``{"kLeftShoulderPitch.q": ..., ...}`` dict -- the same shape
        ``G1Arms.get_arm_positions()``/``get_full_snapshot()`` already return."""
        if self._rr is None:
            return
        try:
            fk = _import_g1_arm_fk()
        except Exception:  # noqa: BLE001
            logger.debug("g1_arm_fk unavailable; skipping skeleton log", exc_info=True)
            return
        for side, joints, hand_offset, color in (
            ("left", fk.LEFT_ARM, fk.HAND_OFFSET_LEFT, (100, 136, 234)),
            ("right", fk.RIGHT_ARM, fk.HAND_OFFSET_RIGHT, (224, 96, 58)),
        ):
            angles = _to_fk_angles(pose, side)
            if not angles:
                continue
            points = _arm_chain_points(fk, joints, angles, hand_offset)
            self._rr.log(f"g1/{side}_arm/skeleton", self._rr.LineStrips3D([points], colors=[color]))
            self._rr.log(f"g1/{side}_arm/joints", self._rr.Points3D(points, colors=[color], radii=0.02))

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
            self.log_arm_skeleton(measured)
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
