"""Robot safety helpers: graceful stop / damp / zero-torque.

Ctrl+C during a live demo should **not** leave ``arm_sdk`` engaged forever.
Default interrupt behaviour matches ``remote-vla-inference``:

  1. ``LocoClient.StopMove()`` — zero loco velocity, stay standing
  2. release ``arm_sdk`` (ramp weight → 0) so the balancer takes the arms

The headed killswitch can additionally enter ``Damp`` / ``ZeroTorque`` FSM
states (same actions as the physical pendant ``L2+B`` / ``L2+A``).
"""

from __future__ import annotations

import logging
from typing import Any

from soccerbot.deps import LOCAL_VLA_DIR, _ensure_front

logger = logging.getLogger(__name__)


def _ensure_dds(iface: str | None = None) -> None:
    _ensure_front(LOCAL_VLA_DIR)
    from dds_init import ensure_dds

    ensure_dds(iface)


def init_loco(iface: str | None = None) -> Any:
    _ensure_dds(iface)
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    loco = LocoClient()
    loco.SetTimeout(3.0)
    loco.Init()
    return loco


def stop_loco(loco: Any | None = None, *, iface: str | None = None) -> None:
    """Zero locomotion velocity; robot stays up on the balancer."""
    client = loco
    if client is None:
        try:
            client = init_loco(iface)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not init LocoClient for StopMove: %s", exc)
            return
    try:
        rc = client.StopMove()
        logger.info("LocoClient.StopMove() -> %s", rc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("StopMove failed: %s", exc)


def release_arms(arms: Any | None = None, *, iface: str | None = None) -> None:
    """Hand arms back to the stock balancer (ramp arm_sdk weight → 0).

    If ``arms`` is None (e.g. interrupt during a scripted stage that owns its
    own short-lived ``G1Arms``), open a temporary publisher solely to release.
    """
    owned = False
    if arms is None:
        try:
            _ensure_dds(iface)
            _ensure_front(LOCAL_VLA_DIR)
            from g1_arms import G1Arms

            arms = G1Arms(kp=60.0, kd=1.5)
            arms.connect()
            owned = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not open G1Arms to release arm_sdk: %s", exc)
            return
    try:
        if hasattr(arms, "release"):
            arms.release()
        elif hasattr(arms, "disconnect"):
            arms.disconnect()
            return
        if hasattr(arms, "detach"):
            arms.detach()
        logger.info("arm_sdk released (arms back to balancer)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("arm release failed: %s", exc)
    finally:
        if owned and arms is not None and hasattr(arms, "detach"):
            try:
                arms.detach()
            except Exception:  # noqa: BLE001
                pass


def graceful_reset(
    *,
    arms: Any | None = None,
    loco: Any | None = None,
    iface: str | None = None,
    camera: Any | None = None,
) -> None:
    """Ctrl+C / abort path: stop motion, release arms, disconnect camera.

    Leaves the robot standing (does **not** enter damp/zero-torque unless the
    operator uses the killswitch). Always attempts arm_sdk release even when
    the caller has no ``G1Arms`` handle.
    """
    logger.warning("Graceful reset: StopMove + release arm_sdk")
    stop_loco(loco, iface=iface)
    release_arms(arms, iface=iface)
    if camera is not None:
        try:
            camera.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("camera disconnect failed: %s", exc)


def enter_damp(*, iface: str | None = None, loco: Any | None = None) -> None:
    """Passive damp mode (pendant L2+B). Soft-falls the robot if unsupported."""
    client = loco or init_loco(iface)
    stop_loco(client)
    try:
        # Best-effort: release arm overlay before going passive.
        release_arms(iface=iface)
        rc = client.Damp()
        logger.warning("LocoClient.Damp() -> %s  (passive damping)", rc)
    except Exception as exc:  # noqa: BLE001
        logger.error("Damp failed: %s", exc)
        raise


def enter_zero_torque(*, iface: str | None = None, loco: Any | None = None) -> None:
    """Zero-torque mode (pendant L2+A). Robot goes fully limp — spotter ready."""
    client = loco or init_loco(iface)
    stop_loco(client)
    try:
        release_arms(iface=iface)
        rc = client.ZeroTorque()
        logger.warning("LocoClient.ZeroTorque() -> %s  (motors limp)", rc)
    except Exception as exc:  # noqa: BLE001
        logger.error("ZeroTorque failed: %s", exc)
        raise


def balance_stand(*, iface: str | None = None, loco: Any | None = None) -> None:
    """Re-engage balancer stand / start FSM after a killswitch event."""
    client = loco or init_loco(iface)
    try:
        if hasattr(client, "Start"):
            rc = client.Start()
            logger.info("LocoClient.Start() -> %s", rc)
        else:
            rc = client.SetFsmId(500)
            logger.info("LocoClient.SetFsmId(500) -> %s", rc)
    except Exception as exc:  # noqa: BLE001
        logger.error("balance stand / Start failed: %s", exc)
        raise
