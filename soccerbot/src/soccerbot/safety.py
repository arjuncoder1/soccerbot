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

logger = logging.getLogger(__name__)


def init_loco(iface: str | None = None) -> Any:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

    # ChannelFactoryInitialize is process-global; calling twice raises on some
    # SDK builds, so tolerate an already-initialized factory.
    try:
        if iface:
            ChannelFactoryInitialize(0, iface)
        else:
            ChannelFactoryInitialize(0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("DDS already initialized or init soft-failed: %s", exc)

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


def release_arms(arms: Any | None) -> None:
    """Hand arms back to the stock balancer (ramp arm_sdk weight → 0)."""
    if arms is None:
        return
    try:
        if hasattr(arms, "release"):
            arms.release()
        elif hasattr(arms, "disconnect"):
            arms.disconnect()
        logger.info("arm_sdk released (arms back to balancer)")
    except Exception as exc:  # noqa: BLE001
        logger.warning("arm release failed: %s", exc)


def graceful_reset(
    *,
    arms: Any | None = None,
    loco: Any | None = None,
    iface: str | None = None,
    camera: Any | None = None,
) -> None:
    """Ctrl+C / abort path: stop motion, release arms, disconnect camera.

    Leaves the robot standing (does **not** enter damp/zero-torque unless the
    operator uses the killswitch).
    """
    logger.warning("Graceful reset: StopMove + release arm_sdk")
    stop_loco(loco, iface=iface)
    release_arms(arms)
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
