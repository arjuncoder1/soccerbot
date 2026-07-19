"""Process-wide DDS ChannelFactoryInitialize guard.

``ChannelFactoryInitialize`` may only be called once per process. Every
entry point that talks to the robot (ACT loop, scripted stages, killswitch)
must go through ``ensure_dds`` instead of calling the SDK directly.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_initialized = False


def ensure_dds(iface: str | None = None) -> None:
    """Idempotent ``ChannelFactoryInitialize``. Safe to call from any stage."""
    global _initialized
    if _initialized:
        return
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if iface:
        logger.info("DDS ChannelFactoryInitialize(0, %r)", iface)
        ChannelFactoryInitialize(0, iface)
    else:
        logger.info("DDS ChannelFactoryInitialize(0) [default interface]")
        ChannelFactoryInitialize(0)
    _initialized = True


def dds_initialized() -> bool:
    return _initialized
