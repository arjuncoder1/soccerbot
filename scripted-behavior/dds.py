"""Process-wide DDS init guard.

Every live stage that talks to the robot must call ``ensure_dds`` before
opening any ``unitree_sdk2py`` channels; the SDK's
``ChannelFactoryInitialize`` may only be called once per process.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("scripted_behavior.dds")

_initialized = False


def ensure_dds(iface: str | None) -> None:
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
