"""Process-wide DDS init guard (re-exports local-vla-inference's singleton).

Every live stage that talks to the robot must call ``ensure_dds`` before
opening any ``unitree_sdk2py`` channels; the SDK's
``ChannelFactoryInitialize`` may only be called once per process.
"""

from __future__ import annotations

import sys
from pathlib import Path

_LOCAL_VLA = Path(__file__).resolve().parent.parent / "local-vla-inference"
if str(_LOCAL_VLA) not in sys.path:
    sys.path.insert(0, str(_LOCAL_VLA))

from dds_init import dds_initialized, ensure_dds  # noqa: E402

__all__ = ["ensure_dds", "dds_initialized"]
