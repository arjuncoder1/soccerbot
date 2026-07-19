"""Run the full 4-stage scripted-behavior demo in Isaac Sim.

Monkey-patches sys.modules so that scripted-behavior stage imports get
sim backends instead of real DDS/hardware drivers. Zero changes needed
to the stage code itself.

Usage (on Brev instance with Isaac Sim):
    cd soccerbot
    python3 -m sim.run_demo
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

logger = logging.getLogger("sim.run_demo")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTED_DIR = REPO_ROOT / "scripted-behavior"


def _patch_modules() -> None:
    """Inject sim backends into sys.modules before stages import them."""

    # 1. Patch g1_arms module (stages do: from g1_arms import G1Arms)
    from sim.sim_arms import SimG1Arms

    fake_g1_arms = types.ModuleType("g1_arms")
    fake_g1_arms.G1Arms = SimG1Arms  # type: ignore[attr-defined]
    sys.modules["g1_arms"] = fake_g1_arms

    # 2. Patch unitree_sdk2py.g1.loco.g1_loco_client (stages do:
    #    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient)
    from sim.sim_loco import LocoClient as SimLocoClient

    # Build the module hierarchy so the import resolves.
    for mod_name in [
        "unitree_sdk2py",
        "unitree_sdk2py.g1",
        "unitree_sdk2py.g1.loco",
        "unitree_sdk2py.g1.loco.g1_loco_client",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["unitree_sdk2py.g1.loco.g1_loco_client"].LocoClient = SimLocoClient  # type: ignore[attr-defined]

    # 3. Patch unitree_sdk2py.core.channel (for dds.py: ensure_dds)
    fake_channel = types.ModuleType("unitree_sdk2py.core.channel")
    fake_channel.ChannelFactoryInitialize = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    if "unitree_sdk2py.core" not in sys.modules:
        sys.modules["unitree_sdk2py.core"] = types.ModuleType("unitree_sdk2py.core")
    sys.modules["unitree_sdk2py.core.channel"] = fake_channel

    # 4. Patch realsense_human_avoid (avoid stage does:
    #    from realsense_human_avoid import HumanDetector)
    from sim.sim_detect import HumanDetector as MockHumanDetector

    fake_realsense = types.ModuleType("realsense_human_avoid")
    fake_realsense.HumanDetector = MockHumanDetector  # type: ignore[attr-defined]
    sys.modules["realsense_human_avoid"] = fake_realsense


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Patching sys.modules for Isaac Sim backends...")
    _patch_modules()

    # Add scripted-behavior to path so its modules resolve.
    sys.path.insert(0, str(SCRIPTED_DIR))

    # Import the orchestrator main after patching.
    from config import OrchestratorConfig, PickupBackend
    from main import run_demo as run_pipeline

    cfg = OrchestratorConfig(
        backend=PickupBackend.REPLAY,
        iface=None,  # no DDS interface in sim
    )

    logger.info("Starting 4-stage pipeline in Isaac Sim...")
    try:
        run_pipeline(cfg)
    except FileNotFoundError as e:
        # Expected if throw.json trajectory not yet recorded.
        logger.error("Pipeline stopped: %s", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        return 130

    logger.info("Demo complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
