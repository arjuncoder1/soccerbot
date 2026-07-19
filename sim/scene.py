"""Spawn G1 robot + table + ball in a running Isaac Sim instance.

Run standalone on the Brev instance to verify assets load correctly:
    python3 sim/scene.py

Requires Isaac Sim 6.0.1+ Python environment (isaacsim package).

IMPORTANT: SimulationApp must be instantiated before any other isaacsim
imports (Carbonite framework requirement).
"""

from __future__ import annotations

import logging

logger = logging.getLogger("sim.scene")

# Isaac Sim asset paths — adjust if your Nucleus/local paths differ.
G1_USD = "/Isaac/Robots/Unitree/G1/g1.usd"
TABLE_USD = "/Isaac/Environments/Simple_Room/Props/table_low.usd"

# Scene layout (metres, Z-up).
G1_POSITION = (0.0, 0.0, 0.0)
TABLE_POSITION = (0.8, 0.0, 0.0)  # ~80 cm in front of robot
BALL_POSITION = (0.8, 0.0, 0.75)  # on the table surface
BALL_RADIUS = 0.04  # soccer ball ~8 cm diameter

# Module-level handle; set by ensure_sim_app().
_simulation_app = None


def ensure_sim_app(headless: bool = True):
    """Instantiate SimulationApp exactly once (must happen before other imports)."""
    global _simulation_app
    if _simulation_app is not None:
        return _simulation_app
    from isaacsim import SimulationApp

    _simulation_app = SimulationApp({"headless": headless})
    logger.info("SimulationApp instantiated (headless=%s)", headless)
    return _simulation_app


def build_scene():
    """Create the demo scene. Returns (world, robot_articulation, ball_prim).

    Calls ensure_sim_app() automatically — safe to call multiple times.
    """
    ensure_sim_app()

    from isaacsim.core import World
    from isaacsim.core.objects import DynamicSphere
    from isaacsim.core.utils.stage_utils import add_reference_to_stage

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # G1 robot
    robot_prim_path = "/World/G1"
    add_reference_to_stage(usd_path=G1_USD, prim_path=robot_prim_path)
    from isaacsim.core.articulations import Articulation

    robot = world.scene.add(
        Articulation(prim_path=robot_prim_path, name="g1", position=G1_POSITION)
    )

    # Table
    table_prim_path = "/World/Table"
    add_reference_to_stage(usd_path=TABLE_USD, prim_path=table_prim_path)
    from isaacsim.core.prims import XFormPrim

    world.scene.add(
        XFormPrim(prim_path=table_prim_path, name="table", position=TABLE_POSITION)
    )

    # Ball
    ball = world.scene.add(
        DynamicSphere(
            prim_path="/World/Ball",
            name="ball",
            radius=BALL_RADIUS,
            position=BALL_POSITION,
            color=(1.0, 0.5, 0.0),  # orange
        )
    )

    world.reset()
    logger.info("Scene built: G1 @ %s, table @ %s, ball @ %s", G1_POSITION, TABLE_POSITION, BALL_POSITION)
    return world, robot, ball


def discover_joint_names():
    """Print robot DOF names — run once on Brev to get the mapping."""
    world, robot, _ = build_scene()
    world.step()
    print("=== G1 DOF names ===")
    for i, name in enumerate(robot.dof_names):
        print(f"  [{i:2d}] {name}")
    print(f"Total DOFs: {robot.num_dof}")
    return robot.dof_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    discover_joint_names()
