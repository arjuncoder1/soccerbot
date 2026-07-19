from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

# Round 2: find add_reference_to_stage and Articulation
candidates = {
    "add_reference_to_stage": [
        "from isaacsim.core.utils.stage import add_reference_to_stage",
        "from isaacsim.core.api.utils.stage import add_reference_to_stage",
        "from isaacsim.core.prims.utils import add_reference_to_stage",
        "from pxr import UsdUtils; add_reference_to_stage = None",
    ],
    "Articulation": [
        "from isaacsim.core.api import Articulation",
        "from isaacsim.core.prims import Articulation",
        "from isaacsim.core.api.robots import Articulation",
    ],
}

for name, stmts in candidates.items():
    for stmt in stmts:
        try:
            exec(stmt)
            print("SUCCESS:", name, "->", stmt)
            break
        except Exception as e:
            print("FAIL:", stmt, "->", e)

# Also list what's actually in isaacsim.core.api.articulations
print("\n=== dir(isaacsim.core.api.articulations) ===")
try:
    import isaacsim.core.api.articulations as art_mod
    print([x for x in dir(art_mod) if not x.startswith("_")])
except Exception as e:
    print("ERROR:", e)

# And what's in isaacsim.core.utils
print("\n=== isaacsim.core.utils submodules ===")
try:
    import isaacsim.core.utils as utils_mod
    print([x for x in dir(utils_mod) if not x.startswith("_")])
except Exception as e:
    print("ERROR:", e)

# Check if stage_utils is a function in isaacsim.core.utils
print("\n=== isaacsim.core.utils.stage ===")
try:
    import isaacsim.core.utils.stage as stage_mod
    print([x for x in dir(stage_mod) if not x.startswith("_")])
except Exception as e:
    print("ERROR:", e)

app.close()
