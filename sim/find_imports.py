from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

candidates = {
    "stage_utils": [
        "from isaacsim.core.api.utils.stage_utils import add_reference_to_stage",
        "from isaacsim.core.utils.stage_utils import add_reference_to_stage",
        "from omni.isaac.core.utils.stage import add_reference_to_stage",
    ],
    "DynamicSphere": [
        "from isaacsim.core.api.objects import DynamicSphere",
        "from isaacsim.core.objects import DynamicSphere",
        "from omni.isaac.core.objects import DynamicSphere",
    ],
    "Articulation": [
        "from isaacsim.core.api.articulations import Articulation",
        "from isaacsim.core.articulations import Articulation",
        "from omni.isaac.core.articulations import Articulation",
    ],
    "XFormPrim": [
        "from isaacsim.core.api.prims import XFormPrim",
        "from isaacsim.core.prims import XFormPrim",
        "from omni.isaac.core.prims import XFormPrim",
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

app.close()
