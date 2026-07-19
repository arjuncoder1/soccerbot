from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

candidates = [
    "from isaacsim.core.api import World",
    "from omni.isaac.core import World",
    "from isaacsim.core.world import World",
]
for stmt in candidates:
    try:
        exec(stmt)
        print("SUCCESS:", stmt)
        break
    except Exception as e:
        print("FAIL:", stmt, "->", e)

app.close()
