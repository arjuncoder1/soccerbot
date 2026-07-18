# Dummy LeRobot Dataset (1 episode)

Generated dummy episode in **LeRobot v2.1** format.

- Frames: 60 @ 30 FPS
- State: 6 (joint positions)
- Action: 10
- Cameras: camera_1, camera_2 (480x640)
- Task: "pick up the red cube"

Load with:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="local/dummy", root="/Users/aryan/Desktop/lerobot_dummy_1episode")
print(ds[0].keys())
```
