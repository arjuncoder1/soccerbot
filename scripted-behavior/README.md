# scripted-behavior

Hardcoded (non-learned) post-pickup robot control. No model, no training.

## `throw.py` — gentle goalkeeper forward push

Call `throw(arms)` once the ball is already in the G1's hands. Unlike an
earlier version of this file, it does **not** drive to some fixed pose
first — it reads whatever the current arm position actually is (assumed to
already be a "holding the ball in front of the torso" pose) and applies a
small, gentle forward push *relative* to that: nudges `shoulder_pitch` down
and `elbow` up by a fixed amount, leaves `shoulder_roll`/`shoulder_yaw`
completely alone (so whatever left/right hand placement the current hold
already has is preserved), then returns to the exact starting pose
afterward.

```python
from throw import throw
# arms = a connected, already-engaged g1_arms.G1Arms instance
throw(arms)
```

Since there's no single fixed trajectory to verify anymore, the push *delta*
is checked against a family of plausible starting poses
(`_CANDIDATE_HOLDING_POSES` in `throw.py`) rather than one hand-picked
pose — for every candidate: real G1 URDF joint limits respected (`g1_arm_fk.py`,
sourced from `unitreerobotics/unitree_ros`), hand moves forward by a real
amount, elbow never swings behind the torso, hand never crosses the body's
centerline. This runs at **import time** (`_self_check()`) — importing
`throw.py` raises immediately if any candidate fails. `pytest test_throw.py`
covers the same ground (9 tests).

```bash
python throw.py              # print the delta + forward-gain table per candidate, no hardware
python throw.py --execute    # run for real (see the SAFETY note in throw.py first)
pytest test_throw.py
```

Both `throw.py` and `g1_arm_fk.py` are stdlib-only (no deps needed for
`--dry-run` or the tests); `--execute` needs `unitree_sdk2py` from the
sibling `local-vla-inference/` package (Python 3.12, see its README).

## Not yet built

Turn 180°, depth-based human detection, and the side-to-side shuffle. The
push is the last step of that sequence and can be called on its own once the
others exist.
