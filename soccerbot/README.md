# soccerbot

Core orchestrator for the Unitree G1 soccer-ball pickup demo.

This package is **workspace-local** (not published to PyPI). The logic pieces live
in sibling packages that soccerbot imports in-process:

| Dependency | Role |
|---|---|
| `local-vla-inference/` | ACT pickup (`ajkoder/g1-pickup-ball-act`), slew clamp, teleimager RGB, Rerun |
| `scripted-behavior/` | Turn 180°, avoid/shuffle, throw, JSON trajectory replay |

## Run

```bash
# one-time robot setup (Python 3.12 + CycloneDDS)
./install.sh

# headed killswitch in a second terminal (keep open during demos)
./killswitch.sh --iface enp5s0

# full demo: ACT pickup → turn → avoid → throw
./run_soccerbot.sh --iface enp5s0

# safer smoke: replay recorded pickup trajectory instead of ACT
./run_soccerbot.sh --iface enp5s0 --backend replay

# ACT policy + record full demo to .rrd (headless / Waldo)
mkdir -p logs
./run_soccerbot.sh --iface enp5s0 --backend local \
  --record-path logs/demo.rrd --no-display
# then: scp logs/demo.rrd … && rerun demo.rrd
```

Defaults match the validated local ACT command:

- `--layout 14d`
- `--policy ajkoder/g1-pickup-ball-act`
- `--clamp 0.002`
- `--camera zmq://192.168.123.164:55555` (working teleimager head JPEG port)

## Safety

- Every arm command path is slew-clamped (ACT + replay + throw).
- **Ctrl+C** → graceful reset: `LocoClient.StopMove()` + release `arm_sdk`.
- **`./killswitch.sh`** → headed GUI: Stop Move / Damp / Zero Torque / Start.

## Diagnose

```bash
./diagnose.sh --iface enp5s0
```
