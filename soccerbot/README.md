# soccerbot

FastAPI control plane for the G1 soccer-ball demo orchestrator
(`scripted-behavior/main.py`).

## API

```bash
# from repo root
uv sync --package soccerbot
uv run soccerbot-api
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness |
| `GET` | `/api/orchestrator/status` | Current run state + log tail |
| `POST` | `/api/orchestrator/start` | Start demo (JSON body) |
| `POST` | `/api/orchestrator/stop` | Terminate running demo |

Example start body:

```json
{
  "backend": "replay",
  "iface": "eth0",
  "pickup_duration_s": 30
}
```

Pair with the Nuxt UI in [`../ui`](../ui).
