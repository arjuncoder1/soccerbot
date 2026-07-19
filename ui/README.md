# Soccerbot UI

Nuxt control surface for starting the G1 demo orchestrator via the FastAPI
service in `soccerbot/`.

## Run

```bash
# terminal 1 — API
cd ..
uv sync --package soccerbot
uv run soccerbot-api

# terminal 2 — UI
cd ui
npm install
NUXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run dev
```

Open http://localhost:3000
