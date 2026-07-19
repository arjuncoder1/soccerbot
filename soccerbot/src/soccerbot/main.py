"""FastAPI app: start / stop / status for the scripted-behavior orchestrator."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from soccerbot.orchestrator_runner import StartRequest, runner

logger = logging.getLogger("soccerbot.api")

app = FastAPI(
    title="Soccerbot Orchestrator API",
    description="HTTP control plane for scripted-behavior/main.py",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PickupBackendName(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"
    REPLAY = "replay"


class StartBody(BaseModel):
    backend: PickupBackendName = PickupBackendName.REPLAY
    iface: str | None = Field(default=None, description="DDS network interface, e.g. eth0")
    pickup_duration_s: float = Field(default=30.0, gt=0, le=600)
    remote_server: str | None = Field(
        default=None,
        description="HOST:PORT for backend=remote",
    )
    pickup_extra_args: list[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    state: Literal["idle", "running", "succeeded", "failed", "stopped"]
    pid: int | None = None
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    command: list[str] = Field(default_factory=list)
    error: str | None = None
    log_tail: list[str] = Field(default_factory=list)


def _to_response() -> StatusResponse:
    status = runner.status()
    return StatusResponse(
        state=status.state.value,
        pid=status.pid,
        exit_code=status.exit_code,
        started_at=status.started_at,
        finished_at=status.finished_at,
        command=status.command,
        error=status.error,
        log_tail=status.log_tail,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/orchestrator/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    return _to_response()


@app.post("/api/orchestrator/start", response_model=StatusResponse)
def start_orchestrator(body: StartBody) -> StatusResponse:
    if body.backend is PickupBackendName.REMOTE and not body.remote_server:
        raise HTTPException(
            status_code=400,
            detail="remote_server is required when backend=remote",
        )
    try:
        runner.start(
            StartRequest(
                backend=body.backend.value,
                iface=body.iface,
                pickup_duration_s=body.pickup_duration_s,
                remote_server=body.remote_server,
                pickup_extra_args=list(body.pickup_extra_args),
            )
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response()


@app.post("/api/orchestrator/stop", response_model=StatusResponse)
def stop_orchestrator() -> StatusResponse:
    runner.stop()
    return _to_response()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "soccerbot.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
