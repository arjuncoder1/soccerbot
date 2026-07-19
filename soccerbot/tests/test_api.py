"""Smoke tests for the orchestrator control API."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from soccerbot.main import app
from soccerbot.orchestrator_runner import OrchestratorRunner, OrchestratorState, StartRequest


def test_health() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_status_idle() -> None:
    client = TestClient(app)
    body = client.get("/api/orchestrator/status").json()
    assert body["state"] in {"idle", "succeeded", "failed", "stopped", "running"}


def test_start_requires_remote_server() -> None:
    client = TestClient(app)
    resp = client.post(
        "/api/orchestrator/start",
        json={"backend": "remote", "iface": "eth0"},
    )
    assert resp.status_code == 400


def test_runner_builds_command() -> None:
    runner = OrchestratorRunner()
    cmd = runner._build_command(
        StartRequest(backend="replay", iface="eth0", pickup_duration_s=12.0)
    )
    assert "scripted-behavior/main.py" in cmd[3].replace("\\", "/")
    assert "--backend" in cmd and "replay" in cmd
    assert "--iface" in cmd and "eth0" in cmd
    assert "--pickup-duration" in cmd and "12.0" in cmd


def test_start_conflict_when_running() -> None:
    client = TestClient(app)
    fake_status = MagicMock()
    fake_status.state = OrchestratorState.RUNNING
    fake_status.pid = 123
    fake_status.exit_code = None
    fake_status.started_at = 1.0
    fake_status.finished_at = None
    fake_status.command = ["uv", "run", "python", "main.py"]
    fake_status.error = None
    fake_status.log_tail = []

    with (
        patch("soccerbot.main.runner.start", side_effect=RuntimeError("orchestrator is already running")),
        patch("soccerbot.main.runner.status", return_value=fake_status),
    ):
        resp = client.post(
            "/api/orchestrator/start",
            json={"backend": "replay", "iface": "eth0"},
        )
    assert resp.status_code == 409
