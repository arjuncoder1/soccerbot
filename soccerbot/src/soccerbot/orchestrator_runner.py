"""Manage a single long-running scripted-behavior orchestrator subprocess."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("soccerbot.orchestrator_runner")

REPO_ROOT = Path(__file__).resolve().parents[3]
ORCHESTRATOR_MAIN = REPO_ROOT / "scripted-behavior" / "main.py"
MAX_LOG_LINES = 500


class OrchestratorState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class StartRequest:
    backend: str = "replay"
    iface: str | None = None
    pickup_duration_s: float = 30.0
    remote_server: str | None = None
    pickup_extra_args: list[str] = field(default_factory=list)


@dataclass
class OrchestratorStatus:
    state: OrchestratorState
    pid: int | None = None
    exit_code: int | None = None
    started_at: float | None = None
    finished_at: float | None = None
    command: list[str] = field(default_factory=list)
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)


class OrchestratorRunner:
    """Thread-safe singleton-style runner (at most one demo at a time)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._state = OrchestratorState.IDLE
        self._exit_code: int | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._command: list[str] = []
        self._error: str | None = None
        self._log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._reader: threading.Thread | None = None

    def _build_command(self, req: StartRequest) -> list[str]:
        if not ORCHESTRATOR_MAIN.is_file():
            raise FileNotFoundError(f"orchestrator entrypoint missing: {ORCHESTRATOR_MAIN}")

        cmd = [
            "uv",
            "run",
            "python",
            str(ORCHESTRATOR_MAIN),
            "--backend",
            req.backend,
            "--pickup-duration",
            str(req.pickup_duration_s),
        ]
        if req.iface:
            cmd.extend(["--iface", req.iface])
        if req.remote_server:
            cmd.extend(["--remote-server", req.remote_server])
        if req.pickup_extra_args:
            cmd.append("--")
            cmd.extend(req.pickup_extra_args)
        return cmd

    def start(self, req: StartRequest) -> OrchestratorStatus:
        with self._lock:
            self._refresh_locked()
            if self._state is OrchestratorState.RUNNING:
                raise RuntimeError("orchestrator is already running")

            cmd = self._build_command(req)
            self._log.clear()
            self._error = None
            self._exit_code = None
            self._finished_at = None
            self._command = cmd
            self._started_at = time.time()

            logger.info("Starting orchestrator: %s", " ".join(cmd))
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(ORCHESTRATOR_MAIN.parent),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                self._state = OrchestratorState.FAILED
                self._error = str(exc)
                self._finished_at = time.time()
                raise

            self._state = OrchestratorState.RUNNING
            self._reader = threading.Thread(
                target=self._drain_stdout,
                name="orchestrator-log-reader",
                daemon=True,
            )
            self._reader.start()
            return self._status_locked()

    def stop(self) -> OrchestratorStatus:
        with self._lock:
            self._refresh_locked()
            if self._proc is None or self._state is not OrchestratorState.RUNNING:
                return self._status_locked()

            logger.info("Stopping orchestrator pid=%s", self._proc.pid)
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)

            self._exit_code = self._proc.returncode
            self._finished_at = time.time()
            self._state = OrchestratorState.STOPPED
            self._proc = None
            return self._status_locked()

    def status(self) -> OrchestratorStatus:
        with self._lock:
            self._refresh_locked()
            return self._status_locked()

    def _drain_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                text = line.rstrip("\n")
                with self._lock:
                    self._log.append(text)
        except Exception:  # noqa: BLE001 -- reader must never crash the API
            logger.exception("orchestrator log reader failed")

    def _refresh_locked(self) -> None:
        if self._proc is None:
            return
        code = self._proc.poll()
        if code is None:
            self._state = OrchestratorState.RUNNING
            return
        self._exit_code = code
        self._finished_at = self._finished_at or time.time()
        if code == 0:
            self._state = OrchestratorState.SUCCEEDED
        elif self._state is not OrchestratorState.STOPPED:
            self._state = OrchestratorState.FAILED
        self._proc = None

    def _status_locked(self) -> OrchestratorStatus:
        pid = self._proc.pid if self._proc is not None else None
        return OrchestratorStatus(
            state=self._state,
            pid=pid,
            exit_code=self._exit_code,
            started_at=self._started_at,
            finished_at=self._finished_at,
            command=list(self._command),
            error=self._error,
            log_tail=list(self._log),
        )


runner = OrchestratorRunner()
