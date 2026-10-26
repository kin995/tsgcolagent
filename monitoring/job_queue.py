from __future__ import annotations

import os
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import ExperimentRegistry


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentJob:
    experiment_id: str
    command: list[str]
    env: dict[str, str]
    log_path: Path


class ExperimentJobQueue:
    """Runs directory experiments serially so one dataset root finishes before the next starts."""

    def __init__(self, registry: ExperimentRegistry, processes: dict[str, subprocess.Popen[bytes]]):
        self.registry = registry
        self.processes = processes
        self.condition = threading.Condition()
        self.pending: deque[ExperimentJob] = deque()
        self.active_id: str | None = None
        self.worker = threading.Thread(target=self._run, name="tsg-experiment-queue", daemon=True)
        self.worker.start()

    def enqueue(self, job: ExperimentJob) -> int:
        with self.condition:
            identifiers = [item.experiment_id for item in self.pending]
            if job.experiment_id == self.active_id or job.experiment_id in identifiers:
                raise ValueError("experiment is already running or queued")
            self.pending.append(job)
            position = len(self.pending)
            self.registry.update(
                job.experiment_id,
                execution_status="queued",
                queued_at=_now(),
                process_id=None,
                return_code=None,
                last_error=None,
            )
            self.condition.notify()
            return position

    def state(self, experiment_id: str) -> dict[str, Any]:
        with self.condition:
            if experiment_id == self.active_id:
                process = self.processes.get(experiment_id)
                return {"status": "running", "position": 0, "process_id": process.pid if process else None}
            identifiers = [item.experiment_id for item in self.pending]
            if experiment_id in identifiers:
                return {"status": "queued", "position": identifiers.index(experiment_id) + 1, "process_id": None}
        return {}

    def _run(self) -> None:
        while True:
            with self.condition:
                while not self.pending:
                    self.condition.wait()
                job = self.pending.popleft()
                self.active_id = job.experiment_id
            self._execute(job)
            with self.condition:
                self.active_id = None

    def _execute(self, job: ExperimentJob) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process: subprocess.Popen[bytes] | None = None
        try:
            with job.log_path.open("ab") as log_handle:
                process = subprocess.Popen(
                    job.command,
                    cwd=str(Path(__file__).resolve().parents[1]),
                    env=job.env,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    close_fds=os.name != "nt",
                )
            self.processes[job.experiment_id] = process
            self.registry.update(job.experiment_id, execution_status="running", process_id=process.pid, started_at=_now())
            return_code = process.wait()
            self.registry.update(
                job.experiment_id,
                execution_status="completed" if return_code == 0 else "failed",
                process_id=process.pid,
                return_code=return_code,
                finished_at=_now(),
                last_error=None if return_code == 0 else f"experiment process exited with code {return_code}",
            )
        except Exception as exc:
            try:
                self.registry.update(
                    job.experiment_id,
                    execution_status="failed",
                    finished_at=_now(),
                    last_error=f"{exc.__class__.__name__}: {exc}",
                )
            except OSError:
                pass
        finally:
            if process is None or process.poll() is not None:
                self.processes.pop(job.experiment_id, None)
