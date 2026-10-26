"""内存任务队列：每次生成在后台线程跑，前端轮询状态。"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .config import Settings
from .engine import SOURCE_SUFFIXES, run_source, write_job_record

Runner = Callable[[Path, Path, Settings, Callable[[str], None], Callable[[], bool]], dict[str, object]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    source: str
    out_dir: str
    settings: dict[str, object]
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, object] | None = None
    log: list[str] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def add_log(self, message: str) -> None:
        self.log.append(f"{time.strftime('%H:%M:%S')} {message}")
        if len(self.log) > 400:
            del self.log[: len(self.log) - 400]

    def public(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "out_dir": self.out_dir,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result": self.result,
            "log": self.log,
            "settings": self.settings,
        }


class JobManager:
    def __init__(self, settings: Settings, runner: Runner | None = None):
        self.settings = settings
        self.runner: Runner = runner or run_source
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, source: str, **overrides: object) -> Job:
        path = Path(source).expanduser()
        if not path.exists():
            raise ValueError(f"路径不存在：{path}")
        if path.is_file() and path.suffix.lower() not in SOURCE_SUFFIXES:
            raise ValueError("只支持 .py 或 .java 文件，或者一个目录")
        settings = Settings(**{**self.settings.__dict__})
        for key, value in overrides.items():
            if value is not None and hasattr(settings, key):
                setattr(settings, key, value)

        job_id = uuid.uuid4().hex[:12]
        out_root = Path(settings.runs_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        stem = path.stem if path.is_file() else path.name
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem) or "run"
        out_dir = out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}_{job_id[:4]}"
        job = Job(
            id=job_id,
            source=str(path.resolve()),
            out_dir=str(out_dir.resolve()),
            settings=settings.as_public(),
        )
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._execute, args=(job, settings), daemon=True, name=f"tsgtest-{job_id}")
        thread.start()
        return job

    def _execute(self, job: Job, settings: Settings) -> None:
        job.status = "running"
        job.started_at = _now()
        job.add_log("任务开始")
        payload: dict[str, object] = {}
        try:
            job.result = self.runner(
                Path(job.source), Path(job.out_dir), settings, job.add_log, job.cancel_event.is_set
            )
            job.status = "cancelled" if job.cancel_event.is_set() else "done"
            job.add_log("任务结束")
        except BaseException as exc:  # noqa: BLE001 - 任务线程必须吞掉异常
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.add_log(f"失败：{job.error}")
        finally:
            job.finished_at = _now()
            payload = job.public()
            write_job_record(Path(job.out_dir), payload)

    def list(self) -> list[dict[str, object]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.public() for job in jobs]

    def get(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.public() if job else None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.status in {"done", "failed", "cancelled"}:
            return False
        job.cancel_event.set()
        job.add_log("收到取消请求（单文件任务会在本轮结束后停止）")
        return True
