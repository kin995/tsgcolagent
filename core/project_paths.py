from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OTHER_COMPUTER_PROJECT_ROOT = Path(r"<DATASETS>")


def resolve_datasets_root(configured: str | Path | None = None) -> Path:
    if configured:
        return Path(configured).expanduser().resolve()
    for name in ("TSG_DATASETS_ROOT", "STATEFLOW_DATASETS_ROOT"):
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    project_local = PROJECT_ROOT / "datasets"
    if project_local.is_dir():
        return project_local.resolve()
    other_computer = OTHER_COMPUTER_PROJECT_ROOT / "datasets"
    if other_computer.is_dir():
        return other_computer.resolve()
    return project_local.resolve()


def resolve_runs_root(configured: str | Path | None = None) -> Path:
    return Path(configured).expanduser().resolve() if configured else (PROJECT_ROOT / "runs").resolve()
