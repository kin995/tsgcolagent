"""临时工作目录：优先 STATEFLOW_TMP_DIR，避免 0700 目录在受限环境被拒。"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def temp_workspace(prefix: str = "stateflow_"):
    base = os.environ.get("STATEFLOW_TMP_DIR") or tempfile.gettempdir()
    Path(base).mkdir(parents=True, exist_ok=True)
    path = Path(base) / f"{prefix}{uuid.uuid4().hex[:16]}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)
