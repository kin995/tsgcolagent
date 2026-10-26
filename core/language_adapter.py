from __future__ import annotations

from pathlib import Path


def detect_language(source_path: str | Path) -> str:
    suffix = Path(source_path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".java":
        return "java"
    return "unknown"
