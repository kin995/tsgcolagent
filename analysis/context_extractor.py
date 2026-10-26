from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.language_adapter import detect_language
from core.structured_context import StructuredContext


class StructuredContextExtractor(ABC):
    @abstractmethod
    def extract(self, source_path: str | Path, project_root: str | Path | None = None) -> StructuredContext:
        raise NotImplementedError


def extractor_for(source_path: str | Path) -> StructuredContextExtractor:
    language = detect_language(source_path)
    if language == "python":
        from analysis.python_context_extractor import PythonStructuredContextExtractor

        return PythonStructuredContextExtractor()
    if language == "java":
        from analysis.java_context_extractor import JavaStructuredContextExtractor

        return JavaStructuredContextExtractor()
    raise ValueError(f"Unsupported source language for {source_path}")

