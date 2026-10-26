from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable


CACHE_SCHEMA_VERSION = 2


class EvaluationCache:
    """Small persistent cache for deterministic tool results within one task."""

    def __init__(self, path: str | Path, max_entries: int = 48):
        self.path = Path(path)
        self.max_entries = max(8, int(max_entries))

    def key(
        self,
        *,
        language: str,
        stage: str,
        profile: str,
        files: Iterable[str | Path],
        settings: dict[str, object] | None = None,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(f"tsg-evaluation-cache-v{CACHE_SCHEMA_VERSION}\0{language}\0{stage}\0{profile}\0".encode())
        digest.update(json.dumps(runtime_fingerprint(language), sort_keys=True, ensure_ascii=True).encode())
        digest.update(b"\0")
        for raw_path in sorted({str(Path(item).resolve()) for item in files}):
            path = Path(raw_path)
            digest.update(raw_path.encode("utf-8", errors="surrogatepass"))
            digest.update(b"\0")
            if path.is_file():
                digest.update(path.read_bytes())
            else:
                digest.update(b"<missing>")
            digest.update(b"\0")
        digest.update(json.dumps(settings or {}, sort_keys=True, ensure_ascii=True, default=str).encode())
        return digest.hexdigest()

    def get(self, stage: str, key: str) -> dict[str, object] | None:
        payload = self._load()
        item = payload.get("entries", {}).get(f"{stage}:{key}")
        if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
            return None
        return dict(item["result"])

    def put(self, stage: str, key: str, result: dict[str, object]) -> None:
        payload = self._load()
        entries = payload.setdefault("entries", {})
        entries[f"{stage}:{key}"] = {"result": result}
        while len(entries) > self.max_entries:
            entries.pop(next(iter(entries)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        content = json.dumps(payload, indent=2, ensure_ascii=False)
        temp_path.write_text(content, encoding="utf-8")
        try:
            temp_path.replace(self.path)
        except OSError:
            self.path.write_text(content, encoding="utf-8")
            try:
                temp_path.unlink()
            except OSError:
                pass

    def _load(self) -> dict[str, object]:
        if not self.path.is_file():
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(payload.get("entries"), dict):
            return {"schema_version": CACHE_SCHEMA_VERSION, "entries": {}}
        return payload


def dependency_files(root: str | Path, language: str) -> list[Path]:
    root = Path(root).resolve()
    names = (
        ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradle.properties")
        if language == "java"
        else ("pyproject.toml", "setup.py", "setup.cfg", "pytest.ini", "tox.ini", "requirements.txt")
    )
    files = [root / name for name in names if (root / name).is_file()]
    if language == "python":
        files.extend(sorted(path for path in root.glob("requirements*.txt") if path.is_file()))
    elif language == "java":
        wrapper = root / "gradle" / "wrapper" / "gradle-wrapper.properties"
        if wrapper.is_file():
            files.append(wrapper)
    return files


@lru_cache(maxsize=4)
def runtime_fingerprint(language: str) -> dict[str, object]:
    """Keep cached evidence tied to the tool/runtime environment that produced it."""
    common = {
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if language == "java":
        return {
            **common,
            "java_home": os.getenv("JAVA_HOME", ""),
            "maven_home": os.getenv("MAVEN_HOME", os.getenv("M2_HOME", "")),
            "gradle_home": os.getenv("GRADLE_HOME", ""),
        }
    versions: dict[str, str] = {}
    for package in ("pytest", "coverage", "pytest-cov"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return {
        **common,
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "packages": versions,
    }


def source_dependency_files(source: str | Path, root: str | Path, language: str) -> list[Path]:
    """Return local source files whose changes can alter evaluation evidence."""
    source_path = Path(source).resolve()
    root_path = Path(root).resolve()
    if language == "java":
        main_root = root_path / "src" / "main"
        test_root = root_path / "src" / "test"
        files = list(main_root.rglob("*.java")) if main_root.is_dir() else list(source_path.parent.glob("*.java"))
        if test_root.is_dir():
            files.extend(test_root.rglob("*.java"))
        return sorted({path.resolve() for path in [source_path, *files] if path.is_file()})
    return _python_dependency_closure(source_path, root_path)


def _python_dependency_closure(source: Path, root: Path, limit: int = 256) -> list[Path]:
    search_roots = []
    for candidate in (root, source.parent):
        if candidate.is_dir() and candidate not in search_roots:
            search_roots.append(candidate)
    pending = [source]
    visited: set[Path] = set()
    while pending and len(visited) < limit:
        path = pending.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for imported in _local_import_paths(tree, path, search_roots):
            if imported not in visited:
                pending.append(imported)
    return sorted(visited)


def _local_import_paths(tree: ast.AST, current: Path, roots: list[Path]) -> set[Path]:
    found: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[tuple[str, Path | None]] = []
        if isinstance(node, ast.Import):
            modules.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = current.parent
                for _ in range(max(0, node.level - 1)):
                    base = base.parent
                if node.module:
                    modules.append((node.module, base))
                else:
                    modules.extend((alias.name, base) for alias in node.names)
            elif node.module:
                modules.append((node.module, None))
        for module, relative_base in modules:
            fragment = Path(*[part for part in module.split(".") if part])
            bases = [relative_base] if relative_base is not None else roots
            for base in bases:
                if base is None:
                    continue
                for candidate in (base / fragment.with_suffix(".py"), base / fragment / "__init__.py"):
                    if candidate.is_file():
                        found.add(candidate.resolve())
    return found
