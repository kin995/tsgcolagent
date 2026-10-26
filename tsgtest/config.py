"""运行配置：默认值 → tsgtest.toml → 环境变量，后者覆盖前者。"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

ENV_PREFIX = "TSGTEST_"


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8770
    runs_dir: Path = field(default_factory=lambda: Path("tsgtest_runs"))
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:14b"
    temperature: float = 0.5
    max_rounds: int = 3
    request_timeout: int = 120
    max_files: int = 20
    knowledge_store: Path | None = None

    def as_public(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "runs_dir": str(self.runs_dir),
            "ollama_base_url": self.ollama_base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_rounds": self.max_rounds,
            "max_files": self.max_files,
        }


def _env(name: str) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return value if value not in (None, "") else None


def load_settings(root: Path, config_path: Path | None = None) -> Settings:
    """读取 root/tsgtest.toml（若存在），再用环境变量覆盖。"""
    settings = Settings(runs_dir=root / "tsgtest_runs")
    path = config_path or root / "tsgtest.toml"
    if path.is_file():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        section = data.get("tsgtest") if isinstance(data.get("tsgtest"), dict) else data
        if isinstance(section, dict):
            for key, value in section.items():
                if hasattr(settings, key) and value is not None:
                    setattr(settings, key, value)
    if isinstance(settings.runs_dir, str):
        settings.runs_dir = Path(settings.runs_dir)
        if not settings.runs_dir.is_absolute():
            settings.runs_dir = root / settings.runs_dir
    if settings.knowledge_store is not None and isinstance(settings.knowledge_store, str):
        settings.knowledge_store = Path(settings.knowledge_store)

    overrides = {
        "host": _env("HOST"),
        "port": _env("PORT"),
        "ollama_base_url": _env("OLLAMA_BASE_URL"),
        "model": _env("MODEL"),
        "temperature": _env("TEMPERATURE"),
        "max_rounds": _env("MAX_ROUNDS"),
        "max_files": _env("MAX_FILES"),
    }
    for key, raw in overrides.items():
        if raw is None:
            continue
        try:
            if key in ("port", "max_rounds", "max_files"):
                setattr(settings, key, int(raw))
            elif key == "temperature":
                setattr(settings, key, float(raw))
            else:
                setattr(settings, key, raw)
        except ValueError:
            continue
    if _env("RUNS_DIR"):
        settings.runs_dir = Path(_env("RUNS_DIR"))  # type: ignore[arg-type]
    return settings


def with_overrides(settings: Settings, **changes: object) -> Settings:
    allowed = {k: v for k, v in changes.items() if v is not None and hasattr(settings, k)}
    return replace(settings, **allowed)  # type: ignore[arg-type]
