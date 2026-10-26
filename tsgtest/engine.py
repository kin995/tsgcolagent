"""引擎适配层：把现有论文实现包装成“一次运行”。

只在真正执行任务时才导入 ``main``，因此没装 langgraph 也能打开界面。
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .config import Settings

SOURCE_SUFFIXES = {".py", ".java"}
MAX_PREVIEW_BYTES = 200_000


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def dependency_status() -> dict[str, bool]:
    modules = {
        "langgraph": "langgraph",
        "langgraph_sqlite": "langgraph.checkpoint.sqlite",
        "tree_sitter": "tree_sitter",
        "tree_sitter_java": "tree_sitter_java",
        "pytest": "pytest",
        "coverage": "coverage",
    }
    status = {label: _has_module(name) for label, name in modules.items()}
    status["java"] = shutil.which("java") is not None
    status["mvn"] = shutil.which("mvn") is not None
    return status


def ollama_models(base_url: str, timeout: float = 4.0) -> tuple[list[str], str | None]:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return [], f"无法连接 Ollama（{exc.reason}）"
    except (OSError, ValueError) as exc:
        return [], f"读取 Ollama 模型列表失败：{exc}"
    names = []
    for item in payload.get("models", []) if isinstance(payload, dict) else []:
        name = item.get("name") or item.get("model")
        if name:
            names.append(str(name))
    return sorted(names), None


def _build_llm_config(settings: Settings):
    from llm_config import LLMConfig

    return LLMConfig(
        enabled=True,
        required=True,
        provider="ollama",
        base_url=settings.ollama_base_url,
        model=settings.model,
        api_key="local",
        timeout=settings.request_timeout,
        temperature=settings.temperature,
    )


def collect_sources(source: Path, limit: int) -> list[Path]:
    if source.is_file():
        return [source]
    files = [p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES]
    files = [p for p in files if "__pycache__" not in p.parts]
    return files[:limit]


def _read_preview(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        with candidate.open("rb") as fh:
            data = fh.read(MAX_PREVIEW_BYTES)
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


def summarize(summary: dict, source: Path, out_dir: Path) -> dict[str, object]:
    final = summary.get("final_report") if isinstance(summary.get("final_report"), dict) else {}
    paper = summary.get("paper_metrics") if isinstance(summary.get("paper_metrics"), dict) else {}
    test_path = summary.get("test_path")
    return {
        "source": str(source),
        "output_dir": str(out_dir),
        "test_path": test_path,
        "test_text": _read_preview(test_path),
        "status": summary.get("quality_status") or summary.get("stop_reason"),
        "grade": summary.get("final_grade"),
        "stop_reason": summary.get("stop_reason"),
        "iterations": summary.get("iterations"),
        "pytest_passed": final.get("pytest_passed"),
        "metrics": {
            "line_coverage": final.get("line_coverage"),
            "branch_coverage": final.get("branch_coverage"),
            "ae": final.get("ae"),
            "mutation_score": final.get("mutation_score"),
            "pass_rate": paper.get("pass_rate"),
            "tir": paper.get("tir"),
            "tir_comparable": paper.get("tir_comparable"),
            "avg_exec": paper.get("avg_exec"),
            "duration_seconds": paper.get("duration_seconds"),
            "collected_count": final.get("collected_count"),
            "effective_mutants": final.get("effective_mutants"),
            "effective_assertions": final.get("effective_assertions"),
            "assertion_count": final.get("assertion_count"),
        },
        "llm": summary.get("llm") or {},
        "error": summary.get("last_error"),
    }


def run_one(source: Path, out_dir: Path, settings: Settings, log: Callable[[str], None]) -> dict[str, object]:
    from graph.experience_memory import configure_knowledge_store_path
    from main import MultiAgentUnitTestSystem, Thresholds

    out_dir.mkdir(parents=True, exist_ok=True)
    if settings.knowledge_store:
        configure_knowledge_store_path(settings.knowledge_store)
    log(f"分析源码：{source.name}")
    system = MultiAgentUnitTestSystem(
        source_path=source,
        output_dir=out_dir,
        thresholds=Thresholds(),
        max_iterations=settings.max_rounds,
        llm_config=_build_llm_config(settings),
    )
    started = time.time()
    summary = system.run()
    elapsed = round(time.time() - started, 1)
    result = summarize(summary, source, out_dir)
    log(f"完成：{source.name}（{elapsed}s，状态 {result['status']}）")
    return result


def run_source(
    source: Path,
    out_dir: Path,
    settings: Settings,
    log: Callable[[str], None],
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, object]:
    """单文件或目录。目录模式下逐个文件执行，返回聚合结果。"""
    if settings.knowledge_store:
        from graph.experience_memory import configure_knowledge_store_path

        configure_knowledge_store_path(settings.knowledge_store)

    if source.is_file():
        return {"kind": "single", "items": [run_one(source, out_dir, settings, log)]}

    sources = collect_sources(source, settings.max_files)
    if not sources:
        raise ValueError(f"目录中没有 .py/.java 文件：{source}")
    log(f"目录模式：共 {len(sources)} 个文件（上限 {settings.max_files}）")
    items = []
    for index, path in enumerate(sources, 1):
        if cancelled():
            log("已取消，停止后续文件")
            break
        log(f"[{index}/{len(sources)}] {path.name}")
        items.append(run_one(path, out_dir / path.stem, settings, log))
    return {"kind": "batch", "items": items}


def write_job_record(out_dir: Path, payload: dict[str, object]) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "tsgtest_job.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass
