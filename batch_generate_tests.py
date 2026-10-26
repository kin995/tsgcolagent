from __future__ import annotations

import argparse
import ast
import concurrent.futures
import html
import fnmatch
import hashlib
import json
import os
import re
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from analysis.context_extractor import extractor_for
from analysis.context_rewriter import ContextRewriter
from core.project_paths import resolve_datasets_root
from core.public_metrics import public_result, public_text
from core.experiment_outcome import (
    SECOND_ROUND_GRADES,
    THIRD_ROUND_GRADES,
    VALID_GRADES,
    aggregate_results,
    default_solution,
    failure_category as outcome_failure_category,
    failure_reason as outcome_failure_reason,
    grade_for,
    round_label,
)
from agents.analysis_agent import SourceAnalysisAgent
from agents.generation_agent import TestGenerationAgent
from graph.experience_memory import AgentExperienceMemory, configure_knowledge_store_path, resolve_knowledge_store_path
from llm_client import OpenAICompatibleLLM
from llm_config import LLMConfig
from main import MultiAgentUnitTestSystem, Thresholds
from memory.shared_graph import SharedGraphMemory
from metrics.measurement import batch_paper_metrics, reconcile_round


DEFAULT_SOURCE_DIR = resolve_datasets_root() / "HumanEval"
DEFAULT_TASK_PREFIXES = ("task_", "TestEvatask_", "TestEvaltask_")
DEFAULT_TASK_PATTERN = f"{DEFAULT_TASK_PREFIXES[0]}*.py"
SUCCESSFUL_STATUSES = {
    "generated",
    "quality_high",
    "quality_acceptable",
    "valid_usable",
}
INCOMPLETE_STATUSES = {"failed", "generation_failed", "execution_invalid", "not_started", "interrupted"}
VALID_STATUSES = {"quality_high", "quality_acceptable", "valid_usable"}
FAILED_STATUSES = {"failed", "generation_failed", "execution_invalid"}
JAVA_IGNORED_DIR_NAMES = {
    ".git",
    ".gradle",
    ".idea",
    ".mvn",
    ".pytest_cache",
    ".venv",
    "build",
    "dist",
    "env",
    "generated-sources",
    "generated-test-sources",
    "generated_tests",
    "node_modules",
    "out",
    "runs",
    "test",
    "tests",
    "testfixtures",
    "target",
    "venv",
    "__pycache__",
}
JAVA_IGNORED_DIR_PREFIXES = (
    "_codex_",
    "_report_",
    "_teste_",
    "generated_tests",
)


@dataclass
class BatchItemResult:
    source: str
    output_dir: str
    status: str
    test_path: str | None = None
    graph_path: str | None = None
    summary_path: str | None = None
    final_report: dict[str, object] | None = None
    visualization: dict[str, object] | None = None
    error: str | None = None
    final_grade: str = "E"
    experiment_round: int = 1
    grade_history: list[dict[str, object]] | None = None
    knowledge_history: list[dict[str, object]] | None = None
    failure_reason: str | None = None
    previous_solutions: list[str] | None = None
    failure_report: str | None = None
    round_strategy: str | None = None
    round_failure_diagnostic: dict[str, object] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float = 0.0
    attempt_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return public_result({
            "source": self.source,
            "output_dir": self.output_dir,
            "status": self.status,
            "test_path": self.test_path,
            "graph_path": self.graph_path,
            "summary_path": self.summary_path,
            "final_report": self.final_report,
            "visualization": self.visualization,
            "error": self.error,
            "final_grade": self.final_grade,
            "grade_label": round_label(self.final_grade, self.experiment_round),
            "experiment_round": self.experiment_round,
            "grade_history": self.grade_history or [],
            "knowledge_history": self.knowledge_history or [],
            "failure_reason": self.failure_reason,
            "previous_solutions": self.previous_solutions or [],
            "failure_report": self.failure_report,
            "round_strategy": self.round_strategy,
            "round_failure_diagnostic": self.round_failure_diagnostic,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "attempt_count": self.attempt_count,
        })


@dataclass(frozen=True)
class RoundKnowledgeIsolation:
    live_path: Path
    base_path: Path
    baseline_keys: frozenset[str]
    session_id: str
    experiment_round: int

    def journal_dir(self, case_dir: Path) -> Path:
        return case_dir / ".knowledge_journals" / f"{self.session_id}_R{self.experiment_round}"


def _run_initial_source(
    *,
    source_path: Path,
    case_dir: Path,
    execute: bool,
    thresholds: Thresholds,
    max_iterations: int,
    retry_rounds: int,
    llm_config: LLMConfig,
    execution_budget: int | None,
    llm_call_budget: int | None,
    token_budget: int | None,
    time_budget_seconds: float | None,
    resume: bool,
    rerun_existing: bool,
    knowledge_isolation: RoundKnowledgeIsolation | None = None,
) -> BatchItemResult:
    timing_token = _begin_task_timing(case_dir, source_path, experiment_round=1, strategy="initial_generation")
    try:
        if execute:
            item = _generate_and_execute(
                source_path=source_path,
                case_dir=case_dir,
                thresholds=thresholds,
                max_iterations=max_iterations,
                retry_rounds=retry_rounds,
                llm_config=llm_config,
                execution_budget=execution_budget,
                llm_call_budget=llm_call_budget,
                token_budget=token_budget,
                time_budget_seconds=time_budget_seconds,
                resume=resume,
                rerun_existing=rerun_existing,
                experience_read_path=knowledge_isolation.base_path if knowledge_isolation else None,
                experience_journal_dir=knowledge_isolation.journal_dir(case_dir) if knowledge_isolation else None,
            )
        else:
            item = _generate_only(source_path=source_path, case_dir=case_dir, llm_config=llm_config)
    except Exception as exc:
        error_text = f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc(limit=8)}"
        failure_artifacts = _write_failed_task_artifacts(
            source_path=source_path,
            case_dir=case_dir,
            error=error_text,
            execute=execute,
            llm_config=llm_config,
        )
        item = BatchItemResult(
            source=str(source_path),
            output_dir=str(case_dir),
            status="generation_failed",
            summary_path=failure_artifacts.get("summary_path"),
            visualization=failure_artifacts.get("visualization"),
            error=error_text,
        )
    item = _prepare_and_record_round_item(item, experiment_round=1, execute=execute, llm_config=llm_config)
    timing = _finish_task_timing(case_dir, timing_token, item.status)
    _apply_task_timing(item, timing)
    _persist_round_metadata(item)
    return item


def _effective_worker_count(language: str, sources: list[Path], requested: int | None, fail_fast: bool) -> int:
    if fail_fast or not sources:
        return 1
    python_only = language == "python" or (language == "auto" and all(path.suffix.lower() == ".py" for path in sources))
    humaneval_java = (
        language in {"java", "auto"}
        and all(path.suffix.lower() == ".java" and path.parent.name.casefold() == "humanevaljava" for path in sources)
        and all(not any((path.parent / name).exists() for name in ("pom.xml", "build.gradle", "build.gradle.kts")) for path in sources)
    )
    if not python_only and not humaneval_java:
        return 1
    available = max(1, os.cpu_count() or 1)
    default_workers = 2 if humaneval_java else 4
    target = default_workers if requested is None else max(1, int(requested))
    if humaneval_java:
        target = min(4, target)
    return min(target, available, len(sources))


def _is_humaneval_java_class_dataset(source_root: Path, sources: list[Path], language: str) -> bool:
    return bool(
        language in {"java", "auto"}
        and source_root.name.casefold() == "humanevaljava"
        and sources
        and not any((source_root / name).exists() for name in ("pom.xml", "build.gradle", "build.gradle.kts"))
        and all(path.suffix.lower() == ".java" and path.parent.resolve() == source_root.resolve() for path in sources)
    )


def _knowledge_record_key(record: dict[str, object]) -> str:
    direct = record.get("experience_id") or record.get("strategy_fingerprint") or record.get("candidate_hash")
    if direct:
        return str(direct)
    return hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _prepare_round_knowledge_isolation(
    output_root: Path,
    *,
    experiment_round: int,
    enabled: bool,
) -> RoundKnowledgeIsolation | None:
    if not enabled:
        return None
    live_path = resolve_knowledge_store_path()
    records = AgentExperienceMemory(live_path).load()
    session_id = hashlib.sha256(f"{output_root.resolve()}|{experiment_round}|{_utc_now()}".encode("utf-8")).hexdigest()[:12]
    base_path = output_root / ".knowledge_snapshots" / f"{session_id}_R{experiment_round}.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps({"records": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    return RoundKnowledgeIsolation(
        live_path=live_path,
        base_path=base_path,
        baseline_keys=frozenset(_knowledge_record_key(item) for item in records),
        session_id=session_id,
        experiment_round=experiment_round,
    )


def _merge_round_knowledge(
    isolation: RoundKnowledgeIsolation | None,
    case_dirs: list[Path],
) -> dict[str, object]:
    if isolation is None:
        return {"enabled": False, "merged_records": 0}
    updates: list[dict[str, object]] = []
    baseline = _read_json_object(isolation.base_path).get("records", [])
    for case_dir in sorted(case_dirs):
        journal_dir = isolation.journal_dir(case_dir)
        if not journal_dir.is_dir():
            continue
        isolated = AgentExperienceMemory(
            isolation.live_path,
            read_base_path=isolation.base_path,
            journal_dir=journal_dir,
            publish_canonical=False,
        )
        journal_records = []
        for path in isolated._journal_paths():
            journal_records.extend(_read_json_object(path).get("records", []))
        updates.extend(isolated._dedupe_records(journal_records))
    merged = AgentExperienceMemory(isolation.live_path).merge_records(
        updates, baseline=baseline, merge_id=f"{isolation.session_id}:R{isolation.experiment_round}",
    ) if updates else 0
    evidence = {
        "enabled": True,
        "experiment_round": isolation.experiment_round,
        "snapshot_path": str(isolation.base_path.resolve()),
        "isolated_task_count": len(case_dirs),
        "candidate_updates": len(updates),
        "merged_records": merged,
    }
    evidence_path = isolation.base_path.with_suffix(".merge.json")
    evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    evidence["evidence_path"] = str(evidence_path.resolve())
    return evidence


def _recover_round_knowledge(output_root: Path) -> None:
    """Replay unmerged journals after interruption; merge IDs make replay idempotent."""
    for base_path in sorted((output_root / ".knowledge_snapshots").glob("*_R*.json")):
        if base_path.name.endswith(".merge.json") or base_path.with_suffix(".merge.json").exists():
            continue
        match = re.fullmatch(r"([a-f0-9]+)_R([123])\.json", base_path.name)
        if not match:
            continue
        records = _read_json_object(base_path).get("records", [])
        isolation = RoundKnowledgeIsolation(
            resolve_knowledge_store_path(), base_path,
            frozenset(_knowledge_record_key(item) for item in records),
            match.group(1), int(match.group(2)),
        )
        case_dirs = [path for path in output_root.iterdir() if path.is_dir() and isolation.journal_dir(path).is_dir()]
        if case_dirs:
            _merge_round_knowledge(isolation, case_dirs)


def generate_tests_for_directory(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = "generated_tests_humaneval",
    pattern: str = DEFAULT_TASK_PATTERN,
    language: str = "auto",
    recursive: bool = False,
    execute: bool = False,
    max_iterations: int = 3,
    retry_rounds: int = 2,
    thresholds: Thresholds | None = None,
    llm_config: LLMConfig | None = None,
    limit: int | None = None,
    fail_fast: bool = False,
    verbose: bool = False,
    execution_budget: int | None = None,
    llm_call_budget: int | None = None,
    token_budget: int | None = None,
    time_budget_seconds: float | None = None,
    retry_failed_rounds: int = 2,
    resume: bool = True,
    rerun_existing: bool = False,
    workers: int | None = None,
    knowledge_store: str | Path | None = None,
) -> dict[str, object]:
    configure_knowledge_store_path(knowledge_store)
    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    if resume and not rerun_existing and limit is None:
        completed = _load_completed_dataset(
            output_root=output_root,
            source_root=source_root,
            execute=execute,
            language=language,
            pattern=pattern,
            recursive=recursive,
        )
        if completed is not None:
            return completed
    llm_config = llm_config or LLMConfig.from_env()
    thresholds = thresholds or Thresholds()
    language = _normalize_language(language)
    effective_pattern = _effective_pattern(source_root, pattern=pattern, recursive=recursive, language=language)
    effective_recursive = recursive or effective_pattern.lower().endswith(".java")
    sources, skipped_sources, discovered_count = _discover_sources(source_root, pattern=effective_pattern, recursive=effective_recursive, language=language)
    if language == "auto" and sources:
        suffixes = {path.suffix.lower() for path in sources}
        if suffixes in ({".java"}, {".py"}):
            language = "java" if suffixes == {".java"} else "python"
    official_tasks = len(sources)
    if limit is not None:
        sources = sources[: max(0, limit)]
    humaneval_java_mode = _is_humaneval_java_class_dataset(source_root, sources, language)
    if humaneval_java_mode:
        _recover_round_knowledge(output_root)
    worker_count = _effective_worker_count(language, sources, workers, fail_fast)
    knowledge_round_history: list[dict[str, object]] = []

    started_at = _utc_now()
    previous_batch = _read_json_object(output_root / "batch_summary.json") if resume and not rerun_existing else {}
    accumulated_before_seconds = float(
        previous_batch.get("cumulative_duration_seconds")
        or previous_batch.get("duration_seconds")
        or 0.0
    )
    first_started_at = str(previous_batch.get("first_started_at") or previous_batch.get("started_at") or started_at)
    batch_runtime_path = output_root / "batch_runtime.json"
    _write_batch_runtime(
        batch_runtime_path,
        status="running",
        language=language,
        task_workers=worker_count,
        selected_tasks=len(sources),
        resumed_tasks=0,
        started_at=started_at,
        accumulated_before_seconds=accumulated_before_seconds,
        first_started_at=first_started_at,
        java_parallel=humaneval_java_mode and worker_count > 1,
    )
    result_slots: list[BatchItemResult | None] = [None] * len(sources)
    previous_results = previous_batch.get("results") if isinstance(previous_batch.get("results"), list) else []
    previous_by_source = {
        str(Path(str(item.get("source"))).resolve()).casefold(): item
        for item in previous_results
        if isinstance(item, dict) and item.get("source")
    }
    retry_history = [
        dict(item)
        for item in (previous_batch.get("retry_history") if isinstance(previous_batch.get("retry_history"), list) else [])
        if isinstance(item, dict)
    ]
    resumed_count = 0
    pending: list[tuple[int, Path, Path]] = []
    for index, source_path in enumerate(sources, start=1):
        case_dir = output_root / _case_name(source_root, source_path)
        if resume and not rerun_existing:
            existing = _resume_existing_item(
                source_path=source_path,
                case_dir=case_dir,
                execute=execute,
                batch_item=previous_by_source.get(str(source_path.resolve()).casefold()),
            )
            if existing is not None:
                resumed_count += 1
                result_slots[index - 1] = _prepare_round_item(existing, experiment_round=max(1, existing.experiment_round), execute=execute)
                if verbose:
                    print(f"[{index}/{len(sources)}] resumed {source_path.name}: {existing.status}")
                continue
        pending.append((index - 1, source_path, case_dir))

    round_knowledge = _prepare_round_knowledge_isolation(
        output_root,
        experiment_round=1,
        enabled=bool(execute and humaneval_java_mode and pending),
    )

    _write_batch_runtime(
        batch_runtime_path,
        status="running",
        language=language,
        task_workers=worker_count,
        selected_tasks=len(sources),
        resumed_tasks=resumed_count,
        started_at=started_at,
        accumulated_before_seconds=accumulated_before_seconds,
        first_started_at=first_started_at,
        java_parallel=humaneval_java_mode and worker_count > 1,
    )

    def run_pending(slot: int, source_path: Path, case_dir: Path) -> tuple[int, BatchItemResult]:
        return slot, _run_initial_source(
            source_path=source_path,
            case_dir=case_dir,
            execute=execute,
            thresholds=thresholds,
            max_iterations=max_iterations,
            retry_rounds=retry_rounds,
            llm_config=llm_config,
            execution_budget=execution_budget,
            llm_call_budget=llm_call_budget,
            token_budget=token_budget,
            time_budget_seconds=time_budget_seconds,
            resume=resume,
            rerun_existing=rerun_existing,
            knowledge_isolation=round_knowledge,
        )

    if worker_count == 1:
        for position, (slot, source_path, case_dir) in enumerate(pending, start=1):
            if verbose:
                print(f"[{position}/{len(pending)}] {source_path.name}")
            _, item = run_pending(slot, source_path, case_dir)
            result_slots[slot] = item
            if fail_fast and item.final_grade in {"D", "E"}:
                break
    elif pending:
        thread_prefix = "tsg-humanevaljava" if language == "java" else "tsg-python"
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=thread_prefix) as pool:
            future_map = {
                pool.submit(run_pending, slot, source_path, case_dir): (slot, source_path)
                for slot, source_path, case_dir in pending
            }
            for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                slot, source_path = future_map[future]
                _, item = future.result()
                result_slots[slot] = item
                if verbose:
                    print(f"[{completed}/{len(pending)}] completed {source_path.name}: {item.final_grade}")

    results = [item for item in result_slots if item is not None]
    round_knowledge_evidence = _merge_round_knowledge(round_knowledge, [case_dir for _, _, case_dir in pending])
    if round_knowledge_evidence.get("enabled"):
        knowledge_round_history.append(round_knowledge_evidence)

    batch_summary_path = output_root / "batch_summary.json"
    summary = _build_batch_summary(
        source_root=source_root,
        output_root=output_root,
        pattern=pattern,
        effective_pattern=effective_pattern,
        language=language,
        recursive=effective_recursive,
        execute=execute,
        llm_config=llm_config,
        started_at=started_at,
        discovered_count=discovered_count,
        official_tasks=official_tasks,
        selected_count=len(sources),
        skipped_sources=skipped_sources,
        results=results,
        retry_history=retry_history,
        resumed_count=resumed_count,
        worker_count=worker_count,
        accumulated_before_seconds=accumulated_before_seconds,
        first_started_at=first_started_at,
        knowledge_round_history=knowledge_round_history,
    )
    _write_batch_summary(batch_summary_path, summary)
    _append_batch_experience(summary, output_root, llm_config, batch_summary_path)

    if execute and retry_failed_rounds > 0 and not fail_fast:
        max_experiment_round = min(3, max(1, int(retry_failed_rounds) + 1))
        for experiment_round in range(2, max_experiment_round + 1):
            retry_indexes = [index for index, item in enumerate(results) if _should_enter_experiment_round(item, experiment_round)]
            if not retry_indexes:
                continue
            round_knowledge = _prepare_round_knowledge_isolation(
                output_root,
                experiment_round=experiment_round,
                enabled=humaneval_java_mode,
            )
            retry_record = next(
                (item for item in retry_history if int(item.get("round", 0) or 0) == experiment_round),
                None,
            )
            if retry_record is None:
                retry_record = {"round": experiment_round, "attempted": 0, "items": []}
                retry_history.append(retry_record)
            existing_retry_items = retry_record.get("items") if isinstance(retry_record.get("items"), list) else []
            retry_record["items"] = existing_retry_items
            retry_record["attempted"] = len(existing_retry_items) + len(retry_indexes)
            retry_record["status"] = "running"
            def run_followup(result_index: int) -> dict[str, object]:
                previous = results[result_index]
                source_path = Path(previous.source).resolve()
                case_dir = output_root / _case_name(source_root, source_path)
                round_strategy = _round_strategy(previous, experiment_round)
                timing_token = _begin_task_timing(
                    case_dir,
                    source_path,
                    experiment_round=experiment_round,
                    strategy=round_strategy,
                )
                _write_round_progress(
                    case_dir,
                    source_path=source_path,
                    experiment_round=experiment_round,
                    status="running",
                    strategy=round_strategy,
                    previous=previous,
                )
                artifact_snapshot = _snapshot_best_artifacts(case_dir)
                _write_round_backup(case_dir, experiment_round, artifact_snapshot)
                try:
                    retry_item = _generate_and_execute(
                        source_path=source_path,
                        case_dir=case_dir,
                        thresholds=thresholds,
                        max_iterations=max_iterations,
                        retry_rounds=retry_rounds,
                        llm_config=llm_config,
                        execution_budget=execution_budget,
                        llm_call_budget=llm_call_budget,
                        token_budget=token_budget,
                        time_budget_seconds=time_budget_seconds,
                        resume=False,
                        rerun_existing=True,
                        experiment_round=experiment_round,
                        followup_strategy=round_strategy,
                        baseline_report=_followup_baseline(previous),
                        existing_test_path=previous.test_path if previous.test_path and Path(previous.test_path).is_file() else None,
                        previous_grade=previous.final_grade,
                        experience_read_path=round_knowledge.base_path if round_knowledge else None,
                        experience_journal_dir=round_knowledge.journal_dir(case_dir) if round_knowledge else None,
                    )
                except Exception as exc:
                    error_text = f"{exc.__class__.__name__}: {exc}\n{traceback.format_exc(limit=8)}"
                    failure_artifacts = _write_failed_task_artifacts(
                        source_path=source_path,
                        case_dir=case_dir,
                        error=error_text,
                        execute=True,
                        llm_config=llm_config,
                    )
                    retry_item = BatchItemResult(
                        source=str(source_path),
                        output_dir=str(case_dir),
                        status="generation_failed",
                        summary_path=failure_artifacts.get("summary_path"),
                        visualization=failure_artifacts.get("visualization"),
                        error=error_text,
                    )
                retry_item = _prepare_and_record_round_item(
                    retry_item,
                    experiment_round=experiment_round,
                    execute=True,
                    llm_config=llm_config,
                )
                retry_item.round_strategy = round_strategy
                timing = _finish_task_timing(case_dir, timing_token, retry_item.status)
                _apply_task_timing(retry_item, timing)
                _persist_round_metadata(retry_item)
                return {
                    "result_index": result_index,
                    "previous": previous,
                    "source_path": source_path,
                    "case_dir": case_dir,
                    "round_strategy": round_strategy,
                    "artifact_snapshot": artifact_snapshot,
                    "retry_item": retry_item,
                }

            if worker_count == 1:
                followup_results = []
                for position, result_index in enumerate(retry_indexes, start=1):
                    if verbose:
                        print(f"[R{experiment_round} {position}/{len(retry_indexes)}] {Path(results[result_index].source).name}")
                    followup_results.append(run_followup(result_index))
            else:
                followup_results = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"tsg-r{experiment_round}") as pool:
                    future_map = {pool.submit(run_followup, result_index): result_index for result_index in retry_indexes}
                    for position, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
                        outcome = future.result()
                        followup_results.append(outcome)
                        if verbose:
                            print(f"[R{experiment_round} {position}/{len(retry_indexes)}] completed {Path(str(outcome['source_path'])).name}")

            round_knowledge_evidence = _merge_round_knowledge(
                round_knowledge,
                [Path(str(outcome["case_dir"])) for outcome in followup_results],
            )
            if round_knowledge_evidence.get("enabled"):
                knowledge_round_history.append(round_knowledge_evidence)

            for outcome in followup_results:
                result_index = int(outcome["result_index"])
                previous = outcome["previous"]
                source_path = outcome["source_path"]
                case_dir = outcome["case_dir"]
                round_strategy = str(outcome["round_strategy"])
                artifact_snapshot = outcome["artifact_snapshot"]
                retry_item = outcome["retry_item"]
                before_status = previous.status
                before_grade = previous.final_grade
                selected, improved = _select_best_round_result(previous, retry_item, experiment_round)
                selected.round_strategy = round_strategy
                if not improved:
                    _restore_best_artifacts(case_dir, artifact_snapshot)
                reconcile_round(case_dir, selected.final_report, experiment_round, improved, selected.duration_seconds)
                results[result_index] = selected
                _persist_round_metadata(selected)
                _write_round_progress(
                    case_dir,
                    source_path=source_path,
                    experiment_round=experiment_round,
                    status="completed",
                    strategy=round_strategy,
                    previous=previous,
                    selected=selected,
                    improved=improved,
                )
                retry_record["items"].append(
                    {
                        "source": str(source_path),
                        "before_status": before_status,
                        "after_status": retry_item.status,
                        "before_grade": before_grade,
                        "attempt_grade": retry_item.final_grade,
                        "final_grade": selected.final_grade,
                        "improved": improved,
                        "recovered": before_grade not in VALID_GRADES and selected.final_grade in VALID_GRADES,
                        "strategy": round_strategy,
                    }
                )
                _clear_round_backup(case_dir, experiment_round)
                live_summary = _build_batch_summary(
                    source_root=source_root,
                    output_root=output_root,
                    pattern=pattern,
                    effective_pattern=effective_pattern,
                    language=language,
                    recursive=effective_recursive,
                    execute=execute,
                    llm_config=llm_config,
                    started_at=started_at,
                    discovered_count=discovered_count,
                    official_tasks=official_tasks,
                    selected_count=len(sources),
                    skipped_sources=skipped_sources,
                    results=results,
                    retry_history=retry_history,
                    resumed_count=resumed_count,
                    worker_count=worker_count,
                    accumulated_before_seconds=accumulated_before_seconds,
                    first_started_at=first_started_at,
                    knowledge_round_history=knowledge_round_history,
                )
                _write_batch_summary(batch_summary_path, live_summary)
            retry_record["status"] = "completed"
            summary = _build_batch_summary(
                source_root=source_root,
                output_root=output_root,
                pattern=pattern,
                effective_pattern=effective_pattern,
                language=language,
                recursive=effective_recursive,
                execute=execute,
                llm_config=llm_config,
                started_at=started_at,
                discovered_count=discovered_count,
                official_tasks=official_tasks,
                selected_count=len(sources),
                skipped_sources=skipped_sources,
                results=results,
                retry_history=retry_history,
                resumed_count=resumed_count,
                worker_count=worker_count,
                accumulated_before_seconds=accumulated_before_seconds,
                first_started_at=first_started_at,
                knowledge_round_history=knowledge_round_history,
            )
            _write_batch_summary(batch_summary_path, summary)
            _append_batch_experience(summary, output_root, llm_config, batch_summary_path)

    experience = AgentExperienceMemory(resolve_knowledge_store_path())
    for item in results:
        if item.final_grade in {"D", "E"}:
            _write_final_failure_report(item, experience)
    summary = _build_batch_summary(
        source_root=source_root,
        output_root=output_root,
        pattern=pattern,
        effective_pattern=effective_pattern,
        language=language,
        recursive=effective_recursive,
        execute=execute,
        llm_config=llm_config,
        started_at=started_at,
        discovered_count=discovered_count,
        official_tasks=official_tasks,
        selected_count=len(sources),
        skipped_sources=skipped_sources,
        results=results,
        retry_history=retry_history,
        resumed_count=resumed_count,
        worker_count=worker_count,
        accumulated_before_seconds=accumulated_before_seconds,
        first_started_at=first_started_at,
        knowledge_round_history=knowledge_round_history,
    )
    _write_batch_summary(batch_summary_path, summary)
    _append_batch_experience(summary, output_root, llm_config, batch_summary_path)
    try:
        from visualization.knowledge_library_visualizer import TestKnowledgeGuidanceLibraryVisualizer

        summary["test_knowledge_guidance_library"] = TestKnowledgeGuidanceLibraryVisualizer().render(experience)
        _write_batch_summary(batch_summary_path, summary)
    except Exception as exc:
        summary["test_knowledge_guidance_library"] = {"status": "visualization_failed", "error": f"{exc.__class__.__name__}: {exc}"}
        _write_batch_summary(batch_summary_path, summary)
    try:
        from visualization.graph_visualizer import StateFlowGraphVisualizer

        batch_report = StateFlowGraphVisualizer().render_batch_summary(public_result(summary), output_root)
        summary["batch_report"] = batch_report
        batch_summary_path.write_text(
            json.dumps(public_result(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        summary["batch_report"] = {
            "status": "visualization_failed",
            "batch_report_html": str(report_path.resolve()),
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        batch_summary_path.write_text(
            json.dumps(public_result(summary), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    summary["dataset_completion"] = _write_dataset_completion(output_root, summary, limit=limit, requested_recursive=recursive)
    _write_batch_summary(batch_summary_path, summary)
    _write_batch_runtime(
        batch_runtime_path,
        status="completed",
        language=language,
        task_workers=worker_count,
        selected_tasks=len(sources),
        resumed_tasks=resumed_count,
        started_at=started_at,
        finished_at=str(summary.get("finished_at") or _utc_now()),
        accumulated_before_seconds=accumulated_before_seconds,
        first_started_at=first_started_at,
        java_parallel=humaneval_java_mode and worker_count > 1,
    )
    return public_result(summary)


def _build_batch_summary(
    source_root: Path,
    output_root: Path,
    pattern: str,
    effective_pattern: str,
    language: str,
    recursive: bool,
    execute: bool,
    llm_config: LLMConfig,
    started_at: str,
    discovered_count: int,
    official_tasks: int,
    selected_count: int,
    skipped_sources: list[dict[str, str]],
    results: list[BatchItemResult],
    retry_history: list[dict[str, object]],
    resumed_count: int = 0,
    worker_count: int = 1,
    accumulated_before_seconds: float = 0.0,
    first_started_at: str | None = None,
    knowledge_round_history: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    finished_at = _utc_now()
    session_duration = _duration_seconds(started_at, finished_at)
    cumulative_duration = round(max(0.0, accumulated_before_seconds) + session_duration, 4)
    for item in results:
        if not item.grade_history:
            _prepare_round_item(item, experiment_round=max(1, int(item.experiment_round or 1)), execute=execute)
    # Batch learning consumes the full in-memory reports. Public projection
    # happens only when writing or returning the summary.
    result_dicts = [
        {**vars(item), "grade_label": round_label(item.final_grade, item.experiment_round)}
        for item in results
    ]
    outcome = aggregate_results(result_dicts)
    retry_summary = _retry_summary(retry_history, results)
    humaneval_java_mode = language == "java" and source_root.name.casefold() == "humanevaljava"
    return {
        "source_dir": str(source_root),
        "output_dir": str(output_root),
        "pattern": pattern,
        "effective_pattern": effective_pattern,
        "language": language,
        "recursive": recursive,
        "mode": "execute" if execute else "generate-only",
        "execution_policy": {
            "task_workers": worker_count,
            "python_parallel": language == "python" and worker_count > 1,
            "java_parallel": humaneval_java_mode and worker_count > 1,
            "java_workspace_isolation": "per_task" if humaneval_java_mode else "unchanged",
            "repair_rounds": "R1 establishes all baselines; R2 recovers D/E only; R3 performs focused optimization for every C result; each round allows at most three internal repair or optimization attempts with early stopping",
            "mutation_profiles": "screening evidence is reused by the final evaluation; repair candidates use at most 6 screening mutants",
            "knowledge_visibility": "frozen_per_formal_round" if humaneval_java_mode else "unchanged",
        },
        "llm": llm_config.public_dict(),
        "knowledge_round_isolation": list(knowledge_round_history or []),
        "started_at": first_started_at or started_at,
        "first_started_at": first_started_at or started_at,
        "current_session_started_at": started_at,
        "finished_at": finished_at,
        "session_duration_seconds": session_duration,
        "duration_seconds": cumulative_duration,
        "cumulative_duration_seconds": cumulative_duration,
        "discovered_files": discovered_count,
        "official_tasks": official_tasks,
        "dataset_integrity": {
            "reference_expected_tasks": 164 if humaneval_java_mode else None,
            "discovered_testable_tasks": official_tasks,
            "complete": official_tasks == 164 if humaneval_java_mode else None,
            "note": "HumanEvalJava参考任务数为164；扫描数不等于官方完整性校验。" if humaneval_java_mode else "",
        },
        "selected_tasks": selected_count,
        "resumed_existing": resumed_count,
        "skipped": len(skipped_sources),
        "skipped_reason_counts": _skip_reason_counts(skipped_sources),
        "skipped_examples": skipped_sources[:10],
        "skipped_files": skipped_sources,
        "retry_policy": {
            "enabled": bool(execute),
            "experiment_rounds": 3 if execute else 1,
            "strategy": "R1 runs all tasks. R2 handles only D/E using failure-oriented knowledge and recovery generation. R3 then keeps every C-grade passing suite and performs knowledge/LLM-guided focused optimization, with at most three internal attempts and early stopping. No R4 is allowed.",
        },
        "retry_history": retry_history,
        "retry_summary": retry_summary,
        "experiment_round_summary": retry_summary,
        "total": len(results),
        "generated": outcome["valid_results"],
        "valid": outcome["valid_results"],
        "quality_acceptable": outcome["grade_counts"]["B"],
        "quality_high": outcome["grade_counts"]["A"],
        "weak_success": outcome["grade_counts"]["C"],
        "plateau_reached": sum(1 for item in results if item.status == "plateau_reached"),
        "incomplete": outcome["not_completed_execution"],
        "succeeded": outcome["valid_results"],
        "failed": outcome["not_completed_execution"],
        "invalid": outcome["invalid_results"],
        "unfinished": outcome["unfinished_results"],
        "outcome_summary": outcome,
        "statistics_definition": {
            "balance": "total_tasks = A + B + C + D + E",
            "completed_execution": "A + B + C",
            "not_completed_execution": "D + E",
            "round_marker": "A·R1, B·R2 and similar labels show the last experiment round attempted",
        },
        "failure_analysis": _batch_failure_analysis(results),
        "weak_success_analysis": _weak_success_analysis(results),
        "quality_overview": _batch_quality_overview(results),
        "paper_metrics": batch_paper_metrics(results, selected_count),
        "results": result_dicts,
    }


def _write_batch_summary(path: Path, summary: dict[str, object]) -> None:
    _write_json_atomic(path, public_result(summary))


def _write_batch_runtime(
    path: Path,
    *,
    status: str,
    language: str,
    task_workers: int,
    selected_tasks: int,
    resumed_tasks: int,
    started_at: str,
    finished_at: str | None = None,
    accumulated_before_seconds: float = 0.0,
    first_started_at: str | None = None,
    java_parallel: bool = False,
) -> None:
    try:
        llm_concurrency = max(1, int(os.getenv("TSG_LLM_CONCURRENCY", "2") or 2))
    except ValueError:
        llm_concurrency = 2
    interval_end = finished_at or _utc_now()
    session_duration = _duration_seconds(started_at, interval_end)
    cumulative_duration = round(max(0.0, accumulated_before_seconds) + session_duration, 4)
    payload = {
        "status": status,
        "language": language,
        "task_workers": task_workers,
        "python_parallel": language == "python" and task_workers > 1,
        "java_parallel": bool(java_parallel),
        "llm_concurrency": llm_concurrency,
        "selected_tasks": selected_tasks,
        "resumed_tasks": resumed_tasks,
        "started_at": first_started_at or started_at,
        "first_started_at": first_started_at or started_at,
        "active_session_started_at": started_at if status == "running" else None,
        "accumulated_duration_before_session": round(max(0.0, accumulated_before_seconds), 4),
        "session_duration_seconds": session_duration,
        "duration_seconds": cumulative_duration,
        "cumulative_duration_seconds": cumulative_duration,
        "finished_at": finished_at,
        "knowledge_store": str(resolve_knowledge_store_path()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, payload)


def _append_batch_experience(
    summary: dict[str, object],
    output_root: Path,
    llm_config: LLMConfig,
    batch_summary_path: Path,
) -> None:
    try:
        experience = AgentExperienceMemory(resolve_knowledge_store_path())
        summary["batch_experience"] = experience.append_batch_summary(summary, model=llm_config.model)
    except Exception as exc:
        summary["batch_experience"] = {"status": "experience_update_failed", "error": f"{exc.__class__.__name__}: {exc}"}
    _write_batch_summary(batch_summary_path, summary)


def _should_retry_item(item: BatchItemResult) -> bool:
    return _should_enter_experiment_round(item, 2)


def _should_enter_experiment_round(item: BatchItemResult, experiment_round: int) -> bool:
    if int(item.experiment_round or 1) >= experiment_round:
        return False
    if experiment_round == 2:
        return item.final_grade in SECOND_ROUND_GRADES
    if experiment_round == 3:
        return item.final_grade in THIRD_ROUND_GRADES
    return False


def _round_strategy(item: BatchItemResult, experiment_round: int) -> str:
    if experiment_round == 3 and item.final_grade == "C" and item.test_path and Path(item.test_path).is_file():
        return "optimize_existing"
    if item.final_grade == "D":
        return "regenerate_after_failure"
    return "restart_incomplete"


def _followup_baseline(item: BatchItemResult) -> dict[str, object] | None:
    report = dict(item.final_report) if isinstance(item.final_report, dict) else {}
    if item.final_grade in {"D", "E"}:
        diagnostic = item.round_failure_diagnostic or {}
        category = diagnostic.get("failure_category") or outcome_failure_category(item.to_dict())
        reason = diagnostic.get("failure_reason") or item.failure_reason
        solutions = list(item.previous_solutions or diagnostic.get("suggested_solutions", []) or [default_solution(str(category))])
        report.setdefault("failure_category", category)
        report.setdefault("failure_detail", reason)
        report.setdefault("quality_status", item.status)
        report.setdefault("pytest_passed", False)
        report["failure_recovery"] = {
            "category": category,
            "reason": reason,
            "stage": diagnostic.get("failure_stage"),
            "error": diagnostic.get("error") or item.error,
            "attempted_measures": list(diagnostic.get("attempted_measures", []) or []),
            "required_solutions": solutions,
            "failed_tests": list(report.get("failed_tests", []) or []),
        }
    return report or None


def _write_round_progress(
    case_dir: Path,
    *,
    source_path: Path,
    experiment_round: int,
    status: str,
    strategy: str,
    previous: BatchItemResult,
    selected: BatchItemResult | None = None,
    improved: bool | None = None,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(source_path.resolve()),
        "experiment_round": experiment_round,
        "status": status,
        "strategy": strategy,
        "previous_grade": previous.final_grade,
        "previous_status": previous.status,
        "final_grade": selected.final_grade if selected else previous.final_grade,
        "result_status": selected.status if selected else None,
        "improved": improved,
        "updated_at": _utc_now(),
    }
    (case_dir / "experiment_round_progress.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _retry_summary(
    retry_history: list[dict[str, object]],
    results: list[BatchItemResult] | None = None,
) -> dict[str, object]:
    attempts = 0
    recovered = 0
    improved = 0
    followup_tasks: set[str] = set()
    by_round: dict[str, dict[str, int]] = {}
    strategy_counts: dict[str, int] = {}
    normalized = [dict(record) for record in retry_history]
    known = {
        (int(record.get("round", 0) or 0), str(item.get("source") or ""))
        for record in normalized
        for item in (record.get("items", []) if isinstance(record.get("items"), list) else [])
        if isinstance(item, dict)
    }
    for result in results or []:
        history = [entry for entry in (result.grade_history or []) if isinstance(entry, dict)]
        history.sort(key=lambda entry: int(entry.get("round", 0) or 0))
        for index, entry in enumerate(history):
            round_number = int(entry.get("round", 0) or 0)
            key = (round_number, result.source)
            if round_number <= 1 or key in known:
                continue
            previous_grade = str(history[index - 1].get("grade") or "E") if index else "E"
            attempt_grade = str(entry.get("grade") or result.final_grade)
            record = next((item for item in normalized if int(item.get("round", 0) or 0) == round_number), None)
            if record is None:
                record = {"round": round_number, "status": "reconstructed", "items": []}
                normalized.append(record)
            items = record.get("items") if isinstance(record.get("items"), list) else []
            items.append(
                {
                    "source": result.source,
                    "before_grade": previous_grade,
                    "attempt_grade": attempt_grade,
                    "final_grade": result.final_grade if round_number == result.experiment_round else attempt_grade,
                    "improved": _grade_rank(attempt_grade) > _grade_rank(previous_grade),
                    "recovered": previous_grade in {"D", "E"} and attempt_grade in VALID_GRADES,
                    "strategy": "optimize_existing" if round_number == 3 and previous_grade == "C" else "regenerate_after_failure" if previous_grade == "D" else "restart_incomplete",
                    "reconstructed": True,
                }
            )
            record["items"] = items
            known.add(key)
    for record in sorted(normalized, key=lambda item: int(item.get("round", 0) or 0)):
        items = record.get("items", []) if isinstance(record.get("items"), list) else []
        round_name = f"R{int(record.get('round', 0) or 0)}"
        attempts += len(items)
        followup_tasks.update(str(item.get("source")) for item in items if isinstance(item, dict))
        recovered += sum(1 for item in items if isinstance(item, dict) and item.get("recovered"))
        improved += sum(1 for item in items if isinstance(item, dict) and item.get("improved"))
        for item in items:
            if not isinstance(item, dict):
                continue
            strategy = str(item.get("strategy") or "unknown")
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        by_round[round_name] = {
            "tasks": len(items),
            "recovered": sum(1 for item in items if isinstance(item, dict) and item.get("recovered")),
            "improved": sum(1 for item in items if isinstance(item, dict) and item.get("improved")),
        }
    return {
        "rounds_after_initial": len(normalized),
        "followup_task_count": len(followup_tasks),
        "attempts": attempts,
        "recovered": recovered,
        "improved": improved,
        "by_round": by_round,
        "strategy_counts": strategy_counts,
    }


def _grade_rank(grade: str) -> int:
    return {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}.get(str(grade), 0)


def _prepare_round_item(item: BatchItemResult, experiment_round: int, execute: bool) -> BatchItemResult:
    report = item.final_report if isinstance(item.final_report, dict) else {}
    if report.get("pytest_passed") and str(report.get("quality_status") or "") in VALID_STATUSES:
        item.status = str(report["quality_status"])
    item.final_grade = grade_for(item.status, item.final_report, execute=execute)
    item.experiment_round = experiment_round
    item.round_failure_diagnostic = _round_failure_diagnostic(item) if item.final_grade in {"D", "E"} else None
    item.failure_reason = (
        str((item.round_failure_diagnostic or {}).get("failure_reason") or outcome_failure_reason(item.to_dict()))
        if item.final_grade in {"D", "E"}
        else None
    )
    knowledge = list(item.knowledge_history or [])
    if not knowledge:
        knowledge.append(
            {
                "round": experiment_round,
                "selected_ids": [],
                "status": "retrieved_without_selected_match",
            }
        )
    else:
        for entry in knowledge:
            entry.setdefault("round", experiment_round)
    item.knowledge_history = knowledge
    history = [
        entry
        for entry in (item.grade_history or [])
        if isinstance(entry, dict) and int(entry.get("round", 0) or 0) != experiment_round
    ]
    history.append(
        {
            "round": experiment_round,
            "grade": item.final_grade,
            "status": item.status,
            "label": round_label(item.final_grade, experiment_round),
            "failure_reason": item.failure_reason,
            "metrics": _round_metrics(item.final_report),
            "knowledge_selected_ids": [
                selected
                for context in knowledge
                for selected in (context.get("selected_ids", []) if isinstance(context, dict) else [])
            ],
        }
    )
    item.grade_history = sorted(history, key=lambda entry: int(entry.get("round", 0) or 0))
    return item


def _prepare_and_record_round_item(
    item: BatchItemResult,
    *,
    experiment_round: int,
    execute: bool,
    llm_config: LLMConfig,
) -> BatchItemResult:
    item = _prepare_round_item(item, experiment_round=experiment_round, execute=execute)
    if execute and item.final_grade in {"D", "E"}:
        _write_round_failure_report(item, llm_config)
    _persist_round_metadata(item)
    return item


def _round_failure_diagnostic(item: BatchItemResult) -> dict[str, object]:
    summary: dict[str, object] = {}
    if item.summary_path:
        try:
            loaded = json.loads(Path(item.summary_path).read_text(encoding="utf-8"))
            summary = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            summary = {}
    evidence = summary.get("failure_evidence") if isinstance(summary.get("failure_evidence"), dict) else {}
    validation = evidence.get("candidate_validation") if isinstance(evidence.get("candidate_validation"), dict) else {}
    validation_output = str(validation.get("pytest_output") or "").strip()
    stop_reason = str(summary.get("stop_reason") or item.status or "").strip()
    error = str(
        item.error
        or summary.get("last_error")
        or summary.get("error")
        or validation_output
        or ""
    ).strip()
    detail_by_status = {
        "generation_failed": "测试生成阶段没有形成通过候选验证的可执行测试文件",
        "test_state_invalid": "测试状态智能体完成自修复后，测试状态仍不满足后续生成要求",
        "not_started": "任务未进入测试生成与执行阶段",
        "interrupted": "任务在生成完整执行证据前被中断",
    }
    detail = error or detail_by_status.get(stop_reason) or detail_by_status.get(item.status) or stop_reason
    category_payload = {
        **item.to_dict(),
        "round_failure_diagnostic": None,
        "last_error": error,
        "stop_reason": stop_reason,
    }
    category = outcome_failure_category(category_payload)
    reason = outcome_failure_reason({**category_payload, "error": detail})
    stage = str(validation.get("stage") or evidence.get("stage") or "").strip()
    if not stage:
        stage = "source_analysis" if category == "source_analysis_failed" else "test_generation" if item.final_grade == "E" else "test_execution"
    attempted = []
    repair_attempts = int(evidence.get("state_repair_attempts", 0) or 0)
    if repair_attempts:
        attempted.append(f"测试状态智能体已执行{repair_attempts}次状态自修复")
    if validation:
        attempted.append(f"测试生成智能体已执行候选验证（阶段：{validation.get('stage') or 'unknown'}）")
    if evidence.get("generation_strategy"):
        attempted.append(f"测试生成策略：{evidence.get('generation_strategy')}")
    return {
        "failure_category": category,
        "failure_reason": reason,
        "failure_stage": stage,
        "error": error or detail,
        "stop_reason": stop_reason,
        "attempted_measures": attempted,
        "suggested_solutions": [default_solution(category)],
        "evidence": evidence,
    }


def _write_round_failure_report(item: BatchItemResult, llm_config: LLMConfig) -> None:
    diagnostic = dict(item.round_failure_diagnostic or _round_failure_diagnostic(item))
    category = str(diagnostic.get("failure_category") or "unknown")
    reason = str(diagnostic.get("failure_reason") or outcome_failure_reason(item.to_dict()))
    experience = AgentExperienceMemory(resolve_knowledge_store_path())
    learned = experience.append_task_failure(
        source=item.source,
        experiment_round=item.experiment_round,
        category=category,
        reason=reason,
        solution=default_solution(category),
        stage=str(diagnostic.get("failure_stage") or "") or None,
        error=str(diagnostic.get("error") or item.error or "") or None,
        model=llm_config.model,
    )
    guidance = experience.failure_guidance(item.source, category)
    item.failure_reason = reason
    item.previous_solutions = list(guidance.get("previous_solutions", []) or [])
    payload = {
        "source": item.source,
        "final_grade": item.final_grade,
        "grade_label": round_label(item.final_grade, item.experiment_round),
        "experiment_round": item.experiment_round,
        **diagnostic,
        "previous_solutions": item.previous_solutions,
        "grade_history": item.grade_history or [],
        "knowledge_history": item.knowledge_history or [],
        "matched_experience_ids": guidance.get("matched_experience_ids", []),
        "knowledge_record_id": (learned or {}).get("experience_id"),
        "generated_at": _utc_now(),
    }
    item.round_failure_diagnostic = payload
    case_dir = Path(item.output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    json_path = case_dir / f"task_failure_report_R{item.experiment_round}.json"
    html_path = case_dir / f"task_failure_report_R{item.experiment_round}.html"
    payload = public_result(payload)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(_final_failure_html(payload), encoding="utf-8")
    item.failure_report = str(html_path.resolve())
    summary_path = Path(item.summary_path) if item.summary_path else case_dir / "stateflow_summary.json"
    try:
        task_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        task_summary = {}
    task_summary["round_failure_diagnostic"] = payload
    history = task_summary.get("round_failure_history") if isinstance(task_summary.get("round_failure_history"), list) else []
    history = [entry for entry in history if isinstance(entry, dict) and int(entry.get("experiment_round", 0) or 0) != item.experiment_round]
    task_summary["round_failure_history"] = [*history, payload]
    task_summary.setdefault("visualization", {})
    if isinstance(task_summary["visualization"], dict):
        task_summary["visualization"]["failure_report_html"] = item.failure_report
    summary_path.write_text(json.dumps(public_result(task_summary), indent=2, ensure_ascii=False), encoding="utf-8")


def _round_metrics(report: dict[str, object] | None) -> dict[str, object]:
    report = report if isinstance(report, dict) else {}
    return {
        key: report.get(key)
        for key in ("pytest_passed", "line_coverage", "branch_coverage", "ae", "mutation_score")
    }


def _select_best_round_result(
    previous: BatchItemResult,
    candidate: BatchItemResult,
    experiment_round: int,
) -> tuple[BatchItemResult, bool]:
    previous_key = _item_quality_key(previous)
    candidate_key = _item_quality_key(candidate)
    improved = candidate_key > previous_key
    selected = candidate if improved else previous
    selected.experiment_round = experiment_round
    selected.grade_history = [*(previous.grade_history or []), *(candidate.grade_history or [])]
    selected.knowledge_history = [*(previous.knowledge_history or []), *(candidate.knowledge_history or [])]
    selected.started_at = candidate.started_at or previous.started_at
    selected.finished_at = candidate.finished_at or previous.finished_at
    selected.duration_seconds = max(candidate.duration_seconds, previous.duration_seconds)
    selected.attempt_count = max(candidate.attempt_count, previous.attempt_count)
    if selected.final_grade in VALID_GRADES:
        selected.failure_reason = None
        selected.previous_solutions = []
    return selected, improved


def _item_quality_key(item: BatchItemResult) -> tuple[float, ...]:
    grade_score = {"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0}.get(item.final_grade, 0.0)
    report = item.final_report if isinstance(item.final_report, dict) else {}
    return (
        grade_score,
        float(report.get("tsq", report.get("sfq", 0.0)) or 0.0),
        float(report.get("ae", 0.0) or 0.0),
        float(report.get("sfc", 0.0) or 0.0),
        float(report.get("mutation_score", 0.0) or 0.0),
        float(report.get("line_coverage", 0.0) or 0.0),
    )


def _snapshot_best_artifacts(case_dir: Path) -> dict[Path, bytes]:
    if not case_dir.is_dir():
        return {}
    selected: dict[Path, bytes] = {}
    for path in case_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in {"stateflow_summary.json", "state_flow_graph.json"} or path.suffix in {".py", ".java"}:
            try:
                selected[path] = path.read_bytes()
            except OSError:
                continue
    return selected


def _restore_best_artifacts(case_dir: Path, snapshot: dict[Path, bytes]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for path, content in snapshot.items():
        path.write_bytes(content)


def _round_backup_dir(case_dir: Path, experiment_round: int) -> Path:
    return case_dir / f".round_backup_R{max(1, int(experiment_round or 1))}"


def _write_round_backup(case_dir: Path, experiment_round: int, snapshot: dict[Path, bytes]) -> None:
    backup_dir = _round_backup_dir(case_dir, experiment_round)
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path, content in snapshot.items():
        (backup_dir / path.name).write_bytes(content)


def _clear_round_backup(case_dir: Path, experiment_round: int) -> None:
    backup_dir = _round_backup_dir(case_dir, experiment_round)
    if backup_dir.is_dir():
        shutil.rmtree(backup_dir)


def _recover_interrupted_round(case_dir: Path) -> None:
    progress_path = case_dir / "experiment_round_progress.json"
    progress = _read_json_object(progress_path)
    if str(progress.get("status") or "") != "running":
        return
    experiment_round = int(progress.get("experiment_round", 1) or 1)
    backup_dir = _round_backup_dir(case_dir, experiment_round)
    if not backup_dir.is_dir():
        return
    for backup in backup_dir.iterdir():
        if backup.is_file():
            (case_dir / backup.name).write_bytes(backup.read_bytes())
    shutil.rmtree(backup_dir)
    progress.update({"status": "interrupted_recovered", "updated_at": _utc_now()})
    progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def _persist_round_metadata(item: BatchItemResult) -> None:
    if not item.summary_path:
        return
    summary_path = Path(item.summary_path)
    summary = _read_json_object(summary_path)
    if not summary:
        return
    measurement = _read_json_object(summary_path.parent / "paper_metrics" / "summary.json")
    if measurement:
        summary["paper_metrics"] = measurement
    summary.update(
        {
            "experiment_round": item.experiment_round,
            "final_grade": item.final_grade,
            "grade_label": round_label(item.final_grade, item.experiment_round),
            "grade_history": item.grade_history or [],
            "knowledge_history": item.knowledge_history or [],
            "followup_strategy": item.round_strategy or summary.get("followup_strategy") or "initial_generation",
            "quality_status": item.status,
            "started_at": item.started_at or summary.get("started_at"),
            "finished_at": item.finished_at or summary.get("finished_at"),
            "duration_seconds": item.duration_seconds or summary.get("duration_seconds", 0.0),
            "cumulative_duration_seconds": item.duration_seconds or summary.get("cumulative_duration_seconds", 0.0),
            "attempt_count": item.attempt_count or summary.get("attempt_count", 0),
        }
    )
    if item.failure_reason:
        summary["failure_reason"] = item.failure_reason
    elif item.final_grade in VALID_GRADES:
        summary.pop("failure_reason", None)
    summary_path.write_text(json.dumps(public_result(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _migrate_public_result(path: Path, value: dict[str, object]) -> dict[str, object]:
    cleaned = public_result(value)
    if cleaned != value:
        _write_json_atomic(path, cleaned)
    return cleaned


def _dataset_completion_path(output_root: Path) -> Path:
    return output_root / "dataset_completion.json"


def _load_completed_dataset(
    output_root: Path,
    source_root: Path,
    execute: bool,
    language: str,
    pattern: str,
    recursive: bool,
) -> dict[str, object] | None:
    marker_path = _dataset_completion_path(output_root)
    summary_path = output_root / "batch_summary.json"
    if not marker_path.exists() or not summary_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    marker_language = str(marker.get("language") or "auto")
    language_matches = language == "auto" or marker_language == language
    requested_recursive_matches = bool(marker.get("requested_recursive", recursive)) == bool(recursive)
    current_signature = _source_tree_signature(source_root)
    if source_root.name.casefold() == "humanevaljava" and current_signature["file_count"] != 164:
        return None
    if not (
        marker.get("skip_ready") is True
        and str(Path(str(marker.get("source_dir") or "")).resolve()) == str(source_root.resolve())
        and bool(marker.get("execute")) == bool(execute)
        and language_matches
        and str(marker.get("pattern")) == str(pattern)
        and requested_recursive_matches
        and marker.get("source_signature") == current_signature["signature"]
        and int(marker.get("source_file_count", -1) or -1) == current_signature["file_count"]
    ):
        return None
    summary = _migrate_public_result(summary_path, summary)
    for item in summary.get("results", []):
        if not isinstance(item, dict):
            continue
        case_summary_path = Path(str(item.get("summary_path") or Path(str(item.get("output_dir") or "")) / "stateflow_summary.json")).resolve()
        if not case_summary_path.is_relative_to(output_root.resolve()) or not case_summary_path.is_file():
            continue
        case_summary = _read_json_object(case_summary_path)
        if case_summary:
            _migrate_public_result(case_summary_path, case_summary)
    try:
        from visualization.graph_visualizer import StateFlowGraphVisualizer

        summary["batch_report"] = StateFlowGraphVisualizer().render_batch_summary(summary, output_root)
    except Exception as exc:
        report_path = output_root / "batch_quality_report.html"
        report_path.write_text(
            "<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>批量报告</title>"
            "<body><h1>批量报告暂不可用</h1><p>请查看 batch_summary.json。</p></body></html>",
            encoding="utf-8",
        )
        summary["batch_report"] = {"status": "visualization_failed", "error": f"{exc.__class__.__name__}: {exc}"}
    _write_batch_summary(summary_path, summary)
    summary["dataset_skipped"] = True
    summary["dataset_skip_reason"] = "all eligible tasks already have complete A/B/C results"
    summary["resumed_existing"] = int(summary.get("total", 0) or 0)
    summary["session_duration_seconds"] = 0.0
    return public_result(summary)


def _write_dataset_completion(
    output_root: Path,
    summary: dict[str, object],
    limit: int | None,
    requested_recursive: bool = False,
) -> dict[str, object]:
    outcome = summary.get("outcome_summary") if isinstance(summary.get("outcome_summary"), dict) else {}
    selected = int(summary.get("selected_tasks", 0) or 0)
    official = int(summary.get("official_tasks", 0) or 0)
    skip_ready = bool(
        limit is None
        and selected == official
        and bool(outcome.get("balance_valid"))
        and int(outcome.get("not_completed_execution", 0) or 0) == 0
        and (summary.get("dataset_integrity") or {}).get("complete") is not False
    )
    source_signature = _source_tree_signature(Path(str(summary.get("source_dir") or "")))
    marker = {
        "schema_version": 1,
        "source_dir": summary.get("source_dir"),
        "output_dir": summary.get("output_dir"),
        "language": summary.get("language"),
        "execute": summary.get("mode") == "execute",
        "pattern": summary.get("pattern"),
        "recursive": summary.get("recursive"),
        "requested_recursive": requested_recursive,
        "effective_pattern": summary.get("effective_pattern"),
        "official_tasks": official,
        "selected_tasks": selected,
        "source_signature": source_signature["signature"],
        "source_file_count": source_signature["file_count"],
        "grade_counts": outcome.get("grade_counts", {}),
        "skip_ready": skip_ready,
        "reason": "all eligible tasks have A/B/C results" if skip_ready else "at least one task is missing, invalid, unfinished, or the run used a limit",
        "updated_at": _utc_now(),
    }
    marker_path = _dataset_completion_path(output_root)
    marker_path.write_text(json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8")
    marker["path"] = str(marker_path.resolve())
    return marker


def _source_tree_signature(source_root: Path) -> dict[str, object]:
    """Detect dataset changes without parsing or executing every task."""
    digest = hashlib.sha256()
    count = 0
    if source_root.is_dir():
        for path in sorted(
            (item for item in source_root.rglob("*") if item.is_file() and item.suffix.lower() in {".py", ".java"}),
            key=lambda item: item.relative_to(source_root).as_posix().casefold(),
        ):
            relative = path.relative_to(source_root).as_posix()
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8", errors="surrogatepass"))
            count += 1
    return {"signature": digest.hexdigest(), "file_count": count}


def _write_final_failure_report(item: BatchItemResult, experience: AgentExperienceMemory) -> None:
    category = outcome_failure_category(item.to_dict())
    guidance = experience.failure_guidance(item.source, category)
    item.failure_reason = item.failure_reason or outcome_failure_reason(item.to_dict())
    item.previous_solutions = list(guidance.get("previous_solutions", []) or [])
    diagnostic = dict(item.round_failure_diagnostic or {})
    case_dir = Path(item.output_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    json_path = case_dir / "task_failure_report.json"
    html_path = case_dir / "task_failure_report.html"
    payload = {
        "source": item.source,
        "final_grade": item.final_grade,
        "grade_label": round_label(item.final_grade, item.experiment_round),
        "experiment_round": item.experiment_round,
        "failure_category": category,
        "failure_reason": item.failure_reason,
        "failure_stage": diagnostic.get("failure_stage"),
        "attempted_measures": diagnostic.get("attempted_measures", []),
        "suggested_solutions": diagnostic.get("suggested_solutions", [default_solution(category)]),
        "evidence": diagnostic.get("evidence", {}),
        "previous_solutions": item.previous_solutions,
        "grade_history": item.grade_history or [],
        "knowledge_history": item.knowledge_history or [],
        "matched_experience_ids": guidance.get("matched_experience_ids", []),
        "error": diagnostic.get("error") or item.error,
        "generated_at": _utc_now(),
    }
    payload = public_result(payload)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(_final_failure_html(payload), encoding="utf-8")
    item.failure_report = str(html_path.resolve())
    summary_path = Path(item.summary_path) if item.summary_path else case_dir / "stateflow_summary.json"
    try:
        task_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        task_summary = {}
    task_summary["experiment_outcome"] = payload
    task_summary.setdefault("visualization", {})
    if isinstance(task_summary["visualization"], dict):
        task_summary["visualization"]["failure_report_html"] = item.failure_report
    summary_path.write_text(json.dumps(public_result(task_summary), indent=2, ensure_ascii=False), encoding="utf-8")


def _final_failure_html(payload: dict[str, object]) -> str:
    history_rows = "".join(
        "<tr>"
        f"<td>R{int(entry.get('round', 1) or 1)}</td>"
        f"<td>{html.escape(str(entry.get('grade') or 'E'))}</td>"
        f"<td>{html.escape(str(entry.get('status') or ''))}</td>"
        f"<td>{html.escape(str(entry.get('failure_reason') or ''))}</td></tr>"
        for entry in payload.get("grade_history", [])
        if isinstance(entry, dict)
    ) or '<tr><td colspan="4">没有轮次记录</td></tr>'
    solutions = "".join(f"<li>{html.escape(str(value))}</li>" for value in payload.get("previous_solutions", [])) or "<li>没有匹配到历史解决措施</li>"
    knowledge = "".join(
        f"<li>R{int(entry.get('round', 1) or 1)}：{html.escape(', '.join(str(value) for value in entry.get('selected_ids', []) or []) or '已检索，但没有匹配经验')}</li>"
        for entry in payload.get("knowledge_history", [])
        if isinstance(entry, dict)
    ) or "<li>没有知识检索记录</li>"
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>任务失败报告</title>
<style>body{{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:0;background:#f7f9fc;color:#172033}}main{{max-width:1100px;margin:auto;padding:28px}}section{{background:white;border:1px solid #dce2ea;border-radius:8px;padding:18px;margin-bottom:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #e7ebf0;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap;background:#111827;color:#e5e7eb;padding:14px;overflow:auto}}</style></head><body><main>
<section><h1>任务失败报告</h1><p><b>任务：</b>{html.escape(str(payload.get('source') or ''))}</p><p><b>最终结果：</b>{html.escape(str(payload.get('grade_label') or ''))}</p><p><b>具体失败原因：</b>{html.escape(str(payload.get('failure_reason') or ''))}</p></section>
<section><h2>轮次结果</h2><table><thead><tr><th>轮次</th><th>等级</th><th>状态</th><th>原因</th></tr></thead><tbody>{history_rows}</tbody></table></section>
<section><h2>之前采用或建议的解决措施</h2><ul>{solutions}</ul></section><section><h2>各轮知识检索</h2><ul>{knowledge}</ul></section>
<section><h2>原始错误</h2><pre>{html.escape(str(payload.get('error') or '没有额外错误文本'))}</pre></section></main></body></html>'''


def _find_sources(source_root: Path, pattern: str, recursive: bool, language: str = "auto") -> list[Path]:
    return _discover_sources(source_root, pattern=pattern, recursive=recursive, language=language)[0]


def _discover_sources(source_root: Path, pattern: str, recursive: bool, language: str = "auto") -> tuple[list[Path], list[dict[str, str]], int]:
    language = _normalize_language(language)
    effective_pattern = _effective_pattern(source_root, pattern=pattern, recursive=recursive, language=language)
    extensions = _source_extensions_for_pattern(effective_pattern, language=language)
    iterators: list[Iterable[Path]] = [
        source_root.rglob(f"*{extension}") if recursive else source_root.glob(f"*{extension}")
        for extension in extensions
    ]
    discovered = sorted({path.resolve() for iterator in iterators for path in iterator if path.is_file()})
    selected: list[Path] = []
    skipped: list[dict[str, str]] = []
    for path in discovered:
        if path.suffix.lower() == ".java" and _is_generated_or_non_source_java_path(source_root, path):
            skipped.append({"source": str(path), "status": "skipped", "reason": "java_non_main_source"})
        elif _is_test_or_generated_source_path(source_root, path):
            skipped.append({"source": str(path), "status": "skipped", "reason": "test_or_generated_source"})
        elif path.name == "__init__.py":
            skipped.append({"source": str(path), "status": "skipped", "reason": "package_initializer"})
        elif not _matches_source_pattern(path, effective_pattern):
            skipped.append({"source": str(path), "status": "skipped", "reason": "pattern_mismatch"})
        elif not _has_public_testable_function(path):
            skipped.append({"source": str(path), "status": "skipped", "reason": "no_public_testable_function"})
        else:
            selected.append(path)
    return sorted(selected, key=_task_sort_key), skipped, len(discovered)


def _is_generated_or_non_source_java_path(source_root: Path, path: Path) -> bool:
    if path.suffix.lower() != ".java":
        return False
    try:
        parts = path.relative_to(source_root).parts
    except ValueError:
        parts = path.parts
    lowered = [part.lower() for part in parts]
    for part in lowered[:-1]:
        if part in JAVA_IGNORED_DIR_NAMES or any(part.startswith(prefix) for prefix in JAVA_IGNORED_DIR_PREFIXES):
            return True
    for index in range(len(lowered) - 2):
        if lowered[index : index + 3] == ["src", "test", "java"]:
            return True
    return False


def _is_test_or_generated_source_path(source_root: Path, path: Path) -> bool:
    if _is_generated_or_non_source_java_path(source_root, path):
        return True
    try:
        relative_parts = [part.lower() for part in path.relative_to(source_root).parts]
    except ValueError:
        relative_parts = [part.lower() for part in path.parts]
    if any(part in JAVA_IGNORED_DIR_NAMES for part in relative_parts[:-1]):
        return True
    name = path.name.lower()
    if path.suffix.lower() == ".py":
        return name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py")
    if path.suffix.lower() == ".java":
        stem = path.stem.lower()
        return stem.endswith("test") or stem.endswith("tests") or stem.endswith("testcase")
    return False


def _has_public_testable_function(path: Path) -> bool:
    if path.suffix.lower() == ".java":
        try:
            context = ContextRewriter().rewrite(extractor_for(path).extract(path))
        except Exception:
            return False
        return any(cls.methods or cls.constructors for cls in context.classes) or bool(context.functions)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            return True
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            if any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_")
                for item in node.body
            ):
                return True
    return False


def _matches_source_pattern(path: Path, pattern: str) -> bool:
    if path.suffix.lower() == ".java":
        return fnmatch.fnmatch(path.name, pattern)
    configured_patterns = tuple(f"{prefix}*.py" for prefix in DEFAULT_TASK_PREFIXES)
    allowed_patterns = tuple(dict.fromkeys((pattern, *configured_patterns)))
    return any(fnmatch.fnmatch(path.name, item) for item in allowed_patterns)


def _source_extensions_for_pattern(pattern: str, language: str = "auto") -> tuple[str, ...]:
    language = _normalize_language(language)
    if language == "python":
        return (".py",)
    if language == "java":
        return (".java",)
    if language == "all":
        return (".py", ".java")
    lowered = pattern.lower()
    if lowered.endswith(".java"):
        return (".java",)
    if lowered in {"*", "*.*"}:
        return (".py", ".java")
    return (".py",)


def _normalize_language(language: str | None) -> str:
    value = (language or "auto").lower()
    if value not in {"auto", "python", "java", "all"}:
        raise ValueError(f"Unsupported language: {language}")
    return value


def _effective_pattern(source_root: Path, pattern: str, recursive: bool, language: str = "auto") -> str:
    language = _normalize_language(language)
    if language == "python":
        return "*.py" if pattern in {"auto", "*", "*.*"} else pattern
    if language == "java":
        return "*.java" if pattern in {"auto", DEFAULT_TASK_PATTERN, "*", "*.*"} else pattern
    if language == "all":
        return "*.*" if pattern in {"auto", DEFAULT_TASK_PATTERN} else pattern
    if pattern not in {"auto", DEFAULT_TASK_PATTERN}:
        return pattern
    py_files = [path for path in (source_root.rglob("*.py") if recursive else source_root.glob("*.py")) if path.is_file()]
    if any(_matches_source_pattern(path, DEFAULT_TASK_PATTERN) for path in py_files):
        return DEFAULT_TASK_PATTERN
    if py_files:
        return "*.py"
    java_iter = source_root.rglob("*.java")
    if any(path.is_file() for path in java_iter):
        return "*.java"
    return DEFAULT_TASK_PATTERN


def _resume_existing_item(
    source_path: Path,
    case_dir: Path,
    execute: bool,
    batch_item: dict[str, object] | None = None,
) -> BatchItemResult | None:
    _recover_interrupted_round(case_dir)
    summary_path = case_dir / "stateflow_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(Path(str(summary.get("source", source_path))).resolve()) != str(source_path.resolve()):
        return None
    if summary.get("source_sha256") and summary["source_sha256"] != hashlib.sha256(source_path.read_bytes()).hexdigest():
        return None
    final_report = summary.get("final_report") if isinstance(summary.get("final_report"), dict) else None
    final_report = _restore_internal_report(case_dir, source_path, final_report, summary.get("graph_version"))
    if execute and final_report is None and not _summary_has_terminal_attempt(summary):
        return None
    status = _status_from_task_summary(summary, execute=execute)
    if not _resume_status_is_reusable(status, execute=execute, summary=summary):
        return None
    summary = _migrate_public_result(summary_path, summary)
    progress = {}
    progress_path = case_dir / "experiment_round_progress.json"
    if progress_path.exists():
        try:
            loaded = json.loads(progress_path.read_text(encoding="utf-8"))
            progress = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            progress = {}
    experiment_round = max(
        int(summary.get("experiment_round") or 1),
        int(progress.get("experiment_round") or 1) if progress.get("status") == "completed" else 1,
    )
    batch_item = batch_item if isinstance(batch_item, dict) else {}
    timing = _task_timing_snapshot(case_dir)
    return BatchItemResult(
        source=str(source_path),
        output_dir=str(case_dir),
        status=status,
        test_path=str(summary.get("test_path")) if summary.get("test_path") else None,
        graph_path=str(summary.get("graph_path")) if summary.get("graph_path") else None,
        summary_path=str(summary_path.resolve()),
        final_report=final_report,
        experiment_round=experiment_round,
        round_strategy=str(progress.get("strategy") or summary.get("followup_strategy") or "initial_generation"),
        grade_history=list(summary.get("grade_history") or batch_item.get("grade_history") or []),
        knowledge_history=list(summary.get("knowledge_history") or batch_item.get("knowledge_history") or []),
        failure_reason=str(summary.get("failure_reason") or batch_item.get("failure_reason") or "") or None,
        previous_solutions=list(batch_item.get("previous_solutions") or []),
        failure_report=str(
            batch_item.get("failure_report")
            or ((summary.get("visualization") or {}).get("failure_report_html") if isinstance(summary.get("visualization"), dict) else "")
            or ""
        ) or None,
        round_failure_diagnostic=(
            dict(summary.get("round_failure_diagnostic"))
            if isinstance(summary.get("round_failure_diagnostic"), dict)
            else dict(batch_item.get("round_failure_diagnostic"))
            if isinstance(batch_item.get("round_failure_diagnostic"), dict)
            else None
        ),
        started_at=str(timing.get("started_at") or summary.get("started_at") or batch_item.get("started_at") or "") or None,
        finished_at=str(timing.get("finished_at") or summary.get("finished_at") or batch_item.get("finished_at") or "") or None,
        duration_seconds=float(timing.get("duration_seconds") or summary.get("cumulative_duration_seconds") or summary.get("duration_seconds") or batch_item.get("duration_seconds") or 0.0),
        attempt_count=int(timing.get("attempt_count") or summary.get("attempt_count") or batch_item.get("attempt_count") or 0),
    )


def _restore_internal_report(
    case_dir: Path,
    source_path: Path,
    report: dict[str, object] | None,
    graph_version: object,
) -> dict[str, object] | None:
    """Recover process metrics from the matching private graph for continuation."""
    if not isinstance(report, dict):
        return report
    graph = _read_json_object(case_dir / "state_flow_graph.json")
    try:
        if Path(str(graph.get("source_path") or "")).resolve() != source_path.resolve():
            return report
    except (OSError, ValueError):
        return report
    if graph_version is not None and graph.get("version") != graph_version:
        return report
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    saved_report = metadata.get("last_report") if isinstance(metadata.get("last_report"), dict) else {}
    for key in ("pytest_passed", "collected_count", "line_coverage", "branch_coverage", "ae", "mutation_score"):
        if key in report and key in saved_report and report[key] != saved_report[key]:
            return report
    restored = dict(report)
    for key in (
        "sfc", "nrr", "tsq", "sfq", "csg", "pv", "dvs", "tic", "rcs",
        "quality_assessment", "threshold_failures", "weak_quality_flags",
    ):
        if key in saved_report and key not in restored:
            restored[key] = saved_report[key]
    return restored


def _status_from_task_summary(summary: dict[str, object], execute: bool) -> str:
    if not execute:
        return "generated"
    final_report = summary.get("final_report") if isinstance(summary.get("final_report"), dict) else None
    if not final_report:
        return "generation_failed"
    if final_report.get("pytest_passed"):
        quality_status = str(summary.get("quality_status") or final_report.get("quality_status") or "valid_usable")
        return quality_status
    return "execution_invalid"


def _resume_status_is_reusable(
    status: str,
    execute: bool,
    summary: dict[str, object] | None = None,
) -> bool:
    if not execute:
        return status == "generated"
    if status in VALID_STATUSES:
        return True
    if status not in FAILED_STATUSES or not isinstance(summary, dict):
        return False
    return bool(
        summary.get("failure_reason")
        or summary.get("error")
        or summary.get("last_error")
        or summary.get("failure_evidence")
        or summary.get("round_failure_diagnostic")
    )


def _summary_has_terminal_attempt(summary: dict[str, object]) -> bool:
    if summary.get("finished_at"):
        return True
    if summary.get("error") or summary.get("last_error") or summary.get("failure_evidence"):
        return True
    status = str(summary.get("status") or summary.get("quality_status") or summary.get("stop_reason") or "")
    return status in FAILED_STATUSES or status in {"test_state_invalid", "budget_limited", "interrupted"}


def _skip_reason_counts(skipped_sources: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in skipped_sources:
        reason = item.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _batch_failure_analysis(results: list[BatchItemResult]) -> dict[str, object]:
    categories: dict[str, int] = {}
    rejected_low_quality: list[dict[str, str]] = []
    low_mutation_reliability: list[str] = []
    unknown_details: list[dict[str, str]] = []
    for item in results:
        category = _failure_category(item)
        if category:
            categories[category] = categories.get(category, 0) + 1
            if category in {"exception_only", "no_normal_behavior", "weak_assertions_only"}:
                rejected_low_quality.append({"source": item.source, "reason": category})
            if category == "unknown":
                unknown_details.append(
                    {
                        "source": item.source,
                        "status": item.status,
                        "error_summary": (item.error or "没有错误摘要，也没有执行报告")[:240],
                    }
                )
        report = item.final_report or {}
        effective = int(report.get("effective_mutants", 0) or 0) if isinstance(report, dict) else 0
        reliability = str(report.get("mutation_reliability") or "") if isinstance(report, dict) else ""
        if report and (reliability in {"low", "unavailable"} or effective < 3):
            low_mutation_reliability.append(item.source)
    return {
        "failure_categories": categories,
        "rejected_low_quality": rejected_low_quality,
        "low_mutation_reliability": low_mutation_reliability,
        "unknown_details": unknown_details,
    }


def _weak_success_analysis(results: list[BatchItemResult]) -> dict[str, object]:
    flag_counts: dict[str, int] = {}
    examples: list[dict[str, object]] = []
    for item in results:
        if item.status != "valid_usable" or not isinstance(item.final_report, dict):
            continue
        raw_flags = item.final_report.get("weak_quality_flags")
        flags = [str(flag) for flag in raw_flags if public_text(str(flag)) == str(flag)] if isinstance(raw_flags, list) else []
        if not flags:
            flags = ["below_quality_threshold"]
        for flag in flags:
            flag_counts[str(flag)] = flag_counts.get(str(flag), 0) + 1
        examples.append(
            {
                "source": item.source,
                "weak_quality_flags": flags,
                "threshold_failures": item.final_report.get("threshold_failures", {}),
                "line_coverage": item.final_report.get("line_coverage"),
                "branch_coverage": item.final_report.get("branch_coverage"),
                "ae": item.final_report.get("ae"),
                "mutation_score": item.final_report.get("mutation_score"),
            }
        )
    return {
        "weak_success_count": len(examples),
        "weak_quality_flag_counts": flag_counts,
        "examples": examples[:100],
    }


def _batch_quality_overview(results: list[BatchItemResult]) -> dict[str, object]:
    executed_reports = [item.final_report for item in results if isinstance(item.final_report, dict) and item.final_report.get("pytest_passed")]
    # Dataset quality averages describe valid A/B/C outcomes.  A task whose pytest
    # process happened to pass but which failed semantic usability remains D and must
    # not silently depress or distort the reported quality of completed results.
    def is_valid_result(item: BatchItemResult) -> bool:
        grade = getattr(item, "final_grade", None)
        return grade is None or grade in VALID_GRADES

    reports = [
        item.final_report
        for item in results
        if is_valid_result(item)
        and isinstance(item.final_report, dict)
        and item.final_report.get("pytest_passed")
    ]

    def avg(key: str) -> float | None:
        values = [float(report[key]) for report in reports if report.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    def total(key: str) -> int:
        return sum(int(report.get(key, 0) or 0) for report in reports)

    mutation_reports = [report for report in reports if report.get("mutation_score") is not None]
    low_mutation = [
        {
            "source": item.source,
            "effective_mutants": int((item.final_report or {}).get("effective_mutants", 0) or 0),
            "reason": str((item.final_report or {}).get("mutation_low_sample_reason") or "有效变异体少于 3 个"),
        }
        for item in results
        if is_valid_result(item)
        and isinstance(item.final_report, dict)
        and item.final_report.get("pytest_passed")
        and int(item.final_report.get("effective_mutants", 0) or 0) < 3
    ]
    return {
        "executed_tasks": len(reports),
        "all_executed_tasks": len(executed_reports),
        "excluded_invalid_tasks": max(0, len(executed_reports) - len(reports)),
        "average_line_coverage": avg("line_coverage"),
        "average_branch_coverage": avg("branch_coverage"),
        "average_ae": avg("ae"),
        "average_mutation_score": avg("mutation_score"),
        "mutation_scored_tasks": len(mutation_reports),
        "total_assertions": total("assertion_count"),
        "total_effective_assertions": total("effective_assertions"),
        "total_weak_assertions": total("weak_assertions"),
        "average_effective_mutants": avg("effective_mutants"),
        "low_mutation_reliability": low_mutation[:100],
        "assertion_explanation": "有效断言=总断言数减去弱断言；弱断言包括恒真、只和 None 比较、对象存在性检查等无法证明具体行为的断言。",
    }


def _failure_category(item: BatchItemResult) -> str | None:
    if item.status not in {"failed", "generation_failed", "execution_invalid"}:
        return None
    if item.final_report and item.final_report.get("failure_category"):
        return str(item.final_report.get("failure_category"))
    error = item.error or ""
    for category in [
        "exception_only",
        "no_normal_behavior",
        "weak_assertions_only",
        "syntax_error",
        "import_error",
        "oracle_error",
        "collection_error",
        "fixture_error",
        "runtime_error",
        "baseline test invalid",
        "no mutation candidates",
        "timeout",
    ]:
        if category in error:
            return category.replace(" ", "_")
    if "no valid generated test candidate" in error:
        return "candidate_rejected"
    if "RuntimeError" in error:
        return "runtime_error"
    if "ValueError" in error:
        return "value_error"
    return "unknown"


def _task_sort_key(path: Path) -> tuple[int, str]:
    prefix_pattern = "|".join(re.escape(prefix) for prefix in DEFAULT_TASK_PREFIXES)
    match = re.fullmatch(rf"(?:{prefix_pattern})(\d+)\.py", path.name)
    return (int(match.group(1)), path.name) if match else (10**9, path.name)


def _case_name(source_root: Path, source_path: Path) -> str:
    relative = source_path.relative_to(source_root).with_suffix("")
    parts = [part.replace(" ", "_") for part in relative.parts]
    return "__".join(parts)


def _generate_only(source_path: Path, case_dir: Path, llm_config: LLMConfig) -> BatchItemResult:
    case_dir.mkdir(parents=True, exist_ok=True)
    graph_path = case_dir / "state_flow_graph.json"
    memory = SharedGraphMemory(graph_path)
    llm = OpenAICompatibleLLM(llm_config) if llm_config.enabled else None
    SourceAnalysisAgent(memory, llm=llm, llm_required=llm_config.required).run(source_path)
    test_path = TestGenerationAgent(memory, llm=llm, llm_required=llm_config.required).run(case_dir)
    graph = memory.load()
    try:
        from visualization.graph_visualizer import StateFlowGraphVisualizer

        visualization = StateFlowGraphVisualizer().render(graph_path, case_dir)
        graph.metadata["visualization"] = visualization
        memory.save(graph)
    except Exception as exc:
        visualization = {"status": "visualization_failed", "error": f"{exc.__class__.__name__}: {exc}"}
    summary = {
        "source": str(source_path),
        "test_path": str(test_path.resolve()),
        "graph_path": str(graph_path.resolve()),
        "mode": "generate-only",
        "final_report": None,
        "graph_version": graph.version,
        "visualization": visualization,
        "llm": llm_config.public_dict(),
    }
    summary_path = case_dir / "stateflow_summary.json"
    summary_path.write_text(json.dumps(public_result(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return BatchItemResult(
        source=str(source_path),
        output_dir=str(case_dir),
        status="generated",
        test_path=str(test_path.resolve()),
        graph_path=str(graph_path.resolve()),
        summary_path=str(summary_path.resolve()),
        visualization=visualization,
    )


def _generate_and_execute(
    source_path: Path,
    case_dir: Path,
    thresholds: Thresholds,
    max_iterations: int,
    retry_rounds: int,
    llm_config: LLMConfig,
    execution_budget: int | None = None,
    llm_call_budget: int | None = None,
    token_budget: int | None = None,
    time_budget_seconds: float | None = None,
    resume: bool = True,
    rerun_existing: bool = False,
    experiment_round: int = 1,
    followup_strategy: str = "initial_generation",
    baseline_report: dict[str, object] | None = None,
    existing_test_path: str | Path | None = None,
    previous_grade: str | None = None,
    experience_read_path: str | Path | None = None,
    experience_journal_dir: str | Path | None = None,
) -> BatchItemResult:
    system = MultiAgentUnitTestSystem(
        source_path=source_path,
        output_dir=case_dir,
        thresholds=thresholds,
        max_iterations=max_iterations,
        retry_rounds=retry_rounds,
        llm_config=llm_config,
        experience_path=resolve_knowledge_store_path(),
        experience_read_path=experience_read_path,
        experience_journal_dir=experience_journal_dir,
        isolate_humaneval_java=experience_read_path is not None,
        execution_budget=execution_budget,
        llm_call_budget=llm_call_budget,
        token_budget=token_budget,
        time_budget_seconds=time_budget_seconds,
        resume=resume,
        rerun_existing=rerun_existing,
        experiment_round=experiment_round,
        followup_strategy=followup_strategy,
        baseline_report=baseline_report,
        existing_test_path=existing_test_path,
        previous_grade=previous_grade,
    )
    summary = system.run()
    final_report = summary.get("final_report") if isinstance(summary.get("final_report"), dict) else None
    final_report = _restore_internal_report(case_dir, source_path, final_report, summary.get("graph_version"))
    final_report = _enrich_final_report(case_dir, final_report)
    if final_report and final_report.get("pytest_passed"):
        quality_status = str(summary.get("quality_status") or final_report.get("quality_status") or "valid_usable")
        status = quality_status
    else:
        status = "generation_failed" if not final_report else "execution_invalid"
    return BatchItemResult(
        source=str(source_path),
        output_dir=str(case_dir),
        status=status,
        test_path=str(summary.get("test_path")),
        graph_path=str(summary.get("graph_path")),
        summary_path=str((case_dir / "stateflow_summary.json").resolve()),
        final_report=final_report,
        visualization=summary.get("visualization") if isinstance(summary.get("visualization"), dict) else None,
        error=str(summary.get("last_error") or summary.get("error") or "").strip() or None,
        knowledge_history=[
            {
                **context,
                "outcome": summary.get("knowledge_outcome") if isinstance(summary.get("knowledge_outcome"), dict) else {},
            }
            for context in (
                summary.get("knowledge_history", [])
                if isinstance(summary.get("knowledge_history"), list)
                else [summary.get("knowledge_context", {})]
            )
            if isinstance(context, dict)
        ],
        round_strategy=followup_strategy,
    )


def _write_failed_task_artifacts(
    source_path: Path,
    case_dir: Path,
    error: str,
    execute: bool,
    llm_config: LLMConfig,
) -> dict[str, object]:
    case_dir.mkdir(parents=True, exist_ok=True)
    started_finished = _utc_now()
    failure_report = case_dir / "task_failure_report.html"
    failure_report.write_text(_failure_html(source_path, error), encoding="utf-8")
    summary = {
        "source": str(source_path.resolve()),
        "output_dir": str(case_dir.resolve()),
        "mode": "execute" if execute else "generate-only",
        "status": "generation_failed",
        "started_at": started_finished,
        "finished_at": started_finished,
        "duration_seconds": 0.0,
        "final_report": None,
        "error": error,
        "visualization": {
            "status": "generated",
            "failure_report_html": str(failure_report.resolve()),
            "generated_at": started_finished,
        },
        "llm": llm_config.public_dict(),
    }
    summary_path = case_dir / "stateflow_summary.json"
    summary_path.write_text(json.dumps(public_result(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"summary_path": str(summary_path.resolve()), "visualization": summary["visualization"]}


def _failure_html(source_path: Path, error: str) -> str:
    escaped_source = html.escape(str(source_path.resolve()))
    escaped_error = html.escape(public_text(error))
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>任务失败报告</title>"
        "<style>body{margin:0;background:#f8fafc;color:#0f172a;font-family:Segoe UI,Microsoft YaHei,sans-serif}"
        "main{max-width:980px;margin:auto;padding:32px}.panel{background:white;border:1px solid #e2e8f0;border-radius:14px;padding:22px}"
        "h1{margin-top:0;font-size:26px}.muted{color:#64748b}pre{white-space:pre-wrap;overflow:auto;background:#111827;color:#e5e7eb;padding:16px;border-radius:10px}</style>"
        "</head><body><main><section class=\"panel\"><h1>任务失败报告</h1>"
        f"<p class=\"muted\">源文件：{escaped_source}</p>"
        "<p>该任务已被记录为失败。可以根据下面的错误信息修复环境、依赖或代码后，再次运行同一条批量命令；系统会跳过已完成任务，只继续未完成或失败的任务。</p>"
        f"<pre>{escaped_error}</pre></section></main></body></html>"
    )


def _enrich_final_report(case_dir: Path, final_report: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(final_report, dict):
        return final_report
    graph_path = case_dir / "state_flow_graph.json"
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return final_report
    metadata = graph.get("metadata", {}) if isinstance(graph.get("metadata"), dict) else {}
    usability = metadata.get("last_test_usability", {}) if isinstance(metadata.get("last_test_usability"), dict) else {}
    mutation = metadata.get("last_mutation", {}) if isinstance(metadata.get("last_mutation"), dict) else {}
    final_report.setdefault("assertion_count", int(usability.get("assertion_count", final_report.get("assertion_count", 0)) or 0))
    final_report.setdefault("weak_assertions", int(usability.get("weak_assertions", final_report.get("weak_assertions", 0)) or 0))
    final_report.setdefault(
        "effective_assertions",
        max(0, int(final_report.get("assertion_count", 0) or 0) - int(final_report.get("weak_assertions", 0) or 0)),
    )
    final_report.setdefault(
        "assertion_quality_explanation",
        "有效断言=总断言数减去弱断言；弱断言包括恒真、None 比较、对象存在性检查等。",
    )
    final_report["mutation_low_sample_reason"] = mutation.get("low_sample_reason", "")
    final_report["mutation_autonomy_action"] = mutation.get("autonomy_action", "")
    final_report["mutation_timeout_count"] = mutation.get("timeout", 0)
    final_report["mutation_timeout_retries"] = mutation.get("timeout_retries", 0)
    final_report["mutation_candidate_count"] = mutation.get("candidate_count", 0)
    final_report["mutation_attempted_mutants"] = mutation.get("attempted_mutants", 0)
    return final_report


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_runtime_path(case_dir: Path) -> Path:
    return case_dir / "task_runtime.json"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    temporary.write_text(content, encoding="utf-8")
    try:
        temporary.replace(path)
    except OSError:
        # Some Windows/Python combinations cannot atomically rename paths under a
        # non-ASCII user profile.  Keep persistence usable in that environment.
        path.write_text(content, encoding="utf-8")
        try:
            temporary.unlink()
        except OSError:
            pass


def _parse_time(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _begin_task_timing(
    case_dir: Path,
    source_path: Path,
    *,
    experiment_round: int,
    strategy: str,
) -> str:
    case_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = _task_runtime_path(case_dir)
    payload = _read_json_object(runtime_path)
    sessions = [dict(item) for item in payload.get("sessions", []) if isinstance(item, dict)] if isinstance(payload.get("sessions"), list) else []

    # Migrate old summaries once so an upgrade does not erase already measured work.
    if not sessions:
        legacy = _read_json_object(case_dir / "stateflow_summary.json")
        legacy_seconds = float(
            legacy.get("cumulative_duration_seconds")
            or legacy.get("duration_seconds")
            or legacy.get("runtime_seconds")
            or 0.0
        )
        if legacy_seconds > 0:
            sessions.append(
                {
                    "id": "legacy",
                    "round": int(legacy.get("experiment_round", 1) or 1),
                    "strategy": legacy.get("followup_strategy") or "legacy_run",
                    "status": "completed_legacy",
                    "started_at": legacy.get("started_at"),
                    "finished_at": legacy.get("finished_at"),
                    "duration_seconds": round(legacy_seconds, 4),
                }
            )

    # If the previous process stopped abruptly, close its active interval at the
    # latest task artifact update rather than at resume time (which would count the
    # computer's idle/offline period as experiment time).
    for session in sessions:
        if session.get("status") != "running":
            continue
        started = _parse_time(session.get("started_at"))
        artifact_epochs = []
        for path in case_dir.iterdir():
            if not path.is_file() or path.name in {runtime_path.name, runtime_path.name + ".tmp"}:
                continue
            try:
                artifact_epochs.append(path.stat().st_mtime)
            except OSError:
                # A report may be atomically replaced while an interrupted task is
                # being recovered.  One disappearing artifact must not block resume.
                continue
        stopped = datetime.fromtimestamp(max(artifact_epochs, default=started.timestamp() if started else 0.0), timezone.utc)
        session["finished_at"] = stopped.isoformat()
        session["duration_seconds"] = round(max(0.0, (stopped - started).total_seconds()), 4) if started else 0.0
        session["status"] = "interrupted"

    started_at = _utc_now()
    token = hashlib.sha256(f"{source_path.resolve()}|{experiment_round}|{started_at}".encode("utf-8")).hexdigest()[:20]
    sessions.append(
        {
            "id": token,
            "round": max(1, int(experiment_round or 1)),
            "strategy": strategy,
            "status": "running",
            "started_at": started_at,
            "finished_at": None,
            "duration_seconds": 0.0,
        }
    )
    _write_json_atomic(
        runtime_path,
        {
            "schema_version": 1,
            "source": str(source_path.resolve()),
            "sessions": sessions,
            "updated_at": started_at,
        },
    )
    return token


def _finish_task_timing(case_dir: Path, token: str, status: str) -> dict[str, object]:
    runtime_path = _task_runtime_path(case_dir)
    payload = _read_json_object(runtime_path)
    sessions = [dict(item) for item in payload.get("sessions", []) if isinstance(item, dict)] if isinstance(payload.get("sessions"), list) else []
    finished_at = _utc_now()
    for session in sessions:
        if str(session.get("id") or "") != token:
            continue
        started = _parse_time(session.get("started_at"))
        finished = _parse_time(finished_at)
        session["finished_at"] = finished_at
        session["duration_seconds"] = round(max(0.0, (finished - started).total_seconds()), 4) if started and finished else 0.0
        session["status"] = status
        break
    payload.update({"schema_version": 1, "sessions": sessions, "updated_at": finished_at})
    _write_json_atomic(runtime_path, payload)
    return _task_timing_snapshot(case_dir)


def _task_timing_snapshot(case_dir: Path) -> dict[str, object]:
    payload = _read_json_object(_task_runtime_path(case_dir))
    sessions = [item for item in payload.get("sessions", []) if isinstance(item, dict)] if isinstance(payload.get("sessions"), list) else []
    completed = sum(float(item.get("duration_seconds", 0.0) or 0.0) for item in sessions if item.get("status") != "running")
    running = next((item for item in reversed(sessions) if item.get("status") == "running"), None)
    active_elapsed = 0.0
    if running:
        started = _parse_time(running.get("started_at"))
        if started:
            active_elapsed = max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    started_values = [str(item.get("started_at")) for item in sessions if item.get("started_at")]
    finished_values = [str(item.get("finished_at")) for item in sessions if item.get("finished_at")]
    return {
        "started_at": min(started_values) if started_values else None,
        "finished_at": max(finished_values) if finished_values else None,
        "duration_seconds": round(completed + active_elapsed, 4),
        "accumulated_duration_seconds": round(completed, 4),
        "active_session_started_at": running.get("started_at") if running else None,
        "attempt_count": len(sessions),
        "sessions": sessions,
    }


def _apply_task_timing(item: BatchItemResult, timing: dict[str, object]) -> None:
    item.started_at = str(timing.get("started_at") or "") or None
    item.finished_at = str(timing.get("finished_at") or "") or None
    item.duration_seconds = float(timing.get("duration_seconds") or 0.0)
    item.attempt_count = int(timing.get("attempt_count") or 0)
    if not item.summary_path:
        return
    summary_path = Path(item.summary_path)
    summary = _read_json_object(summary_path)
    if not summary:
        return
    summary.update(
        {
            "started_at": item.started_at,
            "finished_at": item.finished_at,
            "duration_seconds": item.duration_seconds,
            "cumulative_duration_seconds": item.duration_seconds,
            "attempt_count": item.attempt_count,
            "timing": timing,
        }
    )
    if isinstance(summary.get("paper_metrics"), dict):
        summary["paper_metrics"]["duration_seconds"] = item.duration_seconds
    _write_json_atomic(summary_path, public_result(summary))


def _duration_seconds(started_at: str, finished_at: str) -> float:
    try:
        started = datetime.fromisoformat(started_at)
        finished = datetime.fromisoformat(finished_at)
    except ValueError:
        return 0.0
    return round(max(0.0, (finished - started).total_seconds()), 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate unit tests for Python or Java source files.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Directory that contains Python or Java source files.")
    parser.add_argument("--out", default="generated_tests_humaneval", help="Output directory for generated tests.")
    parser.add_argument("--pattern", default=DEFAULT_TASK_PATTERN, help=f"File glob pattern. Default: {DEFAULT_TASK_PATTERN}; Java directories are auto-detected.")
    parser.add_argument("--language", choices=["auto", "python", "java", "all"], default="auto", help="Source language selection. Default: auto.")
    parser.add_argument("--recursive", action="store_true", help="Search source directory recursively.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N source files.")
    parser.add_argument("--workers", type=int, default=None, help="Task workers. Default: Python 4, HumanEvalJava 2 (isolated), all other Java 1.")
    parser.add_argument("--execute", action="store_true", help="Run pytest/coverage/mutation quality loop after generation.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum repair iterations when --execute is used.")
    parser.add_argument("--retry-rounds", type=int, default=2, help="Maximum pytest timeout retry rounds per execution before repair.")
    parser.add_argument("--execution-budget", type=int, default=None, help="Maximum execution iterations per task.")
    parser.add_argument("--llm-call-budget", type=int, default=None, help="Maximum LLM calls per task.")
    parser.add_argument("--token-budget", type=int, default=None, help="Maximum recorded/estimated tokens per task.")
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Maximum wall-clock seconds per task.")
    parser.add_argument("--retry-failed-rounds", type=int, default=2, help="Compatibility option: 2 enables R1/R2/R3 (R2 recovers D/E; R3 optimizes C; no R4).")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing per-task summaries from a previous interrupted run.")
    parser.add_argument("--rerun-existing", action="store_true", help="Force rerunning tasks even if a completed summary already exists.")
    parser.add_argument("--line-coverage", type=float, default=0.80, help="Acceptable line coverage threshold.")
    parser.add_argument("--branch-coverage", type=float, default=0.70, help="Acceptable branch coverage threshold.")
    parser.add_argument("--sfc", type=float, default=0.65, help="Acceptable test-state coverage threshold.")
    parser.add_argument("--ae", type=float, default=0.75, help="Minimum assertion-effectiveness threshold.")
    parser.add_argument("--mutation", type=float, default=0.60, help="Acceptable reliable mutation score threshold.")
    parser.add_argument("--tsq", "--sfq", dest="sfq", type=float, default=0.70, help="Acceptable Test State Quality threshold; --sfq is retained as a compatibility alias.")
    parser.add_argument("--llm", action="store_true", help="Use the configured OpenAI-compatible LLM for generation and repair.")
    parser.add_argument("--llm-required", action="store_true", help="Fail instead of falling back if the LLM call fails.")
    parser.add_argument("--llm-base-url", default=None, help="LLM host URL. For Ollama, use http://127.0.0.1:11434.")
    parser.add_argument("--llm-model", default=None, help="Model name, such as deepseek-r1:14b.")
    parser.add_argument("--llm-api-key", default=None, help="API key. For many local servers, any non-empty value is fine.")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM request timeout in seconds.")
    parser.add_argument("--llm-temperature", type=float, default=None, help="LLM sampling temperature.")
    parser.add_argument("--knowledge-store", default=None, help="External long-term knowledge JSON path. Existing files are never overwritten by project defaults.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first failed source file.")
    parser.add_argument("--quiet", action="store_true", help="Hide per-file progress.")
    parser.add_argument("--no-ui", action="store_true", help="Do not start or open the real-time experiment dashboard.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from monitoring import finish_dashboard, launch_dashboard

    configure_knowledge_store_path(args.knowledge_store)
    llm_config = LLMConfig.from_env(
        enabled=args.llm,
        required=args.llm_required,
        base_url=args.llm_base_url,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout=args.llm_timeout,
        temperature=args.llm_temperature,
    )
    launch_dashboard(args.out, label=Path(args.source_dir).name, enabled=not args.no_ui, llm_config=llm_config)
    failure = None
    try:
        summary = generate_tests_for_directory(
            source_dir=args.source_dir,
            output_dir=args.out,
            pattern=args.pattern,
            language=args.language,
            recursive=args.recursive,
            execute=args.execute,
            max_iterations=args.max_iterations,
            retry_rounds=args.retry_rounds,
            thresholds=Thresholds(
                line_coverage=args.line_coverage,
                branch_coverage=args.branch_coverage,
                sfc=args.sfc,
                ae=args.ae,
                mutation=args.mutation,
                sfq=args.sfq,
            ),
            llm_config=llm_config,
            limit=args.limit,
            fail_fast=args.fail_fast,
            verbose=not args.quiet,
            execution_budget=args.execution_budget,
            llm_call_budget=args.llm_call_budget,
            token_budget=args.token_budget,
            time_budget_seconds=args.time_budget_seconds,
            retry_failed_rounds=args.retry_failed_rounds,
            resume=not args.no_resume,
            rerun_existing=args.rerun_existing,
            workers=args.workers,
            knowledge_store=args.knowledge_store,
        )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finish_dashboard(args.out, success=failure is None, error=failure, enabled=not args.no_ui)
    print(json.dumps(_terminal_summary(summary), indent=2, ensure_ascii=False))
    print(f"Batch summary: {Path(summary['output_dir']) / 'batch_summary.json'}")


def _terminal_summary(summary: dict[str, object]) -> dict[str, object]:
    quality = summary.get("quality_overview") if isinstance(summary.get("quality_overview"), dict) else {}
    return {
        "total": summary.get("total"),
        "discovered_files": summary.get("discovered_files"),
        "selected_tasks": summary.get("selected_tasks"),
        "resumed_existing": summary.get("resumed_existing"),
        "dataset_skipped": summary.get("dataset_skipped", False),
        "outcome_summary": summary.get("outcome_summary"),
        "output_dir": summary.get("output_dir"),
        "mode": summary.get("mode"),
        "language": summary.get("language"),
        "effective_pattern": summary.get("effective_pattern"),
        "experiment_round_summary": summary.get("experiment_round_summary"),
        "weak_success_analysis": summary.get("weak_success_analysis"),
        "paper_metrics": summary.get("paper_metrics", {}),
        "metrics": {
            "executed_tasks": quality.get("executed_tasks"),
            "average_line_coverage": quality.get("average_line_coverage"),
            "average_branch_coverage": quality.get("average_branch_coverage"),
            "average_ae": quality.get("average_ae"),
            "average_mutation_score": quality.get("average_mutation_score"),
        },
        "batch_report": summary.get("batch_report"),
        "test_knowledge_guidance_library": summary.get("test_knowledge_guidance_library"),
    }


if __name__ == "__main__":
    main()
