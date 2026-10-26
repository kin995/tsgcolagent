"""Persist the public data contract used by the experiment pages.

This is a presentation projection, not an archive of execution artifacts.  The
caller passes snapshots that have already crossed the public API filter; this
module narrows them to the fields read by the frontend and applies the public
filter once more before writing a standalone, reproducible page dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from core.public_metrics import public_result


_HIDDEN_TEXT = ("内部过程指标已隐藏", "内部过程信息已隐藏")
_OMIT = object()
_WRITE_LOCK = threading.RLock()


def _pick(source: object, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    return {key: source[key] for key in keys if key in source}


def _items(source: object, project: Any) -> list[dict[str, Any]]:
    return [project(item) for item in source if isinstance(item, Mapping)] if isinstance(source, list) else []


def _event(item: Mapping[str, Any]) -> dict[str, Any]:
    result = _pick(item, (
        "event_id", "event_type", "source_agent", "target_agent", "iteration",
        "related_nodes", "related_edges", "timestamp", "experiment_id",
        "experiment_label", "task_id", "task_name",
    ))
    # These are the only event payload fields consumed by the route and agent
    # cards.  Historical payloads also carry source text and nested tool logs.
    result["payload"] = _pick(item.get("payload"), (
        "experiment_round", "focus", "action", "failure_category", "quality_status",
    ))
    return result


def _task(item: Mapping[str, Any]) -> dict[str, Any]:
    result = _pick(item, (
        "id", "name", "source", "scope", "output_dir", "summary_path",
        "test_path", "test_artifact_available", "status", "grade", "round",
        "round_label", "iterations", "round_strategy", "reports_count",
        "duration_seconds", "started_at", "finished_at", "attempt_count",
        "stop_reason", "experiment_id", "experiment_label", "dataset_id",
        "dataset_name",
    ))
    result["metrics"] = _pick(item.get("metrics"), (
        "tests_passed", "collected_count", "line_coverage", "branch_coverage",
        "ae", "mutation_score", "duration_seconds", "pass_rate", "avg_exec",
        "exec_count", "exec_count_status", "tir", "tir_comparable",
        "tir_improved", "tir_excluded", "tir_status", "knowledge_record_ids",
        "knowledge_use_frequency",
    ))
    result["grade_history"] = _items(item.get("grade_history"), _grade_history)
    failure = _pick(item.get("failure"), (
        "category", "stage", "failed_tests", "weak_flags", "reason", "error",
        "previous_solutions", "suggested_solutions", "attempted_measures",
        "knowledge_record_id",
    ))
    if failure:
        result["failure"] = failure
    return result


def _grade_history(item: Mapping[str, Any]) -> dict[str, Any]:
    result = _pick(item, ("round", "grade", "status", "label", "failure_reason"))
    if isinstance(item.get("metrics"), Mapping):
        result["metrics"] = _pick(item["metrics"], (
            "tests_passed", "pytest_passed", "collected_count", "line_coverage",
            "branch_coverage", "ae", "mutation_score", "pass_rate", "avg_exec",
            "tir", "duration_seconds",
        ))
    return result


def _experiment(item: Mapping[str, Any]) -> dict[str, Any]:
    return _pick(item, (
        "id", "parent_id", "output_dir", "label", "batch_report_path",
        "report_path", "registered_at", "status", "language", "scope",
        "total_tasks", "completed_execution", "not_completed_execution",
        "progress", "queue_position", "started_at", "finished_at",
        "duration_seconds", "duration_source",
    ))


def _catalog(source: object) -> dict[str, Any]:
    value = _pick(source, ("name", "runtime"))
    data = source if isinstance(source, Mapping) else {}
    language_keys = (
        "id", "name", "source_extensions", "test_framework", "coverage_tools",
        "mutation_tools", "analysis", "generated_test", "evidence",
    )
    value["languages"] = _items(data.get("languages"), lambda item: _pick(item, language_keys))
    value["scopes"] = _items(data.get("scopes"), lambda item: _pick(item, ("id", "name", "entry", "description")))
    value["phases"] = _items(data.get("phases"), _phase)
    value["agents"] = _items(data.get("agents"), _catalog_agent)
    value["quality_metrics"] = _items(data.get("quality_metrics"), lambda item: _pick(item, ("id", "name", "description")))
    value["execution_policies"] = _items(data.get("execution_policies"), lambda item: _pick(item, ("id", "name", "value", "description")))
    return value


def _catalog_agent(item: Mapping[str, Any]) -> dict[str, Any]:
    value = _pick(item, ("id", "name", "mission", "inputs", "outputs"))
    value["modules"] = _items(item.get("modules"), lambda module: _pick(module, ("id", "name", "description")))
    return value


def _phase(item: Mapping[str, Any]) -> dict[str, Any]:
    value = _pick(item, ("id", "name", "owner", "description", "artifacts", "status", "event_count"))
    if isinstance(item.get("last_event"), Mapping):
        value["last_event"] = _event(item["last_event"])
    return value


def _agent(item: Mapping[str, Any]) -> dict[str, Any]:
    value = _pick(item, (
        "id", "name", "status", "modules", "current_action", "interaction_role",
        "event_count",
    ))
    value["internal_tasks"] = _items(item.get("internal_tasks"), _internal_task)
    if isinstance(item.get("last_event"), Mapping):
        value["last_event"] = _event(item["last_event"])
    return value


def _internal_task(item: Mapping[str, Any]) -> dict[str, Any]:
    value = _pick(item, ("name", "status", "event_count"))
    if isinstance(item.get("last_event"), Mapping):
        value["last_event"] = _event(item["last_event"])
    return value


def _public(value: object) -> object:
    """Remove any residual process-only placeholder, including nested text."""
    filtered = public_result(value)

    def without_placeholder(item: object) -> object:
        if isinstance(item, str):
            return _OMIT if any(text in item for text in _HIDDEN_TEXT) else item
        if isinstance(item, list):
            values = [without_placeholder(child) for child in item]
            return [child for child in values if child is not _OMIT]
        if isinstance(item, dict):
            values = ((key, without_placeholder(child)) for key, child in item.items())
            return {key: child for key, child in values if child is not _OMIT}
        return item

    return without_placeholder(filtered)


def _generated_at(runtime: Mapping[str, Any]) -> str:
    experiment = runtime.get("experiment")
    overview = runtime.get("overview")
    if isinstance(experiment, Mapping) and experiment.get("status") == "completed":
        finished = experiment.get("finished_at") or (overview.get("finished_at") if isinstance(overview, Mapping) else None)
        if isinstance(finished, str) and finished:
            return finished
    return str(runtime.get("generated_at") or "")


def project_page_data(runtime: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return standalone runtime/knowledge JSON objects for the current pages.

    Inputs must already be public snapshots.  The projection is detached and
    never alters the caller's cached snapshots.
    """
    overview = runtime.get("overview") if isinstance(runtime.get("overview"), Mapping) else {}
    state = runtime.get("state_model") if isinstance(runtime.get("state_model"), Mapping) else {}
    parallel = runtime.get("parallel_execution") if isinstance(runtime.get("parallel_execution"), Mapping) else None
    page_runtime: dict[str, Any] = {
        "schema_version": runtime.get("schema_version", 1),
        "generated_at": _generated_at(runtime),
        "experiment": _experiment(runtime["experiment"]) if isinstance(runtime.get("experiment"), Mapping) else None,
        "experiments": _items(runtime.get("experiments"), _experiment),
        "overview": _pick(overview, (
            "language", "model_name", "model_usage", "mode", "status", "scope",
            "total_tasks", "completed_execution", "not_completed_execution",
            "grade_counts", "followup_task_count", "followup_attempts",
            "started_at", "finished_at", "duration_seconds", "active_session_started_at",
            "duration_source", "accumulated_duration_before_session", "session_duration_seconds",
            "dataset_count", "completed_datasets", "task_workers", "active_task_count",
            "queued_task_count",
        )),
        "catalog": _catalog(runtime.get("catalog")),
        "phase_states": _items(runtime.get("phase_states"), _phase),
        "toolchain": _items(runtime.get("toolchain"), lambda item: _pick(item, ("name", "status", "value"))),
        "agents": _items(runtime.get("agents"), _agent),
        "current_task": _task(runtime["current_task"]) if isinstance(runtime.get("current_task"), Mapping) else None,
        "tasks": _items(runtime.get("tasks"), _task),
        "state_model": {
            **_pick(state, ("version", "iteration")),
            "nodes": _items(state.get("nodes"), lambda item: _pick(item, (
                "id", "type", "name", "visited", "evidenced", "line", "coverage_status",
                "mutation_status", "assertion_effective", "repair_flag",
            ))),
            "edges": _items(state.get("edges"), lambda item: _pick(item, (
                "id", "source", "target", "type", "visited",
            ))),
            "metadata": _pick(state.get("metadata"), (
                "last_test_intent_plan", "last_evaluation_feedback",
                "last_knowledge_context", "last_report",
            )),
        },
        "flow_events": _items(runtime.get("flow_events"), _event),
        "failures": _pick(runtime.get("failures"), ("failure_categories",)),
    }
    page_runtime["overview"]["page_title"] = "实验总览"
    page_runtime["overview"]["quality_averages"] = _pick(overview.get("quality_averages"), (
        "average_line_coverage", "average_branch_coverage", "average_ae",
        "average_mutation_score", "pass_rate", "avg_exec", "average_time_seconds",
        "tir", "tir_improved", "tir_comparable", "tir_excluded",
        "exec_complete_tasks", "selected_tasks",
    ))
    portfolio = runtime.get("portfolio")
    if isinstance(portfolio, Mapping):
        page_runtime["portfolio"] = {
            "view": portfolio.get("view"),
            "counts": portfolio.get("counts") if isinstance(portfolio.get("counts"), Mapping) else {},
            "experiments": _items(portfolio.get("experiments"), _experiment),
        }
    if parallel is not None:
        page_runtime["parallel_execution"] = _pick(parallel, (
            "task_workers", "llm_concurrency", "python_parallel", "java_parallel",
            "active_task_count", "queued_task_count",
        ))
        page_runtime["parallel_execution"]["active_tasks"] = _items(
            parallel.get("active_tasks"), lambda item: _pick(item, ("id", "name", "round_label", "updated_at")))
    if isinstance(runtime.get("control"), Mapping):
        page_runtime["control"] = _pick(runtime["control"], (
            "can_start", "process_id", "process_running", "log_path",
            "queue_position", "queue_size", "execution_status",
        ))
    if isinstance(runtime.get("assistant"), Mapping):
        page_runtime["assistant"] = _pick(runtime["assistant"], (
            "enabled", "provider", "model", "base_url", "backend_managed",
            "api_key_configured",
        ))

    availability = _pick(knowledge.get("availability"), (
        "complete", "note", "recovered_snapshot_records", "recovered_journal_records",
        "run_record_count", "baseline_record_count", "dataset_summary_count",
        "post_import_learning_count", "missing_datasets",
    ))
    page_knowledge: dict[str, Any] = {
        "scope": knowledge.get("scope"),
        "total": knowledge.get("total", 0),
        "counts": knowledge.get("counts") if isinstance(knowledge.get("counts"), Mapping) else {},
        "records": _items(knowledge.get("records"), lambda item: _pick(item, (
            "id", "status", "agent", "action", "failure_category", "accepted",
            "source", "solution", "usage", "topic", "validation_count",
            "support_count", "independent_task_count", "source_type", "dataset",
        ))),
    }
    if availability:
        page_knowledge["availability"] = availability
    public_runtime = _public(page_runtime)
    public_knowledge = _public(page_knowledge)
    return (public_runtime if isinstance(public_runtime, dict) else {},
            public_knowledge if isinstance(public_knowledge, dict) else {})


def project_runtime_page(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Project one public runtime response to exactly the page contract."""
    return project_page_data(runtime, {})[0]


def project_knowledge_page(knowledge: Mapping[str, Any]) -> dict[str, Any]:
    """Project one public knowledge response to exactly the page contract."""
    return project_page_data({}, knowledge)[1]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_if_changed(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(content)
    try:
        temporary.replace(path)
    except OSError:
        # A few Windows Python installs cannot rename beneath Unicode paths.
        path.write_bytes(content)
        temporary.unlink(missing_ok=True)


def write_page_data(directory: str | Path, runtime: Mapping[str, Any], knowledge: Mapping[str, Any]) -> dict[str, Any]:
    """Write runtime.json, knowledge.json and a reproducible manifest.

    The files contain only page-facing projections.  Completed experiments use
    their recorded completion time as the stable export timestamp, so repeated
    API refreshes do not rewrite otherwise identical files.
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    page_runtime, page_knowledge = project_page_data(runtime, knowledge)
    payloads = {
        "runtime.json": _json_bytes(page_runtime),
        "knowledge.json": _json_bytes(page_knowledge),
    }
    manifest = {
        "schema_version": 1,
        "generated_at": page_runtime.get("generated_at"),
        "source": {
            "experiment_id": (page_runtime.get("experiment") or {}).get("id"),
            "run_path": (page_runtime.get("experiment") or {}).get("output_dir"),
        },
        "counts": {
            "tasks": len(page_runtime.get("tasks") or []),
            "knowledge_records": len(page_knowledge.get("records") or []),
        },
        "files": {
            name: {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body)}
            for name, body in payloads.items()
        },
    }
    with _WRITE_LOCK:
        for name, body in payloads.items():
            _write_if_changed(target / name, body)
        _write_if_changed(target / "manifest.json", _json_bytes(manifest))
    return manifest
