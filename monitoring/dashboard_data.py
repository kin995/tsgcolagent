from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.experiment_outcome import default_solution, failure_category as classify_failure
from graph.experience_memory import AgentExperienceMemory, resolve_knowledge_store_path
from core.public_metrics import public_result, public_text
from .system_catalog import system_catalog
from metrics.paper_metrics import aggregate_metrics, direct_metrics


AGENT_NAMES = ["测试状态智能体", "测试生成智能体", "测试知识智能体"]
TASK_CACHE_DIR = Path(__file__).resolve().parents[1] / ".monitoring" / "task_cache"
TASK_CACHE_SCHEMA = 3
AGENT_MODULES = {
    "测试状态智能体": ["状态分析", "状态自修复", "质量评估", "缺口分析", "下一任务规划"],
    "测试生成智能体": ["测试生成", "可行性验证", "状态定位修复", "状态目标优化"],
    "测试知识智能体": ["经验检索", "经验学习", "知识问答"],
}

MODULE_HINTS = {
    "状态分析": ["source", "structure", "graph", "stateinitialized", "stateanalyzed"],
    "状态自修复": ["staterepaired", "stateinvalid", "repairscope"],
    "质量评估": ["evaluat", "quality", "graded"],
    "缺口分析": ["gap", "defect", "uncovered"],
    "下一任务规划": ["intent", "plan", "decision", "route"],
    "测试生成": ["generated", "generation", "candidatecreated"],
    "可行性验证": ["candidatevalid", "syntax", "normaliz", "execution", "validation", "coverage", "mutation", "assertion"],
    "状态定位修复": ["repairscope", "repaired", "patch"],
    "状态目标优化": ["optimiz", "enhance"],
    "经验检索": ["knowledgeretrieved"],
    "经验学习": ["knowledgeupdated"],
    "知识问答": ["conversation", "question"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def experiment_id(output_dir: str | Path) -> str:
    return hashlib.sha256(str(Path(output_dir).resolve()).encode("utf-8")).hexdigest()[:16]


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def build_snapshot(record: dict[str, Any], experiments: list[dict[str, Any]]) -> dict[str, Any]:
    root = Path(str(record.get("output_dir", ""))).resolve()
    batch = read_json(root / "batch_summary.json")
    batch_runtime = read_json(root / "batch_runtime.json")
    master = read_json(root / "all_datasets_summary.json")
    all_summary_paths, all_graph_paths = _task_artifacts(root)
    cache_key = _completed_task_cache_key(root, record, batch, master, all_summary_paths, all_graph_paths)
    tasks = _load_task_cache(root, cache_key) if cache_key else None
    if tasks is None:
        batch_results = batch.get("results") if isinstance(batch.get("results"), list) else []
        result_by_output = {
            str(Path(str(item.get("output_dir", ""))).resolve()).casefold(): item
            for item in batch_results
            if isinstance(item, dict) and item.get("output_dir")
        }
        tasks = [_task_from_summary(path, read_json(path), result_by_output.get(str(path.parent.resolve()).casefold())) for path in all_summary_paths]
        summary_dirs = {path.parent.resolve() for path in all_summary_paths}
        tasks.extend(_task_from_graph(path, read_json(path)) for path in all_graph_paths if path.parent.resolve() not in summary_dirs)
        tasks = [item for item in tasks if item]
        if master:
            for item in tasks:
                artifact = Path(str(item.get("summary_path") or item.get("graph_path") or ""))
                try:
                    relative = artifact.resolve().relative_to(root)
                    dataset_name = relative.parts[0] if len(relative.parts) > 1 else ""
                except (OSError, ValueError):
                    dataset_name = ""
                if dataset_name:
                    item["dataset_id"] = dataset_name
                    item["dataset_name"] = dataset_name
                    item["experiment_label"] = dataset_name
        else:
            dataset_name = str(record.get("label") or root.name)
            for item in tasks:
                item.setdefault("dataset_id", str(record.get("id") or dataset_name))
                item.setdefault("dataset_name", dataset_name)
        tasks.sort(key=lambda item: float(item.get("updated_at_epoch", 0)), reverse=True)
        if cache_key:
            _save_task_cache(root, cache_key, tasks)
    for item in tasks:
        failure = item.get("failure")
        if isinstance(failure, dict):
            failure["failed_tests"] = _failed_test_labels(failure.get("failed_tests"))
            for key in ("weak_flags", "previous_solutions", "suggested_solutions", "attempted_measures"):
                failure[key] = _public_failure_strings(failure.get(key))
    current = tasks[0] if tasks else _task_from_batch_result(batch)
    graph = _current_graph(current)
    events = _merged_live_events(tasks)
    if not events and graph:
        events = list(graph.get("flow_history", []) or [])[-120:]
    model_usage = _aggregate_model_usage(tasks)
    for item in tasks:
        item.pop("_flow_events", None)
        item.pop("_llm_usage", None)
    nodes, edges = _compact_graph(graph)
    stored_outcome = batch.get("outcome_summary") if isinstance(batch.get("outcome_summary"), dict) else {}
    if batch_runtime.get("selected_tasks"):
        stored_outcome = {**stored_outcome, "total_tasks": max(int(stored_outcome.get("total_tasks", 0) or 0), int(batch_runtime.get("selected_tasks", 0) or 0))}
    outcome = _reconciled_outcome(
        stored_outcome,
        tasks,
        count_missing_as_unfinished=str(batch_runtime.get("status") or "") == "completed",
    ) if tasks or stored_outcome else _single_outcome(current)
    quality = _task_quality_averages(tasks, max(len(tasks), int(outcome.get("total_tasks") or batch.get("selected_tasks") or 0))) if tasks else (batch.get("quality_overview") if isinstance(batch.get("quality_overview"), dict) else {})
    grade_counts = outcome.get("grade_counts", _grade_counts(tasks))
    graded_total = sum(int(grade_counts.get(grade, 0) or 0) for grade in "ABCDE")
    overall_quality_index = (
        sum(int(grade_counts.get(grade, 0) or 0) * weight for grade, weight in {"A": 1.0, "B": 0.82, "C": 0.62, "D": 0.2, "E": 0.0}.items()) / graded_total
        if graded_total else 0.0
    )
    completed_marker = read_json(root / "dataset_completion.json")
    activity_times = [float(item.get("updated_at_epoch", 0)) for item in tasks]
    if (root / "batch_summary.json").exists():
        activity_times.append((root / "batch_summary.json").stat().st_mtime)
    status = _experiment_status(batch or master, current, completed_marker, root / "dataset_completion.json", max(activity_times, default=0.0))
    if str(batch_runtime.get("status") or "") == "running":
        status = "running"
    if str(record.get("execution_status") or "") == "completed" and (master.get("finished_at") or batch.get("finished_at")):
        status = "completed"
    timing = _experiment_timing(record, batch, batch_runtime, master, status, tasks, root)
    catalog = system_catalog()
    phase_states = _phase_states(catalog["phases"], events, status)
    active_tasks = [item for item in tasks if item.get("status") == "running"]
    selected_tasks = max(int(batch_runtime.get("selected_tasks", 0) or 0), len(tasks))
    queued_tasks = max(0, selected_tasks - len(tasks))
    runtime_configs = [batch_runtime]
    if master and isinstance(master.get("results"), list):
        runtime_configs = []
        for entry in master["results"]:
            dataset = entry.get("dataset") if isinstance(entry, dict) and isinstance(entry.get("dataset"), dict) else {}
            name = str(dataset.get("name") or "")
            if name and (root / name).is_dir():
                runtime_configs.append(read_json(root / name / "batch_runtime.json"))
    workers = max((int(item.get("task_workers", 1) or 1) for item in runtime_configs), default=1)
    model_channels = max((int(item.get("llm_concurrency", 1) or 1) for item in runtime_configs), default=1)
    parallel_execution = {
        "task_workers": workers,
        "llm_concurrency": model_channels,
        "python_parallel": any(bool(item.get("python_parallel")) for item in runtime_configs),
        "java_parallel": any(bool(item.get("java_parallel")) for item in runtime_configs),
        "active_task_count": len(active_tasks),
        "queued_task_count": queued_tasks,
        "active_tasks": [
            {"id": item.get("id"), "name": item.get("name"), "round_label": item.get("round_label"), "updated_at": item.get("updated_at")}
            for item in active_tasks
        ],
        "knowledge_store": batch_runtime.get("knowledge_store") or str(resolve_knowledge_store_path()),
    }
    overview_language = "mixed" if master else batch.get("language") or _language_from_source(current.get("source") if current else "")
    run_llm = master.get("llm") if isinstance(master.get("llm"), dict) else batch.get("llm") if isinstance(batch.get("llm"), dict) else {}
    return public_result({
        "schema_version": 1,
        "generated_at": utc_now(),
        "experiment": {**record, "status": status},
        "experiments": experiments,
        "overview": {
            "language": overview_language,
            "model_name": run_llm.get("model"),
            "model_usage": model_usage,
            "mode": batch.get("mode") or "execute",
            "status": status,
            "total_tasks": outcome.get("total_tasks", len(tasks)),
            "completed_execution": outcome.get("completed_execution", sum(1 for item in tasks if item.get("grade") in {"A", "B", "C"})),
            "not_completed_execution": outcome.get("not_completed_execution", sum(1 for item in tasks if item.get("grade") in {"D", "E"})),
            "grade_counts": grade_counts,
            "completion_rate": outcome.get("completion_rate", 0.0),
            "followup_task_count": max(_followup_count(batch), sum(1 for item in tasks if int(item.get("round", 1) or 1) > 1)),
            "followup_attempts": max(_followup_attempts(batch), sum(max(0, int(item.get("round", 1) or 1) - 1) for item in tasks)),
            "quality_averages": quality,
            "overall_quality_index": round(overall_quality_index, 4),
            **timing,
            "scope": _scope_for(root, batch, master, current),
            "dataset_count": master.get("total_datasets"),
            "completed_datasets": master.get("completed_datasets"),
            "task_workers": parallel_execution["task_workers"],
            "active_task_count": parallel_execution["active_task_count"],
            "queued_task_count": parallel_execution["queued_task_count"],
        },
        "parallel_execution": parallel_execution,
        "catalog": catalog,
        "phase_states": phase_states,
        "toolchain": _toolchain(current, overview_language),
        "agents": _agent_states(events, status),
        "current_task": current,
        "tasks": tasks[:1000],
        "state_model": {
            "version": graph.get("version", graph.get("graph_version", 0)) if graph else 0,
            "iteration": graph.get("iteration", 0) if graph else 0,
            "nodes": nodes,
            "edges": edges,
            "metadata": _compact_metadata(graph.get("metadata", {}) if graph else {}),
        },
        "flow_events": events,
        "failures": batch.get("failure_analysis", {}),
    })


def _experiment_timing(
    record: dict[str, Any],
    batch: dict[str, Any],
    batch_runtime: dict[str, Any],
    master: dict[str, Any],
    inferred_status: str,
    tasks: list[dict[str, Any]] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    execution_status = str(record.get("execution_status") or inferred_status or "waiting")
    timing_source = master if master else batch_runtime if batch_runtime else batch
    started_at = (
        timing_source.get("first_started_at")
        or record.get("started_at")
        or timing_source.get("started_at")
        or (record.get("registered_at") if execution_status not in {"queued", "waiting"} else None)
    )
    active_session_started_at = (
        timing_source.get("active_session_started_at")
        or timing_source.get("current_session_started_at")
        or (record.get("started_at") if execution_status == "running" else None)
    )
    accumulated_before = float(
        timing_source.get("accumulated_duration_before_session")
        or (
            float(timing_source.get("cumulative_duration_seconds") or timing_source.get("duration_seconds") or 0.0)
            - float(timing_source.get("session_duration_seconds") or 0.0)
        )
        or 0.0
    )
    terminal = execution_status in {"completed", "failed"}
    finished_at = None
    if terminal:
        finished_at = record.get("finished_at") or master.get("finished_at") or batch.get("finished_at")
    if terminal:
        duration_seconds = float(timing_source.get("cumulative_duration_seconds") or timing_source.get("duration_seconds") or 0.0)
        if duration_seconds <= 0 and finished_at:
            duration_seconds = _elapsed_seconds(started_at, finished_at) or 0.0
    else:
        active_elapsed = _elapsed_seconds(active_session_started_at) or 0.0
        duration_seconds = round(accumulated_before + active_elapsed, 3)
    historical_duration = historical_task_duration(tasks or [], root, bool(master), batch_runtime)
    duration_source = "recorded_sessions"
    if historical_duration > duration_seconds:
        duration_seconds = historical_duration
        duration_source = "historical_task_runtime"
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "active_session_started_at": active_session_started_at,
        "accumulated_duration_before_session": round(accumulated_before, 3),
        "session_duration_seconds": timing_source.get("session_duration_seconds"),
        "duration_source": duration_source,
    }


def historical_task_duration(
    tasks: list[dict[str, Any]],
    root: Path | None,
    is_master: bool = False,
    batch_runtime: dict[str, Any] | None = None,
) -> float:
    """Recover active experiment time from task runtimes when a resume summary only records the last short session."""
    if not tasks:
        return 0.0
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        name = str(task.get("dataset_name") or task.get("experiment_label") or "current")
        grouped.setdefault(name, []).append(task)
    total = 0.0
    for name, items in grouped.items():
        runtime = batch_runtime or {}
        if is_master and root is not None:
            runtime = read_json(root / name / "batch_runtime.json")
        language = "java" if any(str(item.get("source") or "").lower().endswith(".java") for item in items) else "python"
        workers = (
            max(1, int(runtime.get("task_workers", 2) or 2))
            if language == "java" and runtime.get("java_parallel")
            else 1
            if language == "java"
            else max(1, int(runtime.get("task_workers", 4) or 4))
        )
        # A single imported task cannot have occupied all configured worker
        # slots. Never divide its elapsed time by idle capacity.
        workers = min(workers, len(items))
        task_seconds = sum(max(0.0, float(item.get("duration_seconds") or 0.0)) for item in items)
        total += task_seconds / workers
    return round(total, 3)


def historical_duration_by_dataset(tasks: list[dict[str, Any]], root: Path) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        name = str(task.get("dataset_name") or task.get("experiment_label") or "")
        if name:
            grouped.setdefault(name, []).append(task)
    return {
        name: historical_task_duration(items, root / name, False, read_json(root / name / "batch_runtime.json"))
        for name, items in grouped.items()
    }


def _elapsed_seconds(started_at: object, finished_at: object | None = None) -> float | None:
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(finished_at).replace("Z", "+00:00")) if finished_at else datetime.now(timezone.utc)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)
        return round(max(0.0, (finished - started).total_seconds()), 3)
    except (TypeError, ValueError):
        return None


def _task_timing(root: Path) -> dict[str, Any]:
    payload = read_json(root / "task_runtime.json")
    sessions = [item for item in payload.get("sessions", []) if isinstance(item, dict)] if isinstance(payload.get("sessions"), list) else []
    accumulated = sum(float(item.get("duration_seconds", 0.0) or 0.0) for item in sessions if item.get("status") != "running")
    active = next((item for item in reversed(sessions) if item.get("status") == "running"), None)
    active_elapsed = _elapsed_seconds(active.get("started_at")) if active else 0.0
    starts = [str(item.get("started_at")) for item in sessions if item.get("started_at")]
    finishes = [str(item.get("finished_at")) for item in sessions if item.get("finished_at")]
    return {
        "started_at": min(starts) if starts else None,
        "finished_at": max(finishes) if finishes else None,
        "duration_seconds": round(accumulated + float(active_elapsed or 0.0), 3),
        "accumulated_duration_seconds": round(accumulated, 3),
        "active_session_started_at": active.get("started_at") if active else None,
        "attempt_count": len(sessions),
        "sessions": sessions,
    }
def _knowledge_usage(item: dict[str, Any]) -> str:
    text = " ".join(str(item.get(key) or "") for key in (
        "topic", "action", "dominant_gap", "failure_category", "solution", "rejection_reason", "record_type"
    )).casefold()
    rules = [
        (("assert", "oracle", "expected"), "断言设计、预期结果与断言失效问题"),
        (("mutation", "mutant"), "变异检出不足与弱断言增强"),
        (("coverage", "branch", "path", "uncovered"), "覆盖路径、分支与边界场景补强"),
        (("timeout", "performance", "slow"), "超时、性能与测试执行效率问题"),
        (("compile", "import", "package", "dependency", "java_tool"), "编译、依赖、包名与导入错误"),
        (("exception", "error_handling"), "异常路径与错误处理测试"),
        (("state", "sfc", "tsq"), "测试状态分析与覆盖检查"),
        (("generate_test_plan", "test_plan", "input"), "测试场景、输入数据与候选用例规划"),
        (("execution", "runtime", "candidate_rejected"), "执行失败定位与候选测试修复"),
    ]
    for keywords, usage in rules:
        if any(keyword in text for keyword in keywords):
            return usage
    return "相似测试任务的分析、生成与质量修复"


def knowledge_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path).expanduser().resolve() if path else resolve_knowledge_store_path()
    records = AgentExperienceMemory(path).load()
    compact = []
    for item in reversed(records[-500:]):
        if not isinstance(item, dict) or item.get("record_type") == "batch_summary":
            continue
        accepted = bool(item.get("accepted") or item.get("candidate_status") == "accepted")
        learning_stage = str(item.get("learning_stage") or "")
        status = "已掌握" if accepted and float(item.get("reward") or 0) >= 0 else "待复核"
        if learning_stage == "learning":
            status = "正在学习"
        elif learning_stage in {"adopted", "mastered"}:
            status = "已掌握"
        elif learning_stage == "pending_validation":
            status = "待复核"
        if item.get("rejection_reason") or item.get("candidate_status") == "rejected" or learning_stage == "avoid":
            status = "应避免"
        compact.append({
            "id": item.get("experience_id") or item.get("strategy_fingerprint") or str(item.get("timestamp", "")),
            "status": status,
            "language": item.get("language") or (item.get("task_signature") or {}).get("language"),
            "agent": _knowledge_owner(item.get("agent")),
            "action": item.get("action") or item.get("dominant_gap") or item.get("record_type"),
            "failure_category": item.get("failure_category"),
            "accepted": accepted,
            "source": item.get("source"),
            "timestamp": item.get("timestamp"),
            "solution": item.get("solution") or item.get("rejection_reason"),
            "question": item.get("question"),
            "usage": _knowledge_usage(item),
            "topic": item.get("topic"),
            "record_type": item.get("record_type"),
            "learning_stage": learning_stage,
            "validation_count": int(item.get("validation_count", 0) or 0),
            "validation_failure_count": int(item.get("validation_failure_count", 0) or 0),
            "support_count": int(item.get("support_count", 0) or 0),
            "independent_task_count": int(item.get("independent_task_count", 0) or 0),
        })
    counts = {name: sum(1 for item in compact if item["status"] == name) for name in ["已掌握", "正在学习", "待复核", "应避免"]}
    return public_result({"path": str(path), "total": len(compact), "stored_record_count": len(records), "counts": counts, "records": compact})


def _task_artifacts(root: Path) -> tuple[list[Path], list[Path]]:
    """Find task summaries and graphs without descending into generated tool workspaces."""
    summaries: list[Path] = []
    graphs: list[Path] = []
    if not root.exists():
        return summaries, graphs
    for parent, directories, files in os.walk(root):
        directories[:] = [name for name in directories if name not in {
            ".git", ".java_workspace", ".langgraph", ".pytest_cache", ".knowledge_snapshots",
            "node_modules", "target", "__pycache__",
        }]
        if "stateflow_summary.json" in files:
            summaries.append(Path(parent) / "stateflow_summary.json")
        if "state_flow_graph.json" in files:
            graphs.append(Path(parent) / "state_flow_graph.json")
    return sorted(summaries), sorted(graphs)


def _completed_task_cache_key(
    root: Path,
    record: dict[str, Any],
    batch: dict[str, Any],
    master: dict[str, Any],
    summaries: list[Path],
    graphs: list[Path],
) -> str | None:
    if str(record.get("execution_status") or "") != "completed" or not (master.get("finished_at") or batch.get("finished_at")):
        return None
    revisions = [root / "all_datasets_summary.json", root / "batch_summary.json", *summaries, *graphs]
    for entry in master.get("results", []) if isinstance(master.get("results"), list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("dataset"), dict):
            name = str(entry["dataset"].get("name") or "")
            if name and (root / name).is_dir():
                revisions.append(root / name / "batch_summary.json")
    digest = hashlib.sha256(str(root.resolve()).encode("utf-8"))
    digest.update(f"task-cache-v{TASK_CACHE_SCHEMA}\n".encode("ascii"))
    for path in revisions:
        try:
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}:{stat.st_mtime_ns}:{stat.st_size}\n".encode("utf-8"))
        except OSError:
            continue
    return digest.hexdigest()


def _task_cache_path(root: Path) -> Path:
    name = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:20]
    return TASK_CACHE_DIR / f"{name}.json"


def _load_task_cache(root: Path, cache_key: str) -> list[dict[str, Any]] | None:
    cached = read_json(_task_cache_path(root))
    tasks = cached.get("tasks")
    return tasks if cached.get("key") == cache_key and isinstance(tasks, list) else None


def _save_task_cache(root: Path, cache_key: str, tasks: list[dict[str, Any]]) -> None:
    path = _task_cache_path(root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        content = json.dumps({"key": cache_key, "tasks": public_result(tasks)}, ensure_ascii=False)
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(path)
        except OSError:
            # Windows can reject a same-directory replace under a non-ASCII user profile.
            path.write_text(content, encoding="utf-8")
            temporary.unlink(missing_ok=True)
    except OSError:
        pass


def _task_from_summary(path: Path, data: dict[str, Any], batch_item: dict[str, Any] | None = None) -> dict[str, Any]:
    if not data:
        return {}
    report = data.get("final_report") if isinstance(data.get("final_report"), dict) else {}
    batch_item = batch_item or {}
    local_graph = path.with_name("state_flow_graph.json")
    graph_path = local_graph if local_graph.is_file() else Path(str(data.get("graph_path") or local_graph))
    round_progress_path = path.parent / "experiment_round_progress.json"
    round_progress = read_json(round_progress_path)
    graph_updated = graph_path.stat().st_mtime if graph_path.exists() else 0.0
    summary_updated = path.stat().st_mtime
    round_updated = round_progress_path.stat().st_mtime if round_progress_path.exists() else 0.0
    graph_data = read_json(graph_path) if graph_updated > summary_updated + 0.05 else {}
    live_report = (graph_data.get("metadata") or {}).get("last_report") if isinstance(graph_data.get("metadata"), dict) else None
    if isinstance(live_report, dict):
        report = live_report
    experiment_outcome = data.get("experiment_outcome") if isinstance(data.get("experiment_outcome"), dict) else {}
    if not experiment_outcome and isinstance(data.get("round_failure_diagnostic"), dict):
        experiment_outcome = data["round_failure_diagnostic"]
    if not experiment_outcome and isinstance(batch_item.get("round_failure_diagnostic"), dict):
        experiment_outcome = batch_item["round_failure_diagnostic"]
    grade_hint = str(batch_item.get("final_grade") or data.get("final_grade") or "")
    if not experiment_outcome and grade_hint in {"D", "E"} and graph_path.exists():
        historical_graph = graph_data or read_json(graph_path)
        metadata = historical_graph.get("metadata") if isinstance(historical_graph.get("metadata"), dict) else {}
        experiment_outcome = _failure_outcome_from_graph(metadata, data, batch_item)
    visualization = data.get("visualization") if isinstance(data.get("visualization"), dict) else {}
    failure = _failure_details(report, experiment_outcome, {**data, **batch_item})
    failure["failure_report"] = (
        visualization.get("failure_report_html")
        or batch_item.get("failure_report")
        or experiment_outcome.get("failure_report")
    )
    if failure.get("failure_report"):
        report_name = Path(str(failure["failure_report"])).name
        local_report = path.parent / report_name
        if local_report.exists():
            failure["failure_report"] = str(local_report.resolve())
    elif (path.parent / "task_failure_report.html").exists():
        failure["failure_report"] = str((path.parent / "task_failure_report.html").resolve())
    progress_completed = str(round_progress.get("status") or "") == "completed"
    grade = (
        round_progress.get("final_grade") if progress_completed else None
    ) or batch_item.get("final_grade") or data.get("final_grade") or _grade_for(data.get("quality_status") or report.get("quality_status"), report)
    # Internal repair iterations and experiment rounds are different concepts.
    round_number = max(
        int(batch_item.get("experiment_round") or 1),
        int(data.get("experiment_round") or data.get("final_round") or 1),
        int(round_progress.get("experiment_round") or 1),
    )
    progress_running = str(round_progress.get("status") or "") == "running" and round_number >= int(batch_item.get("experiment_round") or 1)
    explicit_running = str(batch_item.get("status") or data.get("status") or data.get("quality_status") or "") == "running"
    task_status = (
        "running"
        if progress_running or explicit_running
        else round_progress.get("result_status") if progress_completed and round_progress.get("result_status")
        else batch_item.get("status") or data.get("quality_status") or report.get("quality_status") or data.get("stop_reason")
    )
    timing = _task_timing(path.parent)
    duration = (
        timing.get("duration_seconds")
        if timing.get("attempt_count")
        else batch_item.get("duration_seconds") or data.get("cumulative_duration_seconds") or data.get("duration_seconds") or data.get("runtime_seconds") or report.get("runtime_seconds")
    )
    test_path = _resolved_test_path(path.parent, data.get("test_path") or report.get("test_path"), data.get("source"))
    source = _display_source_path(data.get("source"))
    measurement = read_json(path.parent / "paper_metrics" / "summary.json") or data.get("paper_metrics") or {}
    if measurement.get("source_hash") and data.get("source_sha256") and measurement["source_hash"] != data["source_sha256"]:
        measurement = {}
    paper_values = {key: value for key, value in measurement.items() if key.startswith(("exec_", "tir", "knowledge_")) or key in {"avg_exec", "measurement_version"}}
    if measurement.get("duration_seconds") is not None:
        duration = max(float(duration or 0), float(measurement["duration_seconds"]))
    return {
        "id": hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:16],
        "name": Path(str(source or path.parent.name)).name,
        "source": source,
        "scope": "class" if str(source or "").lower().endswith(".java") else "file",
        "output_dir": str(path.parent.resolve()),
        "summary_path": str(path.resolve()),
        "graph_path": str(graph_path),
        "test_path": test_path,
        "test_artifact_available": bool(test_path and Path(str(test_path)).exists()),
        "failure_report": failure.get("failure_report"),
        "status": task_status,
        "grade": str(grade),
        "round": int(round_number),
        "round_label": f"{grade}·R{round_number}",
        "iterations": data.get("iterations", 0),
        "grade_history": data.get("grade_history") if isinstance(data.get("grade_history"), list) else batch_item.get("grade_history", []),
        "round_strategy": round_progress.get("strategy") or batch_item.get("round_strategy") or data.get("followup_strategy"),
        "round_status": round_progress.get("status"),
        "reports_count": len(data.get("reports", []) or []),
        "duration_seconds": duration,
        "started_at": timing.get("started_at") or batch_item.get("started_at") or data.get("started_at"),
        "finished_at": timing.get("finished_at") or batch_item.get("finished_at") or data.get("finished_at"),
        "attempt_count": timing.get("attempt_count") or batch_item.get("attempt_count") or data.get("attempt_count"),
        "timing_history": timing.get("sessions") or data.get("timing", {}).get("sessions", []) if isinstance(data.get("timing"), dict) else timing.get("sessions", []),
        "stop_reason": data.get("stop_reason"),
        "metrics": {
            **_metrics(report),
            **paper_values,
            **{key: value for key, value in direct_metrics(report, duration).items() if value is not None},
        },
        "failure": failure,
        "updated_at": datetime.fromtimestamp(max(summary_updated, graph_updated, round_updated), timezone.utc).isoformat(),
        "updated_at_epoch": max(summary_updated, graph_updated, round_updated),
        "_flow_events": list(graph_data.get("flow_history", []) or [])[-40:] if graph_data else [],
        "_llm_usage": {key: data["llm_usage"].get(key) for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "duration_seconds")}
        if isinstance(data.get("llm_usage"), dict) else None,
    }


def _resolved_test_path(task_root: Path, declared: object, source: object) -> str | None:
    if declared:
        candidate = Path(str(declared))
        if candidate.exists():
            return str(candidate.resolve())
    pattern = "*Test.java" if str(source or "").lower().endswith(".java") else "test_*.py"
    local = next((item for item in sorted(task_root.glob(pattern)) if item.is_file()), None)
    return str(local.resolve()) if local else (str(declared) if declared else None)


def _display_source_path(declared: object) -> str | None:
    if not declared:
        return None
    source = str(declared)
    path = Path(source)
    if not path.is_absolute() or path.exists():
        return source
    normalized = source.replace("\\", "/")
    marker = "/datasets/"
    index = normalized.casefold().rfind(marker)
    return normalized[index + len(marker):] if index >= 0 else path.name


def _task_from_graph(path: Path, graph: dict[str, Any]) -> dict[str, Any]:
    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    report = metadata.get("last_report") if isinstance(metadata.get("last_report"), dict) else {}
    source = graph.get("source_path") or metadata.get("source_path") or path.parent.name
    failure = _failure_details(report, {}, metadata)
    return {
        "id": hashlib.sha256(str(path.parent.resolve()).encode("utf-8")).hexdigest()[:16],
        "name": Path(str(source)).name,
        "source": source,
        "output_dir": str(path.parent.resolve()),
        "summary_path": None,
        "graph_path": str(path.resolve()),
        "test_path": report.get("test_path") or metadata.get("test_path"),
        "status": "running",
        "grade": "E",
        "round": 1,
        "round_label": "运行中",
        "iterations": graph.get("iteration", 0),
        "stop_reason": None,
        "metrics": _metrics(report),
        "failure": failure,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "updated_at_epoch": path.stat().st_mtime,
        "_flow_events": list(graph.get("flow_history", []) or [])[-40:],
    }


def _merged_live_events(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    active = [item for item in tasks if item.get("status") == "running"][:8]
    for task in active:
        for event in task.get("_flow_events", []) or []:
            if isinstance(event, dict):
                merged.append({**event, "task_id": task.get("id"), "task_name": task.get("name")})
    merged.sort(key=lambda item: str(item.get("timestamp") or ""))
    return merged[-120:]


def _task_from_batch_result(batch: dict[str, Any]) -> dict[str, Any] | None:
    results = batch.get("results") if isinstance(batch.get("results"), list) else []
    if not results:
        return None
    item = results[-1] if isinstance(results[-1], dict) else {}
    summary_path = Path(str(item.get("summary_path", "")))
    return _task_from_summary(summary_path, read_json(summary_path), item) if summary_path.exists() else item


def _current_graph(current: dict[str, Any] | None) -> dict[str, Any]:
    if not current:
        return {}
    path = Path(str(current.get("graph_path", "")))
    return read_json(path) if path.exists() else {}


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "tests_passed": report.get("pytest_passed"),
        "collected_count": report.get("collected_count"),
        "line_coverage": report.get("line_coverage"),
        "branch_coverage": report.get("branch_coverage"),
        "ae": report.get("ae"),
        "mutation_score": report.get("mutation_score"),
        "assertion_count": report.get("assertion_count"),
        "effective_assertions": report.get("effective_assertions"),
        "effective_mutants": report.get("effective_mutants"),
        "mutation_reliability": report.get("mutation_reliability"),
    }


def _failure_details(report: dict[str, Any], outcome: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    category = str(outcome.get("failure_category") or report.get("failure_category") or fallback.get("failure_category") or "").strip()
    error = str(
        outcome.get("error")
        or report.get("failure_detail")
        or report.get("error")
        or report.get("stderr")
        or fallback.get("error")
        or fallback.get("last_error")
        or ""
    ).strip()
    valid_result = bool(report.get("pytest_passed"))
    if valid_result and category in {"valid_baseline", "none", "not_applicable"}:
        category = ""
        error = ""
    if not category and (error or fallback.get("status") in {"failed", "generation_failed", "execution_invalid"}):
        category = classify_failure({**fallback, "final_report": report, "error": error})
    reason = str(
        outcome.get("failure_reason")
        or fallback.get("failure_reason")
        or report.get("repair_failure_reason")
        or error
        or (f"{category}：测试未形成有效执行结果" if category else "")
    ).strip()
    if valid_result and not category:
        reason = ""
    previous = _public_failure_strings(outcome.get("previous_solutions"), fallback.get("previous_solutions"))
    suggestions = _public_failure_strings(
        outcome.get("suggested_solutions"),
        report.get("repair_suggestions"),
        fallback.get("repair_suggestions"),
        [default_solution(category or "unknown")] if category or reason else [],
    )
    grade_history = outcome.get("grade_history") if isinstance(outcome.get("grade_history"), list) else fallback.get("grade_history", [])
    attempted = _public_failure_strings(outcome.get("attempted_measures"), fallback.get("attempted_measures"))
    for item in grade_history if isinstance(grade_history, list) else []:
        if not isinstance(item, dict):
            continue
        summary = " · ".join(str(value) for value in [f"R{item.get('round', 1)}", item.get("grade"), item.get("status"), item.get("failure_reason")] if value)
        if summary:
            attempted.append(summary)
    raw_knowledge_history = outcome.get("knowledge_history", []) if isinstance(outcome.get("knowledge_history"), list) else fallback.get("knowledge_history", [])
    compact_knowledge_history = []
    for item in raw_knowledge_history[-20:] if isinstance(raw_knowledge_history, list) else []:
        if isinstance(item, dict):
            compact_knowledge_history.append({key: item.get(key) for key in ("experience_id", "topic", "action", "failure_category", "accepted", "source") if item.get(key) is not None})
        elif item:
            compact_knowledge_history.append(str(item)[:240])
    return {
        "category": category or None,
        "stage": outcome.get("failure_stage") or fallback.get("failure_stage"),
        "failed_tests": _failed_test_labels(report.get("failed_tests")),
        "weak_flags": _public_failure_strings(report.get("weak_quality_flags")),
        "threshold_failures": dict(report.get("threshold_failures", {}) or {}),
        "reason": reason or None,
        "error": error or None,
        "previous_solutions": previous,
        "suggested_solutions": suggestions,
        "attempted_measures": attempted,
        "knowledge_history": compact_knowledge_history,
        "knowledge_record_id": outcome.get("knowledge_record_id") or fallback.get("knowledge_record_id"),
    }


def _failure_outcome_from_graph(metadata: dict[str, Any], data: dict[str, Any], batch_item: dict[str, Any]) -> dict[str, Any]:
    validation = metadata.get("test_generation_candidate_rejected")
    if not isinstance(validation, dict):
        candidate = metadata.get("test_generation_validation")
        validation = candidate if isinstance(candidate, dict) and candidate.get("accepted") is False else {}
    error = str(
        data.get("last_error")
        or metadata.get("llm_generation_error")
        or validation.get("pytest_output")
        or ""
    ).strip()
    category = str(validation.get("failure_category") or "").strip()
    if not category:
        category = classify_failure(
            {
                **batch_item,
                "status": batch_item.get("status") or data.get("quality_status"),
                "last_error": error,
                "stop_reason": data.get("stop_reason"),
            }
        )
    stage = str(validation.get("stage") or ("test_generation" if str(batch_item.get("final_grade")) == "E" else "test_execution"))
    detail = error.splitlines()[0][:300] if error else f"任务在{stage}阶段停止，未形成完整可执行结果"
    attempted = []
    repair_attempts = int(data.get("orchestration", {}).get("state_repair_attempts", 0) or 0) if isinstance(data.get("orchestration"), dict) else 0
    if repair_attempts:
        attempted.append(f"测试状态智能体已执行{repair_attempts}次状态自修复")
    if validation:
        attempted.append(f"测试生成智能体已执行候选验证（阶段：{stage}）")
    return {
        "failure_category": category,
        "failure_reason": f"{category}: {detail}",
        "failure_stage": stage,
        "error": error or detail,
        "attempted_measures": attempted,
        "suggested_solutions": [default_solution(category)],
    }


def _unique_strings(*values: object) -> list[str]:
    result: list[str] = []
    for value in values:
        items = value if isinstance(value, list) else [value] if isinstance(value, str) else []
        for item in items:
            text = str(item).strip()
            if text and text not in result:
                result.append(text)
    return result


def _public_failure_strings(*values: object) -> list[str]:
    return [item for item in _unique_strings(*values) if public_text(item) == item and "内部过程指标已隐藏" not in item]


def _failed_test_labels(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for item in value[:30]:
        if isinstance(item, dict):
            text = str(item.get("test_name") or item.get("name") or item.get("nodeid") or "").strip()
        else:
            text = str(item).strip() if isinstance(item, str) else ""
        if text and text not in labels:
            labels.append(text)
    return labels


def _compact_graph(graph: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_nodes = graph.get("nodes", {}) if isinstance(graph.get("nodes"), dict) else {}
    nodes = []
    for node_id, node in list(raw_nodes.items())[:400]:
        if not isinstance(node, dict):
            continue
        metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
        mutation_status = str(node.get("mutation_status") or "")
        coverage_status = str(node.get("coverage_status") or "")
        evidenced = bool(
            node.get("visited")
            or int(node.get("visit_count") or 0) > 0
            or coverage_status == "covered"
            or mutation_status in {"killed", "survived"}
            or isinstance(metadata.get("effective"), bool)
        )
        nodes.append({
            "id": node_id,
            "type": node.get("node_type"),
            "name": node.get("name"),
            "visited": node.get("visited", False),
            "evidenced": evidenced,
            "coverage_status": coverage_status or None,
            "mutation_status": mutation_status or None,
            "assertion_effective": metadata.get("effective"),
            "repair_flag": bool(node.get("repair_flag")),
            "line": node.get("line_start"),
        })
    edges = []
    for edge in list(graph.get("edges", []) or [])[:800]:
        if isinstance(edge, dict):
            edges.append({"id": edge.get("edge_id"), "source": edge.get("source"), "target": edge.get("target"), "type": edge.get("edge_type"), "visited": edge.get("visited", False)})
    return nodes, edges


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return public_result({key: metadata.get(key) for key in ["language", "last_test_intent_plan", "last_evaluation_feedback", "last_knowledge_context", "last_knowledge_outcome", "last_report"] if key in metadata})


def state_model_artifact(path: str | Path) -> dict[str, Any]:
    """Return the current shared TSG state in the compact live-dashboard shape."""
    graph = read_json(Path(path))
    nodes, edges = _compact_graph(graph)
    return public_result({
        "version": graph.get("version", graph.get("graph_version", 0)),
        "iteration": graph.get("iteration", 0),
        "source_path": graph.get("source_path"),
        "nodes": nodes,
        "edges": edges,
        "metadata": _compact_metadata(graph.get("metadata", {}) if isinstance(graph.get("metadata"), dict) else {}),
    })


def _agent_states(events: list[dict[str, Any]], experiment_status: str) -> list[dict[str, Any]]:
    result = []
    latest_source = str(events[-1].get("source_agent", "")) if events else ""
    for name in AGENT_NAMES:
        participated = [event for event in events if event.get("source_agent") == name or event.get("target_agent") == name]
        emitted = [event for event in events if event.get("source_agent") == name]
        last = participated[-1] if participated else {}
        interaction_role = "executed" if last.get("source_agent") == name else ("received" if last else None)
        status = "active" if experiment_status == "running" and latest_source == name else ("done" if participated else "standby")
        module_states = []
        for module in AGENT_MODULES[name]:
            hints = MODULE_HINTS.get(module, [])
            matches = [event for event in emitted if any(hint in str(event.get("event_type", "")).lower() for hint in hints)]
            module_states.append({
                "name": module,
                "status": "active" if status == "active" and matches and matches[-1] is emitted[-1] else ("done" if matches else "pending"),
                "event_count": len(matches),
                "last_event": matches[-1] if matches else None,
            })
        result.append({
            "id": name,
            "name": name,
            "status": status,
            "modules": AGENT_MODULES[name],
            "internal_tasks": module_states,
            "current_action": last.get("event_type"),
            "interaction_role": interaction_role,
            "last_event": last,
            "event_count": len(emitted),
        })
    return result


def _single_outcome(current: dict[str, Any] | None) -> dict[str, Any]:
    grade = current.get("grade", "E") if current else "E"
    valid = grade in {"A", "B", "C"}
    return {"total_tasks": 1 if current else 0, "completed_execution": int(valid), "not_completed_execution": int(bool(current) and not valid), "completion_rate": 1.0 if valid else 0.0, "grade_counts": {key: int(key == grade) for key in "ABCDE"}}


def _grade_counts(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return {
        grade: sum(1 for item in tasks if item.get("grade") == grade and item.get("status") != "running")
        for grade in "ABCDE"
    }


def _reconciled_outcome(
    stored: dict[str, Any],
    tasks: list[dict[str, Any]],
    *,
    count_missing_as_unfinished: bool = True,
) -> dict[str, Any]:
    counts = _grade_counts(tasks)
    expected_total = max(int(stored.get("total_tasks", 0) or 0), len(tasks))
    missing = max(0, expected_total - len(tasks))
    if count_missing_as_unfinished:
        counts["E"] += missing
    valid = counts["A"] + counts["B"] + counts["C"]
    return {
        **stored,
        "total_tasks": expected_total,
        "valid_results": valid,
        "invalid_results": counts["D"],
        "unfinished_results": counts["E"],
        "completed_execution": valid,
        "not_completed_execution": counts["D"] + counts["E"],
        "pending_results": 0 if count_missing_as_unfinished else missing + sum(1 for item in tasks if item.get("status") == "running"),
        "completion_rate": round(valid / expected_total, 4) if expected_total else 0.0,
        "grade_counts": counts,
        "balance_valid": expected_total == sum(counts.values()),
    }


def _task_quality_averages(tasks: list[dict[str, Any]], selected_tasks: int | None = None) -> dict[str, Any]:
    metrics = [
        item.get("metrics")
        for item in tasks
        if item.get("grade") in {"A", "B", "C"}
        and isinstance(item.get("metrics"), dict)
        and item["metrics"].get("tests_passed")
    ]

    def average(key: str) -> float | None:
        values = [float(item[key]) for item in metrics if item.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "executed_tasks": len(metrics),
        **{key: value for key, value in aggregate_metrics(tasks, selected_tasks).items()
           if not key.startswith("average_") or key == "average_time_seconds"},
        "average_line_coverage": average("line_coverage"),
        "average_branch_coverage": average("branch_coverage"),
        "average_ae": average("ae"),
        "average_mutation_score": average("mutation_score"),
    }


def _aggregate_model_usage(tasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    totals = {key: 0 for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens")}
    duration = 0.0
    recorded = 0
    for task in tasks:
        usage = task.get("_llm_usage")
        if not isinstance(usage, dict):
            continue
        recorded += 1
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)) and value >= 0:
                totals[key] += int(value)
        value = usage.get("duration_seconds")
        if isinstance(value, (int, float)) and value >= 0:
            duration += float(value)
    return {"recorded_tasks": recorded, **totals, "duration_seconds": round(duration, 3)} if recorded else None


def _grade_for(status: object, report: dict[str, Any]) -> str:
    mapping = {"quality_high": "A", "quality_acceptable": "B", "valid_usable": "C", "execution_invalid": "D"}
    if status in mapping:
        return mapping[str(status)]
    if report.get("pytest_passed") and report.get("collected_count", 0):
        return "C"
    return "D" if report else "E"


def _language_from_source(source: object) -> str:
    return "java" if str(source or "").lower().endswith(".java") else "python"


def _followup_count(batch: dict[str, Any]) -> int:
    summary = batch.get("experiment_round_summary") if isinstance(batch.get("experiment_round_summary"), dict) else {}
    return int(summary.get("followup_task_count") or (batch.get("retry_summary") or {}).get("unique_tasks") or 0)


def _followup_attempts(batch: dict[str, Any]) -> int:
    summary = batch.get("experiment_round_summary") if isinstance(batch.get("experiment_round_summary"), dict) else {}
    return int(summary.get("attempts") or 0)


def _experiment_status(
    batch: dict[str, Any],
    current: dict[str, Any] | None,
    marker: dict[str, Any],
    marker_path: Path,
    latest_activity: float,
) -> str:
    marker_is_current = marker_path.exists() and marker_path.stat().st_mtime >= latest_activity
    if marker_path.exists() and (marker.get("complete") is True or marker.get("completed") is True or marker.get("skip_ready") is True):
        return "completed"
    if marker_is_current and batch and batch.get("dataset_completion"):
        completion = batch.get("dataset_completion")
        if isinstance(completion, dict) and (completion.get("complete") or completion.get("completed")):
            return "completed"
    if current and current.get("status"):
        if current.get("status") == "running":
            return "running"
        return "completed" if not batch else "running"
    return "running" if batch or current else "waiting"


def _knowledge_owner(agent: object) -> str:
    name = str(agent or "")
    if name in {"TestKnowledgeAgent", "测试知识智能体"}:
        return "测试知识智能体"
    if name in {"TestStateAgent", "PlanningAgent", "EvaluationAgent", "测试状态智能体"}:
        return "测试状态智能体"
    if name:
        return "测试生成智能体"
    return "系统归纳"


def _phase_states(phases: list[dict[str, Any]], events: list[dict[str, Any]], experiment_status: str) -> list[dict[str, Any]]:
    phase_hints = {
        "discover": ["source", "structure", "context"],
        "state": ["graph", "state", "node", "edge"],
        "retrieve": ["knowledge"],
        "plan": ["intent", "plan", "decision"],
        "generate": ["generat", "candidate", "normaliz"],
        "execute": ["execution", "validation", "coverage", "mutation", "assertion"],
        "evaluate": ["evaluat", "quality", "gap", "repairscope"],
        "learn": ["knowledgeupdated"],
    }
    states = []
    latest_type = str(events[-1].get("event_type", "")).lower() if events else ""
    for phase in phases:
        hints = phase_hints.get(str(phase.get("id")), [])
        matches = [event for event in events if any(hint in str(event.get("event_type", "")).lower() for hint in hints)]
        active = experiment_status == "running" and any(hint in latest_type for hint in hints)
        states.append({**phase, "status": "active" if active else ("done" if matches else "pending"), "event_count": len(matches), "last_event": matches[-1] if matches else None})
    return states


def _toolchain(current: dict[str, Any] | None, language: str) -> list[dict[str, Any]]:
    metrics = current.get("metrics", {}) if current else {}
    if language == "mixed":
        tools = [("pytest / JUnit 5", True), ("coverage.py / JaCoCo", metrics.get("line_coverage")), ("Python Mutation / PIT", metrics.get("mutation_score")), ("断言评估 AE", metrics.get("ae")), ("分支覆盖 BC", metrics.get("branch_coverage"))]
    elif language == "java":
        tools = [("JUnit 5", metrics.get("tests_passed")), ("JaCoCo", metrics.get("line_coverage")), ("PIT", metrics.get("mutation_score")), ("断言评估 AE", metrics.get("ae")), ("目标改善 TIR", metrics.get("tir"))]
    else:
        tools = [("pytest", metrics.get("tests_passed")), ("coverage.py", metrics.get("line_coverage")), ("变异执行器", metrics.get("mutation_score")), ("断言评估 AE", metrics.get("ae")), ("目标改善 TIR", metrics.get("tir"))]
    return [{"name": name, "status": "ready" if value is not None else "waiting", "value": value} for name, value in tools]


def _scope_for(root: Path, batch: dict[str, Any], master: dict[str, Any], current: dict[str, Any] | None) -> str:
    if master:
        return "datasets"
    if batch:
        return "project"
    source = str((current or {}).get("source") or "")
    if source.lower().endswith(".java"):
        return "class"
    return "file"
