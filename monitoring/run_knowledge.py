"""Recover the knowledge retained in an imported multi-dataset experiment.

The canonical knowledge file referenced by old run summaries may have been
copied independently (or may still be an earlier baseline).  Snapshots and
round journals are the evidence retained with the run itself.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from core.public_metrics import public_result, public_text
from graph.experience_memory import AgentExperienceMemory

from .dashboard_data import _knowledge_owner, _knowledge_usage


_LOCK = threading.RLock()
_PUBLIC_CACHE: dict[str, tuple[tuple[tuple[str, int, int], ...], dict[str, Any]]] = {}
_SCHEMA = 4
_PROCESS_METRIC_PLACEHOLDER = "内部过程指标已隐藏"
_PROCESS_PHRASES = re.compile(
    r"\b(?:line|branch|ae|ms|sfc|nrr|tsq|sfq|csg|pv)[_-](?:coverage[_-])?gap\b|"
    r"\b(?:combined|overall)[ _-](?:quality[ _-])?score\b|"
    r"\boverall[_ -]?quality[_ -]?index\b",
    re.IGNORECASE,
)
_KNOWN_DATASETS = frozenset({"TestEval", "HumanEval", "HumanEvalJava", "Defects4J_cli",
                             "Defects4J_csv", "Defects4J_lang3", "Defects4J_gson"})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _timestamp(value: object) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    return 0.0


def _visible_text(value: object) -> str:
    original = str(value or "")
    if "�" in original or _PROCESS_PHRASES.search(original):
        return ""
    result = public_text(original)
    result = result.replace(_PROCESS_METRIC_PLACEHOLDER, "").replace("内部过程信息已隐藏", "")
    result = re.sub(r"\s+", " ", result).strip(" ,;:，；：·-/")
    if re.search(r"\b(?:line_|branch_|weak_|ae_gap)\b", result, re.IGNORECASE):
        return ""
    if re.fullmatch(r"[a-z][a-z0-9_-]*", result, re.IGNORECASE) and "_" in result:
        return ""
    return result


def _logical_source(value: object) -> tuple[str, str]:
    parts = [part for part in re.split(r"[\\/]+", str(value or "")) if part and part != "."]
    lowered = [part.casefold() for part in parts]
    if "datasets" in lowered:
        parts = parts[lowered.index("datasets") + 1:]
    if not parts:
        return "", ""
    if parts[0].casefold() == "historical:":
        dataset = next((part for part in reversed(parts[1:]) if part in _KNOWN_DATASETS), "")
        return (f"{dataset}/历史知识" if dataset else "历史知识记录"), dataset
    if parts[0].casefold() == "defects4j" and len(parts) >= 2:
        parts = [f"Defects4J_{parts[1]}", *parts[2:]]
    elif re.match(r"^[A-Za-z]:$", parts[0]):
        parts = [parts[-1]]
    safe_parts = [part for part in parts if part not in {"..", "."}]
    logical = "/".join(safe_parts)
    dataset = safe_parts[0] if len(safe_parts) >= 2 else ""
    return logical, dataset


def _sources(root_text: str) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    root = Path(root_text)
    snapshots: list[Path] = []
    journals: list[Path] = []
    batches: list[Path] = []
    for dataset in sorted(path for path in root.iterdir() if path.is_dir()):
        batches.extend(path for path in [dataset / "batch_summary.json"] if path.is_file())
        snapshots.extend(path for path in (dataset / ".knowledge_snapshots").glob("*.json")
                         if ".merge." not in path.name)
        for task in dataset.iterdir():
            if task.is_dir() and not task.name.startswith("."):
                journals.extend((task / ".knowledge_journals").glob("*/records-*.json"))
    return tuple(sorted(snapshots)), tuple(sorted(journals)), tuple(sorted(batches))


def _revision(paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    result = []
    for path in paths:
        try:
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            result.append((str(path), 0, 0))
    return tuple(result)


def _dataset_summary_records(batches: tuple[Path, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in batches:
        experience = _read_json(path).get("batch_experience")
        if not isinstance(experience, dict):
            continue
        dataset = path.parent.name
        stamp = _timestamp(experience.get("timestamp"))
        for index, item in enumerate(experience.get("learning_notes") or []):
            if not isinstance(item, dict):
                continue
            solution = _visible_text(item.get("solution"))
            cause = _visible_text(item.get("cause"))
            if not solution and not cause:
                continue
            reason = _visible_text(item.get("reason_type"))
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            result.append({
                "experience_id": f"dataset-summary:{dataset}:{index}",
                "record_type": "dataset_summary",
                "source_type": "dataset_summary",
                "dataset": dataset,
                "source": f"{dataset}/batch_summary.json",
                "timestamp": stamp + index / 1000,
                "model": experience.get("model"),
                "topic": reason or f"{dataset} 经验摘要 {index + 1}",
                "action": _visible_text(item.get("lesson_type")),
                "failure_category": reason,
                "usage": cause,
                "solution": solution,
                "support_count": int(evidence.get("count", 0) or 0),
                "accepted": False,
            })
    return result


def _original_path(summary: dict[str, Any]) -> Path | None:
    links = summary.get("test_knowledge_guidance_library")
    if not isinstance(links, dict):
        return None
    value = links.get("knowledge") or links.get("knowledge_path")
    return Path(str(value)) if value else None


def _prepare(root: Path, snapshots: tuple[Path, ...], journals: tuple[Path, ...], batches: tuple[Path, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = _read_json(root / "all_datasets_summary.json")
    first = _timestamp(summary.get("first_started_at") or summary.get("started_at"))
    finished = _timestamp(summary.get("finished_at"))
    memory = AgentExperienceMemory(root / "unused.json")
    snapshot_records: list[dict[str, Any]] = []
    journal_records: list[dict[str, Any]] = []
    for path in snapshots:
        snapshot_records.extend(item for item in _read_json(path).get("records", []) if isinstance(item, dict))
    for path in journals:
        journal_records.extend(item for item in _read_json(path).get("records", []) if isinstance(item, dict))
    snapshot_unique = [item for item in memory._dedupe_records(snapshot_records)
                       if item.get("record_type") != "batch_summary"]
    all_recovered = memory._dedupe_records([*snapshot_records, *journal_records])
    recovered = [item for item in all_recovered
                 if item.get("record_type") != "batch_summary"]
    prepared: list[dict[str, Any]] = []
    run_datasets: set[str] = set()
    run_count = 0
    baseline_count = 0
    for item in recovered:
        record = dict(item)
        source, dataset = _logical_source(record.get("source") or record.get("source_dir"))
        if source:
            record["source"] = source
        if dataset:
            record["dataset"] = dataset
        stamp = _timestamp(record.get("timestamp"))
        if first and first <= stamp and (not finished or stamp <= finished):
            record["source_type"] = "run_record"
            run_count += 1
            if dataset:
                run_datasets.add(dataset)
        else:
            record["source_type"] = "historical_baseline"
            baseline_count += 1
        prepared.append(record)
    summaries = _dataset_summary_records(batches)
    prepared.extend(summaries)
    datasets = [str(item.get("dataset", {}).get("name") or "") for item in summary.get("results", [])
                if isinstance(item, dict) and isinstance(item.get("dataset"), dict)]
    external = _original_path(summary)
    external_data = _read_json(external).get("records", []) if external and external.is_file() else []
    external_records = [item for item in external_data if isinstance(item, dict)] if isinstance(external_data, list) else []
    external_is_baseline = bool(external_records and
                                (not first or max(_timestamp(item.get("timestamp"))
                                                  for item in external_records) < first))
    recovered_keys = {memory._record_key(item) for item in all_recovered}
    availability = {
        "complete": False,
        "note": "随运行包保留的快照和任务日志已恢复；实验结束时的完整外置知识库未随包提供。数据集经验摘要与逐条知识记录分开标注。",
        "snapshot_files": len(snapshots),
        "journal_files": len(journals),
        "recovered_snapshot_records": len(snapshot_unique),
        "recovered_journal_records": len(journal_records),
        "run_record_count": run_count,
        "baseline_record_count": baseline_count,
        "dataset_summary_count": len(summaries),
        "missing_datasets": [name for name in datasets if name and name not in run_datasets],
        "external_baseline_available": external_is_baseline,
        "external_baseline_record_count": len(external_records) if external_is_baseline else 0,
        "external_baseline_covered_count": sum(memory._record_key(item) in recovered_keys
                                                for item in external_records) if external_is_baseline else 0,
    }
    return prepared, availability


def imported_knowledge_store(root: str | Path, cache_dir: str | Path) -> Path | None:
    """Materialize a read/write store from retained snapshots without touching the old global store."""
    root = Path(root).resolve()
    if not (root / "all_datasets_summary.json").is_file():
        return None
    snapshots, journals, batches = _sources(str(root))
    if not snapshots and not journals and not batches:
        return None
    digest = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    target = Path(cache_dir) / f"{digest}.json"
    metadata = target.with_suffix(".meta.json")
    signature = hashlib.sha256(repr((_SCHEMA, _revision((root / "all_datasets_summary.json", *snapshots, *journals, *batches)))).encode("utf-8")).hexdigest()
    if target.is_file() and _read_json(metadata).get("signature") == signature:
        return target
    with _LOCK:
        if target.is_file() and _read_json(metadata).get("signature") == signature:
            return target
        records, availability = _prepare(root, snapshots, journals, batches)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{threading.get_ident()}.tmp")
        content = json.dumps({"records": records}, ensure_ascii=False, separators=(",", ":"))
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(target)
        except OSError:
            # The Windows Python runtime used for some installations cannot
            # rename paths below a non-ASCII user profile, despite writing them.
            target.write_text(content, encoding="utf-8")
            try:
                temporary.unlink()
            except OSError:
                pass
        metadata.write_text(json.dumps({"signature": signature, "scope": "imported_run", "root": str(root), "availability": availability}, ensure_ascii=False), encoding="utf-8")
    return target


def _record_status(item: dict[str, Any]) -> str:
    if item.get("source_type") == "dataset_summary":
        return "经验摘要"
    accepted = bool(item.get("accepted") or item.get("candidate_status") == "accepted")
    learning_stage = str(item.get("learning_stage") or "")
    validation_count = int(item.get("validation_count") or 0)
    support_count = int(item.get("support_count") or 0)
    independent_tasks = int(item.get("independent_task_count") or 0)
    confirmed = accepted and (validation_count >= 2 or (support_count >= 2 and independent_tasks >= 2))
    status = "已掌握" if confirmed else "待复核"
    if learning_stage == "learning":
        status = "正在学习"
    elif learning_stage in {"adopted", "mastered"}:
        status = "已掌握"
    elif learning_stage == "pending_validation":
        status = "待复核"
    if item.get("rejection_reason") or item.get("candidate_status") == "rejected" or learning_stage == "avoid":
        status = "应避免"
    return status


def _project(item: dict[str, Any]) -> dict[str, Any]:
    notes = item.get("learning_notes") if isinstance(item.get("learning_notes"), list) else []
    solutions = [_visible_text(note.get("solution")) for note in notes if isinstance(note, dict)]
    solution = _visible_text(item.get("solution") or item.get("rejection_reason")) or "; ".join(
        value for value in dict.fromkeys(solutions) if value)[:1800]
    source_type = item.get("source_type") or "post_import_learning"
    action = _visible_text(item.get("action") or item.get("dominant_gap") or item.get("record_type"))
    if not action:
        action = "历史知识记录" if source_type == "historical_baseline" else "测试经验"
    result = {
        "id": item.get("experience_id") or item.get("strategy_fingerprint") or str(item.get("timestamp", "")),
        "status": _record_status(item),
        "source_type": source_type,
        "dataset": item.get("dataset") or "",
        "language": item.get("language") or (item.get("task_signature") or {}).get("language"),
        "agent": _knowledge_owner(item.get("agent")),
        "action": action,
        "failure_category": _visible_text(item.get("failure_category")),
        "accepted": bool(item.get("accepted") or item.get("candidate_status") == "accepted"),
        "source": item.get("source") or "",
        "timestamp": item.get("timestamp"),
        "solution": solution,
        "question": _visible_text(item.get("question")),
        "usage": _visible_text(item.get("usage")) or _knowledge_usage(item),
        "topic": _visible_text(item.get("topic")),
        "record_type": item.get("record_type"),
        "learning_stage": item.get("learning_stage"),
        "validation_count": int(item.get("validation_count", 0) or 0),
        "validation_failure_count": int(item.get("validation_failure_count", 0) or 0),
        "support_count": int(item.get("support_count", 0) or 0),
        "independent_task_count": int(item.get("independent_task_count", 0) or 0),
    }
    return public_result(result)


def imported_knowledge_snapshot(root: str | Path, store: str | Path) -> dict[str, Any]:
    """Return all recoverable records in the UI's knowledge schema."""
    root, store = Path(root).resolve(), Path(store).resolve()
    journal_dir = store.with_suffix(store.suffix + ".records")
    journal_paths = tuple(sorted(journal_dir.glob("records-*.json"))) if journal_dir.is_dir() else ()
    revision = _revision((store, store.with_suffix(".meta.json"), *journal_paths))
    cache_key = str(store)
    public_path = store.with_suffix(".public.json")
    public_metadata = store.with_suffix(".public.meta.json")
    signature = hashlib.sha256(repr((_SCHEMA, revision)).encode("utf-8")).hexdigest()
    with _LOCK:
        cached = _PUBLIC_CACHE.get(cache_key)
        if cached and cached[0] == revision:
            return cached[1]
        if public_path.is_file() and _read_json(public_metadata).get("signature") == signature:
            result = _read_json(public_path)
            if result.get("scope") == "imported_run":
                _PUBLIC_CACHE[cache_key] = (revision, result)
                return result
        items = AgentExperienceMemory(store).load()
        records = [_project(item) for item in reversed(items)]
        records = [item for item in records if item]
        statuses = ["已掌握", "正在学习", "待复核", "应避免", "经验摘要"]
        counts = {status: sum(item["status"] == status for item in records) for status in statuses}
        availability = dict(_read_json(store.with_suffix(".meta.json")).get("availability", {}))
        availability["post_import_learning_count"] = sum(
            item["source_type"] == "post_import_learning" for item in records)
        result = {
            "path": str(store),
            "scope": "imported_run",
            "total": len(records),
            "stored_record_count": len(items),
            "counts": counts,
            "records": records,
            "availability": availability,
        }
        public_path.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        public_metadata.write_text(json.dumps({"signature": signature}), encoding="utf-8")
        _PUBLIC_CACHE[cache_key] = (revision, result)
    return result


def imported_root_for_store(store: str | Path) -> Path | None:
    value = _read_json(Path(store).with_suffix(".meta.json")).get("root")
    root = Path(str(value)) if value else None
    return root if root and (root / "all_datasets_summary.json").is_file() else None
