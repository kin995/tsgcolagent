from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graph.graph_query import GraphGaps
from graph.state_graph import StateFlowGraph

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_STORE_DIR = PROJECT_ROOT / "knowledge_store"
DEFAULT_KNOWLEDGE_BASE_PATH = KNOWLEDGE_STORE_DIR / "test_knowledge_guidance.json"
LEGACY_KNOWLEDGE_BASE_PATHS = [
    PROJECT_ROOT / "agent_knowledge_base.json",
    PROJECT_ROOT / "agent_experience.json",
]
_KNOWLEDGE_WRITE_LOCK = threading.RLock()


def resolve_knowledge_store_path() -> Path:
    configured = os.getenv("TSG_KNOWLEDGE_STORE")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_KNOWLEDGE_BASE_PATH


def configure_knowledge_store_path(path: str | Path | None) -> Path:
    """Select an external long-term store without overwriting existing knowledge."""
    if path is None or not str(path).strip():
        return resolve_knowledge_store_path()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() and DEFAULT_KNOWLEDGE_BASE_PATH.exists() and target != DEFAULT_KNOWLEDGE_BASE_PATH.resolve():
        try:
            with DEFAULT_KNOWLEDGE_BASE_PATH.open("rb") as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
        except FileExistsError:
            pass
    os.environ["TSG_KNOWLEDGE_STORE"] = str(target)
    return target


AGENT_TO_GAP = {
    "CoverageAgent": "coverage",
    "MutationAgent": "mutation",
    "AssertionAgent": "assertion",
    "BoundaryAgent": "exception",
    "TypeAnalysisAgent": "runtime_type",
    "TestValidityAgent": "validity",
    "OracleRepairAgent": "oracle",
}


@dataclass
class ExperiencePrediction:
    agent: str
    predicted_reward: float
    sample_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "predicted_reward": self.predicted_reward,
            "sample_count": self.sample_count,
            "reason": self.reason,
        }


class AgentExperienceMemory:
    def __init__(
        self,
        path: str | Path,
        mirror_paths: list[str | Path] | None = None,
        *,
        read_base_path: str | Path | None = None,
        journal_dir: str | Path | None = None,
        publish_canonical: bool = True,
    ):
        self.path = Path(path)
        self.read_base_path = Path(read_base_path) if read_base_path else None
        self.journal_dir = Path(journal_dir) if journal_dir else self.path.with_suffix(self.path.suffix + ".records")
        self.publish_canonical = publish_canonical
        # An imported run's recovered base is immutable.  New conversation
        # learning belongs in its journal: publishing the ordinary 3000-record
        # canonical window here would discard older imported records.
        if self.publish_canonical:
            metadata_path = self.path.with_suffix(".meta.json")
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            except (OSError, ValueError):
                metadata = {}
            if isinstance(metadata, dict) and metadata.get("scope") == "imported_run":
                self.publish_canonical = False
        mirrors = [Path(item) for item in (mirror_paths if mirror_paths is not None else self._default_mirror_paths())]
        self.mirror_paths = [item for item in mirrors if item.resolve() != self.path.resolve()]
        self.last_write_errors: list[dict[str, str]] = []

    def load(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        base_paths = [self.read_base_path] if self.read_base_path else self._storage_paths()
        legacy_paths = [] if self.read_base_path else self._legacy_read_paths()
        read_paths = [*base_paths, *legacy_paths, *self._journal_paths()]
        for path in read_paths:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            loaded = data.get("records", data if isinstance(data, list) else [])
            if isinstance(loaded, list):
                records.extend(item for item in loaded if isinstance(item, dict) and not self._ephemeral_test_record(item))
        return self._dedupe_records(records)

    def merge_records(
        self, records: list[dict[str, Any]], limit: int = 3000, *,
        baseline: list[dict[str, Any]] | None = None, merge_id: str | None = None,
    ) -> int:
        """Merge isolated round records into the canonical store after a round barrier."""
        with _KNOWLEDGE_WRITE_LOCK:
            before = self.load()
            current = {self._record_key(item): item for item in before}
            base = {self._record_key(item): item for item in (baseline or [])}
            groups: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                groups.setdefault(self._record_key(record), []).append(record)
            changed = 0
            for key, candidates in groups.items():
                previous = current.get(key, {})
                if merge_id and merge_id in previous.get("_round_merge_ids", []):
                    continue
                selected = dict(max([previous, *candidates], key=lambda item: float(item.get("timestamp", 0) or 0)))
                if baseline is not None and selected.get("record_type") == "conversation_concept":
                    selected["validation_evidence_ids"] = list(dict.fromkeys(
                        evidence_id for item in [previous, *candidates]
                        for evidence_id in item.get("validation_evidence_ids", [])
                    ))[-200:]
                    for field in ("validation_count", "validation_failure_count"):
                        baseline_count = int(base.get(key, {}).get(field, 0) or 0)
                        increments = sum(max(0, int(item.get(field, 0) or 0) - baseline_count) for item in candidates)
                        selected[field] = int(previous.get(field, baseline_count) or 0) + increments
                    if selected["validation_count"] >= 2:
                        selected.update(learning_stage="mastered", candidate_status="mastered", accepted=True)
                    elif selected["validation_failure_count"] >= 2 and not selected["validation_count"]:
                        selected.update(learning_stage="pending_validation", candidate_status="pending", accepted=False)
                if merge_id:
                    selected["_round_merge_ids"] = [*previous.get("_round_merge_ids", []), merge_id][-100:]
                current[key] = selected
                changed += 1
            self._write_records(self._dedupe_records(list(current.values()))[-limit:])
            if self.last_write_errors:
                raise OSError(f"Knowledge merge could not be fully persisted: {self.last_write_errors}")
            return changed

    def _ephemeral_test_record(self, record: dict[str, Any]) -> bool:
        if self.path.resolve() != DEFAULT_KNOWLEDGE_BASE_PATH.resolve():
            return False
        source = str(record.get("source") or record.get("source_dir") or "").replace("\\", "/").lower()
        return "pytest-of-" in source or "/appdata/local/temp/pytest-" in source

    def append_candidate(
        self,
        graph: StateFlowGraph,
        agent: str,
        action: str,
        candidate_text: str,
        accepted: bool,
        pytest_passed: bool,
        failure_category: str | None = None,
        before: Any | None = None,
        after: Any | None = None,
        runtime_seconds: float = 0.0,
        llm_calls: int = 0,
        rejection_reason: str | None = None,
        model: str | None = None,
        prompt_version: str = "unknown",
        schema_version: str = "unknown",
    ) -> dict[str, Any]:
        task_signature = self.task_signature(graph)
        learning_notes = self._candidate_learning_notes(
            graph=graph,
            agent=agent,
            action=action,
            candidate_text=candidate_text,
            accepted=accepted,
            pytest_passed=pytest_passed,
            failure_category=failure_category,
            rejection_reason=rejection_reason,
            before=before,
            after=after,
        )
        record = {
            "experience_id": self._experience_id(graph, agent, action, candidate_text),
            "timestamp": time.time(),
            "source": graph.source_path,
            "model": model or graph.metadata.get("llm_generation_model") or graph.metadata.get("llm_execution_model"),
            "project_version": "pteg-agent-v2",
            "prompt_version": prompt_version,
            "schema_version": schema_version,
            "task_signature": task_signature,
            "target_evidence_ids": list(graph.metadata.get("last_target_evidence_ids", [])),
            "input_partition": self._infer_input_partition(candidate_text),
            "boundary_operator": self._infer_boundary_operator(candidate_text),
            "oracle_kind": self._infer_oracle_kind(candidate_text),
            "candidate_status": "accepted" if accepted else "rejected",
            "failure_category": failure_category,
            "agent": agent,
            "action": action,
            "candidate_hash": hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()[:16],
            "strategy_fingerprint": hashlib.sha256(f"{agent}|{action}|{failure_category}|{self._infer_oracle_kind(candidate_text)}".encode("utf-8")).hexdigest()[:16],
            "accepted": accepted,
            "pytest_passed": pytest_passed,
            "before_metrics": self._metrics(before),
            "after_metrics": self._metrics(after),
            "quality_delta": {
                "sfc": self._delta(before, after, "sfc"),
                "ae": self._delta(before, after, "ae"),
                "mutation": self._delta(before, after, "mutation_score"),
            },
            "delta_sfc": self._delta(before, after, "sfc"),
            "delta_ae": self._delta(before, after, "ae"),
            "delta_mutation": self._delta(before, after, "mutation_score"),
            "runtime_seconds": round(runtime_seconds, 4),
            "llm_calls": llm_calls,
            "reward": self.candidate_reward(accepted, pytest_passed, failure_category, before, after, runtime_seconds, llm_calls),
            "rejection_reason": rejection_reason,
            "learning_notes": learning_notes,
            "prompt_reminders": [item["reminder"] for item in learning_notes if item.get("reminder")],
        }
        self._append_record(record, limit=3000)
        return record

    def hints(self, graph: StateFlowGraph, model: str | None = None, top_k: int = 5) -> dict[str, object]:
        records = self._similar_records(graph, model=model)
        successes = [item for item in records if item.get("accepted") and item.get("pytest_passed") and float(item.get("reward", 0.0)) > 0]
        failures = [item for item in records if not item.get("accepted") or not item.get("pytest_passed")]
        weak_or_no_gain = [
            item for item in records
            if item not in failures and (
                str(item.get("outcome", "")) in {"negative", "no_gain"}
                or str((item.get("after") or {}).get("quality_status", "")) == "valid_usable"
                or bool((item.get("after") or {}).get("weak_quality_flags"))
            )
        ]
        successes = sorted(successes, key=lambda item: self._weighted_reward(item), reverse=True)[:top_k]
        failures = sorted(failures, key=lambda item: self._weighted_reward(item))[:top_k]
        lessons = self._actionable_lessons(records, successes, failures, weak_or_no_gain, top_k=top_k)
        return {
            "matched_experience_count": len(records),
            "experience_sources": [str(path) for path in self._storage_paths() if path.exists()],
            "successful_strategies": [
                {
                    "agent": item.get("agent"),
                    "action": item.get("action"),
                    "failure_category": item.get("failure_category"),
                    "reward": item.get("reward"),
                    "delta_sfc": item.get("delta_sfc"),
                    "delta_ae": item.get("delta_ae"),
                    "delta_mutation": item.get("delta_mutation"),
                }
                for item in successes
            ],
            "failure_patterns_to_avoid": [
                {
                    "agent": item.get("agent"),
                    "action": item.get("action"),
                    "failure_category": item.get("failure_category"),
                    "rejection_reason": item.get("rejection_reason"),
                    "reward": item.get("reward"),
                    "prompt_reminders": item.get("prompt_reminders", []),
                }
                for item in failures
            ],
            "optimization_patterns": [
                {
                    "agent": item.get("agent"),
                    "action": item.get("action"),
                    "dominant_gap": item.get("dominant_gap"),
                    "outcome": item.get("outcome"),
                    "quality_status": (item.get("after") or {}).get("quality_status"),
                    "weak_quality_flags": (item.get("after") or {}).get("weak_quality_flags", []),
                    "prompt_reminders": item.get("prompt_reminders", []),
                }
                for item in sorted(weak_or_no_gain, key=lambda value: float(value.get("timestamp", 0.0) or 0.0), reverse=True)[:top_k]
            ],
            "actionable_lessons": lessons,
            "failure_reminders": [item["reminder"] for item in lessons if item.get("lesson_type") == "failure"],
            "optimization_reminders": [item["reminder"] for item in lessons if item.get("lesson_type") == "optimization"],
            "conversation_concepts": self._conversation_concepts(graph, model=model, top_k=min(3, top_k)),
            "batch_guidance": self._latest_batch_guidance(model=model),
        }

    def append_conversation_concept(
        self,
        question: str,
        answer: str,
        *,
        model: str | None = None,
        conversation_id: str | None = None,
        force: bool = False,
    ) -> dict[str, Any] | None:
        """Persist LLM-assisted unit-test concepts for later task retrieval."""
        if (not force and not self.is_unit_test_concept_question(question)) or not answer.strip():
            return None
        normalized = " ".join(question.casefold().split())
        topic = self._conversation_topic(question)
        experience_id = "conversation-concept:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        previous = next((item for item in self.load() if item.get("experience_id") == experience_id), {})
        record = {
            **previous,
            "record_type": "conversation_concept",
            "experience_id": experience_id,
            "timestamp": time.time(),
            "source": f"conversation:{conversation_id or 'knowledge-agent'}",
            "model": model,
            "project_version": "pteg-agent-v3",
            "agent": "TestKnowledgeAgent",
            "action": f"概念学习：{topic}",
            "topic": topic,
            "question": question.strip()[:1000],
            "solution": answer.strip()[:6000],
            "language": self._conversation_language(question),
            "candidate_status": "learning",
            "learning_stage": "learning",
            "accepted": False,
            "reward": float(previous.get("reward", 0.0) or 0.0),
            "validation_count": int(previous.get("validation_count", 0) or 0),
            "validation_failure_count": int(previous.get("validation_failure_count", 0) or 0),
            "conversation_id": conversation_id,
        }
        if record["validation_count"] >= 2:
            record.update({"candidate_status": "mastered", "learning_stage": "mastered", "accepted": True, "reward": max(0.2, record["reward"])})
        self._append_record(record, limit=3000)
        return record

    def validate_conversation_concepts(
        self,
        record_ids: list[str],
        *,
        successful: bool,
        evidence: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Advance retrieved concepts only when a real execution supplies evidence."""
        wanted = set(record_ids)
        if not wanted:
            return []
        updated: list[dict[str, Any]] = []
        for record in self.load():
            if record.get("record_type") != "conversation_concept" or record.get("experience_id") not in wanted:
                continue
            item = dict(record)
            evidence_id = None
            if evidence and evidence.get("task_source"):
                evidence_id = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
                if evidence_id in item.get("validation_evidence_ids", []):
                    continue
                item["validation_evidence_ids"] = [*item.get("validation_evidence_ids", []), evidence_id][-200:]
            if successful:
                item["validation_count"] = int(item.get("validation_count", 0) or 0) + 1
                if item["validation_count"] >= 2:
                    item.update({"candidate_status": "mastered", "learning_stage": "mastered", "accepted": True, "reward": max(0.2, float(item.get("reward", 0.0) or 0.0))})
                else:
                    item.update({"candidate_status": "learning", "learning_stage": "learning", "accepted": False})
            else:
                item["validation_failure_count"] = int(item.get("validation_failure_count", 0) or 0) + 1
                if not int(item.get("validation_count", 0) or 0) and item["validation_failure_count"] >= 2:
                    item.update({"candidate_status": "pending", "learning_stage": "pending_validation", "accepted": False})
            item["last_validation"] = {"timestamp": time.time(), "successful": successful, "evidence": evidence or {}}
            item["timestamp"] = time.time()
            self._append_record(item, limit=3000)
            updated.append(item)
        return updated

    @staticmethod
    def is_unit_test_concept_question(question: str) -> bool:
        text = question.casefold().strip()
        domain_terms = [
            "单元测试", "测试", "pytest", "junit", "jacoco", "pit", "覆盖", "断言", "预言",
            "oracle", "变异", "mock", "fixture", "参数化", "边界", "桩", "依赖隔离",
        ]
        concept_terms = ["什么", "如何", "怎样", "怎么", "原理", "区别", "方法", "策略", "设计", "提高", "改进", "最佳", "为什么"]
        report_only_terms = ["当前任务", "当前实验", "本次实验", "这份报告", "失败原因", "跑了多少", "结果是多少"]
        return any(term in text for term in domain_terms) and any(term in text for term in concept_terms) and not any(term in text for term in report_only_terms)

    def append_batch_summary(self, batch_summary: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        results = batch_summary.get("results", []) if isinstance(batch_summary.get("results"), list) else []
        failure_categories: Counter[str] = Counter()
        rejected_low_quality: list[dict[str, str]] = []
        low_mutation_reliability: list[str] = []
        failed_tasks: list[str] = []
        timeout_tasks: list[str] = []
        mutation_timeout_tasks: list[str] = []
        weak_quality_flags: Counter[str] = Counter()
        threshold_gap_counts: Counter[str] = Counter()
        weak_success_tasks: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            source = Path(str(item.get("source", ""))).name
            status = str(item.get("status", ""))
            report = item.get("final_report") if isinstance(item.get("final_report"), dict) else {}
            if status in {"failed", "generation_failed", "execution_invalid"}:
                failed_tasks.append(source)
                category = self._failure_category_from_item(item)
                failure_categories[category] += 1
                if category == "timeout":
                    timeout_tasks.append(source)
                if category in {"exception_only", "no_normal_behavior", "weak_assertions_only"}:
                    rejected_low_quality.append({"source": source, "reason": category})
            if isinstance(report, dict):
                if bool(report.get("timed_out")) or str(report.get("failure_category", "")) == "timeout":
                    timeout_tasks.append(source)
                if int(report.get("mutation_timeout_count", report.get("mutation_timeout", 0)) or 0) > 0 or int(report.get("mutation_timeout_retries", 0) or 0) > 0:
                    mutation_timeout_tasks.append(source)
                effective = int(report.get("effective_mutants", 0) or 0)
                reliability = str(report.get("mutation_reliability") or "")
                if reliability in {"low", "unavailable"} or effective < 3:
                    low_mutation_reliability.append(source)
                for flag in report.get("weak_quality_flags", []) or []:
                    weak_quality_flags[str(flag)] += 1
                for gap_name, gap_value in (report.get("threshold_failures", {}) or {}).items():
                    try:
                        if float(gap_value) > 0:
                            threshold_gap_counts[str(gap_name)] += 1
                    except (TypeError, ValueError):
                        continue
                if status == "valid_usable" or report.get("quality_status") == "valid_usable":
                    weak_success_tasks.append(
                        {
                            "source": source,
                            "weak_quality_flags": list(report.get("weak_quality_flags", []) or [])[:8],
                            "threshold_failures": dict(report.get("threshold_failures", {}) or {}),
                        }
                    )
        total = len(results)
        succeeded = int(batch_summary.get("succeeded", 0) or 0)
        learning_notes = self._batch_learning_notes(
            failure_categories=failure_categories,
            rejected_low_quality=rejected_low_quality,
            low_mutation_reliability=low_mutation_reliability,
            timeout_tasks=timeout_tasks,
            mutation_timeout_tasks=mutation_timeout_tasks,
            weak_quality_flags=weak_quality_flags,
            threshold_gap_counts=threshold_gap_counts,
            weak_success_tasks=weak_success_tasks,
        )
        record = {
            "record_type": "batch_summary",
            "timestamp": time.time(),
            "model": model or (batch_summary.get("llm", {}) or {}).get("model"),
            "source_dir": batch_summary.get("source_dir"),
            "output_dir": batch_summary.get("output_dir"),
            "total": total,
            "succeeded": succeeded,
            "failed": int(batch_summary.get("failed", 0) or 0),
            "success_rate": round(succeeded / total, 4) if total else 0.0,
            "failure_categories": dict(failure_categories),
            "failed_tasks": failed_tasks,
            "timeout_tasks": sorted(set(timeout_tasks))[:100],
            "mutation_timeout_tasks": sorted(set(mutation_timeout_tasks))[:100],
            "rejected_low_quality": rejected_low_quality[:50],
            "low_mutation_reliability_tasks": low_mutation_reliability[:100],
            "weak_success_tasks": weak_success_tasks[:100],
            "weak_quality_flag_counts": dict(weak_quality_flags),
            "threshold_gap_counts": dict(threshold_gap_counts),
            "guidance": self._build_batch_guidance(failure_categories, rejected_low_quality, low_mutation_reliability, timeout_tasks, mutation_timeout_tasks),
            "learning_notes": learning_notes,
            "prompt_reminders": [item["reminder"] for item in learning_notes if item.get("reminder")],
        }
        self._append_record(record, limit=3000)
        return record

    def import_historical_records(
        self,
        records: list[dict[str, Any]],
        *,
        source_label: str,
    ) -> dict[str, int]:
        """Condense useful evidence from an old run without copying its noisy raw records."""
        groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        skipped = 0
        for record in records:
            if not isinstance(record, dict) or record.get("record_type") == "batch_summary":
                skipped += 1
                continue
            signature = record.get("task_signature") if isinstance(record.get("task_signature"), dict) else {}
            language = str(signature.get("language") or record.get("language") or "python")
            complexity = str(signature.get("complexity") or "unknown")
            agent = str(record.get("agent") or "TestKnowledgeAgent")
            failure_category = str(record.get("failure_category") or "")
            reward = float(record.get("reward", 0.0) or 0.0)
            outcome = str(record.get("outcome") or "")
            dominant_gap = str(record.get("dominant_gap") or "")
            if record.get("accepted") is False or record.get("pytest_passed") is False:
                kind, topic = "failure", failure_category or str(record.get("rejection_reason") or "generation_failure")
            elif reward > 0 and outcome in {"positive", "improved", "accepted"}:
                kind, topic = "success", dominant_gap or str(record.get("action") or "quality_gain")
            elif dominant_gap and (outcome in {"negative", "no_gain"} or reward <= 0):
                kind, topic = "optimization", dominant_gap
            else:
                skipped += 1
                continue
            groups.setdefault((kind, language, complexity, agent, topic), []).append(record)

        imported: list[dict[str, Any]] = []
        for (kind, language, complexity, agent, topic), samples in groups.items():
            support_count = sum(max(1, int(item.get("support_count", 1) or 1)) for item in samples)
            source_count = len({str(item.get("source") or "") for item in samples if item.get("source")})
            rewards = [float(item.get("reward", 0.0) or 0.0) for item in samples]
            average_reward = sum(rewards) / len(rewards) if rewards else 0.0
            representative = max(samples, key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
            solution = self._historical_solution(representative, kind, topic)
            if kind == "failure":
                stage, accepted, passed, status = "avoid", False, False, "rejected"
                reward = min(-0.1, average_reward)
                reason = f"历史任务反复出现 {topic}，生成前应先应用规避措施。"
            elif kind == "success":
                stage = "adopted" if support_count >= 2 and source_count >= 2 else "pending_validation"
                accepted, passed, status = True, True, "accepted"
                reward = max(0.01, average_reward)
                reason = f"该策略曾在 {support_count} 次历史应用中带来可测质量提升。"
            else:
                stage, accepted, passed, status = "learning", True, True, "learning"
                reward = average_reward
                reason = f"历史任务在 {topic} 维度仍有缺口，应保留已通过测试并定向增强。"
            fingerprint = hashlib.sha256(
                f"{kind}|{language}|{complexity}|{agent}|{topic}".encode("utf-8", errors="replace")
            ).hexdigest()[:20]
            imported.append(
                {
                    "record_type": "historical_pattern",
                    "experience_id": f"historical-pattern:{fingerprint}",
                    "timestamp": max(float(item.get("timestamp", 0.0) or 0.0) for item in samples),
                    "source": f"historical://{Path(source_label).name}",
                    "source_archive": source_label,
                    "project_version": "pteg-agent-v3",
                    "language": language,
                    "task_signature": {**(representative.get("task_signature") or {}), "language": language, "complexity": complexity},
                    "agent": agent,
                    "action": f"历史经验：{topic}",
                    "dominant_gap": topic if kind != "failure" else None,
                    "failure_category": topic if kind == "failure" else None,
                    "candidate_status": status,
                    "learning_stage": stage,
                    "accepted": accepted,
                    "pytest_passed": passed,
                    "outcome": "negative" if kind == "failure" else "positive" if kind == "success" else "no_gain",
                    "reward": round(reward, 4),
                    "support_count": support_count,
                    "independent_task_count": source_count,
                    "validation_count": source_count if stage == "adopted" else 0,
                    "solution": solution,
                    "rejection_reason": reason if kind == "failure" else None,
                    "learning_notes": [{"reason": reason, "solution": solution, "action": solution}],
                    "prompt_reminders": [solution],
                }
            )
        existing = self.load()
        self._write_records(self._dedupe_records([*existing, *imported])[-3000:])
        return {"source_records": len(records), "patterns_imported": len(imported), "records_skipped": skipped}

    def _historical_solution(self, record: dict[str, Any], kind: str, topic: str) -> str:
        for note in record.get("learning_notes", []) or []:
            if isinstance(note, dict):
                value = str(note.get("solution") or note.get("reminder") or "").strip()
                if value:
                    return value
        for reminder in record.get("prompt_reminders", []) or []:
            value = str(reminder).strip()
            if value:
                return value
        if kind == "failure":
            return self._solution_for_category(topic)
        return self._solution_for_dominant_gap({topic: 1.0})

    def append_task_failure(
        self,
        *,
        source: str,
        experiment_round: int,
        category: str,
        reason: str,
        solution: str,
        stage: str | None = None,
        error: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist a round failure before the next experiment round retrieves knowledge."""
        fingerprint = hashlib.sha256(
            f"{Path(source).resolve()}|{experiment_round}|{category}|{reason}".encode("utf-8", errors="replace")
        ).hexdigest()[:24]
        record = {
            "record_type": "task_failure",
            "experience_id": f"failure-{fingerprint}",
            "timestamp": time.time(),
            "model": model,
            "source": source,
            "language": "java" if Path(source).suffix.lower() == ".java" else "python",
            "agent": "TestKnowledgeAgent",
            "action": f"规避{category}失败",
            "experiment_round": max(1, int(experiment_round or 1)),
            "failure_category": category,
            "failure_stage": stage,
            "rejection_reason": reason,
            "solution": solution,
            "error": str(error or "")[:4000] or None,
            "candidate_status": "rejected",
            "learning_stage": "avoid",
            "accepted": False,
            "reward": -1.0,
            "prompt_reminders": [solution],
            "learning_notes": [{"reason": reason, "solution": solution, "action": "下一轮生成前先检索并应用该规避措施"}],
        }
        if self._ephemeral_test_record(record):
            return record
        self._append_record(record, limit=3000)
        return record

    def append(
        self,
        graph: StateFlowGraph,
        agent: str,
        gaps: dict[str, float],
        before: Any,
        after: Any,
        model: str | None = None,
        runtime_seconds: float = 0.0,
        llm_calls: int = 0,
        token_count: int = 0,
        expected_reward: float = 0.0,
    ) -> dict[str, Any]:
        actual_reward = self.reward(before, after, runtime_seconds=runtime_seconds, llm_calls=llm_calls, token_count=token_count)
        learning_notes = self._iteration_learning_notes(
            graph=graph,
            agent=agent,
            gaps=gaps,
            before=before,
            after=after,
            actual_reward=actual_reward,
        )
        record = {
            "timestamp": time.time(),
            "source": graph.source_path,
            "model": model or graph.metadata.get("llm_generation_model") or graph.metadata.get("llm_execution_model"),
            "project_version": "pteg-agent-v3",
            "prompt_version": graph.metadata.get("prompt_version", "unknown"),
            "schema_version": graph.metadata.get("schema_version", "unknown"),
            "agent": agent,
            "dominant_gap": max(gaps.items(), key=lambda item: item[1])[0] if gaps else AGENT_TO_GAP.get(agent, "unknown"),
            "gaps": gaps,
            "task_features": self.task_features(graph),
            "task_signature": self.task_signature(graph),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "delta_sfq": round(after.sfq - before.sfq, 4),
            "runtime_seconds": round(runtime_seconds, 4),
            "llm_calls": llm_calls,
            "token_count": token_count,
            "expected_reward": round(expected_reward, 4),
            "actual_reward": actual_reward,
            "reward_prediction_error": round(actual_reward - expected_reward, 4),
            "outcome": "positive" if actual_reward > 0 else "no_gain" if actual_reward == 0 else "negative",
            "reward": actual_reward,
            "learning_notes": learning_notes,
            "prompt_reminders": [item["reminder"] for item in learning_notes if item.get("reminder")],
        }
        self._append_record(record, limit=1000)
        return record

    def predict(self, agent: str, gaps: GraphGaps, graph: StateFlowGraph, model: str | None = None) -> ExperiencePrediction:
        records = self.load()
        if model:
            records = [item for item in records if item.get("model") == model]
        gap_name = AGENT_TO_GAP.get(agent, "unknown")
        exact = [
            item for item in records
            if item.get("agent") == agent and item.get("dominant_gap") == gap_name
        ]
        same_agent = [item for item in records if item.get("agent") == agent]
        complexity = self._complexity_bucket(self.task_features(graph))
        similar = [
            item for item in exact
            if self._complexity_bucket(item.get("task_features", {})) == complexity
        ]
        pool = similar or exact or same_agent
        if not pool:
            return ExperiencePrediction(agent, 0.0, 0, "no historical samples")
        reward = sum(float(item.get("reward", 0.0)) for item in pool) / len(pool)
        return ExperiencePrediction(agent, round(reward, 4), len(pool), f"historical average for {gap_name}/{complexity}")

    def task_features(self, graph: StateFlowGraph) -> dict[str, int]:
        nodes = list(graph.nodes.values())
        return {
            "node_count": len(nodes),
            "function_count": sum(1 for node in nodes if node.node_type in {"Function", "Async", "Method", "Constructor"}),
            "branch_count": sum(1 for node in nodes if node.node_type in {"Branch", "Match"}),
            "loop_count": sum(1 for node in nodes if node.node_type == "Loop"),
            "exception_count": sum(1 for node in nodes if node.node_type in {"Exception", "Raise"}),
            "mutation_survivor_count": sum(1 for node in nodes if node.mutation_status == "survived"),
        }

    def task_signature(self, graph: StateFlowGraph) -> dict[str, object]:
        nodes = list(graph.nodes.values())
        operators: set[str] = set()
        argument_types: set[str] = set()
        for node in nodes:
            if node.node_type == "Branch":
                for op in ["<=", ">=", "<", ">", "==", "!="]:
                    if op in node.code or op in node.name:
                        operators.add(op)
            if node.node_type in {"Function", "Async", "Method", "Constructor"}:
                annotations = node.metadata.get("arg_annotations", {})
                if isinstance(annotations, dict):
                    argument_types.update(str(value) for value in annotations.values() if value)
        features = self.task_features(graph)
        return {
            **features,
            "language": str(graph.metadata.get("language", "unknown")),
            "argument_types": sorted(argument_types),
            "operators": sorted(operators),
            "state_types": sorted({node.node_type for node in nodes}),
            "target_terms": sorted({node.name for node in nodes if node.node_type in {"Function", "Async", "Method", "Constructor", "Branch", "Loop", "Exception", "Raise", "Throw"}})[:40],
            "complexity": self._complexity_bucket(features),
        }

    def retrieve_hybrid(
        self,
        graph: StateFlowGraph,
        query: dict[str, object] | None = None,
        model: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query = query or {}
        signature = self.task_signature(graph)
        records = [
            item for item in self.load()
            if item.get("task_signature") or (
                item.get("record_type") == "conversation_concept"
                and item.get("learning_stage") not in {"avoid"}
            )
        ]
        if model:
            records = [
                item for item in records
                if item.get("model") in {model, None} or item.get("record_type") == "conversation_concept"
            ]
        query_tokens = self._semantic_tokens({"query": query, "signature": signature})
        ranked: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            record_signature = record.get("task_signature", {})
            semantic = self._token_similarity(query_tokens, self._semantic_tokens(record))
            is_concept = record.get("record_type") == "conversation_concept"
            structural = 0.0 if is_concept else self._similarity(signature, record_signature if isinstance(record_signature, dict) else {})
            reward = max(-1.0, min(1.0, float(record.get("reward", 0.0) or 0.0)))
            quality = (reward + 1.0) / 2.0
            age_days = max(0.0, (time.time() - float(record.get("timestamp", time.time()) or time.time())) / 86400.0)
            recency = math.exp(-age_days / 90.0)
            score = (
                (0.55 * semantic) + (0.15 * quality) + (0.30 * recency)
                if is_concept
                else (0.35 * semantic) + (0.35 * structural) + (0.20 * quality) + (0.10 * recency)
            )
            item = dict(record)
            item["retrieval_score"] = round(score, 4)
            item["retrieval_components"] = {
                "semantic": round(semantic, 4),
                "structural": round(structural, 4),
                "quality": round(quality, 4),
                "recency": round(recency, 4),
            }
            ranked.append((score, item))
        return [item for score, item in sorted(ranked, key=lambda pair: pair[0], reverse=True) if score >= 0.20][:max(0, top_k)]

    def candidate_reward(self, accepted: bool, pytest_passed: bool, failure_category: str | None, before: Any | None, after: Any | None, runtime_seconds: float, llm_calls: int) -> float:
        invalid = {"syntax_error", "import_error", "collection_error", "fixture_error", "no_tests_collected"}
        if failure_category in invalid:
            return -1.0
        if not accepted or not pytest_passed:
            return -0.7
        delta = (
            0.35 * self._delta(before, after, "sfc")
            + 0.30 * self._delta(before, after, "ae")
            + 0.35 * self._delta(before, after, "mutation_score")
            - 0.02 * runtime_seconds
            - 0.05 * llm_calls
        )
        return round(delta, 4)

    def reward(self, before: Any, after: Any, runtime_seconds: float = 0.0, llm_calls: int = 0, token_count: int = 0) -> float:
        before_mutation = before.mutation_score if before.mutation_score is not None else 0.0
        after_mutation = after.mutation_score if after.mutation_score is not None else 0.0
        delta = (
            (after.sfc - before.sfc) * 0.40
            + (after.ae - before.ae) * 0.30
            + (after_mutation - before_mutation) * 0.30
        )
        if after.pytest_passed and not before.pytest_passed:
            delta += 0.10
        if before.pytest_passed and not after.pytest_passed:
            delta -= 0.25
        if delta > 0:
            delta -= min(0.05, runtime_seconds * 0.0001)
            delta -= min(0.05, llm_calls * 0.002)
            delta -= min(0.05, token_count * 0.000001)
        elif abs(delta) < 1e-12:
            # A passing but unchanged patch is explicitly a no-gain example,
            # not a successful strategy.
            delta = 0.0
        return round(delta, 4)

    def summary(self) -> dict[str, object]:
        records = self.load()
        learning_records = [item for item in records if item.get("record_type") != "batch_summary"]
        rewards = [float(item.get("reward", 0.0) or 0.0) for item in learning_records]
        return {
            "experience_sources": [str(path) for path in self._storage_paths() if path.exists()],
            "primary_experience_path": str(self.path),
            "persistent_knowledge_base": str(DEFAULT_KNOWLEDGE_BASE_PATH),
            "positive_experience_count": sum(value > 0 for value in rewards),
            "no_gain_experience_count": sum(value == 0 for value in rewards),
            "negative_experience_count": sum(value < 0 for value in rewards),
            "total_experience_count": len(learning_records),
            "batch_summary_count": sum(1 for item in records if item.get("record_type") == "batch_summary"),
            "actionable_lesson_count": sum(len(item.get("learning_notes", []) or []) for item in records),
        }

    def failure_rate(self, agent: str, model: str | None = None) -> float:
        records = [item for item in self.load() if item.get("agent") == agent]
        if model:
            records = [item for item in records if item.get("model") == model]
        if not records:
            return 0.0
        failures = sum(1 for item in records if float(item.get("reward", 0.0)) <= 0.0)
        return round(failures / len(records), 4)

    def _similar_records(self, graph: StateFlowGraph, model: str | None = None) -> list[dict[str, Any]]:
        signature = self.task_signature(graph)
        records = self.load()
        if model:
            records = [item for item in records if item.get("model") in {model, None}]
        scored = [(self._similarity(signature, item.get("task_signature", {})), item) for item in records if item.get("task_signature")]
        return [item for score, item in sorted(scored, key=lambda pair: pair[0], reverse=True) if score > 0.2]

    def _conversation_concepts(self, graph: StateFlowGraph, model: str | None, top_k: int) -> list[dict[str, Any]]:
        query = {"language": graph.metadata.get("language"), "task_signature": self.task_signature(graph)}
        matched = [item for item in self.retrieve_hybrid(graph, query=query, model=model, top_k=max(8, top_k * 3)) if item.get("record_type") == "conversation_concept"]
        return [
            {
                "experience_id": item.get("experience_id"),
                "topic": item.get("topic"),
                "guidance": item.get("solution"),
                "learning_stage": item.get("learning_stage"),
                "validation_count": item.get("validation_count", 0),
                "retrieval_score": item.get("retrieval_score"),
            }
            for item in matched[:top_k]
        ]

    def _conversation_topic(self, question: str) -> str:
        text = question.casefold()
        topics = [
            (("变异", "pit"), "变异测试与存活变异体"),
            (("断言", "oracle", "预言"), "断言与测试预言"),
            (("覆盖", "jacoco"), "覆盖率与路径设计"),
            (("mock", "桩", "依赖隔离"), "依赖隔离与替身"),
            (("边界", "参数化"), "边界与参数化测试"),
            (("junit", "java"), "Java与JUnit测试"),
            (("pytest", "python", "fixture"), "Python与pytest测试"),
        ]
        return next((label for terms, label in topics if any(term in text for term in terms)), "单元测试设计")

    def _conversation_language(self, question: str) -> str | None:
        text = question.casefold()
        if "java" in text or "junit" in text or "jacoco" in text or "pit" in text:
            return "java"
        if "python" in text or "pytest" in text:
            return "python"
        return None

    def _latest_batch_guidance(self, model: str | None = None) -> dict[str, Any]:
        records = [
            item for item in self.load()
            if item.get("record_type") == "batch_summary"
            and (not model or item.get("model") in {model, None})
        ]
        if not records:
            return {}
        latest = max(records, key=lambda item: float(item.get("timestamp", 0.0) or 0.0))
        return {
            "source_dir": latest.get("source_dir"),
            "success_rate": latest.get("success_rate"),
            "failure_categories": latest.get("failure_categories", {}),
            "guidance": latest.get("guidance", []),
            "learning_notes": latest.get("learning_notes", [])[:20],
            "prompt_reminders": latest.get("prompt_reminders", [])[:20],
            "weak_quality_flag_counts": latest.get("weak_quality_flag_counts", {}),
            "threshold_gap_counts": latest.get("threshold_gap_counts", {}),
            "timeout_tasks": latest.get("timeout_tasks", [])[:20],
            "mutation_timeout_tasks": latest.get("mutation_timeout_tasks", [])[:20],
            "low_mutation_reliability_tasks": latest.get("low_mutation_reliability_tasks", [])[:20],
        }

    def _failure_category_from_item(self, item: dict[str, Any]) -> str:
        report = item.get("final_report")
        if isinstance(report, dict) and report.get("failure_category"):
            return str(report.get("failure_category"))
        error = str(item.get("error") or "")
        lowered_error = error.lower()
        if "timeout" in lowered_error or "timed out" in lowered_error:
            return "timeout"
        for category in [
            "exception_only",
            "no_normal_behavior",
            "weak_assertions_only",
            "syntax_error",
            "import_error",
            "oracle_error",
            "timeout",
        ]:
            if category in error:
                return category
        return "unknown"

    def _build_batch_guidance(
        self,
        failure_categories: Counter[str],
        rejected_low_quality: list[dict[str, str]],
        low_mutation_reliability: list[str],
        timeout_tasks: list[str] | None = None,
        mutation_timeout_tasks: list[str] | None = None,
    ) -> list[str]:
        guidance: list[str] = []
        timeout_tasks = timeout_tasks or []
        mutation_timeout_tasks = mutation_timeout_tasks or []
        if failure_categories.get("timeout") or timeout_tasks:
            guidance.append(
                "上一批出现测试执行超时。下一轮应优先生成小规模输入，避免大范围循环、递归深度过大、无限迭代和真实网络/文件系统等待；对可疑函数先用边界小样本验证，再逐步扩展。"
            )
        if mutation_timeout_tasks:
            guidance.append(
                "上一批出现变异测试超时。下一轮应减少会放大循环或递归的输入，优先使用短列表、小整数、短字符串，并为循环退出条件补充明确断言。"
            )
        if failure_categories.get("exception_only") or failure_categories.get("no_normal_behavior"):
            guidance.append(
                "上一批出现低质量异常-only或缺少正常行为的候选。原因通常是参数类型推断失败或只测试了非法输入；下一轮每个函数应先生成至少一个正常输入/输出断言，再补异常路径。"
            )
            guidance.append(
                "遇到含糊参数名时，应优先从 docstring 示例、源码操作和实际观察中推断类型，不要默认把所有参数当数字。"
            )
        if failure_categories.get("weak_assertions_only"):
            guidance.append("上一批出现弱断言候选。原因是断言只检查对象存在或恒真；下一轮必须使用明确返回值、长度、布尔值或近似数值断言。")
        if low_mutation_reliability:
            guidance.append("变异证据经常是低样本。原因通常是源码可变异位置少、变异超时、无效变异或测试没有区分分支；下一轮应补边界输入、分支区分断言和杀变异断言。")
        if not guidance and failure_categories:
            top = ", ".join(f"{name}:{count}" for name, count in failure_categories.most_common(3))
            guidance.append(f"上一批失败集中在 {top}；下一轮应优先生成可执行、可观察、带明确 oracle 的保守测试，并在报告中保留失败阶段和错误摘要。")
        if rejected_low_quality:
            guidance.append(f"有 {len(rejected_low_quality)} 个任务出现低质量候选被拒绝；这些拒绝会被当作生成失败经验，后续规划应主动避开同类模式。")
        return guidance

    def _candidate_learning_notes(
        self,
        graph: StateFlowGraph,
        agent: str,
        action: str,
        candidate_text: str,
        accepted: bool,
        pytest_passed: bool,
        failure_category: str | None,
        rejection_reason: str | None,
        before: Any | None,
        after: Any | None,
    ) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        if not accepted or not pytest_passed or failure_category:
            category = failure_category or rejection_reason or "candidate_rejected"
            notes.append(self._lesson(
                "failure",
                str(category),
                self._reason_for_category(str(category), after=after),
                self._solution_for_category(str(category), graph=graph, report=after),
                evidence={
                    "agent": agent,
                    "action": action,
                    "oracle_kind": self._infer_oracle_kind(candidate_text),
                    "input_partition": self._infer_input_partition(candidate_text),
                    "rejection_reason": rejection_reason,
                },
            ))
        else:
            deltas = {
                "sfc": self._delta(before, after, "sfc"),
                "ae": self._delta(before, after, "ae"),
                "mutation": self._delta(before, after, "mutation_score"),
            }
            weak_notes = self._weak_quality_lessons(graph, after)
            if weak_notes:
                notes.extend(weak_notes)
            elif any(value > 0 for value in deltas.values()):
                notes.append(self._lesson(
                    "success",
                    "quality_gain",
                    "The accepted candidate improved at least one quality signal.",
                    "Prefer the same input partition and oracle style for similar task signatures.",
                    evidence={"agent": agent, "action": action, "quality_delta": deltas},
                ))
            else:
                notes.append(self._lesson(
                    "optimization",
                    "no_measurable_gain",
                    "The candidate ran successfully but did not improve state-flow, assertion, or mutation evidence.",
                    "Next time add tests that target uncovered branches or survived mutants instead of adding more equivalent examples.",
                    evidence={"agent": agent, "action": action, "quality_delta": deltas},
                ))
        return notes

    def _iteration_learning_notes(
        self,
        graph: StateFlowGraph,
        agent: str,
        gaps: dict[str, float],
        before: Any,
        after: Any,
        actual_reward: float,
    ) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        failure_category = str(getattr(after, "failure_category", "") or "")
        if not bool(getattr(after, "pytest_passed", True)) or failure_category not in {"", "valid_baseline"}:
            category = failure_category or "execution_failed"
            notes.append(self._lesson(
                "failure",
                category,
                self._reason_for_category(category, after=after),
                self._solution_for_category(category, graph=graph, report=after),
                evidence={
                    "agent": agent,
                    "dominant_gap": max(gaps.items(), key=lambda item: item[1])[0] if gaps else None,
                    "failure_location": getattr(after, "failure_location", None),
                    "repair_failure_reason": getattr(after, "repair_failure_reason", None),
                },
            ))
            return notes

        notes.extend(self._weak_quality_lessons(graph, after))
        if actual_reward > 0:
            notes.append(self._lesson(
                "success",
                "optimization_gain",
                "The selected optimization improved the quality score.",
                "Reuse this focus for similar tasks when the same gap is dominant.",
                evidence={
                    "agent": agent,
                    "dominant_gap": max(gaps.items(), key=lambda item: item[1])[0] if gaps else None,
                    "delta_sfc": self._delta(before, after, "sfc"),
                    "delta_ae": self._delta(before, after, "ae"),
                    "delta_mutation": self._delta(before, after, "mutation_score"),
                },
            ))
        elif actual_reward == 0:
            notes.append(self._lesson(
                "optimization",
                "no_gain_after_optimization",
                "The test suite stayed runnable but the selected optimization did not close the main gap.",
                self._solution_for_dominant_gap(gaps),
                evidence={"agent": agent, "gaps": gaps},
            ))
        else:
            notes.append(self._lesson(
                "optimization",
                "negative_quality_delta",
                "The optimization reduced quality after accounting for metric gain and cost.",
                "Prefer a smaller repair scope and keep already passing, unrelated test cases unchanged.",
                evidence={"agent": agent, "gaps": gaps, "reward": actual_reward},
            ))
        return notes

    def _batch_learning_notes(
        self,
        failure_categories: Counter[str],
        rejected_low_quality: list[dict[str, str]],
        low_mutation_reliability: list[str],
        timeout_tasks: list[str],
        mutation_timeout_tasks: list[str],
        weak_quality_flags: Counter[str],
        threshold_gap_counts: Counter[str],
        weak_success_tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        notes: list[dict[str, Any]] = []
        for category, count in failure_categories.most_common(6):
            notes.append(self._lesson(
                "failure",
                category,
                self._reason_for_category(category),
                self._solution_for_category(category),
                evidence={"count": count},
            ))
        for flag, count in weak_quality_flags.most_common(8):
            notes.append(self._lesson(
                "optimization",
                flag,
                self._reason_for_weak_flag(flag),
                self._solution_for_weak_flag(flag),
                evidence={"count": count},
            ))
        for gap_name, count in threshold_gap_counts.most_common(8):
            notes.append(self._lesson(
                "optimization",
                gap_name,
                f"{gap_name} remained below the configured success threshold.",
                self._solution_for_threshold_gap(gap_name),
                evidence={"count": count},
            ))
        if weak_success_tasks:
            notes.append(self._lesson(
                "optimization",
                "valid_but_weak_success",
                "Some tasks were executable and therefore usable, but did not reach the strong quality threshold.",
                "Treat them as successful baselines, then optimize only the reported weak dimensions without deleting passing tests.",
                evidence={"count": len(weak_success_tasks), "examples": weak_success_tasks[:5]},
            ))
        if rejected_low_quality:
            notes.append(self._lesson(
                "failure",
                "low_quality_candidate_rejected",
                "Candidates were rejected because they did not observe meaningful behavior.",
                "Generate at least one normal input/output assertion before adding exception-only or smoke tests.",
                evidence={"count": len(rejected_low_quality), "examples": rejected_low_quality[:5]},
            ))
        if low_mutation_reliability:
            notes.append(self._lesson(
                "optimization",
                "low_mutation_reliability",
                "Mutation evidence was weak or unavailable on several tasks.",
                "Prefer branch-distinguishing assertions and small deterministic inputs; report mutation as unavailable only after the tool evidence says so.",
                evidence={"count": len(low_mutation_reliability), "examples": low_mutation_reliability[:5]},
            ))
        if timeout_tasks or mutation_timeout_tasks:
            notes.append(self._lesson(
                "failure",
                "timeout_or_mutation_timeout",
                "Some generated tests or mutation runs timed out.",
                "Use short strings, small lists, shallow recursion bounds, and avoid network/file waiting unless the source explicitly requires it.",
                evidence={"pytest_timeout_count": len(set(timeout_tasks)), "mutation_timeout_count": len(set(mutation_timeout_tasks))},
            ))
        return notes[:30]

    def _weak_quality_lessons(self, graph: StateFlowGraph, report: Any | None) -> list[dict[str, Any]]:
        if report is None:
            return []
        notes: list[dict[str, Any]] = []
        flags = list(getattr(report, "weak_quality_flags", None) or [])
        threshold_failures = dict(getattr(report, "threshold_failures", None) or {})
        for flag in flags:
            notes.append(self._lesson(
                "optimization",
                str(flag),
                self._reason_for_weak_flag(str(flag)),
                self._solution_for_weak_flag(str(flag)),
                evidence={"source": graph.source_path, "quality_status": getattr(report, "quality_status", None)},
            ))
        for gap_name, gap_value in threshold_failures.items():
            try:
                if float(gap_value) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            notes.append(self._lesson(
                "optimization",
                str(gap_name),
                f"{gap_name} remained below the configured target.",
                self._solution_for_threshold_gap(str(gap_name)),
                evidence={"gap": gap_value, "source": graph.source_path},
            ))
        return notes[:12]

    def _actionable_lessons(
        self,
        records: list[dict[str, Any]],
        successes: list[dict[str, Any]],
        failures: list[dict[str, Any]],
        weak_or_no_gain: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for record in [*failures, *weak_or_no_gain, *successes]:
            for note in record.get("learning_notes", []) or []:
                if isinstance(note, dict):
                    candidates.append(note)
        if not candidates:
            for record in records:
                for reminder in record.get("prompt_reminders", []) or []:
                    candidates.append(self._lesson("experience", "historical_reminder", "Historical reminder from previous run.", str(reminder)))
        scored: dict[str, dict[str, Any]] = {}
        counts: Counter[str] = Counter()
        for note in candidates:
            key = f"{note.get('lesson_type')}|{note.get('reason_type')}|{note.get('reminder')}"
            counts[key] += 1
            scored[key] = note
        result: list[dict[str, Any]] = []
        for key, count in counts.most_common(max(top_k * 3, 12)):
            item = dict(scored[key])
            item["support_count"] = count
            result.append(item)
        return result[: max(top_k * 2, 10)]

    def _lesson(
        self,
        lesson_type: str,
        reason_type: str,
        cause: str,
        solution: str,
        evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reminder = f"Reason: {reason_type}. Cause: {cause} Solution: {solution}"
        return {
            "lesson_type": lesson_type,
            "reason_type": reason_type,
            "cause": cause,
            "solution": solution,
            "reminder": reminder,
            "evidence": evidence or {},
        }

    def _reason_for_category(self, category: str, after: Any | None = None) -> str:
        mapping = {
            "syntax_error": "Generated test code was not syntactically valid.",
            "import_error": "The generated test imported the target or dependency incorrectly.",
            "collection_error": "The test runner could not collect the generated tests.",
            "fixture_error": "The generated test used pytest/JUnit fixtures or setup incorrectly.",
            "no_tests_collected": "The generated file did not contain test methods recognized by the test runner.",
            "oracle_error": "The expected assertion value did not match the real observed behavior.",
            "runtime_error": "The test called the target with invalid objects, invalid argument types, or an unsupported environment.",
            "timeout": "The test input or target call caused long execution, recursion, loop expansion, or external waiting.",
            "exception_only": "The candidate only exercised failing/exception paths and missed normal behavior.",
            "no_normal_behavior": "No useful normal input/output behavior was observed.",
            "weak_assertions_only": "Assertions only checked existence or tautologies instead of real outcomes.",
            "java_tool_missing": "Java quality tools could not run because Maven/JUnit/JaCoCo/PIT were unavailable.",
            "java_execution_unavailable": "The Java source was not staged into a Maven workspace that can run JUnit and JaCoCo.",
            "java_execution_failed": "The generated JUnit test did not compile or did not pass.",
            "java_timeout": "The Java test or Maven quality tool exceeded its timeout.",
        }
        if after is not None:
            detail = str(getattr(after, "repair_failure_reason", "") or getattr(after, "timeout_reason", "") or "")
            if detail:
                return f"{mapping.get(category, 'The previous attempt failed or was rejected.')} Detail: {detail[:240]}"
        return mapping.get(category, "The previous attempt failed or was rejected.")

    def _solution_for_category(self, category: str, graph: StateFlowGraph | None = None, report: Any | None = None) -> str:
        mapping = {
            "syntax_error": "Return only valid test code; keep imports minimal and let the renderer/loaders handle target loading.",
            "import_error": "Use the provided target fixture/loader for Python and package-correct JUnit class names for Java.",
            "collection_error": "Use standard test names: Python def test_* and Java public class SourceNameTest with @Test methods.",
            "fixture_error": "Do not invent fixtures; use the existing target fixture or plain JUnit construction.",
            "no_tests_collected": "Create at least one recognized test function or @Test method with a real assertion.",
            "oracle_error": "Observe behavior first, then assert exact return values, state changes, collection contents, or explicit exception types.",
            "runtime_error": "Infer constructor and argument types from structured context before calling; use small valid objects.",
            "timeout": "Minimize input size and bound loops/recursion-triggering cases; avoid file/network waits.",
            "exception_only": "Add a normal behavior case before exception cases and assert the actual return/state result.",
            "no_normal_behavior": "Prioritize a valid nominal input for each public method/function.",
            "weak_assertions_only": "Replace smoke assertions with specific return, boolean, length, collection, state, or exception-type assertions.",
            "java_tool_missing": "Ensure Maven is available at D:\\wn\\mavenEvo\\apache-maven-3.9.1\\bin\\mvn.cmd or via MAVEN_CMD before Java evaluation.",
            "java_execution_unavailable": "Stage source under java_workspace/src/main/java and generated tests under java_workspace/src/test/java before running Maven.",
            "java_execution_failed": "Repair the smallest failing JUnit method; keep class name SourceNameTest and package declaration aligned with the source.",
            "java_timeout": "Use small deterministic Java inputs and avoid expensive randomized or large-collection tests.",
        }
        return mapping.get(category, self._solution_for_dominant_gap({}))

    def _reason_for_weak_flag(self, flag: str) -> str:
        mapping = {
            "weak_normal_behavior": "The suite runs but does not sufficiently cover normal successful behavior.",
            "coverage_unavailable": "Coverage tools did not produce trustworthy line/branch data.",
            "weak_line_coverage": "Too many executable source lines remain uncovered.",
            "weak_branch_coverage": "Branch alternatives are not distinguished by tests.",
            "weak_state_flow_coverage": "Important state graph nodes or edges lack execution evidence.",
            "weak_assertion_effectiveness": "Assertions do not observe enough return/state/exception evidence.",
            "weak_assertions": "Some assertions are smoke checks or tautologies.",
            "mutation_unavailable": "Mutation testing did not produce a trustworthy score.",
            "weak_mutation_score": "Surviving mutants indicate weak defect-detection ability.",
            "weak_state_flow_quality": "The combined state-flow quality score remains below target.",
        }
        return mapping.get(flag, "The runnable test suite is usable but still below a quality target.")

    def _solution_for_weak_flag(self, flag: str) -> str:
        mapping = {
            "weak_normal_behavior": "Add at least one nominal input/output test for every public target before edge cases.",
            "coverage_unavailable": "Keep the test runnable and make sure the language toolchain can emit coverage reports.",
            "weak_line_coverage": "Target uncovered methods and returns with small concrete examples.",
            "weak_branch_coverage": "Add paired inputs that force true and false branch outcomes.",
            "weak_state_flow_coverage": "Select tests from uncovered state nodes and edges in the graph.",
            "weak_assertion_effectiveness": "Assert concrete return values, object state changes, exception types, and collection contents.",
            "weak_assertions": "Remove or replace assert-not-null, assertTrue(true), and result-is-itself checks.",
            "mutation_unavailable": "Keep mutation as an unavailable evidence flag, then rely on coverage and assertion quality without failing the usable suite.",
            "weak_mutation_score": "Add assertions that distinguish boundary operators, arithmetic changes, and boolean negations.",
            "weak_state_flow_quality": "Optimize the largest remaining gap while preserving already passing tests.",
        }
        return mapping.get(flag, "Optimize only the weak dimension reported by evaluation and keep passing tests unchanged.")

    def _solution_for_threshold_gap(self, gap_name: str) -> str:
        if "line" in gap_name:
            return self._solution_for_weak_flag("weak_line_coverage")
        if "branch" in gap_name:
            return self._solution_for_weak_flag("weak_branch_coverage")
        if "sfc" in gap_name or "state" in gap_name:
            return self._solution_for_weak_flag("weak_state_flow_coverage")
        if "ae" in gap_name or "assert" in gap_name:
            return self._solution_for_weak_flag("weak_assertion_effectiveness")
        if "mutation" in gap_name:
            return self._solution_for_weak_flag("weak_mutation_score")
        if "sfq" in gap_name:
            return self._solution_for_weak_flag("weak_state_flow_quality")
        return "Close this threshold gap with the smallest targeted test addition or assertion repair."

    def _solution_for_dominant_gap(self, gaps: dict[str, float]) -> str:
        if not gaps:
            return "Generate conservative executable tests with clear oracles, then optimize only the reported weak dimension."
        largest = max(gaps.items(), key=lambda item: item[1])[0]
        return self._solution_for_threshold_gap(largest)

    def _append_record(self, record: dict[str, Any], limit: int) -> None:
        with _KNOWLEDGE_WRITE_LOCK:
            records = self._load_process_shard()
            records.append(record)
            records = self._dedupe_records(records)[-limit:]
            self._write_records(records)
            # Measurement records selected/written identifiers, not model
            # reasoning or an assumption that retrieval improved the result.
            from metrics.measurement import CURRENT
            measurement = CURRENT.get()
            if measurement and not self.last_write_errors:
                measurement.append({"type": "knowledge_record", "record_id": self._record_key(record),
                                    "record_type": record.get("record_type", "experience"),
                                    "accepted": record.get("accepted"), "failure_category": record.get("failure_category")})

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        with _KNOWLEDGE_WRITE_LOCK:
            errors: list[dict[str, str]] = []
            wrote_any = False
            for path in self._write_paths():
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
                    content = json.dumps({"records": records}, indent=2, ensure_ascii=False)
                    temp_path.write_text(content, encoding="utf-8")
                    try:
                        temp_path.replace(path)
                    except OSError:
                        path.write_text(content, encoding="utf-8")
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    wrote_any = True
                except OSError as exc:
                    errors.append({"path": str(path), "error": f"{exc.__class__.__name__}: {exc}"})
            self.last_write_errors = errors
            if not wrote_any:
                return
            # Journals are the concurrency-safe source of truth.  Also publish a
            # merged canonical JSON for reports, migration tools and users that
            # open the configured knowledge path directly.
            merged = self.load()
            canonical_content = json.dumps({"records": merged[-3000:]}, indent=2, ensure_ascii=False)
            for path in self._storage_paths() if self.publish_canonical else []:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = path.with_suffix(path.suffix + f".{threading.get_ident()}.tmp")
                    temp_path.write_text(canonical_content, encoding="utf-8")
                    try:
                        temp_path.replace(path)
                    except OSError:
                        path.write_text(canonical_content, encoding="utf-8")
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                except OSError as exc:
                    errors.append({"path": str(path), "error": f"{exc.__class__.__name__}: {exc}"})
            self.last_write_errors = errors

    def _storage_paths(self) -> list[Path]:
        return [self.path, *self.mirror_paths]

    def _write_paths(self) -> list[Path]:
        return [self.journal_dir / f"records-{os.getpid()}.json"]

    def _journal_paths(self) -> list[Path]:
        try:
            return sorted(self.journal_dir.glob("records-*.json")) if self.journal_dir.is_dir() else []
        except OSError:
            return []

    def _load_process_shard(self) -> list[dict[str, Any]]:
        path = self._write_paths()[0]
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        loaded = data.get("records", data if isinstance(data, list) else [])
        return [item for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []

    def _default_mirror_paths(self) -> list[Path]:
        return []

    def _legacy_read_paths(self) -> list[Path]:
        if self.path.resolve() != DEFAULT_KNOWLEDGE_BASE_PATH.resolve():
            return []
        return [path for path in LEGACY_KNOWLEDGE_BASE_PATHS if path.resolve() != self.path.resolve()]

    def failure_guidance(self, source: str, category: str, top_k: int = 8) -> dict[str, object]:
        from core.experiment_outcome import default_solution

        source_name = Path(source).name
        matched: list[dict[str, Any]] = []
        solutions: list[str] = []
        for record in reversed(self.load()):
            record_source = Path(str(record.get("source") or record.get("source_dir") or "")).name
            same_source = bool(source_name and record_source == source_name)
            same_category = str(record.get("failure_category") or "") == category
            categories = record.get("failure_categories", {}) if isinstance(record.get("failure_categories"), dict) else {}
            if not same_source and not same_category and category not in categories:
                continue
            matched.append(record)
            for note in record.get("learning_notes", []) or []:
                if not isinstance(note, dict):
                    continue
                for key in ("solution", "reminder", "action"):
                    value = str(note.get(key) or "").strip()
                    if value and value not in solutions:
                        solutions.append(value)
            for reminder in record.get("prompt_reminders", []) or []:
                value = str(reminder).strip()
                if value and value not in solutions:
                    solutions.append(value)
            if len(matched) >= top_k:
                break
        baseline = default_solution(category)
        if baseline not in solutions:
            solutions.insert(0, baseline)
        return {
            "category": category,
            "matched_experience_ids": [str(item.get("experience_id") or item.get("timestamp") or "") for item in matched],
            "previous_solutions": solutions[:top_k],
            "matched_count": len(matched),
        }

    def _dedupe_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        for record in records:
            key = self._record_key(record)
            previous = deduped.get(key)
            if previous is None or float(record.get("timestamp", 0.0) or 0.0) >= float(previous.get("timestamp", 0.0) or 0.0):
                deduped[key] = record
        return sorted(deduped.values(), key=lambda item: float(item.get("timestamp", 0.0) or 0.0))

    def _record_key(self, record: dict[str, Any]) -> str:
        if record.get("experience_id"):
            return str(record["experience_id"])
        parts = [
            str(record.get("record_type", "experience")),
            str(record.get("timestamp", "")),
            str(record.get("source", record.get("source_dir", ""))),
            str(record.get("agent", "")),
            str(record.get("action", "")),
            str(record.get("candidate_hash", "")),
        ]
        return "|".join(parts)

    def _similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        score = 0.0
        if a.get("language") and a.get("language") == b.get("language"):
            score += 0.1
        for key in ["complexity"]:
            if a.get(key) and a.get(key) == b.get(key):
                score += 0.1
        for key in ["function_count", "branch_count", "loop_count", "exception_count"]:
            av = float(a.get(key, 0) or 0)
            bv = float(b.get(key, 0) or 0)
            score += 0.1 * (1.0 - min(1.0, abs(av - bv) / max(1.0, av, bv)))
        for key in ["argument_types", "operators", "state_types"]:
            aset = set(a.get(key, []) or [])
            bset = set(b.get(key, []) or [])
            if aset or bset:
                score += (0.15 if key == "state_types" else 0.125) * (len(aset & bset) / max(1, len(aset | bset)))
        return round(min(1.0, score), 4)

    def _semantic_tokens(self, value: object) -> set[str]:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).lower()
        return {token for token in re.findall(r"[a-z_][a-z0-9_]{1,}|[\u4e00-\u9fff]{2,}", text) if token not in {"true", "false", "null", "none"}}

    def _token_similarity(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, len(left | right))

    def _weighted_reward(self, record: dict[str, Any]) -> float:
        age_days = max(0.0, (time.time() - float(record.get("timestamp", time.time()))) / 86400.0)
        decay = 1.0 / (1.0 + age_days / 30.0)
        version_weight = 1.0 if record.get("project_version") in {"pteg-agent-v2", "pteg-agent-v3"} else 0.4
        return float(record.get("reward", 0.0)) * decay * version_weight

    def _delta(self, before: Any | None, after: Any | None, field: str) -> float:
        if before is None or after is None:
            return 0.0
        before_value = getattr(before, field, None)
        after_value = getattr(after, field, None)
        before_number = float(before_value) if before_value is not None else 0.0
        after_number = float(after_value) if after_value is not None else 0.0
        return round(after_number - before_number, 4)

    def _experience_id(self, graph: StateFlowGraph, agent: str, action: str, text: str) -> str:
        raw = f"{graph.source_path}|{agent}|{action}|{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _metrics(self, report: Any | None) -> dict[str, object]:
        if report is None:
            return {}
        return {
            "pytest_passed": getattr(report, "pytest_passed", None),
            "sfc": getattr(report, "sfc", None),
            "ae": getattr(report, "ae", None),
            "mutation_score": getattr(report, "mutation_score", None),
            "sfq": getattr(report, "sfq", None),
        }

    def _infer_oracle_kind(self, text: str) -> str:
        if "pytest.raises" in text:
            return "raises"
        if "pytest.approx" in text:
            return "approx"
        if " is True" in text:
            return "is_true"
        if " is False" in text:
            return "is_false"
        if "len(" in text and "==" in text:
            return "length_equals"
        if "==" in text:
            return "equals"
        return "unknown"

    def _infer_boundary_operator(self, text: str) -> str | None:
        for op in ["<=", ">=", "<", ">", "==", "!="]:
            if op in text:
                return op
        return None

    def _infer_input_partition(self, text: str) -> str:
        lowered = text.lower()
        if "none" in lowered:
            return "none"
        if "[]" in text or "''" in text:
            return "empty"
        if "-" in text:
            return "negative"
        if "0" in text:
            return "zero"
        return "nominal"

    def _complexity_bucket(self, features: dict[str, Any]) -> str:
        node_count = int(features.get("node_count", 0))
        branch_count = int(features.get("branch_count", 0))
        loop_count = int(features.get("loop_count", 0))
        score = node_count + (branch_count * 3) + (loop_count * 3)
        if score >= 80:
            return "high"
        if score >= 30:
            return "medium"
        return "low"
