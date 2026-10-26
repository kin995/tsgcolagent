"""Independent paper statistics. Never imported by planning or candidate selection.

Ratios are stored in [0, 1]; the UI converts them to percent. Missing evidence
is never a fabricated zero. Aggregate AvgExec is the mean of tasks with complete
tool-session counts, with its evidence coverage reported alongside the value.
TIR uses preregistered atomic predicates.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any


def number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def ratio(numerator: Any, denominator: Any) -> float | None:
    n, d = number(numerator), number(denominator)
    return n / d if n is not None and d is not None and 0 <= n <= d and d > 0 else None


def direct_metrics(report: dict, duration: Any = None) -> dict:
    mutation = report.get("mutation_evidence") or {}
    killed, survived = number(mutation.get("killed")), number(mutation.get("survived"))
    total_asserts, effective = number(report.get("assertion_count")), number(report.get("effective_assertions"))
    ae = number(report.get("ae"))
    lc = number(report.get("line_coverage")) if report.get("coverage_valid") is True else None
    bc = number(report.get("branch_coverage")) if report.get("coverage_valid") is True else None
    ms = number(report.get("mutation_score")) if report.get("mutation_valid") is True else None
    return {"line_coverage": lc, "branch_coverage": bc, "ae": ae,
            "mutation_score": ms, "duration_seconds": number(duration),
            "pass_rate": float(bool(report.get("pytest_passed")) and (report.get("collected_count") or 0) > 0 and not report.get("timed_out")),
            "metric_definition": "paper-v1", "effective_assertion_count": effective,
            "identified_assertion_count": total_asserts, "mutation_killed": killed,
            "mutation_survived": survived, "mutation_detected": mutation.get("detected_count"),
            "mutation_score_denominator": mutation.get("score_denominator")}


def target_status(goal: dict, evidence: dict) -> str:
    """An unobserved predicate is Unknown, not Unsatisfied.

    Tool adapters can publish explicit facts for any of the five quality
    categories. Recovery, covered line/arc and exact mutant identity have
    built-in adapters. A metric-increase target is preregistered with a fixed
    baseline; the aggregate TIR never feeds back into runtime selection.
    """
    key = str(goal.get("id") or "")
    fact = (evidence.get("facts") or {}).get(key)
    if fact in ("Satisfied", "Unsatisfied", "Unknown", "N/A"):
        return fact
    kind = goal.get("predicate")
    if kind == "runtime_growth":
        previous = goal.get("baseline_observations")
        current = evidence.get("runtime_observations")
        if not previous or not current or not (evidence.get("report") or {}).get("pytest_passed"):
            return "Unknown"
        def signatures(records):
            result = set()
            for record in records:
                if goal.get("category") == "exception":
                    if record.get("exception_type"):
                        result.add((record.get("function"), record.get("exception_type"), tuple(record.get("arg_types") or [])))
                elif record.get("return_type"):
                    result.add((record.get("function"), record.get("return_type"), tuple(record.get("arg_types") or [])))
            return result
        return "Satisfied" if signatures(current) - signatures(previous) else "Unsatisfied"
    if kind == "metric_increase":
        field = goal.get("field")
        report = evidence.get("report") or {}
        if report.get("pytest_passed") is False:
            return "Unsatisfied"
        if field == "mutation_score" and goal.get("baseline_profile") != report.get("evaluation_profile"):
            return "Unknown"
        value = direct_metrics(report).get(field)
        baseline = number(goal.get("baseline"))
        if value is None or baseline is None:
            return "Unknown"
        return "Satisfied" if value > baseline + 1e-9 else "Unsatisfied"
    if kind in {"collected", "executable"}:
        report = evidence.get("report") or {}
        if evidence.get("no_test") is True:
            return "Unsatisfied"
        if not report:
            return "Unknown"
        if report.get("failure_category") in {"java_tool_missing", "java_execution_unavailable"}:
            return "Unknown"
        count = number(report.get("collected_count"))
        if count is None:
            return "Unknown"
        value = count > 0 if kind == "collected" else count > 0 and report.get("pytest_passed") is True and not report.get("timed_out")
        return "Satisfied" if value else "Unsatisfied"
    coverage = evidence.get("coverage") or {}
    if kind == "line" and coverage.get("valid") is True:
        line = goal.get("line")
        if line in coverage.get("covered_lines", []):
            return "Satisfied"
        if line in coverage.get("missing_lines", []):
            return "Unsatisfied"
    if kind == "arc" and coverage.get("valid") is True:
        arc = goal.get("arc")
        if arc in coverage.get("covered_arcs", []):
            return "Satisfied"
        if arc in coverage.get("missing_arcs", []):
            return "Unsatisfied"
    if kind == "mutant":
        # Exact IDs only. A line-level kill must not credit other mutants.
        for mutant in (evidence.get("mutation") or {}).get("details", []):
            if isinstance(mutant, dict) and mutant.get("id") == goal.get("mutant_id"):
                status = str(mutant.get("status", "")).lower()
                return {"killed": "Satisfied", "survived": "Unsatisfied"}.get(status, "Unknown")
    return "Unknown"


def tir_statistics(actions: list[dict]) -> dict:
    seen: dict[tuple, tuple] = {}
    categories: dict[str, Counter] = {}
    excluded: Counter = Counter()
    records = []
    for action in actions:
        for goal in action.get("goals", []):
            identity = (action.get("task_id"), action.get("session_id"), action.get("round"), action.get("step"), goal.get("id"))
            before = target_status(goal, action.get("before") or {})
            after = target_status(goal, action.get("retained_after") or {})
            signature = (before, after, action.get("acted"), action.get("completed"))
            if identity in seen:
                if seen[identity] != signature:
                    raise ValueError("Conflicting duplicate TIR record")
                continue
            seen[identity] = signature
            reason = None
            if not goal.get("id") or action.get("preregistered") is not True:
                reason = "not_preregistered"
            elif not action.get("acted"):
                reason = "not_acted"
            elif not action.get("completed"):
                reason = "interrupted"
            elif before not in {"Satisfied", "Unsatisfied"} or after not in {"Satisfied", "Unsatisfied"}:
                reason = "incomparable_evidence"
            category = str(goal.get("category") or "unknown")
            counts = categories.setdefault(category, Counter())
            improved = before == "Unsatisfied" and after == "Satisfied"
            if reason:
                excluded[reason] += 1
                counts["excluded"] += 1
            else:
                counts["comparable"] += 1
                counts["improved"] += int(improved)
            records.append({"identity": list(identity), "category": category, "before": before,
                            "after": after, "improved": improved if not reason else None, "excluded_reason": reason})
    comparable = sum(c["comparable"] for c in categories.values())
    improved = sum(c["improved"] for c in categories.values())
    return {"tir": ratio(improved, comparable), "tir_comparable": comparable,
            "tir_improved": improved, "tir_excluded": sum(excluded.values()),
            "tir_status": "measured" if comparable else "incomplete_evidence" if excluded else "no_comparable_targets",
            "tir_categories": {k: {**dict(v), "tir": ratio(v["improved"], v["comparable"])} for k, v in categories.items()},
            "tir_exclusion_reasons": dict(excluded), "tir_records": records}


def aggregate_metrics(tasks: list[dict], selected_tasks: int | None = None) -> dict:
    total = selected_tasks if selected_tasks is not None else len(tasks)
    if total < len(tasks):
        raise ValueError("Selected task count is smaller than supplied final task records")
    metrics = [task.get("metrics", task) for task in tasks]
    quality_metrics = [m for task, m in zip(tasks, metrics)
                       if task.get("grade", "C") in {"A", "B", "C"} and m.get("pass_rate") == 1]
    complete = total > 0 and len(tasks) == total
    def mean(key: str) -> float | None:
        values = [number(m.get(key)) for m in (metrics if key == "duration_seconds" else quality_metrics)]
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None
    passed = sum(m.get("pass_rate") == 1 for m in metrics)
    known_exec = [number(m.get("exec_count")) for m in metrics]
    complete_exec = [value for m, value in zip(metrics, known_exec)
                     if m.get("exec_count_status") == "complete" and value is not None and value >= 0]
    exact_exec = complete and len(complete_exec) == total
    comparable = sum(int(m.get("tir_comparable") or 0) for m in metrics)
    improved = sum(int(m.get("tir_improved") or 0) for m in metrics)
    return {"metric_definition": "paper-v1", "selected_tasks": total, "recorded_tasks": len(tasks),
            "pass_rate": ratio(passed, total) if complete else None, "passed_tasks": passed,
            "avg_exec": sum(complete_exec) / len(complete_exec) if complete_exec else None,
            "avg_exec_status": "complete" if exact_exec else "partial_evidence" if complete_exec else "incomplete_evidence",
            "exec_known_total": sum(v for v in known_exec if v is not None),
            "exec_complete_tasks": len(complete_exec),
            "average_time_seconds": mean("duration_seconds") if complete and all(number(m.get("duration_seconds")) is not None for m in metrics) else None,
            "tir": ratio(improved, comparable), "tir_improved": improved, "tir_comparable": comparable,
            "tir_excluded": sum(int(m.get("tir_excluded") or 0) for m in metrics),
            "knowledge_occurrence": len({key for m in metrics for key in m.get("knowledge_record_ids", [])}) if exact_exec else None,
            "knowledge_use_frequency": sum(int(m.get("knowledge_use_frequency") or 0) for m in metrics) if exact_exec else None,
            **{f"average_{key}": mean(key) for key in ("line_coverage", "branch_coverage", "ae", "mutation_score")},
            "valid_metric_tasks": {key: sum(number(m.get(key)) is not None for m in quality_metrics) for key in ("line_coverage", "branch_coverage", "ae", "mutation_score")}}
