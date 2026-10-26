"""Keep process-only quality measures out of user-facing results.

The execution graph and agent decisions may still use these values.  Call
``public_result`` only when data crosses a report, API, or display boundary.
It also cleans historical result objects loaded from disk.
"""

from __future__ import annotations

import re
from typing import Any


_INTERNAL_TOKENS = frozenset({"sfc", "nrr", "tsq", "sfq", "csg", "pv", "dvs", "tic", "rcs"})
_INTERNAL_KEYS = frozenset(
    {
        "combined_coverage",
        "coverage_percent",
        "state_flow_coverage",
        "state_flow_quality",
        "test_state_quality",
        "complete_state_graph",
        "complete_state_graph_report",
        "overall_quality_index",
        "quality_assessment",
        "thresholds",
        "threshold_failures",
        "weak_quality_flags",
        "weak_quality_flag_counts",
        "gaps",
        "gap_values",
        "metric_delta",
        "priority",
        "no_regression_after_repair",
        "average_quality_gain",
        "quality_gain",
        "quality_delta",
        "overall_quality_score",
        "success_rate",
        "success_count",
        "completion_rate",
        "cost",
        "confidence",
        "reward",
        "expected_reward",
        "predicted_reward",
        "expected_gain",
    }
)
_DERIVED_ALIASES = (
    "state_flow_coverage", "state_flow_quality", "test_state_quality",
    "complete_state_graph", "combined_coverage", "coverage_percent",
    "overall_quality_index", "quality_gain", "quality_delta",
    "success_rate", "success_count", "completion_rate",
    "node_coverage", "edge_coverage", "path_coverage", "test_intent_coverage",
)
_PROCESS_SCORE_SUFFIXES = (
    "defect_score", "coverage_score", "quality_score", "validation_score",
    "logic_score", "rule_score", "feasibility_score", "repair_scope_score",
    "path_score", "node_score", "edge_score", "match_score", "ranking_score",
    "knowledge_score", "evidence_score", "confidence_score", "reward_score",
    "completeness_score", "agent_score", "candidate_score", "total_score",
    "oracle_score",
)
_METRIC_LIST_KEYS = frozenset({"metric_names", "metric_keys", "metric_labels", "metrics_available", "visible_metrics", "report_columns", "columns"})
_IDENTITY_LIST_KEYS = frozenset({
    "target_methods", "target_states", "target_branches", "method_names",
    "function_names", "called_functions", "called_methods", "test_names", "sources",
})
_IDENTITY_KEYS = frozenset(
    {
        "task_id", "session_id", "record_id", "source", "source_path",
        "source_hash", "source_sha256", "test_hash", "test_path", "graph_path",
        "output_dir", "summary_path", "checkpoint_path", "ledger_path",
        "file_path", "path", "url", "uri", "node_id", "edge_id",
        "code", "source_code", "test_code",
        *_IDENTITY_LIST_KEYS,
    }
)
_IDENTITY_MAP_KEYS = frozenset({"nodes", "nodes_by_id", "edges_by_id", "records_by_id"})
_METRIC_LABEL_KEYS = frozenset({"metric", "metric_id", "metric_name", "key", "id", "name", "label"})
_METRIC_VALUE_KEYS = frozenset({"value", "raw", "amount", "percentage"})
_OMIT = object()

_NAME_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"state[_ -]?flow[_ -]?coverage|state[_ -]?flow[_ -]?quality|"
    r"test[_ -]?state[_ -]?quality|complete[_ -]?state[_ -]?graph|"
    r"combined[_ -]?coverage|overall[_ -]?quality[_ -]?index|"
    r"(?:average[_ -]?)?quality[_ -]?(?:gain|delta)|"
    r"(?:estimated|screening|predicted|preliminary)[_ -]?mutation[_ -]?score|"
    r"success[_ -]?(?:rate|count)|completion[_ -]?rate|confidence|reward|"
    r"(?:defect|coverage|quality|validation|logic|rule|feasibility|"
    r"assertion[_ -]?quality|repair[_ -]?scope|path)[_ -]?score|"
    r"(?:coverage|assertion|mutation|runtime[_ -]?type|exception|state|quality)[_ -]?gap|"
    r"(?:repair|execution|modification|agent)[_ -]?cost|priority|"
    r"缺陷风险(?:分数|评分)?|风险分数|质量指数|完成率|"
    r"测试状态覆盖|状态流覆盖|测试状态质量|状态流质量|状态图完整度|"
    r"无回归率|无退化率|"
    r"(?:average[_ -]?|avg[_ -]?|delta[_ -]?|estimated[_ -]?)?"
    r"(?:sfc|nrr|tsq|sfq|csg|pv|dvs|tic|rcs)(?:[_ -]?(?:gap|score|delta))?"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_VALUE_PATTERN = re.compile(
    _NAME_PATTERN.pattern + r"(?:\s*(?:[:：=<>]=?|≥|≤)\s*|\s+)[-+]?\d+(?:\.\d+)?%?",
    re.IGNORECASE,
)


def internal_metric_key(key: object) -> bool:
    """Return whether a result field contains an unpublished process metric."""
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).casefold()
    if normalized in {"mutation_score", "average_mutation_score", "pass_rate", "average_pass_rate"}:
        return False
    return (
        normalized in _INTERNAL_KEYS
        or normalized.endswith("_mutation_score")
        or normalized == "score"
        or any(normalized.endswith(suffix) for suffix in _PROCESS_SCORE_SUFFIXES)
        or normalized.endswith(("_gap", "_defect", "_confidence", "_reward", "_cost", "_rate"))
        or any(alias in normalized for alias in _DERIVED_ALIASES)
        or bool(_INTERNAL_TOKENS.intersection(re.findall(r"[a-z0-9]+", normalized)))
    )


def public_text(value: str) -> str:
    """Redact metric names and paired values from free-form result text."""
    cleaned = _VALUE_PATTERN.sub("内部过程指标已隐藏", value)
    return _NAME_PATTERN.sub("内部过程指标已隐藏", cleaned)


def public_result(value: Any) -> Any:
    """Return a detached, recursively filtered public representation."""
    result = _public_result(value)
    return None if result is _OMIT else result


def _public_result(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        if parent_key != "node_record" and _METRIC_VALUE_KEYS.intersection(value) and any(
            internal_metric_key(value.get(key)) for key in _METRIC_LABEL_KEYS
        ):
            return _OMIT
        result = {}
        for key, item in value.items():
            if parent_key not in _IDENTITY_MAP_KEYS and internal_metric_key(key):
                continue
            if parent_key == "node_record" and key in {"name", "target", "source"} and isinstance(item, str):
                projected = item
            else:
                child_context = "node_record" if parent_key in _IDENTITY_MAP_KEYS and isinstance(item, dict) else key if isinstance(key, str) else None
                projected = _public_result(item, child_context)
            if projected is not _OMIT:
                result[key] = projected
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            if parent_key in _METRIC_LIST_KEYS and isinstance(item, str) and internal_metric_key(item):
                continue
            child_context = "node_record" if parent_key in {"nodes", "edges"} and isinstance(item, dict) else parent_key
            projected = _public_result(item, child_context)
            if projected is not _OMIT:
                result.append(projected)
        return result
    if isinstance(value, tuple):
        result = _public_result(list(value), parent_key)
        return tuple(result)
    if isinstance(value, str):
        return value if parent_key in _IDENTITY_KEYS else public_text(value)
    return value
