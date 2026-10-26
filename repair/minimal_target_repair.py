from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from repair.repair_cost import estimate_repair_cost
from repair.repair_scope import RepairScope


@dataclass
class MinimalRepairDecision:
    scope: RepairScope
    expected_gain: float
    cost: dict[str, object]
    utility: float
    constraints: list[str]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["scope"] = self.scope.to_dict()
        return data


class StatePositioningRepairStrategy:
    """状态定位修复策略：根据失败和验证证据定位直接影响测试可行性的最小状态范围。"""
    LEVEL_ORDER = ["assertion-level", "line-level", "test-method-level", "test-class-level", "file-level"]

    def locate(self, graph: Any, test_path: str | Path, suggestions: list[Any] | None = None) -> MinimalRepairDecision:
        validation_errors = graph.metadata.get("unresolved_validation_errors", [])
        failed_tests = graph.metadata.get("last_report", {}).get("failed_tests", [])
        suggestions = suggestions or []
        scope = self._scope_from_state_flow(validation_errors, failed_tests, suggestions)
        expected_gain = self._expected_gain(graph, validation_errors, suggestions)
        cost = estimate_repair_cost(scope.level).to_dict()
        utility = round(expected_gain / max(float(cost["total"]), 0.01), 4)
        return MinimalRepairDecision(
            scope=scope,
            expected_gain=round(expected_gain, 4),
            cost=cost,
            utility=utility,
            constraints=[
                "syntax_must_pass",
                "tests_must_execute",
                "dynamic_validation_must_not_regress",
                "line_and_branch_coverage_must_not_regress",
                "SFC_AE_SFQ_must_not_regress",
                "do_not_delete_unrelated_passing_tests",
            ],
        )

    def _scope_from_state_flow(self, validation_errors: Any, failed_tests: Any, suggestions: list[Any]) -> RepairScope:
        errors = validation_errors if isinstance(validation_errors, list) else []
        failures = failed_tests if isinstance(failed_tests, list) else []
        first_error = errors[0] if errors else {}
        if errors:
            code = str(first_error.get("code", "validation_error")) if isinstance(first_error, dict) else "validation_error"
            related_nodes = list(first_error.get("related_nodes", [])) if isinstance(first_error, dict) else []
            if code in {"assertion_missing", "target_not_observed_in_test"}:
                return RepairScope("assertion-level", code, "validation state flow points to assertion/observation weakness", related_nodes, errors)
            if code in {"python_syntax_error", "junit_signal_missing", "pytest_signal_missing"}:
                return RepairScope("line-level", code, "rule validation points to a local syntax or framework-use defect", related_nodes, errors)
            if code in {"execution_failed", "execution_timeout"}:
                return RepairScope("test-method-level", code, "feasibility validation points to a failing executable test method", related_nodes, errors)
        if failures:
            first = failures[0] if isinstance(failures[0], dict) else {}
            return RepairScope("test-method-level", str(first.get("test_name", "pytest_failure")), "pytest failure state flow identifies one failing test method", [], errors)
        if suggestions:
            target = str(getattr(suggestions[0], "target_id", "graph_gap"))
            issue = str(getattr(suggestions[0], "issue", "graph_gap"))
            level = "assertion-level" if "assert" in issue else "test-method-level" if "mutation" in issue or "path" in issue else "line-level"
            return RepairScope(level, target, f"evaluation feedback prioritizes {issue}", [], errors)
        return RepairScope("file-level", "unknown", "no precise state-flow error was available; use conservative file-level fallback", [], errors)

    def _expected_gain(self, graph: Any, validation_errors: Any, suggestions: list[Any]) -> float:
        completeness = graph.metadata.get("complete_state_graph", {})
        csg_gap = 1.0 - float(completeness.get("csg", 0.0) or 0.0) if isinstance(completeness, dict) else 0.5
        validation_penalty = min(0.4, 0.1 * len(validation_errors)) if isinstance(validation_errors, list) else 0.0
        suggestion_priority = max([float(getattr(item, "priority", 0.0) or 0.0) for item in suggestions], default=0.0)
        return max(0.05, min(1.0, 0.45 * csg_gap + validation_penalty + 0.25 * suggestion_priority))


# Compatibility alias for older result files and imports. New code and documentation
# use the method name StatePositioningRepairStrategy.
MinimalTargetRepair = StatePositioningRepairStrategy
