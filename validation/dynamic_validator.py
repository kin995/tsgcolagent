from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from validation.validation_error import ValidationError, ValidationReport


class DynamicValidator:
    def validate(self, graph: Any, test_code: str | Path | None = None, context: dict[str, Any] | None = None, error_set: list[Any] | None = None) -> ValidationReport:
        context = context or graph.metadata.get("structured_context", {})
        test_text = self._read_test_text(test_code)
        errors: list[ValidationError] = []
        warnings: list[ValidationError] = []
        rule_score = self._rule_validation(test_text, context, errors, warnings)
        logic_score = self._logic_validation(graph, test_text, context, errors, warnings)
        feasibility_score = self._feasibility_validation(graph, errors, warnings)
        coverage_score = self._coverage_validation(graph, errors, warnings)
        repair_scope_score = self._repair_scope_validation(graph, error_set or [], errors, warnings)
        score = round(0.20 * rule_score + 0.25 * logic_score + 0.25 * feasibility_score + 0.20 * coverage_score + 0.10 * repair_scope_score, 4)
        return ValidationReport(
            passed=not errors and score >= 0.75,
            score=score,
            rule_score=round(rule_score, 4),
            logic_score=round(logic_score, 4),
            feasibility_score=round(feasibility_score, 4),
            coverage_score=round(coverage_score, 4),
            repair_scope_score=round(repair_scope_score, 4),
            errors=errors,
            warnings=warnings,
        )

    def _read_test_text(self, test_code: str | Path | None) -> str:
        if test_code is None:
            return ""
        path = Path(str(test_code))
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore")
        return str(test_code)

    def _rule_validation(self, test_text: str, context: dict[str, Any], errors: list[ValidationError], warnings: list[ValidationError]) -> float:
        if not test_text.strip():
            warnings.append(ValidationError("empty_test_code", "No test code was available for rule validation.", "warning"))
            return 0.5
        language = str(context.get("language", "python"))
        if language == "python":
            try:
                ast.parse(test_text)
            except SyntaxError as exc:
                errors.append(ValidationError("python_syntax_error", f"Generated pytest code has syntax error: {exc}", repair_hint="repair the smallest failing test function or assertion"))
                return 0.0
            if "pytest" not in test_text and "assert " not in test_text:
                warnings.append(ValidationError("pytest_signal_missing", "Python tests do not clearly use pytest/assert.", "warning"))
                return 0.75
        if language == "java" and "@Test" not in test_text and "assert" not in test_text:
            warnings.append(ValidationError("junit_signal_missing", "Java tests do not clearly use JUnit @Test/assertions.", "warning"))
            return 0.75
        return 1.0

    def _logic_validation(self, graph: Any, test_text: str, context: dict[str, Any], errors: list[ValidationError], warnings: list[ValidationError]) -> float:
        targets = self._target_names(context)
        if not targets:
            errors.append(ValidationError("no_testable_target", "Structured context contains no public function or method target.", repair_hint="rebuild context or include public methods"))
            return 0.0
        if test_text and not any(name in test_text for name in targets):
            warnings.append(ValidationError("target_not_observed_in_test", "Generated tests do not mention any structured target name.", "warning"))
            return 0.55
        assertion_count = int(graph.metadata.get("last_test_usability", {}).get("assertion_count", 0) or 0)
        if assertion_count <= 0 and test_text:
            errors.append(ValidationError("assertion_missing", "Tests run without effective observable assertions.", repair_hint="add assertion over return value, state change, exception, or collection content"))
            return 0.35
        return 1.0

    def _feasibility_validation(self, graph: Any, errors: list[ValidationError], warnings: list[ValidationError]) -> float:
        report = graph.metadata.get("last_report", {})
        if not report:
            warnings.append(ValidationError("execution_report_missing", "No execution report is available yet.", "warning"))
            return 0.6
        if not report.get("pytest_passed", False):
            errors.append(ValidationError("execution_failed", f"Tests are not executable: {report.get('failure_category', 'unknown')}", repair_hint="locate minimal failing test scope from execution state flow"))
            return 0.0
        if report.get("timed_out", False):
            errors.append(ValidationError("execution_timeout", "Tests timed out.", repair_hint="minimize or bound the timeout-causing test input"))
            return 0.25
        return 1.0

    def _coverage_validation(self, graph: Any, errors: list[ValidationError], warnings: list[ValidationError]) -> float:
        report = graph.metadata.get("last_report", {})
        sfc = float(report.get("sfc", 0.0) or 0.0)
        line = float(report.get("line_coverage", report.get("coverage_percent", 0.0)) or 0.0)
        branch = report.get("branch_coverage")
        branch_value = float(branch) if branch is not None else line
        score = max(0.0, min(1.0, 0.45 * sfc + 0.35 * line + 0.20 * branch_value))
        if line <= 0.0 and report:
            errors.append(ValidationError("coverage_missing", "No executable source coverage was observed.", repair_hint="repair import/path or generated test target"))
        elif score < 0.5:
            warnings.append(ValidationError("coverage_low", "Coverage evidence is still weak.", "warning", repair_hint="plan tests for uncovered state nodes and edges"))
        return score

    def _repair_scope_validation(self, graph: Any, error_set: list[Any], errors: list[ValidationError], warnings: list[ValidationError]) -> float:
        unresolved = graph.metadata.get("unresolved_validation_errors")
        if isinstance(unresolved, list) and unresolved:
            return 0.0
        repair = graph.metadata.get("state_positioning_repair") or graph.metadata.get("minimal_target_repair", {})
        if repair and repair.get("regression_detected"):
            errors.append(ValidationError("repair_regression", "Last repair introduced a validation or metric regression.", repair_hint="preserve max correct and narrow the scope"))
            return 0.0
        return 1.0

    def _target_names(self, context: dict[str, Any]) -> set[str]:
        names: set[str] = set()
        for item in context.get("functions", []):
            if isinstance(item, dict):
                names.add(str(item.get("name", "")))
                names.add(str(item.get("qualified_name", "")))
        for cls in context.get("classes", []):
            if not isinstance(cls, dict):
                continue
            names.add(str(cls.get("name", "")))
            for method in cls.get("methods", []) + cls.get("constructors", []):
                if isinstance(method, dict):
                    names.add(str(method.get("name", "")))
                    names.add(str(method.get("qualified_name", "")))
        return {name for name in names if name}
