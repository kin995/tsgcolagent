from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from .state_graph import StateFlowGraph


class GraphUpdater:
    def __init__(self, graph: StateFlowGraph):
        self.graph = graph

    def reset_current_run_state(self) -> None:
        evidence_types = {"TestCase", "Assertion", "Mutation", "RuntimeState", "RuntimeType", "Execution"}
        for node in self.graph.nodes.values():
            if node.node_type in evidence_types:
                continue
            node.visited = False
            node.visit_count = 0
            node.assert_count = 0
            node.mutation_status = "unknown"
            node.coverage_status = "uncovered"
            node.repair_flag = False
            node.coverage_defect = 0.0
            node.assertion_defect = 0.0
            node.mutation_defect = 0.0
            node.exception_defect = 0.0
            node.type_defect = 0.0
            node.defect_score = 0.0
        for edge in self.graph.edges:
            edge.visited = False
            edge.flow_count = 0
            edge.metadata.pop("coverage_arc_matched", None)

    def apply_coverage_lines(self, covered_lines: set[int]) -> None:
        for node in list(self.graph.nodes.values()):
            if node.node_type in {"TestCase", "Assertion", "Mutation", "RuntimeState", "RuntimeType", "Execution"}:
                continue
            if any(node.line_start <= line <= node.line_end for line in covered_lines):
                node.mark_visited()
                node.metadata.setdefault("first_covered_iteration", self.graph.iteration + 1)
                node.metadata["last_covered_iteration"] = self.graph.iteration + 1

    def apply_assert_counts(self, test_file: str | Path) -> int:
        path = Path(test_file)
        if not path.exists():
            return 0
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return text.count("assert ") + text.count("pytest.raises")
        functions = self._function_lookup()
        assert_count = 0
        for item in tree.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                continue
            called_functions = self._called_target_call_infos(item)
            for child in ast.walk(item):
                if isinstance(child, ast.Assert) or self._is_pytest_raises(child):
                    assert_count += 1
                    specificity = self._assertion_specificity(child)
                    for call_info in called_functions:
                        function_name = str(call_info["function"])
                        function = functions.get((call_info.get("class_name"), function_name)) or functions.get((None, function_name))
                        if function and specificity > 0:
                            function.add_asserts(1)
        return assert_count

    def apply_mutation_results(self, mutation_results: dict[str, object]) -> None:
        killed_lines = set(mutation_results.get("killed_lines", []))
        survived_lines = set(mutation_results.get("survived_lines", []))
        if mutation_results.get("score") is None:
            return
        for node in list(self.graph.nodes.values()):
            node_lines = set(range(node.line_start, node.line_end + 1))
            if node_lines & survived_lines:
                node.mutation_status = "survived"
                node.repair_flag = True
                node.priority = max(node.priority, 0.95)
            elif node_lines & killed_lines:
                node.mutation_status = "killed"

    def apply_java_mutation_results(self, pit_results: dict[str, object]) -> None:
        normalized = dict(pit_results)
        normalized["score"] = pit_results.get("mutation_score", pit_results.get("score"))
        self.apply_mutation_results(normalized)

    def record_java_mutation_evidence(self, execution_id: str, pit_results: dict[str, object]) -> None:
        details = pit_results.get("details", [])
        if not isinstance(details, list):
            return
        test_cases = [node for node in self.graph.nodes.values() if node.node_type == "TestCase" and node.metadata.get("language") == "java"]
        target_types = {"Package", "Class", "Constructor", "Method", "Function", "Branch", "Call", "Return"}
        for index, detail in enumerate(details, start=1):
            if not isinstance(detail, dict):
                continue
            try:
                line = int(detail.get("line", 0) or 0)
            except (TypeError, ValueError):
                line = 0
            status = str(detail.get("category") or detail.get("status") or "unknown").lower()
            mutation = self.graph.add_entity(
                "Mutation",
                f"PIT mutation {index} line {line} {status}",
                line_start=max(1, line),
                line_end=max(1, line),
                priority=1.0 if status == "survived" else 0.3,
                metadata={
                    "stable_id": f"java-mutation:{execution_id}:{index}",
                    "language": "java",
                    "execution_id": execution_id,
                    "iteration": self.graph.iteration,
                    "mutant_id": index,
                    **detail,
                },
            )
            self.graph.add_relation(execution_id, mutation.node_id, "Produce", weight=0.8)
            for target in self.graph.nodes.values():
                if target.node_type in target_types and line > 0 and target.line_start <= line <= target.line_end:
                    self.graph.add_relation(mutation.node_id, target.node_id, "Trigger", weight=0.9)
                    if status == "survived":
                        self.graph.add_relation(mutation.node_id, target.node_id, "RepairTarget", weight=1.0)
            if status == "killed":
                killing_test = str(detail.get("killing_test") or "")
                matched = [test for test in test_cases if str(test.metadata.get("test_name") or "") in killing_test]
                for test_case in matched:
                    self.graph.add_relation(mutation.node_id, test_case.node_id, "Kill", weight=0.8)
            elif status == "survived":
                self.graph.add_relation(mutation.node_id, execution_id, "Survive", weight=1.0)

    def mark_probable_edges(self) -> None:
        if any(edge.metadata.get("coverage_arc_matched") for edge in self.graph.edges):
            return
        for edge in self.graph.edges:
            edge.metadata["coverage_arc_status"] = "unknown"

    def apply_coverage_arcs(self, covered_arcs: set[tuple[int, int]]) -> None:
        if not covered_arcs:
            return
        for edge in self.graph.edges:
            source = self.graph.get_node(edge.source)
            target = self.graph.get_node(edge.target)
            if not source or not target:
                continue
            candidate_arcs = {
                (source.line_start, target.line_start),
                (source.line_end, target.line_start),
            }
            if candidate_arcs & covered_arcs:
                edge.metadata["coverage_arc_matched"] = True
                edge.metadata["traceable"] = True
                edge.metadata["evidence_source"] = "coverage.py executed_branches"
                edge.metadata["covered_arc"] = list(sorted(candidate_arcs & covered_arcs)[0])
                edge.metadata["confidence"] = 1.0
                edge.metadata.setdefault("first_covered_iteration", self.graph.iteration + 1)
                edge.metadata["last_covered_iteration"] = self.graph.iteration + 1
                edge.mark_visited()

    def apply_java_coverage_edges(self) -> None:
        for edge in self.graph.edges:
            source = self.graph.get_node(edge.source)
            target = self.graph.get_node(edge.target)
            if not source or not target:
                continue
            if source.visited and target.visited:
                edge.metadata["coverage_arc_matched"] = True
                edge.metadata["traceable"] = True
                edge.metadata["evidence_source"] = "jacoco_line_coverage"
                edge.metadata["confidence"] = 0.75
                edge.mark_visited()

    def record_java_execution_evidence(
        self,
        test_file: str | Path,
        toolchain_result: dict[str, Any],
        covered_lines: set[int],
    ) -> str:
        path = Path(test_file)
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        jacoco = toolchain_result.get("jacoco", {}) if isinstance(toolchain_result.get("jacoco"), dict) else {}
        pit = toolchain_result.get("pit", {}) if isinstance(toolchain_result.get("pit"), dict) else {}
        execution = self.graph.add_entity(
            "Execution",
            f"JavaExecution_{self.graph.iteration + 1:03d}",
            metadata={
                "language": "java",
                "junit_passed": bool(toolchain_result.get("passed")),
                "build_system": toolchain_result.get("build_system"),
                "line_coverage": toolchain_result.get("line_coverage"),
                "branch_coverage": toolchain_result.get("branch_coverage"),
                "method_coverage": toolchain_result.get("method_coverage"),
                "mutation_score": toolchain_result.get("mutation_score"),
                "test_file": str(path.resolve()) if path.exists() else str(path),
                "jacoco": {
                    "valid": jacoco.get("valid"),
                    "target_class": jacoco.get("target_class"),
                    "covered_lines": jacoco.get("covered_lines", []),
                    "missing_lines": jacoco.get("missing_lines", []),
                    "csv_path": jacoco.get("csv_path"),
                    "xml_path": jacoco.get("xml_path"),
                },
                "pit": {
                    "valid": pit.get("valid"),
                    "total": pit.get("total", 0),
                    "effective_mutants": pit.get("effective_mutants", 0),
                    "killed": pit.get("killed", 0),
                    "survived": pit.get("survived", 0),
                    "score_reliability": pit.get("score_reliability", "unavailable"),
                    "xml_path": pit.get("xml_path"),
                },
            },
        )
        functions = [node for node in self.graph.nodes.values() if node.node_type in {"Function", "Async", "Method", "Constructor"}]
        tests = self._java_test_methods(text)
        for test_name, body, start_line, end_line in tests:
            test_node = self.graph.upsert_node(self.graph.add_entity(
                "TestCase",
                test_name,
                line_start=start_line,
                line_end=end_line,
                code=body,
                priority=0.5,
                metadata={
                    "stable_id": f"java-test:{path.resolve()}::{test_name}",
                    "language": "java",
                    "test_name": test_name,
                    "nodeid": f"{path.name}::{test_name}",
                    "test_file": str(path.resolve()),
                },
            ))
            self.graph.add_relation(execution.node_id, test_node.node_id, "Produce", weight=0.7)
            called = self._java_called_functions(body, functions)
            for function_node in called:
                self.graph.add_relation(test_node.node_id, function_node.node_id, "Execute", weight=0.75)
                self.graph.add_relation(execution.node_id, function_node.node_id, "Execute", weight=0.7)
                state = self.graph.add_entity(
                    "RuntimeState",
                    f"{test_name}:{function_node.metadata.get('scope', '')}.{function_node.name} runtime",
                    line_start=function_node.line_start,
                    line_end=function_node.line_end,
                    priority=0.65,
                    metadata={
                        "language": "java",
                        "execution_id": execution.node_id,
                        "test_name": test_name,
                        "function": function_node.name,
                        "class_name": function_node.metadata.get("scope"),
                        "covered_lines": sorted(line for line in covered_lines if function_node.line_start <= line <= function_node.line_end),
                        "return": "observed_by_junit_assertion",
                    },
                )
                self.graph.add_relation(function_node.node_id, state.node_id, "Produce", weight=0.8)
                self.graph.add_relation(state.node_id, function_node.node_id, "Depend", weight=0.9)
                self._record_runtime_types(function_node, state.node_id)
            for assertion_index, assertion_code in enumerate(self._java_assertions(body), start=1):
                assertion = self.graph.upsert_node(self.graph.add_entity(
                    "Assertion",
                    f"{test_name}:assertion:{assertion_index}",
                    line_start=start_line,
                    line_end=end_line,
                    code=assertion_code,
                    priority=0.75,
                    metadata={
                        "stable_id": f"java-assertion:{path.resolve()}::{test_name}:{assertion_index}",
                        "language": "java",
                        "test_name": test_name,
                        "test_file": str(path.resolve()),
                        "specificity": self._java_assertion_specificity(assertion_code),
                        "effective": self._java_assertion_specificity(assertion_code) >= 0.6,
                    },
                ))
                self.graph.add_relation(test_node.node_id, assertion.node_id, "Depend", weight=0.9)
                for function_node in called:
                    if self._java_assertion_specificity(assertion_code) >= 0.6:
                        function_node.add_asserts(1)
                        self.graph.add_relation(assertion.node_id, function_node.node_id, "Observe", weight=0.85)
        return execution.node_id

    def record_execution_evidence(
        self,
        test_file: str | Path,
        pytest_result: Any,
        coverage_result: Any,
        mutation_result: Any,
        runtime_observations: list[Any] | None = None,
    ) -> str:
        execution = self.graph.add_entity(
            "Execution",
            f"Execution_{self.graph.iteration + 1:03d}",
            metadata={
                "pytest_passed": bool(getattr(pytest_result, "passed", False)),
                "pytest_returncode": getattr(pytest_result, "returncode", None),
                "coverage_percent": getattr(coverage_result, "percent", 0.0),
                "line_coverage": getattr(coverage_result, "line_coverage", 0.0),
                "branch_coverage": getattr(coverage_result, "branch_coverage", None),
                "combined_coverage": getattr(coverage_result, "combined_coverage", getattr(coverage_result, "percent", 0.0)),
                "mutation_score": getattr(mutation_result, "score", 0.0),
                "test_file": str(Path(test_file).resolve()),
            },
        )
        covered_lines = set(getattr(coverage_result, "covered_lines", set()))
        function_runtime_states = self._record_runtime_states(execution.node_id, covered_lines, runtime_observations or [])
        test_cases, assertion_nodes = self._record_test_cases_and_assertions(test_file, function_runtime_states)
        self._record_mutations(execution.node_id, mutation_result, test_cases)
        for runtime_state in function_runtime_states.values():
            for assertion in assertion_nodes:
                if assertion.metadata.get("test_name") == runtime_state.metadata.get("test_name"):
                    self.graph.add_relation(assertion.node_id, runtime_state.node_id, "Observe", weight=0.95)
        return execution.node_id

    def _record_runtime_states(self, execution_id: str, covered_lines: set[int], observations: list[Any]) -> dict[str, Any]:
        runtime_states: dict[str, Any] = {}
        if observations:
            return self._record_observed_runtime_states(execution_id, observations)
        for node in list(self.graph.nodes.values()):
            if node.node_type not in {"Function", "Async", "Method", "Constructor"}:
                continue
            if not any(node.line_start <= line <= node.line_end for line in covered_lines):
                continue
            state = self.graph.add_entity(
                "RuntimeState",
                f"{node.name} runtime",
                line_start=node.line_start,
                line_end=node.line_end,
                priority=0.6,
                metadata={
                    "execution_id": execution_id,
                    "function": node.name,
                    "covered_lines": sorted(line for line in covered_lines if node.line_start <= line <= node.line_end),
                    "args": node.metadata.get("args", []),
                    "return": "observed_by_assertion",
                    "exception": None,
                    "duration": None,
                },
            )
            runtime_states[node.name] = state
            self.graph.add_relation(execution_id, node.node_id, "Execute", weight=0.5)
            self.graph.add_relation(execution_id, state.node_id, "Produce", weight=0.6)
            self.graph.add_relation(state.node_id, node.node_id, "Depend", weight=0.9)
            self._record_runtime_types(node, state.node_id)
        return runtime_states

    def _record_observed_runtime_states(self, execution_id: str, observations: list[Any]) -> dict[str, Any]:
        runtime_states: dict[str, Any] = {}
        functions = self._function_lookup()
        counters: dict[str, int] = {}
        for observation in observations:
            function_name = str(getattr(observation, "function", ""))
            class_name = getattr(observation, "class_name", None)
            class_name = str(class_name) if class_name else None
            test_name = str(getattr(observation, "test_name", ""))
            function_node = functions.get((class_name, function_name)) or functions.get((None, function_name))
            if not function_node:
                continue
            key = self._runtime_state_key(test_name, function_name, class_name)
            legacy_key = f"{test_name}::{function_name}"
            counters[key] = counters.get(key, 0) + 1
            state = self.graph.add_entity(
                "RuntimeState",
                f"{test_name}:{class_name + '.' if class_name else ''}{function_name} runtime",
                line_start=function_node.line_start,
                line_end=function_node.line_end,
                priority=0.7,
                metadata={
                    "execution_id": execution_id,
                    "test_name": test_name,
                    "nodeid": getattr(observation, "nodeid", test_name),
                    "function": function_name,
                    "class_name": class_name,
                    "arg_types": list(getattr(observation, "arg_types", [])),
                    "return_type": getattr(observation, "return_type", None),
                    "exception_type": getattr(observation, "exception_type", None),
                    "exception_message_pattern": getattr(observation, "exception_message_pattern", None),
                    "value_summary": self._safe_value_summary(getattr(observation, "value_summary", None)),
                    "observation_index": counters[key],
                },
            )
            runtime_states[f"{key}::{counters[key]}"] = state
            runtime_states.setdefault(key, state)
            runtime_states.setdefault(legacy_key, state)
            runtime_states.setdefault(function_name, state)
            self.graph.add_relation(execution_id, function_node.node_id, "Execute", weight=0.7)
            self.graph.add_relation(execution_id, state.node_id, "Produce", weight=0.8)
            self.graph.add_relation(function_node.node_id, state.node_id, "Produce", weight=0.85)
            self.graph.add_relation(state.node_id, function_node.node_id, "Depend", weight=0.9)
            self._record_observed_runtime_types(function_node, state.node_id, observation)
        return runtime_states

    def _record_runtime_types(self, function_node: Any, runtime_state_id: str) -> None:
        annotations = function_node.metadata.get("arg_annotations", {})
        for arg_name in function_node.metadata.get("args", []):
            type_name = "observed"
            if isinstance(annotations, dict) and annotations.get(arg_name):
                type_name = str(annotations[arg_name])
            type_node = self.graph.add_entity(
                "RuntimeType",
                f"{function_node.name}.{arg_name}:{type_name}",
                line_start=function_node.line_start,
                line_end=function_node.line_start,
                priority=0.4,
                metadata={"runtime_state_id": runtime_state_id, "function": function_node.name, "variable": arg_name, "type": type_name},
            )
            self.graph.add_relation(runtime_state_id, type_node.node_id, "TypeFlow", weight=0.75)
            self.graph.add_relation(function_node.node_id, type_node.node_id, "TypeFlow", weight=0.75)
        return_type = function_node.metadata.get("returns") or "observed"
        return_type_node = self.graph.add_entity(
            "RuntimeType",
            f"{function_node.name}.return:{return_type}",
            line_start=function_node.line_start,
            line_end=function_node.line_start,
            priority=0.4,
            metadata={"runtime_state_id": runtime_state_id, "function": function_node.name, "variable": "return", "type": return_type},
        )
        self.graph.add_relation(runtime_state_id, return_type_node.node_id, "TypeFlow", weight=0.75)
        self.graph.add_relation(function_node.node_id, return_type_node.node_id, "TypeFlow", weight=0.75)

    def _record_observed_runtime_types(self, function_node: Any, runtime_state_id: str, observation: Any) -> None:
        for index, type_name in enumerate(getattr(observation, "arg_types", [])):
            type_node = self.graph.add_entity(
                "RuntimeType",
                f"{function_node.name}.arg{index}:{type_name}",
                line_start=function_node.line_start,
                line_end=function_node.line_start,
                priority=0.5,
                metadata={
                    "runtime_state_id": runtime_state_id,
                    "function": function_node.name,
                    "variable": f"arg{index}",
                    "type": str(type_name),
                    "source": "runtime_observation",
                },
            )
            self.graph.add_relation(runtime_state_id, type_node.node_id, "TypeFlow", weight=0.85)
            self.graph.add_relation(function_node.node_id, type_node.node_id, "TypeFlow", weight=0.85)
        return_type = getattr(observation, "return_type", None)
        exception_type = getattr(observation, "exception_type", None)
        if return_type:
            self._add_observed_type(function_node, runtime_state_id, "return", str(return_type))
        if exception_type:
            self._add_observed_type(function_node, runtime_state_id, "exception", str(exception_type))

    def _add_observed_type(self, function_node: Any, runtime_state_id: str, variable: str, type_name: str) -> None:
        type_node = self.graph.add_entity(
            "RuntimeType",
            f"{function_node.name}.{variable}:{type_name}",
            line_start=function_node.line_start,
            line_end=function_node.line_start,
            priority=0.55,
            metadata={
                "runtime_state_id": runtime_state_id,
                "function": function_node.name,
                "variable": variable,
                "type": type_name,
                "source": "runtime_observation",
            },
        )
        self.graph.add_relation(runtime_state_id, type_node.node_id, "TypeFlow", weight=0.9)
        self.graph.add_relation(function_node.node_id, type_node.node_id, "TypeFlow", weight=0.9)

    def _record_test_cases_and_assertions(self, test_file: str | Path, runtime_states: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        path = Path(test_file)
        if not path.exists():
            return [], []
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [], []
        test_cases: list[Any] = []
        assertions: list[Any] = []
        functions = self._function_lookup()
        for item in tree.body:
            if not isinstance(item, ast.FunctionDef) or not item.name.startswith("test_"):
                continue
            test_node = self.graph.upsert_node(self.graph.add_entity(
                "TestCase",
                item.name,
                line_start=getattr(item, "lineno", 1),
                line_end=getattr(item, "end_lineno", getattr(item, "lineno", 1)),
                code=ast.get_source_segment(text, item) or "",
                priority=0.5,
                metadata={
                    "stable_id": f"test:{path.resolve()}::{item.name}",
                    "test_name": item.name,
                    "nodeid": f"{path.name}::{item.name}",
                    "test_file": str(path.resolve()),
                },
            ))
            test_cases.append(test_node)
            called_infos = self._called_target_call_infos(item)
            called_functions = {str(info["function"]) for info in called_infos}
            for call_info in called_infos:
                function_name = str(call_info["function"])
                class_name = call_info.get("class_name")
                function_node = functions.get((class_name, function_name)) or functions.get((None, function_name))
                if function_node:
                    self.graph.add_relation(test_node.node_id, function_node.node_id, "Execute", weight=0.5)
                    state = (
                        runtime_states.get(self._runtime_state_key(item.name, function_name, class_name))
                        or runtime_states.get(f"{item.name}::{function_name}")
                        or runtime_states.get(function_name)
                    )
                    if state:
                        state.metadata["test_name"] = item.name
            for child in ast.walk(item):
                if isinstance(child, ast.Assert) or self._is_pytest_raises(child):
                    specificity = self._assertion_specificity(child)
                    observed_infos = self._assert_observed_call_infos(item, child) or list(called_infos)
                    assertion = self.graph.upsert_node(self.graph.add_entity(
                        "Assertion",
                        f"{item.name}:assert@{getattr(child, 'lineno', 1)}",
                        line_start=getattr(child, "lineno", 1),
                        line_end=getattr(child, "end_lineno", getattr(child, "lineno", 1)),
                        code=ast.get_source_segment(text, child) or "",
                        priority=0.7,
                        metadata={
                            "stable_id": f"assertion:{path.resolve()}:{item.name}:{getattr(child, 'lineno', 1)}",
                            "test_name": item.name,
                            "test_file": str(path.resolve()),
                            "specificity": specificity,
                            "effective": specificity > 0,
                        },
                    ))
                    assertions.append(assertion)
                    self.graph.add_relation(test_node.node_id, assertion.node_id, "Depend", weight=0.9)
                    for call_info in observed_infos:
                        function_name = str(call_info["function"])
                        class_name = call_info.get("class_name")
                        state = (
                            runtime_states.get(self._runtime_state_key(item.name, function_name, class_name))
                            or runtime_states.get(f"{item.name}::{function_name}")
                            or runtime_states.get(function_name)
                        )
                        if state:
                            self.graph.add_relation(assertion.node_id, state.node_id, "Observe", weight=0.95)
        return test_cases, assertions

    def _record_mutations(self, execution_id: str, mutation_result: Any, test_cases: list[Any]) -> None:
        status_lines = {
            "killed": list(getattr(mutation_result, "killed_lines", [])),
            "survived": list(getattr(mutation_result, "survived_lines", [])),
            "timeout": list(getattr(mutation_result, "timeout_lines", [])),
            "invalid": list(getattr(mutation_result, "invalid_lines", [])),
            "infra_error": list(getattr(mutation_result, "infra_error_lines", [])),
            "equivalent": list(getattr(mutation_result, "equivalent_lines", [])),
        }
        mutant_id = 0
        for status, lines in status_lines.items():
            for line in lines:
                mutant_id += 1
                self._record_mutation_line(line, status, execution_id, test_cases, mutant_id)

    def _record_mutation_line(self, line: int, status: str, execution_id: str, test_cases: list[Any], mutant_id: int) -> None:
        mutation = self.graph.add_entity(
            "Mutation",
            f"Mutation line {line} {status}",
            line_start=line,
            line_end=line,
            priority=1.0 if status == "survived" else 0.3,
            metadata={
                "line": line,
                "status": status,
                "execution_id": execution_id,
                "iteration": self.graph.iteration,
                "mutant_id": mutant_id,
                "operator": "simple_ast_mutator",
            },
        )
        target_nodes = [
            node for node in self.graph.nodes.values()
            if node.node_type not in {"Mutation", "Execution", "RuntimeState", "RuntimeType", "TestCase", "Assertion"}
            and node.line_start <= line <= node.line_end
        ]
        for target in target_nodes:
            self.graph.add_relation(mutation.node_id, target.node_id, "Trigger", weight=0.9)
            if status == "survived":
                self.graph.add_relation(mutation.node_id, target.node_id, "RepairTarget", weight=1.0)
        if status == "killed":
            for test_case in test_cases:
                self.graph.add_relation(mutation.node_id, test_case.node_id, "Kill", weight=0.2)
        elif status == "survived":
            self.graph.add_relation(mutation.node_id, execution_id, "Survive", weight=1.0)

    def _java_test_methods(self, text: str) -> list[tuple[str, str, int, int]]:
        methods: list[tuple[str, str, int, int]] = []
        pattern = re.compile(r"@Test\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{", re.MULTILINE)
        for match in pattern.finditer(text):
            open_index = text.find("{", match.end() - 1)
            close_index = self._matching_brace(text, open_index)
            body = text[match.start(): close_index + 1]
            start_line = text.count("\n", 0, match.start()) + 1
            end_line = text.count("\n", 0, close_index) + 1
            methods.append((match.group(1), body, start_line, end_line))
        return methods

    def _matching_brace(self, text: str, open_index: int) -> int:
        depth = 0
        for index in range(max(0, open_index), len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return index
        return len(text) - 1

    def _java_called_functions(self, body: str, functions: list[Any]) -> list[Any]:
        called: list[Any] = []
        seen: set[str] = set()
        for node in functions:
            class_name = str(node.metadata.get("scope") or "")
            patterns = [
                rf"\b{re.escape(class_name)}\s*\.\s*{re.escape(node.name)}\s*\(",
                rf"\bnew\s+{re.escape(class_name)}\s*\([^)]*\)\s*\.\s*{re.escape(node.name)}\s*\(",
            ] if class_name else [rf"\b{re.escape(node.name)}\s*\("]
            if any(re.search(pattern, body) for pattern in patterns):
                if node.node_id not in seen:
                    called.append(node)
                    seen.add(node.node_id)
        return called

    def _java_assertions(self, body: str) -> list[str]:
        return [match.group(0).strip() for match in re.finditer(r"\bassert[A-Za-z0-9_]*\s*\([^;]*\);", body, flags=re.DOTALL)]

    def _java_assertion_specificity(self, code: str) -> float:
        if any(name in code for name in ["assertEquals", "assertTrue", "assertFalse", "assertNull", "assertNotNull"]):
            return 1.0
        if "assertThrows" in code:
            return 0.75
        if "assertDoesNotThrow" in code:
            return 0.35
        return 0.2

    def _function_lookup(self) -> dict[tuple[str | None, str], Any]:
        lookup: dict[tuple[str | None, str], Any] = {}
        for node in self.graph.nodes.values():
            if node.node_type not in {"Function", "Async", "Method", "Constructor"}:
                continue
            class_name = self._node_class_name(node)
            lookup[(class_name, node.name)] = node
            lookup.setdefault((None, node.name), node)
        return lookup

    def _node_class_name(self, node: Any) -> str | None:
        scope = str(node.metadata.get("scope") or "")
        return scope if scope and "::" not in scope else None

    def _runtime_state_key(self, test_name: str, function_name: str, class_name: str | None = None) -> str:
        qualified = f"{class_name}.{function_name}" if class_name else function_name
        return f"{test_name}::{qualified}"

    def _called_target_functions(self, test_function: ast.FunctionDef) -> set[str]:
        return {str(info["function"]) for info in self._called_target_call_infos(test_function)}

    def _called_target_call_infos(self, test_function: ast.FunctionDef) -> list[dict[str, str | None]]:
        calls: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str]] = set()
        receiver_call_ids = self._receiver_call_ids(test_function)
        for child in ast.walk(test_function):
            if id(child) in receiver_call_ids:
                continue
            info = self._target_call_info(child) if isinstance(child, ast.Call) else None
            if not info:
                continue
            function_name = str(info["function"])
            class_name = info.get("class_name")
            key = (class_name, function_name)
            if key not in seen:
                calls.append({"function": function_name, "class_name": class_name})
                seen.add(key)
        return calls

    def _receiver_call_ids(self, test_function: ast.FunctionDef) -> set[int]:
        return {
            id(child.func.value)
            for child in ast.walk(test_function)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Call)
        }

    def _is_target_receiver(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Name) and node.id == "target":
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_load_target":
            return True
        return False

    def _target_call_info(self, node: ast.Call) -> dict[str, str | None] | None:
        if not isinstance(node.func, ast.Attribute):
            return None
        receiver = node.func.value
        if self._is_target_receiver(receiver):
            return {"function": node.func.attr, "class_name": None}
        if isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute):
            if self._is_target_receiver(receiver.func.value):
                return {"function": node.func.attr, "class_name": receiver.func.attr}
        return None

    def _assert_observed_functions(self, test_function: ast.FunctionDef, assertion_node: ast.AST) -> set[str]:
        return {str(info["function"]) for info in self._assert_observed_call_infos(test_function, assertion_node)}

    def _assert_observed_call_infos(self, test_function: ast.FunctionDef, assertion_node: ast.AST) -> list[dict[str, str | None]]:
        variable_to_function = self._assigned_target_call_variables_info(test_function)
        receiver_call_ids = self._receiver_call_ids(test_function)
        names = {node.id for node in ast.walk(assertion_node) if isinstance(node, ast.Name)}
        observed: list[dict[str, str | None]] = []
        seen: set[tuple[str | None, str]] = set()

        def add(info: dict[str, str | None] | None) -> None:
            if not info:
                return
            function_name = str(info["function"])
            class_name = info.get("class_name")
            key = (class_name, function_name)
            if key not in seen:
                observed.append({"function": function_name, "class_name": class_name})
                seen.add(key)

        for name in names:
            add(variable_to_function.get(name))
        for child in ast.walk(assertion_node):
            if id(child) in receiver_call_ids:
                continue
            info = self._target_call_info(child) if isinstance(child, ast.Call) else None
            if info and info["function"] != "raises":
                add(info)
        if self._is_pytest_raises(assertion_node) and isinstance(assertion_node, ast.With):
            for body_item in assertion_node.body:
                for child in ast.walk(body_item):
                    if id(child) in receiver_call_ids:
                        continue
                    add(self._target_call_info(child) if isinstance(child, ast.Call) else None)
        return observed

    def _assigned_target_call_variables(self, test_function: ast.FunctionDef) -> dict[str, str]:
        return {name: str(info["function"]) for name, info in self._assigned_target_call_variables_info(test_function).items()}

    def _assigned_target_call_variables_info(self, test_function: ast.FunctionDef) -> dict[str, dict[str, str | None]]:
        mapping: dict[str, dict[str, str | None]] = {}
        receiver_call_ids = self._receiver_call_ids(test_function)
        for child in ast.walk(test_function):
            if not isinstance(child, ast.Assign):
                continue
            if isinstance(child.value, ast.Call) and id(child.value) in receiver_call_ids:
                continue
            info = self._target_call_info(child.value) if isinstance(child.value, ast.Call) else None
            if not info:
                continue
            for target in child.targets:
                if isinstance(target, ast.Name):
                    mapping[target.id] = {"function": str(info["function"]), "class_name": info.get("class_name")}
        return mapping

    def _target_call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            info = self._target_call_info(node)
            if info:
                return str(info["function"])
            for child in ast.walk(node):
                if child is node or not isinstance(child, ast.Call):
                    continue
                info = self._target_call_info(child)
                if info:
                    return str(info["function"])
        return None

    def _is_pytest_raises(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.With):
            return False
        for item in node.items:
            expr = item.context_expr
            if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                if expr.func.attr == "raises":
                    return True
        return False

    def _assertion_specificity(self, node: ast.AST) -> float:
        if self._is_pytest_raises(node):
            return 1.0
        if not isinstance(node, ast.Assert):
            return 0.0
        code = ast.unparse(node.test)
        if code in {"result is not None or result is None", "True"}:
            return 0.0
        if isinstance(node.test, ast.Compare):
            return 1.0
        if isinstance(node.test, ast.Call):
            return 0.6
        if isinstance(node.test, ast.BoolOp):
            return 0.5
        return 0.3

    def _safe_value_summary(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value)
        lowered = text.lower()
        if any(secret in lowered for secret in ["password", "token", "secret", "apikey", "api_key"]):
            return "<redacted>"
        return text[:120]

    def finalize_execution_update(self, reason: str = "execution update") -> None:
        self.graph.bump_version(reason)
