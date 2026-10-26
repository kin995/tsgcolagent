from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

import batch_generate_tests as batch
from agents.execution_agent import ExecutionAgent, ExecutionReport
from agents.generation_agent import TestGenerationAgent
from agents.test_plan import TestCasePlan as CasePlan
from agents.test_plan import TestOracle as Oracle
from agents.test_plan import TestPlan as Plan
from graph.state_graph import StateFlowGraph
from main import Thresholds
from memory.shared_graph import SharedGraphMemory
from monitoring.dashboard_data import build_snapshot
from monitoring.server import DashboardHandler
from monitoring.server import _expanded_experiments
from executor.test_quality import analyze_test_usability


def test_humaneval_java_uses_one_source_per_bundled_workspace_task(tmp_path):
    source_dir = tmp_path / "HumanEvalJava"
    source_dir.mkdir()
    source = source_dir / "ADD.java"
    source.write_text("package humaneval.correct; class ADD {}\n", encoding="utf-8")
    agent = object.__new__(ExecutionAgent)

    source_root, target_package = agent._infer_java_dataset_root(source, "humaneval.correct")

    assert source_root == source.resolve()
    assert target_package == "humaneval.correct"


def test_reliable_severe_mutation_weakness_cannot_be_b_grade():
    report = ExecutionReport(
        pytest_passed=True,
        coverage_percent=0.95,
        mutation_score=0.0,
        sfc=0.85,
        ae=0.9,
        sfq=0.74,
        test_path="test_target.py",
        line_coverage=0.95,
        branch_coverage=0.9,
        combined_coverage=0.93,
        collected_count=3,
        normal_behavior_tests=3,
        assertion_count=3,
        effective_assertions=3,
        effective_mutants=20,
        mutation_reliability="normal",
        mutation_complete=True,
        mutation_valid=True,
    )

    assert Thresholds().classify(report) == "valid_usable"


def test_java_exception_only_suite_is_not_normal_behavior(tmp_path):
    test_path = tmp_path / "TargetTest.java"
    test_path.write_text(
        """
        import org.junit.jupiter.api.Test;
        import static org.junit.jupiter.api.Assertions.*;
        class TargetTest {
          @Test void rejectsInvalid() { assertThrows(IllegalArgumentException.class, () -> fail()); }
        }
        """,
        encoding="utf-8",
    )
    agent = object.__new__(ExecutionAgent)

    profile = agent._java_assertion_profile(test_path)

    assert profile["exception"] == 1
    assert profile["normal"] == 0


def test_java_parameterized_tests_are_counted(tmp_path):
    test_path = tmp_path / "TargetTest.java"
    test_path.write_text("class TargetTest { @ParameterizedTest void works() {} @RepeatedTest(2) void repeats() {} }", encoding="utf-8")
    agent = object.__new__(ExecutionAgent)

    assert agent._java_test_count(test_path) == 2


def test_pytest_class_methods_are_included_in_quality_analysis(tmp_path):
    test_path = tmp_path / "test_target.py"
    test_path.write_text(
        """
class TestTarget:
    def test_value(self, target):
        assert target.solve(2) == 4
""",
        encoding="utf-8",
    )

    usability = analyze_test_usability(test_path)

    assert usability.test_count == 1
    assert usability.normal_behavior_tests == 1
    assert usability.assertion_count == 1


def test_llm_plan_oracles_are_grounded_without_losing_diverse_inputs(tmp_path, monkeypatch):
    source = tmp_path / "target.py"
    source.write_text("def solve(value):\n    return value * 2\n", encoding="utf-8")
    graph = StateFlowGraph(source_path=str(source))
    agent = TestGenerationAgent(SharedGraphMemory(tmp_path / "graph.json"))
    plan = Plan(
        cases=[
            CasePlan(f"case_{value}", "solve", [value], Oracle("equals", value=999), confidence=0.5)
            for value in (1, 2, 5)
        ]
    )
    function = {"name": "solve", "args": ["value"], "node_type": "Function", "class_name": None}

    monkeypatch.setattr(
        agent,
        "_observe_call",
        lambda _name, values, _function, _graph: {"status": "ok", "value": repr(ast.literal_eval(values[0]) * 2)},
    )
    grounded = agent._ground_and_strengthen_llm_plan(graph, plan, [function])

    assert [case.args for case in grounded.cases] == [[1], [2], [5]]
    assert [case.oracle.value for case in grounded.cases] == [2, 4, 10]


def test_nested_collection_annotations_generate_real_matrix_inputs(tmp_path):
    source = tmp_path / "target.py"
    source.write_text("class Solution:\n    def solve(self, heights):\n        return heights\n", encoding="utf-8")
    graph = StateFlowGraph(source_path=str(source))
    agent = TestGenerationAgent(SharedGraphMemory(tmp_path / "graph.json"))
    function = {"name": "solve", "class_name": "Solution", "args": ["heights"]}

    value = agent._value_for_arg("heights", "high", {"heights": "List[List[int]]"}, graph, function)

    parsed = ast.literal_eval(value)
    assert len(parsed) >= 2
    assert all(isinstance(row, list) for row in parsed)


def test_grounding_keeps_the_only_possible_zero_argument_case(tmp_path, monkeypatch):
    source = tmp_path / "target.py"
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    graph = StateFlowGraph(source_path=str(source))
    agent = TestGenerationAgent(SharedGraphMemory(tmp_path / "graph.json"))
    plan = Plan(cases=[CasePlan("answer_case", "answer", [], Oracle("equals", value=0))])
    function = {"name": "answer", "args": []}
    monkeypatch.setattr(agent, "_observe_call", lambda *_args: {"status": "ok", "value": "42"})
    monkeypatch.setattr(agent, "_deterministic_plan", lambda _graph: Plan(cases=[]))

    grounded = agent._ground_and_strengthen_llm_plan(graph, plan, [function])

    assert len(grounded.cases) == 1
    assert grounded.cases[0].oracle.value == 42


def test_task_runtime_accumulates_only_real_run_intervals(tmp_path, monkeypatch):
    source = tmp_path / "ADD.java"
    source.write_text("class ADD {}\n", encoding="utf-8")
    case_dir = tmp_path / "run"
    timestamps = iter(
        [
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:05+00:00",
            "2026-01-02T00:00:00+00:00",
            "2026-01-02T00:00:07+00:00",
        ]
    )
    monkeypatch.setattr(batch, "_utc_now", lambda: next(timestamps))

    first = batch._begin_task_timing(case_dir, source, experiment_round=1, strategy="initial_generation")
    batch._finish_task_timing(case_dir, first, "quality_acceptable")
    second = batch._begin_task_timing(case_dir, source, experiment_round=3, strategy="optimize_existing")
    timing = batch._finish_task_timing(case_dir, second, "quality_high")

    assert timing["duration_seconds"] == 12.0
    assert timing["attempt_count"] == 2


def test_batch_averages_exclude_semantically_invalid_d_tasks(tmp_path):
    valid = batch.BatchItemResult(
        source="valid.py",
        output_dir=str(tmp_path / "valid"),
        status="quality_acceptable",
        final_grade="B",
        final_report={"pytest_passed": True, "line_coverage": 0.9, "mutation_score": 0.8},
    )
    invalid = batch.BatchItemResult(
        source="invalid.py",
        output_dir=str(tmp_path / "invalid"),
        status="execution_invalid",
        final_grade="D",
        final_report={"pytest_passed": True, "line_coverage": 0.1, "mutation_score": 0.0},
    )

    overview = batch._batch_quality_overview([valid, invalid])

    assert overview["executed_tasks"] == 1
    assert overview["all_executed_tasks"] == 2
    assert overview["average_mutation_score"] == 0.8


def test_master_experiment_exposes_each_dataset_as_a_selectable_view(tmp_path):
    runs = tmp_path / "runs"
    for name, language in [("TestEval", "python"), ("HumanEvalJava", "java")]:
        dataset_dir = runs / name
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "batch_summary.json").write_text(json.dumps({"language": language, "total": 1}), encoding="utf-8")
    (runs / "all_datasets_summary.json").write_text(
        json.dumps(
            {
                "results": [
                    {"dataset": {"name": "TestEval", "language": "python"}, "status": "completed", "summary": {"total": 1}},
                    {"dataset": {"name": "HumanEvalJava", "language": "java"}, "status": "completed", "summary": {"total": 1}},
                ]
            }
        ),
        encoding="utf-8",
    )

    visible, masters = _expanded_experiments([{"id": "master", "output_dir": str(runs), "label": "temporary label", "execution_status": "completed"}])

    assert [item["label"] for item in visible] == ["TestEval", "HumanEvalJava"]
    assert [item["language"] for item in visible] == ["python", "java"]
    assert [item["scope"] for item in visible] == ["file_function_class", "file_class"]
    assert masters[0]["label"] == "全部数据集实验"


def test_completed_task_graph_is_not_mistaken_for_live_execution(tmp_path):
    task_dir = tmp_path / "task_001"
    task_dir.mkdir()
    summary_path = task_dir / "stateflow_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": str(tmp_path / "task_001.py"),
                "quality_status": "quality_acceptable",
                "final_grade": "B",
                "final_report": {"pytest_passed": True, "collected_count": 2, "line_coverage": 1.0},
            }
        ),
        encoding="utf-8",
    )
    graph_path = task_dir / "state_flow_graph.json"
    graph_path.write_text(json.dumps({"metadata": {"last_report": {"pytest_passed": True, "collected_count": 2}}}), encoding="utf-8")
    future = time.time() + 2
    os.utime(graph_path, (future, future))
    (tmp_path / "batch_summary.json").write_text(json.dumps({"total": 1}), encoding="utf-8")
    (tmp_path / "dataset_completion.json").write_text(json.dumps({"skip_ready": True}), encoding="utf-8")

    snapshot = build_snapshot({"id": "completed", "output_dir": str(tmp_path), "execution_status": "completed"}, [])

    assert snapshot["tasks"][0]["status"] != "running"
    assert snapshot["overview"]["active_task_count"] == 0


def test_completed_snapshot_cache_refreshes_after_result_file_changes(tmp_path, monkeypatch):
    import monitoring.server as server

    batch_path = tmp_path / "batch_summary.json"
    batch_path.write_text("{}", encoding="utf-8")
    calls: list[int] = []

    def fake_snapshot(_record, _experiments):
        calls.append(len(calls) + 1)
        return {"version": calls[-1]}

    monkeypatch.setattr(server, "build_snapshot", fake_snapshot)
    server.SNAPSHOT_CACHE.clear()
    handler = object.__new__(DashboardHandler)
    record = {"id": "cache-test", "output_dir": str(tmp_path), "execution_status": "completed"}

    assert handler._record_snapshot(record, [record])["version"] == 1
    assert handler._record_snapshot(record, [record])["version"] == 1
    batch_path.write_text('{"changed": true}', encoding="utf-8")
    assert handler._record_snapshot(record, [record])["version"] == 2
    assert calls == [1, 2]
