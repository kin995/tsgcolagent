import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrics.measurement import CURRENT, Measurement, java_execution_count, measured_python_run, reconcile_round
from metrics.paper_metrics import aggregate_metrics, direct_metrics, tir_statistics


def report(**changes):
    return {"pytest_passed": True, "collected_count": 2, "coverage_valid": True,
            "mutation_valid": True, "ae": .725, "line_coverage": .8,
            "branch_coverage": .5, "mutation_score": .75,
            "assertion_count": 3, "effective_assertions": 2,
            "mutation_evidence": {"killed": 2, "survived": 2, "timeout": 4}, **changes}


def action(before_passed=False, after_passed=True, **changes):
    return {"task_id": "task", "session_id": "session", "round": 1, "step": 1,
            "preregistered": True, "acted": True, "completed": True,
            "goals": [{"id": "execution", "predicate": "executable", "category": "recovery"}],
            "before": {"report": report(pytest_passed=before_passed)},
            "retained_after": {"report": report(pytest_passed=after_passed)}, **changes}


def test_preserves_runtime_ae_ms_and_missing_evidence():
    values = direct_metrics(report(), 0)
    assert values["ae"] == .725  # Not the alternative 2/3 formula.
    assert values["mutation_score"] == .75  # Not the alternative killed/(killed+survived).
    assert values["duration_seconds"] == 0
    assert direct_metrics(report(coverage_valid=False))["line_coverage"] is None
    assert direct_metrics(report(mutation_valid=False))["mutation_score"] is None


def test_tir_dedup_exclusions_and_retained_not_candidate():
    measured = action()
    rejected = action(after_passed=False, step=2, candidate_after={"report": report()})
    unknown = action(step=3, retained_after={})
    values = tir_statistics([measured, measured, rejected, unknown])
    assert (values["tir"], values["tir_comparable"], values["tir_excluded"]) == (.5, 2, 1)
    assert tir_statistics([unknown])["tir_status"] == "incomplete_evidence"
    assert tir_statistics([])["tir_status"] == "no_comparable_targets"
    with pytest.raises(ValueError):
        tir_statistics([measured, action(after_passed=False)])


def test_metric_target_is_registered_before_execution():
    goal = {"id": "metric:ae", "predicate": "metric_increase", "category": "observation", "field": "ae", "baseline": .5}
    record = action(goals=[goal], before={"report": report(ae=.5)}, retained_after={"report": report(ae=.6)})
    assert tir_statistics([record])["tir"] == 1
    record["goals"][0]["baseline"] = None
    assert tir_statistics([record])["tir"] is None


def test_aggregate_uses_all_tasks_and_summed_tir_denominators():
    tasks = [{"metrics": {**direct_metrics(report()), "exec_count": 2, "exec_count_status": "complete", "tir_comparable": 1, "tir_improved": 1}},
             {"metrics": {**direct_metrics(report(pytest_passed=False)), "exec_count": 4, "exec_count_status": "complete", "tir_comparable": 3, "tir_improved": 0}}]
    values = aggregate_metrics(tasks)
    assert values["pass_rate"] == .5 and values["avg_exec"] == 3
    assert values["tir"] == .25
    partial = aggregate_metrics(tasks, 3)
    assert partial["avg_exec"] == 3
    assert partial["avg_exec_status"] == "partial_evidence"
    assert partial["exec_complete_tasks"] == 2
    tasks[1]["metrics"]["exec_count_status"] = "incomplete_evidence"
    partial = aggregate_metrics(tasks)
    assert partial["avg_exec"] == 2
    assert partial["avg_exec_status"] == "partial_evidence"
    assert partial["exec_complete_tasks"] == 1


def test_avg_exec_uses_only_complete_execution_counts():
    tasks = [
        {"metrics": {"exec_count": 0, "exec_count_status": "complete"}},
        {"metrics": {"exec_count": 4, "exec_count_status": "complete"}},
        {"metrics": {"exec_count": 9, "exec_count_status": "incomplete_evidence"}},
    ]
    values = aggregate_metrics(tasks)
    assert values["avg_exec"] == 2
    assert values["exec_complete_tasks"] == 2
    assert values["avg_exec_status"] == "partial_evidence"
    assert aggregate_metrics(tasks[2:])["avg_exec"] is None


def test_real_pytest_probe_through_langgraph_and_collect_only(tmp_path):
    from langgraph.graph import StateGraph, START, END
    from typing import TypedDict
    test = tmp_path / "test_sample.py"
    test.write_text("def test_a():\n    assert 1 == 1\ndef test_b():\n    assert 2 == 2\n", encoding="utf-8")
    measurement = Measurement(tmp_path / "run", test, 1)
    token = CURRENT.set(measurement)
    class State(TypedDict):
        done: bool
    def node(state):
        result = measured_python_run([sys.executable, "-m", "pytest", str(test), "-q"], kind="test", cwd=tmp_path, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr + result.stdout
        return {"done": True}
    try:
        graph = StateGraph(State)
        graph.add_node("probe", node)
        graph.add_edge(START, "probe")
        graph.add_edge("probe", END)
        assert graph.compile().invoke({"done": False})["done"]
        measured_python_run([sys.executable, "-m", "pytest", str(test), "-q", "--collect-only"], kind="test", cwd=tmp_path, text=True, capture_output=True, timeout=30)
        values = measurement.summary(report(), completed=True)
        assert values["exec_count"] == 1  # Two cases in one tool session.
        assert values["exec_count_status"] == "complete"
    finally:
        CURRENT.reset(token)


def test_java_combined_session_and_pit_not_mutant_count(tmp_path):
    xml = tmp_path / "target/surefire-reports/TEST-A.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text('<testsuite><testcase name="a"/><testcase name="b"/></testsuite>', encoding="utf-8")
    assert java_execution_count(tmp_path, {}, "", "", 0) == (1, True)
    assert java_execution_count(tmp_path, {}, "Ran 120 tests (4 tests per mutation)", "", 0, True) == (1, True)
    assert java_execution_count(tmp_path, {}, "tool missing", "", 1, True) == (0, False)


def test_knowledge_counts_and_scope_average(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x=1", encoding="utf-8")
    m = Measurement(tmp_path, source, 1)
    graph = SimpleNamespace(nodes={}, metadata={"last_knowledge_context": {"selected_ids": ["a", "a", "b"]}})
    m.begin_action({"selected_agent": "TestValidityAgent"}, graph, initial=True)
    m.append({"type": "knowledge_record", "record_id": "new"})
    m.append({"type": "knowledge_record", "record_id": "new"})
    m.finish_action({"report": report()})
    values = m.summary(report(), completed=True)
    assert values["knowledge_occurrence"] == 1
    assert values["knowledge_use_frequency"] == 2
    aggregate = aggregate_metrics([{"metrics": values}, {"metrics": values}])
    assert aggregate["knowledge_occurrence"] == 1
    assert aggregate["knowledge_use_frequency"] == 4


def test_round_rollback_keeps_cost_but_replaces_target_outcome(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x = 1", encoding="utf-8")
    m = Measurement(tmp_path, source, 3)
    m.last_evidence = {"report": report(ae=.5)}
    m.begin_action({"selected_agent": "AssertionAgent"}, SimpleNamespace(nodes={}), source)
    m.observe(report(ae=.8), SimpleNamespace(metadata={}), source)
    m.append({"type": "execution_start", "call_id": "one"})
    m.append({"type": "execution_end", "call_id": "one", "count": 1, "complete": True})
    assert m.summary(report(ae=.8), completed=True)["tir"] == 1
    reconcile_round(tmp_path, report(ae=.5), 3, False, 12)
    result = json.loads((tmp_path / "paper_metrics/summary.json").read_text(encoding="utf-8"))
    assert result["exec_count"] == 1 and result["tir"] == 0


def test_legacy_prefix_does_not_invent_a_complete_total(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("x=1", encoding="utf-8")
    (tmp_path / "stateflow_summary.json").write_text("{}", encoding="utf-8")
    m = Measurement(tmp_path, source, 2)
    result = m.summary(report(), completed=True)
    assert result["legacy_prefix"] and result["avg_exec"] is None


def test_workflow_persists_measurements_without_a_model(tmp_path):
    from main import MultiAgentUnitTestSystem
    from llm_config import LLMConfig
    source = tmp_path / "source.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    system = MultiAgentUnitTestSystem(source, tmp_path / "output", max_iterations=0,
                                     llm_config=LLMConfig(enabled=False), resume=False)
    result = system.run()
    assert result["paper_metrics"]["measurement_version"] == 1
    assert result["paper_metrics"]["exec_count_status"] == "complete"
    stored = json.loads((tmp_path / "output/stateflow_summary.json").read_text(encoding="utf-8"))
    assert stored["paper_metrics"]["exec_count"] == result["paper_metrics"]["exec_count"]
    from monitoring.dashboard_data import _task_from_summary
    task = _task_from_summary(tmp_path / "output/stateflow_summary.json", stored)
    assert task["metrics"]["avg_exec"] == result["paper_metrics"]["avg_exec"]
