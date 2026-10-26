from __future__ import annotations

from pathlib import Path

from agents.execution_agent import ExecutionAgent, ExecutionReport
from executor.build_tools import maven_local_repo_arg
from executor.evaluation_cache import EvaluationCache
from executor.java_toolchain import JavaToolchain, _java_evaluation_lock
from executor.mutation_runner import MutantStatus, MutationRunner
from graph.structured_graph_builder import StructuredGraphBuilder
from memory.shared_graph import SharedGraphMemory
from main import MultiAgentUnitTestSystem, Thresholds
from orchestration.test_workflow import LangGraphTestWorkflow


def test_humanevaljava_uses_distinct_reusable_workspaces_and_shared_dependencies(tmp_path):
    dataset = tmp_path / "HumanEvalJava"
    dataset.mkdir()
    source_a = dataset / "A.java"
    source_b = dataset / "B.java"
    source_a.write_text("package humaneval.correct; public class A { public static int value() { return 1; } }", encoding="utf-8")
    source_b.write_text("package humaneval.correct; public class B { public static int value() { return 2; } }", encoding="utf-8")
    memory_a = SharedGraphMemory(tmp_path / "run_a" / "state_flow_graph.json")
    memory_b = SharedGraphMemory(tmp_path / "run_b" / "state_flow_graph.json")
    agent_a = ExecutionAgent(memory_a, isolate_humaneval_java=True)
    agent_b = ExecutionAgent(memory_b, isolate_humaneval_java=True)

    workspace_a = agent_a._java_project_root(source_a)
    workspace_b = agent_b._java_project_root(source_b)
    prepared = agent_a._prepare_bundled_java_workspace(
        workspace_a,
        source_a,
        {"package_or_module": "humaneval.correct"},
    )
    sentinel = workspace_a / "src" / "main" / "java" / "keep.incremental"
    sentinel.write_text("keep", encoding="utf-8")
    agent_a._prepare_bundled_java_workspace(workspace_a, source_a, {"package_or_module": "humaneval.correct"})

    assert workspace_a != workspace_b
    assert (workspace_a / ".stateflow_humanevaljava_workspace").is_file()
    assert (workspace_b / ".stateflow_humanevaljava_workspace").is_file()
    assert prepared["workspace_mode"] == "isolated_humanevaljava"
    assert sentinel.is_file()
    assert str(workspace_a / ".m2" / "repository") not in maven_local_repo_arg(workspace_a)
    assert "java_workspace" in maven_local_repo_arg(workspace_a)
    assert _java_evaluation_lock(workspace_a) is not _java_evaluation_lock(workspace_b)


def test_python_screening_runs_at_most_eight_prioritized_mutants(tmp_path, monkeypatch):
    source = tmp_path / "target.py"
    source.write_text(
        "\n".join([f"value_{index} = {index} + {index + 1}" for index in range(30)]) + "\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "test_target.py"
    test_path.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    runner = MutationRunner(max_mutants=20)
    calls = {"count": 0}

    def fake_pytest(*args, **kwargs):
        calls["count"] += 1
        return MutantStatus.SURVIVED if calls["count"] == 1 else MutantStatus.KILLED

    monkeypatch.setattr(runner, "_run_pytest", fake_pytest)
    result = runner.run(source, test_path, cwd=tmp_path, evaluation_profile="screening", priority_lines={25})

    assert result.evaluation_profile == "screening"
    assert result.profile_limit == 8
    assert result.attempted_mutants == 8
    assert 25 in result.selected_candidate_lines
    assert calls["count"] == 9


def test_full_python_mutation_keeps_twenty_mutant_budget(tmp_path, monkeypatch):
    source = tmp_path / "target.py"
    source.write_text("\n".join([f"value_{index} = {index} + 1" for index in range(30)]) + "\n", encoding="utf-8")
    test_path = tmp_path / "test_target.py"
    test_path.write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    runner = MutationRunner(max_mutants=20)
    calls = {"count": 0}

    def fake_pytest(*args, **kwargs):
        calls["count"] += 1
        return MutantStatus.SURVIVED if calls["count"] == 1 else MutantStatus.KILLED

    monkeypatch.setattr(runner, "_run_pytest", fake_pytest)
    result = runner.run(source, test_path, cwd=tmp_path, evaluation_profile="full")

    assert result.evaluation_profile == "full"
    assert result.attempted_mutants == 20
    assert calls["count"] == 21


def test_invalid_python_baseline_skips_coverage_and_mutation(tmp_path, monkeypatch):
    source = tmp_path / "target.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    test_path = tmp_path / "test_target.py"
    test_path.write_text("def test_value():\n    assert False\n", encoding="utf-8")
    memory = SharedGraphMemory(tmp_path / "state_flow_graph.json")
    memory.set(StructuredGraphBuilder.from_source(source).build())
    agent = ExecutionAgent(memory, retry_rounds=0)
    monkeypatch.setattr(agent.coverage_runner, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("coverage should be skipped")))
    monkeypatch.setattr(agent.mutation_runner, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("mutation should be skipped")))

    report = agent.run(test_path, cwd=tmp_path, evaluation_profile="screening")

    assert not report.pytest_passed
    assert report.mutation_score is None
    assert memory.load().metadata["last_mutation"]["skipped_reason"] == "baseline_not_runnable"


def test_java_combines_test_and_coverage_and_defers_pit(monkeypatch, tmp_path):
    (tmp_path / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    toolchain = JavaToolchain()
    calls = {"jacoco": 0, "pit": 0, "junit": 0}

    def jacoco_run(*args, **kwargs):
        calls["jacoco"] += 1
        return {"returncode": 0, "stdout": "", "stderr": "", "valid": True, "line_coverage": 1.0, "branch_coverage": 1.0, "method_coverage": 1.0}

    def pit_run(*args, **kwargs):
        calls["pit"] += 1
        return {"valid": True, "mutation_score": 1.0, "effective_mutants": 4, "score_reliability": "normal"}

    monkeypatch.setattr(toolchain.jacoco, "run", jacoco_run)
    monkeypatch.setattr(toolchain.pit, "run", pit_run)
    monkeypatch.setattr(toolchain.junit, "run", lambda *args, **kwargs: calls.__setitem__("junit", calls["junit"] + 1))

    screening = toolchain.run(tmp_path, test_selector="TargetTest", target_class="demo.Target", evaluation_profile="screening")
    full = toolchain.run(tmp_path, test_selector="TargetTest", target_class="demo.Target", evaluation_profile="full")

    assert screening["passed"] is True
    assert screening["pit"]["skipped_reason"] == "deferred_until_final_candidate"
    assert full["mutation_score"] == 1.0
    assert calls == {"jacoco": 2, "pit": 1, "junit": 0}


def test_evaluation_cache_key_changes_with_source_test_or_dependencies(tmp_path):
    source = tmp_path / "target.py"
    test_path = tmp_path / "test_target.py"
    requirements = tmp_path / "requirements.txt"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    test_path.write_text("def test_value():\n    assert True\n", encoding="utf-8")
    requirements.write_text("pytest\n", encoding="utf-8")
    cache = EvaluationCache(tmp_path / "cache.json")
    first = cache.key(language="python", stage="coverage", profile="shared", files=[source, test_path, requirements])
    cache.put("coverage", first, {"line_coverage": 1.0})

    assert cache.get("coverage", first) == {"line_coverage": 1.0}
    test_path.write_text("def test_value():\n    assert 1 == 1\n", encoding="utf-8")
    second = cache.key(language="python", stage="coverage", profile="shared", files=[source, test_path, requirements])
    assert second != first
    assert cache.get("coverage", second) is None


def test_java_final_evaluation_reuses_screening_validation(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    source = project / "Target.java"
    source.write_text("class Target { int value() { return 1; } }\n", encoding="utf-8")
    test_path = tmp_path / "TargetTest.java"
    test_path.write_text(
        "import org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.assertEquals;\n"
        "class TargetTest { @Test void value() { assertEquals(1, new Target().value()); } }\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "state_flow_graph.json")
    memory.set(StructuredGraphBuilder.from_source(source).build())
    agent = ExecutionAgent(memory)

    class FakeJavaToolchain:
        def __init__(self):
            self.validation_calls = 0
            self.pit_calls = 0

        def _build_system(self, project_root):
            return "maven"

        def run(self, project_root, test_selector=None, target_class=None, timeout=120, evaluation_profile="full"):
            self.validation_calls += 1
            return {
                "language": "java",
                "build_system": "maven",
                "passed": True,
                "junit": {"passed": True},
                "line_coverage": 1.0,
                "branch_coverage": None,
                "method_coverage": 1.0,
                "mutation_score": None,
                "jacoco": {"valid": True, "covered_lines": [1]},
                "pit": {"valid": False, "mutation_score": None, "skipped": True},
                "evaluation_profile": evaluation_profile,
            }

        def run_pit_only(self, project_root, test_selector=None, target_class=None, timeout=180):
            self.pit_calls += 1
            return {"valid": True, "mutation_score": 1.0, "effective_mutants": 3, "score_reliability": "normal", "killed": 3, "survived": 0, "total": 3}

    fake = FakeJavaToolchain()
    agent.java_toolchain = fake
    screening = agent.run(test_path, evaluation_profile="screening")
    full = agent.run(test_path, evaluation_profile="full")

    assert screening.evaluation_profile == "screening"
    assert full.evaluation_profile == "full"
    assert full.mutation_score == 1.0
    assert fake.validation_calls == 1
    assert fake.pit_calls == 1


def test_screening_report_must_be_finalized_before_formal_result(tmp_path):
    source = tmp_path / "target.py"
    source.write_text("def value():\n    return 1\n", encoding="utf-8")
    system = MultiAgentUnitTestSystem(source, tmp_path / "run", max_iterations=1)
    workflow = LangGraphTestWorkflow(system)
    report = ExecutionReport(
        pytest_passed=True,
        coverage_percent=1.0,
        mutation_score=1.0,
        sfc=1.0,
        ae=1.0,
        sfq=1.0,
        test_path=str(tmp_path / "test_target.py"),
        line_coverage=1.0,
        collected_count=1,
        normal_behavior_tests=1,
        assertion_count=1,
        effective_assertions=1,
        evaluation_profile="screening",
        mutation_complete=False,
        mutation_reliability="screening",
        effective_mutants=8,
    )
    try:
        assert workflow._screening_result_needs_final(report.to_dict()) is True
        assert Thresholds().mutation_applicable(report) is False
        report.evaluation_profile = "full"
        report.mutation_complete = True
        report.mutation_reliability = "normal"
        assert workflow._screening_result_needs_final(report.to_dict()) is False
        assert Thresholds().mutation_applicable(report) is True
    finally:
        workflow.close()
