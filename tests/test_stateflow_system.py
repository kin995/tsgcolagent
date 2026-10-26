from __future__ import annotations

import ast
import json
from pathlib import Path

from agents.analysis_agent import SourceAnalysisAgent
from agents.evaluation_agent import EvaluationAgent
from agents.execution_agent import ExecutionAgent, ExecutionReport
from agents.generation_agent import TestGenerationAgent
from agents.planning_agent import PlanningAgent
from agents.repair_agent import RepairOptimizationAgent
from graph.adaptive_scheduler import GraphDrivenAdaptiveScheduler
from graph.experience_memory import AgentExperienceMemory
from graph.graph_query import GraphQuery
from graph.repair_planner import RootCauseRepairPlanner
from llm_config import LLMConfig
from llm_client import OpenAICompatibleLLM
from main import MultiAgentUnitTestSystem, Thresholds
from memory.shared_graph import SharedGraphMemory
from executor.coverage_runner import CoverageRunner
from executor.mutation_runner import MutationResult
from executor.failure_classifier import PytestFailureClassifier
from executor.pytest_runner import PytestResult
from executor.runtime_observer import RuntimeObserver
from executor.test_quality import analyze_test_usability
from agents.test_code_normalizer import normalize_generated_test_code
from agents.candidate_validator import CandidateTestValidator
from agents.test_plan import (
    TestCasePlan as PlanCase,
    TestOracle as PlanOracle,
    TestPlan as GeneratedPlan,
    TestPlanRenderer,
)
from agents.test_repair_operations import RepairOperation
from graph.patch_search import PatchSearchRepair
from graph.state_graph import StateFlowGraph
from graph.graph_updater import GraphUpdater
from metrics.mutation_metric import MutationScoreMetric
from metrics.assertion_metric import AssertionEffectivenessMetric
from metrics.state_quality import StateFlowQualityMetric
from visualization.graph_visualizer import StateFlowGraphVisualizer


class _FakeLLM:
    def __init__(self):
        from types import SimpleNamespace

        self.config = SimpleNamespace(model="fake-llm")
        self.calls: list[str] = []

    def enabled(self) -> bool:
        return True

    def chat(self, messages):
        content = messages[-1]["content"]
        self.calls.append(content)
        if "Test Generation Agent" in content:
            return (
                '{"cases": ['
                '{"id": "llm_generated_classify_score", "function": "classify_score", '
                '"args": [90], "oracle": {"kind": "equals", "value": "excellent"}, '
                '"evidence_ids": [], "rationale": "high score boundary", "confidence": 0.9}'
                ']}'
            )
        if "Evaluation Agent" in content:
            return "[]"
        if "Repair Optimization Agent" in content:
            return (
                '{"operations": ['
                '{"op": "add_test", "new_code": "def test_llm_repair_more_paths(target):\\n'
                '    assert target.classify_score(60) == \\"pass\\"\\n'
                '    assert target.classify_score(1) == \\"fail\\"\\n'
                '    assert target.sum_positive([1, -1, 2]) == 3\\n"}'
                ']}'
            )
        return '{"ok": true}'


def test_analysis_builds_state_graph(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = SourceAnalysisAgent(memory).run(Path("examples/sample_target.py"))
    assert graph.nodes
    assert any(node.node_type == "Branch" for node in graph.nodes.values())
    assert any(node.node_type == "Loop" for node in graph.nodes.values())


def test_generation_execution_and_evaluation(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    SourceAnalysisAgent(memory).run(Path("examples/sample_target.py"))
    test_path = TestGenerationAgent(memory).run(tmp_path)
    assert test_path.exists()
    report = ExecutionAgent(memory).run(test_path, cwd=Path.cwd())
    assert report.pytest_passed
    assert report.sfc >= 0.0
    suggestions = EvaluationAgent(memory).run()
    assert isinstance(suggestions, list)


def test_coverage_runner_tracks_dynamic_loader_source(tmp_path):
    source = tmp_path / "task_000.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    test_path = tmp_path / "out" / "test_task_000_stateflow.py"
    test_path.parent.mkdir()
    rendered = TestPlanRenderer().render(
        source,
        GeneratedPlan([PlanCase("add_case", "add", [1, 2], PlanOracle("equals", value=3))]),
        {"add": {"node_type": "Function"}},
    )
    test_path.write_text(rendered, encoding="utf-8")

    result = CoverageRunner().run(source, test_path, cwd=tmp_path)

    assert result.valid
    assert result.line_coverage == 1.0
    assert result.covered_lines


def test_runtime_observer_records_class_method_calls(tmp_path):
    source = tmp_path / "task_001.py"
    source.write_text(
        "class Solution:\n"
        "    def add(self, a, b):\n"
        "        return a + b\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "test_task_001_stateflow.py"
    test_path.write_text(
        "import importlib.util\n"
        "import pytest\n\n"
        "def _load_target():\n"
        f"    spec = importlib.util.spec_from_file_location('target', r'''{source}''')\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n"
        "@pytest.fixture\n"
        "def target():\n"
        "    return _load_target()\n\n"
        "def test_add(target):\n"
        "    result = target.Solution().add(1, 2)\n"
        "    assert result == 3\n",
        encoding="utf-8",
    )

    observations = RuntimeObserver().collect(source, test_path)

    assert len(observations) == 1
    assert observations[0].function == "add"
    assert observations[0].class_name == "Solution"
    assert observations[0].return_type == "int"


def test_graph_updater_maps_class_method_assertion(tmp_path):
    source = tmp_path / "task_002.py"
    source.write_text("class Solution:\n    def add(self, a, b):\n        return a + b\n", encoding="utf-8")
    graph = StateFlowGraph(source_path=str(source))
    method = graph.add_entity(
        "Function",
        "add",
        2,
        3,
        "def add(self, a, b):",
        metadata={"stable_id": "program:function:Solution:add", "scope": "Solution", "args": ["self", "a", "b"]},
    )
    test_path = tmp_path / "test_task_002_stateflow.py"
    test_path.write_text(
        "def test_add(target):\n"
        "    result = target.Solution().add(1, 2)\n"
        "    assert result == 3\n",
        encoding="utf-8",
    )

    updater = GraphUpdater(graph)
    assert updater._called_target_functions(ast.parse(test_path.read_text(encoding="utf-8")).body[0]) == {"add"}
    assert updater.apply_assert_counts(test_path) == 1
    assert method.assert_count == 1


def test_graph_updater_disambiguates_duplicate_class_methods(tmp_path):
    from types import SimpleNamespace

    source = tmp_path / "task_002.py"
    source.write_text(
        "class A:\n"
        "    def value(self):\n"
        "        return 1\n\n"
        "class B:\n"
        "    def value(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )
    graph = StateFlowGraph(source_path=str(source))
    method_a = graph.add_entity(
        "Function",
        "value",
        2,
        3,
        "def value(self):",
        metadata={"stable_id": "program:function:A:value", "scope": "A", "args": ["self"]},
    )
    method_b = graph.add_entity(
        "Function",
        "value",
        6,
        7,
        "def value(self):",
        metadata={"stable_id": "program:function:B:value", "scope": "B", "args": ["self"]},
    )
    test_path = tmp_path / "test_task_002_stateflow.py"
    test_path.write_text(
        "def test_value(target):\n"
        "    result = target.B().value()\n"
        "    assert result == 2\n",
        encoding="utf-8",
    )

    updater = GraphUpdater(graph)
    assert updater.apply_assert_counts(test_path) == 1
    assert method_a.assert_count == 0
    assert method_b.assert_count == 1

    updater.record_execution_evidence(
        test_path,
        SimpleNamespace(passed=True, returncode=0),
        SimpleNamespace(percent=1.0, line_coverage=1.0, branch_coverage=None, combined_coverage=1.0, covered_lines={6, 7}),
        SimpleNamespace(score=None, killed_lines=[], survived_lines=[], timeout_lines=[], invalid_lines=[], infra_error_lines=[], equivalent_lines=[]),
        [
            SimpleNamespace(
                test_name="test_value",
                nodeid="test_task_002_stateflow.py::test_value",
                function="value",
                class_name="B",
                arg_types=[],
                return_type="int",
                exception_type=None,
                exception_message_pattern=None,
                value_summary="2",
            )
        ],
    )
    runtime_states = [node for node in graph.nodes.values() if node.node_type == "RuntimeState"]
    assert any(node.metadata.get("class_name") == "B" for node in runtime_states)
    test_nodes = [node for node in graph.nodes.values() if node.node_type == "TestCase"]
    assert any(edge.source == test_nodes[0].node_id and edge.target == method_b.node_id and edge.edge_type == "Execute" for edge in graph.edges)
    assert not any(edge.source == test_nodes[0].node_id and edge.target == method_a.node_id and edge.edge_type == "Execute" for edge in graph.edges)


def test_patch_search_accepts_line_coverage_improvement():
    repair = PatchSearchRepair(
        Path("examples/sample_target.py"),
        baseline_report={
            "pytest_passed": True,
            "line_coverage": 0.5,
            "branch_coverage": 0.5,
            "sfc": 0.5,
            "ae": 0.5,
            "mutation_score": 0.5,
            "sfq": 0.5,
        },
    )

    accepted, reason = repair._accept(
        sfc=0.5,
        ae=0.5,
        mutation=0.5,
        sfq=0.5,
        line_coverage=0.6,
        branch_coverage=0.5,
    )

    assert accepted
    assert reason == "accepted by patch search validation"


def test_generation_skips_nested_local_functions(tmp_path):
    source = tmp_path / "task_003.py"
    source.write_text(
        "def outer(value):\n"
        "    def inner(x):\n"
        "        return x + 1\n"
        "    return inner(value)\n\n"
        "class Solution:\n"
        "    def double(self, value):\n"
        "        return value * 2\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = SourceAnalysisAgent(memory).run(source)

    functions = TestGenerationAgent(memory)._top_level_functions(graph)
    names = {(item.get("class_name"), item["name"]) for item in functions}

    assert (None, "outer") in names
    assert ("Solution", "double") in names
    assert (None, "inner") not in names


def test_pteg_evidence_query_and_scheduler(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    SourceAnalysisAgent(memory).run(Path("examples/sample_target.py"))
    test_path = TestGenerationAgent(memory).run(tmp_path)
    ExecutionAgent(memory).run(test_path, cwd=Path.cwd())
    graph = memory.load()
    node_types = {node.node_type for node in graph.nodes.values()}
    assert {"Execution", "TestCase", "Assertion", "Mutation", "RuntimeState", "RuntimeType"} <= node_types
    assert any(edge.edge_type == "Observe" for edge in graph.edges)
    assert any(edge.edge_type == "TypeFlow" for edge in graph.edges)
    assert graph.metadata.get("last_defect_propagation")
    assert graph.metadata["last_defect_propagation"]["root_cause_reasons"]
    assert graph.metadata["last_defect_propagation"]["root_cause_explanation"]
    query = GraphQuery(graph)
    assert query.get_execution_trace()["execution"] is not None
    decision = GraphDrivenAdaptiveScheduler(graph).decide()
    assert decision.agent
    assert graph.metadata["last_defect_propagation"]["root_cause_node"] is not None
    assert graph.metadata["last_agent_decision"]["agent"] in {
        "CoverageAgent",
        "MutationAgent",
        "AssertionAgent",
        "BoundaryAgent",
        "TypeAnalysisAgent",
        "ExecutionAgent",
    }
    assert graph.metadata["last_agent_decision"]["action"] in {"repair", "optimize", "verify", "stop"}


def test_scheduler_and_minimal_repair_plan_on_defect(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = SourceAnalysisAgent(memory).run(Path("examples/sample_target.py"))
    decision = GraphDrivenAdaptiveScheduler(graph).decide()
    assert decision.agent in {"CoverageAgent", "BoundaryAgent", "AssertionAgent", "MutationAgent", "TypeAnalysisAgent"}
    function_node = next(node for node in graph.nodes.values() if node.node_type == "Function")
    function_node.defect_score = 1.0
    function_node.coverage_defect = 1.0
    test_path = tmp_path / "test_minimal.py"
    test_path.write_text("def test_existing(target):\n    assert target is not None\n", encoding="utf-8")
    planner = RootCauseRepairPlanner(graph)
    plan = planner.plan(test_path)
    assert plan is not None
    assert plan.root_name == function_node.name
    assert planner.apply(test_path, plan)
    assert "test_stateflow_minimal_repair" in test_path.read_text(encoding="utf-8")


def test_all_agents_can_use_llm_path(tmp_path):
    fake_llm = _FakeLLM()
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = SourceAnalysisAgent(memory, llm=fake_llm, llm_required=True).run(Path("examples/sample_target.py"))
    assert graph.metadata["analysis_strategy"] == "ast_plus_llm"

    test_path = TestGenerationAgent(memory, llm=fake_llm, llm_required=True).run(tmp_path)
    generated_text = test_path.read_text(encoding="utf-8")
    assert "def _load_target():" in generated_text
    assert "target.classify_score" in generated_text
    assert "test_llm_generated_classify_score" in generated_text
    assert "from task_" not in generated_text

    execution = ExecutionAgent(memory, llm=fake_llm, llm_required=True)
    execution.mutation_runner.max_mutants = 3
    report = execution.run(test_path, cwd=Path.cwd())
    assert report.pytest_passed
    graph = memory.load()
    assert graph.metadata["last_execution_result"]["pytest_passed"] is True
    assert graph.metadata["last_report"]["sfc"] >= 0.0
    assert graph.metadata["last_report"]["ae"] >= 0.0
    assert graph.metadata["last_report"]["mutation_score"] >= 0.0

    suggestions = EvaluationAgent(memory, llm=fake_llm, llm_required=True).run()
    RepairOptimizationAgent(memory, llm=fake_llm, llm_required=True).run(test_path, suggestions)

    called_text = "\n".join(fake_llm.calls)
    assert "Source Analysis Agent" in called_text
    assert "Test Generation Agent" in called_text
    assert "Execution Agent" in called_text
    assert "quality evaluation module inside the Test State Agent" in called_text
    assert "Repair Optimization Agent" in called_text


def test_c_grade_llm_prompt_receives_targeted_optimization_contract(tmp_path):
    fake_llm = _FakeLLM()
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = SourceAnalysisAgent(memory).run(Path("examples/sample_target.py"))
    graph.metadata["c_grade_optimization_contract"] = {
        "dominant_gap": "mutation",
        "baseline_metrics": {"mutation_score": 0.2, "sfq": 0.72},
        "survived_mutation_lines": [8, 12],
        "target_methods": ["classify_score"],
        "constraints": ["preserve passing tests", "do not regress any metric"],
    }
    graph.metadata["minimal_target_repair"] = {"level": "test-method-level", "target": "test_classify_score"}
    graph.metadata["optimization_prompt_version"] = "c-r3-targeted-v2"
    memory.save(graph)

    agent = RepairOptimizationAgent(memory, llm=fake_llm, llm_required=True, focus="mutation")
    operations = agent.optimize_operations_with_llm("def test_old(target):\n    assert target.classify_score(90) == 'excellent'\n", [], plan=None)

    prompt = fake_llm.calls[-1]
    assert operations and operations[0].op == "add_test"
    assert "C-grade targeted optimization contract" in prompt
    assert "survived_mutation_lines" in prompt
    assert "State-guided minimal target scope" in prompt
    assert "never delete a passing test" in prompt


def test_system_iterates_by_quality_thresholds_with_llm(tmp_path):
    fake_llm = _FakeLLM()
    config = LLMConfig(enabled=True, required=True, model="fake-llm")
    system = MultiAgentUnitTestSystem(
        source_path=Path("examples/sample_target.py"),
        output_dir=tmp_path / "run",
        thresholds=Thresholds(sfc=2.0, ae=2.0, mutation=2.0),
        max_iterations=2,
        llm_config=config,
        llm_client=fake_llm,
    )
    system.execution_agent.mutation_runner.max_mutants = 3

    summary = system.run()

    assert summary["iterations"] == 4
    assert summary["repair_rounds_used"] == 2
    assert len(summary["reports"]) == 4
    assert len(summary["agent_decisions"]) == 4
    assert summary["reports"][0]["evaluation_profile"] == "screening"
    assert summary["final_report"]["evaluation_profile"] == "full"
    assert summary["final_report"]["mutation_complete"] is True
    assert "sfc" not in summary["reports"][0]
    assert system.memory.load().metadata["last_report"]["sfc"] < 2.0
    assert summary["reports"][0]["ae"] < 2.0
    assert summary["reports"][0]["mutation_score"] < 2.0
    graph = system.memory.load()
    assert graph.metadata["active_specialized_agent"] in {
        "测试生成智能体/覆盖增强模块",
        "测试生成智能体/断言增强模块",
        "测试生成智能体/变异增强模块",
        "测试生成智能体/边界异常模块",
        "测试生成智能体/运行类型模块",
    }
    assert graph.metadata["repair_agent"]["minimal_plan"] is not None
    assert "minimal_patch_applied" in graph.metadata["repair_agent"]
    assert graph.metadata["test_repair_strategy"] in {"llm_targeted_operations", "llm_root_cause_minimal_patch", "heuristic"}
    assert "patch_search" in graph.metadata
    assert graph.metadata["last_agent_experience"]["reward"] <= 0.0
    assert system.experience.path.exists()
    assert "agent_reward_predictions" in graph.metadata
    assert graph.metadata["collaboration_model"]["primary_agents"] == ["TestStateAgent", "TestGenerationAgent", "TestKnowledgeAgent"]
    assert graph.metadata["last_test_intent_plan"]["owner"] == "测试状态智能体"
    assert any(item.get("event_type") == "KnowledgeUpdated" for item in graph.flow_history)
    assert graph.metadata["last_evaluation_feedback"]["quality_status"] in {"valid_usable", "quality_acceptable", "quality_high"}
    assert summary["test_intent_plans"]
    assert summary["reports"][0]["evaluation_feedback"]["recommended_focus"]


def test_default_llm_config_uses_local_ollama():
    config = LLMConfig()
    assert config.provider == "ollama"
    assert config.base_url == "http://127.0.0.1:11434"
    assert config.model == "deepseek-r1:14b"


def test_ollama_client_chat_uses_native_client(monkeypatch):
    calls = {}

    class _Client:
        def __init__(self, host):
            calls["host"] = host

        def chat(self, model, messages):
            calls["model"] = model
            calls["messages"] = messages
            return {"message": {"content": "local response"}}

    class _OllamaModule:
        Client = _Client

    import sys

    monkeypatch.setitem(sys.modules, "ollama", _OllamaModule())
    llm = OpenAICompatibleLLM(LLMConfig(enabled=True, provider="ollama", base_url="http://127.0.0.1:11434", model="deepseek-r1:14b"))
    result = llm.chat([{"role": "user", "content": "hello"}])
    assert result == "local response"
    assert calls["host"] == "http://127.0.0.1:11434"
    assert calls["model"] == "deepseek-r1:14b"


def test_ollama_client_chat_accepts_object_response(monkeypatch):
    class _Message:
        content = "object response"

    class _Response:
        message = _Message()

    class _Client:
        def __init__(self, host):
            self.host = host

        def chat(self, model, messages):
            return _Response()

    class _OllamaModule:
        Client = _Client

    import sys

    monkeypatch.setitem(sys.modules, "ollama", _OllamaModule())
    llm = OpenAICompatibleLLM(LLMConfig(enabled=True, provider="ollama", base_url="http://127.0.0.1:11434", model="deepseek-r1:14b"))
    assert llm.chat([{"role": "user", "content": "hello"}]) == "object response"


def test_threshold_requires_pytest_passed():
    report = type(
        "Report",
        (),
        {"pytest_passed": False, "sfc": 1.0, "ae": 1.0, "mutation_score": 1.0},
    )()
    assert not Thresholds().reached(report)


def test_mutation_result_na_score_is_serialized():
    result = MutationResult(
        killed=0,
        survived=0,
        timeout=0,
        invalid=0,
        infra_error=0,
        equivalent=0,
        total=0,
        score=None,
        killed_lines=[],
        survived_lines=[],
        timeout_lines=[],
        invalid_lines=[],
        infra_error_lines=[],
        equivalent_lines=[],
        details=["no mutation candidates"],
    )
    assert result.to_dict()["score"] is None
    assert MutationScoreMetric().calculate(0, 0) is None


def test_generated_test_normalizer_fixes_path_loader_mistakes():
    missing_import = (
        "import os\n"
        "DEFAULT_TARGET = Path(r'''D:\\\\dataset\\\\task_001.py''')\n"
        "TARGET_PATH = Path(os.environ.get('STATEFLOW_TARGET', DEFAULT_TARGET)).resolve()\n"
    )
    fixed_missing_import = normalize_generated_test_code(missing_import)
    assert "from pathlib import Path" in fixed_missing_import

    string_resolve = (
        "import os\n"
        "from pathlib import Path\n"
        "DEFAULT_TARGET = os.path.join(r'D:\\\\dataset\\\\task_000.py')\n"
        "TARGET_PATH = os.environ.get('STATEFLOW_TARGET', DEFAULT_TARGET)\n"
        "target_module = TARGET_PATH.resolve()\n"
    )
    fixed_string_resolve = normalize_generated_test_code(string_resolve)
    assert "target_module = Path(TARGET_PATH).resolve()" in fixed_string_resolve


def test_generated_test_normalizer_replaces_function_name_imports():
    code = (
        "import os\n"
        "import sys\n"
        "import has_close_elements\n"
        "\n"
        "def test_empty_list():\n"
        "    assert has_close_elements([] , 0.1) is False\n"
    )
    fixed = normalize_generated_test_code(
        code,
        source_path=Path("dataset/sources/task_000.py"),
        function_names=["has_close_elements"],
    )
    assert "import has_close_elements" not in fixed
    assert "target_module = _load_target()" in fixed
    assert "has_close_elements = target_module.has_close_elements" in fixed


def test_assertion_effectiveness_calibrates_complete_runtime_oracles():
    graph = StateFlowGraph(source_path="sample.py")
    runtime = graph.add_entity("RuntimeState", "foo runtime", metadata={"stable_id": "runtime:foo"})
    assertion = graph.add_entity(
        "Assertion",
        "test_foo:assert",
        metadata={"stable_id": "assertion:test_foo:1", "effective": True, "specificity": 1.0},
    )
    graph.add_relation(assertion.node_id, runtime.node_id, "Observe", weight=1.0)
    ae = AssertionEffectivenessMetric().calculate(graph, total_asserts=1, mutation_score=1.0)
    assert ae >= 0.9


def test_sfq_renormalizes_when_mutation_is_not_applicable():
    score = StateFlowQualityMetric().calculate(sfc=0.7, ae=0.9, mutation_score=None)
    assert score == round((0.30 * 0.7 + 0.20 * 0.9 + 0.15 + 0.15) / 0.80, 4)


def test_layered_thresholds_ignore_unavailable_mutation():
    report = ExecutionReport(
        pytest_passed=True,
        coverage_percent=0.95,
        mutation_score=None,
        sfc=0.8,
        ae=0.8,
        sfq=0.85,
        test_path="test_sample.py",
        line_coverage=0.95,
        branch_coverage=0.90,
        combined_coverage=0.94,
        collected_count=3,
        effective_mutants=0,
        mutation_reliability="unavailable",
        coverage_valid=True,
        mutation_valid=True,
        normal_behavior_tests=2,
        assertion_count=2,
        effective_assertions=2,
    )
    thresholds = Thresholds()
    assert thresholds.classify(report) == "quality_high"
    assert thresholds.reached(report)


def test_observation_timeout_is_local_evidence(monkeypatch, tmp_path):
    import subprocess

    memory = SharedGraphMemory(tmp_path / "graph.json")
    agent = TestGenerationAgent(memory)
    graph = StateFlowGraph(source_path=str(tmp_path / "task_156.py"))

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="observe", timeout=3)

    monkeypatch.setattr("agents.generation_agent.subprocess.run", timeout)
    result = agent._observe_call("int_to_mini_roman", ["-1"], {"node_type": "Function"}, graph)
    assert result == {"status": "timeout"}
    assert graph.metadata["observation_timeouts"][0]["function"] == "int_to_mini_roman"


def test_deterministic_generation_uses_docstring_examples(tmp_path):
    source = tmp_path / "task_043.py"
    source.write_text(
        "def pairs_sum_to_zero(l):\n"
        "    \"\"\"\n"
        "    >>> pairs_sum_to_zero([1, 3, 5, 0])\n"
        "    False\n"
        "    >>> pairs_sum_to_zero([2, 4, -5, 3, 5, 7])\n"
        "    True\n"
        "    \"\"\"\n"
        "    for i, l1 in enumerate(l):\n"
        "        for j in range(i + 1, len(l)):\n"
        "            if l1 + l[j] == 0:\n"
        "                return True\n"
        "    return False\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    SourceAnalysisAgent(memory).run(source)

    test_path = TestGenerationAgent(memory).run(tmp_path / "out")
    rendered = test_path.read_text(encoding="utf-8")

    assert "pairs_sum_to_zero([1, 3, 5, 0])" in rendered
    assert "assert result == False" in rendered
    usability = analyze_test_usability(test_path)
    assert usability.normal_behavior_tests >= 2
    assert not usability.exception_only


def test_batch_experience_guides_next_generation(tmp_path):
    experience = AgentExperienceMemory(tmp_path / "agent_experience.json")
    record = experience.append_batch_summary(
        {
            "source_dir": "sources",
            "output_dir": "out",
            "succeeded": 0,
            "failed": 1,
            "results": [
                {
                    "source": "task_043.py",
                    "status": "failed",
                    "error": "RuntimeError: no valid generated test candidate: exception_only",
                }
            ],
        },
        model="test-model",
    )
    graph = StateFlowGraph(source_path="task_043.py")

    hints = experience.hints(graph, model="test-model")

    assert record["failure_categories"]["exception_only"] == 1
    assert hints["batch_guidance"]["guidance"]
    assert "正常输入/输出断言" in hints["batch_guidance"]["guidance"][0]


def test_experience_memory_records_actionable_failure_and_optimization_lessons(tmp_path):
    experience = AgentExperienceMemory(tmp_path / "agent_experience.json", mirror_paths=[])
    graph = StateFlowGraph(source_path="task_043.py")
    graph.add_entity("Function", "public", metadata={"arg_annotations": {"value": "int"}})

    failure_record = experience.append_candidate(
        graph=graph,
        agent="ExecutionAgent/TestGenerationAgent",
        action="generate_test_plan",
        candidate_text="def test_public(target):\n    assert target.public('bad') == 1\n",
        accepted=False,
        pytest_passed=False,
        failure_category="runtime_error",
        rejection_reason="invalid argument type",
    )
    before_report = ExecutionReport(
        pytest_passed=True,
        coverage_percent=0.2,
        mutation_score=0.1,
        sfc=0.2,
        ae=0.3,
        sfq=0.25,
        test_path="test_task_043_stateflow.py",
        collected_count=1,
        normal_behavior_tests=1,
    )
    weak_report = ExecutionReport(
        pytest_passed=True,
        coverage_percent=0.2,
        mutation_score=0.1,
        sfc=0.2,
        ae=0.3,
        sfq=0.25,
        test_path="test_task_043_stateflow.py",
        line_coverage=0.2,
        branch_coverage=0.0,
        coverage_valid=True,
        mutation_valid=True,
        collected_count=1,
        normal_behavior_tests=1,
        assertion_count=1,
        effective_assertions=1,
        weak_quality_flags=["weak_line_coverage", "weak_mutation_score"],
        threshold_failures={"line_coverage_gap": 0.6, "mutation_gap": 0.7},
        quality_status="valid_usable",
    )
    optimization_record = experience.append(
        graph=graph,
        agent="MutationAgent",
        gaps={"mutation": 0.7, "coverage": 0.6},
        before=before_report,
        after=weak_report,
    )

    hints = experience.hints(graph)

    assert failure_record["learning_notes"][0]["reason_type"] == "runtime_error"
    assert "small valid objects" in failure_record["learning_notes"][0]["solution"]
    assert any(note["reason_type"] == "weak_line_coverage" for note in optimization_record["learning_notes"])
    assert any("runtime_error" in reminder for reminder in hints["failure_reminders"])
    assert any("weak_mutation_score" in reminder for reminder in hints["optimization_reminders"])


def test_batch_experience_records_weak_success_learning_notes(tmp_path):
    experience = AgentExperienceMemory(tmp_path / "agent_experience.json", mirror_paths=[])
    record = experience.append_batch_summary(
        {
            "source_dir": "sources",
            "output_dir": "out",
            "succeeded": 1,
            "failed": 0,
            "results": [
                {
                    "source": "task_001.py",
                    "status": "valid_usable",
                    "final_report": {
                        "pytest_passed": True,
                        "quality_status": "valid_usable",
                        "weak_quality_flags": ["weak_branch_coverage"],
                        "threshold_failures": {"branch_coverage_gap": 0.5},
                    },
                }
            ],
        },
        model="test-model",
    )

    assert record["weak_quality_flag_counts"] == {"weak_branch_coverage": 1}
    assert record["threshold_gap_counts"] == {"branch_coverage_gap": 1}
    assert any(note["reason_type"] == "valid_but_weak_success" for note in record["learning_notes"])
    assert any("weak_branch_coverage" in reminder for reminder in record["prompt_reminders"])


def test_visualizer_creates_offline_graph_and_source_heatmap(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def sample(value):\n    if value:\n        return 1\n    return 0\n", encoding="utf-8")
    graph = StateFlowGraph(source_path=str(source))
    function = graph.add_entity("Function", "sample", 1, 4, "def sample(value):", metadata={"stable_id": "program:function:sample"})
    branch = graph.add_entity("Branch", "value", 2, 3, "if value:", metadata={"stable_id": "program:branch:sample"})
    function.mark_visited(); branch.mark_visited()
    graph.add_relation(function.node_id, branch.node_id, "flow")
    graph.mark_edge_visited(function.node_id, branch.node_id)

    paper_dir = tmp_path / "out" / "paper_metrics"
    paper_dir.mkdir(parents=True)
    (paper_dir / "summary.json").write_text(json.dumps({
        "tir": 0.5, "tir_improved": 1, "tir_comparable": 2, "tir_excluded": 1,
        "avg_exec": 3.0, "duration_seconds": 12.5, "pass_rate": 1.0,
    }), encoding="utf-8")
    result = StateFlowGraphVisualizer().render(graph, tmp_path / "out")

    assert result["status"] == "generated"
    graph_html = (tmp_path / "out" / "state_flow_graph.html").read_text(encoding="utf-8")
    source_html = (tmp_path / "out" / "source_coverage.html").read_text(encoding="utf-8")
    assert "已覆盖" in graph_html and "未覆盖" in graph_html
    assert "program:function:sample" in graph_html
    assert "源码覆盖热力图" in source_html
    assert "ValiFixTest &amp; TSGColAgent" in graph_html
    assert "TIR" in graph_html and "50%" in graph_html
    assert "AvgExec" in source_html and "3.00" in source_html
    assert "SFC" not in graph_html and "TSQ" not in graph_html
    assert "SFC" not in source_html and "TSQ" not in source_html
    assert "https://" not in graph_html


def test_source_heatmap_uses_normalized_metric_filters(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def sample(value):\n    if value:\n        return 1\n    return 0\n", encoding="utf-8")
    graph = StateFlowGraph(source_path=str(source))
    function = graph.add_entity("Function", "sample", 1, 4, "def sample(value):", metadata={"stable_id": "program:function:sample"})
    branch = graph.add_entity("Branch", "value", 2, 3, "if value:", metadata={"stable_id": "program:branch:sample"})
    function.mark_visited()
    branch.mark_visited()
    function.assert_count = 1
    graph.add_relation(function.node_id, branch.node_id, "flow")
    graph.metadata["last_coverage"] = {"covered_lines": [1, 2], "missing_lines": [3, 4], "line_coverage": 0.5}
    graph.metadata["last_mutation"] = {
        "killed": 1,
        "survived": 1,
        "timeout": 1,
        "effective_mutants": 2,
        "killed_lines": [2],
        "survived_lines": [3],
        "timeout_lines": [4],
        "low_sample_reason": "有效变异体少于 3 个",
    }

    StateFlowGraphVisualizer().render(graph, tmp_path / "out")
    source_html = (tmp_path / "out" / "source_coverage.html").read_text(encoding="utf-8")

    assert "只看运行时" not in source_html
    assert "只看分支覆盖" in source_html
    assert "只看有效断言" in source_html
    assert "只看变异检出" in source_html
    assert 'body[data-view="mutation"]' in source_html


def test_sync_test_template_does_not_import_asyncio(tmp_path):
    source = tmp_path / "plain.py"
    source.write_text("def double(value):\n    return value * 2\n", encoding="utf-8")
    plan = GeneratedPlan([PlanCase("double", "double", [2], PlanOracle("equals", value=4))])
    rendered = TestPlanRenderer().render(source, plan, {"double": {"node_type": "Function"}})
    assert "import asyncio" not in rendered
    assert "assert result == 4" in rendered


def test_candidate_validator_rejects_exception_only_suite(tmp_path):
    code = '''import pytest

class Target:
    def run(self, value):
        return value

@pytest.fixture
def target():
    return Target()

def test_only_invalid_input(target):
    with pytest.raises(TypeError):
        target.run() 
'''
    result = CandidateTestValidator(tmp_path / "rejected").validate_and_write(code, tmp_path / "test_candidate.py")
    assert not result.accepted
    assert result.stage == "semantic_usability"
    assert result.failure_category == "exception_only"


def test_repair_operation_accepts_schema_aliases():
    operation = RepairOperation.from_dict({"operation": "append_test", "code": "def test_value(target):\n    assert target.f(1) == 2"})
    assert operation.op == "add_test"


def test_exception_only_assertions_do_not_inflate_ae():
    graph = StateFlowGraph(source_path="sample.py")
    runtime = graph.add_entity("RuntimeState", "runtime")
    assertion = graph.add_entity(
        "Assertion",
        "raises",
        code="with pytest.raises(TypeError): target.run(None)",
        metadata={"effective": True, "specificity": 1.0},
    )
    graph.add_relation(assertion.node_id, runtime.node_id, "Observe")
    assert AssertionEffectivenessMetric().calculate(graph, total_asserts=1) <= 0.30


def test_execution_agent_retries_pytest_timeout(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    memory.save(StateFlowGraph(source_path="sample.py"))
    agent = ExecutionAgent(memory, retry_rounds=2)
    calls = {"count": 0}

    class _Result:
        def __init__(self, timed_out):
            self.timed_out = timed_out

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        return _Result(timed_out=calls["count"] == 1)

    agent.pytest_runner.run = fake_run
    result, attempts = agent._run_pytest_with_retries(tmp_path / "test_sample.py", cwd=tmp_path)

    assert attempts == 2
    assert not result.timed_out


def test_pytest_timeout_records_reason_and_location(tmp_path):
    test_path = tmp_path / "test_timeout.py"
    test_path.write_text(
        "def test_target_hangs(target):\n"
        "    assert target.slow_function(10) == 10\n",
        encoding="utf-8",
    )

    diagnosis = PytestFailureClassifier().classify(
        PytestResult(124, "", "timed out", False, timed_out=True),
        test_path,
        timed_out=True,
    )

    assert diagnosis.failure_category == "timeout"
    assert diagnosis.timeout_kind == "target_call_timeout"
    assert diagnosis.should_repair is True
    assert "slow_function" in (diagnosis.timeout_reason or "")
    assert diagnosis.failed_tests[0].test_name == "test_target_hangs"


def test_scheduler_routes_local_timeout_to_minimal_repair(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = StateFlowGraph(source_path=str(tmp_path / "sample.py"))
    graph.metadata["execution_diagnosis"] = {
        "failure_category": "timeout",
        "timeout_kind": "target_call_timeout",
        "timeout_reason": "target call did not return",
        "should_repair": True,
    }
    memory.save(graph)
    report = ExecutionReport(
        pytest_passed=False,
        coverage_percent=0.0,
        mutation_score=None,
        sfc=0.0,
        ae=0.0,
        sfq=0.0,
        test_path=str(tmp_path / "test_sample.py"),
        failure_category="timeout",
        timed_out=True,
        timeout_kind="target_call_timeout",
        timeout_reason="target call did not return",
        timeout_should_repair=True,
    )

    decision = GraphDrivenAdaptiveScheduler(memory.load()).decide(report=report, thresholds=Thresholds())

    assert decision.action == "repair"
    assert decision.agent == "RepairAgent"


def test_execution_agent_falls_back_to_repair_agent_when_selected_tool_has_no_change(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = StateFlowGraph(source_path=str(tmp_path / "sample.py"))
    memory.save(graph)
    test_path = tmp_path / "test_sample.py"
    test_path.write_text("def test_old():\n    assert True\n", encoding="utf-8")
    agent = ExecutionAgent(memory)

    class _NoChangeTool:
        def run(self, test_path, suggestions=None):
            graph = memory.load()
            graph.metadata["repair_agent"] = {"applied": False}
            memory.save(graph)

    class _RepairTool:
        def run(self, test_path, suggestions):
            Path(test_path).write_text("def test_repaired():\n    assert 1 == 1\n", encoding="utf-8")
            graph = memory.load()
            graph.metadata["repair_agent"] = {"applied": True}
            memory.save(graph)

    plan = type("Plan", (), {"selected_agent": "TestValidityAgent", "action": "repair"})()
    repaired = agent.apply_plan(
        test_path,
        plan,
        repair_agent=_RepairTool(),
        specialized_agents={"TestValidityAgent": _NoChangeTool()},
        evaluation_agent=EvaluationAgent(memory),
        suggestions=[],
    )

    assert repaired
    assert "test_repaired" in test_path.read_text(encoding="utf-8")
    attempts = memory.load().metadata["execution_repair_attempts"]
    assert [item["agent"] for item in attempts] == ["TestValidityAgent", "OracleRepairAgent"]


def test_usable_test_without_patch_stops_after_first_execution(tmp_path, monkeypatch):
    system = MultiAgentUnitTestSystem(
        source_path=Path("examples/sample_target.py"),
        output_dir=tmp_path / "run",
        thresholds=Thresholds(line_coverage=2.0, branch_coverage=2.0, sfc=2.0, mutation=2.0, sfq=2.0),
        max_iterations=3,
        llm_config=LLMConfig(enabled=False),
    )
    monkeypatch.setattr(system, "_run_selected_agent", lambda agent, test_path, intent_plan=None, feedback=None: False)

    summary = system.run()

    assert summary["iterations"] == 2
    assert summary["stop_reason"] in {"quality_high", "quality_acceptable", "valid_usable"}
    assert summary["final_report"]["pytest_passed"] is True
    assert summary["final_report"]["evaluation_profile"] == "full"


def test_planning_agent_replans_from_evaluation_feedback(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    planner = PlanningAgent(memory)
    executor = ExecutionAgent(memory)
    executor.mutation_runner.max_mutants = 3
    test_path, initial_plan = planner.bootstrap(
        Path("examples/sample_target.py"),
        tmp_path / "run",
        AgentExperienceMemory(tmp_path / "agent_experience.json"),
        execution_agent=executor,
    )
    report = executor.run(test_path, cwd=Path.cwd())
    report.quality_status = Thresholds(sfc=2.0, sfq=2.0, mutation=2.0).classify(report)
    feedback = EvaluationAgent(memory).run_feedback(report, Thresholds(sfc=2.0, sfq=2.0, mutation=2.0), AgentExperienceMemory(tmp_path / "agent_experience.json"))

    decision, replanned = planner.replan(feedback, report, Thresholds(sfc=2.0, sfq=2.0, mutation=2.0), AgentExperienceMemory(tmp_path / "agent_experience.json"))
    graph = memory.load()

    assert initial_plan.owner == "测试状态智能体"
    assert decision.agent == replanned.selected_agent
    assert replanned.phase == "feedback_replan"
    assert graph.metadata["last_test_intent_plan"]["phase"] == "feedback_replan"
    assert any(item["artifact_type"] == "EvaluationFeedback" for item in graph.metadata["agent_collaboration_exchange"])


def test_state_planning_delegates_generation_to_test_generation_agent(tmp_path):
    memory = SharedGraphMemory(tmp_path / "graph.json")
    planner = PlanningAgent(memory)
    executor = ExecutionAgent(memory)

    test_path, plan = planner.bootstrap(
        Path("examples/sample_target.py"),
        tmp_path / "run",
        AgentExperienceMemory(tmp_path / "agent_experience.json"),
        execution_agent=executor,
    )
    graph = memory.load()
    exchange = graph.metadata.get("agent_collaboration_exchange", [])

    assert test_path.exists()
    assert plan.selected_agent == "TestGenerationAgent"
    assert not hasattr(planner, "analysis_agent")
    assert not hasattr(planner, "generation_agent")
    assert graph.metadata["execution_subagents"]["analysis"] == "测试状态智能体/源码分析模块"
    assert graph.metadata["execution_subagents"]["generation"] == "测试生成智能体/生成模块"
    assert any(item.get("artifact_type") == "AnalysisResult" for item in exchange)
    assert any(item.get("artifact_type") == "GenerationResult" for item in exchange)
