from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import json
import time
import tomllib
import xml.etree.ElementTree as ET

from analysis.context_rewriter import ContextRewriter
from analysis.java_context_extractor import JavaStructuredContextExtractor
from analysis.python_context_extractor import PythonStructuredContextExtractor
from agents.execution_agent import ExecutionAgent
from agents.generation_agent import TestGenerationAgent
from executor.jacoco_runner import JacocoRunner
from executor.pit_runner import PitRunner
from executor.build_tools import maven_local_repo_arg, maven_project_settings_args, resolve_maven_command
from executor.java_toolchain import JavaToolchain
from graph.completeness_metric import CompleteStateGraphMetric
from graph.experience_memory import AgentExperienceMemory
from graph.structured_graph_builder import StructuredGraphBuilder
from graph.state_graph import MAX_ROLLBACK_SNAPSHOTS, StateFlowGraph
from memory.shared_graph import SharedGraphMemory
from repair.minimal_target_repair import MinimalTargetRepair
from validation.dynamic_validator import DynamicValidator
from main import MultiAgentUnitTestSystem
from orchestration.test_workflow import LangGraphTestWorkflow


def test_python_structured_context_and_graph_builder(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text(
        "import math\n\n"
        "class Solution:\n"
        "    def add(self, a: int, b: int) -> int:\n"
        "        if a > b:\n"
        "            return a + b\n"
        "        return b + a\n\n"
        "def helper(value):\n"
        "    return math.sqrt(value)\n",
        encoding="utf-8",
    )

    context = ContextRewriter().rewrite(PythonStructuredContextExtractor().extract(source))
    graph = StructuredGraphBuilder(context).build()

    assert context.language == "python"
    assert context.classes[0].methods[0].qualified_name.endswith("Solution.add")
    assert any(node.node_type == "Class" and node.name == "Solution" for node in graph.nodes.values())
    assert any(node.node_type == "Function" and node.name == "add" for node in graph.nodes.values())
    assert graph.metadata["structured_context"]["language"] == "python"
    assert any(item.get("event_type") == "InitialGraphBuilt" for item in graph.flow_history)


def test_state_graph_limits_rollback_snapshots():
    graph = StateFlowGraph(source_path="sample.py")

    for index in range(MAX_ROLLBACK_SNAPSHOTS + 4):
        graph.bump_version(f"change {index}")

    assert len(graph._snapshots) == MAX_ROLLBACK_SNAPSHOTS
    assert graph._snapshots[-1]["version"] == graph.version - 1


def test_java_structured_context_extractor(tmp_path):
    source = tmp_path / "Calculator.java"
    source.write_text(
        "package demo;\n"
        "import java.util.List;\n"
        "public class Calculator {\n"
        "  public int add(int a, int b) {\n"
        "    if (a > b) { return a + b; }\n"
        "    return b + a;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    context = ContextRewriter().rewrite(JavaStructuredContextExtractor().extract(source))
    graph = StructuredGraphBuilder(context).build()

    assert context.language == "java"
    assert context.package_or_module == "demo"
    assert context.classes[0].methods[0].qualified_name == "demo.Calculator.add"
    assert any(node.node_type == "Package" and node.name == "demo" for node in graph.nodes.values())
    assert any(node.node_type == "Function" and node.name == "add" for node in graph.nodes.values())


def test_java_tree_sitter_extracts_interfaces_generics_and_override(tmp_path):
    source = tmp_path / "Service.java"
    source.write_text(
        "package demo;\n"
        "interface Contract<T> { T apply(T value); }\n"
        "public class Service<T> implements Contract<T> {\n"
        "  @Override public T apply(T value) throws IllegalArgumentException {\n"
        "    if (value == null) { throw new IllegalArgumentException(); }\n"
        "    return value;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    context = JavaStructuredContextExtractor().extract(source)
    graph = StructuredGraphBuilder(context).build()

    assert context.rewrite_summary["parser"] == "tree_sitter_java"
    assert [item.kind for item in context.classes] == ["interface", "class"]
    service = context.classes[1]
    assert service.type_parameters == ["T"]
    assert service.methods[0].is_override
    assert service.methods[0].thrown_exceptions == ["IllegalArgumentException"]
    assert any(node.node_type == "Interface" for node in graph.nodes.values())
    assert any(node.node_type == "Override" for node in graph.nodes.values())


def test_hybrid_knowledge_retrieval_has_four_score_components(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    graph = StructuredGraphBuilder.from_source(source).build()
    memory_path = tmp_path / "experience.json"
    memory_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "experience_id": "exp-1",
                        "timestamp": time.time(),
                        "reward": 0.8,
                        "task_signature": AgentExperienceMemory(memory_path).task_signature(graph),
                        "learning_notes": [{"solution": "use exact integer boundary assertions"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = AgentExperienceMemory(memory_path, mirror_paths=[]).retrieve_hybrid(graph, query={"dominant_gap": "assertion"}, top_k=1)

    assert result[0]["experience_id"] == "exp-1"
    assert set(result[0]["retrieval_components"]) == {"semantic", "structural", "quality", "recency"}


def test_langgraph_workflow_contains_non_linear_three_agent_routes(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    system = MultiAgentUnitTestSystem(source, tmp_path / "out", max_iterations=1, retry_rounds=0)

    workflow = LangGraphTestWorkflow(system)
    drawable = workflow.graph.get_graph()
    nodes = set(drawable.nodes)
    edges = {(edge.source, edge.target) for edge in drawable.edges}

    assert {"test_state_bootstrap", "test_state_self_repair", "test_knowledge_retrieve", "test_generation_agent", "test_generation_finalize", "test_state_evaluate_plan", "test_knowledge_update"} <= nodes
    assert ("test_state_self_repair", "test_state_self_repair") in edges
    assert ("test_knowledge_update", "test_knowledge_retrieve") in edges
    workflow.close()


def test_test_state_agent_repairs_invalid_model_itself(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    system = MultiAgentUnitTestSystem(source, tmp_path / "out", max_iterations=1, retry_rounds=0)
    graph = StructuredGraphBuilder.from_source(source).build()
    graph.metadata.pop("structured_context", None)
    system.memory.set(graph)

    assert not system.test_state_agent.inspect().valid

    health = system.test_state_agent.self_repair(source, "structured context was missing")

    assert health.valid
    repaired = system.memory.load()
    assert repaired.metadata["test_state_self_repair"]["action"] == "rebuild_from_source"
    assert any(item.get("event_type") == "TestStateRepaired" for item in repaired.flow_history)


def test_java_junit_generation_from_structured_graph(tmp_path):
    source = tmp_path / "Calculator.java"
    source.write_text(
        "package demo;\n"
        "public class Calculator {\n"
        "  public int add(int a, int b) { return a + b; }\n"
        "}\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = StructuredGraphBuilder.from_source(source).build()
    memory.set(graph)

    test_path = TestGenerationAgent(memory).run(tmp_path / "tests")
    text = test_path.read_text(encoding="utf-8")

    assert test_path.name == "CalculatorTest.java"
    assert "package demo;" in text
    assert "import org.junit.jupiter.api.Test;" in text
    assert "new Calculator().add(" in text
    assert "assertEquals" in text


def test_java_junit_generation_sanitizes_final_array_and_static_methods(tmp_path):
    source = tmp_path / "Util.java"
    source.write_text(
        "package cli;\n"
        "public final class Util {\n"
        "  static boolean isEmpty(final Object[] array) { return array == null || array.length == 0; }\n"
        "  public static boolean isEmpty(final String str) { return str == null || str.isEmpty(); }\n"
        "  static String stripLeadingAndTrailingQuotes(final String str) { return str; }\n"
        "  private Util() {}\n"
        "}\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    graph = StructuredGraphBuilder.from_source(source).build()
    memory.set(graph)

    test_path = TestGenerationAgent(memory).run(tmp_path / "tests")
    text = test_path.read_text(encoding="utf-8")

    assert test_path.name == "UtilTest.java"
    assert "class UtilTest" in text
    assert "Util.isEmpty(new Object[]{})" in text
    assert "Util.isEmpty((Object[]) null)" in text
    assert "Util.isEmpty((String) null)" in text
    assert "Util.isEmpty(\"abc\")" in text
    assert 'assertEquals("\\\"", Util.stripLeadingAndTrailingQuotes("\\\""));' in text
    assert "new final object" not in text
    assert "new Util().isEmpty" not in text


def test_java_toolchain_selects_gradle(tmp_path):
    (tmp_path / "build.gradle").write_text("plugins { id 'java' }\n", encoding="utf-8")

    assert JavaToolchain()._build_system(tmp_path) == "gradle"


def test_maven_resolution_reports_missing_tool_without_fake_command(tmp_path, monkeypatch):
    monkeypatch.delenv("MAVEN_CMD", raising=False)
    monkeypatch.delenv("MVN_CMD", raising=False)
    monkeypatch.delenv("MAVEN_HOME", raising=False)
    monkeypatch.delenv("M2_HOME", raising=False)
    monkeypatch.setenv("PATH", "")

    resolution = resolve_maven_command(tmp_path)

    assert resolution.available is False
    assert resolution.command is None
    assert "Maven executable was not found" in resolution.message
    assert any(r"D:\wn\mavenEvo\apache-maven-3.9.1" in item for item in resolution.checked)


def test_maven_local_repo_stays_inside_project(tmp_path):
    arg = maven_local_repo_arg(tmp_path)

    assert arg.startswith("-Dmaven.repo.local=")
    assert str(tmp_path.resolve()) in arg
    assert (tmp_path / ".m2" / "repository").is_dir()


def test_maven_project_settings_uses_project_local_https_central(tmp_path):
    args = maven_project_settings_args(tmp_path)
    settings_path = tmp_path / ".m2" / "settings.xml"
    settings = settings_path.read_text(encoding="utf-8")

    assert args == ["-s", str(settings_path), "-gs", str(settings_path)]
    assert "https://repo.maven.apache.org/maven2" in settings
    assert str((tmp_path / ".m2" / "repository").resolve()).replace("\\", "/") in settings
    assert "maven.aliyun.com" not in settings


def test_project_dependencies_cover_bundled_python_datasets():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = project["dependencies"]

    assert any(item.startswith("sortedcontainers>=2.4") for item in dependencies)
    assert "sortedcontainers>=2.4,<3.0" in Path("依赖.txt").read_text(encoding="utf-8")


def test_java_execution_uses_java_toolchain_branch_without_build_file(tmp_path, monkeypatch):
    monkeypatch.setenv("STATEFLOW_JAVA_WORKSPACE", str(tmp_path / "missing_workspace"))
    source = tmp_path / "Calculator.java"
    source.write_text(
        "package demo;\n"
        "public class Calculator {\n"
        "  public int add(int a, int b) { return a + b; }\n"
        "}\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "CalculatorTest.java"
    test_path.write_text(
        "package demo;\n"
        "import org.junit.jupiter.api.Test;\n"
        "class CalculatorTest {\n"
        "  @Test void addWorks() { new Calculator().add(1, 1); }\n"
        "}\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    memory.set(StructuredGraphBuilder.from_source(source).build())

    report = ExecutionAgent(memory).run(test_path)
    graph = memory.load()

    assert report.failure_category == "java_execution_unavailable"
    assert graph.metadata["last_java_execution"]["build_system"] == "missing"
    assert graph.metadata["execution_diagnosis"]["recommended_focus"] == "java_project_toolchain"


def test_java_execution_stages_generated_test_inside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    source_dir = project / "src" / "main" / "java" / "demo"
    source_dir.mkdir(parents=True)
    source = source_dir / "Calculator.java"
    source.write_text(
        "package demo;\n"
        "public class Calculator {\n"
        "  public int add(int a, int b) { return a + b; }\n"
        "}\n",
        encoding="utf-8",
    )
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    test_path = generated_dir / "CalculatorTest.java"
    test_path.write_text(
        "package demo;\n"
        "import org.junit.jupiter.api.Test;\n"
        "class CalculatorTest {\n"
        "  @Test void addWorks() { new Calculator().add(1, 1); }\n"
        "}\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    memory.set(StructuredGraphBuilder.from_source(source).build())
    agent = ExecutionAgent(memory)

    class FakeJavaToolchain:
        def run(self, project_root, test_selector=None, target_class=None, timeout=120):
            staged = project_root / "src" / "test" / "java" / "demo" / "CalculatorTest.java"
            assert staged.exists()
            assert test_selector == "CalculatorTest"
            assert target_class == "demo.Calculator"
            return {
                "language": "java",
                "build_system": "maven",
                "passed": True,
                "junit": {"passed": True},
                "line_coverage": 0.5,
                "branch_coverage": 0.25,
                "method_coverage": 0.5,
                "mutation_score": 0.2,
                "jacoco": {"valid": True, "covered_lines": [3]},
                "pit": {"valid": True},
            }

        def _build_system(self, project_root):
            return "maven"

    agent.java_toolchain = FakeJavaToolchain()

    report = agent.run(test_path)
    graph = memory.load()

    assert report.pytest_passed
    assert report.collected_count == 1
    assert report.coverage_valid
    assert graph.metadata["last_java_execution"]["staged_test_path"].endswith("CalculatorTest.java")
    assert graph.metadata["last_test_usability"]["assertion_count"] == 0
    assert graph.metadata["last_test_usability"]["test_count"] == 1


def test_java_execution_prepares_bundled_workspace_from_dataset_source(tmp_path, monkeypatch):
    workspace = tmp_path / "java_workspace"
    (workspace / "src" / "main" / "java").mkdir(parents=True)
    (workspace / "src" / "test" / "java").mkdir(parents=True)
    (workspace / "pom.xml").write_text("<project></project>\n", encoding="utf-8")
    monkeypatch.setenv("STATEFLOW_JAVA_WORKSPACE", str(workspace))

    source_root = tmp_path / "datasets" / "Defects4J" / "cli"
    source_dir = source_root / "help"
    source_dir.mkdir(parents=True)
    source = source_dir / "Calculator.java"
    source.write_text(
        "package cli.help;\n"
        "public class Calculator {\n"
        "  public int add(int a, int b) { return a + b; }\n"
        "}\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "generated" / "CalculatorTest.java"
    test_path.parent.mkdir()
    test_path.write_text(
        "package cli.help;\n"
        "import org.junit.jupiter.api.Test;\n"
        "class CalculatorTest {\n"
        "  @Test void addWorks() { new Calculator().add(1, 1); }\n"
        "}\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    memory.set(StructuredGraphBuilder.from_source(source).build())
    agent = ExecutionAgent(memory)

    class FakeJavaToolchain:
        def run(self, project_root, test_selector=None, target_class=None, timeout=120):
            assert (project_root / "src" / "main" / "java" / "cli" / "help" / "Calculator.java").exists()
            assert (project_root / "src" / "test" / "java" / "cli" / "help" / "CalculatorTest.java").exists()
            assert target_class == "cli.help.Calculator"
            return {
                "language": "java",
                "build_system": "maven",
                "passed": True,
                "junit": {"passed": True},
                "line_coverage": 0.75,
                "branch_coverage": 0.5,
                "method_coverage": 1.0,
                "mutation_score": 0.25,
                "jacoco": {"valid": True, "covered_lines": [3]},
                "pit": {"valid": True},
            }

        def _build_system(self, project_root):
            return "maven"

    agent.java_toolchain = FakeJavaToolchain()

    report = agent.run(test_path)
    graph = memory.load()

    assert report.pytest_passed
    assert report.collected_count == 1
    assert graph.metadata["last_java_execution"]["prepared_source"]["target_package"] == "cli"


def test_jacoco_runner_filters_metrics_to_target_class(tmp_path):
    csv_path = tmp_path / "jacoco.csv"
    csv_path.write_text(
        "GROUP,PACKAGE,CLASS,LINE_MISSED,LINE_COVERED,BRANCH_MISSED,BRANCH_COVERED,METHOD_MISSED,METHOD_COVERED\n"
        "demo,demo,Calculator,1,9,1,3,0,2\n"
        "demo,demo,Helper,100,0,20,0,10,0\n",
        encoding="utf-8",
    )

    metrics = JacocoRunner()._parse_csv(csv_path, target_class="demo.Calculator")

    assert metrics["line_coverage"] == 0.9
    assert metrics["branch_coverage"] == 0.75
    assert metrics["method_coverage"] == 1.0
    assert metrics["matched_class_rows"] == 1


def test_pit_runner_parses_target_mutation_evidence(tmp_path):
    report_dir = tmp_path / "target" / "pit-reports"
    report_dir.mkdir(parents=True)
    (report_dir / "mutations.xml").write_text(
        "<mutations>"
        "<mutation detected='true' status='KILLED'><mutatedClass>demo.Calculator</mutatedClass><mutatedMethod>add</mutatedMethod><lineNumber>3</lineNumber><mutator>x.NegateConditionalsMutator</mutator><killingTest>demo.CalculatorTest.addWorks</killingTest><description>negated</description></mutation>"
        "<mutation detected='false' status='SURVIVED'><mutatedClass>demo.Calculator</mutatedClass><mutatedMethod>add</mutatedMethod><lineNumber>4</lineNumber><mutator>x.MathMutator</mutator><killingTest/><description>changed math</description></mutation>"
        "<mutation detected='true' status='KILLED'><mutatedClass>demo.Helper</mutatedClass><mutatedMethod>run</mutatedMethod><lineNumber>9</lineNumber><mutator>x.VoidMethodCallMutator</mutator><killingTest>demo.HelperTest.run</killingTest><description>removed call</description></mutation>"
        "</mutations>",
        encoding="utf-8",
    )

    result = PitRunner()._results_from_xml(tmp_path, target_class="demo.Calculator")

    assert result["total"] == 2
    assert result["killed"] == 1
    assert result["survived"] == 1
    assert result["effective_mutants"] == 2
    assert result["score"] == 0.5
    assert result["killed_lines"] == [3]
    assert result["survived_lines"] == [4]
    assert result["details"][0]["mutator"] == "NegateConditionalsMutator"


def test_java_execution_records_stateflow_evidence(tmp_path):
    source = tmp_path / "Calculator.java"
    source.write_text(
        "package demo;\n"
        "public class Calculator {\n"
        "  public static int add(int a, int b) { return a + b; }\n"
        "}\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "CalculatorTest.java"
    test_path.write_text(
        "package demo;\n"
        "import org.junit.jupiter.api.Test;\n"
        "import static org.junit.jupiter.api.Assertions.assertEquals;\n"
        "class CalculatorTest {\n"
        "  @Test void addWorks() { assertEquals(2, Calculator.add(1, 1)); }\n"
        "}\n",
        encoding="utf-8",
    )
    memory = SharedGraphMemory(tmp_path / "graph.json")
    memory.set(StructuredGraphBuilder.from_source(source).build())
    agent = ExecutionAgent(memory)

    class FakeJavaToolchain:
        def run(self, project_root, test_selector=None, target_class=None, timeout=120):
            return {
                "language": "java",
                "build_system": "maven",
                "passed": True,
                "junit": {"passed": True},
                "line_coverage": 1.0,
                "branch_coverage": 1.0,
                "method_coverage": 1.0,
                "mutation_score": 1.0,
                "jacoco": {"valid": True, "covered_lines": [3]},
                "pit": {
                    "valid": True,
                    "mutation_score": 1.0,
                    "total": 4,
                    "effective_mutants": 4,
                    "killed": 3,
                    "survived": 1,
                    "score_reliability": "normal",
                    "killed_lines": [3],
                    "survived_lines": [3],
                    "details": [
                        {
                            "status": "killed",
                            "category": "killed",
                            "line": 3,
                            "class": "demo.Calculator",
                            "method": "add",
                            "mutator": "MathMutator",
                            "killing_test": "demo.CalculatorTest.addWorks",
                        },
                        {
                            "status": "survived",
                            "category": "survived",
                            "line": 3,
                            "class": "demo.Calculator",
                            "method": "add",
                            "mutator": "ConditionalsBoundaryMutator",
                            "killing_test": "",
                        },
                    ],
                },
            }

        def _build_system(self, project_root):
            return "maven"

    agent.java_toolchain = FakeJavaToolchain()

    report = agent.run(test_path)
    graph = memory.load()

    assert report.sfc > 0
    assert report.ae > 0
    assert report.collected_count == 1
    assert report.effective_mutants == 4
    assert report.mutation_reliability == "normal"
    assert graph.metadata["last_test_usability"]["assertion_count"] == 1
    assert graph.metadata["last_test_usability"]["test_count"] == 1
    assert graph.metadata["last_mutation"]["effective_mutants"] == 4
    assert graph.metadata["last_mutation"]["evidence_source"] == "pit"
    assert any(node.node_type == "Execution" and node.metadata.get("language") == "java" for node in graph.nodes.values())
    assert any(node.node_type == "TestCase" and node.metadata.get("language") == "java" for node in graph.nodes.values())
    assert any(node.node_type == "Assertion" and node.metadata.get("language") == "java" for node in graph.nodes.values())
    assert any(node.node_type == "Mutation" and node.metadata.get("language") == "java" for node in graph.nodes.values())
    assert any(node.node_type == "Function" and node.mutation_status == "survived" for node in graph.nodes.values())
    assert any(edge.edge_type == "Observe" for edge in graph.edges)
    assert any(edge.edge_type == "Kill" for edge in graph.edges)
    assert any(edge.edge_type == "RepairTarget" for edge in graph.edges)
    event_types = {item.get("event_type") for item in graph.flow_history}
    assert {"ExecutionObserved", "CoverageObserved", "MutationObserved"} <= event_types


def test_java_workspace_pom_contains_required_dataset_dependencies():
    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
    root = ET.parse("java_workspace/pom.xml").getroot()
    dependencies = {
        (
            dependency.findtext("m:groupId", default="", namespaces=namespace),
            dependency.findtext("m:artifactId", default="", namespaces=namespace),
        )
        for dependency in root.findall(".//m:dependency", namespace)
    }

    assert ("org.junit.jupiter", "junit-jupiter") in dependencies
    assert ("org.apache.commons", "commons-lang3") in dependencies
    assert ("org.apache.maven.surefire", "surefire-shared-utils") in dependencies
    assert ("br.usp.each.saeg", "saeg-commons") in dependencies
    assert ("com.google.errorprone", "error_prone_annotations") in dependencies
    assert ("com.google.code.gson", "gson") in dependencies
    pit_plugin_dependencies = {
        (
            dependency.findtext("m:groupId", default="", namespaces=namespace),
            dependency.findtext("m:artifactId", default="", namespaces=namespace),
        )
        for dependency in root.findall(".//m:plugin[m:artifactId='pitest-maven']/m:dependencies/m:dependency", namespace)
    }
    assert ("org.pitest", "pitest-junit5-plugin") in pit_plugin_dependencies
    pit_params = {
        param.text
        for param in root.findall(
            ".//m:plugin[m:artifactId='pitest-maven']//m:targetTests/m:param",
            namespace,
        )
    }
    assert "gson.*.*.*.*Test" in pit_params
    assert "lang3.*.*.*.*Test" in pit_params
    assert "cli.*.*.*Test" in pit_params


def test_dynamic_validation_and_complete_state_graph(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    test_path = tmp_path / "test_sample.py"
    test_path.write_text("def test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")
    graph = StructuredGraphBuilder.from_source(source).build()
    for node in list(graph.nodes.values()):
        if node.node_type in {"Function", "Return"}:
            node.visited = True
        if node.node_type == "Function":
            node.assert_count = 1
            runtime_type = graph.add_entity("RuntimeType", "add return int", metadata={"type": "int"})
            graph.add_relation(node.node_id, runtime_type.node_id, "TypeFlow")
    graph.metadata["last_test_usability"] = {"assertion_count": 1}
    graph.metadata["last_report"] = {
        "pytest_passed": True,
        "line_coverage": 1.0,
        "branch_coverage": 1.0,
        "coverage_percent": 1.0,
        "sfc": 1.0,
        "ae": 1.0,
        "mutation_score": 1.0,
    }
    graph.metadata["last_test_intent_plan"] = {"target_states": [node.node_id for node in graph.nodes.values() if node.node_type == "Function"]}

    validation = DynamicValidator().validate(graph, test_path, graph.metadata["structured_context"])
    completeness = CompleteStateGraphMetric(threshold=0.8).calculate(graph, validation_report=validation)

    assert validation.passed
    assert completeness.complete
    assert completeness.csg >= 0.8


def test_minimal_target_repair_uses_validation_state_flow(tmp_path):
    graph = SimpleNamespace(
        metadata={
            "unresolved_validation_errors": [
                {
                    "code": "assertion_missing",
                    "message": "missing assertion",
                    "related_nodes": ["program:function:add"],
                }
            ],
            "complete_state_graph": {"csg": 0.45},
            "last_report": {"failed_tests": []},
        }
    )

    decision = MinimalTargetRepair().locate(graph, tmp_path / "test_sample.py", [])

    assert decision.scope.level == "assertion-level"
    assert decision.utility > 0
