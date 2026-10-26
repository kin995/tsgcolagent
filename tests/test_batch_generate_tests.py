from __future__ import annotations

import json
from pathlib import Path

import batch_generate_tests as batch_module
from batch_generate_tests import _batch_quality_overview, _discover_sources, generate_tests_for_directory
from batch_generate_tests import BatchItemResult
from agents.execution_agent import ExecutionReport
from main import MultiAgentUnitTestSystem, Thresholds
from run_agent import run_target
from visualization.graph_visualizer import StateFlowGraphVisualizer
from visualization.knowledge_library_visualizer import TestKnowledgeGuidanceLibraryVisualizer
from graph.experience_memory import AgentExperienceMemory


def test_only_humanevaljava_classes_get_two_default_workers(tmp_path):
    humaneval = tmp_path / "HumanEvalJava"
    project = tmp_path / "ProjectJava"
    python_dir = tmp_path / "HumanEval"
    for directory in (humaneval, project, python_dir):
        directory.mkdir()
    java_tasks = [humaneval / f"Task{index}.java" for index in range(3)]
    project_tasks = [project / f"Task{index}.java" for index in range(3)]
    python_tasks = [python_dir / f"task_{index:03d}.py" for index in range(5)]

    assert batch_module._effective_worker_count("java", java_tasks, None, False) == 2
    assert batch_module._effective_worker_count("java", java_tasks, 9, False) == 3
    assert batch_module._effective_worker_count("java", project_tasks, None, False) == 1
    assert batch_module._effective_worker_count("python", python_tasks, None, False) == 4


def test_humanevaljava_round_knowledge_is_frozen_then_merged(tmp_path, monkeypatch):
    live_path = tmp_path / "knowledge.json"
    live_path.write_text(json.dumps({"records": [{"experience_id": "base", "timestamp": 1.0, "topic": "base"}]}), encoding="utf-8")
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(live_path))
    output = tmp_path / "runs" / "HumanEvalJava"
    output.mkdir(parents=True)
    case_a = output / "A"
    case_b = output / "B"
    case_a.mkdir()
    case_b.mkdir()
    isolation = batch_module._prepare_round_knowledge_isolation(output, experiment_round=1, enabled=True)
    assert isolation is not None

    memory_a = AgentExperienceMemory(
        live_path,
        read_base_path=isolation.base_path,
        journal_dir=isolation.journal_dir(case_a),
        publish_canonical=False,
    )
    memory_b = AgentExperienceMemory(
        live_path,
        read_base_path=isolation.base_path,
        journal_dir=isolation.journal_dir(case_b),
        publish_canonical=False,
    )
    memory_a.append_conversation_concept("怎样提高JUnit覆盖率？", "补充分支输入", model="test", conversation_id="a", force=True)
    assert all(record.get("conversation_id") != "a" for record in memory_b.load())
    memory_b.append_conversation_concept("怎样提高PIT变异率？", "补充精确断言", model="test", conversation_id="b", force=True)

    evidence = batch_module._merge_round_knowledge(isolation, [case_a, case_b])
    merged = AgentExperienceMemory(live_path).load()

    assert evidence["merged_records"] == 2
    assert {record.get("conversation_id") for record in merged} >= {"a", "b"}
    assert evidence["experiment_round"] == 1


def test_batch_generate_tests_for_directory(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "task_000.py").write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n",
        encoding="utf-8",
    )
    (source_dir / "task_001.py").write_text(
        "def is_positive(value: int) -> bool:\n"
        "    return value > 0\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "generated"
    summary = generate_tests_for_directory(source_dir=source_dir, output_dir=output_dir)

    assert summary["total"] == 2
    assert summary["succeeded"] == 2
    assert (output_dir / "batch_summary.json").exists()
    assert (output_dir / "task_000" / "test_task_000_stateflow.py").exists()
    assert (output_dir / "task_001" / "state_flow_graph.json").exists()


def test_short_launcher_rejects_missing_target(tmp_path):
    missing = tmp_path / "missing.py"
    try:
        run_target(missing, output_root=tmp_path / "out")
    except FileNotFoundError as exc:
        assert "Target not found" in str(exc)
    else:
        raise AssertionError("run_target should reject missing targets")


def test_batch_filters_package_files_and_sorts_tasks_numerically(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "__init__.py").write_text("", encoding="utf-8")
    (source_dir / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    for name in ["task_010.py", "task_002.py", "task_001.py"]:
        (source_dir / name).write_text("def public(value=1):\n    return value\n", encoding="utf-8")

    sources, skipped, discovered = _discover_sources(source_dir, pattern="task_*.py", recursive=False)

    assert discovered == 5
    assert [path.name for path in sources] == ["task_001.py", "task_002.py", "task_010.py"]
    assert any(item["reason"] == "package_initializer" for item in skipped)
    assert any(item["reason"] == "pattern_mismatch" for item in skipped)


def test_batch_discovery_supports_java_sources(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "Calculator.java").write_text(
        "package demo;\n"
        "public class Calculator {\n"
        "  public int add(int a, int b) { return a + b; }\n"
        "}\n",
        encoding="utf-8",
    )
    (source_dir / "Notes.txt").write_text("not a source file\n", encoding="utf-8")

    sources, skipped, discovered = _discover_sources(source_dir, pattern="*.java", recursive=False)

    assert discovered == 1
    assert [path.name for path in sources] == ["Calculator.java"]
    assert skipped == []


def test_batch_auto_discovers_generic_python_folder(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")

    sources, skipped, discovered = _discover_sources(source_dir, pattern="auto", recursive=False)

    assert discovered == 1
    assert [path.name for path in sources] == ["calculator.py"]
    assert skipped == []


def test_batch_auto_discovers_java_folder_with_default_pattern(tmp_path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "Calculator.java").write_text(
        "public class Calculator { public int add(int a, int b) { return a + b; } }\n",
        encoding="utf-8",
    )

    sources, skipped, discovered = _discover_sources(source_dir, pattern=batch_module.DEFAULT_TASK_PATTERN, recursive=False)

    assert discovered == 1
    assert [path.name for path in sources] == ["Calculator.java"]
    assert skipped == []


def test_batch_java_generate_auto_recurses_project_tree(tmp_path):
    source_dir = tmp_path / "java_project"
    package_dir = source_dir / "src" / "main" / "java" / "demo"
    package_dir.mkdir(parents=True)
    (package_dir / "Calculator.java").write_text(
        "package demo;\n"
        "public class Calculator { public int add(int a, int b) { return a + b; } }\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "generated"

    summary = generate_tests_for_directory(source_dir=source_dir, output_dir=output_dir, limit=1)

    assert summary["effective_pattern"] == "*.java"
    assert summary["recursive"] is True
    assert summary["discovered_files"] == 1
    assert summary["succeeded"] == 1


def test_batch_java_project_discovery_skips_tests_and_build_outputs(tmp_path):
    source_dir = tmp_path / "java_project"
    main_dir = source_dir / "src" / "main" / "java" / "demo"
    test_dir = source_dir / "src" / "test" / "java" / "demo"
    target_dir = source_dir / "target" / "generated-sources" / "demo"
    output_dir = source_dir / "generated_tests_llm" / "demo"
    for directory in [main_dir, test_dir, target_dir, output_dir]:
        directory.mkdir(parents=True)
    (main_dir / "Calculator.java").write_text(
        "package demo;\n"
        "public class Calculator { public int add(int a, int b) { return a + b; } }\n",
        encoding="utf-8",
    )
    for directory, name in [
        (test_dir, "CalculatorTest.java"),
        (target_dir, "GeneratedCalculator.java"),
        (output_dir, "OldGeneratedTest.java"),
    ]:
        (directory / name).write_text(
            "package demo;\n"
            "public class Ignored { public void helper() {} }\n",
            encoding="utf-8",
        )

    sources, skipped, discovered = _discover_sources(source_dir, pattern="*.java", recursive=True, language="java")

    assert discovered == 4
    assert [path.name for path in sources] == ["Calculator.java"]
    assert sum(1 for item in skipped if item["reason"] == "java_non_main_source") == 3


def test_batch_quality_overview_reports_assertions_and_mutation_reason(tmp_path):
    task_report = tmp_path / "task_000_report.html"
    task_report.write_text("<html>single task</html>", encoding="utf-8")
    report = {
        "pytest_passed": True,
        "line_coverage": 0.8,
        "branch_coverage": 0.5,
        "sfc": 0.7,
        "ae": 0.6,
        "mutation_score": 0.4,
        "sfq": 0.62,
        "assertion_count": 3,
        "weak_assertions": 1,
        "effective_assertions": 2,
        "effective_mutants": 1,
        "mutation_low_sample_reason": "源码只发现 1 个可变异位置",
    }
    overview = _batch_quality_overview([
        type("Item", (), {"final_report": report, "source": "task_000.py"})()
    ])
    html = StateFlowGraphVisualizer().render_batch_summary(
        {
            "results": [{
                "source": "task_000.py",
                "status": "valid_usable",
                "final_report": report,
                "paper_metrics": {"tir": 0.5, "avg_exec": 4.0, "duration_seconds": 9.0},
                "visualization": {"graph_html": str(task_report)},
            }],
            "succeeded": 1,
            "failed": 0,
            "quality_overview": overview,
            "paper_metrics": {"pass_rate": 1.0, "tir": 0.5, "tir_improved": 1,
                              "tir_comparable": 2, "tir_excluded": 0, "avg_exec": 4.0,
                              "average_time_seconds": 9.0, "exec_complete_tasks": 1,
                              "selected_tasks": 1},
        },
        tmp_path,
    )
    text = (tmp_path / "batch_quality_report.html").read_text(encoding="utf-8")

    assert html["status"] == "generated"
    assert overview["total_effective_assertions"] == 2
    assert "有效断言总数" in text
    assert "源码只发现 1 个可变异位置" in text
    assert "进入后续轮次的任务数" in text
    assert "参与二/三轮的唯一任务" not in text
    assert "单任务报告" in text
    assert "TIR" in text and "50%" in text
    assert "AvgExec" in text and "4.00" in text
    assert "SFC" not in text and "TSQ" not in text
    assert task_report.name in text


def test_batch_summary_counts_passed_plateau_as_c_when_report_is_usable(tmp_path):
    summary = batch_module._build_batch_summary(
        source_root=tmp_path,
        output_root=tmp_path / "out",
        pattern="task_*.py",
        effective_pattern="task_*.py",
        language="python",
        recursive=False,
        execute=True,
        llm_config=batch_module.LLMConfig(enabled=False),
        started_at="2026-07-31T00:00:00+00:00",
        discovered_count=1,
        official_tasks=1,
        selected_count=1,
        skipped_sources=[],
        results=[BatchItemResult(source="task_000.py", output_dir="out", status="plateau_reached", final_report={"pytest_passed": True})],
        retry_history=[],
        resumed_count=0,
    )

    assert summary["succeeded"] == 1
    assert summary["incomplete"] == 0
    assert summary["outcome_summary"]["grade_counts"]["C"] == 1


def test_batch_summary_treats_valid_usable_as_weak_success(tmp_path):
    summary = batch_module._build_batch_summary(
        source_root=tmp_path,
        output_root=tmp_path / "out",
        pattern="task_*.py",
        effective_pattern="task_*.py",
        language="python",
        recursive=False,
        execute=True,
        llm_config=batch_module.LLMConfig(enabled=False),
        started_at="2026-07-31T00:00:00+00:00",
        discovered_count=1,
        official_tasks=1,
        selected_count=1,
        skipped_sources=[],
        results=[
            BatchItemResult(
                source="task_000.py",
                output_dir="out",
                status="valid_usable",
                final_report={
                    "pytest_passed": True,
                    "weak_quality_flags": ["weak_line_coverage"],
                    "threshold_failures": {"line_coverage_gap": 0.2},
                },
            )
        ],
        retry_history=[],
        resumed_count=0,
    )

    assert summary["succeeded"] == 1
    assert summary["weak_success"] == 1
    assert summary["incomplete"] == 0
    assert summary["outcome_summary"]["grade_counts"] == {"A": 0, "B": 0, "C": 1, "D": 0, "E": 0}
    assert summary["weak_success_analysis"]["weak_quality_flag_counts"] == {"weak_line_coverage": 1}


def test_thresholds_treat_runnable_weak_metrics_as_valid_usable():
    report = ExecutionReport(
        pytest_passed=True,
        coverage_percent=0.0,
        mutation_score=None,
        sfc=0.0,
        ae=0.0,
        sfq=0.0,
        test_path="ExampleTest.java",
        coverage_valid=False,
        mutation_valid=False,
        collected_count=1,
        normal_behavior_tests=1,
        assertion_count=0,
        effective_assertions=0,
    )
    thresholds = Thresholds()

    assert not thresholds.usable(report)
    assert thresholds.classify(report) == "execution_invalid"
    assert set(thresholds.weak_quality_flags(report)) >= {"coverage_unavailable", "mutation_unavailable", "weak_assertions"}


def test_batch_status_keeps_passed_plateau_as_weak_success(tmp_path):
    summary = {
        "final_report": {"pytest_passed": True, "quality_status": "valid_usable"},
        "quality_status": "valid_usable",
        "stop_reason": "plateau_reached",
    }

    assert batch_module._status_from_task_summary(summary, execute=True) == "valid_usable"


def test_batch_retries_failed_tasks_after_writing_experience(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    output_dir = tmp_path / "generated"
    calls: list[str] = []

    def fake_generate_and_execute(**kwargs):
        assert "retry_rounds" in kwargs
        calls.append(str(kwargs["source_path"]))
        if len(calls) == 1:
            return BatchItemResult(
                source=str(source_path),
                output_dir=str(output_dir / "task_000"),
                status="execution_invalid",
                final_report={"pytest_passed": False, "timed_out": True, "failure_category": "timeout"},
            )
        return BatchItemResult(
            source=str(source_path),
            output_dir=str(output_dir / "task_000"),
            status="quality_high",
            final_report={
                "pytest_passed": True,
                "line_coverage": 1.0,
                "branch_coverage": 1.0,
                "sfc": 1.0,
                "ae": 1.0,
                "mutation_score": 1.0,
                "sfq": 1.0,
                "effective_assertions": 1,
                "weak_assertions": 0,
                "effective_mutants": 3,
                "mutation_reliability": "normal",
            },
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", fake_generate_and_execute)

    summary = generate_tests_for_directory(
        source_dir=source_dir,
        output_dir=output_dir,
        execute=True,
        retry_failed_rounds=1,
    )
    knowledge_path = batch_module.resolve_knowledge_store_path()
    records = json.loads(knowledge_path.read_text(encoding="utf-8"))["records"]

    assert len(calls) == 2
    assert summary["experiment_round_summary"]["by_round"]["R2"] == {"tasks": 1, "recovered": 1, "improved": 1}
    assert summary["experiment_round_summary"]["followup_task_count"] == 1
    assert summary["succeeded"] == 1
    assert summary["failed"] == 0
    assert any(record.get("failure_categories", {}).get("timeout") == 1 for record in records)


def test_c_grade_r3_optimizes_existing_test_instead_of_regenerating(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_002.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    output_dir = tmp_path / "generated"
    case_dir = output_dir / "task_002"
    test_path = case_dir / "test_task_002_stateflow.py"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(batch_module, "resolve_knowledge_store_path", lambda: tmp_path / "knowledge.json")

    def fake_generate_and_execute(**kwargs):
        calls.append(kwargs)
        case_dir.mkdir(parents=True, exist_ok=True)
        test_path.write_text("def test_public(target):\n    assert target.public(1) == 1\n", encoding="utf-8")
        if len(calls) == 1:
            return BatchItemResult(
                source=str(source_path), output_dir=str(case_dir), status="valid_usable", test_path=str(test_path),
                final_report={"pytest_passed": True, "quality_status": "valid_usable", "line_coverage": 0.7, "ae": 0.9, "mutation_score": 0.25, "threshold_failures": {"mutation": 0.6}},
            )
        assert kwargs["followup_strategy"] == "optimize_existing"
        assert kwargs["existing_test_path"] == str(test_path)
        assert kwargs["baseline_report"]["pytest_passed"] is True
        return BatchItemResult(
            source=str(source_path), output_dir=str(case_dir), status="quality_acceptable", test_path=str(test_path),
            final_report={"pytest_passed": True, "quality_status": "quality_acceptable", "line_coverage": 0.9, "ae": 0.92, "mutation_score": 0.8},
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", fake_generate_and_execute)

    summary = generate_tests_for_directory(source_dir=source_dir, output_dir=output_dir, execute=True, retry_failed_rounds=2)
    progress = json.loads((case_dir / "experiment_round_progress.json").read_text(encoding="utf-8"))

    assert len(calls) == 2
    assert summary["results"][0]["grade_label"] == "B·R3"
    assert summary["results"][0]["round_strategy"] == "optimize_existing"
    assert progress["experiment_round"] == 3
    assert progress["status"] == "completed"


def test_batch_resume_skips_existing_completed_task(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    output_dir = tmp_path / "generated"
    case_dir = output_dir / "task_000"
    case_dir.mkdir(parents=True)
    summary_payload = {
        "source": str(source_path.resolve()),
        "test_path": str((case_dir / "test_task_000_stateflow.py").resolve()),
        "graph_path": str((case_dir / "state_flow_graph.json").resolve()),
        "final_report": {
            "pytest_passed": True,
            "line_coverage": 1.0,
            "branch_coverage": 1.0,
            "sfc": 1.0,
            "ae": 1.0,
            "mutation_score": 1.0,
            "sfq": 1.0,
            "effective_assertions": 1,
            "weak_assertions": 0,
            "effective_mutants": 3,
            "mutation_reliability": "normal",
        },
        "quality_status": "quality_high",
        "stop_reason": "quality_high",
    }
    (case_dir / "stateflow_summary.json").write_text(json.dumps(summary_payload), encoding="utf-8")

    def fail_if_called(**kwargs):
        raise AssertionError("completed task should be resumed instead of rerun")

    monkeypatch.setattr(batch_module, "_generate_and_execute", fail_if_called)

    summary = generate_tests_for_directory(source_dir=source_dir, output_dir=output_dir, execute=True)
    terminal = batch_module._terminal_summary(summary)

    assert summary["resumed_existing"] == 1
    assert summary["succeeded"] == 1
    assert terminal["metrics"]["average_line_coverage"] == 1.0


def test_batch_resume_reruns_existing_invalid_task(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    output_dir = tmp_path / "generated"
    case_dir = output_dir / "task_000"
    case_dir.mkdir(parents=True)
    invalid_payload = {
        "source": str(source_path.resolve()),
        "test_path": str((case_dir / "test_task_000_stateflow.py").resolve()),
        "graph_path": str((case_dir / "state_flow_graph.json").resolve()),
        "stop_reason": "generation_failed",
        "final_report": {"pytest_passed": False, "failure_category": "runtime_error"},
    }
    (case_dir / "stateflow_summary.json").write_text(json.dumps(invalid_payload), encoding="utf-8")
    calls: list[str] = []

    def fake_generate_and_execute(**kwargs):
        calls.append(str(kwargs["source_path"]))
        return BatchItemResult(
            source=str(source_path),
            output_dir=str(case_dir),
            status="quality_high",
            final_report={
                "pytest_passed": True,
                "line_coverage": 1.0,
                "branch_coverage": 1.0,
                "sfc": 1.0,
                "ae": 1.0,
                "mutation_score": 1.0,
                "sfq": 1.0,
                "effective_assertions": 1,
                "weak_assertions": 0,
                "effective_mutants": 3,
                "mutation_reliability": "normal",
            },
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", fake_generate_and_execute)

    summary = generate_tests_for_directory(source_dir=source_dir, output_dir=output_dir, execute=True)

    assert len(calls) == 1
    assert summary["resumed_existing"] == 0
    assert summary["succeeded"] == 1


def test_batch_failure_writes_per_task_summary_and_html(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    output_dir = tmp_path / "generated"

    def fail_generation(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(batch_module, "_generate_and_execute", fail_generation)

    summary = generate_tests_for_directory(
        source_dir=source_dir,
        output_dir=output_dir,
        execute=True,
        retry_failed_rounds=0,
    )
    item = summary["results"][0]

    assert summary["failed"] == 1
    assert Path(item["summary_path"]).exists()
    assert Path(item["visualization"]["failure_report_html"]).exists()
    assert "boom" in Path(item["summary_path"]).read_text(encoding="utf-8")


def test_single_task_resume_returns_existing_summary(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    existing = {
        "source": str(source.resolve()),
        "test_path": str((output_dir / "test_sample_stateflow.py").resolve()),
        "graph_path": str((output_dir / "state_flow_graph.json").resolve()),
        "stop_reason": "quality_high",
        "final_report": {"pytest_passed": True, "line_coverage": 1.0},
    }
    (output_dir / "stateflow_summary.json").write_text(json.dumps(existing), encoding="utf-8")

    summary = MultiAgentUnitTestSystem(
        source_path=source,
        output_dir=output_dir,
        thresholds=Thresholds(),
    ).run()

    assert summary["resumed_existing"] is True
    assert summary["stop_reason"] == "quality_high"


def test_three_round_policy_recovers_de_in_r2_then_optimizes_c_in_r3(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    calls = 0

    def fake_generate_and_execute(**kwargs):
        nonlocal calls
        calls += 1
        test_path = Path(kwargs["case_dir"]) / "test_task_000_stateflow.py"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text("def test_public(target):\n    assert target.public(1) == 1\n", encoding="utf-8")
        if calls == 2:
            assert kwargs["experiment_round"] == 2
            assert kwargs["followup_strategy"] == "regenerate_after_failure"
        if calls == 3:
            assert kwargs["experiment_round"] == 3
            assert kwargs["followup_strategy"] == "optimize_existing"
        status = ["execution_invalid", "valid_usable", "quality_acceptable"][calls - 1]
        passed = status != "execution_invalid"
        return BatchItemResult(
            source=str(source_path),
            output_dir=str(kwargs["case_dir"]),
            status=status,
            test_path=str(test_path),
            final_report={
                "pytest_passed": passed,
                "failure_category": None if passed else "runtime_error",
                "line_coverage": 0.9 if passed else 0.0,
                "sfc": 0.8 if passed else 0.0,
                "ae": 0.9 if passed else 0.0,
                "sfq": 0.85 if passed else 0.0,
            },
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", fake_generate_and_execute)
    summary = generate_tests_for_directory(source_dir, tmp_path / "out", execute=True)
    item = summary["results"][0]

    assert calls == 3
    assert item["grade_label"] == "B·R3"
    assert [entry["grade"] for entry in item["grade_history"]] == ["D", "C", "B"]
    assert summary["outcome_summary"]["grade_counts"]["B"] == 1
    assert summary["outcome_summary"]["balance_valid"] is True


def test_r3_failed_optimization_preserves_best_c_artifact(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public(value=1):\n    return value\n", encoding="utf-8")
    calls = 0

    def fake_generate_and_execute(**kwargs):
        nonlocal calls
        calls += 1
        test_path = Path(kwargs["case_dir"]) / "test_task_000_stateflow.py"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text("# best-c\n" if calls == 1 else "# broken-r3\n", encoding="utf-8")
        if calls == 1:
            return BatchItemResult(
                source=str(source_path), output_dir=str(kwargs["case_dir"]), status="valid_usable",
                test_path=str(test_path), final_report={"pytest_passed": True, "sfq": 0.70, "ae": 0.8, "sfc": 0.7},
            )
        return BatchItemResult(
            source=str(source_path), output_dir=str(kwargs["case_dir"]), status="execution_invalid",
            test_path=str(test_path), final_report={"pytest_passed": False, "failure_category": "assertion_failure"},
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", fake_generate_and_execute)
    summary = generate_tests_for_directory(source_dir, tmp_path / "out", execute=True)

    assert calls == 2
    assert summary["results"][0]["grade_label"] == "C·R3"
    assert (tmp_path / "out" / "task_000" / "test_task_000_stateflow.py").read_text(encoding="utf-8") == "# best-c\n"


def test_completed_dataset_directory_is_skipped_without_source_traversal(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "task_000.py").write_text("def public():\n    return 1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    first = generate_tests_for_directory(source_dir, output_dir, execute=False)
    assert first["dataset_completion"]["skip_ready"] is True

    def fail_discovery(*args, **kwargs):
        raise AssertionError("completed dataset must be skipped before source traversal")

    monkeypatch.setattr(batch_module, "_discover_sources", fail_discovery)
    second = generate_tests_for_directory(source_dir, output_dir, execute=False)

    assert second["dataset_skipped"] is True
    assert second["dataset_skip_reason"]


def test_final_de_task_has_reason_solutions_and_failure_report(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_000.py"
    source_path.write_text("def public():\n    return 1\n", encoding="utf-8")

    calls = 0

    def always_fail(**kwargs):
        nonlocal calls
        calls += 1
        return BatchItemResult(
            source=str(source_path), output_dir=str(kwargs["case_dir"]), status="execution_invalid",
            error="runtime_error: generated test crashed",
            final_report={"pytest_passed": False, "failure_category": "runtime_error", "error": "generated test crashed"},
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", always_fail)
    summary = generate_tests_for_directory(source_dir, tmp_path / "out", execute=True)
    item = summary["results"][0]

    assert calls == 2
    assert item["grade_label"] == "D·R2"
    assert "runtime_error" in item["failure_reason"]
    assert item["previous_solutions"]
    assert Path(item["failure_report"]).exists()
    report_text = Path(item["failure_report"]).read_text(encoding="utf-8")
    assert "具体失败原因" in report_text
    assert "之前采用或建议的解决措施" in report_text


def test_first_round_e_writes_failure_diagnostic_before_retry(tmp_path, monkeypatch):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    source_path = source_dir / "task_098.py"
    source_path.write_text("def public():\n    return 1\n", encoding="utf-8")
    knowledge_path = tmp_path / "knowledge.json"
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(knowledge_path))

    def generation_failed(**kwargs):
        case_dir = kwargs["case_dir"]
        case_dir.mkdir(parents=True, exist_ok=True)
        summary_path = case_dir / "stateflow_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "source": str(source_path),
                    "stop_reason": "generation_failed",
                    "last_error": "candidate_rejected: generated candidate has no normal behavior assertion",
                    "failure_evidence": {
                        "stage": "semantic_usability",
                        "candidate_validation": {
                            "accepted": False,
                            "stage": "semantic_usability",
                            "failure_category": "candidate_rejected",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        return BatchItemResult(
            source=str(source_path),
            output_dir=str(case_dir),
            status="generation_failed",
            summary_path=str(summary_path),
        )

    monkeypatch.setattr(batch_module, "_generate_and_execute", generation_failed)
    summary = generate_tests_for_directory(
        source_dir,
        tmp_path / "out",
        execute=True,
        retry_failed_rounds=0,
    )
    item = summary["results"][0]

    assert item["grade_label"] == "E·R1"
    assert item["round_failure_diagnostic"]["failure_category"] == "no_normal_behavior"
    assert "normal behavior assertion" in item["failure_reason"]
    assert item["previous_solutions"]
    assert Path(item["output_dir"], "task_failure_report_R1.json").exists()
    records = AgentExperienceMemory(knowledge_path, mirror_paths=[]).load()
    assert any(record.get("record_type") == "task_failure" for record in records)


def test_test_knowledge_guidance_library_shows_learning_status(tmp_path, monkeypatch):
    knowledge_path = tmp_path / "knowledge_store" / "test_knowledge_guidance.json"
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(knowledge_path))
    records = []
    for index in range(3):
        records.append(
            {
                "experience_id": f"exp-{index}",
                "timestamp": 1000 + index,
                "source": f"task_{index}.py",
                "agent": "AssertionAgent",
                "action": "strengthen concrete oracle",
                "failure_category": "weak_assertions_only",
                "task_signature": {"language": "python"},
                "reward": 0.2,
                "learning_notes": [{"solution": "使用具体返回值断言"}],
            }
        )
    knowledge_path.parent.mkdir(parents=True)
    knowledge_path.write_text(json.dumps({"records": records}, ensure_ascii=False), encoding="utf-8")

    rendered = TestKnowledgeGuidanceLibraryVisualizer().render(AgentExperienceMemory(knowledge_path, mirror_paths=[]))
    text = Path(rendered["html"]).read_text(encoding="utf-8")

    assert rendered["name"] == "测试知识引导库"
    assert "已掌握" in text
    assert "使用具体返回值断言" in text
