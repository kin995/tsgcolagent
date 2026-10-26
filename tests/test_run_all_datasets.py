from __future__ import annotations

import json
from pathlib import Path

import run_all_datasets as all_module
from core.project_paths import resolve_datasets_root
from llm_config import LLMConfig


def _make_expected_datasets(root: Path) -> None:
    for name in ["TestEval", "HumanEval"]:
        (root / name).mkdir(parents=True)
    for name in ["cli", "csv", "lang3", "gson"]:
        (root / "Defects4J" / name).mkdir(parents=True)


def test_run_all_datasets_orchestrates_known_datasets_with_resume(tmp_path, monkeypatch):
    datasets_root = tmp_path / "datasets"
    output_root = tmp_path / "runs"
    _make_expected_datasets(datasets_root)
    calls: list[dict[str, object]] = []

    def fake_generate_tests_for_directory(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True)
        (output_dir / "batch_summary.json").write_text("{}", encoding="utf-8")
        (output_dir / "batch_quality_report.html").write_text("<html></html>", encoding="utf-8")
        calls.append(kwargs)
        return {
            "total": 2,
            "discovered_files": 2,
            "selected_tasks": 1,
            "resumed_existing": 1 if kwargs["resume"] else 0,
            "succeeded": 1,
            "failed": 0,
            "incomplete": 0,
            "mode": "execute" if kwargs["execute"] else "generate_only",
            "language": kwargs["language"],
            "effective_pattern": "*.java" if kwargs["language"] == "java" else "task_*.py",
            "batch_report": str(output_dir / "batch_quality_report.html"),
            "quality_overview": {
                "executed_tasks": 1 if kwargs["execute"] else 0,
                "average_line_coverage": 0.8,
                "average_branch_coverage": 0.7,
                "average_sfc": 0.6,
                "average_ae": 0.5,
                "average_mutation_score": 0.4,
                "average_sfq": 0.3,
            },
        }

    monkeypatch.setattr(all_module, "generate_tests_for_directory", fake_generate_tests_for_directory)

    summary = all_module.run_all_datasets(
        datasets_root=datasets_root,
        output_root=output_root,
        llm_config=LLMConfig(enabled=True, required=True),
        limit_per_dataset=1,
        quiet=True,
    )

    assert [Path(call["source_dir"]).name for call in calls] == ["TestEval", "HumanEval", "cli", "csv", "lang3", "gson"]
    assert [call["language"] for call in calls] == ["python", "python", "java", "java", "java", "java"]
    assert [call["execute"] for call in calls] == [True, True, True, True, True, True]
    assert all(call["resume"] is True for call in calls)
    assert all(call["rerun_existing"] is False for call in calls)
    assert all(call["limit"] == 1 for call in calls)
    assert summary["total_datasets"] == 6
    assert summary["completed_datasets"] == 6
    assert summary["selected_tasks"] == 6
    assert summary["resumed_existing"] == 6
    assert (output_root / "all_datasets_summary.json").exists()
    assert (output_root / "all_datasets_report.html").exists()

    saved = json.loads((output_root / "all_datasets_summary.json").read_text(encoding="utf-8"))
    assert saved["llm"]["enabled"] is True
    assert "Defects4J_gson" in (output_root / "all_datasets_report.html").read_text(encoding="utf-8")


def test_run_all_datasets_can_skip_java_execution_when_requested(tmp_path, monkeypatch):
    datasets_root = tmp_path / "datasets"
    output_root = tmp_path / "runs"
    _make_expected_datasets(datasets_root)
    java_execute_flags: list[bool] = []

    def fake_generate_tests_for_directory(**kwargs):
        if kwargs["language"] == "java":
            java_execute_flags.append(kwargs["execute"])
        return {
            "total": 0,
            "discovered_files": 0,
            "selected_tasks": 0,
            "resumed_existing": 0,
            "succeeded": 0,
            "failed": 0,
            "incomplete": 0,
            "language": kwargs["language"],
            "batch_report": None,
            "quality_overview": {},
        }

    monkeypatch.setattr(all_module, "generate_tests_for_directory", fake_generate_tests_for_directory)

    all_module.run_all_datasets(
        datasets_root=datasets_root,
        output_root=output_root,
        llm_config=LLMConfig(enabled=False),
        execute_java=False,
        quiet=True,
    )

    assert java_execute_flags == [False, False, False, False]


def test_discovers_humaneval_java_and_executes_auto_discovered_java(tmp_path):
    datasets_root = tmp_path / "datasets"
    output_root = tmp_path / "runs"
    _make_expected_datasets(datasets_root)
    human_eval_java = datasets_root / "HumanEvalJava"
    human_eval_java.mkdir()
    (human_eval_java / "Task.java").write_text("public class Task { public int value() { return 1; } }", encoding="utf-8")
    extra_java = datasets_root / "ProjectJava"
    extra_java.mkdir()
    (extra_java / "ProjectTask.java").write_text("public class ProjectTask { public int value() { return 1; } }", encoding="utf-8")

    specs = all_module.discover_dataset_specs(datasets_root, output_root, execute_java=True)
    selected = {item.name: item for item in specs}

    assert selected["HumanEvalJava"].language == "java"
    assert selected["HumanEvalJava"].execute is True
    assert selected["ProjectJava"].language == "java"
    assert selected["ProjectJava"].execute is True


def test_run_all_only_enables_java_workers_for_humanevaljava(tmp_path, monkeypatch):
    datasets_root = tmp_path / "datasets"
    output_root = tmp_path / "runs"
    _make_expected_datasets(datasets_root)
    humaneval_java = datasets_root / "HumanEvalJava"
    humaneval_java.mkdir()
    (humaneval_java / "Task.java").write_text("public class Task { public int value() { return 1; } }", encoding="utf-8")
    calls = []

    def fake_generate_tests_for_directory(**kwargs):
        calls.append(kwargs)
        return {"total": 0, "selected_tasks": 0, "resumed_existing": 0, "succeeded": 0, "failed": 0, "incomplete": 0, "quality_overview": {}}

    monkeypatch.setattr(all_module, "generate_tests_for_directory", fake_generate_tests_for_directory)
    all_module.run_all_datasets(
        datasets_root=datasets_root,
        output_root=output_root,
        llm_config=LLMConfig(enabled=False),
        human_eval_java_workers=2,
        quiet=True,
    )
    by_name = {Path(call["source_dir"]).name: call for call in calls}

    assert by_name["HumanEvalJava"]["workers"] == 2
    assert by_name["TestEval"]["workers"] == 4
    assert by_name["HumanEval"]["workers"] == 4
    assert all(by_name[name]["workers"] == 1 for name in ["cli", "csv", "lang3", "gson"])


def test_dataset_root_can_match_other_computer_environment(tmp_path, monkeypatch):
    datasets_root = tmp_path / "multi_evo_py" / "datasets"
    datasets_root.mkdir(parents=True)
    monkeypatch.setenv("TSG_DATASETS_ROOT", str(datasets_root))

    assert resolve_datasets_root() == datasets_root.resolve()
