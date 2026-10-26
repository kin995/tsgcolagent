from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

from batch_generate_tests import (
    BatchItemResult,
    _batch_quality_overview,
    _build_batch_summary,
    _restore_internal_report,
    _round_metrics,
    _write_batch_summary,
    generate_tests_for_directory,
)
from llm_config import LLMConfig
from main import _terminal_summary as task_terminal_summary
from run_all_datasets import _render_master_report, _write_master_summary


def test_single_task_finalizes_paper_measurement_before_html_render(tmp_path: Path) -> None:
    from metrics.measurement import CURRENT
    from orchestration.test_workflow import LangGraphTestWorkflow

    source = tmp_path / "source.py"
    source.write_text("def f(): return 1\n", encoding="utf-8")
    graph = SimpleNamespace(metadata={}, version=1)
    paper_path = tmp_path / "paper_metrics" / "summary.json"

    class Measurement:
        def summary(self, report, duration, completed=False):
            assert completed is True
            paper_path.parent.mkdir()
            paper_path.write_text(json.dumps({"duration_seconds": 12.5}), encoding="utf-8")
            return {"duration_seconds": 12.5}

    def render(_graph):
        assert json.loads(paper_path.read_text(encoding="utf-8"))["duration_seconds"] == 12.5
        return {"status": "generated"}

    system = SimpleNamespace(
        memory=SimpleNamespace(load=lambda: graph),
        source_path=source,
        graph_path=tmp_path / "state_flow_graph.json",
        output_dir=tmp_path,
        thresholds=SimpleNamespace(to_dict=lambda: {}),
        max_repair_rounds=1,
        retry_rounds=0,
        experiment_round=1,
        followup_strategy="initial_generation",
        experience=SimpleNamespace(summary=lambda: {}),
        llm_config=SimpleNamespace(public_dict=lambda: {}),
        _usage_snapshot=lambda: {},
        _render_visualizations=render,
    )
    workflow = object.__new__(LangGraphTestWorkflow)
    workflow.system = system
    workflow.checkpoint_path = tmp_path / "checkpoints.sqlite"
    report = {"pytest_passed": True, "collected_count": 1, "quality_status": "quality_high"}
    token = CURRENT.set(Measurement())
    try:
        summary = workflow._summary({"current_report": report, "reports": [dict(report)], "stop_reason": "quality_high"}, time.perf_counter())
    finally:
        CURRENT.reset(token)

    assert summary["paper_metrics"]["duration_seconds"] == 12.5
    assert summary["visualization"] == {"status": "generated"}


def test_batch_result_exposes_paper_metrics_but_keeps_process_metrics_internal(tmp_path: Path) -> None:
    report = {
        "pytest_passed": True,
        "line_coverage": 0.8,
        "branch_coverage": 0.7,
        "ae": 0.6,
        "mutation_score": 0.5,
        "sfc": 0.9,
        "nrr": 1.0,
        "tsq": 0.85,
        "sfq": 0.85,
        "csg": 0.85,
        "pv": 0.75,
        "threshold_failures": {"sfc_gap": 0.1},
    }
    item = BatchItemResult(
        source=str(tmp_path / "task.py"),
        output_dir=str(tmp_path),
        status="quality_acceptable",
        final_grade="B",
        final_report=report,
        grade_history=[{"round": 1, "metrics": {"sfc": 0.9, "ae": 0.6}}],
    )

    serialized = json.dumps(item.to_dict(), ensure_ascii=False)
    for name in ("sfc", "nrr", "tsq", "sfq", "csg", "pv"):
        assert name not in serialized.lower()
    assert item.final_report is report
    assert report["sfc"] == 0.9
    assert item.to_dict()["final_report"]["mutation_score"] == 0.5
    assert _round_metrics(report) == {
        "pytest_passed": True,
        "line_coverage": 0.8,
        "branch_coverage": 0.7,
        "ae": 0.6,
        "mutation_score": 0.5,
    }
    overview = _batch_quality_overview([item])
    assert "average_sfc" not in overview and "average_sfq" not in overview
    assert overview["average_ae"] == 0.6


def test_resume_restores_internal_metrics_only_from_matching_graph(tmp_path: Path) -> None:
    source = tmp_path / "task.py"
    source.write_text("def f(): return 1\n", encoding="utf-8")
    public_report = {"pytest_passed": True, "line_coverage": 0.8, "ae": 0.6, "mutation_score": 0.5}
    graph_path = tmp_path / "state_flow_graph.json"
    graph_path.write_text(json.dumps({
        "source_path": str(source),
        "version": 3,
        "metadata": {"last_report": {
            **public_report,
            "sfc": 0.9,
            "tsq": 0.85,
            "threshold_failures": {"sfc_gap": 0.1},
        }},
    }), encoding="utf-8")

    restored = _restore_internal_report(tmp_path, source, public_report, 3)
    assert restored is not public_report
    assert restored["sfc"] == 0.9
    assert restored["tsq"] == 0.85
    assert restored["threshold_failures"] == {"sfc_gap": 0.1}
    assert "sfc" not in public_report
    assert _restore_internal_report(tmp_path, source, public_report, 4) == public_report
    assert _restore_internal_report(tmp_path, tmp_path / "another.py", public_report, 3) == public_report

    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["metadata"]["last_report"]["line_coverage"] = 0.2
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    assert _restore_internal_report(tmp_path, source, public_report, 3) == public_report


def test_batch_learning_keeps_raw_report_while_saved_summary_is_public(tmp_path: Path) -> None:
    item = BatchItemResult(
        source="task_000.py",
        output_dir=str(tmp_path),
        status="valid_usable",
        final_report={
            "pytest_passed": True,
            "sfc": 0.9,
            "tsq": 0.8,
            "weak_quality_flags": ["weak_state_flow_coverage"],
            "threshold_failures": {"sfc_gap": 0.1},
        },
    )
    summary = _build_batch_summary(
        source_root=tmp_path,
        output_root=tmp_path / "out",
        pattern="task_*.py",
        effective_pattern="task_*.py",
        language="python",
        recursive=False,
        execute=True,
        llm_config=LLMConfig(enabled=False),
        started_at="2026-09-23T00:00:00+00:00",
        discovered_count=1,
        official_tasks=1,
        selected_count=1,
        skipped_sources=[],
        results=[item],
        retry_history=[],
    )
    assert summary["results"][0]["final_report"]["sfc"] == 0.9
    assert summary["results"][0]["final_report"]["threshold_failures"] == {"sfc_gap": 0.1}
    path = tmp_path / "batch_summary.json"
    _write_batch_summary(path, summary)
    saved = path.read_text(encoding="utf-8")
    assert '"sfc"' not in saved and '"tsq"' not in saved
    assert "weak_state_flow_coverage" not in saved


def test_completed_dataset_migrates_existing_reports(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "task_000.py").write_text("def f(): return 1\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    first = generate_tests_for_directory(source_dir, output_dir, execute=False)
    assert first["dataset_completion"]["skip_ready"] is True

    batch_path = output_dir / "batch_summary.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["average_sfc"] = 0.9
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    case_summary_path = Path(first["results"][0]["summary_path"])
    case = json.loads(case_summary_path.read_text(encoding="utf-8"))
    case["tsq"] = 0.8
    case_summary_path.write_text(json.dumps(case), encoding="utf-8")
    (output_dir / "batch_quality_report.html").write_text("SFC TSQ", encoding="utf-8")

    second = generate_tests_for_directory(source_dir, output_dir, execute=False)
    assert second["dataset_skipped"] is True
    assert '"average_sfc"' not in batch_path.read_text(encoding="utf-8")
    assert '"tsq"' not in case_summary_path.read_text(encoding="utf-8")
    rendered = (output_dir / "batch_quality_report.html").read_text(encoding="utf-8")
    assert "SFC" not in rendered and "TSQ" not in rendered


def test_terminal_and_master_reports_hide_process_metrics(tmp_path: Path) -> None:
    task = task_terminal_summary({
        "final_report": {"line_coverage": 0.8, "ae": 0.6, "sfc": 0.9, "tsq": 0.85},
        "visualization": {"sfc": 0.9},
    })
    assert "sfc" not in json.dumps(task).lower()
    assert task["metrics"]["assertion_effectiveness"] == 0.6

    master = {
        "output_root": str(tmp_path),
        "results": [{
            "dataset": {"name": "demo", "language": "python"},
            "status": "completed",
            "summary": {"selected_tasks": 1, "metrics": {
                "average_line_coverage": 0.8,
                "average_sfc": 0.9,
                "average_ae": 0.6,
                "average_sfq": 0.85,
            }},
        }],
    }
    _write_master_summary(tmp_path, master)
    saved = (tmp_path / "all_datasets_summary.json").read_text(encoding="utf-8")
    html = (tmp_path / "all_datasets_report.html").read_text(encoding="utf-8")
    assert "average_sfc" not in saved and "average_sfq" not in saved
    assert "SFC" not in html and "TSQ" not in html
    assert "平均AE" in html
    assert "SFC" not in _render_master_report(master)


def test_master_report_shows_published_tir_and_only_complete_avgexec(tmp_path):
    summary = {
        "output_root": str(tmp_path),
        "results": [
            {"dataset": {"name": "one", "language": "python"}, "status": "completed", "summary": {
                "selected_tasks": 2, "paper_metrics": {"tir": 0.5, "tir_improved": 1, "tir_comparable": 2, "avg_exec": 3.25, "exec_complete_tasks": 2}}},
            {"dataset": {"name": "two", "language": "java"}, "status": "completed", "summary": {
                "selected_tasks": 2, "paper_metrics": {"tir": 1 / 3, "tir_improved": 1, "tir_comparable": 3, "avg_exec": 5.0, "exec_complete_tasks": 1}}},
        ],
        "outcome_summary": {"total_tasks": 4, "completed_execution": 3, "not_completed_execution": 1},
    }
    html = _render_master_report(summary)
    assert "TIR" in html and "40.00%" in html
    assert "AvgExec" in html and "3.25" in html and "5.00" in html and "3.83" in html
    assert "3/4" in html
