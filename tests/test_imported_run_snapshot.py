from __future__ import annotations

import json
import os
import time

import monitoring.dashboard_data as dashboard_data
from monitoring.dashboard_data import _failure_details, build_snapshot


def test_moved_completed_run_uses_local_artifacts_and_finished_status(tmp_path):
    run = tmp_path / "runs-DeepSeek-R1"
    task = run / "TestEval" / "task_000"
    task.mkdir(parents=True)
    (task / "test_case.py").write_text("def test_case(): pass\n", encoding="utf-8")
    graph = task / "state_flow_graph.json"
    graph.write_text(json.dumps({
        "version": 4,
        "nodes": {"method": {"node_type": "Method", "name": "calculate", "visited": True}},
        "edges": [],
        "flow_history": [{"event_type": "TestGenerated", "source_agent": "测试生成智能体"}],
    }, ensure_ascii=False), encoding="utf-8")
    (task / "stateflow_summary.json").write_text(json.dumps({
        "source": r"D:\old_machine\datasets\TestEval\task_000.py",
        "test_path": r"D:\old_machine\runs\TestEval\task_000\test_case.py",
        "graph_path": r"D:\old_machine\runs\TestEval\task_000\state_flow_graph.json",
        "quality_status": "quality_high",
        "final_grade": "A",
        "llm_usage": {"calls": 2, "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "duration_seconds": 4.5},
        "final_report": {"pytest_passed": True, "collected_count": 1, "line_coverage": 0.9, "ae": 0.8, "sfc": 0.7, "nrr": 0.5},
    }, ensure_ascii=False), encoding="utf-8")
    (run / "all_datasets_summary.json").write_text(json.dumps({
        "total_datasets": 1,
        "completed_datasets": 1,
        "selected_tasks": 1,
        "llm": {"provider": "ollama", "model": "deepseek-r1:14b", "enabled": True},
        "finished_at": "2026-09-08T13:50:11+00:00",
        "results": [{"dataset": {"name": "TestEval", "language": "python"}, "status": "completed"}],
    }), encoding="utf-8")
    (run / "TestEval" / "batch_runtime.json").write_text(json.dumps({
        "status": "completed", "task_workers": 4, "python_parallel": True,
        "java_parallel": False, "llm_concurrency": 1,
    }), encoding="utf-8")
    record = {"id": "deepseek", "output_dir": str(run), "label": "DeepSeek-R1", "execution_status": "completed"}

    snapshot = build_snapshot(record, [])

    assert snapshot["overview"]["status"] == "completed"
    assert snapshot["overview"]["total_tasks"] == 1
    assert snapshot["overview"]["model_name"] == "deepseek-r1:14b"
    assert snapshot["overview"]["model_usage"]["calls"] == 2
    assert snapshot["overview"]["model_usage"]["total_tokens"] == 120
    assert "_llm_usage" not in snapshot["tasks"][0]
    assert snapshot["parallel_execution"]["task_workers"] == 4
    assert snapshot["parallel_execution"]["python_parallel"] is True
    assert snapshot["tasks"][0]["source"] == "TestEval/task_000.py"
    assert snapshot["tasks"][0]["graph_path"] == str(graph.resolve())
    assert snapshot["tasks"][0]["test_path"] == str((task / "test_case.py").resolve())
    assert snapshot["state_model"]["nodes"][0]["name"] == "calculate"
    assert snapshot["flow_events"][0]["event_type"] == "TestGenerated"
    assert "sfc" not in snapshot["tasks"][0]["metrics"]
    assert "nrr" not in snapshot["tasks"][0]["metrics"]


def test_completed_run_caches_compact_tasks_and_invalidates_when_a_summary_changes(tmp_path, monkeypatch):
    run = tmp_path / "run"
    task = run / "TestEval" / "task_000"
    task.mkdir(parents=True)
    summary = task / "stateflow_summary.json"
    summary.write_text(json.dumps({
        "source": "task_000.py",
        "final_grade": "A",
        "final_report": {"pytest_passed": True, "collected_count": 1, "line_coverage": 0.9},
    }), encoding="utf-8")
    (run / "all_datasets_summary.json").write_text(json.dumps({"finished_at": "2026-09-08T13:50:11+00:00"}), encoding="utf-8")
    record = {"id": "cache-run", "output_dir": str(run), "label": "DeepSeek-R1", "execution_status": "completed"}
    monkeypatch.setattr(dashboard_data, "TASK_CACHE_DIR", tmp_path / "cache", raising=False)

    first = build_snapshot(record, [])
    assert first["tasks"][0]["metrics"]["line_coverage"] == 0.9
    with monkeypatch.context() as patch:
        def unexpected_reparse(*_args, **_kwargs):
            raise AssertionError("completed task summary was reparsed")
        patch.setattr(dashboard_data, "_task_from_summary", unexpected_reparse)
        second = build_snapshot(record, [])
    assert second["tasks"][0]["metrics"]["line_coverage"] == 0.9

    summary.write_text(json.dumps({
        "source": "task_000.py",
        "final_grade": "A",
        "final_report": {"pytest_passed": True, "collected_count": 1, "line_coverage": 0.5},
    }), encoding="utf-8")
    changed = time.time() + 2
    os.utime(summary, (changed, changed))
    third = build_snapshot(record, [])
    assert third["tasks"][0]["metrics"]["line_coverage"] == 0.5


def test_failed_test_records_are_projected_to_display_names():
    failure = _failure_details({
        "failed_tests": [
            {"test_name": "java_junit_execution", "message": "long execution details"},
            "test_boundary_case",
        ],
    }, {}, {})
    assert failure["failed_tests"] == ["java_junit_execution", "test_boundary_case"]


def test_failure_details_omit_unpublished_process_measurements():
    failure = _failure_details(
        {"weak_quality_flags": ["weak_sfc", "mutation_unavailable"]},
        {"previous_solutions": ["Improve SFC before retry.", "Repair the failing JUnit method."]},
        {},
    )
    assert failure["weak_flags"] == ["mutation_unavailable"]
    assert failure["previous_solutions"] == ["Repair the failing JUnit method."]
