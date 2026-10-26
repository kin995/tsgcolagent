"""The saved page dataset contains only public, rendered information."""

from __future__ import annotations

import copy
import hashlib
import json

from monitoring.page_data import (
    project_knowledge_page,
    project_page_data,
    project_runtime_page,
    write_page_data,
)


def _snapshots(generated_at: str = "2026-09-28T01:00:00+00:00") -> tuple[dict, dict]:
    runtime = {
        "schema_version": 1,
        "generated_at": generated_at,
        "experiment": {
            "id": "deepseek-run", "label": "DeepSeek-R1", "status": "completed",
            "output_dir": "D:/runs-DeepSeek-R1", "finished_at": "2026-09-15T19:00:00+08:00",
            "report_path": "D:/runs-DeepSeek-R1/all_datasets_quality_report.html",
        },
        "experiments": [{"id": "dataset-1", "label": "HumanEval", "total_tasks": 1,
                         "batch_report_path": "D:/runs-DeepSeek-R1/HumanEval/batch_quality_report.html",
                         "unneeded": "discard"}],
        "portfolio": {"view": "master", "counts": {"completed": 1},
                      "experiments": [{"id": "dataset-1", "label": "HumanEval", "total_tasks": 1}]},
        "overview": {
            "status": "completed", "total_tasks": 1, "completed_execution": 1,
            "quality_averages": {
                "average_line_coverage": .91, "average_branch_coverage": .81,
                "average_ae": .71, "average_mutation_score": .61,
                "pass_rate": 1.0, "avg_exec": 2.0,
                "average_time_seconds": 12.3, "tir": .5,
                "tir_improved": 1, "tir_comparable": 2,
                "exec_complete_tasks": 1, "selected_tasks": 1,
                "valid_metric_tasks": {"line_coverage": 1},
                "internal_note": "not part of the page",
            },
        },
        "tasks": [{
            "id": "task-1", "name": "EXAMPLE.py", "grade": "A", "round": 2,
            "round_label": "A·R2", "source": "HumanEval/EXAMPLE.py",
            "output_dir": "D:/runs-DeepSeek-R1/HumanEval/EXAMPLE",
            "summary_path": "D:/runs-DeepSeek-R1/HumanEval/EXAMPLE/stateflow_summary.json",
            "graph_path": "D:/runs-DeepSeek-R1/HumanEval/EXAMPLE/state_flow_graph.json",
            "test_path": "D:/runs-DeepSeek-R1/HumanEval/EXAMPLE/test_example.py",
            "timing_history": [{"command": "pytest", "payload": "large"}],
            "grade_history": [{"round": 1, "grade": "D", "status": "failed",
                               "failure_reason": "compile error", "metrics": {"line_coverage": .3,
                               "mutation_score": .1, "sfc": .9},
                               "knowledge_selected_ids": [f"record-{i}" for i in range(100)]}],
            "metrics": {"tests_passed": True, "collected_count": 3,
                        "line_coverage": .91, "branch_coverage": .81, "ae": .71,
                        "mutation_score": .61, "duration_seconds": 12.3,
                        "pass_rate": 1.0, "avg_exec": 2.0, "exec_count": 2,
                        "exec_count_status": "complete", "tir": .5,
                        "tir_comparable": 2, "tir_improved": 1,
                        "knowledge_record_ids": ["experience-1"],
                        "sfc": .99},
            "failure": {"category": "compile", "reason": "invalid import",
                        "knowledge_history": [{"nested": "large"}],
                        "previous_solutions": ["remove import"],
                        "attempted_measures": ["retry"]},
        }],
        "current_task": None,
        "flow_events": [{"event_type": "KnowledgeRetrieved", "source_agent": "知识智能体",
                         "payload": {"experiment_round": 2, "action": "retrieval",
                                     "source_context": {"large": [1] * 100}, "sfc": .8},
                         "related_nodes": ["n1"]}],
        "state_model": {"nodes": [{"id": "n1", "type": "Source"}], "edges": [],
                        "metadata": {"last_report": {"status": "complete"},
                                     "last_knowledge_outcome": {"unused": "discard"}}},
        "catalog": {"name": "System", "runtime": "Python", "agents": [], "languages": [],
                    "scopes": [], "phases": [], "quality_metrics": [], "execution_policies": [],
                    "internal_architecture": {"large": True}},
        "agents": [], "phase_states": [], "toolchain": [], "failures": {},
        "unneeded": {"large": "discard"},
    }
    knowledge = {
        "path": "D:/writable/imported-knowledge.json", "scope": "imported_run",
        "total": 1, "counts": {"已掌握": 1},
        "availability": {"complete": False, "note": "已恢复记录",
                         "run_record_count": 1, "baseline_record_count": 0,
                         "dataset_summary_count": 0, "missing_datasets": [],
                         "external_baseline_record_count": 100},
        "records": [{"id": "experience-1", "status": "已掌握", "topic": "断言修复",
                     "usage": "处理导入错误", "solution": "移除无效导入", "source_type": "run_record",
                     "dataset": "HumanEval", "support_count": 2, "accepted": True,
                     "question": "not shown by page", "language": "python",
                     "record_type": "repair", "learning_stage": "adopted",
                     "validation_failure_count": 1, "sfc": .5}],
        "stored_record_count": 2,
    }
    return runtime, knowledge


def test_page_projection_keeps_visible_metrics_knowledge_and_report_routes_only():
    runtime, knowledge = _snapshots()
    original_runtime, original_knowledge = copy.deepcopy(runtime), copy.deepcopy(knowledge)
    page_runtime, page_knowledge = project_page_data(runtime, knowledge)
    assert project_runtime_page(runtime) == page_runtime
    assert project_knowledge_page(knowledge) == page_knowledge
    assert page_runtime["overview"]["page_title"] == "实验总览"

    values = page_runtime["overview"]["quality_averages"]
    for key in ("average_line_coverage", "average_branch_coverage", "average_ae",
                "average_mutation_score", "pass_rate", "avg_exec", "average_time_seconds", "tir"):
        assert values[key] == runtime["overview"]["quality_averages"][key]
    assert page_runtime["experiment"]["report_path"] == runtime["experiment"]["report_path"]
    assert page_runtime["experiments"][0]["batch_report_path"] == runtime["experiments"][0]["batch_report_path"]
    task = page_runtime["tasks"][0]
    assert task["test_path"] == runtime["tasks"][0]["test_path"]
    assert task["summary_path"] == runtime["tasks"][0]["summary_path"]
    assert task["metrics"]["tir"] == .5
    assert task["metrics"]["avg_exec"] == 2.0
    assert "timing_history" not in task
    assert "knowledge_history" not in task["failure"]
    assert "knowledge_selected_ids" not in task["grade_history"][0]
    assert "source_context" not in page_runtime["flow_events"][0]["payload"]
    assert page_knowledge["records"][0]["solution"] == "移除无效导入"
    assert "question" not in page_knowledge["records"][0]
    assert not {"language", "record_type", "learning_stage", "validation_failure_count"} & page_knowledge["records"][0].keys()
    assert "path" not in page_knowledge
    assert "external_baseline_record_count" not in page_knowledge["availability"]
    encoded = json.dumps((page_runtime, page_knowledge), ensure_ascii=False).lower()
    assert all(token not in encoded for token in ('"sfc"', '"nrr"', '"tsq"', '"csg"', '"pv"', '内部过程指标已隐藏'))
    assert runtime == original_runtime and knowledge == original_knowledge


def test_page_files_are_separate_hash_checked_and_idempotent(tmp_path):
    runtime, knowledge = _snapshots()
    manifest = write_page_data(tmp_path, runtime, knowledge)
    paths = [tmp_path / name for name in ("runtime.json", "knowledge.json", "manifest.json")]
    assert all(path.is_file() for path in paths)
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    for name in ("runtime.json", "knowledge.json"):
        body = (tmp_path / name).read_bytes()
        assert manifest["files"][name]["sha256"] == hashlib.sha256(body).hexdigest()
        assert manifest["files"][name]["bytes"] == len(body)
    assert manifest["source"]["experiment_id"] == "deepseek-run"
    assert manifest["source"]["run_path"] == "D:/runs-DeepSeek-R1"
    assert manifest["counts"] == {"tasks": 1, "knowledge_records": 1}
    assert manifest["generated_at"] == "2026-09-15T19:00:00+08:00"

    runtime["generated_at"] = "2026-09-29T01:00:00+00:00"
    assert write_page_data(tmp_path, runtime, knowledge) == manifest
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before
