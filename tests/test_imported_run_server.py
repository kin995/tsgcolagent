"""HTTP regressions for browsing a registered multi-dataset result bundle."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

import monitoring.server as server
from monitoring.dashboard_data import experiment_id
from monitoring.registry import ExperimentRegistry


@pytest.fixture
def dashboard(tmp_path, monkeypatch):
    registry = ExperimentRegistry(tmp_path / "registry.json")
    monkeypatch.setattr(server, "REGISTRY", registry)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield registry, f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _make_master(tmp_path, names=("TestEval", "HumanEvalJava")):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "all_datasets_report.html").write_text("<html>Master report</html>", encoding="utf-8")
    results = []
    for name in names:
        child = root / name
        child.mkdir()
        (child / "batch_quality_report.html").write_text(f"<html>{name} report</html>", encoding="utf-8")
        (child / "batch_summary.json").write_text(
            json.dumps({"language": "java" if name.endswith("Java") else "python", "total": 1,
                        "outcome_summary": {"total_tasks": 1, "completed_execution": 1, "not_completed_execution": 0}}),
            encoding="utf-8",
        )
        results.append({"dataset": {"name": name, "language": "java" if name.endswith("Java") else "python"},
                        "status": "completed", "summary": {"total": 1}})
    (root / "all_datasets_summary.json").write_text(
        json.dumps({"results": results, "total_datasets": len(names), "completed_datasets": len(names)}),
        encoding="utf-8",
    )
    return root


def test_virtual_dataset_artifacts_open_and_stay_inside_selected_dataset(tmp_path, dashboard):
    registry, base = dashboard
    root = _make_master(tmp_path)
    child = root / "TestEval"
    sibling = root / "HumanEvalJava"
    result = child / "result.json"
    result.write_text(json.dumps({"line_coverage": 0.75, "sfc": 0.99}), encoding="utf-8")
    report = child / "report.html"
    report.write_text("<html><body>TestEval report</body></html>", encoding="utf-8")
    sibling_result = sibling / "result.json"
    sibling_result.write_text('{"line_coverage": 0.25}', encoding="utf-8")
    task = child / "task_001"
    task.mkdir()
    (task / "stateflow_summary.json").write_text(
        json.dumps({"source": str(sibling_result), "final_grade": "A",
                    "final_report": {"pytest_passed": True, "collected_count": 1}}),
        encoding="utf-8",
    )
    registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})
    child_id = experiment_id(child)

    with urlopen(f"{base}/api/artifact?{urlencode({'experiment': child_id, 'path': str(result)})}") as response:
        payload = json.load(response)
    assert payload["data"]["line_coverage"] == 0.75
    assert "sfc" not in payload["data"]

    with urlopen(f"{base}/api/artifact/raw?{urlencode({'experiment': child_id, 'path': str(report)})}") as response:
        assert response.read().decode("utf-8") == "<html><body>TestEval report</body></html>"

    for endpoint in ("/api/artifact", "/api/artifact/raw"):
        forbidden = sibling_result if endpoint == "/api/artifact" else sibling / "report.html"
        if endpoint.endswith("raw"):
            forbidden.write_text("<html>Sibling report</html>", encoding="utf-8")
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}{endpoint}?{urlencode({'experiment': child_id, 'path': str(forbidden)})}")
        assert error.value.code == 403


def test_default_all_lists_only_latest_master_datasets_but_old_run_remains_selectable(tmp_path, dashboard):
    registry, base = dashboard
    old = tmp_path / "old-preview"
    old.mkdir()
    (old / "batch_summary.json").write_text('{"total": 1}', encoding="utf-8")
    old_record = registry.register(str(old), "旧示例", metadata={"execution_status": "completed"})
    root = _make_master(tmp_path)
    registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})

    with urlopen(f"{base}/api/runtime?experiment=all") as response:
        snapshot = json.load(response)
    assert [card["label"] for card in snapshot["experiments"]] == ["TestEval", "HumanEvalJava"]
    assert [card["label"] for card in snapshot["portfolio"]["experiments"]] == ["TestEval", "HumanEvalJava"]
    assert snapshot["overview"]["dataset_count"] == 2
    assert snapshot["experiment"]["label"] == "DeepSeek-R1"

    child_id = experiment_id(root / "TestEval")
    with urlopen(f"{base}/api/runtime?experiment={child_id}") as response:
        child_snapshot = json.load(response)
    assert [card["label"] for card in child_snapshot["experiments"]] == ["TestEval", "HumanEvalJava"]

    with urlopen(f"{base}/api/runtime?experiment={old_record['id']}") as response:
        old_snapshot = json.load(response)
    assert old_snapshot["experiment"]["id"] == old_record["id"]


def test_master_direct_link_shows_its_own_dataset_portfolio(tmp_path, dashboard):
    registry, base = dashboard
    root = _make_master(tmp_path)
    master = registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})

    with urlopen(f"{base}/api/runtime?experiment={master['id']}") as response:
        snapshot = json.load(response)

    assert snapshot["experiment"]["id"] == master["id"]
    assert snapshot["experiment"]["label"] == "DeepSeek-R1"
    assert [card["label"] for card in snapshot["experiments"]] == ["TestEval", "HumanEvalJava"]
    assert [card["label"] for card in snapshot["portfolio"]["experiments"]] == ["TestEval", "HumanEvalJava"]
    assert snapshot["overview"]["dataset_count"] == 2
    assert snapshot["experiment"]["report_path"] == str(root / "all_datasets_report.html")
    assert snapshot["experiments"][0]["batch_report_path"] == str(root / "TestEval" / "batch_quality_report.html")


def test_imported_batch_report_uses_runtime_metrics_and_local_report_links(tmp_path, dashboard):
    registry, base = dashboard
    root = _make_master(tmp_path, names=("HumanEvalJava",))
    child = root / "HumanEvalJava"
    task = child / "Example"
    task.mkdir()
    (task / "state_flow_graph.html").write_text("<html>Task report</html>", encoding="utf-8")
    (task / "stateflow_summary.json").write_text(json.dumps({
        "source": r"D:\old_machine\datasets\HumanEvalJava\Example.java",
        "graph_path": r"D:\old_machine\runs\HumanEvalJava\Example\state_flow_graph.json",
        "final_grade": "A", "duration_seconds": 12.5,
        "final_report": {"pytest_passed": True, "collected_count": 1, "line_coverage": 0.9, "ae": 0.8},
    }), encoding="utf-8")
    (child / "batch_summary.json").write_text(json.dumps({
        "language": "java", "output_dir": r"D:\old_machine\runs\HumanEvalJava",
        "finished_at": "2026-09-08T13:50:11+00:00", "selected_tasks": 1,
        "outcome_summary": {"total_tasks": 1, "completed_execution": 1, "not_completed_execution": 0, "grade_counts": {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0}},
        "results": [{
            "source": r"D:\old_machine\datasets\HumanEvalJava\Example.java",
            "output_dir": r"D:\old_machine\runs\HumanEvalJava\Example",
            "final_grade": "A", "duration_seconds": 12.5,
            "final_report": {"pytest_passed": True, "collected_count": 1, "line_coverage": 0.9, "ae": 0.8},
            "visualization": {"graph_html": r"D:\old_machine\runs\HumanEvalJava\Example\state_flow_graph.html"},
        }],
    }), encoding="utf-8")
    registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})
    child_id = experiment_id(child)

    with urlopen(f"{base}/api/runtime?experiment={child_id}") as response:
        runtime = json.load(response)
    assert runtime["overview"]["quality_averages"]["pass_rate"] == 1.0
    report_url = f"{base}/api/artifact/raw?{urlencode({'experiment': child_id, 'path': str(child / 'batch_quality_report.html')})}"
    with urlopen(report_url) as response:
        html = response.read().decode("utf-8")
    assert "PassRate" in html and "100%" in html
    assert "平均 Time/s" in html and "12.50" in html
    assert "file://" not in html and "old_machine" not in html
    assert f"/api/artifact/raw?experiment={child_id}" in html


def test_imported_master_report_recomputes_observed_avgexec_from_task_evidence(tmp_path, dashboard):
    registry, base = dashboard
    root = _make_master(tmp_path, names=("TestEval",))
    child = root / "TestEval"
    task = child / "Example"
    task.mkdir()
    (task / "stateflow_summary.json").write_text(json.dumps({
        "source": "Example.py", "final_grade": "A",
        "final_report": {"pytest_passed": True, "collected_count": 1, "coverage_valid": True, "line_coverage": .8, "ae": .8},
    }), encoding="utf-8")
    (task / "paper_metrics").mkdir()
    (task / "paper_metrics" / "summary.json").write_text(json.dumps({
        "exec_count": 4, "exec_count_status": "complete", "avg_exec": 4,
        "tir_improved": 1, "tir_comparable": 2, "tir": .5,
    }), encoding="utf-8")
    master = json.loads((root / "all_datasets_summary.json").read_text(encoding="utf-8"))
    master["results"][0]["summary"] = {
        "selected_tasks": 1,
        "paper_metrics": {"avg_exec": None, "exec_complete_tasks": 0, "tir": None, "tir_improved": 0, "tir_comparable": 0},
    }
    (root / "all_datasets_summary.json").write_text(json.dumps(master), encoding="utf-8")
    registered = registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})

    url = f"{base}/api/artifact/raw?{urlencode({'experiment': registered['id'], 'path': str(root / 'all_datasets_report.html')})}"
    with urlopen(url) as response:
        page = response.read().decode("utf-8")
    assert "AvgExec" in page and "4.00" in page and "1/1" in page
    assert "TIR" in page and "50.00%" in page


def test_virtual_dataset_rejects_paths_outside_registered_master(tmp_path, dashboard):
    registry, base = dashboard
    root = _make_master(tmp_path, names=("TestEval",))
    outside = tmp_path / "secret"
    outside.mkdir()
    secret = outside / "report.html"
    secret.write_text("<html>private</html>", encoding="utf-8")
    summary = json.loads((root / "all_datasets_summary.json").read_text(encoding="utf-8"))
    summary["results"].append({
        "dataset": {"name": "../secret", "language": "python"},
        "summary_path": str(outside / "batch_summary.json"),
        "status": "completed",
    })
    summary["results"].append({
        "dataset": {"name": "missing", "language": "python"},
        "summary_path": str(outside / "batch_summary.json"),
        "status": "completed",
    })
    (outside / "batch_summary.json").write_text("{}", encoding="utf-8")
    (root / "all_datasets_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})

    with urlopen(f"{base}/api/runtime?experiment=all") as response:
        snapshot = json.load(response)
    assert [item["label"] for item in snapshot["experiments"]] == ["TestEval"]
    for endpoint in ("/api/artifact", "/api/artifact/raw"):
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}{endpoint}?{urlencode({'experiment': experiment_id(outside), 'path': str(secret)})}")
        assert error.value.code in {403, 404}


def test_failure_report_shows_public_failure_details_without_old_paths_or_process_metrics(tmp_path, dashboard):
    registry, base = dashboard
    root = _make_master(tmp_path, names=("HumanEvalJava",))
    child = root / "HumanEvalJava"
    task = child / "Example"
    task.mkdir()
    report = task / "task_failure_report.html"
    report.write_text("<html>旧路径 D:\\old_machine\\datasets\\Example.java SFC 0.5</html>", encoding="utf-8")
    report.with_suffix(".json").write_text(json.dumps({
        "source": r"D:\old_machine\datasets\Example.java",
        "final_grade": "D",
        "failure_category": "java_execution_failed",
        "failure_stage": "compile",
        "failure_reason": "candidate did not compile",
        "previous_solutions": ["Keep the JUnit package aligned", "SFC 0.5"],
        "suggested_solutions": ["Repair the failing method"],
    }), encoding="utf-8")
    registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})
    url = f"{base}/api/artifact/raw?{urlencode({'experiment': experiment_id(child), 'path': str(report)})}"
    with urlopen(url) as response:
        page = response.read().decode("utf-8")
    assert "candidate did not compile" in page
    assert "Repair the failing method" in page
    assert "Example.java" in page
    assert "SFC" not in page and "old_machine" not in page
    assert "内部过程指标已隐藏" not in page


def test_report_link_rewriting_drops_unsupported_schemes_and_stale_paths(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    page = (
        '<a href="javascript:alert(1)">unsafe</a>'
        '<a href="D:\\old_machine\\library.html">old</a>'
        '<a href="#details">details</a>'
    )
    cleaned = server._rewrite_report_links(page, root, r"D:\old_machine\runs", "run", master=True)
    assert "javascript:" not in cleaned and r"D:\old_machine" not in cleaned
    assert ">unsafe<" not in cleaned and "unsafe" in cleaned
    assert '<a href="#details">details</a>' in cleaned


def test_public_runtime_snapshot_omits_process_metric_placeholders_but_keeps_useful_advice():
    snapshot = {
        "tasks": [{"failure": {"previous_solutions": [
            "Keep the JUnit package aligned",
            "Reason: weak_sfc. Improve SFC 0.5 before retrying",
        ]}}],
        "phase_states": [{"last_event": {"payload": {"learning_notes": [
            {"reason_type": "weak_sfc", "reminder": "SFC 0.5 below target"},
        ]}}}],
    }
    public = server._public_snapshot(snapshot)
    assert public["tasks"][0]["failure"]["previous_solutions"] == ["Keep the JUnit package aligned"]
    assert public["phase_states"][0]["last_event"]["payload"]["learning_notes"] == [{}]
    assert "内部过程指标已隐藏" not in json.dumps(public, ensure_ascii=False)
