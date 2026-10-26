from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from html.parser import HTMLParser
from urllib.parse import urlencode
from urllib.request import urlopen

from graph.experience_memory import AgentExperienceMemory
from monitoring.dashboard_data import build_snapshot, state_model_artifact
from monitoring.registry import ExperimentRegistry
from monitoring.server import DashboardHandler, _public_report_html
from visualization.graph_visualizer import TestStateModelVisualizer
from visualization.knowledge_library_visualizer import TestKnowledgeGuidanceLibraryVisualizer


HIDDEN = re.compile(r"(?i)(?<![a-z0-9])(?:sfc|nrr|tsq|sfq|csg|pv)(?![a-z0-9])")


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_script = False
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        if self.in_script:
            self.scripts[-1] += data


def _scripts(document):
    parser = _Scripts()
    parser.feed(document)
    return parser.scripts


def _legacy_graph(case):
    source = case / "calculator.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    graph_path = case / "state_flow_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "source_path": str(source),
                "version": 3,
                "nodes": {
                    "function": {
                        "node_id": "function",
                        "node_type": "Function",
                        "name": "add",
                        "line_start": 1,
                        "line_end": 2,
                        "visited": True,
                        "defect_score": 0.63,
                        "priority": 5,
                        "metadata": {"pv": 0.91827},
                    }
                },
                "edges": [],
                "flow_history": [
                    {
                        "event_type": "QualityEvaluated",
                        "source_agent": "测试状态智能体",
                        "message": "CSG=0.45671; NRR=1",
                        "payload": {"complete_state_graph": {"csg": 0.45671}, "tsq": 0.71234},
                    }
                ],
                "metadata": {
                    "last_report": {
                        "pytest_passed": True,
                        "collected_count": 1,
                        "line_coverage": 0.9,
                        "branch_coverage": 0.8,
                        "ae": 0.75,
                        "mutation_score": 0.6,
                        "sfc": 0.81913,
                        "nrr": 1,
                        "sfq": 0.71234,
                        "tsq": 0.71234,
                        "pv": 0.91827,
                    },
                    "last_evaluation_feedback": {"baseline_metrics": {"sfc": 0.81913}, "description": "SFC=0.81913"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return graph_path


def test_dashboard_snapshot_and_state_artifact_hide_nested_process_metrics(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    graph_path = _legacy_graph(case)
    record = {"id": "legacy-run", "output_dir": str(tmp_path), "label": "legacy"}

    snapshot = build_snapshot(record, [record])
    artifact = state_model_artifact(graph_path)
    serialized = json.dumps({"snapshot": snapshot, "artifact": artifact}, ensure_ascii=False)

    assert snapshot["current_task"]["metrics"]["line_coverage"] == 0.9
    assert snapshot["state_model"]["nodes"][0]["name"] == "add"
    assert "defect_score" not in serialized and "priority" not in serialized
    assert not HIDDEN.search(serialized)
    assert "0.81913" not in serialized and "0.91827" not in serialized and "0.45671" not in serialized


def test_source_identifiers_are_not_rewritten_as_metric_labels(tmp_path):
    from graph.state_graph import StateFlowGraph

    source = tmp_path / "sample.py"
    source.write_text("def calculate_quality_gain():\n    return 1\n", encoding="utf-8")
    graph = StateFlowGraph(str(source))
    graph.add_entity("Function", "calculate_quality_gain", 1, 2, "def calculate_quality_gain():")
    graph_path = tmp_path / "state_flow_graph.json"
    graph.save(graph_path)
    TestStateModelVisualizer().render(graph_path, tmp_path)

    graph_html = _public_report_html(tmp_path / "state_flow_graph.html")
    source_html = _public_report_html(tmp_path / "source_coverage.html")
    assert "calculate_quality_gain" in graph_html
    assert "def calculate_quality_gain():" in source_html
    assert "无法在可视化平台展示" not in graph_html
    assert "无法在可视化平台展示" not in source_html


def test_new_graph_and_knowledge_html_hide_process_scores(tmp_path, monkeypatch):
    case = tmp_path / "case"
    case.mkdir()
    graph_path = _legacy_graph(case)
    visualizer = TestStateModelVisualizer()
    original_model = visualizer._load_model(graph_path)
    assert visualizer._summary(visualizer._public_model(original_model))["label"] == visualizer._summary(original_model)["label"]
    rendered = visualizer.render(graph_path, case)
    graph_html = (case / "state_flow_graph.html").read_text(encoding="utf-8")
    source_html = (case / "source_coverage.html").read_text(encoding="utf-8")
    svg = (case / "state_flow_graph.svg").read_text(encoding="utf-8")

    assert rendered["status"] == "generated"
    for output in (graph_html, source_html, svg):
        assert not HIDDEN.search(output)
        assert "0.81913" not in output and "0.91827" not in output
        assert "63%" not in output
    assert "LC" in graph_html and "AE" in graph_html
    for document, template in ((graph_html, visualizer._script()), (source_html, visualizer._source_script())):
        generated_scripts = _scripts(document)
        assert generated_scripts == _scripts(template)
        if shutil.which("node"):
            for script in generated_scripts:
                checked = subprocess.run(["node", "--check", "-"], input=script, text=True, capture_output=True, timeout=10)
                assert checked.returncode == 0, checked.stderr

    knowledge_path = tmp_path / "knowledge" / "records.json"
    knowledge_path.parent.mkdir()
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(knowledge_path))
    knowledge_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "experience_id": "item",
                        "timestamp": 1,
                        "source": "calculator.py",
                        "action": "improve SFC=0.81913",
                        "failure_category": "weak_assertions_only",
                        "reward": 0.45671,
                        "learning_notes": [{"solution": "raise TSQ=0.71234"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    knowledge = TestKnowledgeGuidanceLibraryVisualizer().render(AgentExperienceMemory(knowledge_path, mirror_paths=[]))
    knowledge_html = (knowledge_path.parent / "test_knowledge_guidance_library.html").read_text(encoding="utf-8")
    knowledge_json = (knowledge_path.parent / "test_knowledge_guidance_summary.json").read_text(encoding="utf-8")
    assert knowledge["status"] == "generated"
    for output in (knowledge_html, knowledge_json):
        assert not HIDDEN.search(output)
        assert "0.81913" not in output and "0.71234" not in output and "0.45671" not in output
        assert "reward" not in output and "confidence" not in output and "success_rate" not in output


def test_legacy_html_and_artifact_endpoints_do_not_serve_internal_metrics(tmp_path, monkeypatch):
    import monitoring.server as server
    from http.server import ThreadingHTTPServer

    case = tmp_path / "case"
    case.mkdir()
    _legacy_graph(case)
    (tmp_path / "batch_summary.json").write_text(json.dumps({"results": [], "quality_overview": {"average_sfc": 0.81913}}), encoding="utf-8")
    batch_report = tmp_path / "batch_quality_report.html"
    batch_report.write_text("<table><tr><th>SFC</th><td>0.81913</td></tr></table>", encoding="utf-8")
    old_report = tmp_path / "old_report.html"
    old_report.write_text("<p>NRR=1; TSQ=0.71234</p>", encoding="utf-8")
    old_risk_report = tmp_path / "old_risk_report.html"
    old_risk_report.write_text("<p>defect_score=0.92345</p>", encoding="utf-8")
    old_json = tmp_path / "old_result.json"
    old_json.write_text(
        json.dumps({"metrics": {"sfc": 0.81913, "nrr": 1, "tsq": 0.71234}, "line_coverage": 0.9, "message": "CSG=0.45671"}),
        encoding="utf-8",
    )

    rerendered = _public_report_html(batch_report)
    fallback = _public_report_html(old_report)
    risk_fallback = _public_report_html(old_risk_report)
    assert "LC" in rerendered and not HIDDEN.search(rerendered)
    assert "无法在可视化平台展示" in fallback and not HIDDEN.search(fallback)
    assert "无法在可视化平台展示" in risk_fallback and "0.92345" not in risk_fallback
    assert "0.81913" not in rerendered and "0.71234" not in fallback

    registry = ExperimentRegistry(tmp_path / "registry.json")
    experiment = registry.register(str(tmp_path), "legacy")
    monkeypatch.setattr(server, "REGISTRY", registry)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        for report in (old_report, old_risk_report, batch_report):
            params = urlencode({"experiment": experiment["id"], "path": str(report)})
            with urlopen(f"{base}/api/artifact/raw?{params}", timeout=5) as response:
                body = response.read().decode("utf-8")
            assert not HIDDEN.search(body)
            assert "0.81913" not in body and "0.71234" not in body and "0.92345" not in body

        params = urlencode({"experiment": experiment["id"], "path": str(old_json)})
        with urlopen(f"{base}/api/artifact?{params}", timeout=5) as response:
            payload = json.load(response)
        serialized = json.dumps(payload, ensure_ascii=False)
        assert payload["data"]["line_coverage"] == 0.9
        assert not HIDDEN.search(serialized)
        assert "0.81913" not in serialized and "0.71234" not in serialized and "0.45671" not in serialized

        with urlopen(f"{base}/api/events?{urlencode({'experiment': experiment['id']})}", timeout=5) as response:
            data = next(line for line in response if line.startswith(b"data: "))
        event = data.decode("utf-8")[len("data: "):]
        assert not HIDDEN.search(event)
        assert "0.81913" not in event and "0.71234" not in event and "0.45671" not in event
        assert json.loads(event)["current_task"]["metrics"]["line_coverage"] == 0.9
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
