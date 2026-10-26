"""Recovered knowledge from an imported run stays distinct from the live store."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from urllib.parse import urlencode
from urllib.request import urlopen

import monitoring.server as server
from agents.test_knowledge_agent import TestKnowledgeAgent as KnowledgeAgent
from monitoring.dashboard_data import experiment_id
from monitoring.registry import ExperimentRegistry
from monitoring.run_knowledge import (_logical_source, _project, _record_status,
                                      imported_knowledge_store, imported_knowledge_snapshot)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _imported_run(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    _write(root / "all_datasets_summary.json", {
        "started_at": "2026-09-07T08:00:00+00:00",
        "finished_at": "2026-09-08T14:00:00+00:00",
        "test_knowledge_guidance_library": {"knowledge": r"D:\missing\test_knowledge_guidance.json"},
        "results": [{"dataset": {"name": "HumanEvalJava"}}, {"dataset": {"name": "Defects4J_cli"}}],
    })
    java = root / "HumanEvalJava"
    _write(java / "batch_summary.json", {
        "batch_experience": {"timestamp": 1788810000, "learning_notes": [
            {"reason_type": "java_compile_error", "cause": "JUnit import failed",
             "solution": "Align the JUnit import.", "evidence": {"count": 2}},
        ]},
    })
    _write(root / "Defects4J_cli" / "batch_summary.json", {
        "batch_experience": {"timestamp": 1788820000, "learning_notes": [
            {"reason_type": "weak_assertion", "cause": "Weak assertion",
             "solution": "Use concrete expected values.", "evidence": {"count": 3}},
        ]},
    })
    baseline = {"experience_id": "old", "timestamp": 1788000000, "model": "deepseek-r1:14b",
                "source": r"D:\old\datasets\Defects4J\cli\Old.java", "action": "old advice"}
    current = {"experience_id": "current", "timestamp": 1788810001, "model": "deepseek-r1:14b",
               "source": r"D:\old\datasets\HumanEvalJava\Current.java", "action": "current advice",
               "learning_notes": [{"solution": "Exercise the exception path."}]}
    _write(java / ".knowledge_snapshots" / "R3.json", {"records": [baseline, current]})
    _write(java / "Current" / ".knowledge_journals" / "R3" / "records-1.json", {
        "records": [current, {"experience_id": "last", "timestamp": 1788811000,
                              "model": "deepseek-r1:14b", "source": r"D:\old\datasets\HumanEvalJava\Last.java",
                              "action": "last advice"}],
    })
    return root


def test_imported_run_recovers_snapshot_journals_and_dataset_notes(tmp_path):
    root = _imported_run(tmp_path)
    store = imported_knowledge_store(root, tmp_path / "cache")
    snapshot = imported_knowledge_snapshot(root, store)

    assert store is not None and store.is_file()
    assert snapshot["scope"] == "imported_run"
    assert snapshot["availability"]["complete"] is False
    assert snapshot["availability"]["recovered_snapshot_records"] == 2
    assert snapshot["availability"]["recovered_journal_records"] == 2
    assert snapshot["availability"]["run_record_count"] == 2
    assert snapshot["availability"]["baseline_record_count"] == 1
    assert snapshot["availability"]["dataset_summary_count"] == 2
    assert snapshot["availability"]["missing_datasets"] == ["Defects4J_cli"]
    assert snapshot["total"] == 5
    assert snapshot["counts"]["经验摘要"] == 2
    by_id = {record["id"]: record for record in snapshot["records"]}
    assert by_id["current"]["source_type"] == "run_record"
    assert by_id["current"]["source"] == "HumanEvalJava/Current.java"
    assert by_id["current"]["solution"] == "Exercise the exception path."
    assert by_id["old"]["source_type"] == "historical_baseline"
    assert by_id["old"]["source"] == "Defects4J_cli/Old.java"
    assert "D:\\old" not in json.dumps(snapshot)


def test_imported_run_knowledge_api_and_agent_use_same_store(tmp_path, monkeypatch):
    root = _imported_run(tmp_path)
    registry = ExperimentRegistry(tmp_path / "registry.json")
    master = registry.register(str(root), "DeepSeek-R1", metadata={"execution_status": "completed"})
    monkeypatch.setattr(server, "REGISTRY", registry)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        for item_id in (master["id"], experiment_id(root / "HumanEvalJava")):
            with urlopen(f"{base}/api/knowledge?{urlencode({'experiment': item_id})}") as response:
                payload = json.load(response)
            assert payload["total"] == 5
            assert payload["scope"] == "imported_run"
            handler = server.DashboardHandler.__new__(server.DashboardHandler)
            store = handler._knowledge_store(item_id)
            assert "path" not in payload
            assert handler._knowledge_snapshot_for_store(store)["path"] == str(store)
            assert "record_type" not in payload["records"][0]
            assert [item["id"] for item in handler._knowledge_snapshot_for_store(store)["records"]] == [item["id"] for item in payload["records"]]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_imported_run_has_no_arbitrary_500_record_limit(tmp_path):
    root = _imported_run(tmp_path)
    snapshot_path = root / "HumanEvalJava" / ".knowledge_snapshots" / "R3.json"
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    data["records"].extend({"experience_id": f"extra-{index}", "timestamp": 1788810200 + index / 1000,
                            "source": r"D:\old\datasets\HumanEvalJava\Bulk.java", "action": "bulk"}
                           for index in range(501))
    _write(snapshot_path, data)
    store = imported_knowledge_store(root, tmp_path / "cache")
    payload = imported_knowledge_snapshot(root, store)
    assert payload["total"] == 506
    assert len(payload["records"]) == 506


def test_imported_run_new_conversation_is_separate_from_original_run(tmp_path):
    root = _imported_run(tmp_path)
    store = imported_knowledge_store(root, tmp_path / "cache")
    before = imported_knowledge_snapshot(root, store)
    _write(store.with_suffix(store.suffix + ".records") / "records-1.json", {
        "records": [{"experience_id": "later", "timestamp": 1789000000,
                     "record_type": "conversation_concept", "topic": "new concept"}],
    })
    after = imported_knowledge_snapshot(root, store)
    assert after["total"] == before["total"] + 1
    assert after["availability"]["run_record_count"] == before["availability"]["run_record_count"]
    assert after["availability"]["post_import_learning_count"] == 1
    assert after["records"][0]["source_type"] == "post_import_learning"


def test_earlier_external_store_is_counted_as_baseline_not_new_run_knowledge(tmp_path):
    root = _imported_run(tmp_path)
    old_store = tmp_path / "old-baseline.json"
    _write(old_store, {"records": [{"experience_id": "old", "timestamp": 1788000000,
                                    "source": r"D:\old\datasets\Defects4J\cli\Old.java"}]})
    master_path = root / "all_datasets_summary.json"
    master = json.loads(master_path.read_text(encoding="utf-8"))
    master["test_knowledge_guidance_library"]["knowledge"] = str(old_store)
    _write(master_path, master)
    store = imported_knowledge_store(root, tmp_path / "cache")
    availability = imported_knowledge_snapshot(root, store)["availability"]
    assert availability["external_baseline_available"] is True
    assert availability["external_baseline_record_count"] == 1
    assert availability["external_baseline_covered_count"] == 1
    assert availability["baseline_record_count"] == 1


def test_imported_knowledge_projection_omits_internal_metric_words():
    record = _project({
        "experience_id": "metric-note", "source_type": "run_record",
        "action": "Improve SFC and TSQ", "solution": "SFC 0.8; add boundary cases",
        "topic": "NRR", "learning_notes": [{"solution": "PV 0.2; add meaningful assertions"}],
    })
    visible = json.dumps(record, ensure_ascii=False).casefold()
    for token in ("sfc", "nrr", "tsq", "sfq", "csg", "pv", "内部过程指标已隐藏"):
        assert token not in visible


def test_dataset_summary_process_reasons_leave_a_useful_solution(tmp_path):
    root = _imported_run(tmp_path)
    batch_path = root / "Defects4J_cli" / "batch_summary.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    batch["batch_experience"]["learning_notes"] = [
        {"reason_type": "line_coverage_gap", "cause": "line_coverage_gap remained below target",
         "solution": "Target uncovered methods with small concrete examples."},
        {"reason_type": "ae_gap", "cause": "ae_gap remained below target",
         "solution": "Assert concrete return values."},
        {"reason_type": "weak_overall_quality_index", "cause": "The combined score remains below target.",
         "solution": "Preserve passing tests and add boundary examples."},
        {"reason_type": "weak_branch_coverage", "cause": "内部过程信息已隐藏",
         "solution": "Add inputs for both branch outcomes."},
    ]
    _write(batch_path, batch)
    store = imported_knowledge_store(root, tmp_path / "cache")
    records = [item for item in imported_knowledge_snapshot(root, store)["records"]
               if item["source_type"] == "dataset_summary" and item["dataset"] == "Defects4J_cli"]
    assert len(records) == 4
    for index, record in enumerate(reversed(records), 1):
        assert record["topic"] == f"Defects4J_cli 经验摘要 {index}"
        assert record["solution"]
        visible = json.dumps(record, ensure_ascii=False).casefold()
        for fragment in ("line_", "branch_", "ae_gap", "combined score", "内部过程信息已隐藏"):
            assert fragment not in visible


def test_mastered_status_requires_explicit_learning_or_repeat_evidence():
    assert _record_status({"accepted": True, "reward": 0}) == "待复核"
    assert _record_status({"accepted": True, "validation_count": 2}) == "已掌握"
    assert _record_status({"accepted": True, "support_count": 2,
                           "independent_task_count": 2}) == "已掌握"
    assert _record_status({"learning_stage": "mastered"}) == "已掌握"
    assert _record_status({"accepted": True, "rejection_reason": "failed"}) == "应避免"


def test_historical_source_uses_real_dataset_label():
    assert _logical_source("historical:/runs.rar/TestEval") == ("TestEval/历史知识", "TestEval")
    assert _logical_source("historical:/runs.rar/unknown") == ("历史知识记录", "")


def test_imported_run_discovers_newly_copied_journal_file(tmp_path):
    root = _imported_run(tmp_path)
    store = imported_knowledge_store(root, tmp_path / "cache")
    before = imported_knowledge_snapshot(root, store)
    _write(root / "HumanEvalJava" / "Later" / ".knowledge_journals" / "R3" / "records-2.json", {
        "records": [{"experience_id": "copied-later", "timestamp": 1788811010,
                     "source": r"D:\old\datasets\HumanEvalJava\Later.java"}],
    })
    assert imported_knowledge_store(root, tmp_path / "cache") == store
    after = imported_knowledge_snapshot(root, store)
    assert after["total"] == before["total"] + 1
    assert any(item["id"] == "copied-later" for item in after["records"])


def test_new_knowledge_agent_learning_preserves_all_imported_records(tmp_path):
    root = _imported_run(tmp_path)
    snapshot_path = root / "HumanEvalJava" / ".knowledge_snapshots" / "R3.json"
    data = json.loads(snapshot_path.read_text(encoding="utf-8"))
    data["records"].extend({"experience_id": f"extra-{index}", "timestamp": 1788810200 + index / 1000,
                            "source": r"D:\old\datasets\HumanEvalJava\Bulk.java", "action": "bulk"}
                           for index in range(3729))
    _write(snapshot_path, data)
    store = imported_knowledge_store(root, tmp_path / "cache")
    before = imported_knowledge_snapshot(root, store)
    assert before["total"] == 3734
    old_ids = {record["id"] for record in before["records"]}

    result = KnowledgeAgent.learn_from_conversation(
        "How should a Java unit test check an exception path?",
        "Use an exact exception assertion for the invalid input.",
        model="test-model", conversation_id="after-import", knowledge_store=store, force=True,
    )
    assert result is not None
    after = imported_knowledge_snapshot(root, store)
    assert after["total"] == 3735
    assert old_ids.issubset({record["id"] for record in after["records"]})
