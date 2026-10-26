from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

import batch_generate_tests as batch
from agents.execution_agent import ExecutionAgent
from agents.java_test_renderer import JavaJUnitTestRenderer
from agents.test_knowledge_agent import TestKnowledgeAgent as KnowledgeAgent
from graph.experience_memory import AgentExperienceMemory
from llm_config import LLMConfig
from memory.shared_graph import SharedGraphMemory
from graph.state_graph import StateFlowGraph
from monitoring.learning_queue import PendingKnowledgeLearningQueue
from executor.java_process import run_java_process
from main import MultiAgentUnitTestSystem


def test_auto_humanevaljava_freezes_knowledge_and_reports_java_slots(tmp_path, monkeypatch):
    sources = tmp_path / "HumanEvalJava"
    sources.mkdir()
    for name in ("A", "B"):
        (sources / f"{name}.java").write_text(f"public class {name} {{ public int value() {{ return 1; }} }}", encoding="utf-8")
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return batch.BatchItemResult(
            source=str(kwargs["source_path"]), output_dir=str(kwargs["case_dir"]),
            status="quality_high", final_report={"pytest_passed": True, "quality_status": "quality_high"},
        )

    monkeypatch.setattr(batch, "_generate_and_execute", execute)
    result = batch.generate_tests_for_directory(
        sources, tmp_path / "out", execute=True, language="auto",
        retry_failed_rounds=0, llm_config=LLMConfig(enabled=False),
    )
    assert result["language"] == "java"
    assert len({str(call["experience_read_path"]) for call in calls}) == 1
    assert all(call["experience_read_path"] is not None for call in calls)
    assert len({str(call["experience_journal_dir"]) for call in calls}) == 2
    runtime = json.loads((tmp_path / "out" / "batch_runtime.json").read_text("utf-8"))
    assert runtime["java_parallel"] is True
    assert runtime["python_parallel"] is False
    assert result["dataset_integrity"]["reference_expected_tasks"] == 164
    assert result["dataset_completion"]["skip_ready"] is False


def test_direct_single_java_does_not_implicitly_enable_batch_workspace(tmp_path):
    source = tmp_path / "HumanEvalJava" / "A.java"
    source.parent.mkdir()
    source.write_text("public class A {}", encoding="utf-8")
    agent = ExecutionAgent(SharedGraphMemory(tmp_path / "out" / "graph.json"))
    assert agent._human_eval_java_workspace(source) is None


def test_round_merge_keeps_existing_concept_updates_and_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "knowledge.json"
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(path))
    live = AgentExperienceMemory(path)
    record = live.append_conversation_concept("怎样提高断言质量？", "精确断言", force=True)
    output = tmp_path / "out"
    isolation = batch._prepare_round_knowledge_isolation(output, experiment_round=1, enabled=True)
    cases = [output / "A", output / "B"]
    for case in cases:
        memory = AgentExperienceMemory(path, read_base_path=isolation.base_path,
            journal_dir=isolation.journal_dir(case), publish_canonical=False)
        memory.validate_conversation_concepts([record["experience_id"]], successful=True)
        assert next(r for r in memory.load() if r["experience_id"] == record["experience_id"])["validation_count"] == 1
    batch._merge_round_knowledge(isolation, cases)
    merged = next(r for r in live.load() if r["experience_id"] == record["experience_id"])
    assert merged["validation_count"] == 2
    assert merged["learning_stage"] == "mastered"
    batch._merge_round_knowledge(isolation, cases)
    assert next(r for r in live.load() if r["experience_id"] == record["experience_id"])["validation_count"] == 2


def test_interrupted_round_journal_is_recovered(tmp_path, monkeypatch):
    path = tmp_path / "knowledge.json"
    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(path))
    output = tmp_path / "out"
    isolation = batch._prepare_round_knowledge_isolation(output, experiment_round=2, enabled=True)
    case = output / "A"
    memory = AgentExperienceMemory(path, read_base_path=isolation.base_path,
        journal_dir=isolation.journal_dir(case), publish_canonical=False)
    memory.append_conversation_concept("怎样设计JUnit测试？", "先验证正常行为", force=True)
    assert not AgentExperienceMemory(path).load()
    batch._recover_round_knowledge(output)
    assert len(AgentExperienceMemory(path).load()) == 1
    batch._recover_round_knowledge(output)
    assert len(AgentExperienceMemory(path).load()) == 1


def test_learning_write_failure_cannot_be_reported_as_learned(tmp_path, monkeypatch):
    def fail_write(self, records):
        self.last_write_errors = [{"error": "disk unavailable"}]
    monkeypatch.setattr(AgentExperienceMemory, "_write_records", fail_write)
    with pytest.raises(OSError, match="未成功写入"):
        KnowledgeAgent.learn_from_conversation("怎样设计测试？", "验证结果",
            model="test", conversation_id="a", knowledge_store=tmp_path / "kb.json", force=True)


def test_pending_questions_keep_all_chats_and_are_not_silently_evicted(tmp_path):
    queue = PendingKnowledgeLearningQueue(tmp_path / "queue.json")
    args = dict(experiment_id="run", knowledge_store=tmp_path / "kb.json", context_label="test")
    a = queue.enqueue("问题", chat_id="a", **args)
    b = queue.enqueue("问题", chat_id="b", **args)
    assert a["id"] != b["id"]
    queue._data["items"].extend({"id": f"p{i}", "status": "pending"} for i in range(500))
    queue.enqueue("新问题", chat_id="c", **args)
    assert len(queue.list()) == 503
    with pytest.raises(ValueError):
        queue.complete(a["id"], answer="未落盘", record_id=None)


def test_long_catalog_does_not_cut_off_knowledge_or_conversation_history():
    captured = []
    class Model:
        def enabled(self):
            return True
        def chat(self, messages):
            captured.append(messages[-1]["content"])
            return "模型答复"
    result = KnowledgeAgent.answer_question("怎样提高PIT变异率？",
        snapshot={"catalog": {"component_map": [{"description": "x" * 50000}] * 50}},
        knowledge={"path": "kb", "records": [{"topic": "PIT变异", "solution": "MUST_KEEP_KNOWLEDGE"}]},
        llm=Model(), conversation_history=[{"role": "user", "content": "PREVIOUS_MESSAGE"}])
    payload = json.loads(captured[0].split("可用事实与知识检索结果：", 1)[1])
    assert "MUST_KEEP_KNOWLEDGE" in str(payload["relevant_knowledge"])
    assert "PREVIOUS_MESSAGE" in str(payload["conversation_history"])
    assert result["model_used"] is True


def test_unknown_java_concept_is_queued_instead_of_generic_dataset_answer():
    result = KnowledgeAgent.answer_question("请解释Java ZetaProbe协议的原理",
        snapshot={}, knowledge={"records": []}, llm=None)
    assert result["needs_learning"] is True


def test_knowledge_answer_uses_learned_solution_not_current_task_metrics():
    result = KnowledgeAgent.answer_question("怎样提高PIT变异率？",
        snapshot={"current_task": {"name": "A.java", "metrics": {"mutation_score": 0.1}}},
        knowledge={"records": [{"topic": "PIT变异", "solution": "对存活变异增加精确断言", "status": "正在学习"}]},
        llm=None)
    assert "存活变异" in result["answer"]
    assert result["response_mode"] == "knowledge_autonomous"


@pytest.mark.parametrize("grade,round_number,expected", [
    ("A", 2, False), ("B", 2, False), ("C", 2, False), ("D", 2, True), ("E", 2, True),
    ("A", 3, False), ("B", 3, False), ("C", 3, True), ("D", 3, False), ("E", 3, False),
    ("C", 4, False), ("D", 4, False),
])
def test_formal_round_routing_matrix(grade, round_number, expected):
    item = batch.BatchItemResult(source="task.py", output_dir="out", status="test", final_grade=grade)
    assert batch._should_enter_experiment_round(item, round_number) is expected


@pytest.mark.parametrize("hint,expected", [
    ("ArrayList<Integer>", "new java.util.ArrayList<>()"),
    ("List<String>", "new java.util.ArrayList<>()"),
    ("String[]", "new String[]{}"), ("HashMap<String, Integer>", "new java.util.HashMap<>()"),
    ("float", "1.0f"), ("short", "(short) 1"),
])
def test_java_default_arguments_match_declared_types(hint, expected):
    assert JavaJUnitTestRenderer()._default_value({"type_hint": hint}) == expected


def test_humanevaljava_named_build_project_stays_serial(tmp_path):
    root = tmp_path / "HumanEvalJava"
    root.mkdir()
    (root / "pom.xml").write_text("<project/>", encoding="utf-8")
    sources = [root / "A.java", root / "B.java"]
    assert batch._effective_worker_count("java", sources, 2, False) == 1
    assert not batch._is_humaneval_java_class_dataset(root, sources, "auto")


def test_modified_source_cannot_reuse_new_summary(tmp_path):
    source = tmp_path / "target.py"
    source.write_text("def f(): return 1", encoding="utf-8")
    output = tmp_path / "out"
    output.mkdir()
    (output / "stateflow_summary.json").write_text(json.dumps({
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "quality_status": "quality_high", "final_report": {"pytest_passed": True},
    }), encoding="utf-8")
    source.write_text("def f(): return 2", encoding="utf-8")
    assert MultiAgentUnitTestSystem(source, output)._load_existing_summary() is None
    assert batch._resume_existing_item(source, output, execute=True) is None


def test_java_timeout_terminates_spawned_child(tmp_path):
    marker = tmp_path / "orphan.txt"
    child = "import time,pathlib; time.sleep(1.5); pathlib.Path(" + repr(str(marker)) + ").write_text('orphan')"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(10)"
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_java_process([sys.executable, "-c", parent], cwd=tmp_path, timeout=0.3)
    assert time.monotonic() - started < 8
    time.sleep(1.6)
    assert not marker.exists()


def test_java_pending_generation_is_not_written_as_negative_knowledge(tmp_path):
    from types import SimpleNamespace
    agent = object.__new__(ExecutionAgent)
    graph = SimpleNamespace(metadata={"test_generation_validation": {"stage": "pending_compile_and_intent_validation"}})
    class Memory:
        def append_candidate(self, **kwargs):
            raise AssertionError("pending candidate must not become negative experience")
    agent._record_generation_experience(graph, tmp_path / "A.java", Memory())


def test_screening_and_duplicate_evidence_do_not_promote_knowledge(tmp_path):
    experience = AgentExperienceMemory(tmp_path / "knowledge.json")
    record = experience.append_conversation_concept("怎样提高测试覆盖？", "增加边界输入", force=True)
    graph = StateFlowGraph(source_path=str(tmp_path / "A.java"))
    graph.metadata["last_knowledge_context"] = {"selected_ids": [record["experience_id"]]}
    memory = SharedGraphMemory(tmp_path / "graph.json")
    memory.set(graph)
    agent = KnowledgeAgent(memory, experience)
    report = {"pytest_passed": True, "quality_status": "valid_usable", "evaluation_profile": "screening"}
    agent.record_outcome(report)
    agent.record_outcome(report)
    assert experience.load()[0]["validation_count"] == 0
    report["evaluation_profile"] = "full"
    agent.record_outcome(report)
    agent.record_outcome(report)
    assert experience.load()[0]["validation_count"] == 1
