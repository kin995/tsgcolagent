from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agents.test_knowledge_agent import TestKnowledgeAgent as KnowledgeAgent
from graph.experience_memory import AgentExperienceMemory
from llm_client import OpenAICompatibleLLM
from llm_config import LLMConfig
from monitoring.dashboard_data import _experiment_timing, build_snapshot, historical_task_duration, knowledge_snapshot, state_model_artifact
from monitoring.chat_store import ChatStore
from monitoring.job_queue import ExperimentJob, ExperimentJobQueue
from monitoring.learning_queue import PendingKnowledgeLearningQueue
from monitoring.registry import ExperimentRegistry
from monitoring.server import _portfolio_timing
from monitoring.system_catalog import system_catalog


class FakeKnowledgeLLM:
    def enabled(self):
        return True

    def chat(self, messages):
        prompt = messages[-1]["content"]
        if "当前覆盖率" in prompt:
            return "当前任务行覆盖率100.00%，变异得分75.00%。"
        if "你是谁" in prompt:
            return "我是测试知识智能体，与测试状态智能体、测试生成智能体协作；R2恢复D/E，R3优化C，我不直接修改测试文件。"
        if "代码结构" in prompt:
            return "A-E分级位于core/experiment_outcome.py和agents/test_state_agent.py。"
        if "单元测试的概念" in prompt:
            return "单元测试通过准备输入、执行目标和精确断言验证行为，并用变异检测衡量缺陷发现能力。"
        return "已结合模型和知识库回答。"


def test_single_task_duration_is_not_divided_by_idle_worker_slots():
    tasks = [{"source": "source.py", "duration_seconds": 75.8353}]
    assert historical_task_duration(tasks, None) == 75.835


def test_imported_completed_task_without_finish_timestamp_uses_recorded_time():
    timing = _experiment_timing(
        {"execution_status": "completed", "registered_at": "2020-01-01T00:00:00+00:00"},
        {}, {}, {}, "completed",
        tasks=[{"source": "source.py", "duration_seconds": 75.8353}],
    )
    assert timing["duration_seconds"] == 75.835
    assert timing["duration_source"] == "historical_task_runtime"


def test_portfolio_uses_completed_experiment_duration_without_finish_timestamp():
    timing = _portfolio_timing(
        [{"status": "completed", "started_at": "2026-09-23T14:14:25+00:00", "finished_at": None, "duration_seconds": 75.835}],
        "completed",
    )
    assert timing["duration_seconds"] == 75.835


def test_dashboard_snapshot_uses_experiment_round_not_internal_repairs(tmp_path):
    case = tmp_path / "Calculator"
    case.mkdir()
    graph_path = case / "state_flow_graph.json"
    summary_path = case / "stateflow_summary.json"
    graph_path.write_text(
        json.dumps(
            {
                "source_path": "Calculator.java",
                "version": 3,
                "iteration": 2,
                "nodes": {"n1": {"node_type": "Method", "name": "add", "visited": True}},
                "edges": [],
                "flow_history": [{"event_type": "KnowledgeRetrieved", "source_agent": "测试知识智能体", "target_agent": "测试状态智能体"}],
                "metadata": {"language": "java"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "source": "Calculator.java",
                "graph_path": str(graph_path),
                "repair_rounds_used": 2,
                "quality_status": "quality_acceptable",
                "final_report": {"pytest_passed": True, "collected_count": 4, "line_coverage": 1.0, "mutation_score": 0.75, "failure_category": "valid_baseline"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_summary.json").write_text(
        json.dumps(
            {
                "language": "java",
                "results": [{"output_dir": str(case), "summary_path": str(summary_path), "final_grade": "B", "experiment_round": 1, "status": "quality_acceptable"}],
                "outcome_summary": {"total_tasks": 1, "completed_execution": 1, "not_completed_execution": 0, "grade_counts": {"A": 0, "B": 1, "C": 0, "D": 0, "E": 0}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dataset_completion.json").write_text(json.dumps({"skip_ready": True}), encoding="utf-8")
    record = {"id": "run", "output_dir": str(tmp_path), "label": "Java", "registered_at": "now"}

    snapshot = build_snapshot(record, [record])

    assert snapshot["overview"]["status"] == "completed"
    assert snapshot["current_task"]["round_label"] == "B·R1"
    assert snapshot["current_task"]["metrics"]["line_coverage"] == 1.0
    assert snapshot["current_task"]["failure"]["reason"] is None
    assert [item["name"] for item in snapshot["agents"]] == ["测试状态智能体", "测试生成智能体", "测试知识智能体"]
    state_agent, _, knowledge_agent = snapshot["agents"]
    assert state_agent["current_action"] == "KnowledgeRetrieved"
    assert state_agent["interaction_role"] == "received"
    assert state_agent["event_count"] == 0
    assert knowledge_agent["interaction_role"] == "executed"
    assert knowledge_agent["event_count"] == 1

    artifact = state_model_artifact(graph_path)
    assert artifact["version"] == 3
    assert artifact["nodes"][0]["name"] == "add"
    assert artifact["nodes"][0]["evidenced"] is True


def test_dashboard_uses_live_round_progress_before_batch_round_finishes(tmp_path):
    case = tmp_path / "TestEvatask_002"
    case.mkdir()
    summary_path = case / "stateflow_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": "TestEvatask_002.py",
                "quality_status": "valid_usable",
                "final_grade": "C",
                "experiment_round": 1,
                "final_report": {"pytest_passed": True, "line_coverage": 0.73, "ae": 0.9},
            }
        ),
        encoding="utf-8",
    )
    (case / "experiment_round_progress.json").write_text(
        json.dumps(
            {
                "source": "TestEvatask_002.py",
                "experiment_round": 2,
                "status": "running",
                "strategy": "optimize_existing",
                "previous_grade": "C",
                "final_grade": "C",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_summary.json").write_text(
        json.dumps(
            {
                "language": "python",
                "results": [{"output_dir": str(case), "summary_path": str(summary_path), "final_grade": "C", "experiment_round": 1}],
            }
        ),
        encoding="utf-8",
    )
    record = {"id": "live-r2", "output_dir": str(tmp_path), "label": "TestEval", "registered_at": "now"}

    task = build_snapshot(record, [record])["current_task"]

    assert task["round_label"] == "C·R2"
    assert task["status"] == "running"
    assert task["round_strategy"] == "optimize_existing"


def test_dashboard_reports_humanevaljava_isolated_workers(tmp_path):
    (tmp_path / "batch_summary.json").write_text(json.dumps({"language": "java", "total": 2}), encoding="utf-8")
    (tmp_path / "batch_runtime.json").write_text(
        json.dumps({"status": "running", "language": "java", "task_workers": 2, "java_parallel": True, "python_parallel": False, "selected_tasks": 2}),
        encoding="utf-8",
    )
    snapshot = build_snapshot({"id": "human-java", "output_dir": str(tmp_path), "label": "HumanEvalJava", "execution_status": "running"}, [])

    assert snapshot["parallel_execution"]["task_workers"] == 2
    assert snapshot["parallel_execution"]["java_parallel"] is True
    assert snapshot["parallel_execution"]["python_parallel"] is False


def test_knowledge_agent_factual_answer_is_read_only():
    response = KnowledgeAgent.answer_question(
        "当前覆盖率和变异得分是多少？",
        snapshot={
            "experiment": {"id": "run"},
            "overview": {},
            "current_task": {
                "name": "Calculator.java",
                "grade": "B",
                "round": 1,
                "metrics": {"line_coverage": 1.0, "branch_coverage": 0.9, "sfc": 0.8, "ae": 0.85, "mutation_score": 0.75, "tsq": 0.82},
                "failure": {},
                "summary_path": "summary.json",
            },
        },
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=FakeKnowledgeLLM(),
    )

    assert response["agent"] == "测试知识智能体"
    assert response["read_only"] is False
    assert response["system_assistant"] is True
    assert response["knowledge_learning_enabled"] is True
    assert "100.00%" in response["answer"]
    assert "75.00%" in response["answer"]


def test_knowledge_agent_model_prompt_uses_only_public_evaluation_metrics():
    captured = []

    class CaptureLLM:
        def enabled(self):
            return True

        def chat(self, messages):
            captured.extend(messages)
            return "已依据公开评价指标回答。"

    KnowledgeAgent.answer_question(
        "请解释当前实验的评价指标。",
        snapshot={"experiment": {"id": "run"}, "overview": {}, "current_task": {}, "catalog": system_catalog()},
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=CaptureLLM(),
    )

    system_prompt = captured[0]["content"]
    for name in ("LC", "BC", "AE", "MS", "PassRate", "AvgExec", "Time", "TIR"):
        assert name in system_prompt
    for hidden in ("overall_quality_index", "SFC", "NRR", "TSQ", "SFQ", "CSG", "PV"):
        assert hidden not in system_prompt


def test_knowledge_agent_understands_its_system_identity_and_round_policy():
    response = KnowledgeAgent.answer_question(
        "你是谁？三个智能体和R1、R2、R3分别做什么？",
        snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=FakeKnowledgeLLM(),
    )

    answer = response["answer"]
    assert "测试知识智能体" in answer
    assert "测试状态智能体" in answer
    assert "测试生成智能体" in answer
    assert "R2" in answer and "D/E" in answer
    assert "R3" in answer and "C" in answer
    assert "不直接修改测试文件" in answer

    code_answer = KnowledgeAgent.answer_question(
        "代码结构中哪个模块实现A-E分级？",
        snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=FakeKnowledgeLLM(),
    )["answer"]
    assert "core/experiment_outcome.py" in code_answer
    assert "agents/test_state_agent.py" in code_answer


def test_knowledge_agent_answers_open_unit_test_question_with_model_and_knowledge():
    response = KnowledgeAgent.answer_question(
        "简单介绍一下单元测试的概念和原理。",
        snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=FakeKnowledgeLLM(),
    )

    assert "准备输入" in response["answer"]
    assert "精确断言" in response["answer"]
    assert "变异检测" in response["answer"]
    assert response["model_used"] is True
    assert response["knowledge_used"] is True


def test_knowledge_agent_answers_autonomously_when_model_is_unavailable():
    response = KnowledgeAgent.answer_question(
        "简单介绍单元测试的概念和原理",
        snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=None,
    )

    assert response["response_mode"] == "knowledge_autonomous"
    assert response["model_used"] is False
    assert response["knowledge_used"] is True
    assert response["needs_learning"] is False
    assert "单元测试" in response["answer"]


def test_unknown_question_enters_pending_learning_when_model_is_unavailable():
    response = KnowledgeAgent.answer_question(
        "请解释ZetaProbe协议的可靠性证明",
        snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
        knowledge={"path": "knowledge.json", "counts": {}, "records": []},
        llm=None,
    )

    assert response["response_mode"] == "pending_learning"
    assert response["needs_learning"] is True
    assert "已经记住这个问题" in response["answer"]


def test_knowledge_agent_chinese_query_retrieves_matching_guidance_and_calls_llm():
    calls = []

    class FakeLLM:
        def enabled(self):
            return True

        def chat(self, messages):
            calls.append(messages)
            return "应针对PIT存活变异补充分支输入和精确断言。"

    response = KnowledgeAgent.answer_question(
        "怎样提高Java变异检测率？",
        snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
        knowledge={
            "path": "knowledge.json",
            "counts": {},
            "records": [
                {"id": "pit", "topic": "Java PIT变异检测", "solution": "针对存活变异体补充分支输入和精确断言"},
                {"id": "unrelated", "topic": "页面配色", "solution": "修改颜色"},
            ],
        },
        llm=FakeLLM(),
    )

    assert response["model_used"] is True
    assert len(calls) == 1
    assert "针对存活变异体" in calls[0][1]["content"]
    assert "页面配色" not in calls[0][1]["content"]


def test_real_openai_compatible_client_receives_retrieved_knowledge():
    captured = []

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            captured.append(json.loads(self.rfile.read(length).decode("utf-8")))
            body = json.dumps({"choices": [{"message": {"content": "模型已依据存活变异知识给出回答"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    model_server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    worker = threading.Thread(target=model_server.serve_forever, daemon=True)
    worker.start()
    try:
        llm = OpenAICompatibleLLM(
            LLMConfig(enabled=True, provider="openai", base_url=f"http://127.0.0.1:{model_server.server_port}", model="integration-model", api_key="test")
        )
        response = KnowledgeAgent.answer_question(
            "怎样提高Java变异检测率？",
            snapshot={"experiment": None, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
            knowledge={
                "path": "knowledge.json",
                "stored_record_count": 1,
                "counts": {"已掌握": 1},
                "records": [{"id": "pit", "topic": "Java PIT变异检测", "solution": "针对存活变异体补充分支输入和精确断言"}],
            },
            llm=llm,
        )
    finally:
        model_server.shutdown()
        model_server.server_close()

    assert response["model_used"] is True
    assert response["knowledge_used"] is True
    assert response["knowledge_retrieval"]["matched_record_count"] == 1
    assert "针对存活变异体" in captured[0]["messages"][1]["content"]


def test_pending_learning_is_persisted_and_completed_after_model_recovers(tmp_path, monkeypatch):
    import monitoring.server as server

    queue = PendingKnowledgeLearningQueue(tmp_path / "pending.json")
    chats = ChatStore(tmp_path / "chats.json")
    chat = chats.create()
    knowledge_path = tmp_path / "knowledge.json"
    pending = queue.enqueue(
        "请解释ZetaProbe协议的可靠性证明",
        chat_id=chat["id"],
        experiment_id="run",
        knowledge_store=knowledge_path,
        context_label="自动学习验证",
    )
    assert PendingKnowledgeLearningQueue(tmp_path / "pending.json").list()[0]["status"] == "pending"

    class ModelHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"choices": [{"message": {"content": "ZetaProbe应通过确定性输入和可复现断言验证可靠性。"}}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            return

    model_server = ThreadingHTTPServer(("127.0.0.1", 0), ModelHandler)
    worker = threading.Thread(target=model_server.serve_forever, daemon=True)
    worker.start()
    monkeypatch.setattr(server, "LEARNING_QUEUE", queue)
    monkeypatch.setattr(server, "CHATS", chats)
    monkeypatch.setattr(
        server.DashboardHandler,
        "_llm_config",
        lambda _self, _id: LLMConfig(enabled=True, required=True, provider="openai", base_url=f"http://127.0.0.1:{model_server.server_port}", model="learning-model", api_key="test"),
    )
    monkeypatch.setattr(
        server.DashboardHandler,
        "_snapshot",
        lambda _self, _id: {"experiment": {"id": "run"}, "overview": {}, "current_task": {}, "catalog": system_catalog(), "parallel_execution": {}},
    )
    try:
        completed = server._process_pending_learning_once()
    finally:
        model_server.shutdown()
        model_server.server_close()

    learned = next(item for item in queue.list() if item["id"] == pending["id"])
    records = AgentExperienceMemory(knowledge_path).load()
    updated_chat = chats.get(chat["id"])
    assert completed == 1, learned.get("last_error")
    assert learned["status"] == "learned"
    assert learned["record_id"]
    assert any(record.get("question") == "请解释ZetaProbe协议的可靠性证明" for record in records)
    assert any(message.get("response_mode") == "autonomous_learning_completed" for message in updated_chat["messages"])


def test_conversation_concept_requires_real_validation_before_mastery(tmp_path, monkeypatch):
    knowledge_path = tmp_path / "knowledge.json"
    memory = AgentExperienceMemory(knowledge_path)
    record = memory.append_conversation_concept(
        "怎样提高Java的PIT变异检测率？",
        "针对存活变异体补充分支输入，并使用可区分运算符变化的具体断言。",
        model="test-model",
        conversation_id="chat-1",
    )

    assert record is not None
    assert record["learning_stage"] == "learning"
    memory.validate_conversation_concepts([record["experience_id"]], successful=True, evidence={"grade": "B"})
    assert memory.load()[0]["learning_stage"] == "learning"
    memory.validate_conversation_concepts([record["experience_id"]], successful=True, evidence={"grade": "A"})
    assert memory.load()[0]["learning_stage"] == "mastered"

    monkeypatch.setenv("TSG_KNOWLEDGE_STORE", str(knowledge_path))
    snapshot = knowledge_snapshot()
    assert snapshot["counts"]["已掌握"] == 1
    assert snapshot["records"][0]["validation_count"] == 2


def test_dashboard_surfaces_final_d_grade_reason_and_solutions(tmp_path):
    case = tmp_path / "Broken"
    case.mkdir()
    failure_report = case / "task_failure_report.html"
    failure_report.write_text("<html>failure</html>", encoding="utf-8")
    summary_path = case / "stateflow_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": "Broken.java",
                "quality_status": "execution_invalid",
                "final_grade": "D",
                "final_report": {"pytest_passed": False, "failure_category": "compile_error", "failed_tests": ["BrokenTest.compiles"]},
                "experiment_outcome": {
                    "failure_category": "compile_error",
                    "failure_reason": "compile_error: cannot find symbol add(int,int)",
                    "previous_solutions": ["核对目标方法签名"],
                    "grade_history": [{"round": 1, "grade": "D", "status": "execution_invalid", "failure_reason": "cannot find symbol"}],
                    "error": "javac: cannot find symbol",
                },
                "visualization": {"failure_report_html": str(failure_report)},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_summary.json").write_text(
        json.dumps({"language": "java", "results": [{"output_dir": str(case), "summary_path": str(summary_path), "final_grade": "D", "experiment_round": 3}]}),
        encoding="utf-8",
    )
    record = {"id": "d-run", "output_dir": str(tmp_path), "label": "D run", "registered_at": "now"}

    task = build_snapshot(record, [record])["current_task"]

    assert task["grade"] == "D"
    assert task["failure"]["reason"] == "compile_error: cannot find symbol add(int,int)"
    assert task["failure"]["previous_solutions"] == ["核对目标方法签名"]
    assert task["failure"]["suggested_solutions"]
    assert task["failure_report"] == str(failure_report)


def test_dashboard_surfaces_first_round_e_diagnostic(tmp_path):
    case = tmp_path / "Task098"
    case.mkdir()
    summary_path = case / "stateflow_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "source": "TestEvatask_098.py",
                "quality_status": "generation_failed",
                "final_grade": "E",
                "experiment_round": 1,
                "stop_reason": "generation_failed",
                "round_failure_diagnostic": {
                    "failure_category": "candidate_rejected",
                    "failure_reason": "candidate_rejected: candidate has no normal behavior assertion",
                    "failure_stage": "semantic_usability",
                    "error": "candidate has no normal behavior assertion",
                    "attempted_measures": ["测试生成智能体已执行候选验证"],
                    "suggested_solutions": ["补充正常行为断言"],
                    "knowledge_record_id": "failure-098",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "batch_summary.json").write_text(
        json.dumps({"language": "python", "results": [{"summary_path": str(summary_path), "output_dir": str(case), "status": "generation_failed", "final_grade": "E", "experiment_round": 1}]}),
        encoding="utf-8",
    )
    record = {"id": "e-r1", "output_dir": str(tmp_path), "label": "E R1", "registered_at": "now"}

    task = build_snapshot(record, [record])["current_task"]

    assert task["round_label"] == "E·R1"
    assert task["failure"]["category"] == "candidate_rejected"
    assert task["failure"]["stage"] == "semantic_usability"
    assert task["failure"]["attempted_measures"] == ["测试生成智能体已执行候选验证"]
    assert task["failure"]["knowledge_record_id"] == "failure-098"


def test_dashboard_recovers_legacy_e_reason_from_shared_tsg(tmp_path):
    case = tmp_path / "Task150"
    case.mkdir()
    graph_path = case / "state_flow_graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "test_generation_candidate_rejected": {
                        "accepted": False,
                        "stage": "collect_only",
                        "failure_category": "import_error",
                        "pytest_output": "ImportError: cannot import target function",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary_path = case / "stateflow_summary.json"
    summary_path.write_text(
        json.dumps({"source": "TestEvatask_150.py", "graph_path": str(graph_path), "quality_status": "generation_failed", "final_grade": "E", "experiment_round": 1}),
        encoding="utf-8",
    )
    (tmp_path / "batch_summary.json").write_text(
        json.dumps({"results": [{"summary_path": str(summary_path), "output_dir": str(case), "status": "generation_failed", "final_grade": "E", "experiment_round": 1}]}),
        encoding="utf-8",
    )
    record = {"id": "legacy-e", "output_dir": str(tmp_path), "label": "legacy E", "registered_at": "now"}

    task = build_snapshot(record, [record])["current_task"]

    assert task["failure"]["category"] == "import_error"
    assert task["failure"]["stage"] == "collect_only"
    assert "cannot import target function" in task["failure"]["reason"]


def test_running_graph_is_visible_before_final_summary_and_refreshes(tmp_path):
    case = tmp_path / "live-task"
    case.mkdir()
    graph_path = case / "state_flow_graph.json"

    def write_graph(coverage: float, version: int) -> None:
        graph_path.write_text(
            json.dumps(
                {
                    "source_path": "Live.java",
                    "version": version,
                    "iteration": version,
                    "nodes": {"method": {"node_type": "Method", "name": "run", "visited": coverage > 0.5}},
                    "edges": [],
                    "flow_history": [{"event_type": "ValidationCompleted", "source_agent": "测试生成智能体", "target_agent": "测试状态智能体", "iteration": version}],
                    "metadata": {"language": "java", "last_report": {"pytest_passed": True, "collected_count": 1, "line_coverage": coverage}},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    record = {"id": "live", "output_dir": str(tmp_path), "label": "live", "registered_at": "now"}
    write_graph(0.4, 1)
    first = build_snapshot(record, [record])
    write_graph(0.9, 2)
    second = build_snapshot(record, [record])

    assert first["overview"]["status"] == "running"
    assert first["current_task"]["round_label"] == "运行中"
    assert first["current_task"]["metrics"]["line_coverage"] == 0.4
    assert second["current_task"]["metrics"]["line_coverage"] == 0.9
    assert second["state_model"]["version"] == 2


def test_registry_keeps_llm_secret_out_of_persistent_file(tmp_path):
    path = tmp_path / "registry.json"
    registry = ExperimentRegistry(path)
    item = registry.register(
        str(tmp_path / "run"),
        "run",
        {"enabled": True, "provider": "ollama", "model": "demo-model", "base_url": "http://127.0.0.1:11434", "api_key": "secret"},
    )

    assert registry.runtime_config(item["id"])["api_key"] == "secret"
    assert '"api_key"' not in path.read_text(encoding="utf-8")
    restarted = ExperimentRegistry(path)
    restored = restarted.runtime_config(item["id"])
    assert restored["enabled"] is True
    assert restored["provider"] == "ollama"
    assert restored["model"] == "demo-model"
    assert "api_key" not in restored


def test_all_view_uses_active_experiment_llm_config_after_restart(tmp_path, monkeypatch):
    import monitoring.server as server

    registry_path = tmp_path / "registry.json"
    registry = ExperimentRegistry(registry_path)
    registry.register(str(tmp_path / "run"), "run", {"enabled": True, "provider": "ollama", "model": "demo-model"})
    restarted = ExperimentRegistry(registry_path)
    monkeypatch.setattr(server, "REGISTRY", restarted)
    handler = object.__new__(server.DashboardHandler)

    config = handler._llm_config("all")

    assert config.enabled is True
    assert config.model == "demo-model"


def test_legacy_experiment_recovers_llm_config_from_summary(tmp_path, monkeypatch):
    import monitoring.server as server

    output = tmp_path / "run"
    output.mkdir()
    (output / "all_datasets_summary.json").write_text(
        json.dumps({"llm": {"enabled": True, "provider": "ollama", "model": "legacy-model", "base_url": "http://127.0.0.1:11434", "timeout": 45}}),
        encoding="utf-8",
    )
    registry = ExperimentRegistry(tmp_path / "registry.json")
    registry.register(str(output), "legacy")
    monkeypatch.setattr(server, "REGISTRY", registry)
    handler = object.__new__(server.DashboardHandler)

    config = handler._llm_config("all")

    assert config.enabled is True
    assert config.model == "legacy-model"
    assert config.timeout == 45


def test_dashboard_experiment_timing_uses_process_lifecycle(tmp_path):
    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=5)
    record = {
        "id": "timed-run",
        "output_dir": str(tmp_path),
        "label": "timed",
        "registered_at": (now - timedelta(minutes=2)).isoformat(),
        "execution_status": "running",
        "started_at": started.isoformat(),
    }
    (tmp_path / "batch_summary.json").write_text(
        json.dumps(
            {
                "started_at": (now - timedelta(minutes=1)).isoformat(),
                "finished_at": (now - timedelta(seconds=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    running = build_snapshot(record, [record])["overview"]

    assert running["started_at"] == started.isoformat()
    assert running["finished_at"] is None
    assert 4 <= running["duration_seconds"] <= 10

    finished = now + timedelta(seconds=3)
    record.update({"execution_status": "completed", "finished_at": finished.isoformat()})
    completed = build_snapshot(record, [record])["overview"]
    assert completed["finished_at"] == finished.isoformat()
    assert completed["duration_seconds"] == 8.0


def test_dashboard_recovers_full_duration_when_resume_summary_only_contains_last_session(tmp_path):
    task = tmp_path / "task_001"
    task.mkdir()
    (tmp_path / "batch_summary.json").write_text(
        json.dumps({"language": "java", "duration_seconds": 3.0, "outcome_summary": {"total_tasks": 1}}),
        encoding="utf-8",
    )
    (task / "stateflow_summary.json").write_text(
        json.dumps({"source": str(tmp_path / "Example.java"), "runtime_seconds": 125.0, "final_grade": "D"}),
        encoding="utf-8",
    )
    record = {"id": "resumed", "output_dir": str(tmp_path), "label": "resumed", "execution_status": "completed"}

    overview = build_snapshot(record, [record])["overview"]

    assert overview["duration_seconds"] == 125.0
    assert overview["duration_source"] == "historical_task_runtime"


def test_registry_clears_old_timing_when_same_output_is_restarted(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.json")
    output = tmp_path / "run"
    previous = registry.register(
        str(output),
        "old",
        metadata={"execution_status": "completed", "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:01:00+00:00"},
    )

    restarted = registry.register(str(output), "new", metadata={"execution_status": "queued"})

    assert restarted["id"] == previous["id"]
    assert restarted["started_at"] is None
    assert restarted["finished_at"] is None


def test_experiment_queue_finishes_one_directory_before_starting_next(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.json")
    processes = {}
    queue = ExperimentJobQueue(registry, processes)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = registry.register(str(first_dir), "first")
    second = registry.register(str(second_dir), "second")
    trace = tmp_path / "trace.txt"
    writer = (
        "import pathlib,time,sys; p=pathlib.Path(sys.argv[1]); "
        "p.open('a',encoding='utf-8').write(sys.argv[2]+'-start\\n'); "
        "time.sleep(float(sys.argv[3])); "
        "p.open('a',encoding='utf-8').write(sys.argv[2]+'-end\\n')"
    )
    queue.enqueue(ExperimentJob(first["id"], [sys.executable, "-c", writer, str(trace), "first", "0.25"], os.environ.copy(), first_dir / "process.log"))
    queue.enqueue(ExperimentJob(second["id"], [sys.executable, "-c", writer, str(trace), "second", "0"], os.environ.copy(), second_dir / "process.log"))

    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if registry.get(first["id"])["execution_status"] == "completed" and registry.get(second["id"])["execution_status"] == "completed":
            break
        time.sleep(0.05)

    assert registry.get(first["id"])["execution_status"] == "completed"
    assert registry.get(second["id"])["execution_status"] == "completed"
    assert trace.read_text(encoding="utf-8").splitlines() == ["first-start", "first-end", "second-start", "second-end"]
