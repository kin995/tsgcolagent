from __future__ import annotations

import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llm_client
import monitoring.server as server
from llm_config import LLMConfig
from monitoring.chat_store import ChatStore
from monitoring.learning_queue import PendingKnowledgeLearningQueue
from monitoring.registry import ExperimentRegistry


def test_http_offline_queue_recovery_and_offline_knowledge_reuse(tmp_path, monkeypatch):
    state = {"online": False, "calls": 0}

    class Model(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            state["calls"] += 1
            body = (
                {"choices": [{"message": {"content": "ZetaProbe的可靠性需要用固定输入、故障注入和独立断言验证。"}}]}
                if state["online"] else {"error": "temporarily unavailable"}
            )
            encoded = json.dumps(body).encode()
            self.send_response(200 if state["online"] else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    model = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    dashboard = ThreadingHTTPServer(("127.0.0.1", 0), server.DashboardHandler)
    threads = [threading.Thread(target=app.serve_forever, daemon=True) for app in (model, dashboard)]
    queue = PendingKnowledgeLearningQueue(tmp_path / "queue.json")
    chats = ChatStore(tmp_path / "chats.json")
    registry = ExperimentRegistry(tmp_path / "registry.json")
    knowledge_path = tmp_path / "knowledge.json"
    monkeypatch.setattr(server, "LEARNING_QUEUE", queue)
    monkeypatch.setattr(server, "CHATS", chats)
    monkeypatch.setattr(server, "REGISTRY", registry)
    monkeypatch.setattr(server.DashboardHandler, "_snapshot", lambda *_args: {"experiment": {"id": "run"}, "tasks": []})
    monkeypatch.setattr(server.DashboardHandler, "_knowledge_store", lambda *_args: knowledge_path)
    monkeypatch.setattr(server.DashboardHandler, "_llm_config", lambda *_args: LLMConfig(
        enabled=True, provider="openai", base_url=f"http://127.0.0.1:{model.server_port}", model="audit-model", timeout=2,
    ))
    for thread in threads:
        thread.start()

    def request(path, data=None):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{dashboard.server_port}{path}", data=raw,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    try:
        chat = request("/api/chats", {"title": "isolated audit"})
        url = f"/api/chats/{chat['id']}/messages"
        question = "请解释Java ZetaProbe协议的原理"
        offline = request(url, {"question": question})
        assert offline["answer"]["response_mode"] == "pending_learning"
        assert PendingKnowledgeLearningQueue(tmp_path / "queue.json").list()[0]["status"] == "pending"
        state["online"] = True
        # Simulate expiry of the connection-failure cooldown, not a real user endpoint.
        llm_client._ENDPOINT_FAILURES.clear()
        assert server._process_pending_learning_once() == 1
        restored = request(f"/api/chats/{chat['id']}")
        assert restored["messages"][-1]["response_mode"] == "autonomous_learning_completed"
        assert restored["messages"][-1]["learning"]["record_id"]
        state["online"] = False
        reused = request(url, {"question": question})
        assert reused["answer"]["response_mode"] == "knowledge_autonomous"
        assert "故障注入" in reused["answer"]["answer"]
        assert not reused["answer"]["needs_learning"]
        assert len(queue.list()) == 1
    finally:
        for app in (dashboard, model):
            app.shutdown()
            app.server_close()
        llm_client._ENDPOINT_FAILURES.clear()
