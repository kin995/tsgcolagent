"""tsgtest 的接口测试：只用标准库，不依赖 langgraph / Ollama。"""
from __future__ import annotations

import json
import shutil
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from tsgtest.config import Settings
from tsgtest.jobs import JobManager
from tsgtest.server import make_server


def fake_runner(source: Path, out_dir: Path, settings, log, cancelled) -> dict[str, object]:
    log("fake run")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test_generated.py"
    test_path.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    return {
        "kind": "single",
        "items": [{
            "source": str(source),
            "output_dir": str(out_dir),
            "test_path": str(test_path),
            "test_text": test_path.read_text(encoding="utf-8"),
            "status": "quality_high",
            "iterations": 1,
            "pytest_passed": True,
            "metrics": {"line_coverage": 0.9, "branch_coverage": 0.8, "ae": 0.85, "mutation_score": 0.7,
                        "pass_rate": 1.0, "tir": 0.5, "avg_exec": 3.0, "duration_seconds": 12.5},
        }],
    }


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        # 不用 tempfile.TemporaryDirectory：它按 0700 建目录，在本机沙箱下会拒写。
        base = Path(__file__).resolve().parents[1] / ".tsgtest_test"
        base.mkdir(exist_ok=True)
        self.root = base / uuid.uuid4().hex[:8]
        self.root.mkdir()
        self.source = self.root / "sample.py"
        self.source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        self.settings = Settings(runs_dir=self.root / "runs")
        self.manager = JobManager(self.settings, runner=fake_runner)
        self.server = make_server(self.settings, self.manager, port=0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.root, ignore_errors=True)

    def get(self, path: str) -> tuple[int, bytes, str]:
        with urllib.request.urlopen(self.base + path, timeout=10) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_health(self) -> None:
        status, body, _ = self.get("/api/health")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertTrue(data["ok"])
        self.assertIn("dependencies", data)

    def test_index_and_static(self) -> None:
        status, body, content_type = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("TSGColAgent", body.decode("utf-8"))
        self.assertIn("text/html", content_type)
        status, body, content_type = self.get("/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", content_type)

    def test_unknown_api(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/nope")
        self.assertEqual(ctx.exception.code, 404)

    def test_browse(self) -> None:
        status, body, _ = self.get("/api/browse?path=" + urllib.parse.quote(str(self.source.parent)))
        self.assertEqual(status, 200)
        names = [item["name"] for item in json.loads(body)["files"]]
        self.assertIn("sample.py", names)

    def test_reject_missing_source(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/jobs", {"source": str(self.source.parent / "nope.py")})
        self.assertEqual(ctx.exception.code, 400)

    def test_reject_wrong_suffix(self) -> None:
        other = self.source.parent / "notes.txt"
        other.write_text("x", encoding="utf-8")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("/api/jobs", {"source": str(other)})
        self.assertEqual(ctx.exception.code, 400)

    def test_job_lifecycle_and_test_download(self) -> None:
        status, data = self.post("/api/jobs", {"source": str(self.source), "max_rounds": 2})
        self.assertEqual(status, 201)
        job_id = data["job"]["id"]
        deadline = time.time() + 10
        job = None
        while time.time() < deadline:
            _, body, _ = self.get(f"/api/jobs/{job_id}")
            job = json.loads(body)
            if job["status"] in {"done", "failed", "cancelled"}:
                break
            time.sleep(0.1)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["items"][0]["status"], "quality_high")
        _, body, content_type = self.get(f"/api/jobs/{job_id}/test")
        payload = json.loads(body)
        self.assertIn("application/json", content_type)
        self.assertIn("def test_ok", payload["files"][0]["text"])

    def test_cancel_unknown(self) -> None:
        status, data = self.post("/api/jobs/aaaaaaaaaaaa/cancel", {})
        self.assertEqual(status, 200)
        self.assertFalse(data["cancelled"])


if __name__ == "__main__":
    unittest.main()
