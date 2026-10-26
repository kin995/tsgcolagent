"""HTTP 服务：JSON 接口 + 单页前端。仅依赖标准库。"""
from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import __version__
from .config import Settings
from .engine import SOURCE_SUFFIXES, dependency_status, ollama_models
from .jobs import JobManager

WEB_DIR = Path(__file__).resolve().parent / "web"
JOB_PATH = re.compile(r"^/api/jobs/([0-9a-f]{6,32})(?:/(test|cancel|summary))?$")


class Handler(BaseHTTPRequestHandler):
    manager: JobManager
    settings: Settings
    server_version = "tsgtest/" + __version__

    # ---------- 基础输出 ----------
    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _text(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - 保持安静
        pass

    # ---------- 路由 ----------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        if route == "/api/health":
            return self._health()
        if route == "/api/models":
            return self._models(query)
        if route == "/api/browse":
            return self._browse(query)
        if route == "/api/jobs":
            return self._json({"jobs": self.manager.list()})
        match = JOB_PATH.match(route)
        if match:
            return self._job_get(match.group(1), match.group(2))
        if route in ("/", "/index.html"):
            return self._static("index.html")
        if route.startswith("/api/"):
            return self._error("未知接口", HTTPStatus.NOT_FOUND)
        return self._static(route.lstrip("/"))

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/jobs":
            return self._create_job()
        match = JOB_PATH.match(route)
        if match and match.group(2) == "cancel":
            ok = self.manager.cancel(match.group(1))
            return self._json({"cancelled": ok})
        return self._error("未知接口", HTTPStatus.NOT_FOUND)

    # ---------- 接口实现 ----------
    def _health(self) -> None:
        self._json({
            "ok": True,
            "version": __version__,
            "dependencies": dependency_status(),
            "ollama_base_url": self.settings.ollama_base_url,
            "runs_dir": str(self.settings.runs_dir),
        })

    def _models(self, query: dict[str, list[str]]) -> None:
        base_url = (query.get("base_url") or [self.settings.ollama_base_url])[0]
        models, error = ollama_models(base_url)
        self._json({"models": models, "error": error, "base_url": base_url})

    def _browse(self, query: dict[str, list[str]]) -> None:
        raw = (query.get("path") or [""])[0]
        base = Path(raw).expanduser() if raw else Path.home()
        if not base.is_absolute():
            base = (Path.cwd() / base)
        if not base.is_dir():
            return self._error(f"目录不存在：{base}", HTTPStatus.NOT_FOUND)
        try:
            entries = list(base.iterdir())
        except OSError as exc:
            return self._error(f"无法读取目录：{exc}", HTTPStatus.FORBIDDEN)
        dirs = sorted((p for p in entries if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name.lower())
        files = sorted(
            (p for p in entries if p.is_file() and p.suffix.lower() in SOURCE_SUFFIXES),
            key=lambda p: p.name.lower(),
        )
        self._json({
            "path": str(base.resolve()),
            "parent": str(base.resolve().parent) if base.resolve().parent != base.resolve() else None,
            "dirs": [{"name": p.name, "path": str(p.resolve())} for p in dirs[:200]],
            "files": [{"name": p.name, "path": str(p.resolve()), "language": p.suffix.lstrip(".")} for p in files[:400]],
        })

    def _create_job(self) -> None:
        try:
            payload = self._read_json()
        except ValueError as exc:
            return self._error(str(exc), HTTPStatus.BAD_REQUEST)
        source = str(payload.get("source") or "").strip()
        if not source:
            return self._error("缺少 source（源文件或目录）", HTTPStatus.BAD_REQUEST)
        overrides: dict[str, object] = {}
        for key in ("model", "ollama_base_url"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                overrides[key] = value.strip()
        if payload.get("temperature") is not None:
            try:
                overrides["temperature"] = float(payload["temperature"])
            except (TypeError, ValueError):
                return self._error("temperature 必须是数字", HTTPStatus.BAD_REQUEST)
        if payload.get("max_rounds") is not None:
            try:
                overrides["max_rounds"] = max(1, min(5, int(payload["max_rounds"])))
            except (TypeError, ValueError):
                return self._error("max_rounds 必须是整数", HTTPStatus.BAD_REQUEST)
        try:
            job = self.manager.submit(source, **overrides)
        except ValueError as exc:
            return self._error(str(exc), HTTPStatus.BAD_REQUEST)
        self._json({"job": job.public()}, HTTPStatus.CREATED)

    def _job_get(self, job_id: str, action: str | None) -> None:
        job = self.manager.get(job_id)
        if job is None:
            return self._error("任务不存在", HTTPStatus.NOT_FOUND)
        if action is None:
            return self._json(job)
        if action == "summary":
            return self._json({"id": job_id, "result": job["result"], "status": job["status"]})
        if action == "test":
            result = job.get("result") or {}
            items = result.get("items") if isinstance(result, dict) else None
            if not items:
                return self._error("该任务还没有生成测试", HTTPStatus.NOT_FOUND)
            out_dir = Path(str(job["out_dir"])).resolve()
            collected = []
            for item in items:
                path = item.get("test_path")
                if not path:
                    continue
                candidate = Path(str(path)).resolve()
                if out_dir not in candidate.parents and candidate != out_dir:
                    continue
                text = item.get("test_text")
                if text is None and candidate.is_file():
                    text = candidate.read_text(encoding="utf-8", errors="replace")
                collected.append({"source": item.get("source"), "test_path": str(candidate), "text": text})
            if not collected:
                return self._error("测试文件不在任务目录内", HTTPStatus.FORBIDDEN)
            return self._json({"files": collected})
        return self._error("未知接口", HTTPStatus.NOT_FOUND)

    # ---------- 工具 ----------
    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1_048_576:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def _static(self, relative: str) -> None:
        target = (WEB_DIR / relative).resolve()
        if WEB_DIR.resolve() not in target.parents and target != WEB_DIR.resolve():
            return self._error("非法路径", HTTPStatus.FORBIDDEN)
        if not target.is_file():
            return self._error("文件不存在", HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self._text(target.read_bytes(), content_type)


def make_server(settings: Settings, manager: JobManager | None = None, host: str | None = None, port: int | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {
        "manager": manager or JobManager(settings),
        "settings": settings,
    })
    server = ThreadingHTTPServer((host or settings.host, settings.port if port is None else port), handler)
    return server


def serve(settings: Settings, open_browser: bool = False) -> None:
    server = make_server(settings)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}"
    print(f"tsgtest 已启动：{url}")
    print(f"运行目录：{settings.runs_dir}")
    deps = dependency_status()
    missing = [name for name, ok in deps.items() if not ok]
    if missing:
        print(f"[提示] 缺少依赖或工具：{', '.join(missing)}（界面可打开，任务会失败）")
    if open_browser:
        import webbrowser

        webbrowser.open(url, new=2)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        server.server_close()
