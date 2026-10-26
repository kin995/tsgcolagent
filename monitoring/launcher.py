from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from urllib.parse import urlencode

from .dashboard_data import experiment_id


FALSE_VALUES = {"0", "false", "no", "off"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def launch_dashboard(
    output_dir: str | Path,
    *,
    label: str | None = None,
    enabled: bool = True,
    open_browser: bool = True,
    llm_config: object | None = None,
) -> str | None:
    if not enabled or os.environ.get("TSG_UI_ENABLED", "1").strip().lower() in FALSE_VALUES:
        return None
    host = os.environ.get("TSG_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("TSG_UI_PORT", "8765"))
    base_url = f"http://{host}:{port}"
    if not _healthy(base_url):
        _start_server(host, port)
        for _ in range(40):
            if _healthy(base_url):
                break
            time.sleep(0.125)
    if not _healthy(base_url):
        print("[dashboard] 前端监控服务未能启动，实验将继续运行。", file=sys.stderr)
        return None
    resolved = str(Path(output_dir).resolve())
    config_data = None
    if llm_config is not None:
        public = getattr(llm_config, "public_dict", None)
        config_data = dict(public()) if callable(public) else {}
        api_key = getattr(llm_config, "api_key", None)
        if api_key:
            config_data["api_key"] = api_key
    payload = json.dumps(
        {
            "output_dir": resolved,
            "label": label,
            "llm_config": config_data,
            "metadata": {
                "execution_status": "running",
                "started_at": _utc_now(),
                "knowledge_store": os.environ.get("TSG_KNOWLEDGE_STORE"),
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(base_url + "/api/register", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=2):
            pass
    except (OSError, urllib.error.URLError):
        return None
    # The experiment is registered above; the dashboard opens the portfolio so
    # newly queued, running, and completed directory experiments stay visible.
    url = base_url + "/?" + urlencode({"experiment": "all"})
    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    print(f"[dashboard] {url}")
    return url


def finish_dashboard(
    output_dir: str | Path,
    *,
    success: bool,
    error: str | None = None,
    enabled: bool = True,
) -> None:
    """Report terminal-launched experiment completion without affecting its result files."""
    if not enabled or os.environ.get("TSG_UI_ENABLED", "1").strip().lower() in FALSE_VALUES:
        return
    host = os.environ.get("TSG_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("TSG_UI_PORT", "8765"))
    base_url = f"http://{host}:{port}"
    if not _healthy(base_url):
        return
    payload = json.dumps(
        {"output_dir": str(Path(output_dir).resolve()), "success": success, "error": error},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/api/experiments/finish",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2):
            pass
    except (OSError, urllib.error.URLError):
        return


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/api/health", timeout=0.4) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _start_server(host: str, port: int) -> None:
    command = [sys.executable, "-B", "-m", "monitoring.server", "--host", host, "--port", str(port)]
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    try:
        subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=os.name != "nt",
        )
    except OSError:
        return
