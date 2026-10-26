from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dashboard_data import experiment_id


class ExperimentRegistry:
    PERSISTED_LLM_FIELDS = {"enabled", "required", "provider", "base_url", "model", "timeout", "temperature"}
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self._data = self._load()
        self._runtime_configs: dict[str, dict[str, Any]] = {}

    def register(
        self,
        output_dir: str,
        label: str | None = None,
        llm_config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = str(Path(output_dir).resolve())
        item_id = experiment_id(resolved)
        with self.lock:
            previous = next((dict(entry) for entry in self._data.get("experiments", []) if entry.get("id") == item_id), {})
        item = {**previous,
            "id": item_id,
            "output_dir": resolved,
            "label": label or Path(resolved).name or "experiment",
            "registered_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {}),
        }
        if llm_config:
            item["llm_config"] = {
                key: value
                for key, value in llm_config.items()
                if key in self.PERSISTED_LLM_FIELDS
            }
        lifecycle = str(item.get("execution_status") or "")
        if lifecycle == "queued":
            item.update({"started_at": None, "finished_at": None, "return_code": None, "last_error": None})
        elif lifecycle == "running":
            item.update({"finished_at": None, "return_code": None, "last_error": None})
        with self.lock:
            if llm_config:
                self._runtime_configs[item["id"]] = dict(llm_config)
            experiments = [entry for entry in self._data.get("experiments", []) if entry.get("id") != item["id"]]
            experiments.insert(0, item)
            self._data = {"active_id": item["id"], "experiments": experiments}
            self._save()
        return item

    def update(self, item_id: str, **changes: Any) -> dict[str, Any] | None:
        allowed = {
            "execution_status", "process_id", "queued_at", "started_at", "finished_at",
            "return_code", "last_error", "launch_request", "label", "knowledge_store",
        }
        safe_changes = {key: value for key, value in changes.items() if key in allowed}
        with self.lock:
            experiments = self._data.get("experiments", [])
            item = next((entry for entry in experiments if entry.get("id") == item_id), None)
            if item is None:
                return None
            item.update(safe_changes)
            self._save()
            return dict(item)

    def runtime_config(self, item_id: str | None = None) -> dict[str, Any]:
        item = self.get(item_id)
        with self.lock:
            persisted = (item or {}).get("llm_config")
            persisted = dict(persisted) if isinstance(persisted, dict) else {}
            runtime = self._runtime_configs.get(str((item or {}).get("id", "")), {})
            return {**persisted, **dict(runtime)}

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(item) for item in self._data.get("experiments", [])]

    def get(self, item_id: str | None = None) -> dict[str, Any] | None:
        with self.lock:
            wanted = item_id or self._data.get("active_id")
            return next((dict(item) for item in self._data.get("experiments", []) if item.get("id") == wanted), None)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"experiments": []}
        except (OSError, json.JSONDecodeError):
            return {"experiments": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        content = json.dumps(self._data, indent=2, ensure_ascii=False)
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(self.path)
        except OSError:
            # os.replace can fail on some Windows installations when the user
            # profile contains non-ASCII characters.  Direct writing is safer
            # than losing the registry entirely in that environment.
            self.path.write_text(content, encoding="utf-8")
            try:
                temporary.unlink()
            except OSError:
                pass
