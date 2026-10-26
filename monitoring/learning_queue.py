from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PendingKnowledgeLearningQueue:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self._data = self._load()

    def enqueue(
        self,
        question: str,
        *,
        chat_id: str | None,
        experiment_id: str | None,
        knowledge_store: str | Path,
        context_label: str | None,
    ) -> dict[str, Any]:
        normalized = " ".join(question.casefold().split())
        store = str(Path(knowledge_store).expanduser().resolve())
        item_id = "learn_" + hashlib.sha256(f"{store}|{experiment_id}|{chat_id}|{normalized}".encode("utf-8")).hexdigest()[:20]
        with self.lock:
            existing = next((item for item in self._data.get("items", []) if item.get("id") == item_id), None)
            if existing and existing.get("status") == "pending":
                return dict(existing)
            if existing:
                self._data["items"].remove(existing)
            item = {
                "id": item_id,
                "question": question.strip()[:2000],
                "chat_id": chat_id,
                "experiment_id": experiment_id,
                "knowledge_store": str(Path(knowledge_store).expanduser().resolve()),
                "context_label": context_label,
                "status": "pending",
                "attempts": 0,
                "created_at": _now(),
                "updated_at": _now(),
                "next_retry_at": time.time(),
                "last_error": None,
            }
            self._data.setdefault("items", []).append(item)
            pending = [entry for entry in self._data["items"] if entry.get("status") == "pending"]
            finished = [entry for entry in self._data["items"] if entry.get("status") != "pending"]
            self._data["items"] = [*finished[-500:], *pending]
            self._save()
            return dict(item)

    def due(self, limit: int = 5) -> list[dict[str, Any]]:
        now = time.time()
        with self.lock:
            return [
                dict(item)
                for item in self._data.get("items", [])
                if item.get("status") == "pending" and float(item.get("next_retry_at", 0) or 0) <= now
            ][:limit]

    def retry_later(self, item_id: str, error: str) -> dict[str, Any] | None:
        with self.lock:
            item = self._find(item_id)
            if item is None:
                return None
            attempts = int(item.get("attempts", 0) or 0) + 1
            delay = min(1800, 60 * (2 ** min(5, attempts - 1)))
            item.update({
                "attempts": attempts,
                "last_error": error[:2000],
                "next_retry_at": time.time() + delay,
                "updated_at": _now(),
            })
            self._save()
            return dict(item)

    def complete(self, item_id: str, *, answer: str, record_id: str | None) -> dict[str, Any] | None:
        if not record_id:
            raise ValueError("Learning cannot complete without a persisted knowledge record")
        with self.lock:
            item = self._find(item_id)
            if item is None:
                return None
            item.update({
                "status": "learned",
                "answer": answer[:6000],
                "record_id": record_id,
                "learned_at": _now(),
                "updated_at": _now(),
                "last_error": None,
            })
            self._save()
            return dict(item)

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(item) for item in self._data.get("items", [])]

    def _find(self, item_id: str) -> dict[str, Any] | None:
        return next((item for item in self._data.get("items", []) if item.get("id") == item_id), None)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"items": []}
        except (OSError, json.JSONDecodeError):
            return {"items": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            temporary.replace(self.path)
        except OSError:
            self.path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
            temporary.unlink(missing_ok=True)
