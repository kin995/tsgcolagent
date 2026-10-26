from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        self._data = self._load()

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            chats = [self._summary(item) for item in self._data.get("chats", []) if isinstance(item, dict)]
        return sorted(chats, key=lambda item: str(item.get("updated_at", "")), reverse=True)

    def create(self, title: str | None = None) -> dict[str, Any]:
        now = _now()
        chat = {
            "id": f"chat_{uuid4().hex[:12]}",
            "title": (title or "新对话").strip()[:80] or "新对话",
            "created_at": now,
            "updated_at": now,
            "bound_experiment": None,
            "bound_task": None,
            "messages": [],
        }
        with self.lock:
            self._data.setdefault("chats", []).append(chat)
            self._save()
        return dict(chat)

    def get(self, chat_id: str) -> dict[str, Any] | None:
        with self.lock:
            chat = next((item for item in self._data.get("chats", []) if item.get("id") == chat_id), None)
            return json.loads(json.dumps(chat, ensure_ascii=False)) if chat else None

    def append(self, chat_id: str, role: str, content: str, **extra: Any) -> dict[str, Any] | None:
        message = {"id": f"msg_{uuid4().hex[:12]}", "role": role, "content": content, "created_at": _now(), **extra}
        with self.lock:
            chat = next((item for item in self._data.get("chats", []) if item.get("id") == chat_id), None)
            if chat is None:
                return None
            chat.setdefault("messages", []).append(message)
            chat["updated_at"] = message["created_at"]
            if len(chat["messages"]) == 1 and chat.get("title") == "新对话":
                chat["title"] = _title_from(content)
            self._save()
        return message

    def bind(self, chat_id: str, experiment_id: str | None, task_id: str | None, context_label: str | None) -> None:
        with self.lock:
            chat = next((item for item in self._data.get("chats", []) if item.get("id") == chat_id), None)
            if chat is None:
                return
            chat["bound_experiment"] = experiment_id
            chat["bound_task"] = task_id
            chat["context_label"] = context_label
            chat["updated_at"] = _now()
            self._save()

    def rename(self, chat_id: str, title: str) -> dict[str, Any] | None:
        with self.lock:
            chat = next((item for item in self._data.get("chats", []) if item.get("id") == chat_id), None)
            if chat is None:
                return None
            chat["title"] = title.strip()[:80] or chat.get("title") or "新对话"
            chat["updated_at"] = _now()
            self._save()
            return self._summary(chat)

    def delete(self, chat_id: str) -> bool:
        with self.lock:
            before = len(self._data.get("chats", []))
            self._data["chats"] = [item for item in self._data.get("chats", []) if item.get("id") != chat_id]
            changed = len(self._data["chats"]) != before
            if changed:
                self._save()
            return changed

    def _summary(self, chat: dict[str, Any]) -> dict[str, Any]:
        messages = chat.get("messages") if isinstance(chat.get("messages"), list) else []
        return {key: chat.get(key) for key in ["id", "title", "created_at", "updated_at", "bound_experiment", "bound_task", "context_label"]} | {"message_count": len(messages), "last_message": messages[-1].get("content") if messages else None}

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {"chats": []}
        except (OSError, json.JSONDecodeError):
            return {"chats": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        content = json.dumps(self._data, indent=2, ensure_ascii=False)
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(self.path)
        except OSError:
            self.path.write_text(content, encoding="utf-8")
            try:
                temporary.unlink()
            except OSError:
                pass


def _title_from(content: str) -> str:
    cleaned = " ".join(content.strip().split())
    return (cleaned[:24] + ("…" if len(cleaned) > 24 else "")) or "新对话"
