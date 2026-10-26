from __future__ import annotations

import json
import os
import re
import threading
import urllib.error
import urllib.request
import time
from typing import Any

from llm_config import LLMConfig


try:
    _OLLAMA_CONCURRENCY = max(1, int(os.getenv("TSG_LLM_CONCURRENCY", "2") or 2))
except ValueError:
    _OLLAMA_CONCURRENCY = 2
_OLLAMA_SEMAPHORE = threading.BoundedSemaphore(_OLLAMA_CONCURRENCY)
_ENDPOINT_FAILURES: dict[str, tuple[float, str]] = {}
_ENDPOINT_FAILURE_LOCK = threading.Lock()
_CIRCUIT_BREAK_SECONDS = 60.0


class OpenAICompatibleLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_seconds": 0.0}

    def enabled(self) -> bool:
        return self.config.enabled

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.config.enabled:
            raise RuntimeError("LLM is disabled.")
        endpoint_key = self._endpoint_key()
        with _ENDPOINT_FAILURE_LOCK:
            blocked = _ENDPOINT_FAILURES.get(endpoint_key)
        if blocked and blocked[0] > time.monotonic():
            raise RuntimeError(f"LLM endpoint circuit is open: {blocked[1]}")
        started = time.perf_counter()
        self.usage["calls"] += 1
        try:
            if self.config.provider.lower() == "ollama":
                with _OLLAMA_SEMAPHORE:
                    result = self._chat_ollama(messages)
            else:
                result = self._chat_openai_compatible(messages)
            with _ENDPOINT_FAILURE_LOCK:
                _ENDPOINT_FAILURES.pop(endpoint_key, None)
            return result
        except Exception as exc:
            if _is_endpoint_failure(exc):
                with _ENDPOINT_FAILURE_LOCK:
                    _ENDPOINT_FAILURES[endpoint_key] = (time.monotonic() + _CIRCUIT_BREAK_SECONDS, str(exc))
            raise
        finally:
            self.usage["duration_seconds"] = round(float(self.usage["duration_seconds"]) + (time.perf_counter() - started), 4)

    def health_check(self, timeout: int | float = 3) -> tuple[bool, str]:
        """Verify the configured service and model before a long experiment or chat call."""
        if not self.config.enabled:
            return False, "LLM is disabled"
        provider = self.config.provider.lower()
        url = self.config.base_url.rstrip("/") + ("/api/tags" if provider == "ollama" else "/models")
        headers = {}
        if provider != "ollama":
            headers["Authorization"] = f"Bearer {self.config.api_key or 'local'}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=max(1, float(timeout))) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return False, f"{exc.__class__.__name__}: {exc}"
        if provider == "ollama":
            models = data.get("models", []) if isinstance(data, dict) else []
            names = {
                str(item.get("name") or item.get("model") or "")
                for item in models
                if isinstance(item, dict)
            }
            if self.config.model not in names:
                return False, f"Ollama model is not installed: {self.config.model}"
        return True, "ok"

    def _endpoint_key(self) -> str:
        return f"{self.config.provider.lower()}|{self.config.base_url.rstrip('/')}|{self.config.model}"

    def usage_snapshot(self) -> dict[str, int | float]:
        return dict(self.usage)

    def _chat_ollama(self, messages: list[dict[str, str]]) -> str:
        try:
            from ollama import Client
        except ImportError as exc:
            raise RuntimeError("Ollama Python package is not installed. Install with: pip install ollama") from exc

        try:
            client = Client(
                host=self.config.base_url.rstrip("/"),
                timeout=max(1, int(self.config.timeout)),
            )
        except TypeError:
            # Older Ollama SDK releases accept only ``host``.  The request still
            # remains protected by the server-side/model timeout configuration.
            client = Client(host=self.config.base_url.rstrip("/"))
        try:
            response = client.chat(model=self.config.model, messages=messages)
        except Exception as exc:
            raise RuntimeError(f"Ollama connection failed: {exc}") from exc
        content = _extract_ollama_content(response)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama response has empty content: {response}")
        self._record_usage(response, messages, content)
        return content

    def _chat_openai_compatible(self, messages: list[dict[str, str]]) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key or 'local'}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM connection failed: {exc}") from exc
        content = _extract_message(data)
        self._record_usage(data, messages, content)
        return content

    def _record_usage(self, response: Any, messages: list[dict[str, str]], content: str) -> None:
        data = response if isinstance(response, dict) else None
        if data is None:
            try:
                data = response.model_dump()
            except Exception:
                data = {}
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(data, dict):
            prompt = prompt if prompt is not None else data.get("prompt_eval_count")
            completion = completion if completion is not None else data.get("eval_count")
        # Some local servers omit usage.  Keep an explicit estimate rather than
        # silently reporting zero tokens.
        prompt_tokens = int(prompt) if prompt is not None else max(1, sum(len(item.get("content", "")) for item in messages) // 4)
        completion_tokens = int(completion) if completion is not None else max(1, len(content) // 4)
        self.usage["prompt_tokens"] += prompt_tokens
        self.usage["completion_tokens"] += completion_tokens
        self.usage["total_tokens"] += prompt_tokens + completion_tokens


def extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip() + "\n"
    return text.strip() + "\n"


def extract_json_data(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    return json.loads(candidate)


def _extract_message(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"LLM response has no choices: {data}")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM response has empty content: {data}")
    return content


def _extract_ollama_content(response: Any) -> str | None:
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return message.get("content")
        return getattr(message, "content", None)
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content")
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    try:
        data = response.model_dump()
    except Exception:
        data = None
    if isinstance(data, dict):
        message = data.get("message", {})
        if isinstance(message, dict):
            return message.get("content")
    return None


def _is_endpoint_failure(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in ("connection", "timed out", "timeout", "refused", "unreachable", "reset by peer", "http 5")
    )
