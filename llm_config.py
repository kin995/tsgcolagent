from __future__ import annotations

import os
from dataclasses import dataclass


TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass
class LLMConfig:
    enabled: bool = False
    required: bool = False
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "deepseek-r1:14b"
    api_key: str = "local"
    timeout: int = 120
    temperature: float = 0.2

    @classmethod
    def from_env(
        cls,
        enabled: bool | None = None,
        required: bool | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
        provider: str | None = None,
    ) -> "LLMConfig":
        env_enabled = os.environ.get("STATEFLOW_LLM_ENABLED") or os.environ.get("LLM_ENABLED")
        env_required = os.environ.get("STATEFLOW_LLM_REQUIRED") or os.environ.get("LLM_REQUIRED")
        return cls(
            enabled=_as_bool(env_enabled) if enabled is None else enabled,
            required=_as_bool(env_required) if required is None else required,
            provider=provider or os.environ.get("STATEFLOW_LLM_PROVIDER") or os.environ.get("LLM_PROVIDER") or cls.provider,
            base_url=base_url or os.environ.get("STATEFLOW_LLM_BASE_URL") or os.environ.get("OLLAMA_HOST") or os.environ.get("OPENAI_BASE_URL") or cls.base_url,
            model=model or os.environ.get("STATEFLOW_LLM_MODEL") or os.environ.get("OPENAI_MODEL") or cls.model,
            api_key=api_key or os.environ.get("STATEFLOW_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or cls.api_key,
            timeout=timeout if timeout is not None else int(os.environ.get("STATEFLOW_LLM_TIMEOUT", str(cls.timeout))),
            temperature=temperature if temperature is not None else float(os.environ.get("STATEFLOW_LLM_TEMPERATURE", str(cls.temperature))),
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "required": self.required,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "timeout": self.timeout,
            "temperature": self.temperature,
        }


def _as_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in TRUE_VALUES
