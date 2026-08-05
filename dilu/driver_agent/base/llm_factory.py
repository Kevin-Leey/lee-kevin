"""Single-path LLM factory for the online slow path."""

import os
from typing import Any, Dict, Optional, Tuple


class LLMFactory:
    """Create cached online chat instances."""

    PRESETS: Dict[str, Dict[str, Any]] = {
        "slow": {"temperature": 0.0, "max_tokens": 64},
        "default": {"temperature": 0.0, "max_tokens": 256},
    }
    _instances: Dict[Tuple[str, float, int, str, str], Any] = {}

    @staticmethod
    def _temperature(default: float) -> float:
        raw = os.environ.get("QWEN_TEMPERATURE", "").strip()
        return float(raw) if raw else float(default)

    @classmethod
    def create(
        cls,
        purpose: str = "default",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        force_new: bool = False,
    ) -> Any:
        preset = dict(cls.PRESETS.get(purpose, cls.PRESETS["default"]))
        resolved_temperature = cls._temperature(float(temperature if temperature is not None else preset["temperature"]))
        resolved_tokens = int(max_tokens if max_tokens is not None else preset["max_tokens"])
        if resolved_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {resolved_tokens}")
        model = str(os.environ["QWEN_CHAT_MODEL"]).strip()
        provider = str(os.environ.get("LLM_PROVIDER", "siliconflow") or "siliconflow").strip().lower()
        if provider == "tokenplan":
            base_url = str(os.environ["TOKENPLAN_BASE_URL"]).strip().rstrip("/")
        elif provider in {"openai_compatible", "openai", "bizdecipher"}:
            base_url = str(
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("BIZDECIPHER_BASE_URL")
                or "https://bizdecipher.com/v1"
            ).strip().rstrip("/")
        else:
            base_url = str(os.environ["SILICONFLOW_BASE_URL"]).strip().rstrip("/")
        cache_key = (str(purpose), float(resolved_temperature), int(resolved_tokens), model, provider, base_url)
        if not force_new and cache_key in cls._instances:
            return cls._instances[cache_key]

        from dilu.driver_agent.qwen_adapter import QwenChatModel

        instance = QwenChatModel(
            temperature=float(resolved_temperature),
            max_tokens=int(resolved_tokens),
        )
        cls._instances[cache_key] = instance
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        cls._instances.clear()
