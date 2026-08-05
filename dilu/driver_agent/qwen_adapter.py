"""Online chat-completions client for the RGD slow path."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class QwenChatModel:
    """Minimal OpenAI-compatible chat-completions client."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.model_name = str(model_name or os.environ["QWEN_CHAT_MODEL"]).strip()
        expected_model = str(os.environ.get("QWEN_CHAT_MODEL_CONFIG", "") or "").strip()
        if expected_model and self.model_name != expected_model:
            raise ValueError(f"QWEN model drift: expected {expected_model!r}, got {self.model_name!r}")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.request_timeout_s = max(0.1, float(os.environ["QWEN_REQUEST_TIMEOUT_S"]))
        self.provider = str(os.environ.get("LLM_PROVIDER", "siliconflow") or "siliconflow").strip().lower()
        self.base_url, self.api_key = self._resolve_endpoint(self.provider)
        if not self.api_key:
            raise ValueError(f"{self.provider} API key is empty")
        self.enable_thinking = self._optional_bool(os.environ.get("QWEN_ENABLE_THINKING", ""))
        self.thinking_budget = self._optional_int(os.environ.get("QWEN_THINKING_BUDGET", ""))
        force_json = self._optional_bool(os.environ.get("LLM_FORCE_JSON_RESPONSE", ""))
        # Default: force JSON for siliconflow/tokenplan; optional elsewhere.
        if force_json is None:
            self.force_json_response = self.provider in {"siliconflow", "tokenplan"}
        else:
            self.force_json_response = bool(force_json)

    @staticmethod
    def _resolve_endpoint(provider: str) -> tuple:
        if provider == "tokenplan":
            return (
                str(os.environ["TOKENPLAN_BASE_URL"]).strip().rstrip("/"),
                str(os.environ["TOKENPLAN_API_KEY"]).strip(),
            )
        if provider in {"openai_compatible", "openai", "bizdecipher"}:
            base = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("BIZDECIPHER_BASE_URL")
                or "https://bizdecipher.com/v1"
            )
            key = os.environ.get("OPENAI_API_KEY") or os.environ.get("BIZDECIPHER_API_KEY") or ""
            return str(base).strip().rstrip("/"), str(key).strip()
        return (
            str(os.environ["SILICONFLOW_BASE_URL"]).strip().rstrip("/"),
            str(os.environ["SILICONFLOW_API_KEY"]).strip(),
        )

    @staticmethod
    def _optional_bool(value: Any) -> Optional[bool]:
        text = str(value or "").strip().lower()
        if not text:
            return None
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid boolean value: {value!r}")

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = int(text)
        if parsed <= 0:
            raise ValueError(f"expected positive integer, got {value!r}")
        return parsed

    @staticmethod
    def _extract_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _resolve_proxies() -> Dict[str, str]:
        proxies: Dict[str, str] = {}
        http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
        https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        return proxies

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            "temperature": float(self.temperature),
            "stream": False,
        }
        if self.force_json_response:
            payload["response_format"] = {"type": "json_object"}

        # tokenplan-style max_completion_tokens; OpenAI-compatible and SiliconFlow use max_tokens.
        if self.provider == "tokenplan":
            payload["max_completion_tokens"] = int(self.max_tokens)
        else:
            payload["max_tokens"] = int(self.max_tokens)

        if self.enable_thinking is not None:
            payload["enable_thinking"] = bool(self.enable_thinking)
        if self.thinking_budget is not None:
            payload["thinking_budget"] = int(self.thinking_budget)

        with requests.Session() as session:
            session.trust_env = False
            session.proxies.update(self._resolve_proxies())
            response = session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.request_timeout_s,
            )

        if int(response.status_code) >= 400:
            raise RuntimeError(f"{self.provider} HTTP {response.status_code}: {str(response.text or '')[:300]}")
        body = dict(response.json() or {})
        choices = list(body.get("choices", []) or [])
        if not choices:
            raise RuntimeError(f"{self.provider} response contains no choices")
        message = dict((choices[0] or {}).get("message", {}) or {})
        content = self._extract_text(message.get("content", ""))
        if not content.strip():
            # Some models put text under reasoning/content variants.
            content = self._extract_text(message.get("reasoning_content", "")) or content
        if not content.strip():
            raise RuntimeError(f"{self.provider} response content is empty")
        return content

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.provider == "tokenplan":
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
