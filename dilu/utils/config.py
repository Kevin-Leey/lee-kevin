"""Configuration loading and online LLM environment setup."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Tuple


DEFAULT_REQUEST_TIMEOUT_S = 120.0


def load_config(path: str = "config.yaml") -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, encoding="utf-8-sig") as handle:
        payload = yaml_safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"config file must contain a YAML mapping: {path}")
    return payload


def yaml_safe_load(handle) -> Any:
    import yaml

    return yaml.safe_load(handle)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any], *, copy_base: bool = False) -> Dict[str, Any]:
    target = deepcopy(dict(base or {})) if copy_base else (base if base is not None else {})
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value, copy_base=False)
        else:
            target[key] = deepcopy(value) if copy_base else value
    return target


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _required_text(cfg: Dict[str, Any], key: str) -> str:
    value = _first_text(cfg.get(key, ""), os.environ.get(key, ""))
    if not value:
        raise ValueError(f"missing required config key: {key}")
    return value


def resolve_llm_endpoint(
    cfg: Dict[str, Any], *, require_api_key: bool = True
) -> Tuple[str, str, str, str]:
    """Return (provider, model, api_key, base_url).

    Supported providers:
      - siliconflow
      - bizdecipher / openai / openai_compatible
      - tokenplan / mimo (legacy)

    Provenance-only callers may set ``require_api_key=False`` so offline
    manifests can record public endpoint metadata without loading a credential.
    """
    configured_provider = str(cfg.get("LLM_PROVIDER", "") or "").strip().lower()
    legacy_api_type = str(cfg.get("OPENAI_API_TYPE", "") or "").strip().lower()
    if not configured_provider and legacy_api_type == "openai":
        provider = "openai_compatible"
    elif not configured_provider and legacy_api_type == "azure":
        raise ValueError(
            "the reconstructed runtime uses an OpenAI-compatible chat endpoint; "
            "configure LLM_PROVIDER=openai_compatible instead of legacy Azure settings"
        )
    else:
        provider = configured_provider or "siliconflow"
    if provider in {"tokenplan", "mimo", "xiaomi-mimo"}:
        model = _required_text(cfg, "TOKENPLAN_CHAT_MODEL")
        api_key = _first_text(cfg.get("TOKENPLAN_API_KEY"), os.environ.get("TOKENPLAN_API_KEY"))
        if require_api_key and not api_key:
            raise ValueError("missing required config key: TOKENPLAN_API_KEY")
        base_url = _first_text(
            cfg.get("TOKENPLAN_BASE_URL"), "https://token-plan-cn.xiaomimimo.com/v1"
        ).rstrip("/")
        return "tokenplan", model, api_key, base_url
    if provider in {"bizdecipher", "openai", "openai_compatible", "openai-compatible"}:
        model = _first_text(
            cfg.get("OPENAI_CHAT_MODEL"),
            cfg.get("BIZDECIPHER_CHAT_MODEL"),
            cfg.get("SILICONFLOW_CHAT_MODEL"),
            os.environ.get("OPENAI_CHAT_MODEL"),
            os.environ.get("BIZDECIPHER_CHAT_MODEL"),
        )
        api_key = _first_text(
            cfg.get("OPENAI_API_KEY"),
            cfg.get("OPENAI_KEY"),
            cfg.get("BIZDECIPHER_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
            os.environ.get("OPENAI_KEY"),
            os.environ.get("BIZDECIPHER_API_KEY"),
        )
        default_base_url = (
            "https://api.openai.com/v1"
            if legacy_api_type == "openai" and not configured_provider
            else "https://bizdecipher.com/v1"
        )
        base_url = _first_text(
            cfg.get("OPENAI_BASE_URL"),
            cfg.get("BIZDECIPHER_BASE_URL"),
            os.environ.get("OPENAI_BASE_URL"),
            os.environ.get("BIZDECIPHER_BASE_URL"),
            default_base_url,
        ).rstrip("/")
        if not model:
            raise ValueError("openai-compatible provider requires OPENAI_CHAT_MODEL (or BIZDECIPHER_CHAT_MODEL)")
        if require_api_key and not api_key:
            raise ValueError("openai-compatible provider requires OPENAI_API_KEY (or BIZDECIPHER_API_KEY)")
        return "openai_compatible", model, api_key, base_url

    # default: siliconflow
    model = _required_text(cfg, "SILICONFLOW_CHAT_MODEL")
    api_key = _first_text(cfg.get("SILICONFLOW_API_KEY"), os.environ.get("SILICONFLOW_API_KEY"))
    if require_api_key and not api_key:
        raise ValueError("missing required config key: SILICONFLOW_API_KEY")
    base_url = _first_text(
        cfg.get("SILICONFLOW_BASE_URL"),
        os.environ.get("SILICONFLOW_BASE_URL"),
        "https://api.siliconflow.cn/v1",
    ).rstrip("/")
    return "siliconflow", model, api_key, base_url


def setup_api(cfg: Dict) -> None:
    provider, model, api_key, base_url = resolve_llm_endpoint(cfg)
    timeout_s = max(
        0.1,
        float(cfg.get("QWEN_REQUEST_TIMEOUT_S", DEFAULT_REQUEST_TIMEOUT_S) or DEFAULT_REQUEST_TIMEOUT_S),
    )

    os.environ["LLM_PROVIDER"] = provider
    os.environ["QWEN_CHAT_MODEL"] = model
    os.environ["QWEN_CHAT_MODEL_CONFIG"] = model
    os.environ["QWEN_TEMPERATURE"] = str(float(cfg.get("QWEN_TEMPERATURE", 0.0) or 0.0))
    os.environ["QWEN_REQUEST_TIMEOUT_S"] = str(timeout_s)

    # Clear sibling provider env vars so a stale key cannot leak across runs.
    for key in (
        "TOKENPLAN_API_KEY",
        "TOKENPLAN_BASE_URL",
        "TOKENPLAN_CHAT_MODEL",
        "SILICONFLOW_API_KEY",
        "SILICONFLOW_BASE_URL",
        "SILICONFLOW_CHAT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_CHAT_MODEL",
        "BIZDECIPHER_API_KEY",
        "BIZDECIPHER_BASE_URL",
        "BIZDECIPHER_CHAT_MODEL",
    ):
        os.environ.pop(key, None)

    if provider == "tokenplan":
        os.environ["TOKENPLAN_API_KEY"] = api_key
        os.environ["TOKENPLAN_BASE_URL"] = base_url
        os.environ["TOKENPLAN_CHAT_MODEL"] = model
    elif provider == "openai_compatible":
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_CHAT_MODEL"] = model
        os.environ["BIZDECIPHER_API_KEY"] = api_key
        os.environ["BIZDECIPHER_BASE_URL"] = base_url
        os.environ["BIZDECIPHER_CHAT_MODEL"] = model
    else:
        os.environ["SILICONFLOW_API_KEY"] = api_key
        os.environ["SILICONFLOW_BASE_URL"] = base_url
        os.environ["SILICONFLOW_CHAT_MODEL"] = model

    if "QWEN_ENABLE_THINKING" in cfg:
        os.environ["QWEN_ENABLE_THINKING"] = str(bool(cfg.get("QWEN_ENABLE_THINKING", False))).lower()
    else:
        os.environ.pop("QWEN_ENABLE_THINKING", None)
    if str(cfg.get("QWEN_THINKING_BUDGET", "") or "").strip():
        os.environ["QWEN_THINKING_BUDGET"] = str(int(cfg["QWEN_THINKING_BUDGET"]))
    else:
        os.environ.pop("QWEN_THINKING_BUDGET", None)

    # Optional response-format control: some third-party models reject json_object.
    if "LLM_FORCE_JSON_RESPONSE" in cfg:
        os.environ["LLM_FORCE_JSON_RESPONSE"] = str(bool(cfg.get("LLM_FORCE_JSON_RESPONSE", True))).lower()
    else:
        os.environ.pop("LLM_FORCE_JSON_RESPONSE", None)
    if "LLM_STRICT_JSON_SCHEMA" in cfg:
        os.environ["LLM_STRICT_JSON_SCHEMA"] = str(bool(cfg.get("LLM_STRICT_JSON_SCHEMA", False))).lower()
    else:
        os.environ.pop("LLM_STRICT_JSON_SCHEMA", None)

    proxy_cfg = dict(cfg.get("proxy", {}) or {})
    for env_key, cfg_key in (("http_proxy", "http_proxy"), ("https_proxy", "https_proxy")):
        value = str(proxy_cfg.get(cfg_key, "") or "").strip()
        if value:
            os.environ[env_key] = value
            os.environ[env_key.upper()] = value
