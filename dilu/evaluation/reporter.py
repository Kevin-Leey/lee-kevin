"""Experiment manifests and episode-result persistence for the RGD runtime."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit


_SECRET_MARKERS = ("api_key", "apikey", "password", "secret", "token", "authorization")
_PROTOCOL_HASH_EXCLUDED = {
    "protocol_id",
    "protocol_hash",
    "config_hash",
    "source_hash",
    "timestamp",
    "source_control",
    "runtime_environment",
    "git_hash",
    "git_dirty",
}
_RETRYABLE_HTTP_STATUSES = (408, 409, 425, 429, 500, 502, 503, 504)
_RETRYABLE_TRANSPORT_FAILURES = ("timeout", "connection_error")


def _canonical_json(payload: Any) -> str:
    # Match the independently distributed acceptance verifier's canonical
    # runtime-hash encoding exactly.
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False)


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _is_secret_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    return any(marker in normalized for marker in _SECRET_MARKERS) or normalized in {"openai_key", "key"}


def _public_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(item)
            for key, item in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, (list, tuple, set)):
        return [_public_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _base_url_origin(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.hostname:
        return ""
    host = parsed.hostname
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}"


def _protocol_executor_default(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = dict(cfg.get("_paper_protocol_config", {}) or {})
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    table = dict(submission.get("table_vii", {}) or {})
    models = list(table.get("slow_executor_models", []) or [])
    if models and isinstance(models[0], Mapping):
        return dict(models[0])
    return {}


def _resolve_llm_backend_snapshot(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a credential-free description of the configured slow executor."""
    config = dict(cfg or {})
    protocol_default = _protocol_executor_default(config)
    requested_provider = str(
        config.get("LLM_PROVIDER", protocol_default.get("provider", "")) or ""
    ).strip().lower()
    provider = requested_provider.replace("-", "_") or "openai_compatible"
    if provider == "siliconflow":
        model = str(
            config.get(
                "SILICONFLOW_CHAT_MODEL",
                config.get("QWEN_CHAT_MODEL", protocol_default.get("model", "")),
            )
            or ""
        )
        base_url = config.get("SILICONFLOW_BASE_URL", protocol_default.get("base_url", ""))
    elif provider in {"openai", "openai_compatible", "azure"}:
        model = str(config.get("OPENAI_CHAT_MODEL", config.get("QWEN_CHAT_MODEL", "")) or "")
        base_url = config.get("OPENAI_BASE_URL", config.get("AZURE_API_BASE", ""))
    else:
        model = str(config.get("QWEN_CHAT_MODEL", config.get("OPENAI_CHAT_MODEL", "")) or "")
        base_url = config.get("QWEN_BASE_URL", config.get("OPENAI_BASE_URL", ""))
    retryable_statuses = config.get("LLM_RETRYABLE_HTTP_STATUSES", _RETRYABLE_HTTP_STATUSES)
    if not isinstance(retryable_statuses, (list, tuple)):
        retryable_statuses = _RETRYABLE_HTTP_STATUSES
    retryable_failures = config.get(
        "LLM_RETRYABLE_TRANSPORT_FAILURES", _RETRYABLE_TRANSPORT_FAILURES
    )
    if not isinstance(retryable_failures, (list, tuple)):
        retryable_failures = _RETRYABLE_TRANSPORT_FAILURES
    return {
        "provider": provider,
        "requested_model": model,
        "resolved_chat_model": model,
        "base_url_origin": _base_url_origin(base_url),
        "temperature": float(config.get("QWEN_TEMPERATURE", 0.0) or 0.0),
        "request_timeout_s": float(config.get("QWEN_REQUEST_TIMEOUT_S", 0.0) or 0.0),
        "reasoning_effort": config.get("LLM_REASONING_EFFORT"),
        "retry_contract": {
            "max_attempts": int(config.get("LLM_MAX_ATTEMPTS", 1) or 1),
            "initial_backoff_s": float(config.get("LLM_RETRY_BACKOFF_S", 0.0) or 0.0),
            "retryable_http_statuses": [int(value) for value in retryable_statuses],
            "retryable_transport_failures": [str(value) for value in retryable_failures],
        },
    }


def _build_runtime_experiment_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a serializable public configuration without credentials."""
    config = dict(_public_value(dict(cfg or {})))
    # The runtime uses private names internally.  Publish stable aliases so
    # independent acceptance tools never need to infer an implementation key.
    if isinstance(config.get("_rgd_runtime_contract"), Mapping):
        config["runtime_contract"] = dict(config["_rgd_runtime_contract"])
    if isinstance(config.get("_v12_floor_overlay"), Mapping):
        config["v12_floor_overlay"] = dict(config["_v12_floor_overlay"])
    return config


def build_runtime_source_hash(repo_root: Optional[Path] = None) -> str:
    """Hash source files that define runtime behavior, independent of result output."""
    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    candidates = []
    for relative in ("dilu", "tools"):
        directory = root / relative
        if directory.is_dir():
            candidates.extend(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_state_memory_artifact(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Describe the optional state-memory input without serializing its data."""
    config = dict(cfg or {})
    raw_path = str(config.get("memory_path", "") or "").strip()
    path = Path(raw_path).expanduser() if raw_path else None
    resolved = path.resolve() if path is not None else None
    exists = bool(resolved is not None and resolved.exists())
    return {
        "path": str(resolved) if resolved is not None else "",
        "exists": exists,
        "size_bytes": int(resolved.stat().st_size) if exists and resolved.is_file() else 0,
        "sha256": _sha256_file(resolved) if exists and resolved.is_file() else None,
        "runtime_enabled": bool(config.get("enable_memory_retrieval", False)),
    }


def _source_control_snapshot(root: Path) -> Dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(root), stderr=subprocess.DEVNULL
        ).decode("utf-8")
        return {"git_hash": revision, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_hash": "", "git_dirty": None}


def _protocol_source_path(cfg: Mapping[str, Any]) -> str:
    protocol = dict(cfg.get("_paper_protocol_config", {}) or {})
    source = str(
        protocol.get("_source_path")
        or dict(cfg.get("training", {}) or {}).get("protocol", {}).get("source_path")
        or ""
    ).strip()
    return str(Path(source).resolve()) if source else ""


def _selected_group_spec(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = dict(cfg.get("_paper_protocol_config", {}) or {})
    group_name = str(cfg.get("group_name", protocol.get("selected_group", "")) or "")
    groups = dict(protocol.get("groups", {}) or {})
    spec = groups.get(group_name, {})
    return dict(spec) if isinstance(spec, Mapping) else {}


def _protocol_manifest(cfg: Mapping[str, Any], public_config: Mapping[str, Any]) -> Dict[str, Any]:
    protocol = dict(cfg.get("_paper_protocol_config", {}) or {})
    overlay = public_config.get("v12_floor_overlay")
    return {
        "protocol_name": str(cfg.get("protocol_name", protocol.get("protocol_name", "rgd_runtime")) or "rgd_runtime"),
        "protocol_version": protocol.get("protocol_version", cfg.get("protocol_version")),
        "selected_group": str(cfg.get("group_name", protocol.get("selected_group", "")) or ""),
        "selected_environment": str(cfg.get("env_type", protocol.get("selected_environment", "")) or ""),
        "source_path": _protocol_source_path(cfg),
        "config": {
            "v12_floor_overlay": dict(overlay) if isinstance(overlay, Mapping) else None,
        },
        "v12_floor_overlay": dict(overlay) if isinstance(overlay, Mapping) else None,
    }


def _runtime_manifest_preimage(
    cfg: Mapping[str, Any], public_config: Mapping[str, Any], seed: Optional[int]
) -> Dict[str, Any]:
    protocol = dict(cfg.get("_paper_protocol_config", {}) or {})
    group = _selected_group_spec(cfg)
    guardrails = dict(protocol.get("claim_guardrails", {}) or {})
    slow = dict(cfg.get("slow_thinking", {}) or {})
    trace_cache = dict(slow.get("trace_cache", {}) or {})
    root = Path(__file__).resolve().parents[2]
    resolved_seed = public_config.get("fixed_seed_override") if seed is None else int(seed)
    return {
        "schema_version": "rgd_runtime_manifest_v2",
        "config": dict(public_config),
        "fixed_seed_override": resolved_seed,
        "seed_start": resolved_seed,
        "resolved_seeds": [] if resolved_seed is None else [int(resolved_seed)],
        "protocol_name": str(cfg.get("protocol_name", "rgd_runtime") or "rgd_runtime"),
        "experiment_name": str(cfg.get("experiment_name", cfg.get("group_name", "")) or ""),
        "env_type": str(cfg.get("env_type", "") or ""),
        "scenario_type": str(cfg.get("scenario_type", "") or ""),
        "simulation_duration": public_config.get("simulation_duration"),
        "experiment_plan_path": _protocol_source_path(cfg),
        "config_scope": "protocol_bound_setting",
        "single_core_method_name": str(
            guardrails.get("single_core_method_name", "Recoverability-Gated Deliberation")
        ),
        "primary_evaluation_subject": str(
            guardrails.get("primary_evaluation_subject", "fixed-policy RGD")
        ),
        "recoverability_core_variables": list(
            guardrails.get("recoverability_core_variables", []) or []
        ),
        "recoverability_object_status": str(
            guardrails.get("recoverability_object_status", "") or ""
        ),
        "paper_role": str(group.get("paper_role", "paper_baseline") or "paper_baseline"),
        "publication_track": str(group.get("publication_track", "main_text") or "main_text"),
        "theory_family": str(group.get("theory_family", "unclassified") or "unclassified"),
        "alternative_explanation_axis": str(
            group.get("alternative_explanation_axis", "unspecified") or "unspecified"
        ),
        "ablation_dimension": str(group.get("ablation_dimension", "none") or "none"),
        "enable_memory_retrieval": bool(cfg.get("enable_memory_retrieval", False)),
        "few_shot_num": int(cfg.get("few_shot_num", 0) or 0),
        "memory_artifacts": {"state_memory": _build_state_memory_artifact(cfg)},
        "slow_path_provenance": {
            "executor": str(slow.get("executor", "") or ""),
            "trace_cache_enabled": bool(trace_cache.get("enable", False)),
        },
        "llm_backend": _resolve_llm_backend_snapshot(cfg),
        "protocol_manifest": _protocol_manifest(cfg, public_config),
        "runtime_environment": _runtime_environment(),
        "source_control": _source_control_snapshot(root),
    }


def build_experiment_identity(cfg: Mapping[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    """Build stable configuration, protocol, and source identity hashes."""
    public_config = _build_runtime_experiment_config(cfg)
    if seed is not None:
        public_config["fixed_seed_override"] = int(seed)
    resolved_seed = public_config.get("fixed_seed_override")
    preimage = _runtime_manifest_preimage(cfg, public_config, seed)
    protocol_hash = _sha256_payload(
        {
            key: value
            for key, value in preimage.items()
            if key not in _PROTOCOL_HASH_EXCLUDED
        }
    )
    experiment_name = str(preimage.get("experiment_name", "") or "rgd_runtime")
    root = Path(__file__).resolve().parents[2]
    return {
        "protocol_id": f"{experiment_name}::{protocol_hash[:16]}",
        "protocol_hash": protocol_hash,
        "config": public_config,
        "config_hash": _sha256_payload(public_config),
        "source_hash": build_runtime_source_hash(root),
        "fixed_seed_override": resolved_seed,
        "resolved_seeds": [] if resolved_seed is None else [int(resolved_seed)],
        "protocol_manifest": dict(preimage["protocol_manifest"]),
        "_runtime_manifest_preimage": preimage,
    }


def _package_version(name: str) -> str:
    try:
        return str(importlib_metadata.version(name))
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _runtime_environment() -> Dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pkg_numpy": _package_version("numpy"),
        "pkg_gymnasium": _package_version("gymnasium"),
        "pkg_highway-env": _package_version("highway-env"),
    }


def save_experiment_snapshot(cfg: Mapping[str, Any], result_dir: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Persist matching runtime and experiment manifests for a setting/seed."""
    root = Path(result_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = build_experiment_identity(cfg, seed)
    preimage = dict(identity.pop("_runtime_manifest_preimage"))
    manifest = {**preimage, **identity}
    snapshot = {
        **preimage,
        **identity,
        "schema_version": "rgd_experiment_snapshot_v2",
        "seed_start": identity["fixed_seed_override"],
        "seeds_used": list(identity["resolved_seeds"]),
        "protocol_manifest_path": str((root / "runtime_manifest.json").resolve()),
    }
    for name, payload in (("runtime_manifest.json", manifest), ("experiment_snapshot.json", snapshot)):
        (root / name).write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def save_episode_results(**kwargs: Any) -> Dict[str, Any]:
    """Persist recorder outputs and add them to an optional aggregate collector."""
    result_dir = Path(str(kwargs.get("result_dir", ".")))
    prefix = str(kwargs.get("prefix", "episode"))
    ep = int(kwargs.get("ep", 0))
    physical = kwargs.get("phys_rec")
    reasoning = kwargs.get("reas_rec")
    metrics_agg = kwargs.get("metrics_agg")

    physical_payload = None
    reasoning_payload = None
    for recorder, method_name, label in (
        (physical, "save", "physical"),
        (reasoning, "save", "reasoning"),
    ):
        method = getattr(recorder, method_name, None)
        if callable(method):
            payload = method()
            if label == "physical":
                physical_payload = payload
            else:
                reasoning_payload = payload
    add = getattr(metrics_agg, "add_episode", None)
    if callable(add):
        add(physical_payload=physical_payload, reasoning_payload=reasoning_payload)

    payload = {
        "episode_id": ep,
        "prefix": prefix,
        "collision_frame": int(kwargs.get("collision_frame", -1)),
        "frame_count": len(list(kwargs.get("frame_runtimes", []) or [])),
        "event_log": str(kwargs.get("event_log_path", "")),
    }
    output = result_dir / f"episode_result_{prefix}_{ep}.json"
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "_build_state_memory_artifact",
    "_build_runtime_experiment_config",
    "_resolve_llm_backend_snapshot",
    "build_experiment_identity",
    "build_runtime_source_hash",
    "save_episode_results",
    "save_experiment_snapshot",
]
