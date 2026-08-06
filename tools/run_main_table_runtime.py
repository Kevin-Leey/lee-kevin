"""Configuration assembly for protocol-bound experiment runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in dict(override).items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_formal_protocol(path: Path | str) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"formal protocol must be a mapping: {source}")
    protocol = deepcopy(dict(payload))
    groups = protocol.get("groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("formal protocol must define at least one group")
    protocol["_source_path"] = str(source)
    return protocol


def load_formal_base_config(
    protocol: Mapping[str, Any], config_path: Path | str | None = None
) -> dict[str, Any]:
    source = Path(config_path) if config_path is not None else REPO_ROOT / "config.yaml"
    base_payload = yaml.safe_load(source.read_text(encoding="utf-8-sig")) or {}
    if not isinstance(base_payload, Mapping):
        raise ValueError(f"base config must be a mapping: {source}")
    runtime = protocol.get("runtime_config", {})
    if runtime is None:
        runtime = {}
    if not isinstance(runtime, Mapping):
        raise ValueError("protocol runtime_config must be a mapping")
    config = _deep_merge(base_payload, runtime)
    config.setdefault("protocol_name", protocol.get("protocol_name", "rgd_runtime"))
    config.setdefault("protocol_version", protocol.get("protocol_version"))
    config.setdefault("fixed_seed_override", None)
    config.setdefault("training", {})
    config["training"] = _deep_merge(
        config["training"], {"protocol": {"source_path": str(protocol.get("_source_path", ""))}}
    )
    return config


def iter_selected_groups(
    groups: Mapping[str, Any], requested: Sequence[str] | None = None
) -> Iterable[tuple[str, dict[str, Any]]]:
    available = dict(groups or {})
    selected = list(requested) if requested else list(available)
    resolved: list[tuple[str, dict[str, Any]]] = []
    for name in selected:
        if name not in available:
            raise ValueError(f"formal protocol does not define requested group: {name}")
        spec = available[name]
        if not isinstance(spec, Mapping):
            raise ValueError(f"group {name} must be a mapping")
        resolved.append((str(name), deepcopy(dict(spec))))
    return iter(resolved)


def _scenario_type(env_type: str) -> str:
    normalized = str(env_type).lower()
    if "merge" in normalized:
        return "merge"
    if "roundabout" in normalized:
        return "roundabout"
    if "intersection" in normalized:
        return "intersection"
    return "highway"


def build_group_config(
    base_cfg: Mapping[str, Any],
    group_name: str,
    group_cfg: Mapping[str, Any],
    env_type: str,
    episodes: int,
    result_dir: Path | str,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one group/environment setting to the immutable protocol contract."""
    if int(episodes) <= 0:
        raise ValueError("episodes must be positive")
    spec = dict(group_cfg or {})
    overrides = spec.get("runtime_overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError(f"runtime overrides for {group_name} must be a mapping")
    by_env = protocol.get("runtime_config_by_env", {}) or {}
    if not isinstance(by_env, Mapping):
        raise ValueError("protocol runtime_config_by_env must be a mapping")
    env_overrides = by_env.get(str(env_type), {}) or {}
    if not isinstance(env_overrides, Mapping):
        raise ValueError(f"runtime_config_by_env.{env_type} must be a mapping")
    cfg = _deep_merge(base_cfg, env_overrides)
    cfg = _deep_merge(cfg, overrides)
    cfg.update(
        {
            "experiment_name": str(group_name),
            "group_name": str(group_name),
            "group_id": str(spec.get("id", group_name)),
            "protocol_name": str(cfg.get("protocol_name", group_name) or group_name),
            "protocol_version": protocol.get("protocol_version"),
            "env_type": str(env_type),
            "scenario_type": _scenario_type(str(env_type)),
            "episodes_num": int(episodes),
            "result_folder": str(Path(result_dir)),
        }
    )
    cfg.setdefault("fixed_seed_override", None)
    cfg.setdefault("slow_thinking", {})
    cfg.setdefault("system_routing", {})

    protocol_snapshot = deepcopy(dict(protocol))
    # The snapshot contains the resolved public runtime configuration used by
    # this setting.  Keep it separate from cfg itself to avoid recursive data.
    snapshot_runtime = deepcopy(cfg)
    snapshot_runtime.pop("_paper_protocol_config", None)
    protocol_snapshot["runtime_config"] = snapshot_runtime
    protocol_snapshot["selected_group"] = str(group_name)
    protocol_snapshot["selected_environment"] = str(env_type)
    cfg["_paper_protocol_config"] = protocol_snapshot
    return cfg


__all__ = [
    "REPO_ROOT",
    "build_group_config",
    "iter_selected_groups",
    "load_formal_base_config",
    "load_formal_protocol",
]
