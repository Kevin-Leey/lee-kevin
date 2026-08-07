from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tools.protocol_io import dump_json


def _ordered_unique(items: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        ordered.append(str(item))
        seen.add(item)
    return ordered


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        resolved = float(value)
        return resolved if math.isfinite(resolved) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered == "true":
            return 1.0
        if lowered == "false":
            return 0.0
        if lowered in {"none", "null", "nan"}:
            return None
        try:
            resolved = float(text)
            return resolved if math.isfinite(resolved) else None
        except ValueError:
            return None
    return None


def write_result_bundle_manifest(
    root: Path,
    run_mode: str,
    run_stamp: str,
    groups: Sequence[str],
    envs: Sequence[str],
    seeds: int,
    episodes: int,
    seed_value: Optional[int],
    simulation_duration: Optional[int],
    protocol_path: str,
    group_env_matrix: Dict[str, Sequence[str]],
    seed_start: int = 0,
    partition: str = "auto",
    execution_horizon_by_group_env: Optional[
        Mapping[str, Mapping[str, Mapping[str, Any]]]
    ] = None,
) -> Path:
    bundle_root = root / run_mode / run_stamp
    bundle_root.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_root / "result_bundle_manifest.json"
    effective_groups = list(_ordered_unique(list(groups)))
    effective_envs = list(_ordered_unique(list(envs)))
    effective_group_env_matrix: Dict[str, List[str]] = {
        str(group_name): _ordered_unique(list(env_list or []))
        for group_name, env_list in dict(group_env_matrix or {}).items()
    }
    resolved_seed_labels = (
        [int(seed_value)] * int(seeds)
        if seed_value is not None
        else list(range(int(seed_start), int(seed_start) + int(seeds)))
    )
    horizon_matrix = {
        str(group): {
            str(env): dict(horizon)
            for env, horizon in dict(env_horizons or {}).items()
        }
        for group, env_horizons in dict(execution_horizon_by_group_env or {}).items()
    }
    unique_horizons = {
        json.dumps(horizon, sort_keys=True, separators=(",", ":")): horizon
        for env_horizons in horizon_matrix.values()
        for horizon in env_horizons.values()
    }
    uniform_horizon = (
        next(iter(unique_horizons.values())) if len(unique_horizons) == 1 else {}
    )
    effective_duration = uniform_horizon.get("episode_duration_s")
    manifest = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bundle_kind": str(run_mode),
        "partition": str(partition),
        "bundle_root": str(bundle_root),
        "groups": effective_groups,
        "group_env_matrix": effective_group_env_matrix,
        "envs": effective_envs,
        "seeds": int(seeds),
        "episodes": int(episodes),
        "seed_policy": "fixed_per_setting",
        "seed_start": int(seed_start),
        "seed_labels": resolved_seed_labels,
        "seed_value": (None if seed_value is None else int(seed_value)),
        "simulation_duration": (
            effective_duration
            if effective_duration is not None
            else (None if simulation_duration is None else int(simulation_duration))
        ),
        "episode_duration_s": uniform_horizon.get("episode_duration_s"),
        "policy_frequency_hz": uniform_horizon.get("policy_frequency_hz"),
        "simulation_frequency_hz": uniform_horizon.get("simulation_frequency_hz"),
        "expected_policy_steps": uniform_horizon.get("expected_policy_steps"),
        "execution_horizon_by_group_env": horizon_matrix,
        "formal_protocol_path": protocol_path,
        "entry_artifacts": ["result_bundle_manifest.json", "overall_group_comparison.csv", "overall_group_comparison.json"],
    }
    dump_json(manifest_path, manifest)
    return manifest_path


def summarise_rows(rows: Sequence[Dict[str, Any]], group_name: str, group_id: str, output_path: Path) -> None:
    if not rows:
        return
    numeric_keys = [key for key in rows[0].keys() if key not in {"group", "group_id", "env", "result_dir"}]
    aggregate: Dict[str, Any] = {"group_name": group_name, "group_id": group_id, "updated": datetime.now().strftime("%Y-%m-%d %H:%M"), "total_runs": len(rows), "aggregate": {}}
    for key in numeric_keys:
        values: List[float] = []
        for row in rows:
            value = _coerce_numeric(row.get(key))
            if value is not None:
                values.append(value)
        if values:
            aggregate["aggregate"][f"{key}_mean"] = sum(values) / len(values)
    dump_json(output_path, aggregate)
