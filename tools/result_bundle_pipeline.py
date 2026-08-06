from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
        return float(value)
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
            return float(text)
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
) -> Path:
    bundle_root = root / run_mode / run_stamp
    bundle_root.mkdir(parents=True, exist_ok=True)
    manifest_path = bundle_root / "result_bundle_manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
    effective_groups = list(_ordered_unique(list(existing.get("groups", []) or []) + list(groups)))
    effective_envs = list(_ordered_unique(list(existing.get("envs", []) or []) + list(envs)))
    effective_group_env_matrix: Dict[str, List[str]] = {
        str(group_name): list(env_list or [])
        for group_name, env_list in dict(existing.get("group_env_matrix", {}) or {}).items()
    }
    for group_name, env_list in dict(group_env_matrix or {}).items():
        effective_group_env_matrix[str(group_name)] = _ordered_unique(
            list(effective_group_env_matrix.get(str(group_name), [])) + list(env_list or [])
        )
    resolved_seed_labels = (
        [int(seed_value)] * int(seeds)
        if seed_value is not None
        else list(range(int(seed_start), int(seed_start) + int(seeds)))
    )
    manifest = {
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "bundle_kind": str(run_mode),
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
        "simulation_duration": (None if simulation_duration is None else int(simulation_duration)),
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
