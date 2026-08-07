"""Run protocol-bound closed-loop main-table experiments.

Each output cell owns one simulator seed and its immutable runtime snapshot.
The runner deliberately writes a row only after the cell's traces, metrics, and
identity manifests have been persisted.  Consequently, ``--resume`` can skip
only cells whose recorded identity still matches the currently assembled
configuration.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.formal_surface import COMPARISON_HEADLINE_FIELDS  # noqa: E402
from dilu.evaluation.metrics_aggregator import MetricsAggregator  # noqa: E402
from dilu.evaluation.reporter import (  # noqa: E402
    build_experiment_identity,
    build_runtime_source_hash,
    save_experiment_snapshot,
)
from dilu.runtime_episode_finalize import finalize_episode_outputs  # noqa: E402
from dilu.runtime_episode_setup import (  # noqa: E402
    create_episode_agent,
    create_episode_env,
    create_episode_recorders,
)
from dilu.runtime_frame_trace import (  # noqa: E402
    collect_runtime_integrity,
    create_episode_runtime_state,
)
from dilu.runtime_support import (  # noqa: E402
    exclude_policy_pacing_sleep,
    execute_episode_step,
)
from dilu.safety import UnifiedSafetySystem  # noqa: E402
from dilu.scenario import create_scenario  # noqa: E402
from dilu.scenario.env_ids import infer_env_label  # noqa: E402
from tools.protocol_io import dump_json, load_csv_rows  # noqa: E402
from tools.result_bundle_pipeline import (  # noqa: E402
    summarise_rows,
    write_result_bundle_manifest,
)
from tools.run_main_table_runtime import (  # noqa: E402
    build_group_config,
    iter_selected_groups,
    load_formal_base_config,
    load_formal_protocol,
    resolve_policy_execution_horizon,
    validate_policy_execution_horizon,
)
from tools.run_main_table_support import (  # noqa: E402
    write_overall_comparison_assets,
    write_rows_csv,
)


V12_MAIN_GROUPS = (
    "rgd_fixed_policy",
    "always_fast",
    "always_slow",
    "random_budget",
    "uncertainty_budget",
    "risk_budget",
)
V12_MAIN_ENV = "highway-v0"
V12_MAIN_SEEDS = tuple(range(4000, 4030))
V13_PROTOCOL_NAME = "rgd_tvt_action_aligned_release_v13"

RUN_ROW_FIELDS = (
    "group",
    "group_id",
    "env",
    "seed_idx",
    "fixed_seed_override",
    "seed_start",
    "requested_seed_start",
    "episodes_run",
    "total_frames",
    "result_dir",
    "protocol_id",
    "protocol_hash",
    "config_hash",
    "source_hash",
    "evaluation_runtime_stable",
    "runtime_integrity_clean",
    "runtime_integrity_violation_rate",
    "slow_call_rate",
    "slow_call_success_rate",
    "request_count",
    "request_issued_count",
    "request_terminal_count",
    "valid_response_count",
    "timeout_count",
    "failure_count",
    "dropped_at_episode_end_count",
    "request_pending_count",
    "request_lifecycle_closed",
    "terminal_wall_latency_count",
    "terminal_wall_latency_mean_s",
    "simulator_e2e_latency_count",
    "simulator_e2e_latency_mean_s",
    *COMPARISON_HEADLINE_FIELDS,
)


def _as_exact_int(value: Any) -> Optional[int]:
    """Return an integer only when the supplied value denotes one exactly."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and text.lstrip("+-").isdigit():
            return int(text)
    return None


def _sync_protocol_runtime_snapshot(cfg: MutableMapping[str, Any]) -> None:
    """Keep the embedded protocol snapshot aligned with effective overrides."""
    protocol = cfg.get("_paper_protocol_config")
    if not isinstance(protocol, MutableMapping):
        return
    runtime = copy.deepcopy(dict(cfg))
    runtime.pop("_paper_protocol_config", None)
    protocol["runtime_config"] = runtime


def _bind_setting_seed(cfg: MutableMapping[str, Any], seed: int) -> int:
    """Bind one fresh simulator seed to a setting and its saved protocol view."""
    resolved = _as_exact_int(seed)
    if resolved is None or resolved < 0:
        raise ValueError("setting seed must be a nonnegative integer")
    cfg["fixed_seed_override"] = int(resolved)
    _sync_protocol_runtime_snapshot(cfg)
    return int(resolved)


def _apply_cli_runtime_overrides(
    cfg: MutableMapping[str, Any], args: argparse.Namespace, seed_label: int
) -> int:
    """Apply only declared runtime overrides before a setting is snapshotted."""
    explicit_seed = getattr(args, "seed_value", None)
    selected_seed = seed_label if explicit_seed is None else explicit_seed
    resolved_seed = _bind_setting_seed(cfg, int(selected_seed))
    duration = getattr(args, "simulation_duration", None)
    if duration is not None:
        value = _as_exact_int(duration)
        if value is None or value <= 0:
            raise ValueError("simulation duration must be a positive integer")
        cfg["simulation_duration"] = int(value)
        _sync_protocol_runtime_snapshot(cfg)
    return resolved_seed


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, Optional[int]]:
    """A row is uniquely identified by the group, environment, and seed cell."""
    fixed_seed = _as_exact_int(row.get("fixed_seed_override"))
    seed = fixed_seed if fixed_seed is not None else _as_exact_int(row.get("seed_idx"))
    return (
        str(row.get("group", "") or ""),
        str(row.get("env", "") or ""),
        seed,
    )


def _merge_group_run_rows(
    existing_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]]
) -> list[Dict[str, Any]]:
    """Replace an old row by a freshly completed row for the same seed cell."""
    merged: Dict[tuple[str, str, Optional[int]], Dict[str, Any]] = {}
    for row in list(existing_rows) + list(new_rows):
        payload = dict(row)
        merged[_row_key(payload)] = payload
    return sorted(
        merged.values(),
        key=lambda row: (
            str(row.get("group", "") or ""),
            str(row.get("env", "") or ""),
            _as_exact_int(row.get("seed_idx"))
            if _as_exact_int(row.get("seed_idx")) is not None
            else -1,
        ),
    )


def _filter_active_cohort_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_name: str,
    envs: Sequence[str],
    seed_labels: Sequence[int],
) -> list[Dict[str, Any]]:
    active_envs = {str(env) for env in envs}
    active_seeds = {int(seed) for seed in seed_labels}
    return [
        dict(row)
        for row in rows
        if str(row.get("group", "") or "") == str(group_name)
        and str(row.get("env", "") or "") in active_envs
        and _row_key(row)[2] in active_seeds
    ]


def _read_mapping(path: Path) -> Optional[Dict[str, Any]]:
    def _reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON constant is not permitted: {token}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _sha256_file(path: Path) -> Optional[str]:
    """Hash one artifact without allowing a missing path to look complete."""
    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, UnicodeError):
        return None


def _identity_signature(identity: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        field: identity.get(field)
        for field in ("protocol_id", "protocol_hash", "config_hash", "source_hash")
    }


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in (
        "protocol_id", "protocol_hash", "config_hash", "source_hash"
    ))


CELL_COMPLETION_SCHEMA = "rgd_cell_completion_manifest_v1"


def _relative_cell_path(root: Path, path: Path) -> str:
    """Return a portable path and reject references outside the cell root."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"cell artifact escapes result directory: {path}") from exc
    return relative.as_posix()


def _artifact_entry(root: Path, path: Path) -> Dict[str, Any]:
    digest = _sha256_file(path)
    if digest is None:
        raise RuntimeError(f"missing or unreadable cell artifact: {path}")
    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        raise RuntimeError(f"cannot stat cell artifact: {path}") from exc
    return {
        "path": _relative_cell_path(root, path),
        "size_bytes": size,
        "sha256": digest,
    }


def _validate_artifact_entries(root: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    for entry in entries:
        relative = str(entry.get("path", "") or "")
        if not relative or relative in seen:
            raise RuntimeError("cell completion manifest has duplicate/empty artifact path")
        seen.add(relative)
        path = (root / relative).resolve()
        if _relative_cell_path(root, path) != relative.replace("\\", "/"):
            raise RuntimeError("cell completion manifest contains a non-canonical path")
        actual = _artifact_entry(root, path)
        if actual != {
            "path": relative.replace("\\", "/"),
            "size_bytes": entry.get("size_bytes"),
            "sha256": entry.get("sha256"),
        }:
            raise RuntimeError(f"cell artifact hash/size drift: {relative}")


def _episode_identity(
    seed_label: int, episodes: int, offset: int, env_type: str
) -> tuple[int, str]:
    episode_id = (
        int(seed_label)
        if int(episodes) == 1
        else int(seed_label) * int(episodes) + int(offset)
    )
    return episode_id, f"{infer_env_label(env_type)}_{episode_id}"


def _validate_release_references(
    *,
    root: Path,
    event_payload: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Validate the event-to-release-snapshot file references and hashes."""
    release_events = [
        event for event in events
        if bool(event.get("closed_loop_latency_release_event", False))
    ]
    count = _as_exact_int(event_payload.get("release_snapshot_count"))
    if count is None or count < 0 or count > len(release_events):
        raise RuntimeError("release snapshot count exceeds release events")
    bundle_ref = event_payload.get("release_snapshot_bundle")
    manifest_ref = event_payload.get("release_snapshot_manifest")
    digest = event_payload.get("release_snapshot_bundle_sha256")
    if count == 0:
        if bundle_ref not in (None, "") or manifest_ref not in (None, "") or digest not in (None, ""):
            raise RuntimeError("empty release snapshot set has non-empty references")
        return []
    if not all(isinstance(value, str) and value.strip() for value in (bundle_ref, manifest_ref, digest)):
        raise RuntimeError("release snapshot references are incomplete")
    bundle_path = (root / str(bundle_ref)).resolve()
    manifest_path = (root / str(manifest_ref)).resolve()
    if _relative_cell_path(root, bundle_path) != str(bundle_ref).replace("\\", "/"):
        raise RuntimeError("release snapshot bundle escapes cell root")
    if _relative_cell_path(root, manifest_path) != str(manifest_ref).replace("\\", "/"):
        raise RuntimeError("release snapshot manifest escapes cell root")
    if _sha256_file(bundle_path) != str(digest):
        raise RuntimeError("release snapshot bundle hash drift")
    snapshot_manifest = _read_mapping(manifest_path)
    if snapshot_manifest is None:
        raise RuntimeError("release snapshot manifest is missing or invalid")
    if _as_exact_int(snapshot_manifest.get("snapshot_count")) != count:
        raise RuntimeError("release snapshot manifest count drift")
    if str(snapshot_manifest.get("bundle_sha256", "") or "") != str(digest):
        raise RuntimeError("release snapshot manifest bundle hash drift")
    if str(snapshot_manifest.get("bundle_file", "") or "") != bundle_path.name:
        raise RuntimeError("release snapshot manifest bundle filename drift")
    snapshot_rows = snapshot_manifest.get("snapshots")
    if not isinstance(snapshot_rows, list) or len(snapshot_rows) != count:
        raise RuntimeError("release snapshot manifest row count drift")
    release_ids = {
        str(event.get("closed_loop_latency_terminal_request_id", "") or "")
        for event in release_events
    }
    snapshot_ids = {
        str(row.get("request_id", "") or "")
        for row in snapshot_rows
        if isinstance(row, Mapping)
    }
    if "" in release_ids or not snapshot_ids <= release_ids:
        raise RuntimeError("release snapshot request-ID coverage drift")
    return [bundle_path, manifest_path]


def _collect_cell_artifact_closure(
    root: Path,
    *,
    group_name: str,
    env_type: str,
    seed_label: int,
    episodes: int,
    expected_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate and inventory every required artifact in one completed cell."""
    root = Path(root).resolve()
    runtime_path = root / "runtime_manifest.json"
    snapshot_path = root / "experiment_snapshot.json"
    metrics_path = root / f"{group_name}_rgd_metrics.json"
    runtime = _read_mapping(runtime_path)
    snapshot = _read_mapping(snapshot_path)
    metrics_payload = _read_mapping(metrics_path)
    if runtime is None or snapshot is None or metrics_payload is None:
        raise RuntimeError("cell artifact closure is missing a required manifest or metrics file")
    if not _identity_matches(runtime, expected_identity, int(seed_label)):
        raise RuntimeError("runtime manifest identity drift")
    if not _identity_matches(snapshot, expected_identity, int(seed_label)):
        raise RuntimeError("experiment snapshot identity drift")
    if _as_exact_int(snapshot.get("seed_start")) != int(seed_label):
        raise RuntimeError("experiment snapshot seed_start drift")
    if list(snapshot.get("seeds_used", []) or []) != [int(seed_label)]:
        raise RuntimeError("experiment snapshot seeds_used drift")
    metrics = metrics_payload.get("comprehensive_metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError("cell metrics payload is missing comprehensive_metrics")

    artifacts: list[Dict[str, Any]] = [
        _artifact_entry(root, runtime_path),
        _artifact_entry(root, snapshot_path),
        _artifact_entry(root, metrics_path),
    ]
    episode_rows: list[Dict[str, Any]] = []
    total_frames = 0
    for offset in range(int(episodes)):
        episode_id, prefix = _episode_identity(seed_label, episodes, offset, env_type)
        ep_dir = root / f"ep_{episode_id}"
        event_path = root / "event_logs" / f"event_log_{prefix}_{episode_id}.json"
        reasoning_path = ep_dir / f"{prefix}_reasoning_records.json"
        physical_path = ep_dir / f"{prefix}_physical_frames.json"
        result_path = root / f"episode_result_{prefix}_{episode_id}.json"
        event_payload = _read_mapping(event_path)
        reasoning_payload = _read_mapping(reasoning_path)
        physical_payload = _read_mapping(physical_path)
        result_payload = _read_mapping(result_path)
        if any(payload is None for payload in (event_payload, reasoning_payload, physical_payload, result_payload)):
            raise RuntimeError(f"episode {episode_id} is missing a canonical artifact")
        if (
            _as_exact_int(event_payload.get("episode_id")) != episode_id
            or str(event_payload.get("prefix", "") or "") != prefix
            or _as_exact_int(reasoning_payload.get("episode_id")) != episode_id
            or _as_exact_int(physical_payload.get("episode_id")) != episode_id
            or _as_exact_int(result_payload.get("episode_id")) != episode_id
            or str(result_payload.get("prefix", "") or "") != prefix
        ):
            raise RuntimeError(f"episode {episode_id} artifact identity drift")
        events = list(event_payload.get("events", []) or [])
        records = list(reasoning_payload.get("analysis_records", []) or [])
        frames = list(physical_payload.get("frames", []) or [])
        event_count = _as_exact_int(event_payload.get("event_count"))
        reasoning_count = _as_exact_int(reasoning_payload.get("record_count"))
        physical_count = _as_exact_int(physical_payload.get("frame_count"))
        result_count = _as_exact_int(result_payload.get("frame_count"))
        if event_count != len(events) or reasoning_count != len(records) or physical_count != len(frames):
            raise RuntimeError(f"episode {episode_id} trace count closure failed")
        if event_count != physical_count or reasoning_count != physical_count or result_count != physical_count:
            raise RuntimeError(f"episode {episode_id} cross-trace frame counts differ")
        if not events or not frames:
            raise RuntimeError(f"episode {episode_id} has no frames")
        last_event = events[-1]
        last_frame = frames[-1]
        if not bool(last_event.get("done", last_event.get("episode_done", False))):
            raise RuntimeError(f"episode {episode_id} has no terminal event")
        if not bool(last_frame.get("done", False)):
            raise RuntimeError(f"episode {episode_id} physical trace has no terminal frame")
        if str(event_payload.get("terminal_cause", "") or "") != str(
            last_event.get("terminal_cause", "") or ""
        ):
            raise RuntimeError(f"episode {episode_id} terminal cause drift")
        physical_metrics = physical_payload.get("metrics")
        if not isinstance(physical_metrics, Mapping) or _as_exact_int(
            physical_metrics.get("total_frames")
        ) != physical_count:
            raise RuntimeError(f"episode {episode_id} physical metrics count drift")
        event_log_ref = str(result_payload.get("event_log", "") or "")
        if not event_log_ref:
            raise RuntimeError(f"episode {episode_id} result has no event-log reference")
        referenced_event = Path(event_log_ref)
        if referenced_event.is_absolute():
            event_candidates = {referenced_event.resolve()}
        else:
            event_candidates = {
                (root / referenced_event).resolve(),
                (REPO_ROOT / referenced_event).resolve(),
            }
        if event_path.resolve() not in event_candidates:
            raise RuntimeError(f"episode {episode_id} result event-log reference drift")
        if _as_exact_int(event_payload.get("pending_release_count")) != len(
            list(event_payload.get("pending_releases_dropped_at_episode_end", []) or [])
        ):
            raise RuntimeError(f"episode {episode_id} pending-release count drift")
        release_paths = _validate_release_references(
            root=root, event_payload=event_payload, events=events
        )
        for path in (event_path, reasoning_path, physical_path, result_path, *release_paths):
            artifacts.append(_artifact_entry(root, path))
        total_frames += physical_count
        episode_rows.append(
            {
                "episode_id": int(episode_id),
                "prefix": str(prefix),
                "event_path": _relative_cell_path(root, event_path),
                "reasoning_path": _relative_cell_path(root, reasoning_path),
                "physical_path": _relative_cell_path(root, physical_path),
                "result_path": _relative_cell_path(root, result_path),
                "event_count": int(event_count),
                "reasoning_count": int(reasoning_count),
                "physical_frame_count": int(physical_count),
                "pending_release_count": len(
                    list(event_payload.get("pending_releases_dropped_at_episode_end", []) or [])
                ),
            }
        )
    metric_frames = _as_exact_int(metrics.get("total_frames"))
    if metric_frames != total_frames:
        raise RuntimeError("cell metrics total_frames does not match trace closure")
    artifacts.sort(key=lambda entry: str(entry["path"]))
    return {
        "episodes": episode_rows,
        "artifacts": artifacts,
        "total_frames": int(total_frames),
    }


def _write_cell_completion_manifest(
    root: Path,
    *,
    group_name: str,
    env_type: str,
    seed_label: int,
    episodes: int,
    expected_identity: Mapping[str, Any],
    runtime_integrity_checks: Sequence[bool],
    ending_identity: Optional[Mapping[str, Any]] = None,
    runtime_integrity_records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Path:
    closure = _collect_cell_artifact_closure(
        root,
        group_name=group_name,
        env_type=env_type,
        seed_label=seed_label,
        episodes=episodes,
        expected_identity=expected_identity,
    )
    checks = [bool(value) for value in runtime_integrity_checks]
    if len(checks) != int(episodes) or not checks or not all(checks):
        raise RuntimeError("cell runtime integrity was not clean")
    identity_end = dict(ending_identity or expected_identity)
    if not _same_identity(expected_identity, identity_end):
        raise RuntimeError("cell start/end experiment identities differ")
    integrity_records = [dict(row) for row in list(runtime_integrity_records or [])]
    if len(integrity_records) != len(checks):
        raise RuntimeError("runtime identity record/check counts differ")
    for record in integrity_records:
        if (
            record.get("clean") is not True
            or record.get("identity_start") != record.get("identity_end")
        ):
            raise RuntimeError("runtime identity record contains drift")
    marker = {
        "schema_version": CELL_COMPLETION_SCHEMA,
        "group": str(group_name),
        "env": str(env_type),
        "seed": int(seed_label),
        "episodes": int(episodes),
        "total_frames": int(closure["total_frames"]),
        "identity": _identity_signature(expected_identity),
        "identity_start": _identity_signature(expected_identity),
        "identity_end": _identity_signature(identity_end),
        "identity_clean": True,
        "runtime_integrity_clean": bool(all(checks)),
        "runtime_integrity_checks": len(checks),
        "runtime_integrity_violation_count": int(sum(not check for check in checks)),
        "runtime_identity_records": integrity_records,
        "episodes_manifest": closure["episodes"],
        "artifacts": closure["artifacts"],
    }
    path = Path(root) / "cell_completion_manifest.json"
    dump_json(path, marker)
    return path


def _validate_cell_completion_manifest(
    root: Path,
    *,
    group_name: str,
    env_type: str,
    seed_label: int,
    episodes: int,
    expected_identity: Mapping[str, Any],
) -> bool:
    marker = _read_mapping(Path(root) / "cell_completion_manifest.json")
    if marker is None or marker.get("schema_version") != CELL_COMPLETION_SCHEMA:
        return False
    if (
        str(marker.get("group", "") or "") != str(group_name)
        or str(marker.get("env", "") or "") != str(env_type)
        or _as_exact_int(marker.get("seed")) != int(seed_label)
        or _as_exact_int(marker.get("episodes")) != int(episodes)
        or marker.get("identity") != _identity_signature(expected_identity)
        or marker.get("identity_start") != _identity_signature(expected_identity)
        or marker.get("identity_end") != _identity_signature(expected_identity)
        or marker.get("identity_clean") is not True
        or marker.get("runtime_integrity_clean") is not True
        or _as_exact_int(marker.get("runtime_integrity_violation_count")) != 0
    ):
        return False
    integrity_records = marker.get("runtime_identity_records")
    if not isinstance(integrity_records, list):
        return False
    check_count = _as_exact_int(marker.get("runtime_integrity_checks"))
    if check_count != int(episodes) or len(integrity_records) != check_count:
        return False
    if any(
        not isinstance(record, Mapping)
        or record.get("clean") is not True
        or record.get("identity_start") != record.get("identity_end")
        for record in integrity_records
    ):
        return False
    try:
        closure = _collect_cell_artifact_closure(
            Path(root),
            group_name=group_name,
            env_type=env_type,
            seed_label=seed_label,
            episodes=episodes,
            expected_identity=expected_identity,
        )
    except Exception:
        return False
    if _as_exact_int(marker.get("total_frames")) != closure["total_frames"]:
        return False
    if marker.get("episodes_manifest") != closure["episodes"]:
        return False
    recorded = marker.get("artifacts")
    if not isinstance(recorded, list):
        return False
    try:
        _validate_artifact_entries(Path(root), recorded)
    except Exception:
        return False
    return sorted(recorded, key=lambda entry: str(entry.get("path", ""))) == closure["artifacts"]


def _completed_run_row(
    *,
    metrics: Mapping[str, Any],
    identity: Mapping[str, Any],
    group_name: str,
    group_id: str,
    env_type: str,
    seed_label: int,
    episodes: int,
    result_dir: Path,
) -> Dict[str, Any]:
    return {
        "group": str(group_name),
        "group_id": str(group_id),
        "env": str(env_type),
        "seed_idx": int(seed_label),
        "fixed_seed_override": int(seed_label),
        "seed_start": int(seed_label),
        "episodes_run": int(episodes),
        "total_frames": int(metrics["total_frames"]),
        "result_dir": str(Path(result_dir).resolve()),
        **_identity_signature(identity),
        "evaluation_runtime_stable": bool(metrics["evaluation_runtime_stable"]),
        "runtime_integrity_clean": bool(metrics["runtime_integrity_clean"]),
        "runtime_integrity_violation_rate": metrics.get(
            "runtime_integrity_violation_rate"
        ),
        "slow_call_rate": float(metrics["slow_call_rate"]),
        "slow_call_success_rate": float(metrics["slow_call_success_rate"]),
        **{
            field: metrics.get(field)
            for field in (
                "request_count",
                "request_issued_count",
                "request_terminal_count",
                "valid_response_count",
                "timeout_count",
                "failure_count",
                "dropped_at_episode_end_count",
                "request_pending_count",
                "request_lifecycle_closed",
                "terminal_wall_latency_count",
                "terminal_wall_latency_mean_s",
                "simulator_e2e_latency_count",
                "simulator_e2e_latency_mean_s",
            )
        },
        **{field: float(metrics[field]) for field in COMPARISON_HEADLINE_FIELDS},
    }


def _recover_completed_cell_row(
    root: Path,
    *,
    group_name: str,
    env_type: str,
    seed_label: int,
    episodes: int,
    expected_identity: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Rebuild a shared CSV row after a worker completed before main merged it."""
    if not _validate_cell_completion_manifest(
        root,
        group_name=group_name,
        env_type=env_type,
        seed_label=seed_label,
        episodes=episodes,
        expected_identity=expected_identity,
    ):
        return None
    metrics_payload = _read_mapping(Path(root) / f"{group_name}_rgd_metrics.json")
    if metrics_payload is None or not isinstance(
        metrics_payload.get("comprehensive_metrics"), Mapping
    ):
        return None
    metrics = dict(metrics_payload["comprehensive_metrics"])
    try:
        config = dict(expected_identity.get("config", {}) or {})
        return _completed_run_row(
            metrics=metrics,
            identity=expected_identity,
            group_name=group_name,
            group_id=str(config.get("group_id", group_name)),
            env_type=env_type,
            seed_label=seed_label,
            episodes=episodes,
            result_dir=Path(root),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _identity_matches(
    payload: Mapping[str, Any], expected_identity: Mapping[str, Any], seed_label: int
) -> bool:
    for field in ("protocol_id", "protocol_hash", "config_hash", "source_hash"):
        if payload.get(field) != expected_identity.get(field):
            return False
    if _as_exact_int(payload.get("fixed_seed_override")) != int(seed_label):
        return False
    resolved_seeds = payload.get("resolved_seeds")
    if resolved_seeds is not None and list(resolved_seeds) != [int(seed_label)]:
        return False
    expected_config = expected_identity.get("config")
    if isinstance(expected_config, Mapping) and payload.get("config") != expected_config:
        return False
    return True


def _resume_row_is_complete(
    row: Mapping[str, Any],
    result_dir: Path | str,
    *,
    group_name: str,
    env_type: str,
    seed_label: int,
    episodes: int,
    expected_identity: Mapping[str, Any],
) -> bool:
    """Return true only for a row and manifests from the same complete cell."""
    if str(row.get("group", "") or "") != str(group_name):
        return False
    if str(row.get("env", "") or "") != str(env_type):
        return False
    if _as_exact_int(row.get("seed_idx")) != int(seed_label):
        return False
    if _as_exact_int(row.get("fixed_seed_override")) != int(seed_label):
        return False
    if _as_exact_int(row.get("episodes_run")) != int(episodes):
        return False
    for field in ("protocol_id", "protocol_hash", "config_hash", "source_hash"):
        if row.get(field) != expected_identity.get(field):
            return False

    try:
        return _validate_cell_completion_manifest(
            Path(result_dir),
            group_name=group_name,
            env_type=env_type,
            seed_label=int(seed_label),
            episodes=int(episodes),
            expected_identity=expected_identity,
        )
    except Exception:
        # Resume is deliberately fail-closed for malformed, legacy, or
        # partially-written cells.  The caller may archive and regenerate it.
        return False


def _v12_required_groups(protocol: Mapping[str, Any]) -> tuple[str, ...]:
    artifacts = dict(
        dict(protocol.get("tvt_submission_contract", {}) or {})
        .get("evidence_artifacts", {})
        or {}
    )
    main = dict(artifacts.get("artifacts", {}) or {}).get("main_results", {})
    declared = tuple(dict(main or {}).get("required_groups", []) or [])
    return tuple(str(name) for name in declared)


def _validate_v12_main_preflight(
    *,
    protocol: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    partition: str,
    selected_groups: Sequence[str],
    envs: Sequence[str],
    seed_labels: Sequence[int],
    episodes: int,
) -> None:
    """Reject any deviation from the frozen six-arm v12 paper matrix."""
    del base_cfg
    if str(partition) != "main":
        raise ValueError("v12 main preflight requires partition='main'")
    declared = _v12_required_groups(protocol)
    if declared != V12_MAIN_GROUPS or tuple(selected_groups) != V12_MAIN_GROUPS:
        raise ValueError("v12 main preflight requires the complete ordered group contract")
    if tuple(str(env) for env in envs) != (V12_MAIN_ENV,):
        raise ValueError("v12 main preflight requires highway-v0 only")
    if tuple(int(seed) for seed in seed_labels) != V12_MAIN_SEEDS:
        raise ValueError("v12 main preflight requires the frozen 4000-4029 seed block")
    if int(episodes) != 1:
        raise ValueError("v12 main preflight requires one episode per seed cell")


def _declared_seed_block(protocol: Mapping[str, Any], field: str) -> tuple[int, ...]:
    contract = dict(protocol.get("tvt_submission_contract", {}) or {})
    value = contract.get(field)
    if isinstance(value, Mapping):
        start = _as_exact_int(value.get("start"))
        end = _as_exact_int(value.get("end"))
        count = _as_exact_int(value.get("count"))
        if start is None or end is None or end < start:
            raise ValueError(f"invalid v13 seed contract: {field}")
        seeds = tuple(range(start, end + 1))
        if count is not None and count != len(seeds):
            raise ValueError(f"inconsistent v13 seed count: {field}")
        return seeds
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seeds = tuple(_as_exact_int(item) for item in value)
        if not seeds or any(item is None or item < 0 for item in seeds):
            raise ValueError(f"invalid v13 seed contract: {field}")
        return tuple(int(item) for item in seeds)
    text = str(value or "").strip()
    if "-" in text:
        left, right = text.split("-", 1)
        start = _as_exact_int(left)
        end = _as_exact_int(right)
        if start is not None and end is not None and 0 <= start <= end:
            return tuple(range(start, end + 1))
    raise ValueError(f"invalid v13 seed contract: {field}")


def _validate_v13_preflight(
    *,
    protocol: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    partition: str,
    selected_groups: Sequence[str],
    envs: Sequence[str],
    seed_labels: Sequence[int],
    episodes: int,
    simulation_duration: Optional[int],
) -> None:
    """Fail closed before a v13 bundle can be labeled main or mechanism evidence."""
    contract = dict(protocol.get("tvt_submission_contract", {}) or {})
    if partition == "main":
        expected_seeds = _declared_seed_block(protocol, "main_seeds")
        artifacts = dict(dict(contract.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {})
        expected_groups = tuple(
            str(name)
            for name in dict(artifacts.get("main_results", {}) or {}).get(
                "required_groups", ()
            )
        )
    elif partition == "mechanism":
        expected_seeds = _declared_seed_block(protocol, "mechanism_evaluation_seeds")
        declared_groups = contract.get("mechanism_source_groups")
        if not isinstance(declared_groups, Sequence) or isinstance(
            declared_groups, (str, bytes)
        ):
            raise ValueError("v13 mechanism source groups must be declared explicitly")
        expected_groups = tuple(str(name) for name in declared_groups)
    else:
        raise ValueError("v13 formal preflight supports partition='main' or 'mechanism'")
    if tuple(selected_groups) != expected_groups:
        raise ValueError(
            f"v13 {partition} preflight requires ordered groups {expected_groups}"
        )
    if tuple(str(env) for env in envs) != ("highway-v0",):
        raise ValueError(f"v13 {partition} preflight requires highway-v0 only")
    if tuple(int(seed) for seed in seed_labels) != expected_seeds:
        raise ValueError(f"v13 {partition} preflight seed cohort drift")
    if int(episodes) != 1:
        raise ValueError(f"v13 {partition} preflight requires one episode per seed")
    formal_horizon = resolve_policy_execution_horizon(
        dict(contract.get("formal_execution_contract", {}) or {}),
        context="v13 formal execution contract",
    )
    if simulation_duration is not None and float(simulation_duration) != float(
        formal_horizon.episode_duration_s
    ):
        raise ValueError(
            f"v13 {partition} preflight requires "
            f"{formal_horizon.episode_duration_s:g} seconds"
        )
    if str(base_cfg.get("LLM_PROVIDER", "") or "").strip().lower() != "siliconflow":
        raise ValueError("v13 formal preflight requires the SiliconFlow backend")
    if str(base_cfg.get("SILICONFLOW_CHAT_MODEL", "") or "") != "Qwen/Qwen3-8B":
        raise ValueError("v13 formal preflight requires Qwen/Qwen3-8B")
    asynchronous = dict(base_cfg.get("asynchronous_slow_path", {}) or {})
    replay = dict(base_cfg.get("closed_loop_latency_replay", {}) or {})
    if not bool(asynchronous.get("enable", False)) or bool(replay.get("enable", False)):
        raise ValueError(
            "v13 live source requires native async execution and disables scripted replay"
        )
    if float(base_cfg.get("rgd_predicted_slow_latency_s", -1.0)) != 1.7:
        raise ValueError("v13 live source requires the locked 1.7-s prediction")


def _v13_partition_execution_contract(
    protocol: Mapping[str, Any], partition: str
) -> Mapping[str, Any]:
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    if partition == "mechanism":
        contract = submission.get("mechanism_source_execution_contract")
    elif partition == "main":
        artifacts = dict(
            dict(submission.get("evidence_artifacts", {}) or {}).get("artifacts", {})
            or {}
        )
        contract = dict(artifacts.get("main_results", {}) or {}).get(
            "execution_contract"
        )
    else:
        raise ValueError(f"unsupported v13 formal partition: {partition}")
    if not isinstance(contract, Mapping):
        raise ValueError(f"v13 {partition} execution contract is missing")
    return contract


def _validate_v13_resolved_cell(
    protocol: Mapping[str, Any], partition: str, cfg: Mapping[str, Any]
):
    context = (
        f"v13 {partition} cell "
        f"{cfg.get('group_name', '')}/{cfg.get('env_type', '')}"
    )
    horizon = validate_policy_execution_horizon(
        cfg,
        _v13_partition_execution_contract(protocol, partition),
        context=context,
    )
    if str(cfg.get("LLM_PROVIDER", "") or "").strip().lower() != "siliconflow":
        raise ValueError(f"{context}: provider drift")
    if str(cfg.get("SILICONFLOW_CHAT_MODEL", "") or "") != "Qwen/Qwen3-8B":
        raise ValueError(f"{context}: slow model drift")
    asynchronous = dict(cfg.get("asynchronous_slow_path", {}) or {})
    replay = dict(cfg.get("closed_loop_latency_replay", {}) or {})
    if not bool(asynchronous.get("enable", False)) or bool(replay.get("enable", False)):
        raise ValueError(f"{context}: live latency-engine drift")
    if float(cfg.get("rgd_predicted_slow_latency_s", -1.0)) != 1.7:
        raise ValueError(f"{context}: predicted latency drift")
    return horizon


def _config_with_runtime_defaults(cfg: Mapping[str, Any], *, v12: bool) -> Dict[str, Any]:
    result = copy.deepcopy(dict(cfg))
    result.update(
        {
            "enable_physical_metrics": True,
            "enable_reasoning_recording": True,
            "event_log_schema_version": "rgd_event_log_v2" if v12 else "rgd_event_log_v3",
        }
    )
    _sync_protocol_runtime_snapshot(result)
    return result


def _identity_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Exclude frame-local state that the runtime intentionally updates in-place."""
    result = copy.deepcopy(dict(cfg))
    result.pop("_current_runtime_state", None)
    return result


def _record_to_dict(record: Any) -> Dict[str, Any]:
    if hasattr(record, "to_dict") and callable(record.to_dict):
        payload = record.to_dict()
    elif is_dataclass(record):
        payload = asdict(record)
    elif isinstance(record, Mapping):
        payload = dict(record)
    else:
        raise TypeError(f"unsupported reasoning record: {type(record)!r}")
    return dict(payload)


def _write_episode_trace_counts(
    *,
    result_dir: Path,
    ep_dir: Path,
    prefix: str,
    episode_id: int,
    events: Sequence[Mapping[str, Any]],
    reasoning_records: Sequence[Mapping[str, Any]],
    event_schema: str,
) -> Dict[str, Any]:
    """Add count closures required for independently auditable per-cell traces."""
    event_path = result_dir / "event_logs" / f"event_log_{prefix}_{episode_id}.json"
    event_payload = _read_mapping(event_path)
    if event_payload is None:
        raise RuntimeError(f"episode event log was not written: {event_path}")
    terminal = str(events[-1].get("terminal_cause", "") if events else "")
    event_payload.update(
        {
            "schema_version": str(event_schema),
            "event_count": len(events),
            "pending_release_count": len(
                list(event_payload.get("pending_releases_dropped_at_episode_end", []) or [])
            ),
            "terminal_cause": terminal,
            "events": [dict(event) for event in events],
        }
    )
    for event in event_payload["events"]:
        event["episode_done"] = bool(event.get("done", event.get("episode_done", False)))
    dump_json(event_path, event_payload)

    reasoning_path = ep_dir / f"{prefix}_reasoning_records.json"
    dump_json(
        reasoning_path,
        {
            "episode_id": int(episode_id),
            "record_count": len(reasoning_records),
            "analysis_records": [dict(record) for record in reasoning_records],
        },
    )
    physical_path = ep_dir / f"physical_frames_{episode_id}.json"
    physical_payload = _read_mapping(physical_path)
    if physical_payload is None:
        raise RuntimeError(f"physical trace was not written: {physical_path}")
    frames = list(physical_payload.get("frames", []) or [])
    physical_payload["frame_count"] = len(frames)
    canonical_physical_path = ep_dir / f"{prefix}_physical_frames.json"
    dump_json(canonical_physical_path, physical_payload)
    if canonical_physical_path != physical_path:
        physical_path.unlink(missing_ok=True)
    return event_payload


def _finite_nonnegative(value: Any) -> Optional[float]:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) and resolved >= 0.0 else None


def _completed_metrics(
    aggregate: MetricsAggregator,
    *,
    group_name: str,
    runtime_seconds: Sequence[float],
    events: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    dropped_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    runtime_integrity_checks: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    metrics = aggregate.calculate_comprehensive_metrics()
    frame_count = int(metrics.get("total_frames", 0) or 0)
    if frame_count <= 0:
        raise RuntimeError(f"{group_name}: completed setting has no physical frames")
    if len(events) != frame_count:
        raise RuntimeError(
            f"{group_name}: event/physical frame mismatch ({len(events)} != {frame_count})"
        )
    issued: Dict[str, str] = {}
    terminal: Dict[str, str] = {}
    dropped: Dict[str, Mapping[str, Any]] = {}
    for event in events:
        if bool(event.get("closed_loop_latency_issuance_event", False)):
            request_id = str(
                event.get("closed_loop_latency_issued_request_id", "") or ""
            )
            if not request_id or request_id in issued:
                raise RuntimeError(
                    f"{group_name}: malformed or duplicate slow-request issuance"
                )
            issued[request_id] = str(
                event.get("closed_loop_latency_issued_response_outcome", "pending")
                or "pending"
            )
        if bool(event.get("closed_loop_latency_terminal_event", False)):
            request_id = str(
                event.get("closed_loop_latency_terminal_request_id", "") or ""
            )
            outcome = str(
                event.get("closed_loop_latency_terminal_response_outcome", "") or ""
            )
            if not request_id or request_id in terminal or request_id not in issued:
                raise RuntimeError(
                    f"{group_name}: malformed, duplicate, or orphan slow-request terminal"
                )
            if outcome not in {"valid", "timeout", "failure"}:
                raise RuntimeError(f"{group_name}: invalid slow-request terminal outcome")
            terminal[request_id] = outcome
    for row in list(dropped_rows or []):
        dropped_row = dict(row)
        request_id = str(dropped_row.get("request_id", "") or "").strip()
        if not request_id or request_id in dropped or request_id not in issued:
            raise RuntimeError(f"{group_name}: malformed, duplicate, or orphan dropped request")
        if request_id in terminal:
            raise RuntimeError(f"{group_name}: request is both terminal and dropped")
        if str(dropped_row.get("terminal_outcome", "") or "") != "dropped_at_episode_end":
            raise RuntimeError(f"{group_name}: dropped request has invalid terminal outcome")
        dropped[request_id] = row
    if issued or terminal or dropped:
        attempts = len(issued)
        successes = sum(outcome == "valid" for outcome in terminal.values())
        timeout_count = sum(outcome == "timeout" for outcome in terminal.values())
        failure_count = sum(outcome == "failure" for outcome in terminal.values())
        failures = timeout_count + failure_count
        pending_ids = set(issued) - set(terminal) - set(dropped)
        pending = len(pending_ids)
        if pending:
            raise RuntimeError(
                f"{group_name}: completed cell retains {pending} unterminated slow requests"
            )
        lifecycle_closed = set(issued) == set(terminal) | set(dropped)
    else:
        attempts = sum(bool(event.get("slow_request_attempted", False)) for event in events)
        successes = sum(
            bool(event.get("slow_request_attempted", False))
            and bool(event.get("slow_request_valid_return", False))
            and bool(event.get("slow_reasoning_success", False))
            and not bool(event.get("slow_request_failed", False))
            for event in events
        )
        timeout_count = sum(
            bool(event.get("closed_loop_latency_timeout_event", False))
            for event in events
        )
        explicit_failure_count = sum(
            bool(event.get("closed_loop_latency_failure_event", False))
            for event in events
        )
        failure_count = (
            explicit_failure_count
            if timeout_count or explicit_failure_count
            else sum(bool(event.get("slow_request_failed", False)) for event in events)
        )
        failures = timeout_count + failure_count
        pending = 0
        lifecycle_closed = True
    artifacts = dict(
        dict(
            dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                "evidence_artifacts", {}
            )
            or {}
        ).get("artifacts", {})
        or {}
    )
    failure_policy = str(
        dict(artifacts.get("main_results", {}) or {}).get(
            "technical_slow_failure_policy", ""
        )
        or ""
    )
    if failures and failure_policy == "reject_any_scheduled_slow_failure_or_fallback":
        raise RuntimeError(
            f"{group_name}: slow-executor failures cannot be reported as completed main-table results"
        )
    wall_latencies = [
        latency
        for event in events
        if bool(event.get("closed_loop_latency_terminal_event", False))
        for latency in [_finite_nonnegative(event.get("slow_response_wall_latency_s"))]
        if latency is not None
    ]
    simulator_e2e_latencies = [
        latency
        for event in events
        if bool(event.get("closed_loop_latency_terminal_event", False))
        and str(event.get("closed_loop_latency_realized_source", "") or "")
        == "simulator_frame_delta"
        for latency in [_finite_nonnegative(event.get("closed_loop_latency_realized_seconds"))]
        if latency is not None
    ]
    integrity_checks = [bool(value) for value in list(runtime_integrity_checks or [])]
    integrity_clean = bool(integrity_checks) and all(integrity_checks)
    integrity_violation_rate = (
        float(sum(not value for value in integrity_checks) / len(integrity_checks))
        if integrity_checks
        else None
    )
    guardrails = dict(protocol.get("claim_guardrails", {}) or {})
    metrics.update(
        {
            "evaluation_protocol_name": str(group_name),
            "single_core_method_name": str(
                guardrails.get("single_core_method_name", "Recoverability-Gated Deliberation")
            ),
            "primary_evaluation_subject": str(
                guardrails.get("primary_evaluation_subject", "fixed-policy RGD")
            ),
            "evaluation_runtime_stable": integrity_clean,
            "runtime_integrity_clean": integrity_clean,
            "runtime_integrity_violation_rate": integrity_violation_rate,
            "slow_call_rate": float(attempts / frame_count),
            "slow_call_success_rate": float(successes / attempts) if attempts else 0.0,
            "slow_call_terminal_count": int(len(terminal) if issued else successes + failures),
            "slow_call_pending_at_episode_end": int(pending),
            "slow_request_lifecycle_request_scoped": bool(issued or not attempts),
            "slow_attempts": int(attempts),
            "slow_attempt_successes": int(successes),
            "slow_attempt_failures": int(failures),
            "slow_attempt_terminal_outcomes": int(len(terminal)),
            "slow_attempt_pending": int(pending),
            "request_count": int(attempts),
            "request_issued_count": int(attempts),
            "request_terminal_count": int(len(terminal)),
            "valid_response_count": int(successes),
            "timeout_count": int(timeout_count),
            "failure_count": int(failure_count),
            "dropped_at_episode_end_count": int(len(dropped)),
            "request_pending_count": int(pending),
            "request_lifecycle_closed": bool(lifecycle_closed and pending == 0),
            "terminal_wall_latency_count": int(len(wall_latencies)),
            "terminal_wall_latency_mean_s": (
                float(sum(wall_latencies) / len(wall_latencies))
                if wall_latencies
                else None
            ),
            "simulator_e2e_latency_count": int(len(simulator_e2e_latencies)),
            "simulator_e2e_latency_mean_s": (
                float(sum(simulator_e2e_latencies) / len(simulator_e2e_latencies))
                if simulator_e2e_latencies
                else None
            ),
            "simulator_e2e_latency_source": "simulator_frame_delta" if simulator_e2e_latencies else None,
            "avg_runtime_per_frame": (
                float(sum(float(value) for value in runtime_seconds) / len(runtime_seconds))
                if runtime_seconds
                else 0.0
            ),
        }
    )
    return metrics


def _run_setting(
    *,
    cfg: Mapping[str, Any],
    protocol: Mapping[str, Any],
    group_name: str,
    seed_label: int,
    episodes: int,
    result_dir: Path,
    verbose: bool,
) -> Dict[str, Any]:
    """Run one group/environment/seed setting and persist its trace bundle."""
    runtime_cfg = copy.deepcopy(dict(cfg))
    seed = _as_exact_int(runtime_cfg.get("fixed_seed_override"))
    if seed != int(seed_label):
        raise RuntimeError("setting configuration is not bound to its requested seed")
    setting_identity = build_experiment_identity(_identity_config(runtime_cfg), seed)
    save_experiment_snapshot(runtime_cfg, str(result_dir), seed)
    aggregate = MetricsAggregator(group_name, str(result_dir))
    all_events: list[Dict[str, Any]] = []
    all_dropped_rows: list[Dict[str, Any]] = []
    frame_runtimes: list[float] = []
    runtime_integrity_checks: list[bool] = []
    runtime_integrity_records: list[Dict[str, Any]] = []
    event_schema = str(runtime_cfg.get("event_log_schema_version", "rgd_event_log_v3"))

    for offset in range(int(episodes)):
        episode_id = int(seed_label) if int(episodes) == 1 else int(seed_label) * int(episodes) + offset
        env = None
        agent = None
        close_after = True
        try:
            env, obs, raw_ep_dir, prefix, resolved_seed, close_after = create_episode_env(
                episode_id, runtime_cfg, str(result_dir), [int(seed_label)]
            )
            if int(resolved_seed) != int(seed_label):
                raise RuntimeError("environment reset seed differs from the setting seed")
            ep_dir = Path(raw_ep_dir)
            scenario = create_scenario(
                getattr(env, "unwrapped", env),
                str(runtime_cfg.get("env_type", "") or ""),
                int(resolved_seed),
                str(ep_dir / "scenario.db"),
            )
            agent = create_episode_agent(scenario, runtime_cfg, str(result_dir))
            integrity_start = getattr(agent, "_runtime_integrity_start", None)
            if not isinstance(integrity_start, Mapping):
                raise RuntimeError("episode agent did not publish its runtime identity")
            physical, reasoning = create_episode_recorders(
                episode_id, int(resolved_seed), str(ep_dir), runtime_cfg
            )
            if physical is None or reasoning is None:
                raise RuntimeError("main-table runtime requires physical and reasoning recorders")
            safety = UnifiedSafetySystem(runtime_cfg)
            history = deque(maxlen=max(1, int(runtime_cfg.get("history_window", 16) or 16)))
            state = create_episode_runtime_state()
            episode_runtimes: list[float] = []
            env_type = str(runtime_cfg.get("env_type", "") or "")
            if env_type.startswith("metadrive-"):
                max_frames = int(runtime_cfg.get("simulation_duration", 0) or 0)
                if max_frames <= 0:
                    raise RuntimeError("simulation_duration must be positive")
            else:
                max_frames = resolve_policy_execution_horizon(
                    runtime_cfg,
                    context=f"{group_name}/{env_type}",
                ).expected_policy_steps
            for frame in range(max_frames):
                started = time.perf_counter()
                obs, done = execute_episode_step(
                    frame=frame,
                    env=env,
                    sce=scenario,
                    agent=agent,
                    obs=obs,
                    cfg=runtime_cfg,
                    safety_system=safety,
                    phys_rec=physical,
                    reas_rec=reasoning,
                    history_buffer=history,
                    episode_state=state,
                )
                elapsed = exclude_policy_pacing_sleep(
                    time.perf_counter() - started,
                    state,
                )
                episode_runtimes.append(elapsed)
                if verbose:
                    print(
                        f"{group_name}/{runtime_cfg['env_type']}/seed={seed_label} frame={frame}",
                        flush=True,
                    )
                if done:
                    break
            if not state["event_log"]:
                raise RuntimeError("episode ended without any event trace")
            if not bool(state["event_log"][-1].get("done", state["event_log"][-1].get("episode_done", False))):
                # The configured evaluation horizon is an explicit truncation
                # boundary. Preserve that provenance instead of representing a
                # horizon-limited trajectory as an environment termination.
                terminal_event = state["event_log"][-1]
                terminal_event.update(
                    {
                        "done": True,
                        "term": False,
                        "trunc": True,
                        "episode_done": True,
                        "terminal_cause": "truncated",
                        "evaluation_horizon_truncated": True,
                    }
                )
                if physical.frames:
                    physical.frames[-1]["done"] = True
            events = [dict(event) for event in state["event_log"]]
            records = [_record_to_dict(record) for record in list(reasoning.records)]
            if len(records) != len(events):
                raise RuntimeError("reasoning/event trace frame counts differ")
            finalize_episode_outputs(
                ep=episode_id,
                cfg=runtime_cfg,
                agent=agent,
                metrics_agg=aggregate,
                docs=[],
                collision_frame=int(state.get("collision_frame", -1)),
                result_dir=str(result_dir),
                prefix=prefix,
                frame_runtimes=episode_runtimes,
                phys_rec=physical,
                reas_rec=reasoning,
                event_log=events,
                pending_latency_queue=list(state.get("latency_replay_queue", []) or []),
                release_snapshots=dict(state.get("release_snapshots", {}) or {}),
            )
            event_payload = _write_episode_trace_counts(
                result_dir=result_dir,
                ep_dir=ep_dir,
                prefix=prefix,
                episode_id=episode_id,
                events=events,
                reasoning_records=records,
                event_schema=event_schema,
            )
            integrity_end = collect_runtime_integrity(agent)
            integrity_clean = dict(integrity_start) == dict(integrity_end)
            runtime_integrity_checks.append(integrity_clean)
            runtime_integrity_records.append(
                {
                    "episode_id": int(episode_id),
                    "identity_start": dict(integrity_start),
                    "identity_end": dict(integrity_end),
                    "clean": bool(integrity_clean),
                }
            )
            if not integrity_clean:
                raise RuntimeError("episode runtime identity drifted during execution")
            all_events.extend(events)
            all_dropped_rows.extend(
                dict(row)
                for row in list(
                    event_payload.get("pending_releases_dropped_at_episode_end", []) or []
                )
                if isinstance(row, Mapping)
            )
            frame_runtimes.extend(episode_runtimes)
        finally:
            if agent is not None:
                close_agent = getattr(agent, "close", None)
                if callable(close_agent):
                    close_agent()
            if env is not None and close_after:
                close_env = getattr(env, "close", None)
                if callable(close_env):
                    close_env()

    ending_identity = build_experiment_identity(_identity_config(runtime_cfg), seed)
    if not _same_identity(setting_identity, ending_identity):
        raise RuntimeError("cell experiment identity drifted during execution")
    aggregate.all_event_records = all_events
    metrics = _completed_metrics(
        aggregate,
        group_name=group_name,
        runtime_seconds=frame_runtimes,
        events=all_events,
        protocol=protocol,
        dropped_rows=all_dropped_rows,
        runtime_integrity_checks=runtime_integrity_checks,
    )
    dump_json(result_dir / f"{group_name}_rgd_metrics.json", {"comprehensive_metrics": metrics})
    _write_cell_completion_manifest(
        result_dir,
        group_name=group_name,
        env_type=str(runtime_cfg.get("env_type", "") or ""),
        seed_label=int(seed_label),
        episodes=int(episodes),
        expected_identity=setting_identity,
        runtime_integrity_checks=runtime_integrity_checks,
        ending_identity=ending_identity,
        runtime_integrity_records=runtime_integrity_records,
    )
    return _completed_run_row(
        metrics=metrics,
        identity=setting_identity,
        group_name=group_name,
        group_id=str(runtime_cfg.get("group_id", group_name)),
        env_type=str(runtime_cfg.get("env_type", "") or ""),
        seed_label=seed_label,
        episodes=episodes,
        result_dir=result_dir,
    )


def _v13_required_main_groups(protocol: Mapping[str, Any]) -> list[str]:
    contract = dict(protocol.get("tvt_submission_contract", {}) or {})
    artifacts = dict(
        dict(contract.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {}
    )
    return [
        str(name)
        for name in list(dict(artifacts.get("main_results", {}) or {}).get("required_groups", []) or [])
    ]


def _default_selection(
    protocol: Mapping[str, Any], mode: str, partition_name: str = "auto"
) -> tuple[list[str], list[str]]:
    protocol_name = str(protocol.get("protocol_name", "") or "")
    if protocol_name == V13_PROTOCOL_NAME and mode == "formal_run":
        if partition_name in {"auto", "main"}:
            return _v13_required_main_groups(protocol), [V12_MAIN_ENV]
        if partition_name == "mechanism":
            contract = dict(protocol.get("tvt_submission_contract", {}) or {})
            return [str(name) for name in list(contract.get("mechanism_source_groups", []) or [])], [V12_MAIN_ENV]
    execution = dict(protocol.get("execution", {}) or {})
    partition = dict(dict(execution.get("partitions", {}) or {}).get("highway_env", {}) or {})
    groups = list(partition.get("groups", []) or list(dict(protocol.get("groups", {}) or {})))
    envs = list(partition.get("envs", []) or [V12_MAIN_ENV])
    if mode == "quick_check":
        return groups, envs
    return groups, envs


def _resolve_group_env_matrix(
    protocol: Mapping[str, Any], groups: Sequence[str], envs: Sequence[str]
) -> Dict[str, list[str]]:
    declared = dict(dict(protocol.get("execution", {}) or {}).get("group_env_matrix", {}) or {})
    result: Dict[str, list[str]] = {}
    for group in groups:
        allowed = [str(value) for value in list(declared.get(group, envs) or [])]
        selected = [str(env) for env in envs if str(env) in allowed]
        if not selected:
            raise ValueError(f"group {group} has no selected supported environment")
        result[str(group)] = selected
    return result


def _archive_incomplete_cell(result_dir: Path) -> None:
    """Move an incomplete cell aside rather than mixing it with a fresh run."""
    if not result_dir.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = result_dir.with_name(f"{result_dir.name}.incomplete_{stamp}")
    suffix = 1
    while archived.exists():
        archived = result_dir.with_name(f"{result_dir.name}.incomplete_{stamp}_{suffix}")
        suffix += 1
    os.replace(result_dir, archived)


def _load_existing_group_rows(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [dict(row) for row in load_csv_rows(path)]


def _bundle_identity(config_path: Path, protocol_path: Path) -> Dict[str, Any]:
    config = Path(config_path).resolve()
    protocol = Path(protocol_path).resolve()
    config_digest = _sha256_file(config)
    protocol_digest = _sha256_file(protocol)
    if config_digest is None or protocol_digest is None:
        raise RuntimeError("bundle config/protocol source file is missing")
    return {
        "config_path": str(config),
        "config_sha256": config_digest,
        "protocol_path": str(protocol),
        "protocol_sha256": protocol_digest,
        "runtime_source_sha256": build_runtime_source_hash(REPO_ROOT),
    }


def _validate_existing_bundle_contract(
    path: Path,
    *,
    mode: str,
    partition: str,
    groups: Sequence[str],
    envs: Sequence[str],
    group_env_matrix: Mapping[str, Sequence[str]],
    seed_labels: Sequence[int],
    episodes: int,
    identity_start: Mapping[str, Any],
) -> None:
    if not path.is_file():
        return
    existing = _read_mapping(path)
    if existing is None:
        raise RuntimeError("existing result bundle manifest is invalid JSON")
    expected = {
        "bundle_kind": str(mode),
        "partition": str(partition),
        "groups": [str(value) for value in groups],
        "envs": [str(value) for value in envs],
        "group_env_matrix": {
            str(group): [str(env) for env in values]
            for group, values in group_env_matrix.items()
        },
        "seed_labels": [int(seed) for seed in seed_labels],
        "seeds": len(seed_labels),
        "episodes": int(episodes),
    }
    drift = [key for key, value in expected.items() if existing.get(key) != value]
    recorded_identity = existing.get("bundle_identity_start")
    if recorded_identity is not None and recorded_identity != dict(identity_start):
        drift.append("bundle_identity_start")
    if drift:
        raise RuntimeError(
            "existing run-stamp bundle contract differs for: "
            + ", ".join(sorted(set(drift)))
        )


def _apply_v12_floor_overlay(
    base_cfg: Mapping[str, Any],
    protocol: Mapping[str, Any],
    args: argparse.Namespace,
    seed_labels: Sequence[int],
) -> Dict[str, Any]:
    """Apply a verified v12 overlay when and only when the protocol requires it."""
    protocol_name = str(protocol.get("protocol_name", "") or "")
    if protocol_name != "rgd_tvt_identifiable_gate_v12":
        if getattr(args, "floor_overlay", None) is not None or getattr(args, "calibration_manifest", None) is not None:
            raise ValueError("v12 floor-overlay inputs require the identifiable-gate v12 protocol")
        return copy.deepcopy(dict(base_cfg))
    from tools.v12_floor_overlay import (
        DEFAULT_LOCK_PATH,
        apply_floor_overlay,
        enforce_v12_floor_overlay_contract,
        load_optional_verified_floor_overlay,
    )

    verified = load_optional_verified_floor_overlay(
        getattr(args, "floor_overlay", None),
        calibration_manifest_path=getattr(args, "calibration_manifest", None),
        protocol_path=Path(args.protocol),
        lock_path=(
            Path(args.calibration_lock)
            if getattr(args, "calibration_lock", None) is not None
            else DEFAULT_LOCK_PATH
        ),
    )
    enforce_v12_floor_overlay_contract(
        protocol_name,
        [int(seed) for seed in seed_labels],
        verified,
        allow_nonformal=bool(getattr(args, "allow_nonformal_v12", False)),
    )
    return apply_floor_overlay(base_cfg, verified) if verified is not None else copy.deepcopy(dict(base_cfg))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--result-root", type=Path, default=Path("results/highway_result"))
    parser.add_argument("--mode", choices=("quick_check", "formal_run"), default="quick_check")
    parser.add_argument("--run-stamp", default=None)
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--envs", nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--seed-value", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--simulation-duration", type=int, default=None)
    parser.add_argument(
        "--partition",
        choices=("auto", "main", "mechanism", "nonformal"),
        default="auto",
    )
    parser.add_argument(
        "--randomize-group-order",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Randomize arm order within each simulator-seed block.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--floor-overlay", type=Path, default=None)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--calibration-lock", type=Path, default=None)
    parser.add_argument("--allow-nonformal-v12", action="store_true")
    parser.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="Explicitly authorize a diagnostic/nonformal result bundle.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run independent simulator-seed blocks in parallel.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _ordered_groups_for_seed(
    groups: Sequence[str], seed_label: int, randomize_group_order: bool
) -> list[str]:
    ordered = [str(group) for group in groups]
    if randomize_group_order:
        random.Random(20260807 + int(seed_label)).shuffle(ordered)
    return ordered


def _run_seed_block(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Execute one seed block; all writes remain below that seed's cell dirs."""
    rows: list[Dict[str, Any]] = []
    for cell in list(task.get("cells", []) or []):
        payload = dict(cell)
        row = _run_setting(
            cfg=payload["cfg"],
            protocol=payload["protocol"],
            group_name=str(payload["group_name"]),
            seed_label=int(payload["seed_label"]),
            episodes=int(payload["episodes"]),
            result_dir=Path(payload["result_dir"]),
            verbose=bool(payload.get("verbose", False)),
        )
        row["requested_seed_start"] = int(payload["requested_seed_start"])
        rows.append(row)
    return {"seed": int(task["seed"]), "rows": rows}


def _execute_seed_blocks(
    tasks: Sequence[Mapping[str, Any]], workers: int
) -> list[Dict[str, Any]]:
    """Run seed blocks concurrently while preserving deterministic result order."""
    ordered_tasks = [dict(task) for task in tasks]
    if not ordered_tasks:
        return []
    if int(workers) == 1:
        return [_run_seed_block(task) for task in ordered_tasks]
    with ProcessPoolExecutor(
        max_workers=min(int(workers), len(ordered_tasks))
    ) as executor:
        return list(executor.map(_run_seed_block, ordered_tasks))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if int(args.workers) <= 0:
        raise ValueError("--workers must be a positive integer")
    protocol = load_formal_protocol(args.protocol)
    base_cfg = load_formal_base_config(protocol, args.config)
    protocol_name = str(protocol.get("protocol_name", "") or "")
    is_v12 = protocol_name == "rgd_tvt_identifiable_gate_v12"
    is_v13 = protocol_name == V13_PROTOCOL_NAME
    partition = str(args.partition)
    if partition == "auto":
        if str(args.mode) == "formal_run" and is_v13:
            partition = "main"
        elif str(args.mode) == "formal_run" and is_v12:
            partition = "main"
        else:
            partition = "nonformal"
    if partition in {"main", "mechanism"} and str(args.mode) != "formal_run":
        raise ValueError("formal partitions require --mode formal_run")
    if partition == "main" and not (is_v12 or is_v13):
        raise ValueError("the main partition requires a recognized formal protocol")
    if partition == "mechanism" and not is_v13:
        raise ValueError("the mechanism partition requires the v13 protocol")
    if partition == "nonformal" and not bool(args.allow_nonformal):
        raise ValueError("nonformal execution requires explicit --allow-nonformal")

    execution = dict(protocol.get("execution", {}) or {})
    mode_cfg = dict(execution.get(str(args.mode), {}) or {})
    default_groups, default_envs = _default_selection(
        protocol, str(args.mode), partition
    )
    selected_groups = list(args.groups) if args.groups else default_groups
    selected_envs = list(args.envs) if args.envs else default_envs
    group_specs = dict(iter_selected_groups(protocol.get("groups", {}), selected_groups))
    group_env_matrix = _resolve_group_env_matrix(protocol, selected_groups, selected_envs)
    episodes = int(args.episodes if args.episodes is not None else mode_cfg.get("episodes", 1))
    formal_seed_block: Optional[tuple[int, ...]] = None
    if is_v13 and partition == "main":
        formal_seed_block = _declared_seed_block(protocol, "main_seeds")
    elif is_v13 and partition == "mechanism":
        formal_seed_block = _declared_seed_block(
            protocol, "mechanism_evaluation_seeds"
        )
    elif is_v12 and partition == "main":
        formal_seed_block = V12_MAIN_SEEDS
    seeds = int(
        args.seeds
        if args.seeds is not None
        else len(formal_seed_block)
        if formal_seed_block is not None
        else mode_cfg.get("seeds", 1)
    )
    seed_start = int(
        args.seed_start
        if args.seed_start is not None
        else formal_seed_block[0]
        if formal_seed_block is not None
        else 0
    )
    if episodes <= 0 or seeds <= 0 or seed_start < 0:
        raise ValueError("episodes and seeds must be positive; seed-start must be nonnegative")
    if args.seed_value is not None and seeds != 1:
        raise ValueError("--seed-value is only valid for a single setting; use --seed-start for a seed block")
    seed_labels = list(range(seed_start, seed_start + seeds))
    if args.seed_value is not None:
        seed_labels = [int(args.seed_value)]

    if is_v12 and partition == "main":
        _validate_v12_main_preflight(
            protocol=protocol,
            base_cfg=base_cfg,
            partition="main",
            selected_groups=selected_groups,
            envs=selected_envs,
            seed_labels=seed_labels,
            episodes=episodes,
        )
    if is_v13 and partition in {"main", "mechanism"}:
        _validate_v13_preflight(
            protocol=protocol,
            base_cfg=base_cfg,
            partition=partition,
            selected_groups=selected_groups,
            envs=selected_envs,
            seed_labels=seed_labels,
            episodes=episodes,
            simulation_duration=args.simulation_duration,
        )
    randomize_group_order = (
        bool(args.randomize_group_order)
        if args.randomize_group_order is not None
        else bool(is_v13 and partition == "main")
    )
    if is_v13 and partition == "main" and not randomize_group_order:
        raise ValueError("v13 main preflight requires seed-block randomized group order")
    base_cfg = _apply_v12_floor_overlay(base_cfg, protocol, args, seed_labels)

    stamp = str(args.run_stamp or datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))
    prospective_bundle_root = Path(args.result_root) / str(args.mode) / stamp
    bundle_identity_start = _bundle_identity(Path(args.config), Path(args.protocol))
    execution_horizon_by_group_env: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if is_v13 and partition in {"main", "mechanism"}:
        representative_seed = int(seed_labels[0])
        for group_name in selected_groups:
            execution_horizon_by_group_env[group_name] = {}
            for env_type in group_env_matrix[group_name]:
                scenario = infer_env_label(env_type)
                result_dir = (
                    prospective_bundle_root
                    / group_name
                    / scenario
                    / f"seed_{representative_seed:02d}"
                )
                resolved_cfg = build_group_config(
                    base_cfg,
                    group_name,
                    group_specs[group_name],
                    env_type,
                    episodes,
                    result_dir,
                    protocol,
                )
                _apply_cli_runtime_overrides(resolved_cfg, args, representative_seed)
                resolved_cfg = _config_with_runtime_defaults(resolved_cfg, v12=is_v12)
                horizon = _validate_v13_resolved_cell(
                    protocol, partition, resolved_cfg
                )
                execution_horizon_by_group_env[group_name][env_type] = (
                    horizon.as_manifest()
                )
    prospective_manifest_path = prospective_bundle_root / "result_bundle_manifest.json"
    _validate_existing_bundle_contract(
        prospective_manifest_path,
        mode=str(args.mode),
        partition=partition,
        groups=selected_groups,
        envs=selected_envs,
        group_env_matrix=group_env_matrix,
        seed_labels=seed_labels,
        episodes=episodes,
        identity_start=bundle_identity_start,
    )
    manifest_path = write_result_bundle_manifest(
        Path(args.result_root),
        str(args.mode),
        stamp,
        selected_groups,
        selected_envs,
        len(seed_labels),
        episodes,
        args.seed_value,
        args.simulation_duration,
        str(Path(args.protocol).resolve()),
        group_env_matrix,
        seed_start=seed_labels[0],
        partition=partition,
        execution_horizon_by_group_env=execution_horizon_by_group_env,
    )
    bundle_root = manifest_path.parent
    manifest = _read_mapping(manifest_path)
    if manifest is None:
        raise RuntimeError("result bundle manifest was not written")
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    manifest.update(
        {
            "bundle_completion_state": "running",
            "bundle_identity_start": bundle_identity_start,
            "bundle_identity_end": None,
            "bundle_identity_clean": None,
            "workers": int(args.workers),
            "partition": partition,
            "method_version": (
                str(submission.get("rgd_method_version", "") or "")
                if is_v13 and partition in {"main", "mechanism"}
                else manifest.get("method_version")
            ),
            "query_gate_method_version": (
                str(submission.get("query_gate_method_version", "") or "")
                if is_v13 and partition in {"main", "mechanism"}
                else manifest.get("query_gate_method_version")
            ),
            "release_contract_version": (
                str(submission.get("release_contract_version", "") or "")
                if is_v13 and partition in {"main", "mechanism"}
                else manifest.get("release_contract_version")
            ),
            "group_order_policy": (
                "seed_block_randomized" if randomize_group_order else "declared_order"
            ),
            "group_order_randomization_seed_rule": (
                "20260807 + simulator_seed" if randomize_group_order else ""
            ),
        }
    )
    dump_json(manifest_path, manifest)
    prior_rows_by_group = {
        group_name: _load_existing_group_rows(
            bundle_root / group_name / f"{group_name}_run_rows.csv"
        )
        for group_name in selected_groups
    }
    new_rows_by_group: Dict[str, list[Dict[str, Any]]] = {
        group_name: [] for group_name in selected_groups
    }
    execution_order: list[Dict[str, Any]] = []
    seed_block_tasks: list[Dict[str, Any]] = []
    expected_identities: Dict[tuple[str, str, Optional[int]], Dict[str, Any]] = {}

    for seed_label in seed_labels:
        ordered_groups = _ordered_groups_for_seed(
            selected_groups, int(seed_label), randomize_group_order
        )
        block_cells: list[Dict[str, Any]] = []
        for order_index, group_name in enumerate(ordered_groups):
            execution_order.append(
                {
                    "seed": int(seed_label),
                    "order": int(order_index),
                    "group": str(group_name),
                }
            )
            for env_type in group_env_matrix[group_name]:
                scenario = infer_env_label(env_type)
                result_dir = bundle_root / group_name / scenario / f"seed_{int(seed_label):02d}"
                cfg = build_group_config(
                    base_cfg,
                    group_name,
                    group_specs[group_name],
                    env_type,
                    episodes,
                    result_dir,
                    protocol,
                )
                _apply_cli_runtime_overrides(cfg, args, int(seed_label))
                cfg = _config_with_runtime_defaults(cfg, v12=is_v12)
                if is_v13 and partition in {"main", "mechanism"}:
                    _validate_v13_resolved_cell(protocol, partition, cfg)
                expected_identity = build_experiment_identity(cfg, int(seed_label))
                if expected_identity.get("source_hash") != bundle_identity_start.get(
                    "runtime_source_sha256"
                ):
                    raise RuntimeError("cell source hash differs from bundle start identity")
                key = (str(group_name), str(env_type), int(seed_label))
                expected_identities[key] = expected_identity
                if args.resume and result_dir.exists():
                    recovered = _recover_completed_cell_row(
                        result_dir,
                        group_name=group_name,
                        env_type=env_type,
                        seed_label=int(seed_label),
                        episodes=episodes,
                        expected_identity=expected_identity,
                    )
                    if recovered is not None:
                        recovered["requested_seed_start"] = int(seed_labels[0])
                        new_rows_by_group[group_name].append(recovered)
                        continue
                if result_dir.exists():
                    if not args.restart_incomplete:
                        raise RuntimeError(
                            f"incomplete or incompatible cell exists: {result_dir}; "
                            "rerun with --restart-incomplete to archive it before execution"
                        )
                    _archive_incomplete_cell(result_dir)
                block_cells.append(
                    {
                        "cfg": cfg,
                        "protocol": protocol,
                        "group_name": str(group_name),
                        "seed_label": int(seed_label),
                        "episodes": int(episodes),
                        "result_dir": str(result_dir),
                        "verbose": bool(args.verbose),
                        "requested_seed_start": int(seed_labels[0]),
                    }
                )
        if block_cells:
            seed_block_tasks.append({"seed": int(seed_label), "cells": block_cells})

    manifest = _read_mapping(manifest_path)
    if manifest is None:
        raise RuntimeError("result bundle manifest disappeared before execution")
    manifest["randomized_block_run_order"] = execution_order
    dump_json(manifest_path, manifest)

    block_results = _execute_seed_blocks(seed_block_tasks, int(args.workers))
    for block in block_results:
        for row in list(block.get("rows", []) or []):
            key = _row_key(row)
            expected_identity = expected_identities.get(key)
            if expected_identity is None or not _same_identity(row, expected_identity):
                raise RuntimeError(f"worker returned an unexpected or drifted cell row: {key}")
            result_dir = Path(str(row.get("result_dir", "") or ""))
            if not _resume_row_is_complete(
                row,
                result_dir,
                group_name=key[0],
                env_type=key[1],
                seed_label=int(key[2]),
                episodes=episodes,
                expected_identity=expected_identity,
            ):
                raise RuntimeError(f"worker returned a cell without artifact closure: {key}")
            new_rows_by_group[key[0]].append(dict(row))

    rows_by_group: Dict[str, list[Dict[str, Any]]] = {}
    for group_name in selected_groups:
        prior_rows = prior_rows_by_group[group_name]
        new_rows = new_rows_by_group[group_name]
        rows = _merge_group_run_rows(prior_rows, new_rows)
        # Keep only rows for the active bundle's declared matrix.  Rows from a
        # previous incompatible invocation remain on disk only when explicitly
        # archived, never silently folded into the current summary.
        rows = _filter_active_cohort_rows(
            rows,
            group_name=group_name,
            envs=group_env_matrix[group_name],
            seed_labels=seed_labels,
        )
        write_rows_csv(bundle_root / group_name / f"{group_name}_run_rows.csv", rows, RUN_ROW_FIELDS)
        summarise_rows(
            rows,
            group_name,
            str(group_specs[group_name].get("id", group_name)),
            bundle_root / group_name / f"{group_name}_summary.json",
        )
        rows_by_group[group_name] = rows

    actual_keys = {
        _row_key(row)
        for rows in rows_by_group.values()
        for row in rows
    }
    expected_keys = {
        (str(group), str(env), int(seed))
        for group in selected_groups
        for env in group_env_matrix[group]
        for seed in seed_labels
    }
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise RuntimeError(
            f"active result matrix is incomplete or contaminated; missing={missing}, extra={extra}"
        )
    write_overall_comparison_assets(bundle_root, rows_by_group, selected_groups, selected_envs)

    bundle_identity_end = _bundle_identity(Path(args.config), Path(args.protocol))
    identity_clean = bundle_identity_end == bundle_identity_start
    manifest = _read_mapping(manifest_path)
    if manifest is None:
        raise RuntimeError("result bundle manifest disappeared during execution")
    manifest.update(
        {
            "bundle_identity_end": bundle_identity_end,
            "bundle_identity_clean": bool(identity_clean),
            "bundle_completion_state": "complete" if identity_clean else "identity_drift",
            "completed_cell_count": len(actual_keys),
            "expected_cell_count": len(expected_keys),
        }
    )
    dump_json(manifest_path, manifest)
    if not identity_clean:
        raise RuntimeError("bundle config/protocol/runtime source identity drifted during execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
