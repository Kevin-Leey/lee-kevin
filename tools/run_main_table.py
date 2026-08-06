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
import json
import os
import sys
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.formal_surface import COMPARISON_HEADLINE_FIELDS  # noqa: E402
from dilu.evaluation.metrics_aggregator import MetricsAggregator  # noqa: E402
from dilu.evaluation.reporter import build_experiment_identity, save_experiment_snapshot  # noqa: E402
from dilu.runtime_episode_finalize import finalize_episode_outputs  # noqa: E402
from dilu.runtime_episode_setup import (  # noqa: E402
    create_episode_agent,
    create_episode_env,
    create_episode_recorders,
)
from dilu.runtime_frame_trace import create_episode_runtime_state  # noqa: E402
from dilu.runtime_support import execute_episode_step  # noqa: E402
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
    "slow_call_rate",
    "slow_call_success_rate",
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


def _read_mapping(path: Path) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


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

    root = Path(result_dir)
    manifest = _read_mapping(root / "runtime_manifest.json")
    snapshot = _read_mapping(root / "experiment_snapshot.json")
    if manifest is None or snapshot is None:
        return False
    if not _identity_matches(manifest, expected_identity, int(seed_label)):
        return False
    if not _identity_matches(snapshot, expected_identity, int(seed_label)):
        return False
    if _as_exact_int(snapshot.get("seed_start")) != int(seed_label):
        return False
    if list(snapshot.get("seeds_used", [])) != [int(seed_label)]:
        return False
    return True


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
) -> None:
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


def _completed_metrics(
    aggregate: MetricsAggregator,
    *,
    group_name: str,
    runtime_seconds: Sequence[float],
    events: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = aggregate.calculate_comprehensive_metrics()
    frame_count = int(metrics.get("total_frames", 0) or 0)
    if frame_count <= 0:
        raise RuntimeError(f"{group_name}: completed setting has no physical frames")
    if len(events) != frame_count:
        raise RuntimeError(
            f"{group_name}: event/physical frame mismatch ({len(events)} != {frame_count})"
        )
    attempts = sum(bool(event.get("slow_request_attempted", False)) for event in events)
    successes = sum(
        bool(event.get("slow_request_attempted", False))
        and bool(event.get("slow_request_valid_return", False))
        and bool(event.get("slow_reasoning_success", False))
        and not bool(event.get("slow_request_failed", False))
        for event in events
    )
    failures = sum(bool(event.get("slow_request_failed", False)) for event in events)
    if failures or successes != attempts:
        raise RuntimeError(
            f"{group_name}: slow-executor failures cannot be reported as completed main-table results"
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
            "evaluation_runtime_stable": True,
            "runtime_integrity_clean": True,
            "runtime_integrity_violation_rate": 0.0,
            "slow_call_rate": float(attempts / frame_count),
            "slow_call_success_rate": float(successes / attempts) if attempts else 0.0,
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
    setting_identity = build_experiment_identity(runtime_cfg, seed)
    save_experiment_snapshot(runtime_cfg, str(result_dir), seed)
    aggregate = MetricsAggregator(group_name, str(result_dir))
    all_events: list[Dict[str, Any]] = []
    frame_runtimes: list[float] = []
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
            physical, reasoning = create_episode_recorders(
                episode_id, int(resolved_seed), str(ep_dir), runtime_cfg
            )
            if physical is None or reasoning is None:
                raise RuntimeError("main-table runtime requires physical and reasoning recorders")
            safety = UnifiedSafetySystem(runtime_cfg)
            history = deque(maxlen=max(1, int(runtime_cfg.get("history_window", 16) or 16)))
            state = create_episode_runtime_state()
            episode_runtimes: list[float] = []
            max_frames = int(runtime_cfg.get("simulation_duration", 0) or 0)
            if max_frames <= 0:
                raise RuntimeError("simulation_duration must be positive")
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
                elapsed = time.perf_counter() - started
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
            _write_episode_trace_counts(
                result_dir=result_dir,
                ep_dir=ep_dir,
                prefix=prefix,
                episode_id=episode_id,
                events=events,
                reasoning_records=records,
                event_schema=event_schema,
            )
            all_events.extend(events)
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

    aggregate.all_event_records = all_events
    metrics = _completed_metrics(
        aggregate,
        group_name=group_name,
        runtime_seconds=frame_runtimes,
        events=all_events,
        protocol=protocol,
    )
    dump_json(result_dir / f"{group_name}_rgd_metrics.json", {"comprehensive_metrics": metrics})
    return {
        "group": str(group_name),
        "group_id": str(runtime_cfg.get("group_id", group_name)),
        "env": str(runtime_cfg.get("env_type", "") or ""),
        "seed_idx": int(seed_label),
        "fixed_seed_override": int(seed),
        "seed_start": int(seed),
        "episodes_run": int(episodes),
        "total_frames": int(metrics["total_frames"]),
        "result_dir": str(result_dir.resolve()),
        **{
            field: setting_identity[field]
            for field in ("protocol_id", "protocol_hash", "config_hash", "source_hash")
        },
        "slow_call_rate": float(metrics["slow_call_rate"]),
        "slow_call_success_rate": float(metrics["slow_call_success_rate"]),
        **{field: float(metrics[field]) for field in COMPARISON_HEADLINE_FIELDS},
    }


def _default_selection(protocol: Mapping[str, Any], mode: str) -> tuple[list[str], list[str]]:
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
    parser.add_argument("--partition", default="auto")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--floor-overlay", type=Path, default=None)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--calibration-lock", type=Path, default=None)
    parser.add_argument("--allow-nonformal-v12", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    protocol = load_formal_protocol(args.protocol)
    base_cfg = load_formal_base_config(protocol, args.config)
    execution = dict(protocol.get("execution", {}) or {})
    mode_cfg = dict(execution.get(str(args.mode), {}) or {})
    default_groups, default_envs = _default_selection(protocol, str(args.mode))
    selected_groups = list(args.groups) if args.groups else default_groups
    selected_envs = list(args.envs) if args.envs else default_envs
    group_specs = dict(iter_selected_groups(protocol.get("groups", {}), selected_groups))
    group_env_matrix = _resolve_group_env_matrix(protocol, selected_groups, selected_envs)
    episodes = int(args.episodes if args.episodes is not None else mode_cfg.get("episodes", 1))
    seeds = int(args.seeds if args.seeds is not None else mode_cfg.get("seeds", 1))
    seed_start = int(args.seed_start if args.seed_start is not None else 0)
    if episodes <= 0 or seeds <= 0 or seed_start < 0:
        raise ValueError("episodes and seeds must be positive; seed-start must be nonnegative")
    if args.seed_value is not None and seeds != 1:
        raise ValueError("--seed-value is only valid for a single setting; use --seed-start for a seed block")
    seed_labels = list(range(seed_start, seed_start + seeds))
    if args.seed_value is not None:
        seed_labels = [int(args.seed_value)]

    partition = str(args.partition)
    is_v12 = str(protocol.get("protocol_name", "") or "") == "rgd_tvt_identifiable_gate_v12"
    if is_v12 and partition == "auto":
        partition = "main" if tuple(seed_labels) == V12_MAIN_SEEDS else "nonformal"
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
    base_cfg = _apply_v12_floor_overlay(base_cfg, protocol, args, seed_labels)

    stamp = str(args.run_stamp or datetime.now().strftime("%Y-%m-%d/%H-%M-%S"))
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
    )
    bundle_root = manifest_path.parent
    rows_by_group: Dict[str, list[Dict[str, Any]]] = {}

    for group_name in selected_groups:
        prior_rows = _load_existing_group_rows(bundle_root / group_name / f"{group_name}_run_rows.csv")
        new_rows: list[Dict[str, Any]] = []
        for env_type in group_env_matrix[group_name]:
            scenario = infer_env_label(env_type)
            for seed_label in seed_labels:
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
                expected_identity = build_experiment_identity(cfg, int(seed_label))
                existing = next(
                    (
                        row
                        for row in prior_rows
                        if _row_key(row) == (str(group_name), str(env_type), int(seed_label))
                    ),
                    None,
                )
                if args.resume and existing is not None and _resume_row_is_complete(
                    existing,
                    result_dir,
                    group_name=group_name,
                    env_type=env_type,
                    seed_label=int(seed_label),
                    episodes=episodes,
                    expected_identity=expected_identity,
                ):
                    continue
                if result_dir.exists():
                    if not args.restart_incomplete:
                        raise RuntimeError(
                            f"incomplete or incompatible cell exists: {result_dir}; "
                            "rerun with --restart-incomplete to archive it before execution"
                        )
                    _archive_incomplete_cell(result_dir)
                row = _run_setting(
                    cfg=cfg,
                    protocol=protocol,
                    group_name=group_name,
                    seed_label=int(seed_label),
                    episodes=episodes,
                    result_dir=result_dir,
                    verbose=bool(args.verbose),
                )
                row["requested_seed_start"] = int(seed_labels[0])
                new_rows.append(row)

        rows = _merge_group_run_rows(prior_rows, new_rows)
        # Keep only rows for the active bundle's declared matrix.  Rows from a
        # previous incompatible invocation remain on disk only when explicitly
        # archived, never silently folded into the current summary.
        rows = [
            row
            for row in rows
            if str(row.get("group", "") or "") == group_name
            and str(row.get("env", "") or "") in group_env_matrix[group_name]
        ]
        write_rows_csv(bundle_root / group_name / f"{group_name}_run_rows.csv", rows, RUN_ROW_FIELDS)
        summarise_rows(
            rows,
            group_name,
            str(group_specs[group_name].get("id", group_name)),
            bundle_root / group_name / f"{group_name}_summary.json",
        )
        rows_by_group[group_name] = rows

    write_overall_comparison_assets(bundle_root, rows_by_group, selected_groups, selected_envs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
