"""Fail-closed acceptance guard for the paper-facing v12 main experiment.

This verifier reads a completed ``run_main_table.py`` result bundle.  It does
not run a simulator or contact an LLM endpoint.  Acceptance is intentionally
narrow: one highway-v0 episode for every arm/seed cell in the frozen
six-arm, 4000--4029 design, produced by the current identifiable-gate v12
runtime and the Qwen3-8B slow executor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.formal_surface import COMPARISON_HEADLINE_FIELDS  # noqa: E402
from dilu.evaluation.reporter import (  # noqa: E402
    _build_runtime_experiment_config,
    build_runtime_source_hash,
)  # noqa: E402
from tools.protocol_io import parse_inclusive_int_range  # noqa: E402
from tools.run_main_table import _bind_setting_seed  # noqa: E402
from tools.run_main_table_runtime import (  # noqa: E402
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
)  # noqa: E402
from tools.run_main_table_support import _overall_rows  # noqa: E402


SCHEMA_VERSION = "identifiable_gate_v12_main_acceptance_v1"
METHOD_VERSION = "identifiable_gate_v12"
ENVIRONMENT = "highway-v0"
SCENARIO = "highway"
EXPECTED_SEEDS = tuple(range(4000, 4030))
EXPECTED_ARMS = (
    "rgd_fixed_policy",
    "always_fast",
    "always_slow",
    "random_budget",
    "uncertainty_budget",
    "risk_budget",
)
ZERO_QUERY_ARMS = frozenset({"always_fast"})
EXPECTED_PROVIDER = "siliconflow"
EXPECTED_MODEL = "Qwen/Qwen3-8B"
EXPECTED_BASE_URL_ORIGIN = "https://api.siliconflow.cn"
EXPECTED_EPISODES_PER_CELL = 1

_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_INTEGER = re.compile(r"^[+-]?\d+$")
_FINAL_TERMINAL_CAUSES = frozenset(
    {"arrived", "collision", "terminated", "truncated"}
)
_IDENTITY_FIELDS = ("protocol_id", "protocol_hash", "config_hash", "source_hash")
_PROTOCOL_DERIVED_FIELDS = frozenset(
    {
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
)
_FAILURE_TEXT_KEYS = frozenset(
    {
        "auth_error",
        "authentication_error",
        "error",
        "error_message",
        "exception",
        "failure_reason",
        "llm_error",
        "parse_error",
        "schema_error",
        "service_error",
        "slow_reasoning_failure_reason",
        "stderr",
        "timeout_error",
    }
)
_HTTP_STATUS_KEYS = frozenset(
    {"http_status", "http_status_code", "response_status", "status_code"}
)
_PROCESS_EXIT_KEYS = frozenset(
    {"exit_code", "process_exit_code", "producer_exit_code", "returncode"}
)
_LOG_FAILURE_PATTERN = re.compile(
    r"(?:\bhttp\s*[45]\d\d\b|\b401\b|unauthori[sz]ed|authentication\s+failed|"
    r"service\s+unavailable|timed?\s*out|\btimeout\b|structured_parse_failed|"
    r"fast_after_slow_failure)",
    flags=re.IGNORECASE,
)


class AcceptanceError(ValueError):
    """Raised when a paper-facing bundle violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def _unique_object(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise AcceptanceError(f"non-finite JSON constant {value!r}")


def _load_json(path: Path, *, allow_nonfinite: bool = False) -> Any:
    _require(path.is_file(), f"required JSON artifact is missing: {path}")
    kwargs: Dict[str, Any] = {"object_pairs_hook": _unique_object}
    if not allow_nonfinite:
        kwargs["parse_constant"] = _reject_nonfinite
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), **kwargs)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError(f"invalid JSON artifact {path}: {exc}") from exc


def _load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    _require(path.is_file(), f"required CSV artifact is missing: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows = list(csv.reader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise AcceptanceError(f"invalid CSV artifact {path}: {exc}") from exc
    _require(bool(raw_rows), f"CSV artifact is empty: {path}")
    header = raw_rows[0]
    _require(bool(header) and all(header), f"CSV header is empty or malformed: {path}")
    _require(len(header) == len(set(header)), f"CSV header contains duplicate columns: {path}")
    rows: List[Dict[str, str]] = []
    for line_number, values in enumerate(raw_rows[1:], start=2):
        _require(
            len(values) == len(header),
            f"CSV row width mismatch at {path}:{line_number}",
        )
        rows.append(dict(zip(header, values)))
    return list(header), rows


def _exact_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise AcceptanceError(f"{field} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    text = str(value).strip() if isinstance(value, str) else ""
    _require(bool(_INTEGER.fullmatch(text)), f"{field} must be an exact integer")
    return int(text)


def _finite_float(value: Any, *, field: str) -> float:
    _require(not isinstance(value, bool), f"{field} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise AcceptanceError(f"{field} must be numeric") from exc
    _require(math.isfinite(parsed), f"{field} must be finite")
    return parsed


def _exact_bool(value: Any, *, field: str) -> bool:
    _require(type(value) is bool, f"{field} must be an explicit JSON boolean")
    return bool(value)


def _canonical_runtime_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_protocol_preimage(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(key): deepcopy(value)
        for key, value in manifest.items()
        if str(key) not in _PROTOCOL_DERIVED_FIELDS
    }


def _resolve_declared_path(raw: Any, *, repo_root: Path) -> Path:
    text = str(raw or "").strip()
    _require(bool(text), "declared artifact path is empty")
    candidate = Path(text)
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()


def _resolve_bundle(bundle_or_manifest: Path) -> Tuple[Path, Path]:
    path = Path(bundle_or_manifest)
    if path.is_dir():
        return path.resolve(), (path / "result_bundle_manifest.json").resolve()
    _require(
        path.name == "result_bundle_manifest.json",
        "input must be a bundle directory or result_bundle_manifest.json",
    )
    return path.parent.resolve(), path.resolve()


def _validate_protocol_contract(protocol: Mapping[str, Any]) -> None:
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    _require(
        str(submission.get("rgd_method_version", "") or "") == METHOD_VERSION,
        "formal protocol method version is not identifiable_gate_v12",
    )
    _require(
        str(submission.get("main_environment", "") or "") == ENVIRONMENT,
        "formal protocol main environment drift",
    )
    _require(
        parse_inclusive_int_range(
            submission.get("main_seeds"), field_name="tvt_submission_contract.main_seeds"
        )
        == (EXPECTED_SEEDS[0], EXPECTED_SEEDS[-1]),
        "formal protocol main seed block drift",
    )
    artifacts = dict(
        (submission.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {}
    )
    main_results = dict(artifacts.get("main_results", {}) or {})
    _require(main_results.get("paper_facing") is True, "main results are not paper-facing")
    _require(
        str(main_results.get("method_version", "") or "") == METHOD_VERSION,
        "main-results evidence method version drift",
    )
    _require(
        str(main_results.get("seed_contract", "") or "") == "main_seeds",
        "main-results seed-contract drift",
    )
    _require(
        tuple(main_results.get("required_groups", []) or []) == EXPECTED_ARMS,
        "main-results required-group contract is missing or drifted",
    )
    _require(
        str(main_results.get("environment", "") or "") == ENVIRONMENT,
        "main-results environment contract is missing or drifted",
    )
    _require(
        _exact_int(
            main_results.get("episodes_per_key"),
            field="main_results.episodes_per_key",
        )
        == EXPECTED_EPISODES_PER_CELL,
        "main-results episodes-per-key contract drift",
    )
    _require(
        tuple(main_results.get("zero_query_allowed_groups", []) or [])
        == tuple(ZERO_QUERY_ARMS),
        "main-results zero-query exception contract drift",
    )
    _require(
        str(main_results.get("technical_slow_failure_policy", "") or "")
        == "reject_any_scheduled_slow_failure_or_fallback",
        "main-results technical slow-failure policy drift",
    )
    groups = dict(protocol.get("groups", {}) or {})
    _require(
        set(EXPECTED_ARMS).issubset(groups),
        "formal protocol omits one or more frozen main-result arms",
    )


def _validate_bundle_manifest(
    manifest: Mapping[str, Any],
    *,
    bundle_root: Path,
    manifest_path: Path,
    protocol_path: Path,
    repo_root: Path,
) -> Optional[int]:
    _require(manifest_path.parent == bundle_root, "bundle manifest is outside bundle root")
    _require(manifest.get("bundle_kind") == "formal_run", "bundle is not a formal_run")
    _require(tuple(manifest.get("groups", []) or []) == EXPECTED_ARMS, "bundle arm registry drift")
    _require(tuple(manifest.get("envs", []) or []) == (ENVIRONMENT,), "bundle environment drift")
    matrix = dict(manifest.get("group_env_matrix", {}) or {})
    _require(tuple(matrix) == EXPECTED_ARMS, "bundle group/environment matrix keys drift")
    for arm in EXPECTED_ARMS:
        _require(tuple(matrix.get(arm, []) or []) == (ENVIRONMENT,), f"{arm}: environment matrix drift")
    _require(_exact_int(manifest.get("seeds"), field="bundle.seeds") == len(EXPECTED_SEEDS), "bundle seed count drift")
    _require(_exact_int(manifest.get("episodes"), field="bundle.episodes") == EXPECTED_EPISODES_PER_CELL, "bundle episode count drift")
    _require(manifest.get("seed_policy") == "fixed_per_setting", "bundle seed policy drift")
    _require(_exact_int(manifest.get("seed_start"), field="bundle.seed_start") == EXPECTED_SEEDS[0], "bundle seed start drift")
    labels = tuple(
        _exact_int(value, field="bundle.seed_labels")
        for value in list(manifest.get("seed_labels", []) or [])
    )
    _require(labels == EXPECTED_SEEDS, "bundle seed labels are incomplete, duplicated, or out of order")
    _require(manifest.get("seed_value") is None, "bundle must not reuse one fixed seed")
    _require(
        _resolve_declared_path(manifest.get("formal_protocol_path"), repo_root=repo_root)
        == protocol_path,
        "bundle formal-protocol path drift",
    )
    _require(
        _resolve_declared_path(manifest.get("bundle_root"), repo_root=repo_root)
        == bundle_root,
        "declared bundle root differs from verifier input",
    )
    entries = tuple(manifest.get("entry_artifacts", []) or [])
    _require(
        entries
        == (
            "result_bundle_manifest.json",
            "overall_group_comparison.csv",
            "overall_group_comparison.json",
        ),
        "bundle entry-artifact registry drift",
    )
    duration = manifest.get("simulation_duration")
    if duration is None:
        return None
    parsed = _exact_int(duration, field="bundle.simulation_duration")
    _require(parsed > 0, "bundle simulation duration must be positive")
    return parsed


def _validate_hash(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    _require(bool(_HEX_256.fullmatch(text)), f"{field} is not a SHA-256 digest")
    return text


def _validate_backend(manifest: Mapping[str, Any], *, cell: str) -> None:
    backend = dict(manifest.get("llm_backend", {}) or {})
    _require(str(backend.get("provider", "") or "").lower() == EXPECTED_PROVIDER, f"{cell}: LLM provider drift")
    _require(backend.get("requested_model") == EXPECTED_MODEL, f"{cell}: requested Qwen identity drift")
    _require(backend.get("resolved_chat_model") == EXPECTED_MODEL, f"{cell}: resolved Qwen identity drift")
    _require(backend.get("base_url_origin") == EXPECTED_BASE_URL_ORIGIN, f"{cell}: Qwen endpoint origin drift")
    _require(_finite_float(backend.get("temperature"), field=f"{cell}.temperature") == 0.0, f"{cell}: non-deterministic LLM temperature")
    _require(_finite_float(backend.get("request_timeout_s"), field=f"{cell}.request_timeout_s") > 0.0, f"{cell}: invalid request timeout")
    retry = dict(backend.get("retry_contract", {}) or {})
    _require(_exact_int(retry.get("max_attempts"), field=f"{cell}.max_attempts") >= 1, f"{cell}: invalid retry count")
    statuses = tuple(_exact_int(item, field=f"{cell}.retryable_http_status") for item in list(retry.get("retryable_http_statuses", []) or []))
    _require(statuses == (408, 409, 425, 429, 500, 502, 503, 504), f"{cell}: retryable HTTP status contract drift")
    _require(
        tuple(retry.get("retryable_transport_failures", []) or [])
        == ("timeout", "connection_error"),
        f"{cell}: retryable transport contract drift",
    )
    provenance = dict(manifest.get("slow_path_provenance", {}) or {})
    _require(provenance.get("executor") == "online_llm", f"{cell}: slow executor is not online_llm")
    _require(provenance.get("trace_cache_enabled") is False, f"{cell}: trace cache must be disabled")


def _validate_identity(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    live_source_hash: str,
    cell: str,
) -> None:
    for field in _IDENTITY_FIELDS:
        row_value = str(row.get(field, "") or "")
        manifest_value = str(manifest.get(field, "") or "")
        snapshot_value = str(snapshot.get(field, "") or "")
        _require(
            row_value == manifest_value == snapshot_value,
            f"{cell}: {field} differs across CSV, runtime manifest, and snapshot",
        )
    protocol_hash = _validate_hash(manifest.get("protocol_hash"), field=f"{cell}.protocol_hash")
    config_hash = _validate_hash(manifest.get("config_hash"), field=f"{cell}.config_hash")
    source_hash = _validate_hash(manifest.get("source_hash"), field=f"{cell}.source_hash")
    _require(
        config_hash == _canonical_runtime_hash(manifest.get("config")),
        f"{cell}: config_hash does not derive from embedded config",
    )
    _require(
        protocol_hash == _canonical_runtime_hash(_runtime_protocol_preimage(manifest)),
        f"{cell}: protocol_hash does not derive from runtime-manifest preimage",
    )
    _require(source_hash == live_source_hash, f"{cell}: source_hash differs from live runtime source")
    expected_protocol_id = f"{manifest.get('experiment_name')}::{protocol_hash[:16]}"
    _require(manifest.get("protocol_id") == expected_protocol_id, f"{cell}: protocol_id drift")


def _validate_manifest_snapshot_contract(
    manifest: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
    result_dir: Path,
    protocol_path: Path,
    repo_root: Path,
    expected_config: Mapping[str, Any],
    expected_group_cfg: Mapping[str, Any],
    cell: str,
) -> None:
    _require(manifest.get("protocol_name") == arm, f"{cell}: protocol name drift")
    _require(manifest.get("experiment_name") == arm, f"{cell}: experiment name drift")
    _require(snapshot.get("experiment_name") == arm, f"{cell}: snapshot experiment name drift")
    _require(manifest.get("env_type") == ENVIRONMENT, f"{cell}: runtime environment drift")
    _require(manifest.get("scenario_type") == SCENARIO, f"{cell}: scenario label drift")
    expected_group_fields = {
        "paper_role": str(expected_group_cfg.get("paper_role", "paper_baseline") or "paper_baseline"),
        "publication_track": str(expected_group_cfg.get("publication_track", "main_text") or "main_text"),
        "theory_family": str(expected_group_cfg.get("theory_family", "unclassified") or "unclassified"),
        "alternative_explanation_axis": str(expected_group_cfg.get("alternative_explanation_axis", "unspecified") or "unspecified"),
        "ablation_dimension": str(expected_group_cfg.get("ablation_dimension", "none") or "none"),
    }
    for field, expected in expected_group_fields.items():
        _require(manifest.get(field) == expected, f"{cell}: {field} differs from formal protocol")
    for payload_name, payload in (("manifest", manifest), ("snapshot", snapshot)):
        _require(_exact_int(payload.get("fixed_seed_override"), field=f"{cell}.{payload_name}.fixed_seed_override") == seed, f"{cell}: fixed seed drift")
        _require(_exact_int(payload.get("seed_start"), field=f"{cell}.{payload_name}.seed_start") == seed, f"{cell}: seed start drift")
    _require(tuple(manifest.get("resolved_seeds", []) or []) == (seed,), f"{cell}: resolved seed list drift")
    _require(tuple(snapshot.get("seeds_used", []) or []) == (seed,), f"{cell}: snapshot seed list drift")
    _require(manifest.get("config") == expected_config, f"{cell}: runtime config differs from formal protocol")
    _require(snapshot.get("config") == expected_config, f"{cell}: snapshot config differs from formal protocol")
    _require(
        _exact_int(manifest.get("simulation_duration"), field=f"{cell}.simulation_duration")
        == _exact_int(expected_config.get("simulation_duration"), field=f"{cell}.config.simulation_duration"),
        f"{cell}: manifest/config simulation duration drift",
    )
    runtime_contract = dict((manifest.get("config", {}) or {}).get("runtime_contract", {}) or {})
    _require(runtime_contract.get("method_version") == METHOD_VERSION, f"{cell}: method version drift")
    slow_cfg = dict((manifest.get("config", {}) or {}).get("slow_thinking", {}) or {})
    _require(slow_cfg.get("executor") == "online_llm", f"{cell}: config slow executor drift")
    _require(
        _resolve_declared_path(manifest.get("experiment_plan_path"), repo_root=repo_root)
        == protocol_path,
        f"{cell}: runtime protocol path drift",
    )
    _require(
        _resolve_declared_path(snapshot.get("experiment_plan_path"), repo_root=repo_root)
        == protocol_path,
        f"{cell}: snapshot protocol path drift",
    )
    runtime_manifest_path = result_dir / "runtime_manifest.json"
    _require(
        _resolve_declared_path(snapshot.get("protocol_manifest_path"), repo_root=repo_root)
        == runtime_manifest_path.resolve(),
        f"{cell}: snapshot protocol-manifest link drift",
    )
    shared_fields = (
        "config_scope",
        "single_core_method_name",
        "primary_evaluation_subject",
        "recoverability_core_variables",
        "recoverability_object_status",
        "paper_role",
        "publication_track",
        "theory_family",
        "alternative_explanation_axis",
        "ablation_dimension",
        "enable_memory_retrieval",
        "few_shot_num",
        "memory_artifacts",
        "slow_path_provenance",
        "runtime_environment",
        "source_control",
    )
    for field in shared_fields:
        _require(manifest.get(field) == snapshot.get(field), f"{cell}: {field} snapshot drift")


def _validate_failure_fields(payload: Any, *, location: str) -> None:
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key).strip().lower()
            child = f"{location}.{raw_key}"
            if key in _FAILURE_TEXT_KEYS:
                text = str(value or "").strip()
                _require(text.lower() in {"", "none", "null"}, f"{child} records a technical failure: {text}")
            elif key in _HTTP_STATUS_KEYS and value not in (None, ""):
                status = _exact_int(value, field=child)
                _require(status < 400, f"{child} records HTTP {status}")
            elif key in _PROCESS_EXIT_KEYS and value not in (None, ""):
                code = _exact_int(value, field=child)
                _require(code == 0, f"{child} records nonzero process exit {code}")
            _validate_failure_fields(value, location=child)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _validate_failure_fields(value, location=f"{location}[{index}]")


def _validate_frame_ids(values: Iterable[Any], *, expected_count: int, field: str) -> Tuple[int, ...]:
    ids = tuple(_exact_int(value, field=field) for value in values)
    _require(ids == tuple(range(expected_count)), f"{field} must be contiguous 0..{expected_count - 1}")
    return ids


def _single_path(paths: Iterable[Path], *, label: str) -> Path:
    resolved = list(paths)
    _require(len(resolved) == 1, f"expected exactly one {label}, found {len(resolved)}")
    return resolved[0]


def _validate_episode(
    result_dir: Path,
    *,
    arm: str,
    seed: int,
    row: Mapping[str, Any],
    cell: str,
) -> Dict[str, int]:
    event_path = _single_path(result_dir.glob("event_logs/event_log_*.json"), label=f"{cell} event log")
    reasoning_path = _single_path(result_dir.glob("ep_*/*_reasoning_records.json"), label=f"{cell} reasoning trace")
    physical_path = _single_path(result_dir.glob("ep_*/*_physical_frames.json"), label=f"{cell} physical trace")
    event_payload = _load_json(event_path, allow_nonfinite=True)
    reasoning_payload = _load_json(reasoning_path, allow_nonfinite=True)
    physical_payload = _load_json(physical_path, allow_nonfinite=True)
    _require(isinstance(event_payload, dict), f"{cell}: event log must be an object")
    _require(isinstance(reasoning_payload, dict), f"{cell}: reasoning trace must be an object")
    _require(isinstance(physical_payload, dict), f"{cell}: physical trace must be an object")
    _validate_failure_fields(event_payload, location=f"{cell}.events")
    _validate_failure_fields(reasoning_payload, location=f"{cell}.reasoning")

    episode_id = seed
    for name, payload in (("event", event_payload), ("reasoning", reasoning_payload), ("physical", physical_payload)):
        _require(_exact_int(payload.get("episode_id"), field=f"{cell}.{name}.episode_id") == episode_id, f"{cell}: {name} episode id drift")
    _require(event_payload.get("schema_version") == "rgd_event_log_v2", f"{cell}: event schema drift")
    events = event_payload.get("events")
    records = reasoning_payload.get("analysis_records")
    frames = physical_payload.get("frames")
    _require(isinstance(events, list), f"{cell}: event list missing")
    _require(isinstance(records, list), f"{cell}: reasoning record list missing")
    _require(isinstance(frames, list), f"{cell}: physical frame list missing")
    frame_count = _exact_int(row.get("total_frames"), field=f"{cell}.total_frames")
    _require(frame_count > 0, f"{cell}: completed episode has no frames")
    _require(_exact_int(event_payload.get("event_count"), field=f"{cell}.event_count") == len(events) == frame_count, f"{cell}: event count does not close")
    _require(_exact_int(reasoning_payload.get("record_count"), field=f"{cell}.record_count") == len(records) == frame_count, f"{cell}: reasoning count does not close")
    _require(_exact_int(physical_payload.get("frame_count"), field=f"{cell}.frame_count") == len(frames) == frame_count, f"{cell}: physical count does not close")
    _validate_frame_ids((item.get("frame") for item in events if isinstance(item, dict)), expected_count=frame_count, field=f"{cell}.event.frame")
    _validate_frame_ids((item.get("frame_id") for item in records if isinstance(item, dict)), expected_count=frame_count, field=f"{cell}.reasoning.frame_id")
    _validate_frame_ids((item.get("frame_id") for item in frames if isinstance(item, dict)), expected_count=frame_count, field=f"{cell}.physical.frame_id")
    _require(all(isinstance(item, dict) for item in events), f"{cell}: non-object event")
    _require(all(isinstance(item, dict) for item in records), f"{cell}: non-object reasoning record")
    _require(all(isinstance(item, dict) for item in frames), f"{cell}: non-object physical frame")

    terminal = str(event_payload.get("terminal_cause", "") or "")
    _require(terminal in _FINAL_TERMINAL_CAUSES, f"{cell}: episode is incomplete ({terminal!r})")
    final_event = events[-1]
    _require(_exact_bool(final_event.get("episode_done"), field=f"{cell}.episode_done"), f"{cell}: final event is not terminal")
    _require(final_event.get("terminal_cause") == terminal, f"{cell}: terminal cause drift")
    pending = event_payload.get("pending_releases_dropped_at_episode_end")
    _require(isinstance(pending, list), f"{cell}: pending-release list missing")
    _require(_exact_int(event_payload.get("pending_release_count"), field=f"{cell}.pending_release_count") == len(pending), f"{cell}: pending-release accounting drift")

    attempts = 0
    successes = 0
    failures = 0
    for index, (event, record) in enumerate(zip(events, records)):
        frame_location = f"{cell}.frame[{index}]"
        _require(event.get("rgd_method_version") == METHOD_VERSION, f"{frame_location}: method version drift")
        for field in ("system_used", "slow_reasoning_success", "slow_reasoning_failure_reason"):
            _require(event.get(field) == record.get(field), f"{frame_location}: event/reasoning {field} drift")
        system = str(event.get("system_used", "") or "")
        reasoning_success = _exact_bool(
            event.get("slow_reasoning_success"),
            field=f"{frame_location}.slow_reasoning_success",
        )
        attempted = _exact_bool(event.get("slow_request_attempted"), field=f"{frame_location}.slow_request_attempted")
        valid = _exact_bool(event.get("slow_request_valid_return"), field=f"{frame_location}.slow_request_valid_return")
        failed = _exact_bool(event.get("slow_request_failed"), field=f"{frame_location}.slow_request_failed")
        inferred_attempt = system in {"slow", "fast_after_slow_failure"}
        _require(attempted == inferred_attempt, f"{frame_location}: slow request plan/attempt drift")
        _require(not (valid and failed), f"{frame_location}: slow request marked success and failure")
        _require(not valid or attempted, f"{frame_location}: valid return without a request")
        _require(not failed or attempted, f"{frame_location}: failure without a request")
        reason = str(event.get("slow_reasoning_failure_reason", "") or "").strip()
        if attempted:
            attempts += 1
            _require(system == "slow", f"{frame_location}: fallback after a slow failure")
            _require(valid is True and failed is False, f"{frame_location}: planned slow request did not succeed")
            _require(reasoning_success is True, f"{frame_location}: slow reasoning did not succeed")
            _require(not reason, f"{frame_location}: slow request failure reason is nonempty")
            successes += 1
        else:
            _require(valid is False and failed is False, f"{frame_location}: non-request has a terminal request outcome")
            _require(reasoning_success is False, f"{frame_location}: non-request claims slow success")
            _require(system != "fast_after_slow_failure", f"{frame_location}: hidden fallback after slow failure")
            _require(not reason, f"{frame_location}: failure reason without an attempted request")
        failures += int(failed)
    _require(attempts == successes and failures == 0, f"{cell}: slow request accounting is not all-successful")
    return {"frames": frame_count, "queries": attempts, "successes": successes, "failures": failures}


def _validate_metrics(
    result_dir: Path,
    *,
    arm: str,
    row: Mapping[str, Any],
    counts: Mapping[str, int],
    cell: str,
) -> None:
    metrics_path = result_dir / f"{arm}_rgd_metrics.json"
    payload = _load_json(metrics_path)
    _require(isinstance(payload, dict), f"{cell}: metrics payload must be an object")
    metrics = payload.get("comprehensive_metrics")
    _require(isinstance(metrics, dict), f"{cell}: comprehensive metrics missing")
    _validate_failure_fields(metrics, location=f"{cell}.metrics")
    _require(_exact_int(metrics.get("total_episodes"), field=f"{cell}.total_episodes") == 1, f"{cell}: metrics episode count drift")
    _require(_exact_int(metrics.get("total_frames"), field=f"{cell}.metrics.total_frames") == counts["frames"], f"{cell}: metrics frame count drift")
    _require(metrics.get("experiment_name") == arm, f"{cell}: metrics experiment name drift")
    _require(metrics.get("evaluation_protocol_name") == arm, f"{cell}: metrics protocol name drift")
    _require(metrics.get("single_core_method_name") == "Recoverability-Gated Deliberation", f"{cell}: metrics method name drift")
    _require(metrics.get("primary_evaluation_subject") == "fixed-policy RGD", f"{cell}: metrics evaluation subject drift")
    _require(metrics.get("evaluation_runtime_stable") is True, f"{cell}: runtime is not stable")
    _require(metrics.get("runtime_integrity_clean") is True, f"{cell}: runtime integrity is not clean")
    _require(_finite_float(metrics.get("runtime_integrity_violation_rate"), field=f"{cell}.runtime_integrity_violation_rate") == 0.0, f"{cell}: runtime integrity violation recorded")
    expected_rate = counts["queries"] / counts["frames"]
    expected_success_rate = 1.0 if counts["queries"] else 0.0
    for source_name, source in (("row", row), ("metrics", metrics)):
        observed_rate = _finite_float(source.get("slow_call_rate"), field=f"{cell}.{source_name}.slow_call_rate")
        observed_success = _finite_float(source.get("slow_call_success_rate"), field=f"{cell}.{source_name}.slow_call_success_rate")
        _require(math.isclose(observed_rate, expected_rate, rel_tol=1e-12, abs_tol=1e-12), f"{cell}: {source_name} Q rate differs from traces")
        _require(math.isclose(observed_success, expected_success_rate, rel_tol=1e-12, abs_tol=1e-12), f"{cell}: {source_name} slow success rate differs from traces")
    for field in COMPARISON_HEADLINE_FIELDS:
        row_value = _finite_float(row.get(field), field=f"{cell}.row.{field}")
        metric_value = _finite_float(metrics.get(field), field=f"{cell}.metrics.{field}")
        _require(math.isclose(row_value, metric_value, rel_tol=1e-12, abs_tol=1e-12), f"{cell}: row/metrics {field} drift")


def _scan_auxiliary_failure_artifacts(bundle_root: Path) -> None:
    for path in bundle_root.rglob("*"):
        if not path.is_file():
            continue
        lowered = path.name.lower()
        if lowered.endswith((".err", ".stderr")) or lowered in {"stderr", "stderr.txt"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            _require(not text.strip(), f"nonempty stderr artifact: {path}")
        elif lowered.endswith(".log") or "process" in lowered or "completion" in lowered:
            if path.suffix.lower() == ".json":
                payload = _load_json(path, allow_nonfinite=False)
                _validate_failure_fields(payload, location=str(path))
            else:
                text = path.read_text(encoding="utf-8", errors="replace")
                _require(not _LOG_FAILURE_PATTERN.search(text), f"technical failure recorded in {path}")


def _assert_overall_row(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, location: str
) -> None:
    for field in ("env", "group"):
        _require(observed.get(field) == expected.get(field), f"{location}: {field} drift")
    for field in ("runs", "seed_count"):
        _require(_exact_int(observed.get(field), field=f"{location}.{field}") == int(expected[field]), f"{location}: {field} drift")
    for field in COMPARISON_HEADLINE_FIELDS:
        actual = _finite_float(observed.get(field), field=f"{location}.{field}")
        target = float(expected[field])
        _require(math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12), f"{location}: {field} drift")


def _validate_overall_assets(
    bundle_root: Path, rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    expected = _overall_rows(
        {arm: list(rows_by_arm[arm]) for arm in EXPECTED_ARMS},
        EXPECTED_ARMS,
        (ENVIRONMENT,),
    )
    _require(len(expected) == len(EXPECTED_ARMS), "derived overall table is incomplete")
    header, csv_rows = _load_csv(bundle_root / "overall_group_comparison.csv")
    expected_header = ["env", "group", "runs", "seed_count", *COMPARISON_HEADLINE_FIELDS]
    _require(header == expected_header, "overall comparison CSV schema drift")
    _require(len(csv_rows) == len(expected), "overall comparison CSV row count drift")
    json_payload = _load_json(bundle_root / "overall_group_comparison.json")
    _require(isinstance(json_payload, dict), "overall comparison JSON must be an object")
    _require(tuple(json_payload.get("metrics", []) or []) == tuple(COMPARISON_HEADLINE_FIELDS), "overall comparison metric registry drift")
    json_rows = json_payload.get("rows")
    _require(isinstance(json_rows, list), "overall comparison JSON rows missing")
    _require(len(json_rows) == len(expected), "overall comparison JSON row count drift")
    for index, target in enumerate(expected):
        _assert_overall_row(csv_rows[index], target, location=f"overall.csv[{index}]")
        _require(isinstance(json_rows[index], dict), f"overall.json[{index}] is not an object")
        _assert_overall_row(json_rows[index], target, location=f"overall.json[{index}]")


def _expected_runtime_config(
    *,
    base_cfg: Mapping[str, Any],
    protocol: Mapping[str, Any],
    arm: str,
    seed: int,
    result_dir: Path,
    simulation_duration: Optional[int],
) -> Dict[str, Any]:
    group_cfg = dict((protocol.get("groups", {}) or {})[arm] or {})
    cfg = build_group_config(
        dict(base_cfg),
        arm,
        group_cfg,
        ENVIRONMENT,
        EXPECTED_EPISODES_PER_CELL,
        result_dir,
        dict(protocol),
    )
    _bind_setting_seed(cfg, seed)
    if simulation_duration is not None:
        cfg["simulation_duration"] = int(simulation_duration)
    return _build_runtime_experiment_config(cfg)


def _verified_floor_overlay_for_bundle(
    bundle_root: Path,
    *,
    protocol_path: Path,
    repo_root: Path,
    floor_overlay_path: Optional[Path],
    calibration_manifest_path: Optional[Path],
    calibration_lock_path: Optional[Path],
) -> Any:
    first_result_dir = (
        bundle_root
        / EXPECTED_ARMS[0]
        / SCENARIO
        / f"seed_{EXPECTED_SEEDS[0]:02d}"
    )
    first_manifest = _load_json(first_result_dir / "runtime_manifest.json")
    _require(isinstance(first_manifest, dict), "first runtime manifest must be an object")
    embedded = dict(
        ((first_manifest.get("config", {}) or {}).get("v12_floor_overlay", {}) or {})
    )
    _require(embedded.get("method_version") == METHOD_VERSION, "runtime floor-overlay method drift")
    declared_overlay = _resolve_declared_path(
        embedded.get("floor_overlay_path"), repo_root=repo_root
    )
    declared_calibration = _resolve_declared_path(
        embedded.get("calibration_manifest_path"), repo_root=repo_root
    )
    if floor_overlay_path is not None:
        supplied = (
            Path(floor_overlay_path)
            if Path(floor_overlay_path).is_absolute()
            else repo_root / Path(floor_overlay_path)
        ).resolve()
        _require(supplied == declared_overlay, "CLI floor overlay differs from runtime provenance")
    if calibration_manifest_path is not None:
        supplied = (
            Path(calibration_manifest_path)
            if Path(calibration_manifest_path).is_absolute()
            else repo_root / Path(calibration_manifest_path)
        ).resolve()
        _require(
            supplied == declared_calibration,
            "CLI calibration manifest differs from runtime provenance",
        )
    from tools.v12_floor_overlay import (  # noqa: PLC0415
        DEFAULT_LOCK_PATH,
        load_verified_floor_overlay,
    )

    lock_path = (
        DEFAULT_LOCK_PATH
        if calibration_lock_path is None
        else (
            Path(calibration_lock_path)
            if Path(calibration_lock_path).is_absolute()
            else repo_root / Path(calibration_lock_path)
        )
    )
    verified = load_verified_floor_overlay(
        declared_overlay,
        calibration_manifest_path=declared_calibration,
        protocol_path=protocol_path,
        lock_path=lock_path,
    )
    expected_embedded = {
        **dict(verified.runtime_binding),
        "floor_overlay_path": str(verified.path),
        "calibration_manifest_path": str(verified.calibration_manifest_path),
        "protocol_path": str(verified.protocol_path),
        "calibration_lock_path": str(verified.lock_path),
    }
    _require(embedded == expected_embedded, "runtime floor-overlay binding is not authentic")
    return verified


def verify(
    bundle_or_manifest: Path,
    *,
    protocol_path: Path,
    repo_root: Path = REPO_ROOT,
    floor_overlay_path: Optional[Path] = None,
    calibration_manifest_path: Optional[Path] = None,
    calibration_lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify and summarize one frozen v12 main-table result bundle."""

    repo_root = Path(repo_root).resolve()
    protocol_path = (
        Path(protocol_path)
        if Path(protocol_path).is_absolute()
        else repo_root / Path(protocol_path)
    ).resolve()
    _require(protocol_path.is_file(), f"formal protocol is missing: {protocol_path}")
    bundle_root, manifest_path = _resolve_bundle(Path(bundle_or_manifest))
    bundle_manifest = _load_json(manifest_path)
    _require(isinstance(bundle_manifest, dict), "bundle manifest must be an object")
    protocol = load_formal_protocol(protocol_path)
    _validate_protocol_contract(protocol)
    simulation_duration = _validate_bundle_manifest(
        bundle_manifest,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        protocol_path=protocol_path,
        repo_root=repo_root,
    )
    live_source_hash = build_runtime_source_hash(repo_root)
    base_cfg = load_formal_base_config(protocol, repo_root / "config.yaml")
    verified_floor_overlay = _verified_floor_overlay_for_bundle(
        bundle_root,
        protocol_path=protocol_path,
        repo_root=repo_root,
        floor_overlay_path=floor_overlay_path,
        calibration_manifest_path=calibration_manifest_path,
        calibration_lock_path=calibration_lock_path,
    )
    from tools.v12_floor_overlay import apply_floor_overlay  # noqa: PLC0415

    base_cfg = apply_floor_overlay(base_cfg, verified_floor_overlay)

    root_dirs = {path.name for path in bundle_root.iterdir() if path.is_dir()}
    unexpected_dirs = root_dirs - set(EXPECTED_ARMS)
    _require(not unexpected_dirs, f"unexpected directories in main bundle: {sorted(unexpected_dirs)}")
    rows_by_arm: Dict[str, List[Dict[str, str]]] = {}
    arm_reports: List[Dict[str, Any]] = []
    seen_global: set = set()
    for arm in EXPECTED_ARMS:
        arm_dir = bundle_root / arm
        _require(arm_dir.is_dir(), f"missing arm directory: {arm_dir}")
        rows_path = arm_dir / f"{arm}_run_rows.csv"
        _, rows = _load_csv(rows_path)
        _require(len(rows) == len(EXPECTED_SEEDS), f"{arm}: expected 30 run rows, found {len(rows)}")
        keys: List[Tuple[str, str, int]] = []
        rows_by_seed: Dict[int, Dict[str, str]] = {}
        for index, row in enumerate(rows):
            seed = _exact_int(row.get("seed_idx"), field=f"{arm}.row[{index}].seed_idx")
            key = (str(row.get("group", "") or ""), str(row.get("env", "") or ""), seed)
            _require(key not in seen_global, f"duplicate arm/environment/seed row: {key}")
            seen_global.add(key)
            keys.append(key)
            _require(seed not in rows_by_seed, f"{arm}: duplicate seed row {seed}")
            rows_by_seed[seed] = row
        expected_keys = {(arm, ENVIRONMENT, seed) for seed in EXPECTED_SEEDS}
        _require(set(keys) == expected_keys, f"{arm}: missing or unexpected arm/environment/seed rows")
        rows_by_arm[arm] = [rows_by_seed[seed] for seed in EXPECTED_SEEDS]

        seed_dirs = {path.name for path in (arm_dir / SCENARIO).glob("seed_*") if path.is_dir()}
        expected_seed_dirs = {f"seed_{seed:02d}" for seed in EXPECTED_SEEDS}
        _require(seed_dirs == expected_seed_dirs, f"{arm}: seed directory grid drift")
        arm_queries = 0
        arm_frames = 0
        for seed in EXPECTED_SEEDS:
            row = rows_by_seed[seed]
            cell = f"{arm}/{ENVIRONMENT}/seed_{seed}"
            _require(row.get("group") == arm, f"{cell}: row arm drift")
            _require(row.get("env") == ENVIRONMENT, f"{cell}: row environment drift")
            _require(_exact_int(row.get("episodes_run"), field=f"{cell}.episodes_run") == 1, f"{cell}: expected one complete episode")
            _require(_exact_int(row.get("fixed_seed_override"), field=f"{cell}.fixed_seed_override") == seed, f"{cell}: row fixed seed drift")
            _require(_exact_int(row.get("seed_start"), field=f"{cell}.seed_start") == seed, f"{cell}: row seed start drift")
            _require(_exact_int(row.get("requested_seed_start"), field=f"{cell}.requested_seed_start") == EXPECTED_SEEDS[0], f"{cell}: requested seed-block start drift")
            result_dir = (arm_dir / SCENARIO / f"seed_{seed:02d}").resolve()
            _require(
                _resolve_declared_path(row.get("result_dir"), repo_root=repo_root) == result_dir,
                f"{cell}: CSV result_dir does not identify the canonical cell directory",
            )
            runtime_manifest = _load_json(result_dir / "runtime_manifest.json")
            snapshot = _load_json(result_dir / "experiment_snapshot.json")
            _require(isinstance(runtime_manifest, dict), f"{cell}: runtime manifest must be an object")
            _require(isinstance(snapshot, dict), f"{cell}: snapshot must be an object")
            _validate_backend(runtime_manifest, cell=cell)
            expected_config = _expected_runtime_config(
                base_cfg=base_cfg,
                protocol=protocol,
                arm=arm,
                seed=seed,
                result_dir=result_dir,
                simulation_duration=simulation_duration,
            )
            _validate_manifest_snapshot_contract(
                runtime_manifest,
                snapshot,
                arm=arm,
                seed=seed,
                result_dir=result_dir,
                protocol_path=protocol_path,
                repo_root=repo_root,
                expected_config=expected_config,
                expected_group_cfg=dict((protocol.get("groups", {}) or {})[arm] or {}),
                cell=cell,
            )
            _validate_identity(
                row,
                runtime_manifest,
                snapshot,
                live_source_hash=live_source_hash,
                cell=cell,
            )
            counts = _validate_episode(result_dir, arm=arm, seed=seed, row=row, cell=cell)
            _validate_metrics(result_dir, arm=arm, row=row, counts=counts, cell=cell)
            arm_queries += counts["queries"]
            arm_frames += counts["frames"]
        if arm in ZERO_QUERY_ARMS:
            _require(arm_queries == 0, f"{arm}: Fast-only must have Q=0")
        else:
            _require(arm_queries > 0, f"{arm}: Q=0 is permitted only for always_fast")
        arm_reports.append(
            {
                "arm": arm,
                "episodes": len(EXPECTED_SEEDS),
                "frames": arm_frames,
                "queries": arm_queries,
                "successful_queries": arm_queries,
                "failed_queries": 0,
            }
        )

    _require(len(seen_global) == len(EXPECTED_ARMS) * len(EXPECTED_SEEDS), "global main-result matrix is incomplete")
    _validate_overall_assets(bundle_root, rows_by_arm)
    _scan_auxiliary_failure_artifacts(bundle_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "accepted": True,
        "method_version": METHOD_VERSION,
        "environment": ENVIRONMENT,
        "seeds": list(EXPECTED_SEEDS),
        "episodes": len(EXPECTED_ARMS) * len(EXPECTED_SEEDS),
        "source_hash": live_source_hash,
        "arms": arm_reports,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed verifier for the paper-facing identifiable-gate v12 main result bundle."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Completed bundle directory or its result_bundle_manifest.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="Historical v12 protocol snapshot associated with the result bundle",
    )
    parser.add_argument(
        "--floor-overlay",
        type=Path,
        default=None,
        help="Optional explicit overlay path; must equal runtime provenance",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=None,
        help="Optional explicit calibration manifest; must equal runtime provenance",
    )
    parser.add_argument(
        "--calibration-lock",
        type=Path,
        default=None,
        help="Optional explicit calibration lock; defaults to the canonical lock",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        report = verify(
            args.bundle,
            protocol_path=args.protocol,
            floor_overlay_path=args.floor_overlay,
            calibration_manifest_path=args.calibration_manifest,
            calibration_lock_path=args.calibration_lock,
        )
    except (AcceptanceError, FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
