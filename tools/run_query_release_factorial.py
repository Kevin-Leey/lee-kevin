"""Run the paired query-gate x release-guard factorial replay.

The runner first freezes a gate-independent always-slow proposal stream, then
replays that stream over the same Fast controller under each factorial arm.
Every request is identified by its source frame and is kept separate from its
later asynchronous terminal event.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_ARMS,
    FORMAL_FACTORIAL_ARMS,
    FACTORIAL_EVENT_SCHEMA,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
    FACTORIAL_RUN_SCHEMA,
    FactorialArm,
    ProposalRecord,
    ProposalReplayAgent,
    canonical_proposal_bank_payload,
    configure_factorial_arm,
    proposal_bank_sha256,
)
from tools.run_main_table_runtime import (  # noqa: E402
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
    resolve_policy_execution_horizon,
    validate_policy_execution_horizon,
)


DEFAULT_SOURCE = Path(
    "results/tvt_final_20260721/main_identifiable_v12_diagnostic/formal_run/"
    "main_v12_20260721/always_slow/highway"
)
DEFAULT_PROPOSAL_SOURCE_POLICY = "scheduled_always_slow"
NATURAL_RGD_PROPOSAL_SOURCE_POLICY = "natural_rgd_issued"
LEGACY_PROPOSAL_SOURCE_POLICY = "legacy_gate_positive_diagnostic"
DISTINCT_ACTION_METRIC_STAGE = (
    "post_release_guard_pre_final_safety_projection"
)
_VALID_OUTCOMES = frozenset({"valid", "timeout", "failure"})
_TERMINAL_FLAGS = (
    "closed_loop_latency_release_event",
    "closed_loop_latency_timeout_event",
    "closed_loop_latency_failure_event",
)
_POLICY_FREQUENCY_HZ = 10.0
V13_PROTOCOL_NAME = "rgd_tvt_action_aligned_release_v13"
FORMAL_FACTORIAL_DESIGN = "matched_five_arm_query_release_with_fast_only"
FORMAL_PROPOSAL_SOURCE_COHORTS = {
    "main": tuple(range(5000, 5030)),
    "mechanism": tuple(range(6000, 6020)),
}
FORMAL_PROPOSAL_SOURCE_VERSIONS = {
    "method_version": "action_aligned_release_gate_v13",
    "query_gate_method_version": "identifiable_gate_v12",
    "release_contract_version": "action_cost_alignment_v2",
}
FORMAL_PROPOSAL_SOURCE_EXECUTION = {
    "episode_duration_s": 30.0,
    "policy_frequency_hz": 10.0,
    "simulation_frequency_hz": 10.0,
    "expected_policy_steps": 300.0,
}
FORMAL_PROPOSAL_SOURCE_GROUP = "always_slow"
NATURAL_RGD_PROPOSAL_SOURCE_GROUP = "rgd_fixed_policy"
FORMAL_PROPOSAL_SOURCE_ENV = "highway-v0"
FORMAL_PROPOSAL_SOURCE_SCENARIO = "highway"
FORMAL_PROPOSAL_SOURCE_PROVIDER = "siliconflow"
FORMAL_PROPOSAL_SOURCE_MODEL = "Qwen/Qwen3-8B"
FORMAL_PROPOSAL_BUDGET = 6
FORMAL_PROPOSAL_MIN_FRAME_GAP = 21
_FORMAL_SOURCE_IDENTITY_FIELDS = (
    "protocol_id",
    "protocol_hash",
    "config_hash",
    "source_hash",
)


def _proposal_source_group(source_policy: str) -> str:
    policy = str(source_policy)
    if policy == DEFAULT_PROPOSAL_SOURCE_POLICY:
        return FORMAL_PROPOSAL_SOURCE_GROUP
    if policy == NATURAL_RGD_PROPOSAL_SOURCE_POLICY:
        return NATURAL_RGD_PROPOSAL_SOURCE_GROUP
    raise ValueError(f"unsupported formal proposal source policy: {policy!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _strict_int(value: Any, field: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not math.isfinite(numeric) or int(numeric) != numeric:
        raise ValueError(f"{field} must be an integer")
    result = int(numeric)
    if nonnegative and result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _exact_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{field} must be boolean")
    return bool(value)


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing file to hash: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any], *, allow_nan: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=allow_nan,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _sanitize_nonfinite_json(value: Any) -> Any:
    """Replace non-finite floats recursively so formal JSON stays RFC-compliant."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _sanitize_nonfinite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_nonfinite_json(item) for item in value]
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError(f"inconsistent CSV fields for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _latency_seconds_to_policy_steps(
    latency_seconds: Any,
    *,
    policy_frequency_hz: float = _POLICY_FREQUENCY_HZ,
) -> int:
    """Convert a measured response duration into a conservative frame delay."""
    seconds = _finite_float(latency_seconds, "latency seconds")
    frequency = _finite_float(policy_frequency_hz, "policy frequency")
    if seconds < 0.0:
        raise ValueError("latency seconds must be nonnegative")
    if frequency <= 0.0:
        raise ValueError("policy frequency must be positive")
    # Subtract a tiny numerical tolerance so an exact integral duration stays
    # on its natural frame boundary after binary floating-point conversion.
    return max(0, int(math.ceil(seconds * frequency - 1e-12)))


def _stress_assignment(request_id: str) -> tuple[int, str]:
    """Return a stable latency/outcome stress assignment for one request ID."""
    identifier = str(request_id or "")
    if not identifier:
        raise ValueError("stress assignment requires a nonempty request ID")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    quantile = int.from_bytes(digest[:8], "big") / float(1 << 64)
    # Stress changes the response stream, not the route-specific gate field.
    return 22, "timeout" if quantile >= 0.75 else "valid"


def _stable_quantile(*parts: Any) -> float:
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _named_latency_assignment(
    profile: str,
    row: Mapping[str, Any],
    *,
    median_steps: int,
) -> tuple[int, str]:
    """Return one deterministic service-perturbation assignment.

    All profiles preserve the frozen response content.  Only response timing
    and, for the drop profile, the terminal transport outcome are varied.
    """
    name = str(profile)
    request_id = str(row.get("request_id", "") or "")
    seed = _strict_int(row.get("seed"), "profile seed", nonnegative=True)
    base = max(0, int(median_steps))
    if name == "jitter":
        offsets = (-8, -4, 0, 4, 8)
        index = min(len(offsets) - 1, int(_stable_quantile("jitter", request_id) * len(offsets)))
        return max(0, base + offsets[index]), str(row["outcome"])
    if name == "burst":
        in_burst = _stable_quantile("burst", seed) < 0.35
        return base + (15 if in_burst else 0), str(row["outcome"])
    if name == "drop":
        dropped = _stable_quantile("drop", request_id) < 0.25
        return base + (10 if dropped else 0), "timeout" if dropped else str(row["outcome"])
    if name == "out_of_order":
        try:
            ordinal = int(request_id.rsplit(":", 1)[-1])
        except ValueError as exc:
            raise ValueError("out-of-order profile requires canonical request ordinals") from exc
        return max(0, base + (10 if ordinal % 2 == 0 else -10)), str(row["outcome"])
    raise ValueError(f"unsupported named latency profile: {profile!r}")


def _empirical_assignment(request_id: str, sample_steps: Sequence[Any]) -> int:
    """Map a request deterministically onto a canonical empirical delay sample."""
    identifier = str(request_id or "")
    if not identifier:
        raise ValueError("empirical assignment requires a nonempty request ID")
    normalized = tuple(
        _strict_int(value, "empirical latency steps", nonnegative=True)
        for value in sample_steps
    )
    if not normalized:
        raise ValueError("empirical latency profile has no samples")
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    return int(normalized[int.from_bytes(digest[:8], "big") % len(normalized)])


def _build_empirical_latency_profile(
    samples: Iterable[tuple[Any, Any, Any]],
) -> Dict[str, Any]:
    """Canonicalize source response latencies for reproducible replay."""
    normalized = []
    for raw_seed, raw_frame, raw_seconds in samples:
        seed = _strict_int(raw_seed, "latency sample seed", nonnegative=True)
        frame = _strict_int(raw_frame, "latency sample frame", nonnegative=True)
        seconds = _finite_float(raw_seconds, "latency sample seconds")
        if seconds < 0.0:
            raise ValueError("latency sample seconds must be nonnegative")
        normalized.append(
            {
                "seed": seed,
                "source_frame": frame,
                "latency_seconds": seconds,
                "latency_steps": _latency_seconds_to_policy_steps(seconds),
            }
        )
    normalized.sort(key=lambda row: (row["seed"], row["source_frame"]))
    identities = [(row["seed"], row["source_frame"]) for row in normalized]
    if len(identities) != len(set(identities)):
        raise ValueError("empirical latency profile has duplicate source frames")
    if not normalized:
        raise ValueError("empirical latency profile has no samples")
    payload = {
        "policy_frequency_hz": _POLICY_FREQUENCY_HZ,
        "samples": normalized,
    }
    return {
        "schema": "rgd_empirical_latency_profile_v1",
        "policy_frequency_hz": _POLICY_FREQUENCY_HZ,
        "sample_count": len(normalized),
        "profile_sha256": _sha256_json(payload),
        "samples": normalized,
        "_sample_steps": tuple(row["latency_steps"] for row in normalized),
    }


def _factorial_group_config(
    protocol: Mapping[str, Any],
    arm: FactorialArm,
    *,
    predicted_latency_s: Any,
) -> Dict[str, Any]:
    """Build an always-Fast inner controller configuration for one arm."""
    groups = dict(protocol.get("groups", {}) or {})
    source_group = dict(groups.get("always_fast", {}) or {})
    if not source_group:
        raise ValueError("factorial protocol requires an always_fast group")
    predicted = _finite_float(predicted_latency_s, "predicted latency")
    if predicted < 0.0:
        raise ValueError("predicted latency must be nonnegative")
    result = copy.deepcopy(source_group)
    overrides = copy.deepcopy(dict(result.get("runtime_overrides", {}) or {}))
    replay = copy.deepcopy(dict(overrides.get("closed_loop_latency_replay", {}) or {}))
    replay.update(
        {
            "enable": True,
            "target_systems": ["slow"],
            "extra_latency_s": float(predicted),
            "delay_steps": _latency_seconds_to_policy_steps(predicted),
            # The replay bank contains authenticated responses from the live
            # slow executor.  Gate evaluation must therefore treat execution
            # as available even though no online client is instantiated.
            "proposal_backed_execution_available": True,
        }
    )
    overrides.update(
        {
            "protocol_name": f"factorial_{arm.name}",
            "system_routing": {"simple": "fast", "complex": "fast"},
            "asynchronous_slow_path": {
                "enable": False,
                "min_release_frames": 1,
            },
            "closed_loop_latency_replay": replay,
            "factorial_predicted_latency_s": float(predicted),
            "factorial_predicted_latency_steps": _latency_seconds_to_policy_steps(predicted),
        }
    )
    result["runtime_overrides"] = overrides
    result.setdefault("id", f"factorial_{arm.name}")
    return result


def _contract_seed_block(contract: Mapping[str, Any], field: str) -> tuple[int, ...]:
    value = contract.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must declare start/end/count")
    start = _strict_int(value.get("start"), f"{field}.start", nonnegative=True)
    end = _strict_int(value.get("end"), f"{field}.end", nonnegative=True)
    count = _strict_int(value.get("count"), f"{field}.count", nonnegative=True)
    if end < start or count != end - start + 1:
        raise ValueError(f"{field} is inconsistent")
    return tuple(range(start, end + 1))


def _validate_formal_factorial_preflight(
    *,
    protocol: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    design: str,
    seeds: Sequence[int],
    latency_profile: str,
    fixed_latency_steps: Optional[int],
    result_root: Path,
    source_policy: str = NATURAL_RGD_PROPOSAL_SOURCE_POLICY,
):
    """Validate the complete v13 five-arm contract before writing artifacts."""

    if str(protocol.get("protocol_name", "") or "") != V13_PROTOCOL_NAME:
        return None
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    contract = dict(submission.get("query_release_factorial", {}) or {})
    if contract.get("design") != FORMAL_FACTORIAL_DESIGN:
        raise ValueError("formal five-arm protocol design drift")
    if str(design) != "five_arm":
        raise ValueError("formal v13 factorial requires --design five_arm")
    if tuple(str(value) for value in contract.get("arms", ())) != tuple(
        arm.name for arm in FORMAL_FACTORIAL_ARMS
    ):
        raise ValueError("formal five-arm protocol arm order drift")
    if tuple(int(seed) for seed in seeds) != _contract_seed_block(
        contract, "seed_range"
    ):
        raise ValueError("formal five-arm seed cohort drift")
    if str(latency_profile) != str(contract.get("latency_profile", "") or ""):
        raise ValueError("formal five-arm latency profile drift")
    declared_fixed = contract.get("fixed_delay_steps")
    if fixed_latency_steps != declared_fixed:
        raise ValueError("formal five-arm fixed-delay contract drift")
    if contract.get("candidate_source_policy") != str(source_policy):
        raise ValueError("formal five-arm candidate-source contract drift")
    if contract.get("proposal_source_group") != _proposal_source_group(source_policy):
        raise ValueError("formal five-arm proposal-source group drift")
    group_cfg = _factorial_group_config(
        protocol,
        FORMAL_FACTORIAL_ARMS[0],
        predicted_latency_s=base_cfg.get("rgd_predicted_slow_latency_s", 0.0),
    )
    cfg = build_group_config(
        base_cfg,
        "factorial_full",
        group_cfg,
        "highway-v0",
        1,
        Path(result_root) / "full" / f"seed_{int(seeds[0])}",
        protocol,
    )
    cfg = configure_factorial_arm(cfg, FORMAL_FACTORIAL_ARMS[0])
    return validate_policy_execution_horizon(
        cfg,
        dict(contract.get("execution_contract", {}) or {}),
        context="formal five-arm factorial",
    )


def _read_json_object(path: Path, *, allow_nan: bool = False) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON artifact: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=None if allow_nan else reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(payload)


def _source_paths(source_root: Path, seed: int) -> tuple[Path, Path, Path]:
    """Resolve the immutable source artifacts for one always-slow seed."""
    root = Path(source_root).resolve()
    seed_value = _strict_int(seed, "source seed", nonnegative=True)
    seed_dir = root / f"seed_{seed_value}"
    if not seed_dir.is_dir() and root.name == f"seed_{seed_value}":
        seed_dir = root
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"missing source seed directory: {seed_dir}")
    event_paths = sorted((seed_dir / "event_logs").glob("event_log_*.json"))
    reasoning_paths = sorted(seed_dir.glob("ep_*/*_reasoning_records.json"))
    snapshot_path = seed_dir / "experiment_snapshot.json"
    if len(event_paths) != 1:
        raise RuntimeError(
            f"seed {seed_value}: expected exactly one source event log, found {len(event_paths)}"
        )
    if len(reasoning_paths) != 1:
        raise RuntimeError(
            f"seed {seed_value}: expected exactly one source reasoning trace, found {len(reasoning_paths)}"
        )
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"seed {seed_value}: experiment snapshot is missing")
    return event_paths[0], reasoning_paths[0], snapshot_path


def _formal_int(value: Any, field: str, *, nonnegative: bool = False) -> int:
    try:
        return _strict_int(value, field, nonnegative=nonnegative)
    except ValueError as exc:
        raise RuntimeError(f"{field} must be an integer") from exc


def _formal_seed_partition(seeds: Sequence[int]) -> tuple[str, tuple[int, ...]]:
    seed_values = tuple(
        _formal_int(seed, "formal proposal source seed", nonnegative=True)
        for seed in seeds
    )
    for partition, expected in FORMAL_PROPOSAL_SOURCE_COHORTS.items():
        if seed_values == expected:
            return partition, seed_values
    raise RuntimeError(
        "formal proposal source seed cohort must be exactly 5000-5029 "
        "or 6000-6019"
    )


def _validate_formal_execution_fields(
    payload: Mapping[str, Any], *, context: str
) -> None:
    for field, expected in FORMAL_PROPOSAL_SOURCE_EXECUTION.items():
        try:
            observed = _finite_float(payload.get(field), f"{context} {field}")
        except ValueError as exc:
            raise RuntimeError(f"{context} has invalid {field}") from exc
        _require(
            observed == expected,
            f"{context} {field} mismatch: expected {expected:g}, observed {observed:g}",
        )


def _formal_bundle_manifest_path(
    source_root: Path, *, source_group: str
) -> tuple[Path, Path]:
    root = Path(source_root).resolve()
    candidates = [
        parent / "result_bundle_manifest.json"
        for parent in (root, *root.parents)
        if (parent / "result_bundle_manifest.json").is_file()
    ]
    _require(candidates, "formal proposal source has no enclosing result bundle manifest")
    _require(
        len(candidates) == 1,
        "formal proposal source has ambiguous enclosing result bundle manifests",
    )
    manifest_path = candidates[0]
    bundle_root = manifest_path.parent.resolve()
    expected_root = (
        bundle_root
        / str(source_group)
        / FORMAL_PROPOSAL_SOURCE_SCENARIO
    ).resolve()
    _require(
        root == expected_root,
        "formal proposal source root must identify the always_slow/highway cell matrix",
    )
    return bundle_root, manifest_path


def _read_csv_objects(path: Path) -> list[Dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise RuntimeError(f"CSV artifact has no header: {path}")
        return [dict(row) for row in reader]


def _validate_formal_source_cell_payload(
    payload: Mapping[str, Any], *, seed: int, artifact: str, source_group: str
) -> Mapping[str, Any]:
    context = f"seed {seed} {artifact}"
    _require(
        str(payload.get("protocol_name", "") or "")
        == str(source_group),
        f"{context} protocol name mismatch",
    )
    _require(
        str(payload.get("env_type", "") or "") == FORMAL_PROPOSAL_SOURCE_ENV,
        f"{context} environment mismatch",
    )
    _require(
        _formal_int(payload.get("fixed_seed_override"), f"{context} fixed seed")
        == seed,
        f"{context} fixed seed mismatch",
    )
    _require(
        _formal_int(payload.get("seed_start"), f"{context} seed start") == seed,
        f"{context} seed start mismatch",
    )
    resolved = payload.get("resolved_seeds")
    _require(
        isinstance(resolved, list)
        and tuple(
            _formal_int(value, f"{context} resolved seed", nonnegative=True)
            for value in resolved
        )
        == (seed,),
        f"{context} resolved seed list mismatch",
    )

    config = payload.get("config")
    _require(isinstance(config, Mapping), f"{context} has no runtime config")
    config = dict(config)
    _require(
        str(config.get("protocol_name", "") or "")
        == str(source_group),
        f"{context} config protocol name mismatch",
    )
    _require(
        _formal_int(config.get("protocol_version"), f"{context} protocol version")
        == 13,
        f"{context} protocol version mismatch",
    )
    _require(
        str(config.get("env_type", "") or "") == FORMAL_PROPOSAL_SOURCE_ENV,
        f"{context} config environment mismatch",
    )
    _require(
        _formal_int(config.get("fixed_seed_override"), f"{context} config seed")
        == seed,
        f"{context} config seed mismatch",
    )
    _require(
        _formal_int(config.get("episodes_num"), f"{context} episode count") == 1,
        f"{context} episode count mismatch",
    )
    config_execution = {
        "episode_duration_s": config.get("simulation_duration"),
        "policy_frequency_hz": config.get("policy_frequency"),
        "simulation_frequency_hz": config.get("simulation_frequency"),
        "expected_policy_steps": (
            _formal_int(
                config.get("simulation_duration"),
                f"{context} simulation duration",
            )
            * _formal_int(
                config.get("policy_frequency"),
                f"{context} policy frequency",
            )
        ),
    }
    _validate_formal_execution_fields(config_execution, context=f"{context} config")
    if str(source_group) == FORMAL_PROPOSAL_SOURCE_GROUP:
        routing = config.get("system_routing")
        _require(
            isinstance(routing, Mapping)
            and routing.get("simple") == "slow"
            and routing.get("complex") == "slow",
            f"{context} must force slow routing",
        )

    protocol_manifest = payload.get("protocol_manifest")
    _require(
        isinstance(protocol_manifest, Mapping),
        f"{context} has no protocol manifest",
    )
    _require(
        _formal_int(
            protocol_manifest.get("protocol_version"),
            f"{context} protocol manifest version",
        )
        == 13,
        f"{context} protocol manifest version mismatch",
    )
    _require(
        str(protocol_manifest.get("selected_group", "") or "")
        == str(source_group),
        f"{context} selected group mismatch",
    )
    _require(
        str(protocol_manifest.get("selected_environment", "") or "")
        == FORMAL_PROPOSAL_SOURCE_ENV,
        f"{context} selected environment mismatch",
    )

    backend = payload.get("llm_backend")
    _require(isinstance(backend, Mapping), f"{context} has no LLM backend")
    _require(
        str(backend.get("provider", "") or "")
        == FORMAL_PROPOSAL_SOURCE_PROVIDER,
        f"{context} LLM provider mismatch",
    )
    _require(
        str(backend.get("requested_model", "") or "")
        == FORMAL_PROPOSAL_SOURCE_MODEL,
        f"{context} requested LLM model mismatch",
    )
    _require(
        str(backend.get("resolved_chat_model", "") or "")
        == FORMAL_PROPOSAL_SOURCE_MODEL,
        f"{context} resolved LLM model mismatch",
    )
    return config


def _validate_formal_proposal_source_bundle(
    source_root: Path,
    seeds: Sequence[int],
    *,
    source_group: str = FORMAL_PROPOSAL_SOURCE_GROUP,
) -> Dict[str, Any]:
    """Close a proposal stream to its formal bundle, cell, and run-row identity."""
    expected_partition, seed_values = _formal_seed_partition(seeds)
    root = Path(source_root).resolve()
    bundle_root, manifest_path = _formal_bundle_manifest_path(
        root, source_group=source_group
    )
    manifest = _read_json_object(manifest_path)
    _require(
        str(manifest.get("bundle_kind", "") or "") == "formal_run",
        "formal proposal source bundle kind mismatch",
    )
    _require(
        str(manifest.get("partition", "") or "") == expected_partition,
        "formal proposal source partition/cohort mismatch",
    )
    for field, expected in FORMAL_PROPOSAL_SOURCE_VERSIONS.items():
        _require(
            str(manifest.get(field, "") or "") == expected,
            f"formal proposal source {field} mismatch",
        )
    groups = manifest.get("groups")
    _require(
        isinstance(groups, list) and str(source_group) in groups,
        f"formal proposal source bundle is missing {source_group}",
    )
    matrix = manifest.get("group_env_matrix")
    _require(
        isinstance(matrix, Mapping)
        and list(matrix.get(str(source_group), []) or [])
        == [FORMAL_PROPOSAL_SOURCE_ENV],
        "formal proposal source always_slow environment matrix mismatch",
    )
    _require(
        _formal_int(manifest.get("seeds"), "formal proposal source seed count")
        == len(seed_values),
        "formal proposal source seed count mismatch",
    )
    _require(
        _formal_int(manifest.get("seed_start"), "formal proposal source seed start")
        == seed_values[0],
        "formal proposal source seed start mismatch",
    )
    labels = manifest.get("seed_labels")
    _require(
        isinstance(labels, list)
        and tuple(
            _formal_int(value, "formal proposal source seed label", nonnegative=True)
            for value in labels
        )
        == seed_values,
        "formal proposal source seed labels mismatch",
    )
    _require(
        _formal_int(manifest.get("episodes"), "formal proposal source episodes") == 1,
        "formal proposal source episode count mismatch",
    )
    _require(
        manifest.get("seed_value") is None,
        "formal proposal source must use a seed block rather than seed_value",
    )
    _validate_formal_execution_fields(manifest, context="formal proposal source bundle")
    try:
        duration = _finite_float(
            manifest.get("simulation_duration"),
            "formal proposal source simulation duration",
        )
    except ValueError as exc:
        raise RuntimeError("formal proposal source has invalid simulation duration") from exc
    _require(duration == 30.0, "formal proposal source simulation duration mismatch")
    horizon_matrix = manifest.get("execution_horizon_by_group_env")
    _require(
        isinstance(horizon_matrix, Mapping),
        "formal proposal source has no execution horizon matrix",
    )
    group_horizons = horizon_matrix.get(str(source_group))
    _require(
        isinstance(group_horizons, Mapping),
        "formal proposal source has no always_slow execution horizon",
    )
    cell_horizon = group_horizons.get(FORMAL_PROPOSAL_SOURCE_ENV)
    _require(
        isinstance(cell_horizon, Mapping),
        "formal proposal source has no highway-v0 execution horizon",
    )
    _validate_formal_execution_fields(
        cell_horizon, context="formal proposal source always_slow/highway-v0"
    )

    run_rows_path = (
        bundle_root
        / str(source_group)
        / f"{source_group}_run_rows.csv"
    )
    rows = _read_csv_objects(run_rows_path)
    _require(
        len(rows) == len(seed_values),
        "formal proposal source run-row count mismatch",
    )
    rows_by_seed: Dict[int, Dict[str, str]] = {}
    for row in rows:
        _require(
            str(row.get("group", "") or "") == str(source_group),
            "formal proposal source run-row group mismatch",
        )
        _require(
            str(row.get("env", "") or "") == FORMAL_PROPOSAL_SOURCE_ENV,
            "formal proposal source run-row environment mismatch",
        )
        row_seed = _formal_int(
            row.get("seed_idx"), "formal proposal source run-row seed", nonnegative=True
        )
        _require(
            row_seed not in rows_by_seed,
            f"formal proposal source has duplicate run row for seed {row_seed}",
        )
        rows_by_seed[row_seed] = row
    _require(
        tuple(sorted(rows_by_seed)) == seed_values,
        "formal proposal source run-row seed cohort mismatch",
    )

    cells: Dict[int, Dict[str, Path]] = {}
    for seed in seed_values:
        row = rows_by_seed[seed]
        for field in ("fixed_seed_override", "seed_start"):
            _require(
                _formal_int(row.get(field), f"seed {seed} run-row {field}") == seed,
                f"seed {seed} run-row {field} mismatch",
            )
        _require(
            _formal_int(
                row.get("requested_seed_start"),
                f"seed {seed} run-row requested seed start",
            )
            == seed_values[0],
            f"seed {seed} run-row requested seed start mismatch",
        )
        _require(
            _formal_int(row.get("episodes_run"), f"seed {seed} run-row episodes")
            == 1,
            f"seed {seed} run-row episode count mismatch",
        )
        seed_dir = root / f"seed_{seed}"
        _require(seed_dir.is_dir(), f"seed {seed}: formal source cell is missing")
        declared_result_dir = str(row.get("result_dir", "") or "")
        _require(
            bool(declared_result_dir)
            and Path(declared_result_dir).resolve() == seed_dir.resolve(),
            f"seed {seed}: run-row result directory mismatch",
        )
        runtime_path = seed_dir / "runtime_manifest.json"
        snapshot_path = seed_dir / "experiment_snapshot.json"
        runtime = _read_json_object(runtime_path)
        snapshot = _read_json_object(snapshot_path)
        runtime_config = _validate_formal_source_cell_payload(
            runtime,
            seed=seed,
            artifact="runtime manifest",
            source_group=source_group,
        )
        snapshot_config = _validate_formal_source_cell_payload(
            snapshot,
            seed=seed,
            artifact="experiment snapshot",
            source_group=source_group,
        )
        _require(
            runtime_config == snapshot_config,
            f"seed {seed}: runtime/snapshot config mismatch",
        )
        _require(
            runtime.get("llm_backend") == snapshot.get("llm_backend"),
            f"seed {seed}: runtime/snapshot LLM backend mismatch",
        )
        _require(
            snapshot.get("seeds_used") == [seed],
            f"seed {seed}: experiment snapshot seed list mismatch",
        )
        for field in _FORMAL_SOURCE_IDENTITY_FIELDS:
            values = (runtime.get(field), snapshot.get(field), row.get(field))
            _require(
                all(isinstance(value, str) and bool(value) for value in values)
                and len(set(values)) == 1,
                f"seed {seed}: {field} identity mismatch across source artifacts",
            )
            if field != "protocol_id":
                value = str(values[0])
                _require(
                    len(value) == 64
                    and all(char in "0123456789abcdef" for char in value),
                    f"seed {seed}: malformed {field} identity",
                )
        cells[seed] = {
            "runtime_manifest_path": runtime_path,
            "experiment_snapshot_path": snapshot_path,
        }
    return {
        "bundle_root": bundle_root,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "run_rows_path": run_rows_path,
        "partition": expected_partition,
        "cells": cells,
    }


def _validate_proposal_source(
    snapshot_path: Path,
    *,
    seed: int,
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> None:
    """Verify that a proposal source is independent of the evaluated gate."""
    if str(source_policy) not in {
        DEFAULT_PROPOSAL_SOURCE_POLICY,
        NATURAL_RGD_PROPOSAL_SOURCE_POLICY,
    }:
        if str(source_policy) == LEGACY_PROPOSAL_SOURCE_POLICY:
            return
        raise RuntimeError(f"unsupported proposal source policy: {source_policy!r}")
    snapshot = _read_json_object(snapshot_path)
    expected_seed = _strict_int(seed, "source seed", nonnegative=True)
    observed_seed = snapshot.get("fixed_seed_override", snapshot.get("seed_start"))
    try:
        source_seed = _strict_int(observed_seed, "source fixed seed", nonnegative=True)
    except ValueError as exc:
        raise RuntimeError("source seed provenance mismatch") from exc
    if source_seed != expected_seed:
        raise RuntimeError("source seed provenance mismatch")
    config = snapshot.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("source snapshot has no configuration")
    expected_group = _proposal_source_group(source_policy)
    if str(config.get("protocol_name", "") or "") != expected_group:
        if str(source_policy) == DEFAULT_PROPOSAL_SOURCE_POLICY:
            raise RuntimeError("gate-independent proposal source must be always_slow")
        raise RuntimeError("natural RGD proposal source must be rgd_fixed_policy")
    if str(source_policy) == DEFAULT_PROPOSAL_SOURCE_POLICY:
        routing = config.get("system_routing")
        if not isinstance(routing, Mapping) or (
            routing.get("simple") != "slow" or routing.get("complex") != "slow"
        ):
            raise RuntimeError("gate-independent proposal source must force slow routing")


def _query_event(
    event: Mapping[str, Any],
    *,
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> bool:
    """Select source-frame request attempts without selecting release frames."""
    row = dict(event or {})
    if bool(row.get("closed_loop_latency_release_event", False)):
        return False
    if str(source_policy) == DEFAULT_PROPOSAL_SOURCE_POLICY:
        if not bool(row.get("slow_request_attempted", False)):
            return False
        frame = row.get("frame")
        source_frame = row.get("closed_loop_latency_source_frame", frame)
        try:
            return _strict_int(frame, "event frame", nonnegative=True) == _strict_int(
                source_frame, "event source frame", nonnegative=True
            )
        except ValueError:
            return False
    if str(source_policy) == LEGACY_PROPOSAL_SOURCE_POLICY:
        return bool(row.get("closed_loop_latency_eligible", False))
    if str(source_policy) == NATURAL_RGD_PROPOSAL_SOURCE_POLICY:
        return bool(row.get("closed_loop_latency_issuance_event", False))
    raise ValueError(f"unsupported proposal source policy: {source_policy!r}")


@contextlib.contextmanager
def _worker_output_context(*, verbose: bool) -> Iterator[None]:
    """Keep spawned workers silent unless the caller explicitly requests logs."""
    if verbose:
        yield
        return
    previous_disable = logging.root.manager.disable
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        logging.disable(logging.CRITICAL)
        try:
            yield
        finally:
            logging.disable(previous_disable)


def _reasoning_records(path: Path) -> Dict[int, Mapping[str, Any]]:
    payload = _read_json_object(path)
    raw_records = payload.get("analysis_records", payload.get("records", []))
    if not isinstance(raw_records, list):
        raise ValueError(f"reasoning trace records must be a list: {path}")
    records: Dict[int, Mapping[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            continue
        frame = raw.get("frame_id", raw.get("frame"))
        try:
            frame_id = _strict_int(frame, "reasoning frame", nonnegative=True)
        except ValueError:
            continue
        if frame_id in records:
            raise ValueError(f"duplicate reasoning record at frame {frame_id}: {path}")
        records[frame_id] = dict(raw)
    return records


def _event_action(event: Mapping[str, Any], reasoning: Mapping[str, Any]) -> int:
    for source in (event, reasoning):
        for name in (
            "query_state_slow_pre_guard_action",
            "slow_response_action",
            "llm_raw_action",
            "query_state_slow_released_action",
            "post_validation_action",
            "final_action",
            "proposed_action",
            "predicted_action_id",
        ):
            if name not in source or source.get(name) is None:
                continue
            action = _strict_int(source[name], f"source {name}")
            if action in range(5):
                return action
    raise ValueError("source query event does not contain a valid slow action")


def _event_outcome(event: Mapping[str, Any], reasoning: Mapping[str, Any]) -> str:
    for source in (event, reasoning):
        for name in (
            "closed_loop_latency_terminal_response_outcome",
            "closed_loop_latency_response_outcome",
            "factorial_shared_response_outcome",
            "slow_reasoning_failure_reason",
        ):
            value = str(source.get(name, "") or "").strip().lower()
            if value in _VALID_OUTCOMES:
                return value
    if bool(event.get("slow_request_failed", False)):
        return "failure"
    return "valid"


def _event_latency_seconds(event: Mapping[str, Any], reasoning: Mapping[str, Any]) -> float:
    for source in (event, reasoning):
        for name in (
            "slow_response_wall_latency_s",
            "inference_latency",
            "latency_seconds",
            "latency_s",
        ):
            if name not in source or source.get(name) is None:
                continue
            seconds = _finite_float(source[name], f"source {name}")
            if seconds >= 0.0:
                return seconds
    raise ValueError("source query event has no finite nonnegative inference latency")


def _response_text(event: Mapping[str, Any], reasoning: Mapping[str, Any]) -> str:
    for source in (reasoning, event):
        for name in ("full_response", "response", "slow_response"):
            value = source.get(name)
            if isinstance(value, str):
                return value
    return ""


def _native_response_identity(
    event: Mapping[str, Any], reasoning: Mapping[str, Any]
) -> tuple[int, str, str]:
    """Validate and return the action, raw response, and runtime response hash."""
    action = _event_action(event, reasoning)
    response = event.get("slow_response_text")
    if not isinstance(response, str):
        raise ValueError("native terminal event has no raw slow response text")
    reported = str(event.get("closed_loop_latency_response_sha256", "") or "")
    if len(reported) != 64 or any(char not in "0123456789abcdef" for char in reported):
        raise ValueError("native terminal event has no valid response SHA256")
    try:
        confidence = _finite_float(
            event.get("slow_response_confidence"), "native slow response confidence"
        )
    except ValueError as exc:
        raise ValueError("native terminal response identity is incomplete") from exc
    slow_reasoning = event.get("slow_response_reasoning")
    if not isinstance(slow_reasoning, str):
        raise ValueError("native terminal response identity is incomplete")
    identity = {
        "action": int(action),
        "confidence": float(confidence),
        "reasoning": slow_reasoning,
        "response": response,
    }
    observed = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if observed != reported:
        raise ValueError("native terminal response SHA256 does not match its payload")
    return action, response, reported


def _native_source_rows(
    *,
    seed: int,
    payload: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reasoning_by_frame: Mapping[int, Mapping[str, Any]],
    reasoning_path: Path,
    source_policy: str,
) -> Optional[list[Dict[str, Any]]]:
    """Join live issuance and terminal frames into immutable proposal rows."""
    native = any(
        bool(event.get("native_async_slow_path", False))
        or bool(event.get("closed_loop_latency_issuance_event", False))
        or bool(event.get("closed_loop_latency_terminal_event", False))
        for event in events
    )
    if not native:
        return None

    issued: Dict[str, tuple[int, Dict[str, Any]]] = {}
    terminal: Dict[str, tuple[int, Dict[str, Any]]] = {}
    for raw in events:
        event = dict(raw or {})
        frame = _strict_int(event.get("frame"), "source event frame", nonnegative=True)
        if bool(event.get("closed_loop_latency_issuance_event", False)):
            request_id = str(event.get("closed_loop_latency_issued_request_id", "") or "")
            if not request_id or request_id in issued:
                raise RuntimeError(f"seed {seed}: malformed or duplicate native issuance")
            if not _query_event(event, source_policy=source_policy):
                raise RuntimeError(f"seed {seed}: native issuance is not a source-frame request")
            source_frame = _strict_int(
                event.get("closed_loop_latency_source_frame", frame),
                "native source frame",
                nonnegative=True,
            )
            if source_frame != frame:
                raise RuntimeError(f"seed {seed}: native issuance source-frame drift")
            issued[request_id] = (frame, event)
        if bool(event.get("closed_loop_latency_terminal_event", False)):
            request_id = str(event.get("closed_loop_latency_terminal_request_id", "") or "")
            if not request_id or request_id in terminal:
                raise RuntimeError(f"seed {seed}: malformed or duplicate native terminal")
            terminal[request_id] = (frame, event)

    dropped = {
        str(dict(row).get("request_id", "") or "")
        for row in list(payload.get("pending_releases_dropped_at_episode_end", []) or [])
        if isinstance(row, Mapping)
    }
    dropped.discard("")
    if dropped:
        raise RuntimeError(
            f"seed {seed}: native proposal source contains dropped requests: {sorted(dropped)}"
        )
    if not issued and str(source_policy) == NATURAL_RGD_PROPOSAL_SOURCE_POLICY:
        return []
    if not issued:
        raise RuntimeError(f"seed {seed}: native proposal source has no issuance events")
    if set(terminal) != set(issued):
        missing = sorted(set(issued) - set(terminal))
        orphan = sorted(set(terminal) - set(issued))
        raise RuntimeError(
            f"seed {seed}: native issuance/terminal coverage mismatch; "
            f"missing={missing}, orphan={orphan}"
        )

    rows: list[Dict[str, Any]] = []
    ordered = sorted(issued.items(), key=lambda item: (item[1][0], item[0]))
    for ordinal, (source_request_id, (source_frame, issuance)) in enumerate(ordered):
        terminal_frame, terminal_event = terminal[source_request_id]
        if terminal_frame < source_frame:
            raise RuntimeError(f"seed {seed}: native terminal precedes issuance")
        outcome = _event_outcome(
            terminal_event, reasoning_by_frame.get(terminal_frame, {})
        )
        issued_outcome = str(
            issuance.get("closed_loop_latency_issued_response_outcome", "pending")
            or "pending"
        ).lower()
        if issued_outcome not in {"pending", outcome}:
            raise RuntimeError(f"seed {seed}: native issuance outcome drift")
        terminal_reasoning = dict(reasoning_by_frame.get(terminal_frame, {}) or {})
        if outcome == "valid":
            action, response_text, response_sha256 = _native_response_identity(
                terminal_event, terminal_reasoning
            )
        else:
            action = _event_action(issuance, reasoning_by_frame.get(source_frame, {}))
            response_text = ""
            response_sha256 = hashlib.sha256(b"").hexdigest()
        rows.append(
            {
                "seed": int(seed),
                "source_frame": int(source_frame),
                "request_id": f"factorial:{seed}:{source_frame}:{ordinal:02d}",
                "raw_slow_action": int(action),
                "outcome": outcome,
                "response_text": response_text,
                "response_sha256": response_sha256,
                "latency_seconds": _event_latency_seconds(
                    terminal_event, terminal_reasoning
                ),
                "source_artifact": str(reasoning_path),
                "source_request_id": source_request_id,
            }
        )
    return rows


def load_proposal_bank(
    source_root: Path,
    seeds: Sequence[int],
    *,
    latency_profile: str = "frozen",
    fixed_latency_steps: Optional[int] = None,
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> Dict[int, Dict[int, ProposalRecord]]:
    """Freeze source proposals and bind each one to a replay delay/outcome."""
    seed_values = tuple(_strict_int(seed, "proposal seed", nonnegative=True) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("proposal seeds must be nonempty and unique")
    supported_profiles = {
        "frozen",
        "fixed",
        "jitter",
        "burst",
        "drop",
        "out_of_order",
        "stress",
    }
    if str(latency_profile) not in supported_profiles:
        raise ValueError(
            "latency profile must be frozen, fixed, jitter, burst, drop, "
            "out_of_order, or stress"
        )
    fixed_steps = None
    if str(latency_profile) == "fixed":
        fixed_steps = _strict_int(
            fixed_latency_steps,
            "fixed latency steps",
            nonnegative=True,
        )
    elif fixed_latency_steps is not None:
        raise ValueError("fixed latency steps require latency_profile='fixed'")
    staged: list[Dict[str, Any]] = []
    for seed in sorted(seed_values):
        event_path, reasoning_path, snapshot_path = _source_paths(source_root, seed)
        _validate_proposal_source(
            snapshot_path, seed=seed, source_policy=source_policy
        )
        event_payload = _read_json_object(event_path, allow_nan=True)
        events = event_payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"seed {seed}: source event log has no event list")
        reasoning_by_frame = _reasoning_records(reasoning_path)
        native_rows = _native_source_rows(
            seed=seed,
            payload=event_payload,
            events=[dict(event) for event in events if isinstance(event, Mapping)],
            reasoning_by_frame=reasoning_by_frame,
            reasoning_path=reasoning_path,
            source_policy=source_policy,
        )
        if native_rows is not None:
            staged.extend(native_rows)
            continue
        selected = []
        for event in events:
            if not isinstance(event, Mapping) or not _query_event(event, source_policy=source_policy):
                continue
            frame = _strict_int(event.get("frame"), "source event frame", nonnegative=True)
            selected.append((frame, dict(event)))
        if not selected:
            raise RuntimeError(f"seed {seed}: no gate-independent source queries")
        for ordinal, (frame, event) in enumerate(sorted(selected)):
            reasoning = dict(reasoning_by_frame.get(frame, {}) or {})
            text = _response_text(event, reasoning)
            staged.append(
                {
                    "seed": seed,
                    "source_frame": frame,
                    "request_id": f"factorial:{seed}:{frame}:{ordinal:02d}",
                    "raw_slow_action": _event_action(event, reasoning),
                    "outcome": _event_outcome(event, reasoning),
                    "response_text": text,
                    "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "latency_seconds": _event_latency_seconds(event, reasoning),
                    "source_artifact": str(reasoning_path),
                }
            )
    profile = _build_empirical_latency_profile(
        (row["seed"], row["source_frame"], row["latency_seconds"]) for row in staged
    )
    empirical_steps = [
        _latency_seconds_to_policy_steps(row["latency_seconds"]) for row in staged
    ]
    median_steps = int(round(float(median(empirical_steps))))
    bank: Dict[int, Dict[int, ProposalRecord]] = {seed: {} for seed in sorted(seed_values)}
    for row in staged:
        request_id = str(row["request_id"])
        if str(latency_profile) == "stress":
            steps, outcome = _stress_assignment(request_id)
        elif str(latency_profile) == "fixed":
            steps = int(fixed_steps)
            outcome = str(row["outcome"])
        elif str(latency_profile) in {"jitter", "burst", "drop", "out_of_order"}:
            steps, outcome = _named_latency_assignment(
                str(latency_profile),
                row,
                median_steps=median_steps,
            )
        else:
            steps = _latency_seconds_to_policy_steps(row["latency_seconds"])
            outcome = str(row["outcome"])
        record = ProposalRecord(
            seed=int(row["seed"]),
            source_frame=int(row["source_frame"]),
            request_id=request_id,
            raw_slow_action=int(row["raw_slow_action"]),
            latency_steps=int(steps),
            outcome=outcome,
            response_text=str(row["response_text"]),
            response_sha256=str(row["response_sha256"]),
            source_artifact=str(row["source_artifact"]),
        )
        if record.source_frame in bank[record.seed]:
            raise RuntimeError(f"seed {record.seed}: duplicate proposal source frame")
        bank[record.seed][record.source_frame] = record
    _validate_formal_proposal_schedule(bank)
    return bank


def _validate_formal_proposal_schedule(
    bank: Mapping[int, Mapping[int, ProposalRecord]],
) -> None:
    """Enforce the predeclared proposal budget and source-frame separation."""
    for raw_seed, raw_records in bank.items():
        seed = _strict_int(raw_seed, "proposal schedule seed", nonnegative=True)
        records = dict(raw_records)
        _require(
            len(records) <= FORMAL_PROPOSAL_BUDGET,
            f"seed {seed}: proposal count exceeds formal budget of {FORMAL_PROPOSAL_BUDGET}",
        )
        frames = sorted(
            _strict_int(frame, f"seed {seed} proposal source frame", nonnegative=True)
            for frame in records
        )
        for earlier, later in zip(frames, frames[1:]):
            _require(
                later - earlier >= FORMAL_PROPOSAL_MIN_FRAME_GAP,
                f"seed {seed}: proposal source-frame gap is below "
                f"{FORMAL_PROPOSAL_MIN_FRAME_GAP}",
            )


def _proposal_manifest(
    bank: Mapping[int, Mapping[int, ProposalRecord]],
    *,
    source_root: Path,
    latency_profile: str,
    fixed_latency_steps: Optional[int] = None,
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> Dict[str, Any]:
    """Return the authenticated portable proposal-bank manifest."""
    source_group = _proposal_source_group(source_policy)
    root = Path(source_root).resolve()
    payload = canonical_proposal_bank_payload(bank)
    seeds = tuple(int(block["seed"]) for block in payload)
    if not seeds or tuple(sorted(seeds)) != seeds:
        raise ValueError("proposal-bank seeds must be nonempty and sorted")
    _validate_formal_proposal_schedule(bank)
    formal_source = _validate_formal_proposal_source_bundle(
        root, seeds, source_group=source_group
    )
    source_artifacts = []
    latency_samples = []
    for seed in seeds:
        records = dict(bank[seed])
        if not records and str(source_policy) != NATURAL_RGD_PROPOSAL_SOURCE_POLICY:
            raise ValueError(f"proposal bank seed {seed} has no candidates")
        event_path, reasoning_path, snapshot_path = _source_paths(root, seed)
        _validate_proposal_source(snapshot_path, seed=seed, source_policy=source_policy)
        runtime_path = formal_source["cells"][seed]["runtime_manifest_path"]
        source_artifacts.append(
            {
                "seed": seed,
                "event_log": {
                    "path": event_path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(event_path),
                },
                "reasoning_trace": {
                    "path": reasoning_path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(reasoning_path),
                },
                "experiment_snapshot": {
                    "path": snapshot_path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(snapshot_path),
                },
                "runtime_manifest": {
                    "path": runtime_path.relative_to(root).as_posix(),
                    "sha256": _sha256_file(runtime_path),
                },
            }
        )
        for frame, record in sorted(records.items()):
            latency_samples.append((seed, frame, record.latency_steps / _POLICY_FREQUENCY_HZ))
    profile = _build_empirical_latency_profile(latency_samples)
    public_profile = {key: value for key, value in profile.items() if not key.startswith("_")}
    manifest = {
        "schema": FACTORIAL_PROPOSAL_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "source_root": str(root),
        "candidate_source_policy": str(source_policy),
        "candidate_source_gate_independent": bool(
            str(source_policy) == DEFAULT_PROPOSAL_SOURCE_POLICY
        ),
        "latency_profile": str(latency_profile),
        "empirical_latency_profile": public_profile,
        "bank_sha256": proposal_bank_sha256(bank),
        "seed_count": len(seeds),
        "proposal_count": sum(len(block["records"]) for block in payload),
        "formal_source_bundle": {
            "bundle_kind": "formal_run",
            "partition": formal_source["partition"],
            "group": source_group,
            "environment": FORMAL_PROPOSAL_SOURCE_ENV,
            **FORMAL_PROPOSAL_SOURCE_VERSIONS,
            **{
                key: int(value) if float(value).is_integer() else float(value)
                for key, value in FORMAL_PROPOSAL_SOURCE_EXECUTION.items()
            },
            "result_bundle_manifest": {
                "path": formal_source["manifest_path"]
                .relative_to(formal_source["bundle_root"])
                .as_posix(),
                "sha256": _sha256_file(formal_source["manifest_path"]),
            },
            "run_rows": {
                "path": formal_source["run_rows_path"]
                .relative_to(formal_source["bundle_root"])
                .as_posix(),
                "sha256": _sha256_file(formal_source["run_rows_path"]),
            },
        },
        "source_artifacts": source_artifacts,
        "bank_payload": payload,
    }
    if str(latency_profile) == "fixed":
        manifest["fixed_latency_steps"] = _strict_int(
            fixed_latency_steps,
            "fixed latency steps",
            nonnegative=True,
        )
        manifest["fixed_latency_seconds"] = (
            float(manifest["fixed_latency_steps"]) / _POLICY_FREQUENCY_HZ
        )
    manifest["latency_profile_contract"] = {
        "frozen": "measured per-request Qwen service duration",
        "fixed": "common fixed policy-step delay",
        "jitter": "measured-profile median plus deterministic {-8,-4,0,4,8}-step jitter",
        "burst": "seed-correlated 15-step service bursts at a fixed 0.35 assignment rate",
        "drop": "request-deterministic 0.25 timeout assignment with a 10-step timeout extension",
        "out_of_order": "alternating +10/-10 steps by within-seed request ordinal",
        "stress": "legacy 22-step delay with deterministic 0.25 timeout assignment",
    }[str(latency_profile)]
    return manifest


def _release_execution_is_distinct(event: Mapping[str, Any]) -> bool:
    """Check primitive release selection without claiming a rollout effect."""
    row = dict(event or {})
    if not bool(row.get("closed_loop_latency_release_event", False)):
        return False
    rejected = bool(row.get("closed_loop_release_opportunity_rejected", False))
    unavailable = bool(row.get("closed_loop_release_action_unavailable", False))
    selected_present = "release_selected_action" in row
    comparator_present = "release_fast_comparator_action" in row
    if selected_present or comparator_present:
        try:
            selected = _strict_int(row.get("release_selected_action"), "release_selected_action")
            fast = _strict_int(row.get("release_fast_comparator_action"), "release_fast_comparator_action")
        except ValueError as exc:
            raise RuntimeError(
                "release_selected_action must be an integer action"
            ) from exc
        if rejected or unavailable:
            if selected != fast:
                raise RuntimeError("rejected/unavailable release must select the Fast action")
            return False
        stage = str(row.get("release_action_comparison_stage", "") or "")
        if stage != DISTINCT_ACTION_METRIC_STAGE:
            raise RuntimeError("release action comparison stage disagrees")
        distinct = selected != fast
        if "release_selection_distinct" not in row or bool(row["release_selection_distinct"]) is not distinct:
            raise RuntimeError("release_selection_distinct disagrees with selected action")
        expected_outcome = "distinct_actuation" if distinct else "fast_equivalent"
        if str(row.get("closed_loop_latency_terminal_outcome", "") or "") != expected_outcome:
            raise RuntimeError("terminal outcome disagrees with release selection")
        return distinct
    if rejected or unavailable:
        return False
    fast = row.get("closed_loop_execution_state_fast_action")
    executed = row.get("closed_loop_latency_executed_action", row.get("final_action"))
    if fast is None or executed is None:
        raise RuntimeError("release action comparison is unavailable")
    return _strict_int(executed, "closed_loop_latency_executed_action") != _strict_int(
        fast, "closed_loop_execution_state_fast_action"
    )


def _primitive_selection_metric_fields(
    primitive_count: Any,
    *,
    aligned_count: Any = 0,
) -> Dict[str, Any]:
    try:
        primitive = _strict_int(
            primitive_count, "primitive distinct selections", nonnegative=True
        )
        aligned = _strict_int(
            aligned_count, "aligned distinct actuations", nonnegative=True
        )
    except ValueError as exc:
        raise ValueError("aligned <= primitive is required") from exc
    if aligned > primitive:
        raise ValueError("aligned <= primitive is required")
    return {
        "distinct_actuations": primitive,
        "primitive_distinct_selections": primitive,
        "aligned_distinct_actuations": aligned,
        "distinct_action_metric_stage": DISTINCT_ACTION_METRIC_STAGE,
        "aligned_distinct_actuations_stage": DISTINCT_ACTION_METRIC_STAGE,
        "effect_distinctness_available": False,
    }


def _validate_request_outcome_accounting(
    issuance_outcomes: Mapping[str, str],
    terminal_outcomes: Mapping[str, str],
    pending_outcomes: Mapping[str, str],
    *,
    context: str,
) -> None:
    issued = {str(key): str(value) for key, value in dict(issuance_outcomes).items()}
    terminal = {str(key): str(value) for key, value in dict(terminal_outcomes).items()}
    pending = {str(key): str(value) for key, value in dict(pending_outcomes).items()}
    if set(terminal) & set(pending):
        raise RuntimeError(f"{context}: request is both terminal and pending")
    if not set(terminal).issubset(issued):
        raise RuntimeError(f"{context}: terminal outcome has no issuance")
    if not set(pending).issubset(issued):
        raise RuntimeError(f"{context}: pending outcome has no issuance")
    if set(terminal) | set(pending) != set(issued):
        raise RuntimeError(f"{context}: issuance lifecycle is incomplete")
    for request_id, outcome in terminal.items():
        if issued[request_id] != outcome:
            raise RuntimeError(f"{context}: issuance/terminal outcome mismatch")
    for request_id, outcome in pending.items():
        if issued[request_id] != outcome:
            raise RuntimeError(f"{context}: issuance/pending outcome mismatch")


def _validate_query_gate_accounting(
    arm: FactorialArm,
    *,
    candidate_events: Sequence[Mapping[str, Any]],
    candidate_count: Any,
    issued_count: Any,
    gate_rejected_count: Any,
    context: str,
) -> None:
    candidates = [dict(event) for event in candidate_events if bool(event.get("factorial_candidate_query", False))]
    expected_candidates = _strict_int(candidate_count, "candidate count", nonnegative=True)
    expected_issued = _strict_int(issued_count, "issued count", nonnegative=True)
    expected_rejected = _strict_int(gate_rejected_count, "gate rejection count", nonnegative=True)
    if len(candidates) != expected_candidates:
        raise RuntimeError(f"{context}: candidate accounting mismatch")
    issued = sum(bool(event.get("factorial_query_issued", False)) for event in candidates)
    rejected = sum(
        not bool(event.get("factorial_query_issued", False))
        and str(event.get("factorial_query_rejection_reason", "") or "")
        in {"query_gate_failed", "fast_only_control"}
        for event in candidates
    )
    if issued != expected_issued:
        raise RuntimeError(f"{context}: issuance accounting mismatch")
    if rejected != expected_rejected:
        raise RuntimeError(f"{context}: rejection accounting mismatch")
    if expected_candidates != expected_issued + expected_rejected:
        raise RuntimeError(f"{context}: query accounting is not a partition")
    if arm.name == "fast_only":
        if expected_issued != 0 or expected_rejected != expected_candidates:
            raise RuntimeError(f"{context}: Fast-only control must suppress every candidate")
        if any(
            str(event.get("factorial_query_rejection_reason", "") or "")
            != "fast_only_control"
            for event in candidates
        ):
            raise RuntimeError(f"{context}: Fast-only rejection provenance drift")
    elif not arm.query_gate_enabled:
        if expected_rejected or any(
            str(event.get("factorial_query_rejection_reason", "") or "")
            for event in candidates
        ):
            raise RuntimeError(f"{context}: query-disabled arm cannot reject candidates")
        if expected_issued != expected_candidates:
            raise RuntimeError(f"{context}: query-disabled arm must issue every candidate")


def _validate_outcome_metrics(
    episode_summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    summary = dict(episode_summary or {})
    if type(summary.get("collision")) is not bool:
        raise RuntimeError(f"{context}: collision outcome must be an explicit boolean")
    payload = dict(metrics or {})
    required = (
        "total_episodes",
        "collision_rate",
        "success_rate",
        "success_number",
        "avg_route_completion",
        "avg_episode_reward",
        "avg_driving_distance",
        "avg_speed_all_frames",
        "avg_runtime_per_frame",
    )
    for name in required:
        if name not in payload:
            raise RuntimeError(f"{context}: required metric {name} is missing")
    try:
        total_episodes = _strict_int(payload["total_episodes"], "total_episodes")
    except ValueError as exc:
        raise RuntimeError(
            f"{context}: total_episodes must report exactly one episode"
        ) from exc
    if total_episodes != 1:
        raise RuntimeError(f"{context}: total_episodes must report exactly one episode")
    try:
        success_number = _strict_int(payload["success_number"], "success_number")
    except ValueError as exc:
        raise RuntimeError(f"{context}: success_number must be binary") from exc
    if success_number not in (0, 1):
        raise RuntimeError(f"{context}: success_number must be binary")
    try:
        numeric = {
            name: _finite_float(payload[name], name)
            for name in (
                "collision_rate",
                "success_rate",
                "avg_route_completion",
                "avg_episode_reward",
                "avg_driving_distance",
                "avg_speed_all_frames",
                "avg_runtime_per_frame",
            )
        }
    except ValueError as exc:
        raise RuntimeError(f"{context}: {exc}") from exc
    for name in ("success_rate", "avg_route_completion"):
        if numeric[name] < 0.0:
            raise RuntimeError(f"{context}: {name} must be at least zero")
        if numeric[name] > 1.0:
            raise RuntimeError(f"{context}: {name} must be at most one")
    for name in ("avg_driving_distance", "avg_speed_all_frames", "avg_runtime_per_frame"):
        if numeric[name] < 0.0:
            raise RuntimeError(f"{context}: {name} must be at least zero")
    collision = int(bool(summary["collision"]))
    if numeric["collision_rate"] != float(collision):
        raise RuntimeError(f"{context}: collision rate disagrees with episode outcome")
    if numeric["success_rate"] != float(success_number):
        raise RuntimeError(f"{context}: success rate disagrees with success_number")
    return {
        "collision": collision,
        "success_rate": numeric["success_rate"],
        "route_completion": numeric["avg_route_completion"],
        "episode_reward": numeric["avg_episode_reward"],
        "driving_distance": numeric["avg_driving_distance"],
        "avg_speed": numeric["avg_speed_all_frames"],
        "runtime_per_frame": numeric["avg_runtime_per_frame"],
    }


def _validate_event_lifecycle_contract(
    events: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    """Fail closed on ambiguous request issuance or terminal attribution."""
    issued: Dict[str, str] = {}
    terminal: set[str] = set()
    for index, raw_event in enumerate(events):
        event = dict(raw_event or {})
        candidate = bool(event.get("factorial_candidate_query", False))
        candidate_issued = bool(event.get("factorial_query_issued", False))
        issuance_event = bool(event.get("closed_loop_latency_issuance_event", False))
        terminal_event = bool(event.get("closed_loop_latency_terminal_event", False))
        flags = [bool(event.get(field, False)) for field in _TERMINAL_FLAGS]
        if sum(flags) > 1:
            raise RuntimeError(f"{context} frame {index}: terminal flags are not mutually exclusive")
        if candidate and candidate_issued:
            if not issuance_event:
                raise RuntimeError(f"{context} frame {index}: issuance event disagrees with factorial issuance")
            candidate_id = str(event.get("factorial_candidate_request_id", "") or "")
            issued_id = str(event.get("closed_loop_latency_issued_request_id", "") or "")
            if not candidate_id or candidate_id != issued_id:
                raise RuntimeError(f"{context} frame {index}: issuance request ID disagrees")
            shared_outcome = str(event.get("factorial_shared_response_outcome", "") or "")
            issued_outcome = str(event.get("closed_loop_latency_issued_response_outcome", "") or "")
            if shared_outcome not in _VALID_OUTCOMES or issued_outcome != shared_outcome:
                raise RuntimeError(f"{context} frame {index}: issuance outcome disagrees")
            if candidate_id in issued:
                raise RuntimeError(f"{context} frame {index}: duplicate request issuance")
            issued[candidate_id] = shared_outcome
        elif issuance_event:
            raise RuntimeError(f"{context} frame {index}: issuance event without factorial candidate")
        elif candidate and not candidate_issued:
            if str(event.get("factorial_query_rejection_reason", "") or "") not in {
                "",
                "query_gate_failed",
                "fast_only_control",
            }:
                raise RuntimeError(f"{context} frame {index}: invalid query rejection reason")

        terminal_id = str(event.get("closed_loop_latency_terminal_request_id", "") or "")
        terminal_response = str(event.get("closed_loop_latency_terminal_response_outcome", "") or "")
        terminal_outcome = str(event.get("closed_loop_latency_terminal_outcome", "") or "")
        if not terminal_event:
            if candidate and candidate_issued and (
                terminal_id or terminal_response or terminal_outcome not in {"", "pending"} or any(flags)
            ):
                raise RuntimeError(f"{context} frame {index}: terminal marker disagrees with terminal fields")
            if not candidate and (terminal_id or terminal_response or terminal_outcome not in {"", "pending"}):
                raise RuntimeError(f"{context} frame {index}: non-terminal event carries terminal fields")
            continue
        if not terminal_id or terminal_response not in _VALID_OUTCOMES:
            raise RuntimeError(f"{context} frame {index}: malformed terminal event")
        if terminal_id not in issued:
            raise RuntimeError(f"{context} frame {index}: terminal precedes issuance")
        if terminal_id in terminal:
            raise RuntimeError(f"{context} frame {index}: duplicate terminal event")
        if issued[terminal_id] != terminal_response:
            raise RuntimeError(f"{context} frame {index}: terminal response outcome disagrees")
        if terminal_response == "valid":
            if flags != [True, False, False]:
                raise RuntimeError(f"{context} frame {index}: valid terminal flags disagree")
            expected = "distinct_actuation" if _release_execution_is_distinct(event) else "fast_equivalent"
            if terminal_outcome != expected:
                raise RuntimeError(f"{context} frame {index}: asynchronous terminal outcome disagrees")
        else:
            expected_flags = [False, terminal_response == "timeout", terminal_response == "failure"]
            if flags != expected_flags:
                raise RuntimeError(f"{context} frame {index}: asynchronous terminal flags disagree")
            if terminal_outcome != terminal_response:
                raise RuntimeError(f"{context} frame {index}: asynchronous terminal outcome disagrees")
        terminal.add(terminal_id)


def _event_request_id(event: Mapping[str, Any]) -> str:
    for field in (
        "closed_loop_latency_terminal_request_id",
        "closed_loop_latency_issued_request_id",
        "closed_loop_latency_request_id",
    ):
        value = str(event.get(field, "") or "")
        if value:
            return value
    return ""


def _normalize_factorial_event(
    event: MutableMapping[str, Any],
    *,
    arm: FactorialArm,
    bank_sha256: str,
    proposals_by_id: Mapping[str, ProposalRecord],
    policy_frequency_hz: float,
) -> None:
    """Fill the versioned audit fields from the executed request ledger."""
    event.update(
        {
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "factorial_arm": arm.name,
            "factorial_query_gate_enabled": bool(arm.query_gate_enabled),
            "factorial_release_guard_enabled": bool(arm.release_guard_enabled),
            "factorial_proposal_bank_sha256": str(bank_sha256),
            "factorial_candidate_query": bool(event.get("factorial_candidate_query", False)),
            "factorial_query_issued": bool(event.get("factorial_query_issued", False)),
            "closed_loop_latency_issuance_event": bool(event.get("closed_loop_latency_issuance_event", False)),
            "closed_loop_latency_terminal_event": bool(event.get("closed_loop_latency_terminal_event", False)),
            "closed_loop_latency_release_event": bool(event.get("closed_loop_latency_release_event", False)),
            "closed_loop_latency_timeout_event": bool(event.get("closed_loop_latency_timeout_event", False)),
            "closed_loop_latency_failure_event": bool(event.get("closed_loop_latency_failure_event", False)),
        }
    )
    terminal_id = str(event.get("closed_loop_latency_terminal_request_id", "") or "")
    issued_id = str(event.get("closed_loop_latency_issued_request_id", "") or "")
    request_id = terminal_id if bool(event["closed_loop_latency_terminal_event"]) else issued_id
    if not request_id:
        request_id = str(event.get("closed_loop_latency_request_id", "") or "")
    proposal = proposals_by_id.get(request_id)
    if proposal is None:
        return
    source_frame = int(proposal.source_frame)
    steps = int(proposal.latency_steps)
    scheduled_frame = source_frame + steps
    event.update(
        {
            "closed_loop_latency_request_id": request_id,
            "closed_loop_latency_source_frame": source_frame,
            "closed_loop_latency_source_system": "slow",
            "closed_loop_latency_delay_steps": steps,
            "closed_loop_latency_scheduled_steps": steps,
            "closed_loop_latency_scheduled_release_frame": scheduled_frame,
            "closed_loop_latency_policy_frequency_hz": float(policy_frequency_hz),
            "closed_loop_latency_scheduled_seconds": steps / float(policy_frequency_hz),
            "closed_loop_latency_response_outcome": proposal.outcome,
        }
    )
    if bool(event["closed_loop_latency_terminal_event"]):
        frame = _strict_int(event.get("frame"), "event frame", nonnegative=True)
        event.update(
            {
                "closed_loop_latency_terminal_request_id": request_id,
                "closed_loop_latency_terminal_response_outcome": proposal.outcome,
                "closed_loop_latency_realized_steps": frame - source_frame,
                "closed_loop_latency_realized_seconds": (frame - source_frame) / float(policy_frequency_hz),
                "closed_loop_latency_realized_available": True,
                "closed_loop_latency_realized_source": "simulator_frame_delta",
            }
        )
        if proposal.outcome == "valid":
            if not bool(event.get("closed_loop_latency_release_event", False)):
                raise RuntimeError("valid factorial terminal did not enter release evaluation")
            fast = _strict_int(event.get("release_fast_comparator_action"), "release fast comparator")
            selected = _strict_int(event.get("release_selected_action"), "release selected action")
            distinct = _release_execution_is_distinct(event)
            event.update(
                {
                    "closed_loop_latency_terminal_outcome": "distinct_actuation" if distinct else "fast_equivalent",
                    "release_selection_distinct": distinct,
                }
            )
        else:
            if bool(event.get("closed_loop_latency_release_event", False)):
                raise RuntimeError("non-valid factorial terminal cannot be a release event")
            event["closed_loop_latency_terminal_outcome"] = proposal.outcome
            event.setdefault("closed_loop_release_snapshot_captured", False)
    else:
        event.update(
            {
                "closed_loop_latency_terminal_outcome": "pending",
                "closed_loop_latency_realized_steps": -1,
                "closed_loop_latency_realized_seconds": float("nan"),
                "closed_loop_latency_realized_available": False,
                "closed_loop_latency_realized_source": "not_released",
            }
        )


def _save_factorial_episode(
    *,
    root: Path,
    seed: int,
    prefix: str,
    events: Sequence[Mapping[str, Any]],
    pending: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    physical_recorder: Any,
    reasoning_recorder: Any,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Persist the one-episode bundle in the schema consumed by the auditor."""
    from dataclasses import asdict as dataclass_asdict

    from dilu.evaluation.release_snapshot import save_release_snapshot_bundle

    root.mkdir(parents=True, exist_ok=True)
    release_bundle = release_manifest = release_digest = None
    if snapshots:
        bundle_path, manifest_path, release_digest = save_release_snapshot_bundle(
            snapshots, root / "release_snapshots", prefix=prefix, episode_id=seed
        )
        release_bundle = bundle_path.relative_to(root).as_posix()
        release_manifest = manifest_path.relative_to(root).as_posix()
    pending_rows = [
        {
            "request_id": str(item.get("request_id", "") or ""),
            "source_frame": int(item.get("source_frame", 0) or 0),
            "release_frame": int(item.get("release_frame", 0) or 0),
            "response_outcome": str(item.get("response_outcome", "") or ""),
            "terminal_outcome": "dropped_at_episode_end",
        }
        for item in pending
    ]
    event_payload = {
        "schema_version": FACTORIAL_EVENT_SCHEMA,
        "episode_id": int(seed),
        "event_count": len(events),
        "pending_release_count": len(pending_rows),
        "pending_releases_dropped_at_episode_end": pending_rows,
        "release_snapshot_count": len(snapshots),
        "release_snapshot_bundle": release_bundle,
        "release_snapshot_manifest": release_manifest,
        "release_snapshot_bundle_sha256": release_digest,
        "terminal_cause": str(events[-1].get("terminal_cause", "truncated") if events else "truncated"),
        "events": [dict(event) for event in events],
    }
    event_path = root / "event_logs" / f"event_log_{prefix}_{seed}.json"
    strict_event_payload = _sanitize_nonfinite_json(event_payload)
    _write_json(event_path, strict_event_payload, allow_nan=False)

    physical_payload: Dict[str, Any] = {}
    if physical_recorder is not None:
        physical_payload = dict(physical_recorder.save())
    reasoning_records = []
    if reasoning_recorder is not None:
        reasoning_records = [dataclass_asdict(record) for record in reasoning_recorder.records]
    reasoning_payload = {"episode_id": int(seed), "analysis_records": reasoning_records}
    reasoning_path = root / f"{prefix}_reasoning_records.json"
    _write_json(reasoning_path, reasoning_payload)
    return physical_payload, reasoning_payload


def _run_arm_seed(
    *,
    protocol: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    protocol_path: Path,
    result_root: Path,
    seed: int,
    arm: FactorialArm,
    proposals: Mapping[int, ProposalRecord],
    bank_sha256: str,
    proposal_source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
    verbose: bool = False,
    query_admission_policy: Any = None,
) -> Dict[str, Any]:
    """Execute one arm/seed cell with an isolated controller and environment."""
    del protocol_path
    from dilu.evaluation.metrics_aggregator import MetricsAggregator
    from dilu.evaluation.reporter import save_experiment_snapshot
    from dilu.runtime_episode_setup import (
        create_episode_agent,
        create_episode_env,
        create_episode_recorders,
    )
    from dilu.runtime_frame_trace import create_episode_runtime_state
    from dilu.runtime_support import (
        exclude_policy_pacing_sleep,
        execute_episode_step,
    )
    from dilu.safety.unified_safety import UnifiedSafetySystem
    from dilu.scenario import create_scenario
    seed = _strict_int(seed, "factorial seed", nonnegative=True)
    if not proposals and str(proposal_source_policy) != NATURAL_RGD_PROPOSAL_SOURCE_POLICY:
        raise ValueError(f"seed {seed}: factorial proposal block is empty")
    expected_frames = {int(frame) for frame in proposals}
    if expected_frames != {int(record.source_frame) for record in proposals.values()}:
        raise ValueError(f"seed {seed}: proposal frame keys drift")
    predicted_latency = _finite_float(
        base_cfg.get("rgd_predicted_slow_latency_s", 0.0), "predicted latency"
    )
    group_cfg = _factorial_group_config(
        protocol, arm, predicted_latency_s=predicted_latency
    )
    cell_root = Path(result_root) / arm.name / f"seed_{seed}"
    cfg = build_group_config(
        base_cfg,
        f"factorial_{arm.name}",
        group_cfg,
        "highway-v0",
        1,
        cell_root,
        protocol,
    )
    cfg = configure_factorial_arm(cfg, arm)
    cfg.update(
        {
            "fixed_seed_override": seed,
            "enable_physical_metrics": True,
            "enable_reasoning_recording": True,
            "event_log_schema_version": FACTORIAL_EVENT_SCHEMA,
        }
    )
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    formal_execution = submission.get("formal_execution_contract")
    if (
        str(protocol.get("protocol_name", "") or "") == V13_PROTOCOL_NAME
        and isinstance(formal_execution, Mapping)
    ):
        execution_horizon = validate_policy_execution_horizon(
            cfg,
            formal_execution,
            context=f"factorial {arm.name}/{seed}",
        )
    else:
        execution_horizon = resolve_policy_execution_horizon(
            cfg, context=f"factorial {arm.name}/{seed}"
        )
    save_experiment_snapshot(cfg, str(cell_root), seed)

    env = None
    inner = None
    try:
        with _worker_output_context(verbose=verbose):
            env, obs, ep_dir, prefix, resolved_seed, close_after = create_episode_env(
                seed, cfg, str(cell_root), [seed]
            )
            if int(resolved_seed) != seed:
                raise RuntimeError("factorial environment seed drift")
            scenario = create_scenario(
                env, str(cfg["env_type"]), seed, str(Path(ep_dir) / "scenario.db")
            )
            inner = create_episode_agent(scenario, cfg, str(cell_root))
            agent = ProposalReplayAgent(
                inner,
                proposals,
                arm=arm,
                bank_sha256=str(bank_sha256),
                query_admission_policy=query_admission_policy,
            )
            physical, reasoning = create_episode_recorders(seed, seed, ep_dir, cfg)
            safety = UnifiedSafetySystem(cfg)
            aggregate = MetricsAggregator(arm.name, str(cell_root))
            history = deque(maxlen=max(1, int(cfg.get("history_window", 16) or 16)))
            episode_state = create_episode_runtime_state()
            runtimes = []
            for frame in range(execution_horizon.expected_policy_steps):
                started = time.perf_counter()
                obs, done = execute_episode_step(
                    frame=frame,
                    env=env,
                    sce=scenario,
                    agent=agent,
                    obs=obs,
                    cfg=cfg,
                    safety_system=safety,
                    phys_rec=physical,
                    reas_rec=reasoning,
                    history_buffer=history,
                    episode_state=episode_state,
                )
                runtimes.append(
                    exclude_policy_pacing_sleep(
                        time.perf_counter() - started,
                        episode_state,
                    )
                )
                if done:
                    break

            events = [dict(event) for event in episode_state["event_log"]]
            proposals_by_id = {
                str(record.request_id): record for record in proposals.values()
            }
            frequency = float(cfg.get("policy_frequency", _POLICY_FREQUENCY_HZ) or _POLICY_FREQUENCY_HZ)
            for event in events:
                _normalize_factorial_event(
                    event,
                    arm=arm,
                    bank_sha256=str(bank_sha256),
                    proposals_by_id=proposals_by_id,
                    policy_frequency_hz=frequency,
                )
            pending = list(episode_state.get("latency_replay_queue", []) or [])
            end_episode = getattr(agent, "end_episode", None)
            if callable(end_episode):
                pending.extend(
                    dict(item)
                    for item in list(end_episode("factorial_episode_finalize") or [])
                )
            physical_payload, reasoning_payload = _save_factorial_episode(
                root=cell_root,
                seed=seed,
                prefix=prefix,
                events=events,
                pending=pending,
                snapshots=dict(episode_state.get("release_snapshots", {}) or {}),
                physical_recorder=physical,
                reasoning_recorder=reasoning,
            )
            aggregate.add_episode(
                physical_payload=physical_payload, reasoning_payload=reasoning_payload
            )
            aggregate.all_event_records = events
            metrics = aggregate.calculate_comprehensive_metrics()
            metrics["avg_runtime_per_frame"] = (
                float(sum(runtimes) / len(runtimes)) if runtimes else 0.0
            )
            physical_metrics = dict(physical_payload.get("metrics", {}) or {})
            collision = bool(physical_metrics.get("collision", False))
            success = int(bool(physical_metrics.get("success_completion", False)))
            metric_input = {
                "total_episodes": 1,
                "collision_rate": float(collision),
                "success_rate": float(success),
                "success_number": success,
                "avg_route_completion": float(success),
                "avg_episode_reward": float(metrics.get("avg_episode_reward", 0.0)),
                "avg_driving_distance": float(metrics.get("avg_driving_distance", 0.0)),
                "avg_speed_all_frames": float(metrics.get("avg_speed_all_frames", 0.0)),
                "avg_runtime_per_frame": float(metrics["avg_runtime_per_frame"]),
            }
            outcomes = _validate_outcome_metrics(
                {"collision": collision}, metric_input, context=f"{arm.name}/{seed}"
            )
            candidate_events = [event for event in events if bool(event.get("factorial_candidate_query", False))]
            issued_events = [event for event in candidate_events if bool(event.get("factorial_query_issued", False))]
            rejected_count = len(candidate_events) - len(issued_events)
            _validate_query_gate_accounting(
                arm,
                candidate_events=candidate_events,
                candidate_count=agent.candidate_count,
                issued_count=agent.issued_count,
                gate_rejected_count=agent.gate_rejected_count,
                context=f"{arm.name}/{seed}",
            )
            _validate_event_lifecycle_contract(events, context=f"{arm.name}/{seed}")
            issuance = {
                str(event["closed_loop_latency_issued_request_id"]): str(event["closed_loop_latency_issued_response_outcome"])
                for event in events
                if bool(event.get("closed_loop_latency_issuance_event", False))
            }
            terminal = {
                str(event["closed_loop_latency_terminal_request_id"]): str(event["closed_loop_latency_terminal_response_outcome"])
                for event in events
                if bool(event.get("closed_loop_latency_terminal_event", False))
            }
            pending = {
                str(item["request_id"]): str(item["response_outcome"])
                for item in list(episode_state.get("latency_replay_queue", []) or [])
            }
            _validate_request_outcome_accounting(
                issuance, terminal, pending, context=f"{arm.name}/{seed}"
            )
            release_events = [
                event for event in events if bool(event.get("closed_loop_latency_release_event", False))
            ]
            primitive = sum(_release_execution_is_distinct(event) for event in release_events)
            aligned = primitive if arm.release_guard_enabled else 0
            row = {
                "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
                "arm": arm.name,
                "query_gate_enabled": bool(arm.query_gate_enabled),
                "release_guard_enabled": bool(arm.release_guard_enabled),
                "seed": seed,
                "proposal_bank_sha256": str(bank_sha256),
                "candidate_source_policy": str(proposal_source_policy),
                "candidate_source_gate_independent": bool(
                    str(proposal_source_policy) == DEFAULT_PROPOSAL_SOURCE_POLICY
                ),
                **execution_horizon.as_manifest(),
                "frames_executed": len(runtimes),
                **outcomes,
                "candidate_queries": len(candidate_events),
                "issued_queries": len(issued_events),
                "query_gate_rejections": rejected_count,
                "scheduled_timeouts": sum(value == "timeout" for value in issuance.values()),
                "timeouts": sum(value == "timeout" for value in terminal.values()),
                "failure_events": sum(value == "failure" for value in terminal.values()),
                "release_events": len(release_events),
                "pending_at_episode_end": len(pending),
                "pending_timeouts_at_episode_end": sum(value == "timeout" for value in pending.values()),
                "snapshot_count": len(episode_state.get("release_snapshots", {}) or {}),
                **_primitive_selection_metric_fields(primitive, aligned_count=aligned),
            }
            return row
    finally:
        if inner is not None:
            close_agent = getattr(inner, "close", None)
            if callable(close_agent):
                close_agent()
        if env is not None and bool(locals().get("close_after", True)):
            close_env = getattr(env, "close", None)
            if callable(close_env):
                close_env()


def _arm_order(
    seed: int,
    arms: Sequence[FactorialArm] = FACTORIAL_ARMS,
) -> list[FactorialArm]:
    arms = list(arms)
    random.Random(20260731 + int(seed)).shuffle(arms)
    return arms


def _run_seed_block(task: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    seed = int(task["seed"])
    rows = []
    order = []
    arms = tuple(task.get("arms") or FACTORIAL_ARMS)
    for index, arm in enumerate(_arm_order(seed, arms)):
        rows.append(
            _run_arm_seed(
                protocol=task["protocol"],
                base_cfg=task["base_cfg"],
                protocol_path=Path(task["protocol_path"]),
                result_root=Path(task["result_root"]),
                seed=seed,
                arm=arm,
                proposals=task["proposals"],
                bank_sha256=str(task["bank_sha256"]),
                proposal_source_policy=str(task["proposal_source_policy"]),
                verbose=bool(task["verbose"]),
            )
        )
        order.append({"seed": seed, "order": index, "arm": arm.name})
    return rows, order


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--source-policy",
        choices=(DEFAULT_PROPOSAL_SOURCE_POLICY, NATURAL_RGD_PROPOSAL_SOURCE_POLICY),
        default=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--design",
        choices=("legacy_four", "five_arm"),
        default="five_arm",
        help="Use the 2x2 gate factorial alone or add a genuine no-query Fast-only control.",
    )
    parser.add_argument(
        "--latency-profile",
        choices=(
            "frozen",
            "fixed",
            "jitter",
            "burst",
            "drop",
            "out_of_order",
            "stress",
        ),
        default="frozen",
    )
    parser.add_argument(
        "--fixed-delay-steps",
        type=int,
        default=None,
        help="Common replay delay in policy steps; required for --latency-profile fixed.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.seed_start < 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError("seed range and worker count must be positive")
    if args.latency_profile == "fixed" and (
        args.fixed_delay_steps is None or args.fixed_delay_steps < 0
    ):
        raise ValueError("fixed latency replay requires nonnegative --fixed-delay-steps")
    if args.latency_profile != "fixed" and args.fixed_delay_steps is not None:
        raise ValueError("--fixed-delay-steps requires --latency-profile fixed")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    arms = FORMAL_FACTORIAL_ARMS if args.design == "five_arm" else FACTORIAL_ARMS
    protocol = load_formal_protocol(args.protocol)
    base_cfg = load_formal_base_config(protocol, REPO_ROOT / "config.yaml")
    execution_horizon = _validate_formal_factorial_preflight(
        protocol=protocol,
        base_cfg=base_cfg,
        design=str(args.design),
        seeds=seeds,
        latency_profile=str(args.latency_profile),
        fixed_latency_steps=args.fixed_delay_steps,
        source_policy=str(args.source_policy),
        result_root=Path(args.result_root),
    )
    bank = load_proposal_bank(
        args.source_root,
        seeds,
        latency_profile=args.latency_profile,
        fixed_latency_steps=args.fixed_delay_steps,
        source_policy=str(args.source_policy),
    )
    proposal_manifest = _proposal_manifest(
        bank,
        source_root=args.source_root,
        latency_profile=args.latency_profile,
        fixed_latency_steps=args.fixed_delay_steps,
        source_policy=str(args.source_policy),
    )
    bank_digest = str(proposal_manifest["bank_sha256"])
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    _write_json(result_root / "proposal_bank_manifest.json", proposal_manifest)
    tasks = [
        {
            "protocol": protocol,
            "base_cfg": base_cfg,
            "protocol_path": str(Path(args.protocol).resolve()),
            "result_root": str(result_root.resolve()),
            "seed": seed,
            "proposals": bank[seed],
            "bank_sha256": bank_digest,
            "proposal_source_policy": str(args.source_policy),
            "arms": arms,
            "verbose": bool(args.verbose),
        }
        for seed in seeds
    ]
    pool = None
    if int(args.workers) == 1:
        results = map(_run_seed_block, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=min(int(args.workers), len(tasks)))
        results = pool.map(_run_seed_block, tasks)
    rows: list[Dict[str, Any]] = []
    order: list[Dict[str, Any]] = []
    try:
        for seed_rows, seed_order in results:
            rows.extend(seed_rows)
            order.extend(seed_order)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    rows.sort(key=lambda row: (int(row["seed"]), str(row["arm"])))
    _write_csv(result_root / "factorial_episode_results.csv", rows)
    _write_json(
        result_root / "factorial_run_manifest.json",
        {
            "schema": FACTORIAL_RUN_SCHEMA,
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "protocol_path": str(Path(args.protocol).resolve()),
            "protocol_sha256": _sha256_file(Path(args.protocol)),
            "proposal_bank_sha256": bank_digest,
            "method_version": str(
                dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                    "rgd_method_version", ""
                )
                or ""
            ),
            "query_gate_method_version": str(
                dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                    "query_gate_method_version", ""
                )
                or ""
            ),
            "release_contract_version": str(
                dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                    "release_contract_version", ""
                )
                or ""
            ),
            "factorial_design": str(args.design),
            "latency_profile": str(args.latency_profile),
            "fixed_latency_steps": (
                int(args.fixed_delay_steps)
                if args.fixed_delay_steps is not None
                else None
            ),
            "candidate_source_policy": str(args.source_policy),
            "candidate_source_gate_independent": bool(
                str(args.source_policy) == DEFAULT_PROPOSAL_SOURCE_POLICY
            ),
            "seed_start": int(args.seed_start),
            "seed_count": int(args.seeds),
            "arms": [asdict(arm) for arm in arms],
            "randomized_block_run_order": order,
            "result_rows": len(rows),
            **(
                execution_horizon.as_manifest()
                if execution_horizon is not None
                else {}
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
