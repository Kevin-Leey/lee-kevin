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
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_ARMS,
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


DEFAULT_SOURCE = Path(
    "results/tvt_final_20260721/main_identifiable_v12_diagnostic/formal_run/"
    "main_v12_20260721/always_slow/highway"
)
DEFAULT_PROPOSAL_SOURCE_POLICY = "scheduled_always_slow"
LEGACY_PROPOSAL_SOURCE_POLICY = "legacy_gate_positive_diagnostic"
DISTINCT_ACTION_METRIC_STAGE = (
    "post_release_guard_and_frame_safety_pre_actuator_bridge"
)
_VALID_OUTCOMES = frozenset({"valid", "timeout", "failure"})
_TERMINAL_FLAGS = (
    "closed_loop_latency_release_event",
    "closed_loop_latency_timeout_event",
    "closed_loop_latency_failure_event",
)
_POLICY_FREQUENCY_HZ = 10.0


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
        }
    )
    overrides.update(
        {
            "protocol_name": f"factorial_{arm.name}",
            "system_routing": {"simple": "fast", "complex": "fast"},
            "closed_loop_latency_replay": replay,
            "factorial_predicted_latency_s": float(predicted),
            "factorial_predicted_latency_steps": _latency_seconds_to_policy_steps(predicted),
        }
    )
    result["runtime_overrides"] = overrides
    result.setdefault("id", f"factorial_{arm.name}")
    return result


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


def _validate_proposal_source(
    snapshot_path: Path,
    *,
    seed: int,
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> None:
    """Verify that a proposal source is independent of the evaluated gate."""
    if str(source_policy) != DEFAULT_PROPOSAL_SOURCE_POLICY:
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
    if str(config.get("protocol_name", "") or "") != "always_slow":
        raise RuntimeError("gate-independent proposal source must be always_slow")
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
            "query_state_slow_released_action",
            "query_state_slow_pre_guard_action",
            "post_validation_action",
            "llm_raw_action",
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
        for name in ("inference_latency", "latency_seconds", "latency_s"):
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


def load_proposal_bank(
    source_root: Path,
    seeds: Sequence[int],
    *,
    latency_profile: str = "frozen",
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> Dict[int, Dict[int, ProposalRecord]]:
    """Freeze source proposals and bind each one to a replay delay/outcome."""
    seed_values = tuple(_strict_int(seed, "proposal seed", nonnegative=True) for seed in seeds)
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("proposal seeds must be nonempty and unique")
    if str(latency_profile) not in {"frozen", "stress"}:
        raise ValueError("latency profile must be 'frozen' or 'stress'")
    staged: list[Dict[str, Any]] = []
    for seed in sorted(seed_values):
        event_path, reasoning_path, snapshot_path = _source_paths(source_root, seed)
        _validate_proposal_source(
            snapshot_path, seed=seed, source_policy=source_policy
        )
        events = _read_json_object(event_path, allow_nan=True).get("events", [])
        if not isinstance(events, list):
            raise ValueError(f"seed {seed}: source event log has no event list")
        reasoning_by_frame = _reasoning_records(reasoning_path)
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
    bank: Dict[int, Dict[int, ProposalRecord]] = {seed: {} for seed in sorted(seed_values)}
    for row in staged:
        request_id = str(row["request_id"])
        if str(latency_profile) == "stress":
            steps, outcome = _stress_assignment(request_id)
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
    return bank


def _proposal_manifest(
    bank: Mapping[int, Mapping[int, ProposalRecord]],
    *,
    source_root: Path,
    latency_profile: str,
    source_policy: str = DEFAULT_PROPOSAL_SOURCE_POLICY,
) -> Dict[str, Any]:
    """Return the authenticated portable proposal-bank manifest."""
    if str(source_policy) != DEFAULT_PROPOSAL_SOURCE_POLICY:
        raise ValueError("paper-facing factorial requires scheduled_always_slow")
    root = Path(source_root).resolve()
    payload = canonical_proposal_bank_payload(bank)
    seeds = tuple(int(block["seed"]) for block in payload)
    if not seeds or tuple(sorted(seeds)) != seeds:
        raise ValueError("proposal-bank seeds must be nonempty and sorted")
    source_artifacts = []
    latency_samples = []
    for seed in seeds:
        records = dict(bank[seed])
        if not records:
            raise ValueError(f"proposal bank seed {seed} has no candidates")
        event_path, reasoning_path, snapshot_path = _source_paths(root, seed)
        _validate_proposal_source(snapshot_path, seed=seed, source_policy=source_policy)
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
            }
        )
        for frame, record in sorted(records.items()):
            latency_samples.append((seed, frame, record.latency_steps / _POLICY_FREQUENCY_HZ))
    profile = _build_empirical_latency_profile(latency_samples)
    public_profile = {key: value for key, value in profile.items() if not key.startswith("_")}
    return {
        "schema": FACTORIAL_PROPOSAL_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "source_root": str(root),
        "candidate_source_policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
        "candidate_source_gate_independent": True,
        "latency_profile": str(latency_profile),
        "empirical_latency_profile": public_profile,
        "bank_sha256": proposal_bank_sha256(bank),
        "seed_count": len(seeds),
        "proposal_count": sum(len(block["records"]) for block in payload),
        "source_artifacts": source_artifacts,
        "bank_payload": payload,
    }


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
        and str(event.get("factorial_query_rejection_reason", "") or "") == "query_gate_failed"
        for event in candidates
    )
    if issued != expected_issued:
        raise RuntimeError(f"{context}: issuance accounting mismatch")
    if rejected != expected_rejected:
        raise RuntimeError(f"{context}: rejection accounting mismatch")
    if expected_candidates != expected_issued + expected_rejected:
        raise RuntimeError(f"{context}: query accounting is not a partition")
    if not arm.query_gate_enabled:
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
            if str(event.get("factorial_query_rejection_reason", "") or "") not in {"", "query_gate_failed"}:
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
            fast = _strict_int(event.get("release_fast_comparator_action"), "release fast comparator")
            selected = _strict_int(event.get("release_selected_action"), "release selected action")
            distinct = selected != fast
            event.update(
                {
                    "closed_loop_latency_release_event": True,
                    "closed_loop_latency_timeout_event": False,
                    "closed_loop_latency_failure_event": False,
                    "closed_loop_latency_terminal_outcome": "distinct_actuation" if distinct else "fast_equivalent",
                    "release_action_comparison_stage": DISTINCT_ACTION_METRIC_STAGE,
                    "release_selection_distinct": distinct,
                    "closed_loop_release_opportunity_rejected": False,
                    "closed_loop_release_action_unavailable": False,
                }
            )
        else:
            event.update(
                {
                    "closed_loop_latency_release_event": False,
                    "closed_loop_latency_timeout_event": proposal.outcome == "timeout",
                    "closed_loop_latency_failure_event": proposal.outcome == "failure",
                    "closed_loop_latency_terminal_outcome": proposal.outcome,
                    "closed_loop_release_snapshot_captured": False,
                }
            )
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
    _write_json(event_path, event_payload, allow_nan=True)

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
    from dilu.runtime_support import execute_episode_step
    from dilu.safety.unified_safety import UnifiedSafetySystem
    from dilu.scenario import create_scenario
    from tools.run_main_table_runtime import build_group_config

    seed = _strict_int(seed, "factorial seed", nonnegative=True)
    if not proposals:
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
            for frame in range(int(cfg.get("simulation_duration", 1) or 1)):
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
                runtimes.append(time.perf_counter() - started)
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
            physical_payload, reasoning_payload = _save_factorial_episode(
                root=cell_root,
                seed=seed,
                prefix=prefix,
                events=events,
                pending=list(episode_state.get("latency_replay_queue", []) or []),
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
                "candidate_source_gate_independent": True,
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


def _arm_order(seed: int) -> list[FactorialArm]:
    arms = list(FACTORIAL_ARMS)
    random.Random(20260731 + int(seed)).shuffle(arms)
    return arms


def _run_seed_block(task: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    seed = int(task["seed"])
    rows = []
    order = []
    for index, arm in enumerate(_arm_order(seed)):
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
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--latency-profile", choices=("frozen", "stress"), default="frozen")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.run_main_table_runtime import load_formal_base_config, load_formal_protocol

    args = parse_args(argv)
    if args.seed_start < 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError("seed range and worker count must be positive")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    protocol = load_formal_protocol(args.protocol)
    base_cfg = load_formal_base_config(protocol, REPO_ROOT / "config.yaml")
    bank = load_proposal_bank(
        args.source_root,
        seeds,
        latency_profile=args.latency_profile,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    proposal_manifest = _proposal_manifest(
        bank,
        source_root=args.source_root,
        latency_profile=args.latency_profile,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
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
            "proposal_source_policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
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
            "latency_profile": str(args.latency_profile),
            "candidate_source_policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
            "candidate_source_gate_independent": True,
            "seed_start": int(args.seed_start),
            "seed_count": int(args.seeds),
            "arms": [asdict(arm) for arm in FACTORIAL_ARMS],
            "randomized_block_run_order": order,
            "result_rows": len(rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
