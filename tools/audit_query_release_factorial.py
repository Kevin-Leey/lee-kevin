"""Fail-closed request and snapshot audit for query x release factorial bundles.

The audit works from immutable bundle artifacts.  It authenticates the frozen
proposal bank, reconstructs every asynchronous request lifecycle, and verifies
that every response-release event has exactly one request-bound online snapshot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_ARMS,
    FACTORIAL_EVENT_SCHEMA,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
    FACTORIAL_RUN_SCHEMA,
)
from dilu.evaluation.release_snapshot import (  # noqa: E402
    RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
    RELEASE_SNAPSHOT_CAPTURE_STAGE,
    RELEASE_SNAPSHOT_SCHEMA,
    snapshot_manifest_row,
    validate_release_snapshot_policy_state,
)


AUDIT_SCHEMA = "rgd_query_release_factorial_request_audit_v3"
LEGACY_REPLAY_VERSION = "rgd_query_release_factorial_v2"
LEGACY_RUN_SCHEMA = "rgd_query_release_factorial_run_v2"
LEGACY_PROPOSAL_SCHEMA = "rgd_factorial_proposal_bank_v2"
LEGACY_EVENT_SCHEMA = "rgd_event_log_v2"
SOURCE_POLICY = "scheduled_always_slow"
ARM_BY_NAME = {arm.name: arm for arm in FACTORIAL_ARMS}
ARM_NAMES = tuple(arm.name for arm in FACTORIAL_ARMS)
SUPPORTED_REPLAY_VERSIONS = frozenset(
    {LEGACY_REPLAY_VERSION, FACTORIAL_REPLAY_VERSION}
)
TERMINAL_FLAGS = (
    "closed_loop_latency_release_event",
    "closed_loop_latency_timeout_event",
    "closed_loop_latency_failure_event",
)
EXPLICIT_LIFECYCLE_FIELDS = (
    "closed_loop_latency_issuance_event",
    "closed_loop_latency_issued_request_id",
    "closed_loop_latency_issued_response_outcome",
    "closed_loop_latency_terminal_event",
    "closed_loop_latency_terminal_request_id",
    "closed_loop_latency_terminal_response_outcome",
)


class AuditError(ValueError):
    """Raised when a factorial artifact cannot satisfy the audit contract."""


def _bundle_schemas(replay_version: str) -> tuple[str, str, str]:
    if replay_version == FACTORIAL_REPLAY_VERSION:
        return FACTORIAL_RUN_SCHEMA, FACTORIAL_PROPOSAL_SCHEMA, FACTORIAL_EVENT_SCHEMA
    if replay_version == LEGACY_REPLAY_VERSION:
        return LEGACY_RUN_SCHEMA, LEGACY_PROPOSAL_SCHEMA, LEGACY_EVENT_SCHEMA
    raise AuditError(f"unsupported factorial replay version: {replay_version!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _load_json(path: Path, *, allow_nonfinite: bool = False) -> Any:
    _require(path.is_file(), f"missing JSON artifact: {path}")

    def reject_constant(value: str) -> None:
        raise AuditError(f"{path}: non-finite JSON constant {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig"),
            parse_constant=(None if allow_nonfinite else reject_constant),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot parse JSON artifact {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    _require(path.is_file(), f"missing hashed artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditError("proposal bank is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"invalid {field}: {value!r}",
    )
    return digest


def _exact_int(value: Any, field: str, *, nonnegative: bool = False) -> int:
    _require(not isinstance(value, bool), f"{field} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid {field}: {value!r}") from exc
    _require(math.isfinite(number) and number == int(number), f"invalid {field}: {value!r}")
    result = int(number)
    if nonnegative:
        _require(result >= 0, f"negative {field}: {value!r}")
    return result


def _exact_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise AuditError(f"invalid {field}: {value!r}")


def _audit_candidate_coverage(
    *,
    candidates: Mapping[str, Mapping[str, Any]],
    proposal_records: Mapping[str, Mapping[str, Any]],
    event_payload: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    cell: str,
) -> tuple[int, set[str], set[str]]:
    """Authenticate candidate exposure while preserving terminal right censoring.

    Proposal records are indexed on the source trajectory.  An intervention arm
    can legitimately end before a later source-frame proposal is reached.  The
    audit therefore requires complete candidate coverage through the final
    executed frame and treats only post-terminal source records as censored.
    """
    _require(bool(events), f"{cell}: cannot assess proposal coverage without events")
    final_frame = len(events) - 1
    reachable_ids = {
        request_id
        for request_id, proposal in proposal_records.items()
        if _exact_int(
            proposal.get("source_frame"),
            f"{cell} proposal {request_id} source frame",
            nonnegative=True,
        )
        <= final_frame
    }
    censored_ids = set(proposal_records) - reachable_ids
    _require(
        set(candidates) == reachable_ids,
        f"{cell}: candidate/reachable-proposal coverage mismatch",
    )
    if censored_ids:
        final_event = _mapping(events[-1], f"{cell} final event")
        _require(
            _exact_bool(final_event.get("episode_done"), f"{cell} final event episode_done"),
            f"{cell}: post-terminal proposal censoring requires an ended episode",
        )
        terminal_cause = str(event_payload.get("terminal_cause", "") or "")
        _require(
            terminal_cause not in {"", "running"},
            f"{cell}: post-terminal proposal censoring lacks a terminal cause",
        )
    return final_frame, reachable_ids, censored_ids


def _audit_shared_candidate_identities(cells: Sequence[Mapping[str, Any]]) -> int:
    """Compare proposal identities only where every factorial arm reached them."""
    _require(bool(cells), "cross-arm audit block is empty")
    candidate_sets = [set(cell["candidate_identities"]) for cell in cells]
    shared_ids = set.intersection(*candidate_sets)
    reference = dict(cells[0]["candidate_identities"])
    expected = {request_id: reference[request_id] for request_id in shared_ids}
    comparisons = 0
    for cell in cells[1:]:
        observed = dict(cell["candidate_identities"])
        _require(
            {request_id: observed[request_id] for request_id in shared_ids} == expected,
            f"seed {cell['seed']}: cross-arm shared proposal identity drift in {cell['arm']}",
        )
        comparisons += len(shared_ids)
    return comparisons


def _finite_float(value: Any, field: str) -> float:
    _require(not isinstance(value, bool), f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"invalid {field}: {value!r}") from exc
    _require(math.isfinite(number), f"non-finite {field}: {value!r}")
    return number


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be a list")
    return value


def _single_path(paths: Iterable[Path], field: str) -> Path:
    resolved = list(paths)
    _require(len(resolved) == 1, f"{field}: expected one artifact, found {len(resolved)}")
    return resolved[0]


def _resolve_within(root: Path, declared: Any, field: str) -> Path:
    text = str(declared or "")
    _require(bool(text), f"{field} is empty")
    candidate = Path(text)
    _require(not candidate.is_absolute(), f"{field} must be relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError(f"{field} escapes its artifact root") from exc
    return resolved


def _read_result_rows(path: Path) -> list[Dict[str, str]]:
    _require(path.is_file(), f"missing factorial result table: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as exc:
        raise AuditError(f"cannot read factorial result table: {exc}") from exc
    _require(bool(rows), "factorial result table is empty")
    return rows


def _audit_proposal_bank(
    bundle: Path,
    run_manifest: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    replay_version = str(run_manifest.get("factorial_replay_version", "") or "")
    _require(replay_version in SUPPORTED_REPLAY_VERSIONS, "unsupported factorial replay version")
    _, proposal_schema, _ = _bundle_schemas(replay_version)
    _require(proposal_manifest.get("schema") == proposal_schema, "proposal-bank schema drift")
    _require(
        proposal_manifest.get("factorial_replay_version") == replay_version,
        "run/proposal replay version drift",
    )
    _require(
        run_manifest.get("candidate_source_policy") == SOURCE_POLICY
        and proposal_manifest.get("candidate_source_policy") == SOURCE_POLICY,
        "proposal source is not the gate-independent always-slow schedule",
    )
    _require(
        run_manifest.get("candidate_source_gate_independent") is True
        and proposal_manifest.get("candidate_source_gate_independent") is True,
        "proposal source is not declared gate-independent",
    )
    _require(
        run_manifest.get("latency_profile") == proposal_manifest.get("latency_profile"),
        "run/proposal latency-profile drift",
    )

    payload = _list(proposal_manifest.get("bank_payload"), "proposal bank payload")
    bank_hash = _valid_sha256(proposal_manifest.get("bank_sha256"), "proposal bank SHA256")
    _require(_sha256_json(payload) == bank_hash, "proposal-bank payload hash mismatch")
    _require(
        _valid_sha256(run_manifest.get("proposal_bank_sha256"), "run proposal-bank SHA256")
        == bank_hash,
        "run manifest is bound to a different proposal bank",
    )

    expected_seed_start = _exact_int(run_manifest.get("seed_start"), "run seed_start", nonnegative=True)
    expected_seed_count = _exact_int(run_manifest.get("seed_count"), "run seed_count", nonnegative=True)
    _require(expected_seed_count > 0, "run manifest has no seeds")
    expected_seeds = tuple(range(expected_seed_start, expected_seed_start + expected_seed_count))
    records_by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    global_request_ids: set[str] = set()
    proposal_count = 0
    observed_block_seeds = []
    for block_index, raw_block in enumerate(payload):
        block = _mapping(raw_block, f"proposal block {block_index}")
        seed = _exact_int(block.get("seed"), f"proposal block {block_index} seed", nonnegative=True)
        _require(seed not in records_by_seed, f"duplicate proposal seed block: {seed}")
        observed_block_seeds.append(seed)
        records: Dict[str, Dict[str, Any]] = {}
        source_frames: set[int] = set()
        for record_index, raw_record in enumerate(_list(block.get("records"), f"seed {seed} records")):
            location = f"proposal seed {seed} record {record_index}"
            record = _mapping(raw_record, location)
            record_seed = _exact_int(record.get("seed"), f"{location} seed", nonnegative=True)
            source_frame = _exact_int(record.get("source_frame"), f"{location} source_frame", nonnegative=True)
            request_id = str(record.get("request_id", "") or "")
            raw_action = _exact_int(record.get("raw_slow_action"), f"{location} raw_slow_action")
            latency_steps = _exact_int(record.get("latency_steps"), f"{location} latency_steps", nonnegative=True)
            outcome = str(record.get("outcome", "") or "")
            response_text = str(record.get("response_text", "") or "")
            response_sha = _valid_sha256(record.get("response_sha256"), f"{location} response SHA256")
            _require(record_seed == seed, f"{location}: seed/key mismatch")
            _require(bool(request_id), f"{location}: empty request ID")
            _require(request_id not in global_request_ids, f"duplicate proposal request ID: {request_id}")
            _require(source_frame not in source_frames, f"seed {seed}: duplicate proposal source frame {source_frame}")
            _require(raw_action in range(5), f"{location}: action outside discrete action universe")
            _require(outcome in {"valid", "timeout", "failure"}, f"{location}: invalid response outcome")
            _require(
                hashlib.sha256(response_text.encode("utf-8")).hexdigest() == response_sha,
                f"{location}: response text/hash mismatch",
            )
            normalized = {
                "seed": seed,
                "source_frame": source_frame,
                "request_id": request_id,
                "raw_slow_action": raw_action,
                "latency_steps": latency_steps,
                "outcome": outcome,
                "response_sha256": response_sha,
            }
            records[request_id] = normalized
            global_request_ids.add(request_id)
            source_frames.add(source_frame)
            proposal_count += 1
        _require(bool(records), f"seed {seed}: proposal bank has no candidates")
        records_by_seed[seed] = records

    _require(tuple(observed_block_seeds) == expected_seeds, "proposal seed blocks are missing, extra, or unsorted")
    _require(
        _exact_int(proposal_manifest.get("seed_count"), "proposal seed_count", nonnegative=True)
        == len(records_by_seed),
        "proposal seed-count mismatch",
    )
    _require(
        _exact_int(proposal_manifest.get("proposal_count"), "proposal_count", nonnegative=True)
        == proposal_count,
        "proposal-count mismatch",
    )
    _require(proposal_count > 0, "proposal bank has no candidates")

    source_root_text = str(proposal_manifest.get("source_root", "") or "")
    _require(bool(source_root_text), "proposal source_root is empty")
    source_root = Path(source_root_text).resolve()
    _require(source_root.is_dir(), f"proposal source root is unavailable: {source_root}")
    source_rows = _list(proposal_manifest.get("source_artifacts"), "proposal source artifacts")
    _require(len(source_rows) == len(expected_seeds), "proposal source-artifact seed coverage mismatch")
    source_hashes_verified = 0
    observed_source_seeds = []
    for row_index, raw_row in enumerate(source_rows):
        row = _mapping(raw_row, f"source-artifact row {row_index}")
        seed = _exact_int(row.get("seed"), f"source-artifact row {row_index} seed", nonnegative=True)
        observed_source_seeds.append(seed)
        resolved_by_kind: Dict[str, Path] = {}
        for kind in ("event_log", "reasoning_trace", "experiment_snapshot"):
            declaration = _mapping(row.get(kind), f"source seed {seed} {kind}")
            path = _resolve_within(source_root, declaration.get("path"), f"source seed {seed} {kind} path")
            expected_hash = _valid_sha256(declaration.get("sha256"), f"source seed {seed} {kind} SHA256")
            _require(_sha256_file(path) == expected_hash, f"source seed {seed} {kind} file hash mismatch")
            resolved_by_kind[kind] = path
            source_hashes_verified += 1
        source_snapshot = _mapping(
            _load_json(resolved_by_kind["experiment_snapshot"]),
            f"source seed {seed} experiment snapshot",
        )
        source_cfg = _mapping(source_snapshot.get("config"), f"source seed {seed} config")
        routing = _mapping(source_cfg.get("system_routing"), f"source seed {seed} routing")
        _require(
            _exact_int(source_snapshot.get("fixed_seed_override"), f"source seed {seed} fixed seed") == seed,
            f"source seed {seed}: seed provenance mismatch",
        )
        _require(source_cfg.get("protocol_name") == "always_slow", f"source seed {seed}: protocol is not always_slow")
        _require(
            routing.get("simple") == "slow" and routing.get("complex") == "slow",
            f"source seed {seed}: routing is not forced slow",
        )
    _require(tuple(observed_source_seeds) == expected_seeds, "proposal source-artifact rows are missing, extra, or unsorted")

    return {
        "bank_sha256": bank_hash,
        "replay_version": replay_version,
        "latency_profile": str(run_manifest.get("latency_profile", "") or ""),
        "seeds": expected_seeds,
        "records_by_seed": records_by_seed,
        "proposal_count": proposal_count,
        "source_artifact_files_verified": source_hashes_verified,
    }


def _audit_run_contract(
    run_manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    bank_sha256: str,
    replay_version: str,
) -> Dict[tuple[int, str], Mapping[str, Any]]:
    run_schema, _, _ = _bundle_schemas(replay_version)
    _require(run_manifest.get("schema") == run_schema, "factorial run schema drift")
    _require(
        run_manifest.get("factorial_replay_version") == replay_version,
        "factorial replay version drift",
    )
    expected_arms = [asdict(arm) for arm in FACTORIAL_ARMS]
    _require(run_manifest.get("arms") == expected_arms, "factorial arm declaration drift")
    expected_cells = {(int(seed), arm) for seed in seeds for arm in ARM_NAMES}
    _require(
        _exact_int(run_manifest.get("result_rows"), "run result_rows", nonnegative=True)
        == len(expected_cells),
        "run result-row count mismatch",
    )
    run_order = _list(run_manifest.get("randomized_block_run_order"), "randomized block run order")
    observed_order_cells: set[tuple[int, str]] = set()
    orders_by_seed: Dict[int, set[int]] = {int(seed): set() for seed in seeds}
    for index, raw in enumerate(run_order):
        item = _mapping(raw, f"run-order row {index}")
        seed = _exact_int(item.get("seed"), f"run-order row {index} seed", nonnegative=True)
        arm = str(item.get("arm", "") or "")
        order = _exact_int(item.get("order"), f"run-order row {index} order", nonnegative=True)
        _require((seed, arm) in expected_cells, f"run-order row {index}: unknown factorial cell")
        _require((seed, arm) not in observed_order_cells, f"run-order row {index}: duplicate factorial cell")
        observed_order_cells.add((seed, arm))
        orders_by_seed[seed].add(order)
    _require(observed_order_cells == expected_cells, "randomized run order does not cover the factorial matrix")
    for seed in seeds:
        _require(orders_by_seed[int(seed)] == set(range(len(ARM_NAMES))), f"seed {seed}: invalid within-block run order")

    row_by_cell: Dict[tuple[int, str], Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        seed = _exact_int(row.get("seed"), f"result row {index} seed", nonnegative=True)
        arm_name = str(row.get("arm", "") or "")
        cell = (seed, arm_name)
        _require(cell in expected_cells, f"result row {index}: unknown factorial cell {cell}")
        _require(cell not in row_by_cell, f"duplicate factorial result row: {cell}")
        arm = ARM_BY_NAME[arm_name]
        _require(row.get("factorial_replay_version") == replay_version, f"{cell}: result replay-version drift")
        _require(_exact_bool(row.get("query_gate_enabled"), f"{cell} query flag") is arm.query_gate_enabled, f"{cell}: query flag drift")
        _require(_exact_bool(row.get("release_guard_enabled"), f"{cell} release flag") is arm.release_guard_enabled, f"{cell}: release flag drift")
        _require(_valid_sha256(row.get("proposal_bank_sha256"), f"{cell} bank SHA256") == bank_sha256, f"{cell}: result row is bound to another proposal bank")
        _require(row.get("candidate_source_policy") == SOURCE_POLICY, f"{cell}: result proposal-source policy drift")
        _require(_exact_bool(row.get("candidate_source_gate_independent"), f"{cell} gate-independent flag"), f"{cell}: result source is not gate-independent")
        row_by_cell[cell] = row
    _require(set(row_by_cell) == expected_cells, "factorial result table is incomplete")
    return row_by_cell


def _required_event_field(event: Mapping[str, Any], name: str, location: str) -> Any:
    _require(name in event, f"{location}: legacy/partial event omits required field {name}")
    return event[name]


def _request_timing(
    event: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    frame: int,
    terminal_kind: Optional[str],
    location: str,
) -> None:
    source_frame = _exact_int(_required_event_field(event, "closed_loop_latency_source_frame", location), f"{location} source frame", nonnegative=True)
    scheduled_steps = _exact_int(_required_event_field(event, "closed_loop_latency_scheduled_steps", location), f"{location} scheduled steps", nonnegative=True)
    delay_steps = _exact_int(_required_event_field(event, "closed_loop_latency_delay_steps", location), f"{location} delay steps", nonnegative=True)
    scheduled_frame = _exact_int(_required_event_field(event, "closed_loop_latency_scheduled_release_frame", location), f"{location} scheduled frame", nonnegative=True)
    frequency = _finite_float(_required_event_field(event, "closed_loop_latency_policy_frequency_hz", location), f"{location} policy frequency")
    scheduled_seconds = _finite_float(_required_event_field(event, "closed_loop_latency_scheduled_seconds", location), f"{location} scheduled seconds")
    _require(source_frame == int(request["source_frame"]), f"{location}: request source-frame drift")
    _require(scheduled_steps == int(request["latency_steps"]) == delay_steps, f"{location}: request latency-step drift")
    _require(scheduled_frame == source_frame + scheduled_steps, f"{location}: scheduled release-frame drift")
    _require(frequency > 0.0, f"{location}: nonpositive policy frequency")
    _require(math.isclose(scheduled_seconds, scheduled_steps / frequency, rel_tol=0.0, abs_tol=1e-9), f"{location}: scheduled seconds/steps drift")
    _require(str(_required_event_field(event, "closed_loop_latency_source_system", location) or "") == "slow", f"{location}: request source system drift")
    _require(str(_required_event_field(event, "closed_loop_latency_response_outcome", location) or "") == request["outcome"], f"{location}: response-outcome drift")
    _require(source_frame <= frame, f"{location}: request appears before its source frame")

    realized_steps = _exact_int(_required_event_field(event, "closed_loop_latency_realized_steps", location), f"{location} realized steps")
    realized_available = _exact_bool(_required_event_field(event, "closed_loop_latency_realized_available", location), f"{location} realized available")
    realized_source = str(_required_event_field(event, "closed_loop_latency_realized_source", location) or "")
    realized_raw = _required_event_field(event, "closed_loop_latency_realized_seconds", location)
    terminal_outcome = str(_required_event_field(event, "closed_loop_latency_terminal_outcome", location) or "")
    if terminal_kind is None:
        _require(terminal_outcome == "pending", f"{location}: nonterminal request is not pending")
        _require(realized_steps == -1 and realized_available is False, f"{location}: pending request claims realized latency")
        _require(realized_source == "not_released", f"{location}: pending realized-latency source drift")
        try:
            realized_value = float(realized_raw)
        except (TypeError, ValueError) as exc:
            raise AuditError(f"{location}: invalid pending realized seconds") from exc
        _require(math.isnan(realized_value), f"{location}: pending request has finite realized seconds")
        return

    _require(frame >= scheduled_frame, f"{location}: request terminates before its scheduled frame")
    _require(realized_available is True, f"{location}: terminal request lacks realized latency")
    _require(realized_source == "simulator_frame_delta", f"{location}: realized-latency source drift")
    _require(realized_steps == frame - source_frame, f"{location}: realized latency/frame delta drift")
    realized_seconds = _finite_float(realized_raw, f"{location} realized seconds")
    _require(math.isclose(realized_seconds, realized_steps / frequency, rel_tol=0.0, abs_tol=1e-9), f"{location}: realized seconds/steps drift")
    if terminal_kind == "timeout":
        _require(terminal_outcome == "timeout" and request["outcome"] == "timeout", f"{location}: timeout outcome drift")
    elif terminal_kind == "failure":
        _require(terminal_outcome == "failure" and request["outcome"] == "failure", f"{location}: failure outcome drift")
    else:
        _require(request["outcome"] == "valid", f"{location}: non-valid response emitted a release")
        _require(terminal_outcome not in {"", "pending", "timeout", "failure"}, f"{location}: invalid release terminal outcome")


def _audit_snapshots(
    seed_dir: Path,
    event_payload: Mapping[str, Any],
    *,
    seed: int,
    release_events: Mapping[str, Mapping[str, Any]],
    timeout_ids: set[str],
    failure_ids: set[str],
    pending_ids: set[str],
    cell: str,
) -> Dict[str, Any]:
    declared_count = _exact_int(event_payload.get("release_snapshot_count"), f"{cell} release snapshot count", nonnegative=True)
    _require(declared_count == len(release_events), f"{cell}: release/snapshot top-level count mismatch")
    snapshot_dir = seed_dir / "release_snapshots"
    snapshot_files = sorted(path for path in snapshot_dir.glob("*") if path.is_file()) if snapshot_dir.is_dir() else []
    if not release_events:
        _require(not snapshot_files, f"{cell}: orphan snapshot files without a release")
        for field in ("release_snapshot_bundle", "release_snapshot_manifest", "release_snapshot_bundle_sha256"):
            _require(event_payload.get(field) in (None, ""), f"{cell}: orphan snapshot declaration {field}")
        return {"snapshot_count": 0, "manifest_sha256": None, "bundle_sha256": None}

    bundle_path = _resolve_within(seed_dir, event_payload.get("release_snapshot_bundle"), f"{cell} snapshot bundle path")
    manifest_path = _resolve_within(seed_dir, event_payload.get("release_snapshot_manifest"), f"{cell} snapshot manifest path")
    _require(bundle_path.parent == snapshot_dir.resolve() and manifest_path.parent == snapshot_dir.resolve(), f"{cell}: snapshot files are outside release_snapshots")
    _require(set(snapshot_files) == {bundle_path, manifest_path}, f"{cell}: missing or orphan snapshot files")
    declared_bundle_sha = _valid_sha256(event_payload.get("release_snapshot_bundle_sha256"), f"{cell} event bundle SHA256")
    actual_bundle_sha = _sha256_file(bundle_path)
    _require(actual_bundle_sha == declared_bundle_sha, f"{cell}: snapshot bundle file SHA256 mismatch")

    manifest = _mapping(_load_json(manifest_path), f"{cell} snapshot manifest")
    _require(manifest.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA, f"{cell}: snapshot manifest schema drift")
    _require(_exact_int(manifest.get("episode_id"), f"{cell} snapshot manifest episode") == seed, f"{cell}: snapshot manifest episode drift")
    _require(_exact_int(manifest.get("snapshot_count"), f"{cell} manifest count", nonnegative=True) == len(release_events), f"{cell}: snapshot manifest count mismatch")
    _require(manifest.get("bundle_file") == bundle_path.name, f"{cell}: snapshot manifest bundle link drift")
    _require(_valid_sha256(manifest.get("bundle_sha256"), f"{cell} manifest bundle SHA256") == actual_bundle_sha, f"{cell}: snapshot manifest bundle hash drift")
    manifest_rows = _list(manifest.get("snapshots"), f"{cell} snapshot rows")
    _require(len(manifest_rows) == len(release_events), f"{cell}: snapshot row count mismatch")

    try:
        with bundle_path.open("rb") as handle:
            bundle_payload = pickle.load(handle)
    except Exception as exc:
        raise AuditError(f"{cell}: cannot load snapshot pickle: {exc}") from exc
    bundle_mapping = _mapping(bundle_payload, f"{cell} snapshot pickle")
    _require(bundle_mapping.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA, f"{cell}: snapshot pickle schema drift")
    _require(_exact_int(bundle_mapping.get("episode_id"), f"{cell} snapshot pickle episode") == seed, f"{cell}: snapshot pickle episode drift")
    snapshots = _mapping(bundle_mapping.get("snapshots"), f"{cell} snapshot pickle map")
    snapshot_ids = {str(request_id) for request_id in snapshots}
    release_ids = set(release_events)
    _require(snapshot_ids == release_ids, f"{cell}: release iff exactly one snapshot violated")
    _require(not snapshot_ids.intersection(timeout_ids | failure_ids | pending_ids), f"{cell}: timeout/failure/pending request has a snapshot")

    rows_by_id: Dict[str, Mapping[str, Any]] = {}
    for row_index, raw_row in enumerate(manifest_rows):
        row = _mapping(raw_row, f"{cell} snapshot row {row_index}")
        request_id = str(row.get("request_id", "") or "")
        _require(bool(request_id) and request_id not in rows_by_id, f"{cell}: duplicate/empty snapshot manifest request ID")
        rows_by_id[request_id] = row
    _require(set(rows_by_id) == release_ids, f"{cell}: snapshot manifest request coverage drift")

    for request_id in sorted(release_ids):
        snapshot = snapshots[request_id]
        try:
            validate_release_snapshot_policy_state(snapshot, context=f"{cell} request {request_id}")
            derived_row = snapshot_manifest_row(snapshot)
        except (TypeError, ValueError) as exc:
            raise AuditError(f"{cell}: unauthenticated snapshot for {request_id}: {exc}") from exc
        _require(rows_by_id[request_id] == derived_row, f"{cell}: snapshot manifest/pickle metadata drift for {request_id}")
        identity = _valid_sha256(getattr(snapshot, "snapshot_identity_sha256", ""), f"{cell} snapshot identity")
        _valid_sha256(getattr(snapshot, "policy_state_sha256", ""), f"{cell} snapshot policy-state SHA256")
        event = release_events[request_id]
        event_frame = _exact_int(event.get("frame"), f"{cell} release frame", nonnegative=True)
        _require(str(getattr(snapshot, "schema", "")) == RELEASE_SNAPSHOT_SCHEMA, f"{cell}: release snapshot schema drift")
        _require(str(getattr(snapshot, "capture_stage", "")) == RELEASE_SNAPSHOT_CAPTURE_STAGE, f"{cell}: release snapshot capture-stage drift")
        _require(str(getattr(snapshot, "request_id", "") or "") == request_id, f"{cell}: snapshot request metadata drift")
        _require(int(getattr(snapshot, "frame", -1)) == event_frame, f"{cell}: snapshot/release frame drift")
        _require(int(getattr(snapshot, "source_frame", -1)) == _exact_int(event.get("closed_loop_latency_source_frame"), f"{cell} event source frame"), f"{cell}: snapshot/event source-frame drift")
        _require(int(getattr(snapshot, "scheduled_release_frame", -1)) == _exact_int(event.get("closed_loop_latency_scheduled_release_frame"), f"{cell} event scheduled frame"), f"{cell}: snapshot/event schedule drift")
        _require(str(event.get("closed_loop_release_snapshot_identity_sha256", "") or "") == identity, f"{cell}: event/snapshot identity hash drift")
        _require(event.get("closed_loop_release_snapshot_schema") == RELEASE_SNAPSHOT_SCHEMA, f"{cell}: event snapshot schema drift")
        _require(event.get("closed_loop_release_snapshot_capture_stage") == RELEASE_SNAPSHOT_CAPTURE_STAGE, f"{cell}: event snapshot capture-stage drift")

    return {
        "snapshot_count": len(snapshot_ids),
        "manifest_sha256": _sha256_file(manifest_path),
        "bundle_sha256": actual_bundle_sha,
    }


def _audit_cell(
    seed_dir: Path,
    *,
    seed: int,
    arm_name: str,
    bank_sha256: str,
    replay_version: str,
    proposal_records: Mapping[str, Mapping[str, Any]],
    result_row: Mapping[str, Any],
) -> Dict[str, Any]:
    cell = f"seed {seed} arm {arm_name}"
    event_path = _single_path(seed_dir.glob("event_logs/event_log_*.json"), f"{cell} event log")
    event_payload = _mapping(_load_json(event_path, allow_nonfinite=True), f"{cell} event log")
    _, _, event_schema = _bundle_schemas(replay_version)
    _require(event_payload.get("schema_version") == event_schema, f"{cell}: unsupported/legacy event schema; request lifecycle cannot be authenticated")
    _require(_exact_int(event_payload.get("episode_id"), f"{cell} episode ID") == seed, f"{cell}: episode ID drift")
    events = _list(event_payload.get("events"), f"{cell} events")
    _require(bool(events), f"{cell}: empty event trace")
    _require(_exact_int(event_payload.get("event_count"), f"{cell} event count", nonnegative=True) == len(events), f"{cell}: event-count mismatch")
    for index, event in enumerate(events):
        _mapping(event, f"{cell} event {index}")
        _require(_exact_int(event.get("frame"), f"{cell} event {index} frame", nonnegative=True) == index, f"{cell}: event frames must be contiguous and ordered")

    explicit_presence = [all(field in event for field in EXPLICIT_LIFECYCLE_FIELDS) for event in events]
    explicit_any = any(any(field in event for field in EXPLICIT_LIFECYCLE_FIELDS) for event in events)
    if replay_version == FACTORIAL_REPLAY_VERSION:
        _require(
            explicit_any and all(explicit_presence),
            f"{cell}: v5 requires the explicit dual-event lifecycle contract",
        )
    if explicit_any:
        _require(all(explicit_presence), f"{cell}: partially upgraded lifecycle schema fails closed")
        lifecycle_mode = "explicit_dual_event_ids"
    else:
        lifecycle_mode = "legacy_v2_single_request_projection"

    arm = ARM_BY_NAME[arm_name]
    candidates: Dict[str, Dict[str, Any]] = {}
    issued: Dict[str, Dict[str, Any]] = {}
    terminals: Dict[str, tuple[str, Dict[str, Any]]] = {}
    request_occurrences: Dict[str, list[Dict[str, Any]]] = {}
    for index, raw_event in enumerate(events):
        event = dict(raw_event)
        location = f"{cell} frame {index}"
        _require(event.get("factorial_replay_version") == replay_version, f"{location}: replay-version drift")
        _require(event.get("factorial_arm") == arm_name, f"{location}: arm identity drift")
        _require(_exact_bool(event.get("factorial_query_gate_enabled"), f"{location} query flag") is arm.query_gate_enabled, f"{location}: query arm flag drift")
        _require(_exact_bool(event.get("factorial_release_guard_enabled"), f"{location} release flag") is arm.release_guard_enabled, f"{location}: release arm flag drift")
        _require(_valid_sha256(event.get("factorial_proposal_bank_sha256"), f"{location} proposal-bank SHA256") == bank_sha256, f"{location}: event bound to another proposal bank")

        candidate = _exact_bool(event.get("factorial_candidate_query"), f"{location} candidate flag")
        candidate_issued = _exact_bool(event.get("factorial_query_issued"), f"{location} issued flag")
        terminal_values = [_exact_bool(event.get(field), f"{location} {field}") for field in TERMINAL_FLAGS]
        _require(sum(terminal_values) <= 1, f"{location}: multiple terminal outcomes")
        terminal_kind = None
        if terminal_values[0]:
            terminal_kind = "release"
        elif terminal_values[1]:
            terminal_kind = "timeout"
        elif terminal_values[2]:
            terminal_kind = "failure"

        if lifecycle_mode == "legacy_v2_single_request_projection":
            _require(not (candidate_issued and terminal_kind), f"{location}: legacy event cannot disambiguate simultaneous issuance and terminal IDs")
            issued_event = candidate_issued
            issued_id = str(event.get("closed_loop_latency_request_id", "") or "") if issued_event else ""
            terminal_event = terminal_kind is not None
            terminal_id = str(event.get("closed_loop_latency_request_id", "") or "") if terminal_event else ""
        else:
            issued_event = _exact_bool(event.get("closed_loop_latency_issuance_event"), f"{location} explicit issuance flag")
            issued_id = str(event.get("closed_loop_latency_issued_request_id", "") or "")
            terminal_event = _exact_bool(event.get("closed_loop_latency_terminal_event"), f"{location} explicit terminal flag")
            terminal_id = str(event.get("closed_loop_latency_terminal_request_id", "") or "")
            _require(issued_event == candidate_issued, f"{location}: explicit/factorial issuance drift")
            _require(terminal_event == (terminal_kind is not None), f"{location}: explicit terminal flag drift")
            _require((bool(issued_id) == issued_event) and (bool(terminal_id) == terminal_event), f"{location}: explicit lifecycle ID presence drift")
            if issued_event:
                _require(event.get("closed_loop_latency_issued_response_outcome") == event.get("factorial_shared_response_outcome"), f"{location}: explicit issued outcome drift")
            if terminal_event:
                _require(event.get("closed_loop_latency_terminal_response_outcome") == event.get("closed_loop_latency_response_outcome"), f"{location}: explicit terminal outcome drift")

        if candidate:
            candidate_id = str(event.get("factorial_candidate_request_id", "") or "")
            _require(candidate_id in proposal_records, f"{location}: candidate is absent from the proposal bank")
            _require(candidate_id not in candidates, f"{location}: duplicate candidate request ID")
            proposal = proposal_records[candidate_id]
            _require(index == proposal["source_frame"], f"{location}: candidate source-frame drift")
            _require(_exact_int(event.get("factorial_shared_raw_slow_action"), f"{location} shared action") == proposal["raw_slow_action"], f"{location}: shared proposal action drift")
            _require(_exact_int(event.get("factorial_shared_latency_steps"), f"{location} shared latency", nonnegative=True) == proposal["latency_steps"], f"{location}: shared proposal latency drift")
            _require(_valid_sha256(event.get("factorial_shared_response_sha256"), f"{location} shared response SHA256") == proposal["response_sha256"], f"{location}: shared proposal response hash drift")
            _require(event.get("factorial_shared_response_outcome") == proposal["outcome"], f"{location}: shared proposal outcome drift")
            gate_pass = _exact_bool(event.get("factorial_query_gate_pass"), f"{location} gate pass")
            expected_issue = bool((not arm.query_gate_enabled) or gate_pass)
            _require(candidate_issued == expected_issue, f"{location}: query-arm issuance contract drift")
            candidates[candidate_id] = {
                "request_id": candidate_id,
                "source_frame": proposal["source_frame"],
                "raw_slow_action": proposal["raw_slow_action"],
                "latency_steps": proposal["latency_steps"],
                "outcome": proposal["outcome"],
                "response_sha256": proposal["response_sha256"],
            }
        else:
            _require(not candidate_issued, f"{location}: issued flag without a candidate")

        if issued_event:
            _require(issued_id in proposal_records and issued_id in candidates, f"{location}: orphan issued request ID")
            _require(issued_id not in issued, f"{location}: duplicate request issuance")
            _require(issued_id == str(event.get("factorial_candidate_request_id", "") or ""), f"{location}: candidate/issued request ID drift")
            _require(_exact_bool(event.get("factorial_policy_state_synchronized"), f"{location} policy-state sync"), f"{location}: issued request lacks policy-state synchronization")
            issued[issued_id] = event

        if terminal_event:
            _require(bool(terminal_id), f"{location}: terminal request ID is empty")
            _require(terminal_id not in terminals, f"{location}: duplicate terminal event for request {terminal_id}")
            terminals[terminal_id] = (str(terminal_kind), event)

        projected_id = str(event.get("closed_loop_latency_request_id", "") or "")
        if projected_id:
            request_occurrences.setdefault(projected_id, []).append(event)
        else:
            _require(terminal_kind is None, f"{location}: terminal event lacks projected request ID")
            _require(not _exact_bool(event.get("closed_loop_release_snapshot_captured"), f"{location} snapshot flag"), f"{location}: snapshot flag without request")

    final_frame, reachable_proposal_ids, censored_proposal_ids = _audit_candidate_coverage(
        candidates=candidates,
        proposal_records=proposal_records,
        event_payload=event_payload,
        events=events,
        cell=cell,
    )
    _require(set(terminals).issubset(issued), f"{cell}: orphan terminal request IDs {sorted(set(terminals) - set(issued))}")
    _require(set(request_occurrences).issubset(issued), f"{cell}: orphan projected request IDs {sorted(set(request_occurrences) - set(issued))}")

    pending_rows = _list(event_payload.get("pending_releases_dropped_at_episode_end"), f"{cell} pending requests")
    _require(_exact_int(event_payload.get("pending_release_count"), f"{cell} pending count", nonnegative=True) == len(pending_rows), f"{cell}: pending count/list mismatch")
    pending: Dict[str, Mapping[str, Any]] = {}
    for row_index, raw_pending in enumerate(pending_rows):
        row = _mapping(raw_pending, f"{cell} pending row {row_index}")
        request_id = str(row.get("request_id", "") or "")
        _require(bool(request_id) and request_id not in pending, f"{cell}: duplicate/empty pending request ID")
        _require(request_id in issued, f"{cell}: orphan pending request ID {request_id}")
        proposal = proposal_records[request_id]
        _require(_exact_int(row.get("source_frame"), f"{cell} pending source frame", nonnegative=True) == proposal["source_frame"], f"{cell}: pending source-frame drift")
        _require(_exact_int(row.get("release_frame"), f"{cell} pending release frame", nonnegative=True) == proposal["source_frame"] + proposal["latency_steps"], f"{cell}: pending scheduled-frame drift")
        _require(row.get("response_outcome") == proposal["outcome"], f"{cell}: pending response-outcome drift")
        # Event-log v3 originally relied on this enclosing list name as the
        # terminal marker.  Newer exports also carry the row-local marker.
        # Accept the former only when the field is absent; an emitted marker
        # must still state the same lifecycle outcome.
        if "terminal_outcome" in row:
            _require(
                row.get("terminal_outcome") == "dropped_at_episode_end",
                f"{cell}: pending terminal marker drift",
            )
        pending[request_id] = row
    pending_ids = set(pending)
    terminal_ids = set(terminals)
    _require(not pending_ids.intersection(terminal_ids), f"{cell}: request is both terminal and pending")
    _require(set(issued) == terminal_ids | pending_ids, f"{cell}: issue != release+timeout+failure+pending")

    for request_id, occurrence_events in request_occurrences.items():
        proposal = proposal_records[request_id]
        terminal = terminals.get(request_id)
        terminal_frame = None if terminal is None else _exact_int(terminal[1].get("frame"), f"{cell} terminal frame", nonnegative=True)
        for event in occurrence_events:
            frame = _exact_int(event.get("frame"), f"{cell} request occurrence frame", nonnegative=True)
            event_terminal = terminal is not None and event is terminal[1]
            _require(terminal_frame is None or frame <= terminal_frame, f"{cell}: request {request_id} appears after termination")
            _request_timing(
                event,
                request=proposal,
                frame=frame,
                terminal_kind=(terminal[0] if event_terminal else None),
                location=f"{cell} request {request_id} frame {frame}",
            )
            captured = _exact_bool(event.get("closed_loop_release_snapshot_captured"), f"{cell} snapshot captured flag")
            identity = str(event.get("closed_loop_release_snapshot_identity_sha256", "") or "")
            _require(captured is bool(event_terminal and terminal[0] == "release"), f"{cell}: release iff snapshot-captured event flag violated")
            _require(bool(identity) == captured, f"{cell}: snapshot identity presence drift")
            if identity:
                _valid_sha256(identity, f"{cell} event snapshot identity")

    release_events = {request_id: event for request_id, (kind, event) in terminals.items() if kind == "release"}
    timeout_ids = {request_id for request_id, (kind, _) in terminals.items() if kind == "timeout"}
    failure_ids = {request_id for request_id, (kind, _) in terminals.items() if kind == "failure"}
    snapshot_report = _audit_snapshots(
        seed_dir,
        event_payload,
        seed=seed,
        release_events=release_events,
        timeout_ids=timeout_ids,
        failure_ids=failure_ids,
        pending_ids=pending_ids,
        cell=cell,
    )

    counts = {
        "candidate_queries": len(candidates),
        "issued_queries": len(issued),
        "query_gate_rejections": len(candidates) - len(issued),
        "timeouts": len(timeout_ids),
        "failure_events": len(failure_ids),
        "release_events": len(release_events),
        "pending_at_episode_end": len(pending_ids),
        "pending_timeouts_at_episode_end": sum(proposal_records[item]["outcome"] == "timeout" for item in pending_ids),
        "snapshot_count": int(snapshot_report["snapshot_count"]),
        "scheduled_timeouts": sum(proposal_records[item]["outcome"] == "timeout" for item in issued),
    }
    for field, observed in counts.items():
        expected = _exact_int(result_row.get(field), f"{cell} result {field}", nonnegative=True)
        _require(expected == observed, f"{cell}: result/event {field} mismatch ({expected} != {observed})")
    _require(
        counts["issued_queries"]
        == counts["release_events"] + counts["timeouts"] + counts["failure_events"] + counts["pending_at_episode_end"],
        f"{cell}: issued-request terminal accounting mismatch",
    )

    return {
        "seed": seed,
        "arm": arm_name,
        "accepted": True,
        "event_schema": event_schema,
        "lifecycle_mode": lifecycle_mode,
        "event_count": len(events),
        "final_executed_frame": final_frame,
        "reachable_proposal_count": len(reachable_proposal_ids),
        "right_censored_proposal_count": len(censored_proposal_ids),
        "right_censored_proposal_ids": sorted(censored_proposal_ids),
        **counts,
        "event_log_sha256": _sha256_file(event_path),
        "snapshot_manifest_sha256": snapshot_report["manifest_sha256"],
        "snapshot_bundle_sha256": snapshot_report["bundle_sha256"],
        "candidate_identities": candidates,
    }


def audit_bundle(bundle: Path) -> Dict[str, Any]:
    """Audit a complete four-arm factorial bundle and return a JSON-ready report."""
    bundle = Path(bundle).resolve()
    _require(bundle.is_dir(), f"factorial bundle does not exist: {bundle}")
    run_manifest = _mapping(_load_json(bundle / "factorial_run_manifest.json"), "factorial run manifest")
    proposal_manifest = _mapping(_load_json(bundle / "proposal_bank_manifest.json"), "proposal bank manifest")
    proposal = _audit_proposal_bank(bundle, run_manifest, proposal_manifest)
    rows = _read_result_rows(bundle / "factorial_episode_results.csv")
    row_by_cell = _audit_run_contract(
        run_manifest,
        rows,
        seeds=proposal["seeds"],
        bank_sha256=proposal["bank_sha256"],
        replay_version=proposal["replay_version"],
    )

    cells = []
    for seed in proposal["seeds"]:
        for arm_name in ARM_NAMES:
            arm_root = bundle / arm_name
            _require(arm_root.is_dir(), f"missing factorial arm directory: {arm_root}")
            expected_seed_dirs = {f"seed_{int(item)}" for item in proposal["seeds"]}
            observed_seed_dirs = {path.name for path in arm_root.glob("seed_*") if path.is_dir()}
            _require(observed_seed_dirs == expected_seed_dirs, f"arm {arm_name}: seed-directory coverage mismatch")
            cells.append(
                _audit_cell(
                    arm_root / f"seed_{int(seed)}",
                    seed=int(seed),
                    arm_name=arm_name,
                    bank_sha256=proposal["bank_sha256"],
                    replay_version=proposal["replay_version"],
                    proposal_records=proposal["records_by_seed"][int(seed)],
                    result_row=row_by_cell[(int(seed), arm_name)],
                )
            )

    by_arm = []
    aggregate_fields = (
        "event_count",
        "reachable_proposal_count",
        "right_censored_proposal_count",
        "candidate_queries",
        "issued_queries",
        "release_events",
        "timeouts",
        "failure_events",
        "pending_at_episode_end",
        "snapshot_count",
    )
    for arm_name in ARM_NAMES:
        selected = [cell for cell in cells if cell["arm"] == arm_name]
        by_arm.append(
            {
                "arm": arm_name,
                "seed_count": len(selected),
                **{field: sum(int(cell[field]) for cell in selected) for field in aggregate_fields},
            }
        )

    cross_arm_comparisons = 0
    for seed in proposal["seeds"]:
        seed_cells = [cell for cell in cells if cell["seed"] == seed]
        _require(len(seed_cells) == len(ARM_NAMES), f"seed {seed}: incomplete cross-arm audit block")
        cross_arm_comparisons += _audit_shared_candidate_identities(seed_cells)

    public_cells = []
    for cell in cells:
        public_cells.append({key: value for key, value in cell.items() if key != "candidate_identities"})
    aggregate = {
        "seed_count": len(proposal["seeds"]),
        "arm_count": len(ARM_NAMES),
        "arm_seed_cells": len(cells),
        "proposal_count": int(proposal["proposal_count"]),
        "source_artifact_files_verified": int(proposal["source_artifact_files_verified"]),
        "cross_arm_candidate_identity_comparisons": int(cross_arm_comparisons),
        **{field: sum(int(cell[field]) for cell in cells) for field in aggregate_fields},
    }
    _require(aggregate["release_events"] == aggregate["snapshot_count"], "aggregate release/snapshot mismatch")
    _require(
        aggregate["issued_queries"]
        == aggregate["release_events"] + aggregate["timeouts"] + aggregate["failure_events"] + aggregate["pending_at_episode_end"],
        "aggregate request lifecycle mismatch",
    )

    return {
        "schema": AUDIT_SCHEMA,
        "accepted": True,
        "bundle": str(bundle),
        "factorial_replay_version": proposal["replay_version"],
        "latency_profile": proposal["latency_profile"],
        "proposal_bank_sha256": proposal["bank_sha256"],
        "audit_contract": {
            "independent_unit": "simulator_seed",
            "proposal_bank_hash_authenticated": True,
            "proposal_source_files_authenticated": True,
            "all_candidate_records_bound_to_bank": True,
            "all_request_ids_lifecycle_closed": True,
            "release_iff_one_authenticated_snapshot": True,
            "timeout_failure_pending_snapshot_forbidden": True,
            "cross_arm_candidate_identity_authenticated": True,
            "cross_arm_comparison_scope": "common_reachable_proposals",
            "right_censoring_policy": "proposal records after a verified terminal frame are reported as right-censored; records at or before that frame require candidate coverage",
            "legacy_event_policy": "accept only unambiguous single-request projection; partial or simultaneous unidentifiable lifecycle fails closed",
        },
        "aggregate": aggregate,
        "by_arm": by_arm,
        "cells": public_cells,
        "errors": [],
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    bundle = args.bundle.resolve()
    output = args.output or (bundle / "analysis" / "factorial_request_audit.json")
    try:
        report = audit_bundle(bundle)
        status = 0
    except Exception as exc:
        report = {
            "schema": AUDIT_SCHEMA,
            "accepted": False,
            "bundle": str(bundle),
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
        status = 1
    _write_json(output.resolve(), report)
    if report["accepted"]:
        aggregate = report["aggregate"]
        print(
            "accepted "
            f"seeds={aggregate['seed_count']} cells={aggregate['arm_seed_cells']} "
            f"issued={aggregate['issued_queries']} release={aggregate['release_events']} "
            f"timeout={aggregate['timeouts']} failure={aggregate['failure_events']} "
            f"pending={aggregate['pending_at_episode_end']} snapshots={aggregate['snapshot_count']}"
        )
    else:
        print(report["errors"][0], file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
