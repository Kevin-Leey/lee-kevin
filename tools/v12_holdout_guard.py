"""One-shot authorization guard for the v12 confirmatory holdout.

The confirmatory simulator seeds are not an ordinary CLI cohort.  Access is
released only after the locked go/no-go analysis passes, and every subsequent
stage is a single-use continuation of that release.  This module owns the
authorization schema and the Windows-compatible, fail-closed state machine
shared by the snapshot producer, target generator, and branch runner.

The files in this module are audit capabilities, not protection from an actor
who can arbitrarily rewrite the repository and every audit artifact.  Within
the experiment workflow they prevent accidental opening, concurrent use,
partial cohorts, stale inputs, and replay of a previously used capability.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from dilu.driver_agent.policy_state import DRIVER_POLICY_STATE_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_TOOL_PATH = Path(__file__).with_name(
    "calibrate_identifiable_gate_v12.py"
)
CALIBRATION_LOCK_PATH = REPO_ROOT / "formal_protocol.yaml"
GATE_SUPPORT_PATH = (
    REPO_ROOT / "dilu" / "driver_agent" / "reasoning" / "rgd_support.py"
)
PRODUCER_PATH = Path(__file__).with_name("run_mechanism_inprocess.py")
BRANCH_ENGINE_PATH = Path(__file__).with_name("analyze_release_state_rollouts.py")
BRANCH_RUNNER_PATH = Path(__file__).with_name("run_v12_branch_labels.py")
BASE_CONFIG_PATH = REPO_ROOT / "config.yaml"

METHOD_VERSION = "identifiable_gate_v12"
AUTHORIZATION_SCHEMA = "identifiable_gate_v12_holdout_authorization_v1"
AUTHORIZATION_ROLE = "one_shot_confirmatory_holdout_capability"
STATE_SCHEMA = "identifiable_gate_v12_holdout_state_v1"
STATE_ROLE = "one_shot_confirmatory_holdout_state"
CLAIM_SCHEMA = "identifiable_gate_v12_holdout_phase_claim_v1"
ISSUANCE_SCHEMA = "identifiable_gate_v12_holdout_issuance_v1"
SINGLETON_SCHEMA = "identifiable_gate_v12_holdout_singleton_v1"
PRODUCER_MANIFEST_SCHEMA = "identifiable_gate_v12_holdout_producer_v1"
TARGET_MANIFEST_SCHEMA = "identifiable_gate_v12_locked_snapshot_targets_v2"
CONFIRMATORY_SEEDS = tuple(range(3000, 3030))
GO_NO_GO_SEEDS = tuple(range(2040, 2060))

_AUTHORIZATION_FILE = "v12_holdout_authorization.json"
_STATE_FILE = "v12_holdout_authorization_state.json"
HOLDOUT_LEDGER_ROOT = (
    REPO_ROOT
    / "results"
    / "tvt_final_20260720"
    / "identifiable_gate_v12"
    / "holdout_authorization_ledger"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{16,128}$")
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE_INVARIANTS = {
    "authorized_unopened": (0, False, False),
    "trace_open": (1, True, False),
    "traces_generated": (1, True, False),
    "target_open": (1, True, False),
    "target_generated": (1, True, False),
    "snapshot_open": (1, True, False),
    "snapshots_generated": (1, True, False),
    "branch_open": (1, True, False),
    "consumed": (1, True, True),
    "trace_failed": (1, True, False),
    "target_failed": (1, True, False),
    "snapshot_failed": (1, True, False),
    "branch_failed": (1, True, False),
}
_ALLOWED_TRANSITIONS = {
    ("authorized_unopened", "trace_open"),
    ("trace_open", "traces_generated"),
    ("trace_open", "trace_failed"),
    ("traces_generated", "target_open"),
    ("target_open", "target_generated"),
    ("target_open", "target_failed"),
    ("target_generated", "snapshot_open"),
    ("snapshot_open", "snapshots_generated"),
    ("snapshot_open", "snapshot_failed"),
    ("snapshots_generated", "branch_open"),
    ("branch_open", "consumed"),
    ("branch_open", "branch_failed"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value {value!r}")
        ),
    )


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    canonical = dict(payload)
    observed = canonical.pop(field, None)
    expected = canonical_sha256(canonical)
    _require(observed == expected, f"{field} mismatch")
    return expected


def _with_payload_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    _require(field not in result, f"payload already contains {field}")
    result[field] = canonical_sha256(result)
    return result


def _atomic_replace_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = _json_bytes(payload)
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_write_json(path: Path, payload: Any) -> None:
    """Create *path* once, using primitives supported on Windows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def authorization_state_path(authorization_path: Path) -> Path:
    return Path(authorization_path).resolve().with_name(_STATE_FILE)


def _claim_path(authorization_path: Path, authorization_id: str, phase: str) -> Path:
    _require(bool(_SAFE_ID.fullmatch(authorization_id)), "unsafe authorization_id")
    _require(phase in {"trace", "target", "snapshot", "branch"}, "unknown holdout phase")
    return HOLDOUT_LEDGER_ROOT.resolve() / "claims" / (
        f"v12_holdout_claim_{authorization_id}_{phase}.json"
    )


def _outcome_path(authorization_id: str, phase: str) -> Path:
    _require(bool(_SAFE_ID.fullmatch(authorization_id)), "unsafe authorization_id")
    _require(phase in {"trace", "target", "snapshot", "branch"}, "unknown holdout phase")
    return HOLDOUT_LEDGER_ROOT.resolve() / "outcomes" / (
        f"v12_holdout_outcome_{authorization_id}_{phase}.json"
    )


def _issuance_path(authorization_id: str) -> Path:
    _require(bool(_SAFE_ID.fullmatch(authorization_id)), "unsafe authorization_id")
    return HOLDOUT_LEDGER_ROOT.resolve() / "issued" / (
        f"v12_holdout_issuance_{authorization_id}.json"
    )


def _singleton_fingerprint(source: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "method_version": METHOD_VERSION,
            "calibration_lock_id": source["calibration_lock_id"],
            "seed_block": seed_block_payload(),
        }
    )


def _singleton_path(source: Mapping[str, Any]) -> Path:
    return HOLDOUT_LEDGER_ROOT.resolve() / "singleton" / (
        f"v12_confirmatory_holdout_{_singleton_fingerprint(source)}.json"
    )


def seed_block_payload(seeds: Sequence[int] = CONFIRMATORY_SEEDS) -> dict[str, Any]:
    normalized = tuple(int(seed) for seed in seeds)
    _require(normalized, "empty holdout seed block")
    _require(
        normalized == tuple(range(normalized[0], normalized[-1] + 1)),
        "holdout seeds must be contiguous and ordered",
    )
    return {
        "start": normalized[0],
        "end": normalized[-1],
        "count": len(normalized),
        "seeds": list(normalized),
    }


def _validate_seed_block(payload: Any, expected: Sequence[int]) -> dict[str, Any]:
    block = dict(payload or {})
    expected_payload = seed_block_payload(expected)
    _require(block == expected_payload, "confirmatory holdout seed block drift")
    return block


def classify_seed_request(seed_start: int, count: int) -> str:
    _require(isinstance(seed_start, int) and not isinstance(seed_start, bool), "seed-start must be an integer")
    _require(isinstance(count, int) and not isinstance(count, bool), "seeds must be an integer")
    _require(count > 0, "--seeds must be positive")
    requested = tuple(range(seed_start, seed_start + count))
    overlap = set(requested).intersection(CONFIRMATORY_SEEDS)
    if not overlap:
        return "ordinary"
    _require(
        requested == CONFIRMATORY_SEEDS,
        "any request touching the v12 confirmatory holdout must be exactly seeds 3000-3029",
    )
    return "confirmatory_holdout"


@dataclass(frozen=True)
class VerifiedAuthorization:
    path: Path
    state_path: Path
    payload: Mapping[str, Any]
    state: Mapping[str, Any]
    raw_sha256: str
    source_paths: Mapping[str, Path]

    @property
    def authorization_id(self) -> str:
        return str(self.payload["authorization_id"])


@dataclass(frozen=True)
class HoldoutPhaseClaim:
    authorization: VerifiedAuthorization
    phase: str
    run_id: str
    claim_path: Path
    open_stage: str
    success_stage: str
    failure_stage: str
    run_binding: Mapping[str, Any]


@dataclass(frozen=True)
class ProducerPermit:
    authorization_id: str
    phase: str
    run_id: str
    seeds: tuple[int, ...]
    runtime_marker: Mapping[str, Any]


def _source_paths(
    *,
    protocol_path: Path,
    lock_path: Path,
    calibration_manifest_path: Path,
    go_no_go_manifest_path: Path,
) -> dict[str, Path]:
    paths = {
        "protocol": Path(protocol_path).resolve(),
        "lock": Path(lock_path).resolve(),
        "calibration_manifest": Path(calibration_manifest_path).resolve(),
        "go_no_go_manifest": Path(go_no_go_manifest_path).resolve(),
    }
    for label, path in paths.items():
        _require(path.is_file(), f"missing holdout {label}: {path}")
    return paths


def _verified_source_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    # Import lazily so the calibration CLI can call issue_holdout_authorization
    # after it has emitted a passing go/no-go manifest without an import cycle.
    from tools.calibrate_identifiable_gate_v12 import (
        GATE_SUPPORT_PATH as SELECTOR_GATE_SUPPORT_PATH,
        load_spec,
        verify_calibration_manifest,
        verify_go_no_go_manifest,
    )

    _require(
        Path(SELECTOR_GATE_SUPPORT_PATH).resolve() == GATE_SUPPORT_PATH.resolve(),
        "selector and holdout guard disagree on gate-support source",
    )
    for path in (
        CALIBRATION_TOOL_PATH,
        GATE_SUPPORT_PATH,
        PRODUCER_PATH,
        BRANCH_ENGINE_PATH,
        BRANCH_RUNNER_PATH,
        BASE_CONFIG_PATH,
        Path(__file__),
    ):
        _require(path.is_file(), f"missing holdout source file: {path}")

    # Strict-load first to reject duplicate-key manifests before the selector's
    # semantic verification reads them.
    calibration_payload = load_json_strict(paths["calibration_manifest"])
    go_payload = load_json_strict(paths["go_no_go_manifest"])
    _require(isinstance(calibration_payload, Mapping), "calibration manifest is not an object")
    _require(isinstance(go_payload, Mapping), "go/no-go manifest is not an object")

    spec = load_spec(paths["lock"])
    verified_calibration, _ = verify_calibration_manifest(
        paths["calibration_manifest"],
        spec=spec,
        protocol_path=paths["protocol"],
        lock_path=paths["lock"],
    )
    calibration_hash = sha256_file(paths["calibration_manifest"])
    protocol_hash = sha256_file(paths["protocol"])
    verified_go = verify_go_no_go_manifest(
        paths["go_no_go_manifest"],
        calibration_manifest_sha256=calibration_hash,
        protocol_sha256=protocol_hash,
    )

    _require(verified_go.get("artifact_role") == "go_no_go_locked_analysis", "go/no-go artifact role drift")
    _require(verified_go.get("method_version") == METHOD_VERSION, "go/no-go method version drift")
    _require(verified_go.get("seed_block") == list(GO_NO_GO_SEEDS), "go/no-go seed block drift")
    _require(verified_go.get("parameter_search_performed") is False, "go/no-go performed parameter search")
    acceptance = dict(verified_go.get("acceptance", {}) or {})
    _require(acceptance.get("validation_evaluated") is False, "go/no-go claims validation access")
    _require(acceptance.get("confirmatory_holdout_evaluated") is False, "go/no-go already evaluated the holdout")
    _require(acceptance.get("passed") is True, "go/no-go did not pass")
    _require(acceptance.get("paper_facing_passed") is False, "go/no-go is incorrectly paper-facing")

    lock_payload = load_json_strict(paths["lock"])
    _validate_seed_block(lock_payload.get("validation_seed_block"), CONFIRMATORY_SEEDS)
    from dilu.evaluation.reporter import build_runtime_source_hash

    return {
        "calibration_lock_id": spec.lock_id,
        "calibration_manifest_sha256": calibration_hash,
        "calibration_selection_digest": verified_calibration["selection_digest"],
        "go_no_go_manifest_sha256": sha256_file(paths["go_no_go_manifest"]),
        "go_no_go_analysis_digest": verified_go["analysis_digest"],
        "protocol_sha256": protocol_hash,
        "lock_sha256": sha256_file(paths["lock"]),
        "selector_sha256": sha256_file(CALIBRATION_TOOL_PATH),
        "gate_support_sha256": sha256_file(GATE_SUPPORT_PATH),
        "producer_sha256": sha256_file(PRODUCER_PATH),
        "branch_engine_sha256": sha256_file(BRANCH_ENGINE_PATH),
        "branch_runner_sha256": sha256_file(BRANCH_RUNNER_PATH),
        "base_config_sha256": sha256_file(BASE_CONFIG_PATH),
        "runtime_source_sha256": build_runtime_source_hash(REPO_ROOT),
        "holdout_guard_sha256": sha256_file(Path(__file__)),
    }


def _initial_state(
    authorization: Mapping[str, Any], authorization_sha256: str
) -> dict[str, Any]:
    payload = {
        "schema": STATE_SCHEMA,
        "artifact_role": STATE_ROLE,
        "authorization_id": authorization["authorization_id"],
        "authorization_sha256": authorization_sha256,
        "authorization_payload_sha256": authorization[
            "authorization_payload_sha256"
        ],
        "stage": "authorized_unopened",
        "open_count": 0,
        "capability_consumed": False,
        "workflow_consumed": False,
        "transition_seq": 0,
        "prior_state_sha256": None,
        "bindings": {
            "source": dict(authorization["source"]),
            "seed_block": dict(authorization["seed_block"]),
        },
        "transition_log": [],
    }
    return _with_payload_hash(payload, "state_payload_sha256")


def issue_holdout_authorization(
    *,
    authorization_path: Path,
    protocol_path: Path,
    lock_path: Path = CALIBRATION_LOCK_PATH,
    calibration_manifest_path: Path,
    go_no_go_manifest_path: Path,
) -> Mapping[str, Any]:
    """Issue the one capability allowed by a passing locked go/no-go run."""

    authorization_path = Path(authorization_path).resolve()
    _require(
        authorization_path.name == _AUTHORIZATION_FILE,
        f"authorization filename must be {_AUTHORIZATION_FILE}",
    )
    paths = _source_paths(
        protocol_path=protocol_path,
        lock_path=lock_path,
        calibration_manifest_path=calibration_manifest_path,
        go_no_go_manifest_path=go_no_go_manifest_path,
    )
    source = _verified_source_payload(paths)
    nonce = secrets.token_hex(32)
    authorization_id = f"v12-holdout-{nonce[:16]}"
    payload = _with_payload_hash(
        {
            "schema": AUTHORIZATION_SCHEMA,
            "artifact_role": AUTHORIZATION_ROLE,
            "method_version": METHOD_VERSION,
            "authorization_id": authorization_id,
            "nonce_hex": nonce,
            "max_open_count": 1,
            "seed_block": seed_block_payload(),
            "source": source,
            "issued_stage": "authorized_unopened",
        },
        "authorization_payload_sha256",
    )
    state_path = authorization_state_path(authorization_path)
    _require(not state_path.exists(), f"holdout state already exists: {state_path}")
    expected_authorization_sha256 = hashlib.sha256(_json_bytes(payload)).hexdigest()
    singleton = _with_payload_hash(
        {
            "schema": SINGLETON_SCHEMA,
            "artifact_role": "confirmatory_holdout_singleton_reservation",
            "scope_fingerprint": _singleton_fingerprint(source),
            "authorization_id": authorization_id,
            "authorization_path": str(authorization_path),
            "state_path": str(state_path),
            "authorization_sha256": expected_authorization_sha256,
            "authorization_payload_sha256": payload[
                "authorization_payload_sha256"
            ],
            "source": source,
            "reserved_at_utc": _utc_now(),
        },
        "singleton_payload_sha256",
    )
    try:
        _exclusive_write_json(_singleton_path(source), singleton)
    except FileExistsError as exc:
        raise ValueError(
            "the v12 confirmatory holdout has already been authorized; a second issuance is forbidden"
        ) from exc
    _exclusive_write_json(authorization_path, payload)
    try:
        _exclusive_write_json(
            state_path, _initial_state(payload, sha256_file(authorization_path))
        )
    except Exception:
        # The authorization remains present deliberately: partial issuance is
        # fail-closed and must be investigated rather than silently retried.
        raise
    issuance = _with_payload_hash(
        {
            "schema": ISSUANCE_SCHEMA,
            "artifact_role": "confirmatory_holdout_issuance_ledger",
            "authorization_id": authorization_id,
            "authorization_path": str(authorization_path),
            "state_path": str(state_path),
            "authorization_sha256": sha256_file(authorization_path),
            "authorization_payload_sha256": payload[
                "authorization_payload_sha256"
            ],
            "source": source,
            "issued_at_utc": _utc_now(),
        },
        "issuance_payload_sha256",
    )
    _exclusive_write_json(_issuance_path(authorization_id), issuance)
    return payload


def _verify_issuance_ledger(
    *,
    authorization_path: Path,
    state_path: Path,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
) -> None:
    singleton_path = _singleton_path(authorization["source"])
    _require(singleton_path.is_file(), "holdout authorization has no singleton reservation")
    singleton = load_json_strict(singleton_path)
    _require(isinstance(singleton, Mapping), "holdout singleton reservation is not an object")
    _require(singleton.get("schema") == SINGLETON_SCHEMA, "holdout singleton schema drift")
    _require(singleton.get("artifact_role") == "confirmatory_holdout_singleton_reservation", "holdout singleton role drift")
    _require(singleton.get("scope_fingerprint") == _singleton_fingerprint(authorization["source"]), "holdout singleton scope drift")
    _require(singleton.get("authorization_id") == authorization["authorization_id"], "holdout singleton authorization id drift")
    _require(singleton.get("authorization_path") == str(authorization_path), "copied or relocated holdout authorization is forbidden")
    _require(singleton.get("state_path") == str(state_path), "holdout singleton state path drift")
    _require(singleton.get("authorization_sha256") == authorization_sha256, "holdout singleton authorization hash drift")
    _require(singleton.get("authorization_payload_sha256") == authorization["authorization_payload_sha256"], "holdout singleton payload hash drift")
    _require(singleton.get("source") == authorization["source"], "holdout singleton source drift")
    _payload_hash(singleton, "singleton_payload_sha256")
    path = _issuance_path(str(authorization["authorization_id"]))
    _require(path.is_file(), "holdout authorization has no immutable issuance ledger entry")
    issuance = load_json_strict(path)
    _require(isinstance(issuance, Mapping), "holdout issuance ledger is not an object")
    _require(issuance.get("schema") == ISSUANCE_SCHEMA, "holdout issuance ledger schema drift")
    _require(issuance.get("artifact_role") == "confirmatory_holdout_issuance_ledger", "holdout issuance ledger role drift")
    _require(issuance.get("authorization_id") == authorization["authorization_id"], "holdout issuance id drift")
    _require(issuance.get("authorization_path") == str(authorization_path), "copied or relocated holdout authorization is forbidden")
    _require(issuance.get("state_path") == str(state_path), "holdout issuance state path drift")
    _require(issuance.get("authorization_sha256") == authorization_sha256, "holdout issuance authorization hash drift")
    _require(issuance.get("authorization_payload_sha256") == authorization["authorization_payload_sha256"], "holdout issuance payload hash drift")
    _require(issuance.get("source") == authorization["source"], "holdout issuance source binding drift")
    _payload_hash(issuance, "issuance_payload_sha256")


def _validate_authorization_payload(
    payload: Any,
    *,
    raw_sha256: str,
    source: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), "holdout authorization is not an object")
    _require(payload.get("schema") == AUTHORIZATION_SCHEMA, "holdout authorization schema drift")
    _require(payload.get("artifact_role") == AUTHORIZATION_ROLE, "holdout authorization role drift")
    _require(payload.get("method_version") == METHOD_VERSION, "holdout authorization method drift")
    _require(payload.get("issued_stage") == "authorized_unopened", "holdout authorization issue stage drift")
    _require(payload.get("max_open_count") == 1, "holdout authorization is not one-shot")
    _validate_seed_block(payload.get("seed_block"), CONFIRMATORY_SEEDS)
    nonce = payload.get("nonce_hex")
    _require(isinstance(nonce, str) and bool(_HEX_256.fullmatch(nonce)), "invalid holdout authorization nonce")
    expected_id = f"v12-holdout-{nonce[:16]}"
    _require(payload.get("authorization_id") == expected_id, "authorization_id/nonce mismatch")
    _require(dict(payload.get("source", {}) or {}) == dict(source), "holdout authorization source hashes are stale or forged")
    _payload_hash(payload, "authorization_payload_sha256")
    _require(bool(_HEX_256.fullmatch(raw_sha256)), "invalid authorization file hash")
    return payload


def _validate_state(
    payload: Any,
    *,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
) -> Mapping[str, Any]:
    _require(isinstance(payload, Mapping), "holdout state is not an object")
    _require(payload.get("schema") == STATE_SCHEMA, "holdout state schema drift")
    _require(payload.get("artifact_role") == STATE_ROLE, "holdout state role drift")
    _require(payload.get("authorization_id") == authorization["authorization_id"], "holdout state authorization id drift")
    _require(payload.get("authorization_sha256") == authorization_sha256, "holdout state references an old authorization")
    _require(
        payload.get("authorization_payload_sha256")
        == authorization["authorization_payload_sha256"],
        "holdout state authorization payload drift",
    )
    _payload_hash(payload, "state_payload_sha256")
    stage = payload.get("stage")
    _require(stage in _STAGE_INVARIANTS, f"unknown holdout stage {stage!r}")
    expected_open, expected_capability, expected_workflow = _STAGE_INVARIANTS[str(stage)]
    _require(payload.get("open_count") == expected_open, "holdout open_count/stage mismatch")
    _require(payload.get("capability_consumed") is expected_capability, "holdout capability_consumed/stage mismatch")
    _require(payload.get("workflow_consumed") is expected_workflow, "holdout workflow_consumed/stage mismatch")
    sequence = payload.get("transition_seq")
    _require(isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0, "invalid holdout transition_seq")
    log = payload.get("transition_log")
    _require(isinstance(log, list) and len(log) == sequence, "holdout transition log length drift")
    previous_stage = "authorized_unopened"
    for index, row in enumerate(log, start=1):
        _require(isinstance(row, Mapping), f"transition {index} is not an object")
        _require(row.get("seq") == index, f"transition {index} sequence drift")
        _require(row.get("to_stage") in _STAGE_INVARIANTS, f"transition {index} has unknown stage")
        _require(row.get("from_stage") == previous_stage, f"transition {index} chain drift")
        pair = (str(row.get("from_stage")), str(row.get("to_stage")))
        _require(pair in _ALLOWED_TRANSITIONS, f"transition {index} is not allowed: {pair}")
        _require(isinstance(row.get("previous_state_sha256"), str), f"transition {index} omits prior hash")
        _require(isinstance(row.get("claim_sha256"), str), f"transition {index} omits claim hash")
        run_id = row.get("run_id")
        _require(isinstance(run_id, str) and run_id, f"transition {index} omits run id")
        phase = run_id.split("-", 1)[0]
        ledger_path = (
            _claim_path(Path("."), str(authorization["authorization_id"]), phase)
            if str(row["to_stage"]).endswith("_open")
            else _outcome_path(str(authorization["authorization_id"]), phase)
        )
        _require(ledger_path.is_file(), f"transition {index} has no append-only ledger record")
        _require(sha256_file(ledger_path) == row["claim_sha256"], f"transition {index} ledger hash drift")
        ledger = load_json_strict(ledger_path)
        _require(ledger.get("authorization_id") == authorization["authorization_id"], f"transition {index} ledger authorization drift")
        _require(ledger.get("run_id") == run_id, f"transition {index} ledger run id drift")
        _require(ledger.get("from_stage") == row["from_stage"], f"transition {index} ledger source stage drift")
        _require(ledger.get("to_stage") == row["to_stage"], f"transition {index} ledger target stage drift")
        _require(ledger.get("prior_state_sha256") == row["previous_state_sha256"], f"transition {index} ledger prior hash drift")
        previous_stage = str(row["to_stage"])
    if log:
        _require(payload.get("prior_state_sha256") == log[-1]["previous_state_sha256"], "holdout prior_state_sha256 drift")
        _require(log[-1].get("to_stage") == stage, "holdout state/log stage drift")
    else:
        _require(payload.get("prior_state_sha256") is None, "initial state has prior hash")
    bindings = dict(payload.get("bindings", {}) or {})
    _require(bindings.get("source") == authorization["source"], "holdout state source binding drift")
    _require(bindings.get("seed_block") == authorization["seed_block"], "holdout state seed binding drift")
    return payload


def verify_holdout_authorization(
    *,
    authorization_path: Path,
    protocol_path: Path,
    lock_path: Path = CALIBRATION_LOCK_PATH,
    calibration_manifest_path: Path,
    go_no_go_manifest_path: Path,
) -> VerifiedAuthorization:
    authorization_path = Path(authorization_path).resolve()
    _require(authorization_path.is_file(), f"missing holdout authorization: {authorization_path}")
    paths = _source_paths(
        protocol_path=protocol_path,
        lock_path=lock_path,
        calibration_manifest_path=calibration_manifest_path,
        go_no_go_manifest_path=go_no_go_manifest_path,
    )
    source = _verified_source_payload(paths)
    raw_hash = sha256_file(authorization_path)
    authorization = _validate_authorization_payload(
        load_json_strict(authorization_path), raw_sha256=raw_hash, source=source
    )
    state_path = authorization_state_path(authorization_path)
    _require(state_path.is_file(), f"missing holdout authorization state: {state_path}")
    _verify_issuance_ledger(
        authorization_path=authorization_path,
        state_path=state_path,
        authorization=authorization,
        authorization_sha256=raw_hash,
    )
    state = _validate_state(
        load_json_strict(state_path),
        authorization=authorization,
        authorization_sha256=raw_hash,
    )
    return VerifiedAuthorization(
        path=authorization_path,
        state_path=state_path,
        payload=authorization,
        state=state,
        raw_sha256=raw_hash,
        source_paths=paths,
    )


def _advance_state(
    verified: VerifiedAuthorization,
    *,
    state: Mapping[str, Any],
    to_stage: str,
    claim_sha256: str,
    run_id: str,
    binding_updates: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _require(to_stage in _STAGE_INVARIANTS, f"unknown target stage {to_stage!r}")
    previous_hash = str(state["state_payload_sha256"])
    sequence = int(state["transition_seq"]) + 1
    bindings = dict(state["bindings"])
    for key, value in dict(binding_updates or {}).items():
        _require(key not in {"source", "seed_block"}, f"cannot replace permanent binding {key}")
        _require(key not in bindings, f"holdout binding {key!r} already exists")
        bindings[key] = value
    open_count, capability_consumed, workflow_consumed = _STAGE_INVARIANTS[to_stage]
    transition = {
        "seq": sequence,
        "from_stage": state["stage"],
        "to_stage": to_stage,
        "run_id": run_id,
        "previous_state_sha256": previous_hash,
        "claim_sha256": claim_sha256,
        "transitioned_at_utc": _utc_now(),
    }
    payload = {
        "schema": STATE_SCHEMA,
        "artifact_role": STATE_ROLE,
        "authorization_id": verified.authorization_id,
        "authorization_sha256": verified.raw_sha256,
        "authorization_payload_sha256": verified.payload[
            "authorization_payload_sha256"
        ],
        "stage": to_stage,
        "open_count": open_count,
        "capability_consumed": capability_consumed,
        "workflow_consumed": workflow_consumed,
        "transition_seq": sequence,
        "prior_state_sha256": previous_hash,
        "bindings": bindings,
        "transition_log": [*list(state["transition_log"]), transition],
    }
    payload = _with_payload_hash(payload, "state_payload_sha256")
    _atomic_replace_json(verified.state_path, payload)
    return payload


def _claim_payload(
    verified: VerifiedAuthorization,
    *,
    phase: str,
    run_id: str,
    from_stage: str,
    to_stage: str,
    run_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    return _with_payload_hash(
        {
            "schema": CLAIM_SCHEMA,
            "artifact_role": "one_shot_holdout_phase_claim",
            "authorization_id": verified.authorization_id,
            "authorization_sha256": verified.raw_sha256,
            "phase": phase,
            "run_id": run_id,
            "status": "claimed",
            "from_stage": from_stage,
            "to_stage": to_stage,
            "prior_state_sha256": verified.state["state_payload_sha256"],
            "run_binding": dict(run_binding),
            "claimed_at_utc": _utc_now(),
        },
        "claim_payload_sha256",
    )


def _begin_phase(
    verified: VerifiedAuthorization,
    *,
    phase: str,
    expected_stage: str,
    open_stage: str,
    success_stage: str,
    failure_stage: str,
    run_binding: Mapping[str, Any],
) -> HoldoutPhaseClaim:
    _require(verified.state.get("stage") == expected_stage, f"holdout {phase} requires stage {expected_stage}, found {verified.state.get('stage')}")
    run_id = f"{phase}-{uuid.uuid4().hex}"
    claim_path = _claim_path(verified.path, verified.authorization_id, phase)
    claim = _claim_payload(
        verified,
        phase=phase,
        run_id=run_id,
        from_stage=expected_stage,
        to_stage=open_stage,
        run_binding=run_binding,
    )
    try:
        _exclusive_write_json(claim_path, claim)
    except FileExistsError as exc:
        raise ValueError(f"holdout {phase} capability was already claimed; replay is forbidden") from exc
    _advance_state(
        verified,
        state=verified.state,
        to_stage=open_stage,
        claim_sha256=sha256_file(claim_path),
        run_id=run_id,
    )
    return HoldoutPhaseClaim(
        authorization=verified,
        phase=phase,
        run_id=run_id,
        claim_path=claim_path,
        open_stage=open_stage,
        success_stage=success_stage,
        failure_stage=failure_stage,
        run_binding=dict(run_binding),
    )


def _reload_claim_state(claim: HoldoutPhaseClaim) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    claim_payload = load_json_strict(claim.claim_path)
    _require(claim_payload.get("schema") == CLAIM_SCHEMA, "holdout claim schema drift")
    _payload_hash(claim_payload, "claim_payload_sha256")
    _require(claim_payload.get("authorization_id") == claim.authorization.authorization_id, "holdout claim authorization drift")
    _require(claim_payload.get("phase") == claim.phase, "holdout claim phase drift")
    _require(claim_payload.get("run_id") == claim.run_id, "holdout claim run id drift")
    _require(claim_payload.get("status") == "claimed", "holdout phase claim is no longer open")
    state = _validate_state(
        load_json_strict(claim.authorization.state_path),
        authorization=claim.authorization.payload,
        authorization_sha256=claim.authorization.raw_sha256,
    )
    _require(state.get("stage") == claim.open_stage, f"holdout {claim.phase} is not open")
    _require(state["transition_log"][-1].get("run_id") == claim.run_id, "holdout state run id drift")
    return claim_payload, state


def _reverify_claim_sources(claim: HoldoutPhaseClaim) -> VerifiedAuthorization:
    paths = claim.authorization.source_paths
    current = verify_holdout_authorization(
        authorization_path=claim.authorization.path,
        protocol_path=paths["protocol"],
        lock_path=paths["lock"],
        calibration_manifest_path=paths["calibration_manifest"],
        go_no_go_manifest_path=paths["go_no_go_manifest"],
    )
    _require(
        current.authorization_id == claim.authorization.authorization_id,
        "holdout authorization changed during the phase",
    )
    _require(
        current.raw_sha256 == claim.authorization.raw_sha256,
        "holdout authorization bytes changed during the phase",
    )
    return current


def _finish_phase(
    claim: HoldoutPhaseClaim,
    *,
    status: str,
    details: Mapping[str, Any],
    binding_updates: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    _reverify_claim_sources(claim)
    claim_payload, state = _reload_claim_state(claim)
    destination = claim.success_stage if status == "completed" else claim.failure_stage
    outcome = _with_payload_hash(
        {
            "schema": CLAIM_SCHEMA,
            "artifact_role": "one_shot_holdout_phase_outcome",
            "authorization_id": claim.authorization.authorization_id,
            "authorization_sha256": claim.authorization.raw_sha256,
            "phase": claim.phase,
            "run_id": claim.run_id,
            "status": status,
            "from_stage": claim.open_stage,
            "to_stage": destination,
            "prior_state_sha256": state["state_payload_sha256"],
            "initial_claim_sha256": sha256_file(claim.claim_path),
            "details": dict(details),
            "finished_at_utc": _utc_now(),
        },
        "claim_payload_sha256",
    )
    outcome_path = _outcome_path(claim.authorization.authorization_id, claim.phase)
    try:
        _exclusive_write_json(outcome_path, outcome)
    except FileExistsError as exc:
        raise ValueError(f"holdout {claim.phase} already has an outcome; replay is forbidden") from exc
    return _advance_state(
        claim.authorization,
        state=state,
        to_stage=destination,
        claim_sha256=sha256_file(outcome_path),
        run_id=claim.run_id,
        binding_updates=binding_updates,
    )


def fail_phase(claim: HoldoutPhaseClaim, error: BaseException | str) -> Mapping[str, Any]:
    """Burn a claimed phase after an error; failed phases are never resumable."""

    return _finish_phase(
        claim,
        status="failed",
        details={
            "error_type": type(error).__name__ if isinstance(error, BaseException) else "RuntimeError",
            "error_message": str(error),
        },
    )


def _target_artifact_paths(target_map_path: Path) -> dict[str, Path]:
    target_map_path = Path(target_map_path).resolve()
    return {
        "target_map": target_map_path,
        "events": target_map_path.with_suffix(".events.csv"),
        "manifest": target_map_path.with_suffix(".manifest.json"),
    }


def _normalized_target_events(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_fields = [
        "seed",
        "delay_s",
        "delay_steps",
        "query_frame",
        "release_frame",
        "candidate_state_id",
        "release_state_id",
    ]
    _require(rows, "confirmatory target event table is empty")
    _require(list(rows[0]) == expected_fields, "confirmatory target event columns drift")
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        seed = int(row["seed"])
        delay_steps = int(row["delay_steps"])
        query = int(row["query_frame"])
        release = int(row["release_frame"])
        _require(seed in CONFIRMATORY_SEEDS, f"target event {index} has out-of-block seed")
        _require(query >= 0 and release >= 0 and release - query == delay_steps, f"target event {index} frame/delay drift")
        _require(row["candidate_state_id"] == f"{seed}:{query}:{delay_steps}", f"target event {index} candidate id drift")
        _require(row["release_state_id"] == f"{seed}:{release}", f"target event {index} release id drift")
        normalized.append(
            {
                "seed": seed,
                "delay_s": float(row["delay_s"]),
                "delay_steps": delay_steps,
                "query_frame": query,
                "release_frame": release,
                "candidate_state_id": row["candidate_state_id"],
                "release_state_id": row["release_state_id"],
            }
        )
    _require(
        normalized
        == sorted(normalized, key=lambda row: (row["seed"], row["delay_steps"], row["query_frame"])),
        "confirmatory target events are not canonically ordered",
    )
    _require(
        len({(row["seed"], row["delay_steps"], row["query_frame"]) for row in normalized})
        == len(normalized),
        "duplicate confirmatory target event",
    )
    return normalized


def validate_target_artifacts(
    verified: VerifiedAuthorization,
    target_map_path: Path,
    *,
    trace_producer_binding: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    paths = _target_artifact_paths(target_map_path)
    for label, path in paths.items():
        _require(path.is_file(), f"missing confirmatory target {label}: {path}")
    target_map = load_json_strict(paths["target_map"])
    _require(isinstance(target_map, Mapping), "confirmatory target map is not an object")
    _require(list(target_map) == [str(seed) for seed in CONFIRMATORY_SEEDS], "target map must contain exactly ordered seeds 3000-3029")
    normalized_map: dict[str, list[int]] = {}
    for seed in CONFIRMATORY_SEEDS:
        frames = target_map[str(seed)]
        _require(isinstance(frames, list), f"target frames for seed {seed} are not a list")
        _require(all(isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0 for frame in frames), f"target frames for seed {seed} are invalid")
        _require(frames == sorted(set(frames)), f"target frames for seed {seed} are not unique and ordered")
        normalized_map[str(seed)] = list(frames)
    events = _normalized_target_events(paths["events"])
    releases = {str(seed): sorted({row["release_frame"] for row in events if row["seed"] == seed}) for seed in CONFIRMATORY_SEEDS}
    _require(normalized_map == releases, "target map and target event releases differ")

    manifest = load_json_strict(paths["manifest"])
    _require(isinstance(manifest, Mapping), "confirmatory target manifest is not an object")
    _require(manifest.get("schema") == TARGET_MANIFEST_SCHEMA, "confirmatory target manifest schema drift")
    _require(manifest.get("artifact_role") == "validation_snapshot_target_lock", "confirmatory target manifest role drift")
    _require(manifest.get("partition") == "validation", "confirmatory target partition drift")
    _require(manifest.get("method_version") == METHOD_VERSION, "confirmatory target method drift")
    _require(manifest.get("seed_block") == list(CONFIRMATORY_SEEDS), "confirmatory target seed block drift")
    _require(manifest.get("event_count") == len(events), "confirmatory target event count drift")
    _require(manifest.get("unique_release_targets") == sum(map(len, normalized_map.values())), "confirmatory target count drift")
    _require(manifest.get("target_map_semantic_hash") == canonical_sha256(normalized_map), "confirmatory target map semantic hash drift")
    _require(manifest.get("target_event_semantic_hash") == canonical_sha256(events), "confirmatory target event semantic hash drift")
    source = verified.payload["source"]
    for field in (
        "calibration_manifest_sha256",
        "go_no_go_manifest_sha256",
        "protocol_sha256",
        "selector_sha256",
        "gate_support_sha256",
        "calibration_selection_digest",
    ):
        _require(manifest.get(field) == source[field], f"confirmatory target {field} drift")
    if trace_producer_binding is not None:
        _require(
            manifest.get("trace_producer_manifest_sha256")
            == trace_producer_binding.get("manifest_sha256"),
            "target manifest is not bound to the authorized trace producer",
        )
        _require(
            manifest.get("trace_raw_file_set_hash")
            == trace_producer_binding.get("trace_raw_file_set_hash"),
            "target manifest trace file-set hash drift",
        )
    return {
        "target_map": {
            "path": str(paths["target_map"]),
            "sha256": sha256_file(paths["target_map"]),
            "semantic_sha256": canonical_sha256(normalized_map),
        },
        "events": {
            "path": str(paths["events"]),
            "sha256": sha256_file(paths["events"]),
            "semantic_sha256": canonical_sha256(events),
        },
        "manifest": {
            "path": str(paths["manifest"]),
            "sha256": sha256_file(paths["manifest"]),
            "semantic_sha256": canonical_sha256(manifest),
        },
        "trace_semantic_hash": manifest.get("trace_semantic_hash"),
        "trace_raw_file_set_hash": manifest.get("trace_raw_file_set_hash"),
    }


def begin_producer_phase(
    *,
    authorization_path: Path,
    protocol_path: Path,
    lock_path: Path,
    calibration_manifest_path: Path,
    go_no_go_manifest_path: Path,
    seeds: Sequence[int],
    result_root: Path,
    run_stamp: str,
    no_snapshots: bool,
    snapshot_targets: Path | None,
) -> tuple[HoldoutPhaseClaim, ProducerPermit]:
    _require(tuple(int(seed) for seed in seeds) == CONFIRMATORY_SEEDS, "producer holdout cohort must be exactly seeds 3000-3029")
    _require(bool(re.fullmatch(r"[A-Za-z0-9_.-]+", run_stamp)) and run_stamp not in {".", ".."}, "unsafe holdout run stamp")
    verified = verify_holdout_authorization(
        authorization_path=authorization_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
        calibration_manifest_path=calibration_manifest_path,
        go_no_go_manifest_path=go_no_go_manifest_path,
    )
    phase_root = (Path(result_root).resolve() / "formal_run" / run_stamp).resolve()
    _require(not phase_root.exists(), f"holdout producer output already exists: {phase_root}")

    if no_snapshots:
        _require(snapshot_targets is None, "trace phase cannot accept snapshot targets")
        phase = "trace"
        expected_stage = "authorized_unopened"
        open_stage = "trace_open"
        success_stage = "traces_generated"
        failure_stage = "trace_failed"
        target_binding = None
    else:
        _require(snapshot_targets is not None, "confirmatory snapshot phase requires locked snapshot targets; full-frame capture is forbidden")
        phase = "snapshot"
        expected_stage = "target_generated"
        open_stage = "snapshot_open"
        success_stage = "snapshots_generated"
        failure_stage = "snapshot_failed"
        state_targets = dict(verified.state.get("bindings", {}).get("targets", {}) or {})
        _require(state_targets, "holdout state has no locked target binding")
        target_binding = validate_target_artifacts(
            verified,
            Path(snapshot_targets),
            trace_producer_binding=verified.state["bindings"].get("trace_producer"),
        )
        _require(target_binding == state_targets, "snapshot targets differ from the single locked target set")

    run_binding = {
        "phase": phase,
        "phase_root": str(phase_root),
        "result_root": str(Path(result_root).resolve()),
        "run_stamp": run_stamp,
        "seed_block": seed_block_payload(seeds),
        "capture_snapshots": not no_snapshots,
        "target_artifacts": target_binding,
        "producer": {"path": str(PRODUCER_PATH.resolve()), "sha256": sha256_file(PRODUCER_PATH)},
        "protocol_sha256": sha256_file(Path(protocol_path).resolve()),
    }
    claim = _begin_phase(
        verified,
        phase=phase,
        expected_stage=expected_stage,
        open_stage=open_stage,
        success_stage=success_stage,
        failure_stage=failure_stage,
        run_binding=run_binding,
    )
    current_state = load_json_strict(verified.state_path)
    marker = _with_payload_hash(
        {
            "schema": "identifiable_gate_v12_runtime_holdout_permit_v1",
            "authorization_path": str(verified.path),
            "authorization_id": verified.authorization_id,
            "authorization_sha256": verified.raw_sha256,
            "state_path": str(verified.state_path),
            "open_state_sha256": current_state["state_payload_sha256"],
            "claim_path": str(claim.claim_path),
            "claim_sha256": sha256_file(claim.claim_path),
            "run_id": claim.run_id,
            "phase": phase,
            "seed_block": seed_block_payload(),
            "source_paths": {
                key: str(path) for key, path in verified.source_paths.items()
            },
        },
        "marker_payload_sha256",
    )
    return claim, ProducerPermit(
        authorization_id=verified.authorization_id,
        phase=phase,
        run_id=claim.run_id,
        seeds=CONFIRMATORY_SEEDS,
        runtime_marker=marker,
    )


def _validate_producer_artifacts(
    claim: HoldoutPhaseClaim, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    phase_root = Path(str(claim.run_binding["phase_root"])).resolve()
    capture_snapshots = bool(claim.run_binding["capture_snapshots"])
    roles = ["experiment_snapshot", "runtime_manifest", "reasoning", "physical"]
    if capture_snapshots:
        roles.append("snapshot_bundle")
    raw_rows = manifest.get("artifacts")
    _require(isinstance(raw_rows, list), "producer manifest artifacts are not a list")
    expected_count = len(CONFIRMATORY_SEEDS) * len(roles)
    _require(len(raw_rows) == expected_count, "producer manifest artifact count drift")
    _require(manifest.get("artifact_count") == expected_count, "producer artifact_count drift")

    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        _require(isinstance(row, Mapping), f"producer artifact {index} is not an object")
        seed = row.get("seed")
        role = row.get("role")
        _require(isinstance(seed, int) and seed in CONFIRMATORY_SEEDS, f"producer artifact {index} seed drift")
        _require(role in roles, f"producer artifact {index} role drift")
        key = (seed, str(role))
        _require(key not in observed, f"duplicate producer artifact {key}")
        observed[key] = row

    trace_files: list[dict[str, Any]] = []
    source = claim.authorization.payload["source"]
    for seed in CONFIRMATORY_SEEDS:
        result_dir = phase_root / "always_fast" / "highway" / f"seed_{seed}"
        expected_paths = {
            "experiment_snapshot": result_dir / "experiment_snapshot.json",
            "runtime_manifest": result_dir / "runtime_manifest.json",
            "reasoning": result_dir / f"ep_{seed}" / f"highway_{seed}_reasoning_records.json",
            "physical": result_dir / f"ep_{seed}" / f"highway_{seed}_physical_frames.json",
        }
        if capture_snapshots:
            expected_paths["snapshot_bundle"] = result_dir / "snapshots.pkl"
        for role, expected_path in expected_paths.items():
            row = observed[(seed, role)]
            resolved = expected_path.resolve()
            _require(Path(str(row.get("path", ""))).resolve() == resolved, f"seed {seed}: producer {role} path drift")
            _require(resolved.is_file(), f"seed {seed}: missing producer {role}")
            current_hash = sha256_file(resolved)
            _require(row.get("sha256") == current_hash, f"seed {seed}: producer {role} hash drift")
            if role == "reasoning":
                trace_files.append(
                    {"seed": seed, "sha256": current_hash, "name": resolved.name}
                )

        expected_artifact_hashes = {
            "reasoning": {
                "path": str(expected_paths["reasoning"].resolve()),
                "sha256": sha256_file(expected_paths["reasoning"]),
            },
            "physical": {
                "path": str(expected_paths["physical"].resolve()),
                "sha256": sha256_file(expected_paths["physical"]),
            },
        }
        if capture_snapshots:
            expected_artifact_hashes["snapshot_bundle"] = {
                "path": str(expected_paths["snapshot_bundle"].resolve()),
                "sha256": sha256_file(expected_paths["snapshot_bundle"]),
            }
        expected_provenance = {
            "schema_version": 2,
            "policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
            "policy_state_integrity": "canonical_json_sha256",
            "producer_path": str(PRODUCER_PATH.resolve()),
            "producer_sha256": source["producer_sha256"],
            "base_config_path": str(BASE_CONFIG_PATH.resolve()),
            "base_config_sha256": source["base_config_sha256"],
            "branch_engine_path": str(BRANCH_ENGINE_PATH.resolve()),
            "branch_engine_sha256": source["branch_engine_sha256"],
            "runtime_source_sha256": source["runtime_source_sha256"],
            "protocol_path": str(claim.authorization.source_paths["protocol"]),
            "protocol_sha256": source["protocol_sha256"],
            "artifact_hashes": expected_artifact_hashes,
        }
        for role in ("experiment_snapshot", "runtime_manifest"):
            payload = load_json_strict(expected_paths[role])
            _require(isinstance(payload, Mapping), f"seed {seed}: {role} is not a JSON object")
            _require(payload.get("snapshot_acquisition") == expected_provenance, f"seed {seed}: {role} acquisition provenance drift")
    _require(
        manifest.get("trace_raw_file_set_hash") == canonical_sha256(trace_files),
        "producer trace raw file-set hash drift",
    )
    return trace_files


def complete_producer_phase(
    claim: HoldoutPhaseClaim, producer_manifest_path: Path
) -> Mapping[str, Any]:
    manifest_path = Path(producer_manifest_path).resolve()
    manifest = load_json_strict(manifest_path)
    _require(isinstance(manifest, Mapping), "producer manifest is not an object")
    _require(manifest.get("schema") == PRODUCER_MANIFEST_SCHEMA, "producer manifest schema drift")
    _require(
        manifest.get("artifact_role")
        == f"confirmatory_holdout_{claim.phase}_acquisition",
        "producer manifest role drift",
    )
    _require(manifest.get("method_version") == METHOD_VERSION, "producer manifest method drift")
    _require(manifest.get("authorization_id") == claim.authorization.authorization_id, "producer manifest authorization drift")
    _require(manifest.get("authorization_sha256") == claim.authorization.raw_sha256, "producer manifest authorization hash drift")
    _require(manifest.get("run_id") == claim.run_id, "producer manifest run id drift")
    _require(manifest.get("phase") == claim.phase, "producer manifest phase drift")
    _require(manifest.get("status") == "completed", "producer manifest is not complete")
    _require(manifest.get("seed_block") == seed_block_payload(), "producer manifest seed block drift")
    _require(manifest.get("run_binding") == claim.run_binding, "producer manifest run binding drift")
    _payload_hash(manifest, "manifest_payload_sha256")
    _validate_producer_artifacts(claim, manifest)
    _, current_state = _reload_claim_state(claim)
    if claim.phase == "snapshot":
        trace_binding = dict(current_state["bindings"].get("trace_producer", {}) or {})
        _require(trace_binding, "snapshot phase has no preceding trace producer")
        _require(
            manifest.get("trace_raw_file_set_hash")
            == trace_binding.get("trace_raw_file_set_hash"),
            "snapshot acquisition trace bytes differ from the authorized target trace",
        )
    binding = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_semantic_sha256": canonical_sha256(manifest),
        "trace_raw_file_set_hash": manifest.get("trace_raw_file_set_hash"),
        "artifact_count": manifest.get("artifact_count"),
        "run_id": claim.run_id,
        "phase_root": claim.run_binding["phase_root"],
    }
    _require(isinstance(binding["trace_raw_file_set_hash"], str) and bool(_HEX_256.fullmatch(binding["trace_raw_file_set_hash"])), "producer manifest omits trace file-set hash")
    return _finish_phase(
        claim,
        status="completed",
        details={"producer_manifest_sha256": binding["manifest_sha256"]},
        binding_updates={f"{claim.phase}_producer": binding},
    )


def begin_target_generation(
    *,
    authorization_path: Path,
    protocol_path: Path,
    lock_path: Path,
    calibration_manifest_path: Path,
    go_no_go_manifest_path: Path,
    trace_root: Path,
    target_map_path: Path,
) -> HoldoutPhaseClaim:
    verified = verify_holdout_authorization(
        authorization_path=authorization_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
        calibration_manifest_path=calibration_manifest_path,
        go_no_go_manifest_path=go_no_go_manifest_path,
    )
    trace_binding = dict(verified.state.get("bindings", {}).get("trace_producer", {}) or {})
    _require(trace_binding, "target generation requires a completed trace producer")
    _require(Path(trace_root).resolve() == Path(trace_binding["phase_root"]).resolve(), "target generator trace root differs from authorized producer output")
    paths = _target_artifact_paths(target_map_path)
    _require(not any(path.exists() for path in paths.values()), "confirmatory target artifacts already exist")
    return _begin_phase(
        verified,
        phase="target",
        expected_stage="traces_generated",
        open_stage="target_open",
        success_stage="target_generated",
        failure_stage="target_failed",
        run_binding={
            "trace_root": str(Path(trace_root).resolve()),
            "trace_producer_manifest_sha256": trace_binding["manifest_sha256"],
            "trace_raw_file_set_hash": trace_binding["trace_raw_file_set_hash"],
            "target_map_path": str(Path(target_map_path).resolve()),
        },
    )


def complete_target_generation(
    claim: HoldoutPhaseClaim, target_map_path: Path
) -> Mapping[str, Any]:
    _require(claim.phase == "target", "not a target-generation claim")
    _, state = _reload_claim_state(claim)
    trace_binding = dict(state.get("bindings", {}).get("trace_producer", {}) or {})
    artifacts = validate_target_artifacts(
        claim.authorization,
        target_map_path,
        trace_producer_binding=trace_binding,
    )
    _require(str(Path(target_map_path).resolve()) == claim.run_binding["target_map_path"], "target output path drift")
    return _finish_phase(
        claim,
        status="completed",
        details={"target_manifest_sha256": artifacts["manifest"]["sha256"]},
        binding_updates={"targets": artifacts},
    )


def _validate_stored_snapshot_binding(
    verified: VerifiedAuthorization, binding: Mapping[str, Any]
) -> Mapping[str, Any]:
    manifest_path = Path(str(binding.get("manifest_path", ""))).resolve()
    _require(manifest_path.is_file(), "authorized snapshot producer manifest is missing")
    _require(sha256_file(manifest_path) == binding.get("manifest_sha256"), "authorized snapshot producer manifest hash drift")
    manifest = load_json_strict(manifest_path)
    _require(isinstance(manifest, Mapping), "authorized snapshot producer manifest is not an object")
    _require(manifest.get("schema") == PRODUCER_MANIFEST_SCHEMA, "authorized snapshot producer schema drift")
    _require(manifest.get("phase") == "snapshot", "authorized snapshot producer phase drift")
    _require(manifest.get("status") == "completed", "authorized snapshot producer is incomplete")
    _require(manifest.get("authorization_id") == verified.authorization_id, "authorized snapshot producer id drift")
    _require(manifest.get("authorization_sha256") == verified.raw_sha256, "authorized snapshot producer auth hash drift")
    _payload_hash(manifest, "manifest_payload_sha256")
    _require(manifest.get("trace_raw_file_set_hash") == binding.get("trace_raw_file_set_hash"), "authorized snapshot trace hash drift")
    claim_path = _claim_path(verified.path, verified.authorization_id, "snapshot")
    _require(claim_path.is_file(), "authorized snapshot phase claim is missing")
    claim_payload = load_json_strict(claim_path)
    _payload_hash(claim_payload, "claim_payload_sha256")
    _require(claim_payload.get("run_id") == binding.get("run_id"), "snapshot claim/binding run id drift")
    _require(claim_payload.get("run_binding") == manifest.get("run_binding"), "snapshot claim run binding drift")
    synthetic_claim = HoldoutPhaseClaim(
        authorization=verified,
        phase="snapshot",
        run_id=str(binding["run_id"]),
        claim_path=claim_path,
        open_stage="snapshot_open",
        success_stage="snapshots_generated",
        failure_stage="snapshot_failed",
        run_binding=dict(claim_payload["run_binding"]),
    )
    _validate_producer_artifacts(synthetic_claim, manifest)
    return manifest


def begin_branch_consumption(
    *,
    authorization_path: Path,
    protocol_path: Path,
    lock_path: Path,
    calibration_manifest_path: Path,
    go_no_go_manifest_path: Path,
    target_map_path: Path,
    branch_output_dir: Path,
    branch_runner_path: Path,
) -> HoldoutPhaseClaim:
    verified = verify_holdout_authorization(
        authorization_path=authorization_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
        calibration_manifest_path=calibration_manifest_path,
        go_no_go_manifest_path=go_no_go_manifest_path,
    )
    _require(verified.state.get("stage") == "snapshots_generated", "branch consumption requires completed targeted snapshots")
    _require(not Path(branch_output_dir).resolve().exists(), "branch output already exists; replay is forbidden")
    target_binding = validate_target_artifacts(
        verified,
        target_map_path,
        trace_producer_binding=verified.state["bindings"].get("trace_producer"),
    )
    _require(target_binding == verified.state["bindings"].get("targets"), "branch target bundle differs from locked target generation")
    snapshot_binding = dict(verified.state["bindings"].get("snapshot_producer", {}) or {})
    _require(snapshot_binding, "branch consumption has no snapshot producer binding")
    snapshot_manifest = _validate_stored_snapshot_binding(verified, snapshot_binding)
    _require(
        snapshot_manifest.get("run_binding", {}).get("target_artifacts")
        == target_binding,
        "snapshot producer target binding drift",
    )
    runner_path = Path(branch_runner_path).resolve()
    _require(runner_path.is_file(), f"missing branch runner: {runner_path}")
    _require(runner_path == BRANCH_RUNNER_PATH.resolve(), "unregistered branch runner path")
    _require(
        sha256_file(runner_path)
        == verified.payload["source"]["branch_runner_sha256"],
        "branch runner differs from the authorized source",
    )
    return _begin_phase(
        verified,
        phase="branch",
        expected_stage="snapshots_generated",
        open_stage="branch_open",
        success_stage="consumed",
        failure_stage="branch_failed",
        run_binding={
            "branch_output_dir": str(Path(branch_output_dir).resolve()),
            "branch_runner": {"path": str(runner_path), "sha256": sha256_file(runner_path)},
            "target_artifacts": target_binding,
            "snapshot_producer": snapshot_binding,
            "snapshot_manifest_sha256": snapshot_binding["manifest_sha256"],
        },
    )


def complete_branch_consumption(
    claim: HoldoutPhaseClaim, branch_manifest_path: Path
) -> Mapping[str, Any]:
    _require(claim.phase == "branch", "not a branch-consumption claim")
    path = Path(branch_manifest_path).resolve()
    _require(str(path.parent) == claim.run_binding["branch_output_dir"], "branch output directory drift")
    _require(path.is_file(), f"missing branch manifest: {path}")
    manifest = load_json_strict(path)
    _require(isinstance(manifest, Mapping), "branch manifest is not an object")
    _require(manifest.get("schema") == "v12_branch_runner_manifest_v1", "branch manifest schema drift")
    _require(manifest.get("status") == "complete", "branch manifest is not complete")
    _require(manifest.get("artifact_role") == "confirmatory_holdout_branch_labels", "branch manifest role drift")
    _require(manifest.get("partition") == "confirmatory_holdout", "branch partition drift")
    _require(manifest.get("method_version") == METHOD_VERSION, "branch method drift")
    _require(manifest.get("authorization_id") == claim.authorization.authorization_id, "branch authorization drift")
    _require(manifest.get("authorization_sha256") == claim.authorization.raw_sha256, "branch authorization hash drift")
    _require(manifest.get("target_artifacts") == claim.run_binding["target_artifacts"], "branch target bundle drift")
    _require(manifest.get("snapshot_producer_manifest_sha256") == claim.run_binding["snapshot_manifest_sha256"], "branch snapshot provenance drift")
    payload_field = "manifest_payload_hash"
    _payload_hash(manifest, payload_field)
    return _finish_phase(
        claim,
        status="completed",
        details={"branch_manifest_sha256": sha256_file(path)},
        binding_updates={
            "branch_manifest": {
                "path": str(path),
                "sha256": sha256_file(path),
                "payload_sha256": manifest[payload_field],
            }
        },
    )


def assert_producer_permit(
    permit: ProducerPermit | None,
    *,
    seed: int,
    capture_snapshots: bool,
) -> None:
    if seed not in CONFIRMATORY_SEEDS:
        return
    _require(isinstance(permit, ProducerPermit), "direct confirmatory run_seed access is forbidden")
    _require(seed in permit.seeds, "producer permit does not cover this seed")
    expected_phase = "snapshot" if capture_snapshots else "trace"
    _require(permit.phase == expected_phase, "producer permit phase/capture mode mismatch")


def validate_runtime_holdout_marker(
    marker: Mapping[str, Any] | None, *, seed: int
) -> Mapping[str, Any]:
    """Validate the config marker immediately before creating a holdout env."""

    _require(seed in CONFIRMATORY_SEEDS, "runtime holdout marker used outside its seed block")
    _require(isinstance(marker, Mapping), "v12 holdout seed has no runtime authorization marker")
    _require(marker.get("schema") == "identifiable_gate_v12_runtime_holdout_permit_v1", "runtime holdout marker schema drift")
    _payload_hash(marker, "marker_payload_sha256")
    _validate_seed_block(marker.get("seed_block"), CONFIRMATORY_SEEDS)
    phase = marker.get("phase")
    _require(phase in {"trace", "snapshot"}, "runtime holdout marker phase drift")
    source_paths = dict(marker.get("source_paths", {}) or {})
    _require(
        set(source_paths)
        == {"protocol", "lock", "calibration_manifest", "go_no_go_manifest"},
        "runtime holdout marker source paths drift",
    )
    verified = verify_holdout_authorization(
        authorization_path=Path(str(marker["authorization_path"])),
        protocol_path=Path(str(source_paths["protocol"])),
        lock_path=Path(str(source_paths["lock"])),
        calibration_manifest_path=Path(str(source_paths["calibration_manifest"])),
        go_no_go_manifest_path=Path(str(source_paths["go_no_go_manifest"])),
    )
    _require(marker.get("authorization_id") == verified.authorization_id, "runtime marker authorization id drift")
    _require(marker.get("authorization_sha256") == verified.raw_sha256, "runtime marker authorization hash drift")
    _require(Path(str(marker.get("state_path", ""))).resolve() == verified.state_path, "runtime marker state path drift")
    expected_stage = f"{phase}_open"
    _require(verified.state.get("stage") == expected_stage, f"runtime marker requires state {expected_stage}")
    _require(marker.get("open_state_sha256") == verified.state["state_payload_sha256"], "runtime marker open-state hash drift")
    _require(verified.state["transition_log"][-1].get("run_id") == marker.get("run_id"), "runtime marker run id drift")
    claim_path = _claim_path(verified.path, verified.authorization_id, str(phase))
    _require(Path(str(marker.get("claim_path", ""))).resolve() == claim_path, "runtime marker claim path drift")
    _require(marker.get("claim_sha256") == sha256_file(claim_path), "runtime marker claim hash drift")
    return marker
