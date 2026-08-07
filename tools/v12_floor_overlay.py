"""Immutable runtime overlay for identifiable-gate v12 calibration floors.

The v12 protocol intentionally retains preregistration placeholders.  A
successful calibration selection is therefore materialized as a separate,
hash-bound overlay instead of rewriting the protocol after selection.  Formal
runtime producers must verify this artifact together with the exact
calibration manifest and apply it before constructing an execution contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "formal_protocol.yaml"
DEFAULT_LOCK_PATH = DEFAULT_PROTOCOL_PATH

METHOD_VERSION = "identifiable_gate_v12"
FLOOR_OVERLAY_SCHEMA = "identifiable_gate_v12_runtime_floor_overlay_v1"
FLOOR_OVERLAY_ROLE = "immutable_calibration_floor_runtime_overlay"
FLOOR_SELECTION_SOURCE = "v12_calibration_selection_manifest"
PROTOCOL_PLACEHOLDER_STATUS = "calibration_placeholder_not_for_deployment"
PROTOCOL_LOCKED_STATUS = "inherited_unchanged_from_v12_calibration_lock"
APPLIED_STATUS = "locked_calibration_overlay_applied"
PROTOCOL_PLACEHOLDER_VALUE = 0.20

FLOOR_FIELDS = {
    "rgd_latency_survival_floor": "lambda_L",
    "rgd_maneuver_breadth_floor": "lambda_A",
    "rgd_corrective_headroom_floor": "lambda_H",
    "rgd_state_need_floor": "lambda_I",
}

V12_PARTITION_RANGES = {
    "calibration": (2000, 2039),
    "go_no_go": (2040, 2059),
    "confirmatory_holdout": (3000, 3029),
    "main": (4000, 4029),
}
V12_OVERLAY_REQUIRED_PARTITIONS = frozenset(
    {"go_no_go", "confirmatory_holdout", "main"}
)

_HEX_256_LENGTH = 64


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _require_exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(payload)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    _require(not missing and not extra, f"{label} keys drift: missing={missing}, extra={extra}")


def _finite_unit(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"{label} must be finite and within [0, 1]")
    return result


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
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


def _verified_calibration(
    calibration_manifest_path: Path,
    *,
    protocol_path: Path,
    lock_path: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    # Strict-load first: the selector's verifier intentionally focuses on
    # semantic content, while this boundary must also reject duplicate keys.
    raw = load_json_strict(calibration_manifest_path)
    _require(isinstance(raw, Mapping), "calibration manifest is not an object")

    from tools.calibrate_identifiable_gate_v12 import (
        load_spec,
        verify_calibration_manifest,
    )

    spec = load_spec(Path(lock_path))
    verified, _ = verify_calibration_manifest(
        Path(calibration_manifest_path),
        spec=spec,
        protocol_path=Path(protocol_path),
        lock_path=Path(lock_path),
    )
    _require(dict(raw) == dict(verified), "strict and semantic calibration loads differ")
    selected = dict(verified.get("selected", {}) or {})
    return verified, selected


def _floor_payload(selected: Mapping[str, Any]) -> dict[str, Any]:
    floors: dict[str, Any] = {}
    for runtime_field, selected_field in FLOOR_FIELDS.items():
        _require(selected_field in selected, f"calibration selection is missing {selected_field}")
        floors[runtime_field] = {
            "selected_field": selected_field,
            "source": FLOOR_SELECTION_SOURCE,
            "value": _finite_unit(selected[selected_field], selected_field),
        }
    return floors


def _build_overlay_payload(
    calibration_manifest_path: Path,
    *,
    protocol_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    calibration, selected = _verified_calibration(
        calibration_manifest_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
    )
    source = dict(calibration.get("source", {}) or {})
    payload: dict[str, Any] = {
        "schema": FLOOR_OVERLAY_SCHEMA,
        "artifact_role": FLOOR_OVERLAY_ROLE,
        "method_version": METHOD_VERSION,
        "floor_source": FLOOR_SELECTION_SOURCE,
        "calibration": {
            "manifest_file": Path(calibration_manifest_path).name,
            "manifest_sha256": sha256_file(calibration_manifest_path),
            "selection_digest": calibration["selection_digest"],
            "lock_id": calibration["lock_id"],
            "candidate_id": selected["candidate_id"],
        },
        "floors": _floor_payload(selected),
        "source": {
            "protocol_sha256": source["protocol_sha256"],
            "lock_sha256": source["lock_sha256"],
            "selector_sha256": source["selector_sha256"],
            "gate_support_sha256": source["gate_support_sha256"],
        },
    }
    payload["overlay_payload_sha256"] = canonical_sha256(payload)
    return payload


def create_floor_overlay(
    output_path: Path,
    *,
    calibration_manifest_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> "VerifiedFloorOverlay":
    """Create an overlay once, then reload it through the strict verifier."""

    payload = _build_overlay_payload(
        Path(calibration_manifest_path),
        protocol_path=Path(protocol_path),
        lock_path=Path(lock_path),
    )
    _atomic_create_json(Path(output_path), payload)
    return load_verified_floor_overlay(
        Path(output_path),
        calibration_manifest_path=Path(calibration_manifest_path),
        protocol_path=Path(protocol_path),
        lock_path=Path(lock_path),
    )


@dataclass(frozen=True)
class VerifiedFloorOverlay:
    path: Path
    protocol_path: Path
    lock_path: Path
    raw_sha256: str
    payload_sha256: str
    calibration_manifest_path: Path
    calibration_manifest_sha256: str
    selection_digest: str
    candidate_id: str
    floors: Mapping[str, float]
    runtime_binding: Mapping[str, Any]
    payload: Mapping[str, Any]


def classify_v12_partition(seeds: Sequence[int]) -> str:
    """Classify a nonempty seed request against the frozen v12 partitions.

    A request may be a strict subset (for resume or a bounded smoke), but it
    may not span two partitions or silently fall into an unregistered range.
    """

    raw_seeds = tuple(seeds)
    _require(raw_seeds, "v12 seed request is empty")
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in raw_seeds),
        "v12 seeds must be integers",
    )
    normalized = tuple(int(seed) for seed in raw_seeds)
    unique = set(normalized)
    matches = []
    for name, (start, end) in V12_PARTITION_RANGES.items():
        if all(start <= seed <= end for seed in unique):
            matches.append(name)
    _require(
        len(matches) == 1,
        "v12 seed request is outside one frozen partition or spans partitions",
    )
    return matches[0]


def enforce_v12_floor_overlay_contract(
    protocol_name: str,
    seeds: Sequence[int],
    verified: "VerifiedFloorOverlay | None",
    *,
    allow_nonformal: bool = False,
) -> str | None:
    """Enforce the common partition/overlay rule at a runner boundary.

    Calibration traces are intentionally produced before selection and must
    not consume a selected overlay.  Every locked evaluation partition must
    consume one verified overlay.  ``allow_nonformal`` is an explicit escape
    hatch for diagnostics outside the preregistered ranges; formal runners
    leave it false.
    """

    is_v12 = str(protocol_name or "") == "rgd_tvt_identifiable_gate_v12"
    if not is_v12:
        _require(verified is None, "v12 floor overlay cannot be used with another protocol")
        return None
    try:
        partition = classify_v12_partition(seeds)
    except ValueError:
        if allow_nonformal:
            return None
        raise
    if partition == "calibration":
        _require(
            verified is None,
            "calibration partition must run before selection and cannot consume a floor overlay",
        )
    else:
        _require(
            partition in V12_OVERLAY_REQUIRED_PARTITIONS and verified is not None,
            f"v12 {partition} partition requires a verified immutable floor overlay",
        )
    return partition


def load_verified_floor_overlay(
    floor_overlay_path: Path,
    *,
    calibration_manifest_path: Path,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> VerifiedFloorOverlay:
    """Load an overlay only if every calibration and source binding is current."""

    floor_overlay_path = Path(floor_overlay_path).resolve()
    calibration_manifest_path = Path(calibration_manifest_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    lock_path = Path(lock_path).resolve()
    payload = load_json_strict(floor_overlay_path)
    _require(isinstance(payload, Mapping), "floor overlay is not an object")
    _require_exact_keys(
        payload,
        {
            "schema",
            "artifact_role",
            "method_version",
            "floor_source",
            "calibration",
            "floors",
            "source",
            "overlay_payload_sha256",
        },
        "floor overlay",
    )
    _require(payload.get("schema") == FLOOR_OVERLAY_SCHEMA, "floor overlay schema drift")
    _require(payload.get("artifact_role") == FLOOR_OVERLAY_ROLE, "floor overlay role drift")
    _require(payload.get("method_version") == METHOD_VERSION, "floor overlay method drift")
    _require(payload.get("floor_source") == FLOOR_SELECTION_SOURCE, "floor overlay source drift")

    observed_payload_sha = payload.get("overlay_payload_sha256")
    without_hash = dict(payload)
    without_hash.pop("overlay_payload_sha256", None)
    expected_payload_sha = canonical_sha256(without_hash)
    _require(observed_payload_sha == expected_payload_sha, "floor overlay payload hash mismatch")

    calibration, selected = _verified_calibration(
        calibration_manifest_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
    )
    expected_payload = _build_overlay_payload(
        calibration_manifest_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
    )
    _require(dict(payload) == expected_payload, "floor overlay differs from the verified calibration selection")

    calibration_block = dict(payload.get("calibration", {}) or {})
    source = dict(payload.get("source", {}) or {})
    floors_block = dict(payload.get("floors", {}) or {})
    _require_exact_keys(
        calibration_block,
        {"manifest_file", "manifest_sha256", "selection_digest", "lock_id", "candidate_id"},
        "floor overlay calibration",
    )
    _require_exact_keys(source, {"protocol_sha256", "lock_sha256", "selector_sha256", "gate_support_sha256"}, "floor overlay source")
    _require_exact_keys(floors_block, set(FLOOR_FIELDS), "floor overlay floors")
    floors: dict[str, float] = {}
    for runtime_field, selected_field in FLOOR_FIELDS.items():
        floor = dict(floors_block.get(runtime_field, {}) or {})
        _require_exact_keys(floor, {"selected_field", "source", "value"}, runtime_field)
        _require(floor["selected_field"] == selected_field, f"{runtime_field} selected-field drift")
        _require(floor["source"] == FLOOR_SELECTION_SOURCE, f"{runtime_field} source drift")
        value = _finite_unit(floor["value"], runtime_field)
        _require(value == _finite_unit(selected[selected_field], selected_field), f"{runtime_field} differs from calibration selection")
        floors[runtime_field] = value

    raw_sha = sha256_file(floor_overlay_path)
    calibration_sha = sha256_file(calibration_manifest_path)
    runtime_binding: dict[str, Any] = {
        "schema": FLOOR_OVERLAY_SCHEMA,
        "method_version": METHOD_VERSION,
        "floor_overlay_sha256": raw_sha,
        "floor_overlay_payload_sha256": expected_payload_sha,
        "calibration_manifest_sha256": calibration_sha,
        "calibration_selection_digest": calibration["selection_digest"],
        "calibration_lock_id": calibration["lock_id"],
        "candidate_id": selected["candidate_id"],
        "floor_source": FLOOR_SELECTION_SOURCE,
        "floors": dict(floors),
        "protocol_sha256": source["protocol_sha256"],
        "lock_sha256": source["lock_sha256"],
        "selector_sha256": source["selector_sha256"],
        "gate_support_sha256": source["gate_support_sha256"],
    }
    return VerifiedFloorOverlay(
        path=floor_overlay_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
        raw_sha256=raw_sha,
        payload_sha256=expected_payload_sha,
        calibration_manifest_path=calibration_manifest_path,
        calibration_manifest_sha256=calibration_sha,
        selection_digest=str(calibration["selection_digest"]),
        candidate_id=str(selected["candidate_id"]),
        floors=floors,
        runtime_binding=runtime_binding,
        payload=payload,
    )


def load_optional_verified_floor_overlay(
    floor_overlay_path: Path | None,
    *,
    calibration_manifest_path: Path | None,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    lock_path: Path = DEFAULT_LOCK_PATH,
) -> VerifiedFloorOverlay | None:
    """Load one verified overlay, or require that both optional inputs are absent.

    Formal runners share this boundary so an overlay can never be supplied
    without the exact calibration manifest that authenticates it (or vice
    versa).  Partition-specific requirements remain the responsibility of
    :func:`enforce_v12_floor_overlay_contract`.
    """

    supplied = {
        "floor_overlay_path": floor_overlay_path,
        "calibration_manifest_path": calibration_manifest_path,
    }
    if not any(value is not None for value in supplied.values()):
        return None
    missing = [name for name, value in supplied.items() if value is None]
    _require(
        not missing,
        "v12 floor overlay inputs must be supplied together; missing "
        + ", ".join(missing),
    )
    return load_verified_floor_overlay(
        Path(floor_overlay_path),
        calibration_manifest_path=Path(calibration_manifest_path),
        protocol_path=Path(protocol_path),
        lock_path=Path(lock_path),
    )


def _core_story(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    slow = cfg.setdefault("slow_thinking", {})
    _require(isinstance(slow, MutableMapping), "slow_thinking config must be an object")
    risk = slow.setdefault("risk_coupling", {})
    _require(isinstance(risk, MutableMapping), "risk_coupling config must be an object")
    core = risk.setdefault("core_story", {})
    _require(isinstance(core, MutableMapping), "core_story config must be an object")
    return core


def _provenance_updates(verified: VerifiedFloorOverlay) -> dict[str, Any]:
    binding = verified.runtime_binding
    return {
        "rgd_floor_selection_source": FLOOR_SELECTION_SOURCE,
        "rgd_floor_calibration_manifest_sha256": binding["calibration_manifest_sha256"],
        "rgd_floor_selection_digest": binding["calibration_selection_digest"],
        "rgd_floor_candidate_id": binding["candidate_id"],
        "rgd_floor_overlay_sha256": binding["floor_overlay_sha256"],
        "rgd_floor_overlay_payload_sha256": binding["floor_overlay_payload_sha256"],
        "rgd_floor_protocol_sha256": binding["protocol_sha256"],
        "rgd_floor_lock_sha256": binding["lock_sha256"],
        "rgd_floor_selector_sha256": binding["selector_sha256"],
        "rgd_floor_gate_support_sha256": binding["gate_support_sha256"],
    }


def apply_floor_overlay(
    base_cfg: Mapping[str, Any],
    verified: VerifiedFloorOverlay,
    *,
    formal_runtime: bool = True,
) -> dict[str, Any]:
    """Return a copied config with the single verified floor tuple injected.

    In formal mode the input must contain either the preregistration placeholder
    or the exact locked tuple embedded in the unified formal protocol. This
    rejects protocol edits, CLI overrides, and second application before any
    runtime contract is constructed.
    """

    cfg = deepcopy(dict(base_cfg))
    _require("_v12_floor_overlay" not in cfg, "runtime config already contains a floor overlay")
    core = _core_story(cfg)
    if formal_runtime:
        status = core.get("v12_floor_status")
        _require(
            status in {PROTOCOL_PLACEHOLDER_STATUS, PROTOCOL_LOCKED_STATUS},
            "formal runtime requires the registered v12 floor status",
        )
        for field in FLOOR_FIELDS:
            _require(field in core, f"formal protocol is missing placeholder {field}")
            observed = _finite_unit(core[field], field)
            expected = (
                PROTOCOL_PLACEHOLDER_VALUE
                if status == PROTOCOL_PLACEHOLDER_STATUS
                else float(verified.floors[field])
            )
            _require(observed == expected, f"formal protocol/CLI floor override detected for {field}")
            _require(
                f"{field}_source" not in core,
                f"formal protocol already overrides {field}_source",
            )
        for field in _provenance_updates(verified):
            _require(field not in core, f"formal protocol already contains runtime provenance {field}")

    for field, value in verified.floors.items():
        core[field] = float(value)
        core[f"{field}_source"] = FLOOR_SELECTION_SOURCE
    core.update(_provenance_updates(verified))
    core["v12_floor_status"] = APPLIED_STATUS
    cfg["_v12_floor_overlay"] = {
        **deepcopy(dict(verified.runtime_binding)),
        "floor_overlay_path": str(verified.path),
        "calibration_manifest_path": str(verified.calibration_manifest_path),
        "protocol_path": str(verified.protocol_path),
        "calibration_lock_path": str(verified.lock_path),
    }

    from dilu.driver_agent.reasoning.rgd_support import build_rgd_execution_contract

    cfg["_rgd_runtime_contract"] = build_rgd_execution_contract(dict(core)).to_dict()
    assert_floor_overlay_applied(cfg, verified)
    return cfg


def assert_floor_overlay_applied(
    cfg: Mapping[str, Any], verified: VerifiedFloorOverlay
) -> None:
    """Fail if any post-injection override changed the locked runtime tuple."""

    observed_binding = dict(cfg.get("_v12_floor_overlay", {}) or {})
    expected_binding = dict(verified.runtime_binding)
    for field, value in expected_binding.items():
        _require(observed_binding.get(field) == value, f"runtime floor overlay binding drift: {field}")
    _require(observed_binding.get("floor_overlay_path") == str(verified.path), "runtime floor overlay path drift")
    _require(
        observed_binding.get("calibration_manifest_path") == str(verified.calibration_manifest_path),
        "runtime calibration manifest path drift",
    )
    _require(observed_binding.get("protocol_path") == str(verified.protocol_path), "runtime protocol path drift")
    _require(observed_binding.get("calibration_lock_path") == str(verified.lock_path), "runtime calibration lock path drift")
    mutable_cfg = deepcopy(dict(cfg))
    core = _core_story(mutable_cfg)
    _require(core.get("v12_floor_status") == APPLIED_STATUS, "runtime floor overlay is not marked applied")
    for field, value in verified.floors.items():
        _require(_finite_unit(core.get(field), field) == value, f"runtime floor drift: {field}")
        _require(core.get(f"{field}_source") == FLOOR_SELECTION_SOURCE, f"runtime floor source drift: {field}")
    for field, value in _provenance_updates(verified).items():
        _require(core.get(field) == value, f"runtime floor provenance drift: {field}")

    contract = dict(cfg.get("_rgd_runtime_contract", {}) or {})
    gate = dict(contract.get("gate_definition", {}) or {})
    for field, value in verified.floors.items():
        gate_field = field.removeprefix("rgd_")
        _require(_finite_unit(gate.get(gate_field), gate_field) == value, f"runtime contract floor drift: {gate_field}")


def _runtime_protocol_name(cfg: Mapping[str, Any]) -> str:
    for key in ("_runtime_experiment_config", "_paper_protocol_config"):
        embedded = cfg.get(key)
        if isinstance(embedded, Mapping):
            name = str(embedded.get("protocol_name", "") or "")
            if name:
                return name
    return str(cfg.get("protocol_name", "") or "")


def assert_v12_runtime_seed_contract(cfg: Mapping[str, Any], seed: int) -> str | None:
    """Final shared guard called before a simulator environment is opened."""

    protocol_name = _runtime_protocol_name(cfg)
    if protocol_name != "rgd_tvt_identifiable_gate_v12":
        return None
    partition = classify_v12_partition([seed])
    binding = cfg.get("_v12_floor_overlay")
    if partition == "calibration":
        enforce_v12_floor_overlay_contract(protocol_name, [seed], None)
        _require(not binding, "calibration runtime cannot contain a selected floor overlay")
        mutable_cfg = deepcopy(dict(cfg))
        core = _core_story(mutable_cfg)
        _require(
            core.get("v12_floor_status") == PROTOCOL_PLACEHOLDER_STATUS,
            "calibration runtime must retain the preregistered floor placeholder",
        )
        return partition

    _require(isinstance(binding, Mapping), f"v12 {partition} runtime is missing its floor overlay binding")
    required_paths = {
        "floor_overlay_path",
        "calibration_manifest_path",
        "protocol_path",
        "calibration_lock_path",
    }
    _require(required_paths.issubset(binding), f"v12 {partition} runtime overlay paths are incomplete")
    verified = load_verified_floor_overlay(
        Path(str(binding["floor_overlay_path"])),
        calibration_manifest_path=Path(str(binding["calibration_manifest_path"])),
        protocol_path=Path(str(binding["protocol_path"])),
        lock_path=Path(str(binding["calibration_lock_path"])),
    )
    enforce_v12_floor_overlay_contract(protocol_name, [seed], verified)
    assert_floor_overlay_applied(cfg, verified)
    if partition == "confirmatory_holdout":
        marker = cfg.get("_v12_holdout_authorization")
        _require(isinstance(marker, Mapping), "confirmatory holdout runtime is missing its central authorization marker")
        from tools.v12_holdout_guard import validate_runtime_holdout_marker

        validate_runtime_holdout_marker(marker, seed=int(seed))
    return partition


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("create", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--floor-overlay", type=Path, required=True)
        sub.add_argument("--calibration-manifest", type=Path, required=True)
        sub.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
        sub.add_argument("--calibration-lock", type=Path, default=DEFAULT_LOCK_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        verified = create_floor_overlay(
            args.floor_overlay,
            calibration_manifest_path=args.calibration_manifest,
            protocol_path=args.protocol,
            lock_path=args.calibration_lock,
        )
    else:
        verified = load_verified_floor_overlay(
            args.floor_overlay,
            calibration_manifest_path=args.calibration_manifest,
            protocol_path=args.protocol,
            lock_path=args.calibration_lock,
        )
    print(
        json.dumps(
            verified.runtime_binding,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
