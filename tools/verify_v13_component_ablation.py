"""Verify the current v13 six-arm component-ablation analysis from raw data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    COMPONENT_ABLATION_ARMS,
    FACTORIAL_REPLAY_VERSION,
)
from tools import analyze_v13_component_interventions as analyzer  # noqa: E402
from tools.analyze_factorial_interventions import (  # noqa: E402
    EVENT_ROW_FIELDS,
)
from tools.analyze_v13_component_interventions import (  # noqa: E402
    ANALYSIS_SCHEMA,
    AUDIT_FILE,
    BY_SEED_FIELDS,
    BY_SEED_FILE,
    CLAIM_SCOPE,
    CURRENT_FINAL_ACTUATOR_STAGE,
    CURRENT_RELEASE_SELECTION_STAGE,
    EPISODE_METRICS,
    EVENTS_FILE,
    MAIN_EFFECTS_FILE,
    MANIFEST_FILE,
    OUTPUT_FILES,
    REQUEST_AUDIT_SCHEMA,
    SUMMARY_FILE,
    compute_analysis,
)
from tools.run_main_table_runtime import (  # noqa: E402
    load_formal_protocol,
    resolve_policy_execution_horizon,
)
from tools.run_v13_component_ablation import (  # noqa: E402
    COMPONENT_ABLATION_DESIGN,
    COMPONENT_ABLATION_RUN_SCHEMA,
    COMPONENT_ABLATION_VERSION,
)


VERIFICATION_SCHEMA = "rgd_v13_component_ablation_verification_v2"
SYSTEM_VERSION = "action_aligned_release_gate_v13"
QUERY_GATE_VERSION = "identifiable_gate_v12"
RELEASE_CONTRACT_VERSION = "action_cost_alignment_v2"

_VERIFIER_COMPONENT_NAMES = {
    "latency_survival": "latency_survival",
    "maneuver_breadth": "relative_support_maneuver_breadth",
    "corrective_headroom": "corrective_recovery_headroom",
    "state_need": "state_need",
}
# Retained for compatibility with the existing formal-contract tests.
ARMS = tuple(
    (
        spec.display_name,
        ";".join(_VERIFIER_COMPONENT_NAMES[name] for name in spec.removed_components)
        or "none",
    )
    for spec in COMPONENT_ABLATION_ARMS
)

_H_X_N_FORMULA = "full - without_h - without_n + without_h_and_n"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    require(path.is_file(), f"missing authenticated file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _same_json(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Dict[str, Any]:
    """Load an object while rejecting JavaScript-style NaN/Infinity constants."""

    require(path.is_file(), f"missing JSON: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant,
        object_pairs_hook=_unique_json_object,
    )
    require(isinstance(payload, Mapping), f"JSON root is not an object: {path}")
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _valid_sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    require(_SHA256_PATTERN.fullmatch(digest) is not None, f"invalid {field}")
    return digest


def _integer(value: Any, field: str, *, nonnegative: bool = False) -> int:
    require(not isinstance(value, bool), f"{field} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    require(math.isfinite(number) and number == int(number), f"{field} must be an integer")
    result = int(number)
    if nonnegative:
        require(result >= 0, f"{field} must be nonnegative")
    return result


def _finite(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    require(math.isfinite(number), f"{field} must be finite")
    return number


def close(left: Any, right: Any, message: str, tol: float = 1e-12) -> None:
    left_value = _finite(left, message)
    right_value = _finite(right, message)
    require(
        math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=tol),
        f"{message}: {left_value!r} != {right_value!r}",
    )


def _seed_block(contract: Mapping[str, Any]) -> list[int]:
    seed_range = contract.get("seed_range")
    require(isinstance(seed_range, Mapping), "protocol component seed range is missing")
    start = _integer(seed_range.get("start"), "seed_range.start", nonnegative=True)
    end = _integer(seed_range.get("end"), "seed_range.end", nonnegative=True)
    count = _integer(seed_range.get("count"), "seed_range.count", nonnegative=True)
    require(end >= start and count == end - start + 1, "protocol seed range drift")
    return list(range(start, end + 1))


def _formal_component_contract(
    protocol: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], int, Dict[str, Any]]:
    """Return the locked formal contract used by the runner and verifier."""

    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    contract = dict(submission.get("component_ablation", {}) or {})
    registry = dict(
        (submission.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {}
    )
    registry_spec = dict(registry.get("component_ablation", {}) or {})
    require(contract.get("design") == COMPONENT_ABLATION_DESIGN, "protocol design drift")
    require(
        tuple(str(value) for value in contract.get("arms", ()))
        == tuple(spec.name for spec in COMPONENT_ABLATION_ARMS),
        "protocol six-arm order drift",
    )
    _seed_block(contract)
    require(contract.get("latency_profile") == "fixed", "protocol latency profile drift")
    delay_steps = _integer(
        contract.get("fixed_delay_steps"), "protocol fixed delay", nonnegative=True
    )
    require(delay_steps == 17, "protocol fixed delay must be 17 steps")
    close(contract.get("predicted_latency_s"), 1.7, "protocol predicted latency drift")
    horizon = resolve_policy_execution_horizon(
        dict(contract.get("execution_contract", {}) or {}),
        context="component verifier execution contract",
    ).as_manifest()

    require(registry_spec.get("paper_facing") is True, "component registry is not paper-facing")
    require(
        list(registry_spec.get("data_files", []) or []) == list(OUTPUT_FILES),
        "component registry output inventory drift",
    )
    require(
        registry_spec.get("summary_file") == SUMMARY_FILE,
        "component registry summary-file drift",
    )
    requirements = dict(registry_spec.get("required_manifest_values", {}) or {})
    require(bool(requirements), "component registry requirements are missing")
    formal_pairs = {
        "design": contract.get("design"),
        "arms": list(contract.get("arms", []) or []),
        "latency_profile": contract.get("latency_profile"),
        "fixed_latency_steps": delay_steps,
        "predicted_latency_s": float(contract.get("predicted_latency_s")),
        "delay_s": [float(value) for value in contract.get("delay_s", [])],
        "horizon_steps": _integer(contract.get("horizon_steps"), "horizon_steps"),
        "gamma": float(contract.get("gamma")),
        "epsilon": float(contract.get("corrective_margin")),
        "state_need_floor": float(contract.get("state_need_floor")),
        "budget": _integer(contract.get("budget"), "budget"),
        "cooldown_frames": _integer(contract.get("cooldown_frames"), "cooldown_frames"),
        "cooldown_minimum_query_frame_gap": _integer(
            contract.get("cooldown_minimum_query_frame_gap"),
            "cooldown_minimum_query_frame_gap",
        ),
        "bootstrap_draws": _integer(contract.get("bootstrap_draws"), "bootstrap_draws"),
    }
    for key, expected in formal_pairs.items():
        require(
            key in requirements and _same_json(requirements[key], expected),
            f"protocol/registry requirement drift at {key}",
        )
    for key, expected in horizon.items():
        require(
            key in requirements
            and math.isclose(
                _finite(requirements[key], f"registry {key}"),
                _finite(expected, f"execution {key}"),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"protocol/registry execution drift at {key}",
        )
    return contract, registry_spec, delay_steps, horizon


def _safe_bundle_file(root: Path, relative: str, *, field: str) -> Path:
    require(bool(relative), f"{field} is empty")
    require("\\" not in relative and ":" not in relative, f"invalid {field}: {relative!r}")
    pure = PurePosixPath(relative)
    require(
        not pure.is_absolute()
        and pure.as_posix() == relative
        and all(part not in ("", ".", "..") for part in pure.parts),
        f"invalid {field}: {relative!r}",
    )
    resolved_root = root.resolve()
    path = (resolved_root / Path(*pure.parts)).resolve()
    require(
        path != resolved_root and resolved_root in path.parents,
        f"{field} escapes its bundle: {relative!r}",
    )
    return path


def _verify_recorded_inputs(
    manifest: Mapping[str, Any], source_bundle: Path
) -> Dict[str, str]:
    raw_inventory = manifest.get("input_sha256")
    require(isinstance(raw_inventory, Mapping) and bool(raw_inventory), "input inventory is missing")
    inventory = {str(key): str(value) for key, value in raw_inventory.items()}
    require(len(inventory) == len(raw_inventory), "duplicate normalized input path")
    for relative, expected in inventory.items():
        _valid_sha256(expected, f"input hash for {relative}")
        path = _safe_bundle_file(source_bundle, relative, field="input path")
        require(path.is_file(), f"missing authenticated input: {path}")
        require(sha256(path) == expected, f"input hash drift: {relative}")
    require(
        _canonical_sha256(inventory) == manifest.get("input_inventory_sha256"),
        "input inventory payload hash drift",
    )
    return inventory


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        require(math.isfinite(value), "recomputed CSV contains a non-finite value")
    return str(value)


def _verify_csv(
    path: Path,
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> list[Dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        table = list(csv.reader(handle))
    require(bool(table), f"empty CSV: {path}")
    header = table[0]
    require(bool(header) and all(header), f"invalid CSV header: {path}")
    require(len(header) == len(set(header)), f"duplicate CSV header: {path}")
    expected_header = list(fieldnames or (list(expected_rows[0]) if expected_rows else []))
    require(header == expected_header, f"CSV schema drift: {path.name}")
    require(
        all(len(row) == len(header) for row in table[1:]),
        f"ragged CSV row: {path.name}",
    )
    expected_table = [
        [_csv_cell(row[column]) for column in expected_header] for row in expected_rows
    ]
    require(
        len(table) - 1 == len(expected_table),
        f"CSV row-count drift: {path.name}",
    )
    for index, (observed, expected) in enumerate(zip(table[1:], expected_table), start=1):
        require(observed == expected, f"CSV row drift: {path.name} row {index}")
    return [dict(zip(header, row)) for row in table[1:]]


def _verify_output_hashes(
    artifact: Path,
    manifest: Mapping[str, Any],
    registry_spec: Mapping[str, Any],
) -> Dict[str, str]:
    names = manifest.get("outputs")
    require(isinstance(names, list), "manifest output inventory is missing")
    require(names == list(OUTPUT_FILES), "analysis output inventory/order drift")
    require(
        names == list(registry_spec.get("data_files", []) or []),
        "analysis output inventory differs from registry",
    )
    raw_hashes = manifest.get("output_sha256")
    require(isinstance(raw_hashes, Mapping), "output hash inventory is missing")
    hashes = {str(key): str(value) for key, value in raw_hashes.items()}
    require(set(hashes) == set(OUTPUT_FILES), "output hash inventory drift")
    for name in OUTPUT_FILES:
        _valid_sha256(hashes[name], f"output hash for {name}")
        path = _safe_bundle_file(artifact, name, field="output path")
        require(sha256(path) == hashes[name], f"output hash drift: {name}")
    return hashes


def _verify_audit(
    artifact: Path,
    manifest: Mapping[str, Any],
    recomputed: Mapping[str, Any],
) -> tuple[Dict[str, Any], str]:
    metadata = manifest.get("request_audit")
    require(isinstance(metadata, Mapping), "request-audit binding is missing")
    require(metadata.get("schema") == REQUEST_AUDIT_SCHEMA, "request-audit schema drift")
    require(metadata.get("file") == AUDIT_FILE, "request-audit filename drift")
    audit_path = _safe_bundle_file(artifact, AUDIT_FILE, field="request-audit path")
    audit_sha256 = _valid_sha256(metadata.get("sha256"), "request-audit file hash")
    require(sha256(audit_path) == audit_sha256, "request-audit file hash drift")
    audit = load_json(audit_path)
    payload_sha256 = _valid_sha256(
        metadata.get("payload_sha256"), "request-audit payload hash"
    )
    require(_canonical_sha256(audit) == payload_sha256, "request-audit payload hash drift")
    require(audit.get("schema") == REQUEST_AUDIT_SCHEMA, "request-audit payload schema drift")
    require(audit.get("accepted") is True, "request audit is not accepted")
    require(audit.get("errors") == [], "request audit reports errors")
    require(
        _same_json(audit, recomputed["request_audit"]),
        "request audit differs from raw-bundle recomputation",
    )
    require(
        payload_sha256 == recomputed["request_audit_payload_sha256"],
        "request-audit recomputed payload hash drift",
    )
    return audit, audit_sha256


def _verify_h_x_n(
    recomputed: Mapping[str, Any], effect_rows: Sequence[Mapping[str, str]]
) -> Dict[str, float]:
    context = recomputed["context"]
    seeds = [int(seed) for seed in context["seeds"]]
    matrix = context["matrix"]
    indexed = {
        (str(row.get("effect", "")), str(row.get("metric", ""))): row
        for row in effect_rows
    }
    require(
        len(indexed) == len(effect_rows),
        "component effect table contains duplicate effect/metric rows",
    )
    estimates: Dict[str, float] = {}
    for metric in EPISODE_METRICS:
        values = [
            _finite(matrix[(seed, "full")][metric], f"full/{seed}/{metric}")
            - _finite(
                matrix[(seed, "without_h")][metric], f"without_h/{seed}/{metric}"
            )
            - _finite(
                matrix[(seed, "without_n")][metric], f"without_n/{seed}/{metric}"
            )
            + _finite(
                matrix[(seed, "without_h_and_n")][metric],
                f"without_h_and_n/{seed}/{metric}",
            )
            for seed in seeds
        ]
        estimate = sum(values) / len(values)
        row = indexed.get(("h_x_n_interaction", metric))
        require(row is not None, f"missing H x N effect for {metric}")
        require(row.get("contrast_formula") == _H_X_N_FORMULA, "H x N formula drift")
        require(
            row.get("estimand") == "paired_mean_per_simulator_seed",
            "H x N estimand drift",
        )
        require(
            _integer(row.get("n_seed_blocks"), "H x N seed blocks", nonnegative=True)
            == len(seeds),
            "H x N seed-block count drift",
        )
        close(row.get("estimate"), estimate, f"H x N estimate drift for {metric}")
        estimates[metric] = float(estimate)
    return estimates


def _verify_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    artifact: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    contract: Mapping[str, Any],
    registry_spec: Mapping[str, Any],
    delay_steps: int,
    execution_horizon: Mapping[str, Any],
) -> Path:
    require(manifest.get("schema") == ANALYSIS_SCHEMA, "analysis schema drift")
    require(
        manifest.get("artifact_role") == "paper_facing_component_ablation_evidence",
        "analysis artifact role drift",
    )
    require(manifest.get("status") == "current", "analysis status drift")
    require(manifest.get("accepted") is True, "analysis manifest is not accepted")
    require(
        manifest.get("component_ablation_version") == COMPONENT_ABLATION_VERSION,
        "component-ablation version drift",
    )
    require(
        manifest.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
        "factorial replay version drift",
    )
    require(manifest.get("design") == COMPONENT_ABLATION_DESIGN, "design drift")
    expected_arms = [spec.name for spec in COMPONENT_ABLATION_ARMS]
    require(manifest.get("arms") == expected_arms, "six-arm analysis order drift")
    require(manifest.get("seeds") == _seed_block(contract), "analysis seed cohort drift")
    require(manifest.get("independent_unit") == "simulator_seed", "independent unit drift")
    require(manifest.get("seed_is_experimental_unit") is True, "seed-unit declaration missing")
    require(
        manifest.get("query_events_nested_within_seed") is True,
        "query-event nesting declaration missing",
    )
    require(manifest.get("claim_scope") == CLAIM_SCOPE, "claim scope drift")
    require(
        manifest.get("release_snapshot_stage") == "pre_release_frame_policy_decision",
        "release snapshot stage drift",
    )
    require(
        manifest.get("release_selection_stage") == CURRENT_RELEASE_SELECTION_STAGE,
        "release selection stage drift",
    )
    require(
        manifest.get("final_actuator_action_stage") == CURRENT_FINAL_ACTUATOR_STAGE,
        "final actuator stage drift",
    )
    require(
        manifest.get("branch_design")
        == "matched_release_state_first_action_then_shared_fast_continuation",
        "branch design drift",
    )

    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    versions = {
        "method_version": submission.get("rgd_method_version"),
        "query_gate_method_version": submission.get("query_gate_method_version"),
        "release_contract_version": submission.get("release_contract_version"),
    }
    require(versions["method_version"] == SYSTEM_VERSION, "formal system version drift")
    require(versions["query_gate_method_version"] == QUERY_GATE_VERSION, "formal query-gate version drift")
    require(versions["release_contract_version"] == RELEASE_CONTRACT_VERSION, "formal release-contract version drift")
    for key, expected in versions.items():
        require(manifest.get(key) == expected, f"analysis {key} drift")
    for key, expected in dict(registry_spec.get("required_manifest_values", {}) or {}).items():
        require(
            key in manifest and _same_json(manifest[key], expected),
            f"registry requirement drift at {key}",
        )

    require(
        _integer(manifest.get("horizon_steps"), "analysis horizon", nonnegative=True)
        == _integer(contract.get("horizon_steps"), "protocol horizon", nonnegative=True),
        "analysis horizon drift",
    )
    close(manifest.get("gamma"), contract.get("gamma"), "analysis gamma drift")
    close(
        manifest.get("epsilon"),
        contract.get("corrective_margin"),
        "analysis epsilon drift",
    )
    require(
        _integer(manifest.get("bootstrap_draws"), "analysis bootstrap draws", nonnegative=True)
        == _integer(contract.get("bootstrap_draws"), "protocol bootstrap draws", nonnegative=True),
        "analysis bootstrap-draw drift",
    )
    require(
        _integer(manifest.get("bootstrap_seed"), "analysis bootstrap seed", nonnegative=True)
        == _integer(contract.get("bootstrap_seed"), "protocol bootstrap seed", nonnegative=True),
        "analysis bootstrap-seed drift",
    )
    require(
        manifest.get("bootstrap")
        == {
            "unit": "simulator_seed",
            "draws": int(contract["bootstrap_draws"]),
            "seed": int(contract["bootstrap_seed"]),
        },
        "analysis bootstrap contract drift",
    )
    require(
        _integer(manifest.get("fixed_latency_steps"), "analysis fixed delay")
        == delay_steps,
        "analysis fixed-delay drift",
    )
    for key, expected in execution_horizon.items():
        close(manifest.get(key), expected, f"execution horizon drift at {key}")

    protocol_path = protocol_path.resolve()
    protocol_text = str(manifest.get("protocol", "") or "")
    require(bool(protocol_text), "analysis protocol path is empty")
    require(
        Path(protocol_text).resolve() == protocol_path,
        "analysis protocol path drift",
    )
    require(sha256(protocol_path) == manifest.get("protocol_sha256"), "protocol hash drift")
    source_bundle_text = str(manifest.get("source_bundle", "") or "")
    require(bool(source_bundle_text), "analysis source bundle path is empty")
    source_bundle = Path(source_bundle_text).resolve()
    require(source_bundle.is_dir(), "analysis source bundle is unavailable")

    expected_source = Path(analyzer.__file__).resolve()
    analysis_source_text = str(manifest.get("analysis_source_path", "") or "")
    require(bool(analysis_source_text), "analysis source path is empty")
    require(
        Path(analysis_source_text).resolve() == expected_source,
        "analysis source path drift",
    )
    require(
        sha256(expected_source) == manifest.get("analysis_source_sha256"),
        "analysis source hash drift",
    )
    require(
        manifest.get("outputs") == list(registry_spec.get("data_files", []) or []),
        "analysis outputs differ from formal registry",
    )
    del artifact
    return source_bundle


def verify(artifact: Path, protocol_path: Path) -> Dict[str, Any]:
    artifact = Path(artifact).resolve()
    protocol_path = Path(protocol_path).resolve()
    require(artifact.is_dir(), f"analysis artifact does not exist: {artifact}")
    manifest_path = artifact / MANIFEST_FILE
    require(
        manifest_path.resolve().parent == artifact,
        "analysis manifest escapes the artifact directory",
    )
    manifest = load_json(manifest_path)
    protocol = load_formal_protocol(protocol_path)
    contract, registry_spec, delay_steps, execution_horizon = _formal_component_contract(
        protocol
    )
    source_bundle = _verify_manifest_contract(
        manifest,
        artifact=artifact,
        protocol_path=protocol_path,
        protocol=protocol,
        contract=contract,
        registry_spec=registry_spec,
        delay_steps=delay_steps,
        execution_horizon=execution_horizon,
    )
    output_hashes = _verify_output_hashes(artifact, manifest, registry_spec)
    recorded_inputs = _verify_recorded_inputs(manifest, source_bundle)

    recomputed = compute_analysis(source_bundle, workers=1)
    require(
        Path(recomputed["context"]["protocol_path"]).resolve() == protocol_path,
        "raw bundle is bound to a different protocol",
    )
    require(
        recorded_inputs == recomputed["input_sha256"],
        "raw input inventory differs from recomputed inventory",
    )
    require(
        manifest.get("input_inventory_sha256")
        == recomputed["input_inventory_sha256"],
        "recomputed input inventory hash drift",
    )
    context = recomputed["context"]
    require(
        manifest.get("proposal_bank_sha256") == context["proposal_bank_sha256"],
        "analysis proposal-bank hash drift",
    )
    source_hashes = {
        "source_run_manifest_sha256": sha256(context["run_path"]),
        "source_proposal_manifest_sha256": sha256(context["proposal_path"]),
        "source_episode_results_sha256": sha256(context["result_path"]),
    }
    for key, expected in source_hashes.items():
        require(manifest.get(key) == expected, f"{key} drift")
    require(
        context["run_manifest"].get("schema") == COMPONENT_ABLATION_RUN_SCHEMA,
        "raw run-manifest schema drift",
    )

    audit, audit_sha256 = _verify_audit(artifact, manifest, recomputed)
    require(
        _integer(manifest.get("event_count"), "manifest event_count", nonnegative=True)
        == len(recomputed["events"]),
        "analysis event-count drift",
    )
    require(
        _same_json(manifest.get("summary"), recomputed["summary"]),
        "manifest summary differs from raw recomputation",
    )
    require(
        _same_json(manifest.get("component_contrasts"), recomputed["main_effects"]),
        "manifest component contrasts differ from raw recomputation",
    )

    summary_rows = _verify_csv(
        artifact / SUMMARY_FILE,
        recomputed["summary"],
    )
    event_rows = _verify_csv(
        artifact / EVENTS_FILE,
        recomputed["events"],
        fieldnames=EVENT_ROW_FIELDS,
    )
    by_seed_rows = _verify_csv(
        artifact / BY_SEED_FILE,
        recomputed["by_seed"],
        fieldnames=BY_SEED_FIELDS,
    )
    effect_rows = _verify_csv(
        artifact / MAIN_EFFECTS_FILE,
        recomputed["main_effects"],
    )
    h_x_n_estimates = _verify_h_x_n(recomputed, effect_rows)

    return {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": VERIFICATION_SCHEMA,
        "accepted": True,
        "artifact": str(artifact),
        "source_bundle": str(source_bundle),
        "manifest_sha256": sha256(manifest_path),
        "analysis_source_sha256": manifest["analysis_source_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "input_inventory_sha256": manifest["input_inventory_sha256"],
        "authenticated_input_files": len(recorded_inputs),
        "request_audit_sha256": audit_sha256,
        "request_audit_cells": len(audit["cells"]),
        "output_sha256": output_hashes,
        "output_files": len(output_hashes),
        "arms": list(context["arms"]),
        "seeds": list(context["seeds"]),
        "release_events": len(event_rows),
        "events": len(event_rows),
        "summary_rows": len(summary_rows),
        "by_seed_rows": len(by_seed_rows),
        "component_effect_rows": len(effect_rows),
        "bootstrap_draws": int(recomputed["parameters"]["draws"]),
        "h_x_n_interaction_estimates": h_x_n_estimates,
        "claim_scope": CLAIM_SCOPE,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = Path(args.artifact).resolve()
    report = verify(artifact, Path(args.protocol).resolve())
    report_path = (
        Path(args.report).resolve()
        if args.report is not None
        else artifact / "v13_component_ablation_verification.json"
    )
    _write_json(report_path, report)
    print(_canonical_json(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
