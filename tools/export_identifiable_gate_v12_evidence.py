"""Publish locked v12 validation as the paper evidence registry bundle.

The exporter is intentionally an adapter, not an analyzer.  It reapplies the
single threshold tuple recorded by ``validate-locked`` only to verify the
upstream tables and to materialize their event-level rows.  It never enumerates
or selects threshold candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.calibrate_identifiable_gate_v12 import (  # noqa: E402
    ARM_FULL,
    COMPONENT_ARMS,
    LABEL_SOURCE,
    LOCK_PATH,
    METHOD_VERSION,
    VALIDATION_ARMS,
    Thresholds,
    _locked_counts_by_seed,
    _locked_exposure_compliance,
    _locked_metrics_payload,
    _semantic_hash,
    _trace_path,
    arm_metrics,
    build_opportunity_table,
    component_variation,
    load_branch_labels,
    load_branch_manifest,
    load_features,
    load_spec,
    locked_acceptance,
    locked_bootstrap,
    locked_by_seed_delay_rows,
    locked_factorial_cohorts,
    locked_geometry,
    locked_partition_spec,
    locked_summary_rows,
    validate_protocol_contract,
)
from tools.run_v12_branch_labels import (  # noqa: E402
    BRANCH_ROWS_FILE,
    LABELS_FILE,
    MANIFEST_FILE as BRANCH_MANIFEST_FILE,
    _derive_release_outcome,
)


VALIDATION_MANIFEST_FILE = "v12_validation_manifest.json"
VALIDATION_SUMMARY_FILE = "v12_validation_summary.csv"
VALIDATION_BY_SEED_FILE = "v12_validation_by_seed_delay.csv"

RELEASE_MANIFEST_FILE = "release_rollout_manifest.json"
RELEASE_SUMMARY_FILE = "release_rollout_summary.csv"
RELEASE_EVENTS_FILE = "release_rollout_events.csv"
RELEASE_BRANCHES_FILE = "release_rollout_branches.csv"

COMPONENT_MANIFEST_FILE = "component_ablation_manifest.json"
COMPONENT_SUMMARY_FILE = "component_ablation_summary.csv"
COMPONENT_EVENTS_FILE = "component_ablation_events.csv"
COMPONENT_BY_SEED_FILE = "component_ablation_by_seed.csv"
COMPONENT_MAIN_EFFECTS_FILE = "component_ablation_main_effects.csv"

BUNDLE_MANIFEST_FILE = "v12_evidence_bundle_manifest.json"
BUNDLE_SCHEMA = "identifiable_gate_v12_evidence_bundle_v1"
RELEASE_SCHEMA = "identifiable_gate_v12_release_rollout_evidence_v1"
COMPONENT_SCHEMA = "identifiable_gate_v12_component_ablation_evidence_v1"
VALIDATION_SCHEMA = "identifiable_gate_v12_locked_analysis_v1"
BRANCH_SCHEMA = "v12_branch_runner_manifest_v1"
CONFIRMATORY_SEEDS = tuple(range(3000, 3030))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    path = Path(path)
    _require(path.is_file(), f"missing input artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_sha256(payload: Mapping[str, Any], field: str) -> str:
    normalized = dict(payload)
    normalized.pop(field, None)
    return _canonical_sha256(normalized)


def _load_json(path: Path) -> Dict[str, Any]:
    def reject_duplicate(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value!r} in {path}")

    payload = json.loads(
        Path(path).read_text(encoding="utf-8-sig"),
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    _require(bool(fields), f"CSV has no header: {path}")
    _require(len(fields) == len(set(fields)), f"CSV has duplicate columns: {path}")
    _require(bool(rows), f"CSV has no data rows: {path}")
    _require(all(None not in row for row in rows), f"CSV has over-wide row: {path}")
    return fields, rows


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    _require(bool(rows), "refusing to render an empty evidence table")
    fields = list(rows[0])
    _require(
        all(list(row) == fields for row in rows),
        "evidence table rows do not share one ordered schema",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _assert_csv_exact(path: Path, rows: Sequence[Mapping[str, Any]], label: str) -> None:
    observed = Path(path).read_text(encoding="utf-8-sig")
    _require(observed == _csv_text(rows), f"{label} table differs from locked derivation")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(str(temporary), str(path))


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_text(path, _csv_text(rows))


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        try:
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return left == right


def _dotted(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for item in dotted.split("."):
        if not isinstance(current, Mapping) or item not in current:
            return None
        current = current[item]
    return current


def _hex_digest(value: Any, field: str, *, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    text = str(value or "")
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{field} is not a lowercase SHA256 digest",
    )
    return text


def _threshold_from_manifest(
    manifest: Mapping[str, Any], spec: Any
) -> Thresholds:
    locked = dict(manifest.get("locked_thresholds", {}) or {})

    def unit(field: str, allowed: Iterable[int]) -> int:
        value = float(locked.get(field, float("nan")))
        scaled = int(round(100.0 * value))
        _require(
            math.isfinite(value)
            and math.isclose(value, scaled / 100.0, rel_tol=0.0, abs_tol=1e-12)
            and scaled in set(allowed),
            f"locked threshold {field} is outside the preregistered grid",
        )
        return scaled

    threshold = Thresholds(
        unit("lambda_L", spec.floor_units),
        unit("lambda_A", spec.floor_units),
        unit("lambda_H", spec.floor_units),
        unit("lambda_I", (spec.i_floor_units,)),
    )
    _require(
        locked.get("candidate_id") == threshold.candidate_id,
        "locked threshold candidate identity drift",
    )
    _require(
        locked == {"candidate_id": threshold.candidate_id, **threshold.as_floats()},
        "locked threshold schema drift",
    )
    return threshold


def validate_validation_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol_sha256: str,
    lock_sha256: str,
) -> None:
    """Fail-closed structural guard used by the exporter and focused tests."""

    _require(manifest.get("schema") == VALIDATION_SCHEMA, "validation manifest schema drift")
    _require(
        manifest.get("artifact_role") == "validation_locked_analysis",
        "validation manifest role drift",
    )
    _require(manifest.get("partition") == "validation", "validation partition drift")
    _require(manifest.get("method_version") == METHOD_VERSION, "validation method version drift")
    _require(
        tuple(int(seed) for seed in manifest.get("seed_block", []) or [])
        == CONFIRMATORY_SEEDS,
        "validation seed block must be exactly 3000-3029",
    )
    _require(
        manifest.get("parameter_search_performed") is False,
        "validation manifest performed parameter search",
    )
    source = dict(manifest.get("source", {}) or {})
    _require(source.get("protocol_sha256") == protocol_sha256, "validation protocol hash drift")
    _require(source.get("lock_sha256") == lock_sha256, "validation lock hash drift")
    _require(source.get("holdout_final_stage") == "consumed", "holdout workflow is not consumed")
    _require(bool(str(source.get("holdout_authorization_id", "") or "")), "holdout authorization id missing")
    for field in (
        "calibration_manifest_sha256",
        "go_no_go_manifest_sha256",
        "trace_semantic_hash",
        "trace_raw_file_set_hash",
        "branch_label_semantic_hash",
        "branch_label_raw_sha256",
        "holdout_authorization_sha256",
        "selector_sha256",
        "gate_support_sha256",
    ):
        _hex_digest(source.get(field), f"validation source.{field}")

    acceptance = dict(manifest.get("acceptance", {}) or {})
    _require(
        manifest.get("paper_acceptance") == acceptance,
        "paper acceptance differs from validation acceptance",
    )
    _require(acceptance.get("scope") == "confirmatory_holdout", "acceptance scope drift")
    _require(acceptance.get("validation_evaluated") is True, "validation was not evaluated")
    _require(acceptance.get("validation_passed") is True, "validation acceptance failed")
    _require(acceptance.get("paper_facing_passed") is True, "paper-facing acceptance failed")
    _require(acceptance.get("passed") is True, "confirmatory acceptance failed")

    digest_payload = {
        "partition": "validation",
        "seed_block": list(CONFIRMATORY_SEEDS),
        "locked_thresholds": manifest.get("locked_thresholds"),
        "source": source,
        "metrics": manifest.get("metrics"),
        "geometry": manifest.get("geometry"),
        "acceptance": acceptance,
    }
    _require(
        manifest.get("analysis_digest") == _semantic_hash(digest_payload),
        "validation analysis digest drift",
    )


def _validate_branch_manifest(
    manifest: Mapping[str, Any],
    *,
    trace_root: Path,
    validation_source: Mapping[str, Any],
) -> None:
    _require(manifest.get("schema") == BRANCH_SCHEMA, "branch manifest schema drift")
    _require(manifest.get("status") == "complete", "branch manifest is not complete")
    _require(
        manifest.get("artifact_role") == "confirmatory_holdout_branch_labels",
        "branch artifact role drift",
    )
    _require(
        manifest.get("partition") == "confirmatory_holdout",
        "branch publication partition drift",
    )
    _require(manifest.get("v12_partition") == "validation", "branch v12 partition drift")
    _require(manifest.get("method_version") == METHOD_VERSION, "branch method version drift")
    _require(manifest.get("gate_selection_performed") is False, "branch runner selected gate parameters")
    _require(manifest.get("exact_action_provenance") == "exact", "branch action provenance is not exact")
    _require(
        tuple(int(seed) for seed in manifest.get("seeds", []) or []) == CONFIRMATORY_SEEDS,
        "branch seed block must be exactly 3000-3029",
    )
    _require(Path(str(manifest.get("trace_root", ""))).resolve() == trace_root.resolve(), "branch trace root drift")
    _require(
        manifest.get("authorization_id") == validation_source.get("holdout_authorization_id"),
        "branch/validation authorization id drift",
    )
    _require(
        manifest.get("authorization_sha256")
        == validation_source.get("holdout_authorization_sha256"),
        "branch/validation authorization hash drift",
    )
    _require(
        manifest.get("manifest_payload_hash")
        == _payload_sha256(manifest, "manifest_payload_hash"),
        "branch manifest payload hash drift",
    )


def _validate_registry_values(
    manifest: Mapping[str, Any], registry_spec: Mapping[str, Any], label: str
) -> None:
    requirements = registry_spec.get("required_manifest_values", {}) or {}
    _require(isinstance(requirements, Mapping), f"{label} registry requirements are invalid")
    for key, expected in requirements.items():
        observed = _dotted(manifest, str(key))
        _require(
            _same(observed, expected),
            f"{label} manifest {key}={observed!r}, expected {expected!r}",
        )


def _arm_mask(arm_label: str) -> Dict[str, int]:
    arm = next(item for item in VALIDATION_ARMS if item.label == arm_label)
    return {"L": int(arm.use_l), "A": int(arm.use_a), "H": int(arm.use_h)}


def _fraction(numerator: int, denominator: int) -> Any:
    return "" if denominator == 0 else float(numerator / denominator)


def _component_summary_rows(
    locked_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source in locked_rows:
        if source["stratum"] != "pooled":
            continue
        rows.append(
            {
                "arm": source["arm"],
                **_arm_mask(str(source["arm"])),
                "Q": source["Q"],
                "R": source["R"],
                "C": source["C"],
                "excluded": source["excluded"],
                "CSet": source["CSet"],
                "Q_per_C": source["Q_per_C"],
                "R_per_C": source["R_per_C"],
                "scheduled_cohort_hash": source["scheduled_cohort_hash"],
                "evaluated_cohort_hash": source["evaluated_cohort_hash"],
            }
        )
    return rows


def _component_by_seed_rows(
    locked_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for row in locked_rows:
        grouped.setdefault((str(row["arm"]), int(row["seed"])), []).append(row)
    rows: List[Dict[str, Any]] = []
    for arm in [item.label for item in VALIDATION_ARMS]:
        for seed in CONFIRMATORY_SEEDS:
            source = grouped[(arm, seed)]
            _require(len(source) == 3, f"{arm} seed {seed}: incomplete delay grid")
            q = sum(int(row["Q"]) for row in source)
            r = sum(int(row["R"]) for row in source)
            c = sum(int(row["C"]) for row in source)
            excluded = sum(int(row["excluded"]) for row in source)
            rows.append(
                {
                    "arm": arm,
                    **_arm_mask(arm),
                    "seed": seed,
                    "Q": q,
                    "R": r,
                    "C": c,
                    "excluded": excluded,
                    "CSet": _fraction(c, r),
                    "Q_per_C": _fraction(q, c),
                    "R_per_C": _fraction(r, c),
                }
            )
    return rows


def _component_main_effect_rows(
    metrics: Mapping[str, Any], bootstrap: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    full = metrics[ARM_FULL]
    marginal = dict(bootstrap.get("cset_margin_marginal_lower", {}) or {})
    simultaneous = float(bootstrap["cset_margin_simultaneous_minimum_lower"])
    rows: List[Dict[str, Any]] = []
    for component in ("L", "A", "H"):
        comparator = COMPONENT_ARMS[component]
        other = metrics[comparator]
        margin = float(full.rate - other.rate)
        lower = float(marginal[comparator])
        rows.append(
            {
                "component": component,
                "contrast": "full_minus_leave_one_out",
                "full_arm": ARM_FULL,
                "comparator_arm": comparator,
                "full_CSet": float(full.rate),
                "comparator_CSet": float(other.rate),
                "margin_CSet": margin,
                "marginal_one_sided_ci_lower": lower,
                "simultaneous_minimum_one_sided_ci_lower": simultaneous,
                "passed": int(margin > 0.0 and lower > 0.0 and simultaneous > 0.0),
            }
        )
    return rows


def _event_rows(
    table: Any,
    cohorts: Mapping[str, Any],
    labels: Any,
    raw_labels: Mapping[Tuple[int, int, int, int], Mapping[str, Any]],
    threshold: Thresholds,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm in [item.label for item in VALIDATION_ARMS]:
        mask = _arm_mask(arm)
        for index in cohorts[arm].evaluated:
            event = table.rows[index]
            key = event.event_key
            raw = raw_labels[key]
            rows.append(
                {
                    "arm": arm,
                    **mask,
                    "seed": event.seed,
                    "delay_s": event.delay_s,
                    "delay_steps": event.delay_steps,
                    "query_frame": event.query_frame,
                    "release_frame": event.release_frame,
                    "candidate_state_id": f"{event.seed}:{event.query_frame}:{event.delay_steps}",
                    "release_state_id": f"{event.seed}:{event.release_frame}",
                    "latency_survival": float(table.l[index]),
                    "maneuver_breadth": float(table.a[index]),
                    "corrective_headroom": float(table.h[index]),
                    "state_need": float(table.i[index]),
                    "lambda_L": threshold.l / 100.0,
                    "lambda_A": threshold.a / 100.0,
                    "lambda_H": threshold.h / 100.0,
                    "lambda_I": threshold.i / 100.0,
                    "corrective_set_nonempty": int(labels.labels[key]),
                    "release_state_identity_sha256": raw["release_state_identity_sha256"],
                    "corrective_set_action_count": raw.get("corrective_set_action_count", ""),
                    "best_advantage": raw.get("best_advantage", ""),
                    "baseline_utility": raw.get("baseline_utility", ""),
                    "baseline_collision": raw.get("baseline_collision", ""),
                    "method_version": METHOD_VERSION,
                    "label_source": LABEL_SOURCE,
                    "exact_action_provenance": 1,
                }
            )
    return rows


def _validate_branch_outcome_derivation(
    raw_branches: Sequence[Mapping[str, Any]],
    raw_labels: Mapping[Tuple[int, int, int, int], Mapping[str, Any]],
    *,
    epsilon: float,
) -> Dict[Tuple[int, int], List[Mapping[str, Any]]]:
    branch_groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in raw_branches:
        key = (int(row["seed"]), int(row["release_frame"]))
        branch_groups.setdefault(key, []).append(row)
    label_groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in raw_labels.values():
        key = (int(row["seed"]), int(row["release_frame"]))
        label_groups.setdefault(key, []).append(row)
    _require(set(branch_groups) == set(label_groups), "branch/label release-state inventory drift")
    for (seed, frame), rows in sorted(branch_groups.items()):
        baselines = [row for row in rows if str(row["branch_role"]) == "matched_fast"]
        candidates = [row for row in rows if str(row["branch_role"]) == "candidate"]
        _require(len(baselines) == 1 and bool(candidates), f"release {(seed, frame)}: invalid branch roles")
        baseline = baselines[0]
        outcome = _derive_release_outcome(
            baseline,
            candidates,
            seed=seed,
            frame=frame,
            predicted_action=int(baseline["fast_action"]),
            epsilon=epsilon,
        )
        expected_corrective = int(bool(outcome["corrective_rows"]))
        expected_count = len(outcome["corrective_rows"])
        for label in label_groups[(seed, frame)]:
            _require(
                int(label["corrective_set_nonempty"]) == expected_corrective,
                f"release {(seed, frame)}: corrective label does not derive from branches",
            )
            _require(
                int(label["corrective_set_action_count"]) == expected_count,
                f"release {(seed, frame)}: corrective action count drift",
            )
    return branch_groups


def _release_summary_rows(
    locked_rows: Sequence[Mapping[str, Any]], spec: Any
) -> List[Dict[str, Any]]:
    seconds = {f"delay_{step}": delay for step, delay in zip(spec.delay_steps, spec.delay_seconds)}
    rows: List[Dict[str, Any]] = []
    for source in locked_rows:
        if source["arm"] != ARM_FULL:
            continue
        stratum = str(source["stratum"])
        rows.append(
            {
                "allocator": "RGD",
                "stratum": stratum,
                "delay_s": "" if stratum == "pooled" else seconds[stratum],
                "Q": source["Q"],
                "R": source["R"],
                "C": source["C"],
                "excluded": source["excluded"],
                "CSet": source["CSet"],
                "Q_per_C": source["Q_per_C"],
                "R_per_C": source["R_per_C"],
                "scheduled_cohort_hash": source["scheduled_cohort_hash"],
                "evaluated_cohort_hash": source["evaluated_cohort_hash"],
            }
        )
    return rows


def _load_protocol(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "formal protocol is not a mapping")
    return dict(payload)


def _build_provenance(paths: Mapping[str, Path]) -> Dict[str, Dict[str, str]]:
    output: Dict[str, Dict[str, str]] = {}
    resolved_seen: set[Path] = set()
    for label, raw_path in sorted(paths.items()):
        path = Path(raw_path).resolve()
        _require(path not in resolved_seen, f"duplicate input path registered as {label}: {path}")
        resolved_seen.add(path)
        output[label] = {"path": str(path), "sha256": _sha256(path)}
    return output


def _revalidate_provenance(provenance: Mapping[str, Mapping[str, str]]) -> None:
    for label, artifact in provenance.items():
        path = Path(str(artifact["path"]))
        _require(_sha256(path) == artifact["sha256"], f"input changed during export: {label}")


def _manifest_output_hashes(root: Path, names: Sequence[str]) -> Dict[str, str]:
    return {name: _sha256(root / name) for name in names}


def export_evidence_bundle(
    *,
    validation_dir: Path,
    branch_dir: Path,
    trace_root: Path,
    output_root: Path,
    protocol_path: Path,
    lock_path: Path,
) -> Path:
    validation_dir = Path(validation_dir).resolve()
    branch_dir = Path(branch_dir).resolve()
    trace_root = Path(trace_root).resolve()
    output_root = Path(output_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    lock_path = Path(lock_path).resolve()
    _require(validation_dir.is_dir(), f"validation directory not found: {validation_dir}")
    _require(branch_dir.is_dir(), f"branch directory not found: {branch_dir}")
    _require(trace_root.is_dir(), f"trace root not found: {trace_root}")
    _require(not output_root.exists(), f"refusing to replace existing evidence bundle: {output_root}")

    validation_manifest_path = validation_dir / VALIDATION_MANIFEST_FILE
    validation_summary_path = validation_dir / VALIDATION_SUMMARY_FILE
    validation_by_seed_path = validation_dir / VALIDATION_BY_SEED_FILE
    branch_manifest_path = branch_dir / BRANCH_MANIFEST_FILE
    branch_labels_path = branch_dir / LABELS_FILE
    branch_rows_path = branch_dir / BRANCH_ROWS_FILE

    base_spec = load_spec(lock_path)
    validate_protocol_contract(protocol_path, base_spec)
    spec = locked_partition_spec(base_spec, "validation")
    _require(spec.seeds == CONFIRMATORY_SEEDS, "locked validation seed contract drift")
    protocol = _load_protocol(protocol_path)
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    registry = dict((submission.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {})
    _require("release_rollout" in registry and "component_ablation" in registry, "v12 evidence registry is incomplete")

    validation_manifest = _load_json(validation_manifest_path)
    validate_validation_manifest(
        validation_manifest,
        protocol_sha256=_sha256(protocol_path),
        lock_sha256=_sha256(lock_path),
    )
    threshold = _threshold_from_manifest(validation_manifest, spec)
    source = dict(validation_manifest["source"])

    raw_branch_manifest = _load_json(branch_manifest_path)
    _validate_branch_manifest(
        raw_branch_manifest,
        trace_root=trace_root,
        validation_source=source,
    )
    branch_manifest = load_branch_manifest(
        branch_manifest_path,
        labels_path=branch_labels_path,
        spec=spec,
        expected_cohort="confirmatory_holdout",
    )
    output_hashes = dict(branch_manifest.get("output_hashes", {}) or {})
    _require(
        output_hashes.get(BRANCH_ROWS_FILE) == _sha256(branch_rows_path),
        "branch row hash drift",
    )

    traces, trace_meta = load_features(trace_root, spec)
    _require(trace_meta["semantic_trace_hash"] == source["trace_semantic_hash"], "trace semantic hash drift")
    _require(trace_meta["raw_file_set_hash"] == source["trace_raw_file_set_hash"], "trace raw file-set hash drift")
    table = build_opportunity_table(traces, spec)
    cohorts = locked_factorial_cohorts(table, threshold, spec)
    required_keys = [
        table.rows[index].event_key
        for arm in VALIDATION_ARMS
        # The locked branch-label table includes scheduled boundary events;
        # only evaluated events are eligible for outcome summaries below.
        for index in cohorts[arm.label].scheduled
    ]
    labels = load_branch_labels(
        branch_labels_path,
        required_keys=sorted(set(required_keys)),
        spec=spec,
        branch_manifest=branch_manifest,
    )
    _require(labels.raw_sha256 == source["branch_label_raw_sha256"], "validation branch-label raw hash drift")
    _require(labels.semantic_hash == source["branch_label_semantic_hash"], "validation branch-label semantic hash drift")

    metrics = {
        arm.label: arm_metrics(table, cohorts[arm.label], labels, spec)
        for arm in VALIDATION_ARMS
    }
    variation = component_variation(table, spec)
    counts = _locked_counts_by_seed(table, cohorts, labels, spec)
    bootstrap = locked_bootstrap(counts, spec)
    geometry = locked_geometry(table, threshold, cohorts, spec)
    exposure = _locked_exposure_compliance(metrics[ARM_FULL], spec, "validation")
    acceptance = locked_acceptance(
        metrics,
        bootstrap,
        geometry,
        exposure,
        partition="validation",
        spec=spec,
    )
    _require(acceptance.get("passed") is True, "recomputed confirmatory acceptance failed")
    for field, expected in (
        ("component_variation", variation),
        ("geometry", geometry),
        ("metrics", _locked_metrics_payload(metrics)),
        ("bootstrap", bootstrap),
        ("acceptance", acceptance),
        ("paper_acceptance", acceptance),
    ):
        _require(
            _canonical_sha256(validation_manifest.get(field)) == _canonical_sha256(expected),
            f"validation manifest {field} differs from locked derivation",
        )

    locked_summary = locked_summary_rows(table, cohorts, labels, spec)
    locked_by_seed = locked_by_seed_delay_rows(table, cohorts, labels, spec)
    _assert_csv_exact(validation_summary_path, locked_summary, "validation summary")
    _assert_csv_exact(validation_by_seed_path, locked_by_seed, "validation per-seed")

    _, raw_label_rows = _read_csv(branch_labels_path)
    raw_labels: Dict[Tuple[int, int, int, int], Mapping[str, Any]] = {}
    for row in raw_label_rows:
        key = (
            int(row["seed"]),
            int(row["delay_steps"]),
            int(row["query_frame"]),
            int(row["release_frame"]),
        )
        _require(key not in raw_labels, f"duplicate branch label event {key}")
        raw_labels[key] = row
    _require(set(raw_labels) == set(labels.labels), "raw/verified branch label key drift")

    _, raw_branches = _read_csv(branch_rows_path)
    required_branch_fields = {
        "seed",
        "release_frame",
        "release_state_id",
        "method_version",
        "exact_action_provenance",
        "branch_role",
        "raw_action",
        "utility",
    }
    _require(required_branch_fields <= set(raw_branches[0]), "branch row schema is incomplete")
    for index, row in enumerate(raw_branches):
        _require(int(row["seed"]) in CONFIRMATORY_SEEDS, f"branch row {index}: seed drift")
        _require(row["method_version"] == METHOD_VERSION, f"branch row {index}: method drift")
        _require(str(row["exact_action_provenance"]) == "1", f"branch row {index}: provenance drift")

    branch_groups = _validate_branch_outcome_derivation(
        raw_branches,
        raw_labels,
        epsilon=spec.epsilon,
    )
    component_events = _event_rows(table, cohorts, labels, raw_labels, threshold)
    component_summary = _component_summary_rows(locked_summary)
    component_by_seed = _component_by_seed_rows(locked_by_seed)
    component_effects = _component_main_effect_rows(metrics, bootstrap)
    release_events = [
        {"allocator": "RGD", **{key: value for key, value in row.items() if key != "arm"}}
        for row in component_events
        if row["arm"] == ARM_FULL
    ]
    release_keys = {(int(row["seed"]), int(row["release_frame"])) for row in release_events}
    _require(release_keys <= set(branch_groups), "full RGD release events lack branch rows")
    for key in release_keys:
        roles = [str(row["branch_role"]) for row in branch_groups[key]]
        _require(roles.count("matched_fast") == 1 and "candidate" in roles, f"release {key}: invalid branch group")
    release_branches = [
        {"allocator": "RGD", **row}
        for row in raw_branches
        if (int(row["seed"]), int(row["release_frame"])) in release_keys
    ]
    release_summary = _release_summary_rows(locked_summary, spec)

    input_paths: Dict[str, Path] = {
        "protocol": protocol_path,
        "calibration_lock": lock_path,
        "validation_manifest": validation_manifest_path,
        "validation_summary": validation_summary_path,
        "validation_by_seed_delay": validation_by_seed_path,
        "branch_manifest": branch_manifest_path,
        "branch_labels": branch_labels_path,
        "branch_rows": branch_rows_path,
    }
    for seed in CONFIRMATORY_SEEDS:
        input_paths[f"trace_reasoning_seed_{seed}"] = _trace_path(trace_root, seed)
    provenance = _build_provenance(input_paths)
    exporter_path = Path(__file__).resolve()
    exporter = {"path": str(exporter_path), "sha256": _sha256(exporter_path)}
    provenance_digest = _canonical_sha256({"inputs": provenance, "exporter": exporter})

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{output_root.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    _require(not staging.exists(), f"staging path collision: {staging}")
    staging.mkdir()
    try:
        _atomic_write_csv(staging / RELEASE_SUMMARY_FILE, release_summary)
        _atomic_write_csv(staging / RELEASE_EVENTS_FILE, release_events)
        _atomic_write_csv(staging / RELEASE_BRANCHES_FILE, release_branches)
        _atomic_write_csv(staging / COMPONENT_SUMMARY_FILE, component_summary)
        _atomic_write_csv(staging / COMPONENT_EVENTS_FILE, component_events)
        _atomic_write_csv(staging / COMPONENT_BY_SEED_FILE, component_by_seed)
        _atomic_write_csv(staging / COMPONENT_MAIN_EFFECTS_FILE, component_effects)

        protocol_component = dict(submission.get("component_ablation", {}) or {})
        registry_component = dict(registry["component_ablation"] or {})
        registry_release = dict(registry["release_rollout"] or {})
        paper_acceptance = {
            **acceptance,
            "metric_passed": True,
            "provenance_passed": True,
            "passed": True,
        }
        common = {
            "status": "current",
            "publication_status": "current",
            "passed": True,
            "paper_facing_passed": True,
            "partition": "validation",
            "method_version": METHOD_VERSION,
            "seeds": list(CONFIRMATORY_SEEDS),
            "parameter_search_performed": False,
            "locked_thresholds": {"candidate_id": threshold.candidate_id, **threshold.as_floats()},
            "validation_analysis_digest": validation_manifest["analysis_digest"],
            "seed_is_experimental_unit": True,
            "horizon_steps": spec.horizon_steps,
            "gamma": spec.gamma,
            "epsilon": spec.epsilon,
            "alternative_metric_source": "relative_support_weighted_maneuver_family_breadth",
            "headroom_metric_source": "incumbent_relative_action_recovery_cost_margin",
            "absolute_alternative_feasibility_non_ablatable": True,
            "viable_cost_threshold": 0.55,
            "source_provenance_sha256": provenance_digest,
            "input_artifacts": provenance,
            "exporter": exporter,
        }
        release_manifest: Dict[str, Any] = {
            "schema": RELEASE_SCHEMA,
            "artifact_role": "paper_facing_release_rollout_evidence",
            **common,
            "ttc_delay_threshold": float(submission["ttc_delay_threshold"]),
            "paper_acceptance": paper_acceptance,
            "summary": release_summary,
            "output_hashes": _manifest_output_hashes(
                staging,
                (RELEASE_SUMMARY_FILE, RELEASE_EVENTS_FILE, RELEASE_BRANCHES_FILE),
            ),
        }
        component_manifest: Dict[str, Any] = {
            "schema": COMPONENT_SCHEMA,
            "artifact_role": "paper_facing_component_ablation_evidence",
            **common,
            "design": "full factorial with leave-one-out arms listed first",
            "legal_action_provenance": "exact",
            "query_events_nested_within_seed": True,
            "need_metric_source": "state_hazard_and_pre_screen_only",
            "delay_s": list(spec.delay_seconds),
            "latency_survival_floor": "calibration_locked",
            "maneuver_breadth_floor": "calibration_locked",
            "corrective_headroom_floor": "calibration_locked",
            "state_need_floor": spec.i_floor_units / 100.0,
            "budget": spec.budget,
            "cooldown_frames": spec.cooldown_complete_frames,
            "cooldown_minimum_query_frame_gap": spec.minimum_query_frame_gap,
            "bootstrap_draws": int(bootstrap["draws"]),
            "bootstrap_seed": int(bootstrap["bootstrap_seed"]),
            "arms": [arm.__dict__ for arm in VALIDATION_ARMS],
            "paper_acceptance": paper_acceptance,
            "validation_acceptance": acceptance,
            "summary": component_summary,
            "component_contrasts": component_effects,
            "output_hashes": _manifest_output_hashes(
                staging,
                (
                    COMPONENT_SUMMARY_FILE,
                    COMPONENT_EVENTS_FILE,
                    COMPONENT_BY_SEED_FILE,
                    COMPONENT_MAIN_EFFECTS_FILE,
                ),
            ),
        }
        _require(protocol_component.get("budget") == spec.budget, "component budget protocol drift")
        _validate_registry_values(release_manifest, registry_release, "release_rollout")
        _validate_registry_values(component_manifest, registry_component, "component_ablation")
        _atomic_write_json(staging / RELEASE_MANIFEST_FILE, release_manifest)
        _atomic_write_json(staging / COMPONENT_MANIFEST_FILE, component_manifest)

        published_names = (
            RELEASE_MANIFEST_FILE,
            RELEASE_SUMMARY_FILE,
            RELEASE_EVENTS_FILE,
            RELEASE_BRANCHES_FILE,
            COMPONENT_MANIFEST_FILE,
            COMPONENT_SUMMARY_FILE,
            COMPONENT_EVENTS_FILE,
            COMPONENT_BY_SEED_FILE,
            COMPONENT_MAIN_EFFECTS_FILE,
        )
        bundle_manifest: Dict[str, Any] = {
            "schema": BUNDLE_SCHEMA,
            "artifact_role": "paper_evidence_registry_adapter",
            "status": "current",
            "passed": True,
            "paper_facing_passed": True,
            "partition": "validation",
            "method_version": METHOD_VERSION,
            "seeds": list(CONFIRMATORY_SEEDS),
            "parameter_search_performed": False,
            "validation_analysis_digest": validation_manifest["analysis_digest"],
            "source_provenance_sha256": provenance_digest,
            "input_artifacts": provenance,
            "exporter": exporter,
            "output_artifacts": _manifest_output_hashes(staging, published_names),
        }
        bundle_manifest["bundle_payload_sha256"] = _payload_sha256(
            bundle_manifest, "bundle_payload_sha256"
        )
        _atomic_write_json(staging / BUNDLE_MANIFEST_FILE, bundle_manifest)
        _revalidate_provenance(provenance)
        _require(_sha256(exporter_path) == exporter["sha256"], "exporter changed during publication")
        _require(not output_root.exists(), f"evidence output appeared during export: {output_root}")
        os.replace(str(staging), str(output_root))
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output_root


def verify_published_bundle(
    root: Path,
    *,
    protocol_path: Optional[Path] = None,
) -> None:
    """Verify immutable publication hashes and registry-facing contracts."""

    root = Path(root).resolve()
    bundle = _load_json(root / BUNDLE_MANIFEST_FILE)
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "evidence bundle schema drift")
    _require(bundle.get("status") == "current", "evidence bundle is not current")
    _require(bundle.get("passed") is True, "evidence bundle is not passed")
    _require(bundle.get("paper_facing_passed") is True, "evidence bundle is not paper-facing passed")
    _require(bundle.get("partition") == "validation", "evidence bundle partition drift")
    _require(bundle.get("method_version") == METHOD_VERSION, "evidence bundle method drift")
    _require(bundle.get("seeds") == list(CONFIRMATORY_SEEDS), "evidence bundle seed drift")
    _require(bundle.get("parameter_search_performed") is False, "evidence bundle searched parameters")
    _require(
        bundle.get("bundle_payload_sha256")
        == _payload_sha256(bundle, "bundle_payload_sha256"),
        "evidence bundle payload hash drift",
    )
    provenance = dict(bundle.get("input_artifacts", {}) or {})
    _require(bool(provenance), "evidence bundle omits input hashes")
    _require(
        bundle.get("source_provenance_sha256")
        == _canonical_sha256({"inputs": provenance, "exporter": bundle.get("exporter")}),
        "evidence source-provenance digest drift",
    )
    _revalidate_provenance(provenance)
    exporter = dict(bundle.get("exporter", {}) or {})
    _require(_sha256(Path(exporter.get("path", ""))) == exporter.get("sha256"), "exporter hash drift")
    for name, expected in dict(bundle.get("output_artifacts", {}) or {}).items():
        _require(_sha256(root / name) == expected, f"published output hash drift: {name}")

    release = _load_json(root / RELEASE_MANIFEST_FILE)
    component = _load_json(root / COMPONENT_MANIFEST_FILE)
    _require(release.get("schema") == RELEASE_SCHEMA, "release evidence schema drift")
    _require(component.get("schema") == COMPONENT_SCHEMA, "component evidence schema drift")
    for label, manifest in (("release", release), ("component", component)):
        _require(manifest.get("status") == "current", f"{label} evidence is not current")
        _require(manifest.get("passed") is True, f"{label} evidence is not passed")
        _require(manifest.get("paper_facing_passed") is True, f"{label} paper acceptance failed")
        _require(manifest.get("partition") == "validation", f"{label} partition drift")
        _require(manifest.get("method_version") == METHOD_VERSION, f"{label} method drift")
        _require(manifest.get("seeds") == list(CONFIRMATORY_SEEDS), f"{label} seed drift")
        _require(manifest.get("parameter_search_performed") is False, f"{label} searched parameters")
        acceptance = dict(manifest.get("paper_acceptance", {}) or {})
        _require(acceptance.get("metric_passed") is True, f"{label} metric acceptance failed")
        _require(acceptance.get("passed") is True, f"{label} acceptance failed")
        _require(
            manifest.get("source_provenance_sha256") == bundle.get("source_provenance_sha256"),
            f"{label} provenance binding drift",
        )
        for name, expected in dict(manifest.get("output_hashes", {}) or {}).items():
            _require(_sha256(root / name) == expected, f"{label} output hash drift: {name}")

    _assert_csv_exact(root / RELEASE_SUMMARY_FILE, release["summary"], "release summary")
    _assert_csv_exact(root / COMPONENT_SUMMARY_FILE, component["summary"], "component summary")
    _assert_csv_exact(
        root / COMPONENT_MAIN_EFFECTS_FILE,
        component["component_contrasts"],
        "component contrasts",
    )
    if protocol_path is not None:
        protocol = _load_protocol(Path(protocol_path).resolve())
        artifacts = dict(
            ((protocol.get("tvt_submission_contract", {}) or {}).get("evidence_artifacts", {}) or {}).get(
                "artifacts", {}
            )
            or {}
        )
        _validate_registry_values(release, dict(artifacts["release_rollout"]), "release_rollout")
        _validate_registry_values(component, dict(artifacts["component_ablation"]), "component_ablation")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--validation-dir", type=Path, required=True)
    export.add_argument("--branch-dir", type=Path, required=True)
    export.add_argument("--trace-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--protocol", type=Path, default=REPO_ROOT / "formal_protocol.yaml")
    export.add_argument("--lock", type=Path, default=LOCK_PATH)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--protocol", type=Path, default=REPO_ROOT / "formal_protocol.yaml")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "export":
        output = export_evidence_bundle(
            validation_dir=args.validation_dir,
            branch_dir=args.branch_dir,
            trace_root=args.trace_root,
            output_root=args.output_root,
            protocol_path=args.protocol,
            lock_path=args.lock,
        )
        print(json.dumps({"status": "current", "passed": True, "artifact": str(output)}))
        return 0
    verify_published_bundle(args.artifact, protocol_path=args.protocol)
    print("PASS: v12 evidence bundle hashes, acceptance, seed partition, and registry schema verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
