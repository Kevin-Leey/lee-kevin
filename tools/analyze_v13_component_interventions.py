"""Analyze the current v13 six-arm component-ablation replay.

Release-event rollouts are descriptive within each arm and are matched to the
exact saved release snapshot. Episode contrasts are paired by simulator seed.
The two estimands are deliberately kept separate in every exported artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    COMPONENT_ABLATION_ARMS,
    FACTORIAL_EVENT_SCHEMA,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
)
from dilu.evaluation.release_snapshot import (  # noqa: E402
    RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
)
from tools.analyze_factorial_interventions import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_HORIZON,
    EVENT_ROW_FIELDS,
    _process_cell,
    _read_csv,
    _sha256_file,
    _write_csv,
    require,
    summarize_events,
)
from tools.analyze_query_release_factorial import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    seed_bootstrap_indices,
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


ANALYSIS_SCHEMA = "rgd_v13_component_ablation_analysis_v2"
REQUEST_AUDIT_SCHEMA = "rgd_v13_component_ablation_request_audit_v1"

MANIFEST_FILE = "component_ablation_analysis_manifest.json"
AUDIT_FILE = "component_ablation_request_audit.json"
SUMMARY_FILE = "component_ablation_summary.csv"
EVENTS_FILE = "component_ablation_events.csv"
BY_SEED_FILE = "component_ablation_by_seed.csv"
MAIN_EFFECTS_FILE = "component_ablation_main_effects.csv"
OUTPUT_FILES = (SUMMARY_FILE, EVENTS_FILE, BY_SEED_FILE, MAIN_EFFECTS_FILE)

CURRENT_RELEASE_SELECTION_STAGE = "post_release_guard_pre_final_safety_projection"
CURRENT_FINAL_ACTUATOR_STAGE = "post_shared_actuator_bridge_pre_environment_step"
CLAIM_SCOPE = (
    "release-event summaries are descriptive within arm; component effects are "
    "paired simulator-seed episode contrasts"
)

EPISODE_METRICS = (
    "collision",
    "route_completion",
    "episode_reward",
    "driving_distance",
    "avg_speed",
)
PAIRWISE_EFFECTS = (
    (
        "full_minus_without_l",
        "full - without_l",
        ("full", "without_l"),
        (1.0, -1.0),
    ),
    (
        "full_minus_without_a",
        "full - without_a",
        ("full", "without_a"),
        (1.0, -1.0),
    ),
    (
        "full_minus_without_h",
        "full - without_h",
        ("full", "without_h"),
        (1.0, -1.0),
    ),
    (
        "full_minus_without_n",
        "full - without_n",
        ("full", "without_n"),
        (1.0, -1.0),
    ),
    (
        "h_x_n_interaction",
        "full - without_h - without_n + without_h_and_n",
        ("full", "without_h", "without_n", "without_h_and_n"),
        (1.0, -1.0, -1.0, 1.0),
    ),
)

BY_SEED_FIELDS = (
    "arm",
    "seed",
    *EPISODE_METRICS,
    "candidate_queries",
    "issued_queries",
    "query_gate_rejections",
    "release_events",
    "timeouts",
    "failure_events",
    "pending_at_episode_end",
    "candidate_evaluable_releases",
    "first_step_distinct_candidates",
    "executed_first_step_interventions",
    "beneficial_candidates",
    "harmful_candidates",
    "neutral_candidates",
    "release_guard_rejections",
    "missed_beneficial_candidates",
    "mean_utility_delta_evaluable",
    "mean_utility_delta_executed",
)

_TERMINAL_FLAGS = (
    "closed_loop_latency_release_event",
    "closed_loop_latency_timeout_event",
    "closed_loop_latency_failure_event",
)
_VALID_OUTCOMES = frozenset({"valid", "timeout", "failure"})
_COMPONENT_GATE_FIELDS = (
    ("latency_survival", "latency_survival_pass"),
    ("maneuver_breadth", "maneuver_breadth_pass"),
    ("corrective_headroom", "corrective_headroom_pass"),
    ("state_need", "state_need_pass"),
)
_NON_ABLATABLE_GATE_FIELDS = (
    "domain_contract_pass",
    "executor_available_pass",
    "latency_prediction_pass",
    "absolute_feasibility_pass",
)


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=reject_constant,
        object_pairs_hook=_unique_json_object,
    )
    require(isinstance(payload, Mapping), f"expected a JSON object: {path}")
    return dict(payload)


def _event_json(path: Path) -> Dict[str, Any]:
    """Read runner event logs, which retain NaN for pending timing fields."""

    require(path.is_file(), f"missing event log: {path}")
    payload = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=_unique_json_object,
    )
    require(isinstance(payload, Mapping), f"event log is not an object: {path}")
    return dict(payload)


def _write_strict_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    if value in (0, 1, 0.0, 1.0):
        return bool(value)
    raise ValueError(f"{field} must be boolean")


def _finite_metric(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"component-ablation row has invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"component-ablation row has non-finite {field}")
    return value


def _finite_or_blank(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _strict_rows(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [
        {str(key): _finite_or_blank(value) for key, value in dict(row).items()}
        for row in rows
    ]


def _valid_sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    require(
        re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"invalid {field}: {value!r}",
    )
    return digest


def _single(paths: Sequence[Path] | Any, *, context: str) -> Path:
    values = list(paths)
    require(len(values) == 1, f"{context}: expected one artifact, found {len(values)}")
    return values[0]


def _seed_block(contract: Mapping[str, Any]) -> tuple[int, ...]:
    block = contract.get("seed_range")
    require(isinstance(block, Mapping), "component contract omits seed_range")
    start = _integer(block.get("start"), "seed_range.start", nonnegative=True)
    end = _integer(block.get("end"), "seed_range.end", nonnegative=True)
    count = _integer(block.get("count"), "seed_range.count", nonnegative=True)
    require(end >= start and count == end - start + 1, "component seed range drift")
    return tuple(range(start, end + 1))


def _proposal_records(
    manifest: Mapping[str, Any], *, seeds: Sequence[int]
) -> Dict[int, Dict[str, Dict[str, Any]]]:
    require(
        manifest.get("schema") == FACTORIAL_PROPOSAL_SCHEMA,
        "component proposal manifest schema drift",
    )
    require(
        manifest.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
        "component proposal replay version drift",
    )
    payload = manifest.get("bank_payload")
    require(isinstance(payload, list), "component proposal bank payload is missing")
    require(
        _canonical_sha256(payload) == manifest.get("bank_sha256"),
        "component proposal bank payload hash drift",
    )
    records_by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    observed_seeds: list[int] = []
    global_request_ids: set[str] = set()
    proposal_count = 0
    for raw_block in payload:
        require(isinstance(raw_block, Mapping), "component proposal seed block is invalid")
        seed = _integer(raw_block.get("seed"), "proposal seed", nonnegative=True)
        require(seed not in records_by_seed, f"duplicate proposal seed {seed}")
        observed_seeds.append(seed)
        records: Dict[str, Dict[str, Any]] = {}
        frames: set[int] = set()
        for raw_record in list(raw_block.get("records", []) or []):
            require(isinstance(raw_record, Mapping), f"seed {seed}: invalid proposal record")
            record = dict(raw_record)
            require(
                _integer(record.get("seed"), "proposal record seed", nonnegative=True)
                == seed,
                f"seed {seed}: proposal record seed drift",
            )
            request_id = str(record.get("request_id", "") or "")
            frame = _integer(
                record.get("source_frame"), "proposal source frame", nonnegative=True
            )
            require(request_id and request_id not in records, f"seed {seed}: duplicate request ID")
            require(
                request_id not in global_request_ids,
                f"proposal request ID is reused across seeds: {request_id}",
            )
            require(frame not in frames, f"seed {seed}: duplicate proposal frame")
            outcome = str(record.get("outcome", "") or "")
            require(outcome in _VALID_OUTCOMES, f"seed {seed}: invalid proposal outcome")
            _integer(record.get("latency_steps"), "proposal latency", nonnegative=True)
            action = _integer(record.get("raw_slow_action"), "proposal action", nonnegative=True)
            require(action in range(5), f"seed {seed}: invalid proposal action")
            response_sha256 = str(record.get("response_sha256", "") or "")
            if response_sha256:
                _valid_sha256(response_sha256, "proposal response_sha256")
            records[request_id] = record
            frames.add(frame)
            global_request_ids.add(request_id)
            proposal_count += 1
        require(bool(records), f"seed {seed}: empty proposal block")
        records_by_seed[seed] = records
    require(tuple(observed_seeds) == tuple(seeds), "proposal seed cohort/order drift")
    require(
        _integer(manifest.get("seed_count"), "proposal seed_count", nonnegative=True)
        == len(seeds),
        "proposal seed-count drift",
    )
    require(
        _integer(manifest.get("proposal_count"), "proposal_count", nonnegative=True)
        == proposal_count,
        "proposal-count drift",
    )
    return records_by_seed


def _load_contract(bundle: Path) -> Dict[str, Any]:
    bundle = Path(bundle).resolve()
    require(bundle.is_dir(), f"component-ablation bundle does not exist: {bundle}")
    run_path = bundle / "component_ablation_run_manifest.json"
    result_path = bundle / "component_ablation_episode_results.csv"
    proposal_path = bundle / "proposal_bank_manifest.json"
    run_manifest = _strict_json(run_path)
    proposal_manifest = _strict_json(proposal_path)
    rows = _read_csv(result_path)

    require(
        run_manifest.get("schema") == COMPONENT_ABLATION_RUN_SCHEMA,
        "component-ablation run schema drift",
    )
    require(
        run_manifest.get("component_ablation_version") == COMPONENT_ABLATION_VERSION,
        "component-ablation version drift",
    )
    require(
        run_manifest.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
        "component-ablation replay version drift",
    )
    expected_arms = tuple(spec.name for spec in COMPONENT_ABLATION_ARMS)
    manifest_arms = list(run_manifest.get("arms", []) or [])
    expected_manifest_arms = [
        {**asdict(spec), "removed_components": list(spec.removed_components)}
        for spec in COMPONENT_ABLATION_ARMS
    ]
    require(
        manifest_arms == expected_manifest_arms,
        "component-ablation arm contract drift",
    )
    seed_start = _integer(run_manifest.get("seed_start"), "seed_start", nonnegative=True)
    seed_count = _integer(run_manifest.get("seed_count"), "seed_count", nonnegative=True)
    require(seed_count > 0, "component-ablation seed cohort is empty")
    seeds = tuple(range(seed_start, seed_start + seed_count))

    require(
        len(rows) == len(seeds) * len(expected_arms),
        "component-ablation episode result count drift",
    )
    matrix = {(int(row["seed"]), str(row["arm"])): dict(row) for row in rows}
    expected_cells = {(seed, arm) for seed in seeds for arm in expected_arms}
    require(set(matrix) == expected_cells, "component-ablation episode matrix drift")

    proposal_hash = _valid_sha256(
        run_manifest.get("proposal_bank_sha256"), "run proposal_bank_sha256"
    )
    require(
        proposal_hash
        == _valid_sha256(proposal_manifest.get("bank_sha256"), "proposal bank_sha256"),
        "run/proposal bank hash drift",
    )
    proposal_records = _proposal_records(proposal_manifest, seeds=seeds)
    for (seed, arm), row in matrix.items():
        spec = next(value for value in COMPONENT_ABLATION_ARMS if value.name == arm)
        require(
            row.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
            f"{arm}/{seed}: result replay-version drift",
        )
        require(
            _boolean(row.get("query_gate_enabled"), f"{arm}/{seed} query_gate_enabled"),
            f"{arm}/{seed}: query gate is not enabled",
        )
        require(
            _boolean(row.get("release_guard_enabled"), f"{arm}/{seed} release_guard_enabled"),
            f"{arm}/{seed}: release guard is not enabled",
        )
        require(
            str(row.get("proposal_bank_sha256", "") or "") == proposal_hash,
            f"{arm}/{seed}: proposal-bank binding drift",
        )
        require(
            str(row.get("component_ablation_arm", "") or "") == arm,
            f"{arm}/{seed}: arm label drift",
        )
        require(
            str(row.get("component_ablation_display_name", "") or "")
            == spec.display_name,
            f"{arm}/{seed}: arm display-name drift",
        )
        require(
            str(row.get("component_ablation_removed_components", "") or "")
            == ";".join(spec.removed_components),
            f"{arm}/{seed}: removed-component label drift",
        )
        for metric in EPISODE_METRICS:
            _finite_metric(row, metric)
        collision = _finite_metric(row, "collision")
        require(collision in (0.0, 1.0), f"{arm}/{seed}: collision must be binary")
        route_completion = _finite_metric(row, "route_completion")
        require(
            0.0 <= route_completion <= 1.0,
            f"{arm}/{seed}: route completion must lie in [0, 1]",
        )
        counts = {
            field: _integer(row.get(field), f"{arm}/{seed} {field}", nonnegative=True)
            for field in (
                "candidate_queries",
                "issued_queries",
                "query_gate_rejections",
                "release_events",
                "timeouts",
                "failure_events",
                "pending_at_episode_end",
                "snapshot_count",
            )
        }
        require(
            counts["candidate_queries"]
            == counts["issued_queries"] + counts["query_gate_rejections"],
            f"{arm}/{seed}: query accounting drift",
        )
        require(
            counts["issued_queries"]
            == counts["release_events"]
            + counts["timeouts"]
            + counts["failure_events"]
            + counts["pending_at_episode_end"],
            f"{arm}/{seed}: request-outcome accounting drift",
        )
        require(
            counts["snapshot_count"] == counts["release_events"],
            f"{arm}/{seed}: result snapshot/release drift",
        )

    protocol_path = Path(str(run_manifest.get("protocol_path", "") or "")).resolve()
    require(protocol_path.is_file(), "component run protocol is unavailable")
    require(
        _sha256_file(protocol_path) == run_manifest.get("protocol_sha256"),
        "component run protocol hash drift",
    )
    protocol = load_formal_protocol(protocol_path)
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    contract = dict(submission.get("component_ablation", {}) or {})
    calibration_lock = dict(
        dict(submission.get("v12_calibration", {}) or {}).get(
            "calibration_lock", {}
        )
        or {}
    )
    contract["support_breadth_trace_formula"] = calibration_lock.get(
        "support_breadth_trace_formula"
    )
    registry = dict(
        ((submission.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {}).get(
            "component_ablation", {}
        )
        or {}
    )
    requirements = dict(registry.get("required_manifest_values", {}) or {})
    require(contract.get("design") == COMPONENT_ABLATION_DESIGN, "protocol design drift")
    require(
        tuple(str(value) for value in contract.get("arms", ())) == expected_arms,
        "protocol six-arm order drift",
    )
    require(_seed_block(contract) == seeds, "protocol component seed cohort drift")
    submission_version_fields = {
        "method_version": "rgd_method_version",
        "query_gate_method_version": "query_gate_method_version",
        "release_contract_version": "release_contract_version",
    }
    for field, submission_field in submission_version_fields.items():
        expected = requirements.get(field, submission.get(submission_field))
        require(bool(expected), f"protocol omits {field}")
        require(run_manifest.get(field) == expected, f"run {field} drift")
    require(run_manifest.get("design") == contract.get("design"), "run design drift")
    require(
        run_manifest.get("latency_profile") == contract.get("latency_profile"),
        "run latency profile drift",
    )
    require(
        run_manifest.get("fixed_latency_steps") == contract.get("fixed_delay_steps"),
        "run fixed latency drift",
    )
    require(
        math.isclose(
            float(run_manifest.get("predicted_latency_s")),
            float(contract.get("predicted_latency_s")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "run predicted latency drift",
    )
    require(
        run_manifest.get("delay_s")
        == [float(value) for value in list(contract.get("delay_s", []) or [])],
        "run delay-stratum drift",
    )
    require(
        run_manifest.get("candidate_source_policy")
        == contract.get("candidate_source_policy"),
        "run candidate-source policy drift",
    )
    require(
        run_manifest.get("candidate_source_gate_independent") is True,
        "run proposal source is not gate-independent",
    )
    require(
        proposal_manifest.get("candidate_source_policy")
        == contract.get("candidate_source_policy"),
        "proposal candidate-source policy drift",
    )
    require(
        proposal_manifest.get("candidate_source_gate_independent") is True,
        "proposal source is not gate-independent",
    )
    require(
        proposal_manifest.get("latency_profile") == contract.get("latency_profile"),
        "proposal latency-profile drift",
    )
    fixed_latency_steps = _integer(
        contract.get("fixed_delay_steps"), "component fixed_delay_steps", nonnegative=True
    )
    require(
        _integer(
            proposal_manifest.get("fixed_latency_steps"),
            "proposal fixed_latency_steps",
            nonnegative=True,
        )
        == fixed_latency_steps,
        "proposal fixed-latency drift",
    )
    for seed, records in proposal_records.items():
        for request_id, record in records.items():
            require(
                _integer(record.get("latency_steps"), "proposal latency", nonnegative=True)
                == fixed_latency_steps,
                f"seed {seed}/{request_id}: proposal latency drifts from fixed profile",
            )
    execution = resolve_policy_execution_horizon(
        dict(contract.get("execution_contract", {}) or {}),
        context="component analysis execution contract",
    ).as_manifest()
    for key, expected in execution.items():
        require(run_manifest.get(key) == expected, f"run execution horizon drift at {key}")
        for (seed, arm), row in matrix.items():
            require(
                math.isclose(
                    _finite_metric(row, key),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{arm}/{seed}: execution horizon drift at {key}",
            )
    require(
        _integer(run_manifest.get("result_rows"), "run result_rows", nonnegative=True)
        == len(rows),
        "run result-row count drift",
    )

    return {
        "bundle": bundle,
        "run_path": run_path,
        "result_path": result_path,
        "proposal_path": proposal_path,
        "run_manifest": run_manifest,
        "proposal_manifest": proposal_manifest,
        "proposal_records": proposal_records,
        "proposal_bank_sha256": proposal_hash,
        "episode_rows": rows,
        "matrix": matrix,
        "seeds": seeds,
        "arms": expected_arms,
        "protocol_path": protocol_path,
        "protocol": protocol,
        "component_contract": contract,
        "registry_spec": registry,
        "required_manifest_values": requirements,
        "execution_horizon": execution,
    }


def _resolve_parameters(
    context: Mapping[str, Any],
    *,
    horizon: Optional[int],
    gamma: Optional[float],
    epsilon: Optional[float],
    draws: Optional[int],
    bootstrap_seed: Optional[int],
) -> Dict[str, Any]:
    contract = dict(context["component_contract"])
    resolved = {
        "horizon": int(contract.get("horizon_steps", DEFAULT_HORIZON))
        if horizon is None
        else int(horizon),
        "gamma": float(contract.get("gamma", DEFAULT_GAMMA))
        if gamma is None
        else float(gamma),
        "epsilon": float(contract.get("corrective_margin", DEFAULT_EPSILON))
        if epsilon is None
        else float(epsilon),
        "draws": int(contract.get("bootstrap_draws", DEFAULT_BOOTSTRAP_DRAWS))
        if draws is None
        else int(draws),
        "bootstrap_seed": int(contract.get("bootstrap_seed", 0))
        if bootstrap_seed is None
        else int(bootstrap_seed),
    }
    require(resolved["horizon"] > 0, "intervention horizon must be positive")
    require(0.0 < resolved["gamma"] <= 1.0, "intervention gamma must lie in (0, 1]")
    require(resolved["epsilon"] >= 0.0, "intervention epsilon must be nonnegative")
    require(resolved["draws"] > 0, "bootstrap draws must be positive")
    for name, contract_key in (
        ("horizon", "horizon_steps"),
        ("gamma", "gamma"),
        ("epsilon", "corrective_margin"),
        ("draws", "bootstrap_draws"),
        ("bootstrap_seed", "bootstrap_seed"),
    ):
        expected = contract.get(contract_key)
        require(expected is not None, f"component contract omits {contract_key}")
        if isinstance(resolved[name], float):
            require(
                math.isclose(float(resolved[name]), float(expected), rel_tol=0.0, abs_tol=1e-12),
                f"analysis parameter {name} drifts from protocol",
            )
        else:
            require(int(resolved[name]) == int(expected), f"analysis parameter {name} drifts from protocol")
    return resolved


def _cell_paths(context: Mapping[str, Any], arm: str, seed: int) -> Dict[str, Path]:
    seed_dir = Path(context["bundle"]) / arm / f"seed_{seed}"
    require(seed_dir.is_dir(), f"missing component cell: {arm}/{seed}")
    event_path = _single(
        sorted((seed_dir / "event_logs").glob("event_log_*.json")),
        context=f"{arm}/{seed} event log",
    )
    require(
        set((seed_dir / "event_logs").iterdir()) == {event_path},
        f"{arm}/{seed}: unexpected event-log artifacts",
    )
    snapshot_path = seed_dir / "experiment_snapshot.json"
    require(snapshot_path.is_file(), f"{arm}/{seed}: missing experiment snapshot")
    result = {"event_log": event_path, "experiment_snapshot": snapshot_path}
    expected_releases = _integer(
        context["matrix"][(seed, arm)].get("release_events"),
        f"{arm}/{seed} release_events",
        nonnegative=True,
    )
    snapshot_root = seed_dir / "release_snapshots"
    snapshot_manifests = sorted(snapshot_root.glob("release_snapshots_*.json"))
    if expected_releases:
        snapshot_manifest_path = _single(
            snapshot_manifests, context=f"{arm}/{seed} snapshot manifest"
        )
        snapshot_manifest = _strict_json(snapshot_manifest_path)
        require(
            snapshot_manifest.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
            f"{arm}/{seed}: snapshot manifest schema drift",
        )
        require(
            _integer(
                snapshot_manifest.get("snapshot_count"),
                f"{arm}/{seed} snapshot_count",
                nonnegative=True,
            )
            == expected_releases,
            f"{arm}/{seed}: snapshot/result release-count drift",
        )
        bundle_name = str(snapshot_manifest.get("bundle_file", "") or "")
        require(
            bool(bundle_name)
            and not Path(bundle_name).is_absolute()
            and len(Path(bundle_name).parts) == 1,
            f"{arm}/{seed}: invalid snapshot bundle name",
        )
        snapshot_bundle_path = snapshot_root / bundle_name
        require(snapshot_bundle_path.is_file(), f"{arm}/{seed}: missing snapshot bundle")
        require(
            _sha256_file(snapshot_bundle_path) == snapshot_manifest.get("bundle_sha256"),
            f"{arm}/{seed}: snapshot bundle hash drift",
        )
        require(
            set(snapshot_root.iterdir()) == {snapshot_manifest_path, snapshot_bundle_path},
            f"{arm}/{seed}: unexpected snapshot artifacts",
        )
        result["snapshot_manifest"] = snapshot_manifest_path
        result["snapshot_bundle"] = snapshot_bundle_path
    else:
        snapshot_files = list(snapshot_root.iterdir()) if snapshot_root.is_dir() else []
        require(not snapshot_files, f"{arm}/{seed}: snapshots without release events")
    return result


def _input_inventory(context: Mapping[str, Any]) -> Dict[str, str]:
    bundle = Path(context["bundle"])
    paths = [
        Path(context["run_path"]),
        Path(context["result_path"]),
        Path(context["proposal_path"]),
    ]
    for arm in context["arms"]:
        for seed in context["seeds"]:
            paths.extend(_cell_paths(context, arm, int(seed)).values())
    inventory: Dict[str, str] = {}
    for path in sorted(set(paths), key=lambda item: item.relative_to(bundle).as_posix()):
        relative = path.relative_to(bundle).as_posix()
        inventory[relative] = _sha256_file(path)
    return inventory


def _snapshot_records(
    context: Mapping[str, Any], arm: str, seed: int
) -> Dict[str, str]:
    paths = _cell_paths(context, arm, seed)
    manifest_path = paths.get("snapshot_manifest")
    if manifest_path is None:
        return {}
    manifest = _strict_json(manifest_path)
    rows = list(manifest.get("snapshots", []) or [])
    records: Dict[str, str] = {}
    for raw_row in rows:
        require(isinstance(raw_row, Mapping), f"{arm}/{seed}: invalid snapshot row")
        row = dict(raw_row)
        request_id = str(row.get("request_id", "") or "")
        require(request_id and request_id not in records, f"{arm}/{seed}: duplicate snapshot IDs")
        identity = _valid_sha256(
            row.get("snapshot_identity_sha256"), "snapshot_identity_sha256"
        )
        records[request_id] = identity
    require(
        len(records)
        == _integer(manifest.get("snapshot_count"), "snapshot_count", nonnegative=True),
        f"{arm}/{seed}: snapshot count drift",
    )
    return records


def _event_flag(event: Mapping[str, Any], field: str, *, context: str) -> bool:
    value = event.get(field)
    require(isinstance(value, bool), f"{context}: {field} must be boolean")
    return bool(value)


def _validate_component_candidate(
    event: Mapping[str, Any],
    *,
    arm: str,
    protocol_contract: Mapping[str, Any],
    required_manifest_values: Mapping[str, Any],
    context: str,
) -> bool:
    gate = event.get("recoverability_gate")
    require(isinstance(gate, Mapping), f"{context}: recoverability gate is missing")
    spec = next(value for value in COMPONENT_ABLATION_ARMS if value.name == arm)
    non_ablatable_pass = all(
        _event_flag(gate, field, context=context)
        for field in _NON_ABLATABLE_GATE_FIELDS
    )
    component_passes = {
        name: _event_flag(gate, field, context=context)
        for name, field in _COMPONENT_GATE_FIELDS
    }
    retained = {
        name for name, _ in _COMPONENT_GATE_FIELDS if name not in spec.removed_components
    }
    admit = bool(
        non_ablatable_pass and all(component_passes[name] for name in retained)
    )
    serial_gate_pass = _event_flag(gate, "serial_gate_pass", context=context)
    string_contract = {
        "method_version": required_manifest_values.get("query_gate_method_version"),
        "gate_composition": "explicit_serial_floors",
        "gate_action_universe_source": "driving_state.effective_action_universe",
        "fast_executor_action_universe_source": "driving_state.effective_action_universe",
        "alternative_metric_source": protocol_contract.get("alternative_metric_source"),
        "headroom_metric_source": protocol_contract.get("headroom_metric_source"),
        "need_metric_source": protocol_contract.get("need_metric_source"),
        "support_breadth_formula": protocol_contract.get(
            "support_breadth_trace_formula"
        ),
        "corrective_headroom_kappa_source": "identifiable_gate_v12.fixed_kappa",
    }
    for field, expected in string_contract.items():
        require(bool(expected), f"component protocol omits {field}")
        require(
            str(gate.get(field, "") or "") == str(expected),
            f"{context}: recoverability-gate {field} drift",
        )
    numeric_contract = {
        "viable_cost_threshold": protocol_contract.get("viable_cost_threshold"),
        "support_breadth_temperature": protocol_contract.get(
            "support_breadth_temperature"
        ),
        "corrective_headroom_kappa": protocol_contract.get(
            "corrective_headroom_kappa"
        ),
        "latency_survival_floor": protocol_contract.get("latency_survival_floor"),
        "maneuver_breadth_floor": protocol_contract.get("maneuver_breadth_floor"),
        "corrective_headroom_floor": protocol_contract.get(
            "corrective_headroom_floor"
        ),
        "state_need_floor": protocol_contract.get("state_need_floor"),
        "effective_delay_steps": protocol_contract.get("fixed_delay_steps"),
        "policy_frequency": protocol_contract.get("latency_policy_frequency_hz"),
        "safety_reserve_seconds": protocol_contract.get(
            "latency_safety_reserve_s"
        ),
    }
    for field, expected in numeric_contract.items():
        require(expected is not None, f"component protocol omits {field}")
        require(
            math.isclose(
                _finite_metric(gate, field),
                float(expected),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{context}: recoverability-gate {field} drift",
        )
    require(
        _event_flag(
            gate,
            "absolute_alternative_feasibility_non_ablatable",
            context=context,
        )
        is bool(protocol_contract.get("absolute_alternative_feasibility_non_ablatable")),
        f"{context}: absolute-feasibility contract drift",
    )
    require(
        str(event.get("component_ablation_policy", "") or "")
        == "serial_predicate_removal",
        f"{context}: component-ablation policy drift",
    )
    require(
        event.get("component_ablation_arm") == arm,
        f"{context}: component-ablation arm audit drift",
    )
    require(
        str(event.get("component_ablation_removed_components", "") or "")
        == ";".join(spec.removed_components),
        f"{context}: removed-component audit drift",
    )
    require(
        _event_flag(event, "component_ablation_non_ablatable_pass", context=context)
        == non_ablatable_pass,
        f"{context}: non-ablatable predicate audit drift",
    )
    require(
        _event_flag(
            event,
            "component_ablation_full_equivalent_to_serial_gate",
            context=context,
        )
        == (admit == serial_gate_pass),
        f"{context}: full-gate equivalence audit drift",
    )
    for name, _ in _COMPONENT_GATE_FIELDS:
        require(
            _event_flag(event, f"component_ablation_{name}_pass", context=context)
            == component_passes[name],
            f"{context}: {name} predicate audit drift",
        )
        require(
            _event_flag(
                event, f"component_ablation_{name}_retained", context=context
            )
            == (name in retained),
            f"{context}: {name} retention audit drift",
        )
    return admit


def _audit_requests(context: Mapping[str, Any], inventory: Mapping[str, str]) -> Dict[str, Any]:
    cells: list[Dict[str, Any]] = []
    aggregate = {
        "event_count": 0,
        "candidate_queries": 0,
        "issued_queries": 0,
        "query_gate_rejections": 0,
        "release_events": 0,
        "timeouts": 0,
        "failure_events": 0,
        "pending_at_episode_end": 0,
        "dropped_at_episode_end": 0,
        "snapshot_count": 0,
    }
    for arm in context["arms"]:
        for seed in context["seeds"]:
            seed = int(seed)
            row = dict(context["matrix"][(seed, arm)])
            paths = _cell_paths(context, arm, seed)
            event_path = paths["event_log"]
            payload = _event_json(event_path)
            require(
                payload.get("schema_version") == FACTORIAL_EVENT_SCHEMA,
                f"{arm}/{seed}: event schema drift",
            )
            require(
                _integer(payload.get("episode_id"), "event episode_id", nonnegative=True)
                == seed,
                f"{arm}/{seed}: event episode identity drift",
            )
            events = list(payload.get("events", []) or [])
            require(bool(events), f"{arm}/{seed}: event log is empty")
            require(
                len(events) == _integer(payload.get("event_count"), "event_count", nonnegative=True),
                f"{arm}/{seed}: event count drift",
            )
            proposal_records = context["proposal_records"][seed]
            candidates: set[str] = set()
            issued: Dict[str, str] = {}
            issued_frames: Dict[str, int] = {}
            terminals: Dict[str, str] = {}
            terminal_kinds: Dict[str, str] = {}
            release_snapshot_identities: Dict[str, str] = {}
            for index, raw_event in enumerate(events):
                require(isinstance(raw_event, Mapping), f"{arm}/{seed}/{index}: invalid event")
                event = dict(raw_event)
                frame = _integer(event.get("frame"), "event frame", nonnegative=True)
                event_context = f"{arm}/{seed}/{frame}"
                require(frame == index, f"{arm}/{seed}: non-contiguous event frames")
                require(event.get("factorial_arm") == arm, f"{event_context}: arm drift")
                require(
                    event.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
                    f"{event_context}: replay-version drift",
                )
                require(
                    event.get("factorial_proposal_bank_sha256")
                    == context["proposal_bank_sha256"],
                    f"{event_context}: proposal-bank binding drift",
                )
                require(
                    _event_flag(
                        event, "factorial_query_gate_enabled", context=event_context
                    ),
                    f"{event_context}: query gate is not enabled",
                )
                require(
                    _event_flag(
                        event, "factorial_release_guard_enabled", context=event_context
                    ),
                    f"{event_context}: release guard is not enabled",
                )
                candidate = _event_flag(
                    event, "factorial_candidate_query", context=event_context
                )
                candidate_issued = _event_flag(
                    event, "factorial_query_issued", context=event_context
                )
                issuance = _event_flag(
                    event, "closed_loop_latency_issuance_event", context=event_context
                )
                if candidate:
                    request_id = str(event.get("factorial_candidate_request_id", "") or "")
                    require(request_id in proposal_records, f"{event_context}: unknown candidate")
                    require(request_id not in candidates, f"{event_context}: duplicate candidate")
                    proposal = proposal_records[request_id]
                    require(
                        _integer(proposal.get("source_frame"), "proposal source frame") == frame,
                        f"{event_context}: candidate source-frame drift",
                    )
                    require(
                        _integer(
                            event.get("factorial_shared_raw_slow_action"),
                            "factorial_shared_raw_slow_action",
                            nonnegative=True,
                        )
                        == _integer(
                            proposal.get("raw_slow_action"),
                            "proposal raw_slow_action",
                            nonnegative=True,
                        ),
                        f"{event_context}: candidate action drift",
                    )
                    require(
                        _integer(
                            event.get("factorial_shared_latency_steps"),
                            "factorial_shared_latency_steps",
                            nonnegative=True,
                        )
                        == _integer(
                            proposal.get("latency_steps"),
                            "proposal latency_steps",
                            nonnegative=True,
                        ),
                        f"{event_context}: candidate latency drift",
                    )
                    require(
                        str(event.get("factorial_shared_response_sha256", "") or "")
                        == str(proposal.get("response_sha256", "") or ""),
                        f"{event_context}: candidate response hash drift",
                    )
                    require(
                        str(event.get("factorial_shared_response_outcome", "") or "")
                        == str(proposal.get("outcome", "") or ""),
                        f"{event_context}: candidate outcome drift",
                    )
                    candidates.add(request_id)
                    gate_passed = _event_flag(
                        event, "factorial_query_gate_pass", context=event_context
                    )
                    require(
                        gate_passed
                        == _validate_component_candidate(
                            event,
                            arm=arm,
                            protocol_contract=context["component_contract"],
                            required_manifest_values=context[
                                "required_manifest_values"
                            ],
                            context=event_context,
                        ),
                        f"{event_context}: component policy decision drift",
                    )
                    require(
                        candidate_issued == issuance == gate_passed,
                        f"{event_context}: issuance/gate flag drift",
                    )
                    if candidate_issued:
                        issued_id = str(event.get("closed_loop_latency_issued_request_id", "") or "")
                        outcome = str(event.get("closed_loop_latency_issued_response_outcome", "") or "")
                        require(issued_id == request_id, f"{event_context}: issued ID drift")
                        require(outcome == proposal.get("outcome"), f"{event_context}: issued outcome drift")
                        require(request_id not in issued, f"{event_context}: duplicate issuance")
                        issued[request_id] = outcome
                        issued_frames[request_id] = frame
                    else:
                        require(
                            not str(event.get("closed_loop_latency_issued_request_id", "") or "")
                            and not str(
                                event.get("closed_loop_latency_issued_response_outcome", "")
                                or ""
                            ),
                            f"{event_context}: rejected candidate retains issuance identity",
                        )
                else:
                    require(
                        not str(event.get("factorial_candidate_request_id", "") or ""),
                        f"{event_context}: noncandidate retains candidate identity",
                    )
                    require(
                        not candidate_issued and not issuance,
                        f"{event_context}: orphan issuance",
                    )
                    require(
                        not str(event.get("closed_loop_latency_issued_request_id", "") or "")
                        and not str(
                            event.get("closed_loop_latency_issued_response_outcome", "")
                            or ""
                        ),
                        f"{event_context}: nonissuance retains issued identity",
                    )

                terminal = _event_flag(
                    event, "closed_loop_latency_terminal_event", context=event_context
                )
                flags = [
                    _event_flag(event, field, context=event_context)
                    for field in _TERMINAL_FLAGS
                ]
                require(sum(flags) <= 1, f"{event_context}: conflicting terminal flags")
                if terminal:
                    request_id = str(event.get("closed_loop_latency_terminal_request_id", "") or "")
                    outcome = str(event.get("closed_loop_latency_terminal_response_outcome", "") or "")
                    require(request_id in issued, f"{event_context}: orphan terminal")
                    require(request_id not in terminals, f"{event_context}: duplicate terminal")
                    require(outcome == issued[request_id], f"{event_context}: terminal outcome drift")
                    proposal = proposal_records[request_id]
                    source_frame = _integer(
                        proposal.get("source_frame"), "proposal source frame", nonnegative=True
                    )
                    latency_steps = _integer(
                        proposal.get("latency_steps"), "proposal latency", nonnegative=True
                    )
                    require(
                        frame == source_frame + latency_steps,
                        f"{event_context}: terminal timing drift",
                    )
                    require(
                        str(event.get("closed_loop_latency_request_id", "") or "")
                        == request_id,
                        f"{event_context}: terminal request alias drift",
                    )
                    require(
                        _integer(
                            event.get("closed_loop_latency_source_frame"),
                            "terminal source frame",
                            nonnegative=True,
                        )
                        == source_frame,
                        f"{event_context}: terminal source-frame drift",
                    )
                    require(
                        _integer(
                            event.get("closed_loop_latency_delay_steps"),
                            "terminal delay steps",
                            nonnegative=True,
                        )
                        == latency_steps,
                        f"{event_context}: terminal delay drift",
                    )
                    require(
                        _integer(
                            event.get("closed_loop_latency_scheduled_release_frame"),
                            "terminal scheduled release frame",
                            nonnegative=True,
                        )
                        == frame,
                        f"{event_context}: scheduled release-frame drift",
                    )
                    if outcome == "valid":
                        require(flags == [True, False, False], f"{event_context}: valid terminal flag drift")
                        selection_distinct = _event_flag(
                            event, "release_selection_distinct", context=event_context
                        )
                        require(
                            str(event.get("closed_loop_latency_terminal_outcome", "") or "")
                            == (
                                "distinct_actuation"
                                if selection_distinct
                                else "fast_equivalent"
                            ),
                            f"{event_context}: valid terminal outcome drift",
                        )
                        require(
                            _event_flag(
                                event,
                                "closed_loop_release_snapshot_captured",
                                context=event_context,
                            ),
                            f"{event_context}: release snapshot was not captured",
                        )
                        release_snapshot_identities[request_id] = _valid_sha256(
                            event.get("closed_loop_release_snapshot_identity_sha256"),
                            "release snapshot identity",
                        )
                        kind = "release"
                    elif outcome == "timeout":
                        require(flags == [False, True, False], f"{event_context}: timeout flag drift")
                        require(
                            str(event.get("closed_loop_latency_terminal_outcome", "") or "")
                            == outcome,
                            f"{event_context}: timeout terminal outcome drift",
                        )
                        require(
                            not _event_flag(
                                event,
                                "closed_loop_release_snapshot_captured",
                                context=event_context,
                            ),
                            f"{event_context}: timeout carries a release snapshot",
                        )
                        kind = "timeout"
                    else:
                        require(flags == [False, False, True], f"{event_context}: failure flag drift")
                        require(
                            str(event.get("closed_loop_latency_terminal_outcome", "") or "")
                            == outcome,
                            f"{event_context}: failure terminal outcome drift",
                        )
                        require(
                            not _event_flag(
                                event,
                                "closed_loop_release_snapshot_captured",
                                context=event_context,
                            ),
                            f"{event_context}: failure carries a release snapshot",
                        )
                        kind = "failure"
                    terminals[request_id] = outcome
                    terminal_kinds[request_id] = kind
                else:
                    require(not any(flags), f"{event_context}: terminal flag without terminal event")
                    require(
                        not _event_flag(
                            event,
                            "closed_loop_release_snapshot_captured",
                            context=event_context,
                        ),
                        f"{event_context}: nonterminal carries a release snapshot",
                    )
                    require(
                        not str(event.get("closed_loop_latency_terminal_request_id", "") or "")
                        and not str(
                            event.get("closed_loop_latency_terminal_response_outcome", "")
                            or ""
                        ),
                        f"{event_context}: nonterminal retains terminal identity",
                    )
                    if issuance:
                        require(
                            str(event.get("closed_loop_latency_request_id", "") or "")
                            == str(event.get("closed_loop_latency_issued_request_id", "") or ""),
                            f"{event_context}: issuance request alias drift",
                        )

            final_frame = len(events) - 1
            reachable = {
                request_id
                for request_id, proposal in proposal_records.items()
                if _integer(proposal.get("source_frame"), "proposal source frame") <= final_frame
            }
            require(candidates == reachable, f"{arm}/{seed}: reachable candidate coverage drift")
            pending_rows = list(payload.get("pending_releases_dropped_at_episode_end", []) or [])
            require(
                len(pending_rows)
                == _integer(payload.get("pending_release_count"), "pending_release_count", nonnegative=True),
                f"{arm}/{seed}: pending count drift",
            )
            pending: Dict[str, str] = {}
            for raw_pending in pending_rows:
                require(isinstance(raw_pending, Mapping), f"{arm}/{seed}: invalid pending row")
                pending_row = dict(raw_pending)
                request_id = str(pending_row.get("request_id", "") or "")
                outcome = str(pending_row.get("response_outcome", "") or "")
                require(request_id in issued, f"{arm}/{seed}: orphan pending request")
                require(request_id not in pending, f"{arm}/{seed}: duplicate pending request")
                require(outcome == issued[request_id], f"{arm}/{seed}: pending outcome drift")
                proposal = proposal_records[request_id]
                source_frame = _integer(
                    proposal.get("source_frame"), "proposal source frame", nonnegative=True
                )
                release_frame = source_frame + _integer(
                    proposal.get("latency_steps"), "proposal latency", nonnegative=True
                )
                require(
                    _integer(
                        pending_row.get("source_frame"),
                        "pending source frame",
                        nonnegative=True,
                    )
                    == source_frame,
                    f"{arm}/{seed}: pending source-frame drift",
                )
                require(
                    _integer(
                        pending_row.get("release_frame"),
                        "pending release frame",
                        nonnegative=True,
                    )
                    == release_frame
                    and release_frame > final_frame,
                    f"{arm}/{seed}: pending release-frame drift",
                )
                require(
                    pending_row.get("terminal_outcome", "dropped_at_episode_end")
                    == "dropped_at_episode_end",
                    f"{arm}/{seed}: pending terminal marker drift",
                )
                pending[request_id] = outcome
            require(not set(terminals) & set(pending), f"{arm}/{seed}: terminal/pending overlap")
            require(set(issued) == set(terminals) | set(pending), f"{arm}/{seed}: request lifecycle does not close")
            budget = _integer(
                context["component_contract"].get("budget"),
                "component budget",
                nonnegative=True,
            )
            cooldown_gap = _integer(
                context["component_contract"].get("cooldown_minimum_query_frame_gap"),
                "component cooldown gap",
                nonnegative=True,
            )
            ordered_issued_frames = sorted(issued_frames.values())
            require(
                len(ordered_issued_frames) <= budget,
                f"{arm}/{seed}: issued-query budget drift",
            )
            require(
                all(
                    right - left >= cooldown_gap
                    for left, right in zip(ordered_issued_frames, ordered_issued_frames[1:])
                ),
                f"{arm}/{seed}: issued-query cooldown drift",
            )

            release_ids = {request_id for request_id, kind in terminal_kinds.items() if kind == "release"}
            timeout_ids = {request_id for request_id, kind in terminal_kinds.items() if kind == "timeout"}
            failure_ids = {request_id for request_id, kind in terminal_kinds.items() if kind == "failure"}
            snapshot_records = _snapshot_records(context, arm, seed)
            snapshot_ids = set(snapshot_records)
            require(snapshot_ids == release_ids, f"{arm}/{seed}: release/snapshot identity drift")
            require(
                snapshot_records == release_snapshot_identities,
                f"{arm}/{seed}: release/snapshot digest drift",
            )
            require(
                _integer(
                    payload.get("release_snapshot_count"),
                    "release_snapshot_count",
                    nonnegative=True,
                )
                == len(snapshot_ids),
                f"{arm}/{seed}: event snapshot-count drift",
            )
            if snapshot_ids:
                seed_dir = Path(context["bundle"]) / arm / f"seed_{seed}"
                expected_manifest = paths["snapshot_manifest"].relative_to(seed_dir).as_posix()
                expected_bundle = paths["snapshot_bundle"].relative_to(seed_dir).as_posix()
                require(
                    payload.get("release_snapshot_manifest") == expected_manifest,
                    f"{arm}/{seed}: event snapshot-manifest reference drift",
                )
                require(
                    payload.get("release_snapshot_bundle") == expected_bundle,
                    f"{arm}/{seed}: event snapshot-bundle reference drift",
                )
                require(
                    payload.get("release_snapshot_bundle_sha256")
                    == _sha256_file(paths["snapshot_bundle"]),
                    f"{arm}/{seed}: event snapshot-bundle hash drift",
                )
            else:
                for field in (
                    "release_snapshot_manifest",
                    "release_snapshot_bundle",
                    "release_snapshot_bundle_sha256",
                ):
                    require(
                        payload.get(field) in (None, ""),
                        f"{arm}/{seed}: empty snapshot set retains {field}",
                    )
            counts = {
                "candidate_queries": len(candidates),
                "issued_queries": len(issued),
                "query_gate_rejections": len(candidates) - len(issued),
                "release_events": len(release_ids),
                "timeouts": len(timeout_ids),
                "failure_events": len(failure_ids),
                "pending_at_episode_end": len(pending),
                "snapshot_count": len(snapshot_ids),
            }
            for field, observed in counts.items():
                require(
                    _integer(row.get(field), f"{arm}/{seed} {field}", nonnegative=True)
                    == observed,
                    f"{arm}/{seed}: result/event {field} drift",
                )
            cell = {
                "arm": arm,
                "seed": seed,
                "accepted": True,
                "event_count": len(events),
                "reachable_proposal_count": len(reachable),
                "right_censored_proposal_count": len(proposal_records) - len(reachable),
                "issued_source_frames": ordered_issued_frames,
                "budget": budget,
                "cooldown_minimum_query_frame_gap": cooldown_gap,
                **counts,
                "dropped_request_ids": sorted(pending),
                "event_log_sha256": _sha256_file(event_path),
            }
            cells.append(cell)
            aggregate["event_count"] += len(events)
            for field in counts:
                if field in aggregate:
                    aggregate[field] += int(counts[field])
            aggregate["dropped_at_episode_end"] += len(pending)

    return {
        "schema": REQUEST_AUDIT_SCHEMA,
        "accepted": True,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "proposal_bank_sha256": context["proposal_bank_sha256"],
        "input_inventory_sha256": _canonical_sha256(dict(inventory)),
        "audit_contract": {
            "all_reachable_candidates_bound_to_bank": True,
            "all_request_ids_lifecycle_closed": True,
            "episode_end_pending_recorded_as_dropped": True,
            "release_iff_authenticated_snapshot": True,
            "seed_is_experimental_unit": True,
            "budget_and_cooldown_bound_to_issuance": True,
        },
        "aggregate": aggregate,
        "cells": cells,
        "errors": [],
    }


def _current_runtime_release_action(event: Mapping[str, Any], *, context: str) -> Dict[str, int]:
    require(bool(event.get("closed_loop_latency_release_event", False)), f"{context}: not a release event")
    require(
        event.get("release_action_comparison_stage") == CURRENT_RELEASE_SELECTION_STAGE,
        f"{context}: release selection stage drift",
    )
    require(
        event.get("final_actuator_action_stage") == CURRENT_FINAL_ACTUATOR_STAGE,
        f"{context}: final actuator stage drift",
    )
    fast = _integer(event.get("release_fast_comparator_action"), f"{context} fast action")
    selected = _integer(event.get("release_selected_action"), f"{context} selected action")
    final = _integer(event.get("final_actuator_action"), f"{context} final action")
    executed = _integer(
        event.get("closed_loop_latency_executed_action"), f"{context} executed action"
    )
    event_final = _integer(event.get("final_action"), f"{context} event final action")
    released = _integer(
        event.get("closed_loop_released_slow_action"), f"{context} released slow action"
    )
    require(
        all(
            action in range(5)
            for action in (fast, selected, final, executed, event_final, released)
        ),
        f"{context}: invalid action",
    )
    require(final == executed == event_final, f"{context}: final/executed action drift")
    rejected = bool(event.get("closed_loop_release_opportunity_rejected", False))
    unavailable = bool(event.get("closed_loop_release_action_unavailable", False))
    raw_legal_actions = event.get("_runtime_available_actions", event.get("legal_actions"))
    require(
        isinstance(raw_legal_actions, Sequence)
        and not isinstance(raw_legal_actions, (str, bytes)),
        f"{context}: release legal-action set is missing",
    )
    legal_actions = {
        _integer(value, f"{context} legal action") for value in raw_legal_actions
    }
    require(bool(legal_actions), f"{context}: release legal-action set is empty")
    require(
        (released not in legal_actions) == unavailable,
        f"{context}: release unavailability/legal-action drift",
    )
    if rejected or unavailable:
        require(selected == fast, f"{context}: rejected release did not retain Fast")
    else:
        require(selected == released, f"{context}: selected/released slow action drift")
    distinct = selected != fast
    require(
        _boolean(event.get("release_selection_distinct"), f"{context} release_selection_distinct")
        == distinct,
        f"{context}: release selection distinctness drift",
    )
    return {
        "fast_action": fast,
        "selected_action": selected,
        "final_action": final,
        "selection_distinct": int(distinct),
    }


def _process_component_cell(task: Mapping[str, Any]) -> list[Dict[str, Any]]:
    legacy_task = dict(task)
    # The shared analyzer's modern contract expects an unavailable same-stage
    # post-safety Fast comparator. Current v13 logs instead expose a pre-safety
    # selection and a post-bridge final action, so use its rollout-only path and
    # authenticate those two stages explicitly below.
    legacy_task["legacy_v2"] = True
    rows = [dict(row) for row in _process_cell(legacy_task)]
    seed_dir = Path(str(task["seed_dir"]))
    event_path = _single(
        sorted((seed_dir / "event_logs").glob("event_log_*.json")),
        context=f"{task['arm']}/{task['seed']} event log",
    )
    payload = _event_json(event_path)
    releases = {
        str(dict(event).get("closed_loop_latency_request_id", "") or ""): dict(event)
        for event in list(payload.get("events", []) or [])
        if isinstance(event, Mapping)
        and bool(dict(event).get("closed_loop_latency_release_event", False))
    }
    require(len(releases) == len(rows), f"{task['arm']}/{task['seed']}: release row drift")
    for row in rows:
        request_id = str(row["request_id"])
        require(request_id in releases, f"{task['arm']}/{task['seed']}: missing release event")
        action = _current_runtime_release_action(
            releases[request_id], context=f"{task['arm']}/{task['seed']}/{request_id}"
        )
        row["release_selected_action"] = action["selected_action"]
        row["selection_stage_primitive_distinct"] = action["selection_distinct"]
        row["final_actuator_action"] = action["final_action"]
        row["executed_action"] = action["final_action"]
        require(tuple(row) == EVENT_ROW_FIELDS, "component intervention event schema drift")
    return rows


def _collect_event_rows(
    context: Mapping[str, Any], parameters: Mapping[str, Any], *, workers: int
) -> list[Dict[str, Any]]:
    require(workers > 0, "worker count must be positive")
    tasks = [
        {
            "arm": arm,
            "seed": int(seed),
            "seed_dir": str((Path(context["bundle"]) / arm / f"seed_{seed}").resolve()),
            "expected_releases": _integer(
                context["matrix"][(int(seed), arm)].get("release_events"),
                f"{arm}/{seed} release_events",
                nonnegative=True,
            ),
            "horizon": int(parameters["horizon"]),
            "gamma": float(parameters["gamma"]),
            "epsilon": float(parameters["epsilon"]),
        }
        for seed in context["seeds"]
        for arm in context["arms"]
        if _integer(
            context["matrix"][(int(seed), arm)].get("release_events"),
            f"{arm}/{seed} release_events",
            nonnegative=True,
        )
        > 0
    ]
    rows: list[Dict[str, Any]] = []
    if workers == 1:
        for task in tasks:
            rows.extend(_process_component_cell(task))
    elif tasks:
        with ProcessPoolExecutor(max_workers=min(int(workers), len(tasks))) as pool:
            futures = [pool.submit(_process_component_cell, task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())
    rows.sort(
        key=lambda row: (
            context["arms"].index(str(row["arm"])),
            int(row["seed"]),
            int(row["release_frame"]),
            str(row["request_id"]),
        )
    )
    return _strict_rows(rows)


def _episode_effects(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    arms: Sequence[str],
    draws: int,
    bootstrap_seed: int,
) -> list[Dict[str, Any]]:
    matrix = {(int(row["seed"]), str(row["arm"])): row for row in rows}
    require(
        set(matrix) == {(int(seed), arm) for seed in seeds for arm in arms},
        "component-ablation outcome matrix drift",
    )
    indices = seed_bootstrap_indices(
        len(seeds), draws=int(draws), bootstrap_seed=int(bootstrap_seed)
    )
    effects: list[Dict[str, Any]] = []
    for name, formula, effect_arms, coefficients in PAIRWISE_EFFECTS:
        require(set(effect_arms).issubset(set(arms)), f"component effect {name} has unknown arm")
        for metric in EPISODE_METRICS:
            values = np.asarray(
                [
                    sum(
                        coefficient * _finite_metric(matrix[(int(seed), arm)], metric)
                        for arm, coefficient in zip(effect_arms, coefficients)
                    )
                    for seed in seeds
                ],
                dtype=float,
            )
            samples = np.mean(values[indices], axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            effects.append(
                {
                    "effect": name,
                    "contrast_formula": formula,
                    "metric": metric,
                    "estimand": "paired_mean_per_simulator_seed",
                    "estimate": float(np.mean(values)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "n_seed_blocks": len(seeds),
                    "bootstrap_draws": int(draws),
                    "valid_bootstrap_draws": int(draws),
                }
            )
    return effects


def _row_flag(row: Mapping[str, Any], field: str) -> bool:
    return bool(_integer(row.get(field, 0), field, nonnegative=True))


def _mean_or_blank(rows: Sequence[Mapping[str, Any]], field: str) -> Any:
    values = [float(row[field]) for row in rows if row.get(field) not in (None, "")]
    return float(np.mean(values)) if values else ""


def _by_seed_rows(
    episode_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    arms: Sequence[str],
) -> list[Dict[str, Any]]:
    matrix = {(int(row["seed"]), str(row["arm"])): row for row in episode_rows}
    output: list[Dict[str, Any]] = []
    for arm in arms:
        for seed in seeds:
            episode = matrix[(int(seed), arm)]
            releases = [
                row
                for row in event_rows
                if str(row.get("arm")) == arm and int(row.get("seed", -1)) == int(seed)
            ]
            evaluable = [row for row in releases if _row_flag(row, "candidate_evaluable")]
            executed = [
                row
                for row in evaluable
                if _row_flag(row, "executed_first_step_actuator_distinct")
            ]
            first_distinct = [
                row for row in evaluable if _row_flag(row, "first_step_actuator_distinct")
            ]
            rejected = [row for row in releases if _row_flag(row, "release_guard_rejected")]
            beneficial = [row for row in evaluable if row.get("classification") == "beneficial"]
            harmful = [row for row in evaluable if row.get("classification") == "harmful"]
            neutral = [row for row in evaluable if row.get("classification") == "neutral"]
            missed = [
                row
                for row in beneficial
                if not _row_flag(row, "executed_first_step_actuator_distinct")
            ]
            result = {
                "arm": arm,
                "seed": int(seed),
                **{metric: _finite_metric(episode, metric) for metric in EPISODE_METRICS},
                **{
                    field: _integer(episode.get(field), f"{arm}/{seed} {field}", nonnegative=True)
                    for field in (
                        "candidate_queries",
                        "issued_queries",
                        "query_gate_rejections",
                        "release_events",
                        "timeouts",
                        "failure_events",
                        "pending_at_episode_end",
                    )
                },
                "candidate_evaluable_releases": len(evaluable),
                "first_step_distinct_candidates": len(first_distinct),
                "executed_first_step_interventions": len(executed),
                "beneficial_candidates": len(beneficial),
                "harmful_candidates": len(harmful),
                "neutral_candidates": len(neutral),
                "release_guard_rejections": len(rejected),
                "missed_beneficial_candidates": len(missed),
                "mean_utility_delta_evaluable": _mean_or_blank(evaluable, "utility_delta"),
                "mean_utility_delta_executed": _mean_or_blank(executed, "utility_delta"),
            }
            require(tuple(result) == BY_SEED_FIELDS, "component by-seed schema drift")
            output.append(result)
    return _strict_rows(output)


def compute_analysis(
    bundle: Path,
    *,
    horizon: Optional[int] = None,
    gamma: Optional[float] = None,
    epsilon: Optional[float] = None,
    draws: Optional[int] = None,
    bootstrap_seed: Optional[int] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Recompute every registry row and the request audit from a raw bundle."""

    context = _load_contract(Path(bundle))
    parameters = _resolve_parameters(
        context,
        horizon=horizon,
        gamma=gamma,
        epsilon=epsilon,
        draws=draws,
        bootstrap_seed=bootstrap_seed,
    )
    inventory = _input_inventory(context)
    audit = _audit_requests(context, inventory)
    event_rows = _collect_event_rows(context, parameters, workers=int(workers))
    summary_rows = _strict_rows(
        summarize_events(
            event_rows,
            seeds=context["seeds"],
            draws=int(parameters["draws"]),
            bootstrap_seed=int(parameters["bootstrap_seed"]),
            arms=context["arms"],
            allow_custom_arms=True,
        )
    )
    by_seed_rows = _by_seed_rows(
        context["episode_rows"],
        event_rows,
        seeds=context["seeds"],
        arms=context["arms"],
    )
    effects = _strict_rows(
        _episode_effects(
            context["episode_rows"],
            seeds=context["seeds"],
            arms=context["arms"],
            draws=int(parameters["draws"]),
            bootstrap_seed=int(parameters["bootstrap_seed"]),
        )
    )
    return {
        "context": context,
        "parameters": parameters,
        "input_sha256": inventory,
        "input_inventory_sha256": _canonical_sha256(inventory),
        "request_audit": audit,
        "request_audit_payload_sha256": _canonical_sha256(audit),
        "events": event_rows,
        "summary": summary_rows,
        "by_seed": by_seed_rows,
        "main_effects": effects,
    }


def _manifest(
    analysis: Mapping[str, Any],
    *,
    audit_sha256: str,
    output_sha256: Mapping[str, str],
) -> Dict[str, Any]:
    context = analysis["context"]
    parameters = analysis["parameters"]
    run_manifest = context["run_manifest"]
    requirements = dict(context["required_manifest_values"])
    manifest: Dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "artifact_role": "paper_facing_component_ablation_evidence",
        "status": "current",
        "accepted": True,
        "component_ablation_version": COMPONENT_ABLATION_VERSION,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "method_version": run_manifest["method_version"],
        "query_gate_method_version": run_manifest["query_gate_method_version"],
        "release_contract_version": run_manifest["release_contract_version"],
        "design": COMPONENT_ABLATION_DESIGN,
        "arms": list(context["arms"]),
        "seeds": list(context["seeds"]),
        "independent_unit": "simulator_seed",
        "seed_is_experimental_unit": True,
        "query_events_nested_within_seed": True,
        "claim_scope": CLAIM_SCOPE,
        "release_snapshot_stage": "pre_release_frame_policy_decision",
        "release_selection_stage": CURRENT_RELEASE_SELECTION_STAGE,
        "final_actuator_action_stage": CURRENT_FINAL_ACTUATOR_STAGE,
        "branch_design": "matched_release_state_first_action_then_shared_fast_continuation",
        "horizon_steps": int(parameters["horizon"]),
        "gamma": float(parameters["gamma"]),
        "epsilon": float(parameters["epsilon"]),
        "bootstrap_draws": int(parameters["draws"]),
        "bootstrap_seed": int(parameters["bootstrap_seed"]),
        "bootstrap": {
            "unit": "simulator_seed",
            "draws": int(parameters["draws"]),
            "seed": int(parameters["bootstrap_seed"]),
        },
        "event_count": len(analysis["events"]),
        "summary": list(analysis["summary"]),
        "component_contrasts": list(analysis["main_effects"]),
        "source_bundle": str(Path(context["bundle"]).resolve()),
        "protocol": str(Path(context["protocol_path"]).resolve()),
        "protocol_sha256": _sha256_file(context["protocol_path"]),
        "source_run_manifest_sha256": _sha256_file(context["run_path"]),
        "source_proposal_manifest_sha256": _sha256_file(context["proposal_path"]),
        "source_episode_results_sha256": _sha256_file(context["result_path"]),
        "proposal_bank_sha256": context["proposal_bank_sha256"],
        "input_sha256": dict(analysis["input_sha256"]),
        "input_inventory_sha256": analysis["input_inventory_sha256"],
        "request_audit": {
            "schema": REQUEST_AUDIT_SCHEMA,
            "file": AUDIT_FILE,
            "sha256": str(audit_sha256),
            "payload_sha256": analysis["request_audit_payload_sha256"],
        },
        "outputs": list(OUTPUT_FILES),
        "output_sha256": dict(output_sha256),
        "analysis_source_path": str(Path(__file__).resolve()),
        "analysis_source_sha256": _sha256_file(Path(__file__).resolve()),
    }
    for key, expected in requirements.items():
        observed = manifest.get(key)
        if observed is None:
            manifest[key] = expected
        else:
            require(observed == expected, f"registry requirement drift at {key}")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    require(args.workers > 0, "worker count must be positive")
    analysis = compute_analysis(
        Path(args.bundle).resolve(),
        horizon=args.horizon,
        gamma=args.gamma,
        epsilon=args.epsilon,
        draws=args.draws,
        bootstrap_seed=args.bootstrap_seed,
        workers=int(args.workers),
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        SUMMARY_FILE: output_dir / SUMMARY_FILE,
        EVENTS_FILE: output_dir / EVENTS_FILE,
        BY_SEED_FILE: output_dir / BY_SEED_FILE,
        MAIN_EFFECTS_FILE: output_dir / MAIN_EFFECTS_FILE,
    }
    _write_csv(paths[SUMMARY_FILE], analysis["summary"])
    _write_csv(paths[EVENTS_FILE], analysis["events"], fieldnames=EVENT_ROW_FIELDS)
    _write_csv(paths[BY_SEED_FILE], analysis["by_seed"], fieldnames=BY_SEED_FIELDS)
    _write_csv(paths[MAIN_EFFECTS_FILE], analysis["main_effects"])

    audit_path = output_dir / AUDIT_FILE
    _write_strict_json(audit_path, analysis["request_audit"])
    output_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    manifest = _manifest(
        analysis,
        audit_sha256=_sha256_file(audit_path),
        output_sha256=output_hashes,
    )
    manifest_path = output_dir / MANIFEST_FILE
    _write_strict_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "accepted": True,
                "events": len(analysis["events"]),
                "manifest": str(manifest_path),
            },
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
