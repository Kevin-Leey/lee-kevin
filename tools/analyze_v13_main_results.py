"""Validate and analyze the formal v13 six-arm closed-loop main table.

The simulator seed is the experimental unit.  Every contrast is computed
within seed, binary endpoints use exact McNemar tests, and continuous
endpoints use paired seed bootstrap intervals with sign-flip tests.  The
validator rebuilds request lifecycle counts from event logs instead of trusting
the summary CSV emitted by the runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PROTOCOL_NAME = "rgd_tvt_action_aligned_release_v13"
METHOD_VERSION = "action_aligned_release_gate_v13"
QUERY_GATE_VERSION = "identifiable_gate_v12"
RELEASE_CONTRACT_VERSION = "action_cost_alignment_v2"
ANALYSIS_VERSION = "rgd_v13_main_paired_analysis_v1"
REFERENCE_GROUP = "rgd_fixed_policy"
FORMAL_GROUPS = (
    "rgd_fixed_policy",
    "always_fast",
    "always_slow",
    "random_budget",
    "uncertainty_budget",
    "risk_budget",
)
EXPECTED_ENVIRONMENT = "highway-v0"

ENDPOINTS = (
    "success_rate",
    "collision_rate",
    "route_completion",
    "episode_reward",
    "driving_distance_m",
    "safety_qualified_speed_mps",
    "all_frame_speed_mps",
    "algorithm_runtime_ms_per_frame",
    "request_count",
    "valid_response_count",
    "release_count",
    "authorized_release_count",
    "distinct_actuation_count",
)
BINARY_ENDPOINTS = frozenset({"success_rate", "collision_rate"})
LOWER_IS_BETTER = frozenset(
    {"collision_rate", "algorithm_runtime_ms_per_frame", "request_count"}
)
INTEGER_ENDPOINTS = frozenset(
    {
        "request_count",
        "valid_response_count",
        "release_count",
        "authorized_release_count",
        "distinct_actuation_count",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {path}: {value}")

    payload = json.loads(
        path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant
    )
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return dict(payload)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    require(path.is_file(), f"missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    require(bool(rows), f"empty CSV artifact: {path}")
    return rows


def _number(value: Any, field: str) -> float:
    require(not isinstance(value, bool), f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    require(math.isfinite(parsed), f"non-finite {field}: {value!r}")
    return parsed


def _integer(value: Any, field: str, *, nonnegative: bool = True) -> int:
    parsed = _number(value, field)
    require(parsed == int(parsed), f"non-integral {field}: {value!r}")
    result = int(parsed)
    if nonnegative:
        require(result >= 0, f"negative {field}: {value!r}")
    return result


def _seed_block(value: Any, field: str) -> tuple[int, ...]:
    if isinstance(value, Mapping):
        start = _integer(value.get("start"), f"{field}.start")
        end = _integer(value.get("end"), f"{field}.end")
        require(end >= start, f"{field} has reversed bounds")
        seeds = tuple(range(start, end + 1))
        if value.get("count") is not None:
            require(
                _integer(value.get("count"), f"{field}.count") == len(seeds),
                f"{field} count drift",
            )
        return seeds
    text = str(value or "").strip()
    parts = text.split("-", 1)
    require(len(parts) == 2, f"{field} must be an inclusive range")
    start = _integer(parts[0], f"{field}.start")
    end = _integer(parts[1], f"{field}.end")
    require(end >= start, f"{field} has reversed bounds")
    return tuple(range(start, end + 1))


def _resolve_cell_path(bundle: Path, group: str, seed: int) -> Path:
    return (bundle / group / "highway" / f"seed_{seed}").resolve()


def _single_path(paths: Iterable[Path], field: str) -> Path:
    resolved = sorted(paths)
    require(len(resolved) == 1, f"expected exactly one {field}, found {len(resolved)}")
    return resolved[0]


def _event_latency(event: Mapping[str, Any]) -> Optional[float]:
    for key in (
        "slow_response_wall_latency_s",
        "closed_loop_latency_terminal_wall_duration_s",
        "closed_loop_latency_realized_wall_seconds",
        "inference_latency",
    ):
        value = event.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed >= 0.0:
            return parsed
    return None


def canonical_distinct_actuation(event: Mapping[str, Any]) -> bool:
    """Return whether a released slow proposal changed executed control."""
    if not bool(event.get("closed_loop_latency_release_event", False)):
        return False
    if not bool(event.get("closed_loop_release_action_alignment_evaluated", False)):
        return False
    if not bool(event.get("closed_loop_release_action_alignment_pass", False)):
        return False
    if bool(event.get("closed_loop_release_opportunity_rejected", False)):
        return False
    if bool(event.get("closed_loop_release_action_unavailable", False)):
        return False
    try:
        fast_action = int(event["closed_loop_execution_state_fast_action"])
        executed_action = int(event["closed_loop_latency_executed_action"])
    except (KeyError, TypeError, ValueError):
        return False
    return executed_action != fast_action


def _authorized_release(event: Mapping[str, Any]) -> bool:
    return bool(
        event.get("closed_loop_latency_release_event", False)
        and event.get("closed_loop_release_action_alignment_evaluated", False)
        and event.get("closed_loop_release_action_alignment_pass", False)
        and not event.get("closed_loop_release_opportunity_rejected", False)
        and not event.get("closed_loop_release_action_unavailable", False)
    )


def _lifecycle(events: Sequence[Mapping[str, Any]], dropped: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    issued: Dict[str, int] = {}
    terminal: Dict[str, str] = {}
    latencies: list[float] = []
    releases = authorized = distinct = 0
    for index, event in enumerate(events):
        if bool(event.get("closed_loop_latency_issuance_event", False)):
            request_id = str(event.get("closed_loop_latency_issued_request_id", "") or "")
            require(request_id and request_id not in issued, f"duplicate issuance at frame {index}")
            issued[request_id] = index
        if bool(event.get("closed_loop_latency_terminal_event", False)):
            request_id = str(event.get("closed_loop_latency_terminal_request_id", "") or "")
            outcome = str(event.get("closed_loop_latency_terminal_response_outcome", "") or "")
            require(request_id in issued, f"orphan terminal at frame {index}")
            require(request_id not in terminal, f"duplicate terminal for {request_id}")
            require(outcome in {"valid", "timeout", "failure"}, f"invalid terminal outcome: {outcome}")
            terminal[request_id] = outcome
            latency = _event_latency(event)
            if latency is not None:
                latencies.append(latency)
        if bool(event.get("closed_loop_latency_release_event", False)):
            releases += 1
            authorized += int(_authorized_release(event))
            distinct += int(canonical_distinct_actuation(event))
    dropped_ids = {
        str(row.get("request_id", "") or "") for row in dropped if row.get("request_id")
    }
    require(not (dropped_ids & set(terminal)), "request is both terminal and episode-end dropped")
    require(dropped_ids <= set(issued), "episode-end drop has no issuance")
    unresolved = set(issued) - set(terminal) - dropped_ids
    require(not unresolved, f"unterminated request IDs: {sorted(unresolved)}")
    return {
        "request_count": len(issued),
        "terminal_count": len(terminal),
        "valid_response_count": sum(value == "valid" for value in terminal.values()),
        "timeout_count": sum(value == "timeout" for value in terminal.values()),
        "failure_count": sum(value == "failure" for value in terminal.values()),
        "dropped_at_episode_end_count": len(dropped_ids),
        "release_count": releases,
        "authorized_release_count": authorized,
        "distinct_actuation_count": distinct,
        "terminal_latencies_s": latencies,
    }


def _validate_protocol(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing formal protocol: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    require(isinstance(payload, Mapping), "formal protocol is not a mapping")
    protocol = dict(payload)
    require(protocol.get("protocol_name") == PROTOCOL_NAME, "protocol name drift")
    require(_integer(protocol.get("protocol_version"), "protocol version") == 13, "protocol version drift")
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    require(submission.get("rgd_method_version") == METHOD_VERSION, "method version drift")
    require(submission.get("query_gate_method_version") == QUERY_GATE_VERSION, "query-gate version drift")
    require(submission.get("release_contract_version") == RELEASE_CONTRACT_VERSION, "release-contract version drift")
    artifacts = dict(dict(submission.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {})
    main = dict(artifacts.get("main_results", {}) or {})
    require(tuple(main.get("required_groups", ())) == FORMAL_GROUPS, "main group contract drift")
    require(main.get("environment") == EXPECTED_ENVIRONMENT, "main environment drift")
    seeds = _seed_block(submission.get("main_seeds"), "main_seeds")
    execution = dict(main.get("execution_contract", {}) or {})
    expected_steps = _integer(execution.get("expected_policy_steps"), "expected policy steps")
    require(expected_steps > 0, "formal horizon is empty")
    return {
        "protocol": protocol,
        "submission": submission,
        "main_contract": main,
        "seeds": seeds,
        "expected_steps": expected_steps,
        "protocol_sha256": _sha256_file(path),
    }


def _validate_identity(
    *,
    row: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    runtime: Mapping[str, Any],
    group: str,
    seed: int,
) -> str:
    for field in ("protocol_hash", "config_hash", "source_hash"):
        value = str(row.get(field, "") or "")
        require(_is_sha256(value), f"{group}/{seed}: invalid {field}")
        require(snapshot.get(field) == value, f"{group}/{seed}: snapshot {field} drift")
        require(runtime.get(field) == value, f"{group}/{seed}: runtime {field} drift")
    protocol_id = str(row.get("protocol_id", "") or "")
    require(protocol_id and snapshot.get("protocol_id") == protocol_id, f"{group}/{seed}: protocol ID drift")
    require(runtime.get("protocol_id") == protocol_id, f"{group}/{seed}: runtime protocol ID drift")
    for payload, label in ((snapshot, "snapshot"), (runtime, "runtime")):
        require(_integer(payload.get("fixed_seed_override"), f"{label} seed") == seed, f"{group}/{seed}: {label} seed drift")
    config = snapshot.get("config")
    require(isinstance(config, Mapping), f"{group}/{seed}: snapshot config is missing")
    require(str(config.get("LLM_PROVIDER", "") or "").lower() == "siliconflow", f"{group}/{seed}: provider drift")
    require(config.get("SILICONFLOW_CHAT_MODEL") == "Qwen/Qwen3-8B", f"{group}/{seed}: model drift")
    require(config.get("rgd_system_method_version") == METHOD_VERSION, f"{group}/{seed}: runtime method drift")
    require(config.get("rgd_query_gate_method_version") == QUERY_GATE_VERSION, f"{group}/{seed}: runtime query-gate drift")
    return str(row["source_hash"])


def validate_bundle(bundle: Path, protocol_path: Path) -> Dict[str, Any]:
    bundle = Path(bundle).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol_info = _validate_protocol(protocol_path)
    manifest = _read_json(bundle / "result_bundle_manifest.json")
    require(manifest.get("bundle_kind") == "formal_run", "bundle is not formal_run")
    require(manifest.get("partition") == "main", "bundle is not the main partition")
    require(tuple(manifest.get("groups", ())) == FORMAL_GROUPS, "bundle group order drift")
    require(tuple(manifest.get("envs", ())) == (EXPECTED_ENVIRONMENT,), "bundle environment drift")
    seeds = tuple(protocol_info["seeds"])
    require(tuple(int(value) for value in manifest.get("seed_labels", ())) == seeds, "bundle seed cohort drift")
    require(_integer(manifest.get("episodes"), "bundle episodes") == 1, "main bundle requires one episode per seed")
    require(_integer(manifest.get("expected_policy_steps"), "bundle horizon") == protocol_info["expected_steps"], "bundle horizon drift")
    require(manifest.get("method_version") == METHOD_VERSION, "bundle method version drift")
    require(manifest.get("query_gate_method_version") == QUERY_GATE_VERSION, "bundle query-gate version drift")
    require(manifest.get("release_contract_version") == RELEASE_CONTRACT_VERSION, "bundle release-contract version drift")

    matrix: Dict[tuple[int, str], Dict[str, Any]] = {}
    input_paths: list[Path] = [bundle / "result_bundle_manifest.json", protocol_path]
    source_hashes: set[str] = set()
    for group in FORMAL_GROUPS:
        run_rows_path = bundle / group / f"{group}_run_rows.csv"
        run_rows = _read_csv(run_rows_path)
        input_paths.append(run_rows_path)
        require(len(run_rows) == len(seeds), f"{group}: run-row count drift")
        rows_by_seed: Dict[int, Dict[str, str]] = {}
        for row in run_rows:
            seed = _integer(row.get("seed_idx"), f"{group} seed")
            require(seed in seeds and seed not in rows_by_seed, f"{group}: unexpected or duplicate seed {seed}")
            require(row.get("group") == group, f"{group}/{seed}: group drift")
            require(row.get("env") == EXPECTED_ENVIRONMENT, f"{group}/{seed}: environment drift")
            rows_by_seed[seed] = row
        require(set(rows_by_seed) == set(seeds), f"{group}: incomplete seed block")

        for seed in seeds:
            row = rows_by_seed[seed]
            cell = _resolve_cell_path(bundle, group, seed)
            declared_cell = Path(str(row.get("result_dir", "") or "")).resolve()
            require(declared_cell == cell, f"{group}/{seed}: result directory drift")
            snapshot_path = cell / "experiment_snapshot.json"
            runtime_path = cell / "runtime_manifest.json"
            metrics_path = cell / f"{group}_rgd_metrics.json"
            event_path = _single_path((cell / "event_logs").glob("event_log_*.json"), f"{group}/{seed} event log")
            physical_path = _single_path(cell.glob("ep_*/*_physical_frames.json"), f"{group}/{seed} physical trace")
            reasoning_path = _single_path(cell.glob("ep_*/*_reasoning_records.json"), f"{group}/{seed} reasoning trace")
            cell_paths = [snapshot_path, runtime_path, metrics_path, event_path, physical_path, reasoning_path]
            input_paths.extend(cell_paths)
            snapshot = _read_json(snapshot_path)
            runtime = _read_json(runtime_path)
            source_hashes.add(
                _validate_identity(
                    row=row,
                    snapshot=snapshot,
                    runtime=runtime,
                    group=group,
                    seed=seed,
                )
            )
            metrics_payload = _read_json(metrics_path)
            metrics = metrics_payload.get("comprehensive_metrics")
            require(isinstance(metrics, Mapping), f"{group}/{seed}: comprehensive metrics missing")
            physical = _read_json(physical_path)
            physical_metrics = physical.get("metrics")
            require(isinstance(physical_metrics, Mapping), f"{group}/{seed}: physical metrics missing")
            events_payload = _read_json(event_path)
            events = list(events_payload.get("events", []) or [])
            require(all(isinstance(event, Mapping) for event in events), f"{group}/{seed}: malformed events")
            reasoning = _read_json(reasoning_path)
            records = list(reasoning.get("analysis_records", []) or [])
            frames = list(physical.get("frames", []) or [])
            frame_count = _integer(metrics.get("total_frames"), f"{group}/{seed} frame count")
            require(frame_count == len(events) == len(records) == len(frames), f"{group}/{seed}: trace count closure failed")
            require(frame_count <= protocol_info["expected_steps"], f"{group}/{seed}: horizon overflow")
            if events_payload.get("event_count") is not None:
                require(_integer(events_payload.get("event_count"), "event count") == len(events), f"{group}/{seed}: event count drift")
            lifecycle = _lifecycle(
                [dict(event) for event in events],
                [dict(item) for item in list(events_payload.get("pending_releases_dropped_at_episode_end", []) or [])],
            )
            endpoint = {
                "group": group,
                "seed": seed,
                "success_rate": _number(metrics.get("success_rate"), f"{group}/{seed} success"),
                "collision_rate": _number(metrics.get("collision_rate"), f"{group}/{seed} collision"),
                "route_completion": _number(metrics.get("avg_route_completion"), f"{group}/{seed} route completion"),
                "episode_reward": _number(metrics.get("avg_episode_reward"), f"{group}/{seed} episode reward"),
                "driving_distance_m": _number(metrics.get("avg_driving_distance"), f"{group}/{seed} distance"),
                "safety_qualified_speed_mps": _number(metrics.get("avg_speed_safety_qualified"), f"{group}/{seed} safety speed"),
                "all_frame_speed_mps": _number(metrics.get("avg_speed_all_frames"), f"{group}/{seed} frame speed"),
                "algorithm_runtime_ms_per_frame": 1000.0 * _number(metrics.get("avg_runtime_per_frame"), f"{group}/{seed} runtime"),
                **{key: lifecycle[key] for key in INTEGER_ENDPOINTS},
                "timeout_count": lifecycle["timeout_count"],
                "failure_count": lifecycle["failure_count"],
                "dropped_at_episode_end_count": lifecycle["dropped_at_episode_end_count"],
                "terminal_count": lifecycle["terminal_count"],
                "frame_count": frame_count,
                "terminal_latencies_s": lifecycle["terminal_latencies_s"],
                "protocol_hash": str(row["protocol_hash"]),
                "config_hash": str(row["config_hash"]),
                "source_hash": str(row["source_hash"]),
            }
            for field in BINARY_ENDPOINTS:
                require(endpoint[field] in {0.0, 1.0}, f"{group}/{seed}: {field} is not binary")
            require(0.0 <= endpoint["route_completion"] <= 1.0, f"{group}/{seed}: route completion outside [0,1]")
            if group == "always_fast":
                require(endpoint["request_count"] == 0, f"{group}/{seed}: slow request observed")
            matrix[(seed, group)] = endpoint

    require(len(source_hashes) == 1, "runtime source hash differs across cells")
    require(len(matrix) == len(seeds) * len(FORMAL_GROUPS), "main matrix is incomplete")
    return {
        **protocol_info,
        "bundle": bundle,
        "bundle_manifest": manifest,
        "matrix": matrix,
        "source_hash": next(iter(source_hashes)),
        "input_paths": input_paths,
    }


def _bootstrap_indices(n: int, draws: int, seed: int) -> np.ndarray:
    require(n > 0 and draws > 0, "bootstrap dimensions must be positive")
    return np.random.default_rng(int(seed)).integers(0, n, size=(int(draws), n))


def _mean_ci(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    point = float(np.mean(values))
    draws = np.mean(values[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    require(total > 0, "Wilson interval requires a positive denominator")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _mcnemar_exact(left: np.ndarray, right: np.ndarray) -> float:
    left_only = int(np.sum((left == 1.0) & (right == 0.0)))
    right_only = int(np.sum((left == 0.0) & (right == 1.0)))
    total = left_only + right_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def _sign_flip_pvalue(values: np.ndarray, *, draws: int, seed: int) -> float:
    if np.allclose(values, 0.0):
        return 1.0
    rng = np.random.default_rng(int(seed))
    observed = abs(float(np.mean(values)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(draws), len(values)))
    permuted = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.sum(permuted >= observed - 1e-15)) / (int(draws) + 1))


def _holm_by_endpoint(rows: list[Dict[str, Any]]) -> None:
    for endpoint in ENDPOINTS:
        family = [row for row in rows if row["endpoint"] == endpoint]
        ordered = sorted(family, key=lambda row: float(row["p_value_raw"]))
        running = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (len(ordered) - index) * float(row["p_value_raw"]))
            running = max(running, adjusted)
            row["p_value_holm_within_endpoint"] = running


def analyze(validated: Mapping[str, Any], *, draws: int, bootstrap_seed: int) -> Dict[str, Any]:
    seeds = tuple(validated["seeds"])
    matrix = dict(validated["matrix"])
    indices = _bootstrap_indices(len(seeds), int(draws), int(bootstrap_seed))
    summary_rows: list[Dict[str, Any]] = []
    lifecycle_rows: list[Dict[str, Any]] = []
    by_seed_rows: list[Dict[str, Any]] = []
    for group in FORMAL_GROUPS:
        rows = [matrix[(seed, group)] for seed in seeds]
        output: Dict[str, Any] = {
            "group": group,
            "n_seeds": len(seeds),
            "successes": int(sum(row["success_rate"] for row in rows)),
            "collisions": int(sum(row["collision_rate"] for row in rows)),
            "bootstrap_draws": int(draws),
            "bootstrap_seed": int(bootstrap_seed),
        }
        for endpoint in ENDPOINTS:
            values = np.asarray([row[endpoint] for row in rows], dtype=float)
            point, low, high = _mean_ci(values, indices)
            output[f"{endpoint}_mean"] = point
            output[f"{endpoint}_ci_low"] = low
            output[f"{endpoint}_ci_high"] = high
            if endpoint in BINARY_ENDPOINTS:
                wilson_low, wilson_high = _wilson(int(np.sum(values)), len(values))
                output[f"{endpoint}_wilson_low"] = wilson_low
                output[f"{endpoint}_wilson_high"] = wilson_high
        latencies = [
            float(value)
            for row in rows
            for value in list(row.get("terminal_latencies_s", []) or [])
        ]
        output.update(
            {
                "request_total": int(sum(row["request_count"] for row in rows)),
                "terminal_total": int(sum(row["terminal_count"] for row in rows)),
                "valid_response_total": int(sum(row["valid_response_count"] for row in rows)),
                "timeout_total": int(sum(row["timeout_count"] for row in rows)),
                "failure_total": int(sum(row["failure_count"] for row in rows)),
                "dropped_total": int(sum(row["dropped_at_episode_end_count"] for row in rows)),
                "terminal_latency_median_s": median(latencies) if latencies else "",
                "terminal_latency_p95_s": float(np.quantile(latencies, 0.95)) if latencies else "",
            }
        )
        summary_rows.append(output)
        lifecycle_rows.append(
            {
                "group": group,
                "n_seeds": len(seeds),
                "request_count": output["request_total"],
                "terminal_count": output["terminal_total"],
                "valid_response_count": output["valid_response_total"],
                "timeout_count": output["timeout_total"],
                "failure_count": output["failure_total"],
                "dropped_at_episode_end_count": output["dropped_total"],
                "release_count": int(sum(row["release_count"] for row in rows)),
                "authorized_release_count": int(sum(row["authorized_release_count"] for row in rows)),
                "distinct_actuation_count": int(sum(row["distinct_actuation_count"] for row in rows)),
                "valid_response_rate": (
                    output["valid_response_total"] / output["request_total"]
                    if output["request_total"]
                    else ""
                ),
                "terminal_latency_median_s": output["terminal_latency_median_s"],
                "terminal_latency_p95_s": output["terminal_latency_p95_s"],
            }
        )
        for row in rows:
            by_seed_rows.append(
                {
                    key: value
                    for key, value in row.items()
                    if key != "terminal_latencies_s"
                }
            )

    contrasts: list[Dict[str, Any]] = []
    for baseline_index, baseline in enumerate(FORMAL_GROUPS[1:]):
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            reference = np.asarray(
                [matrix[(seed, REFERENCE_GROUP)][endpoint] for seed in seeds], dtype=float
            )
            comparator = np.asarray(
                [matrix[(seed, baseline)][endpoint] for seed in seeds], dtype=float
            )
            differences = reference - comparator
            point, low, high = _mean_ci(differences, indices)
            if endpoint in BINARY_ENDPOINTS:
                p_value = _mcnemar_exact(reference, comparator)
                test = "exact_mcnemar"
            else:
                p_value = _sign_flip_pvalue(
                    differences,
                    draws=int(draws),
                    seed=int(bootstrap_seed) + baseline_index * 1009 + endpoint_index,
                )
                test = "paired_sign_flip"
            favorable = -differences if endpoint in LOWER_IS_BETTER else differences
            standard_deviation = (
                float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
            )
            contrasts.append(
                {
                    "reference_group": REFERENCE_GROUP,
                    "baseline_group": baseline,
                    "endpoint": endpoint,
                    "estimate_rgd_minus_baseline": point,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_standardized_effect_dz": (
                        point / standard_deviation if standard_deviation > 0.0 else ""
                    ),
                    "rgd_wins": int(np.sum(favorable > 0.0)),
                    "ties": int(np.sum(favorable == 0.0)),
                    "baseline_wins": int(np.sum(favorable < 0.0)),
                    "p_value_raw": p_value,
                    "test": test,
                    "n_seed_pairs": len(seeds),
                }
            )
    _holm_by_endpoint(contrasts)
    return {
        "summary": summary_rows,
        "by_seed": by_seed_rows,
        "lifecycle": lifecycle_rows,
        "paired_contrasts": contrasts,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validated = validate_bundle(args.bundle, args.protocol)
    contract = dict(validated["main_contract"])
    draws = int(
        args.bootstrap_draws
        if args.bootstrap_draws is not None
        else contract.get("bootstrap_draws", 20000)
    )
    bootstrap_seed = int(
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else contract.get("bootstrap_seed", 20260807)
    )
    require(draws > 0 and bootstrap_seed >= 0, "invalid bootstrap configuration")
    analysis = analyze(validated, draws=draws, bootstrap_seed=bootstrap_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "main_results.csv": analysis["summary"],
        "main_results_by_seed.csv": analysis["by_seed"],
        "main_results_lifecycle.csv": analysis["lifecycle"],
        "main_results_paired_contrasts.csv": analysis["paired_contrasts"],
    }
    for name, rows in output_paths.items():
        _write_csv(output_dir / name, rows)
    manifest = {
        "schema": "rgd_v13_main_results_manifest_v1",
        "accepted": True,
        "analysis_version": ANALYSIS_VERSION,
        "method_version": METHOD_VERSION,
        "query_gate_method_version": QUERY_GATE_VERSION,
        "release_contract_version": RELEASE_CONTRACT_VERSION,
        "bundle": str(validated["bundle"]),
        "protocol": str(Path(args.protocol).resolve()),
        "protocol_sha256": validated["protocol_sha256"],
        "runtime_source_sha256": validated["source_hash"],
        "groups": list(FORMAL_GROUPS),
        "seeds": list(validated["seeds"]),
        "environment": EXPECTED_ENVIRONMENT,
        "matrix_cells": len(validated["matrix"]),
        "bootstrap": {
            "unit": "paired_simulator_seed",
            "draws": draws,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
        "inference": {
            "binary": "two-sided exact McNemar with paired-seed risk-difference bootstrap interval",
            "continuous": "paired-seed mean difference with sign-flip randomization test",
            "multiplicity": "Holm within endpoint across five prespecified RGD-baseline contrasts",
            "proportion_interval": "Wilson 95%",
        },
        "input_sha256": {
            str(path.resolve()): _sha256_file(path)
            for path in sorted(set(validated["input_paths"]), key=lambda item: str(item))
        },
        "output_sha256": {
            name: _sha256_file(output_dir / name) for name in output_paths
        },
    }
    manifest_path = output_dir / "main_results_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"accepted": True, "manifest": str(manifest_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
