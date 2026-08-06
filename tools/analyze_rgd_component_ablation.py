"""Release-state factorial ablation of the RGD gate components.

The analysis reuses the held-out Fast-only traces and matched-action branch
definition from ``analyze_release_state_rollouts.py``.  Removed components are
    set to their multiplicative identity (one); removing A also removes its hard
    support-breadth check. Absolute action feasibility remains fail-closed in every
    arm. Thresholds, budget, cooldown, release delay,
action interface, safety stack, rollout horizon, and simulator seeds remain
fixed across all arms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_common_trajectory_allocators import (  # noqa: E402
    RGD_FLOOR,
    RGD_THRESHOLD,
    scheduled_frames,
)
from tools.analyze_release_state_rollouts import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    RAW_ACTIONS,
    _build_fast_config,
    _capture_release_snapshots,
    _effective_identity,
    _load_trace,
    _query_gate_components,
    _run_branch,
    _write_csv,
)


@dataclass(frozen=True)
class ArmSpec:
    label: str
    use_l: bool
    use_a: bool
    use_h: bool


# Leave-one-out arms come first so the main-paper table remains easy to scan;
# the remaining rows complete the 2^3 design and expose component interactions.
ARM_SPECS: Tuple[ArmSpec, ...] = (
    ArmSpec("RGD", True, True, True),
    ArmSpec("RGD w/o L", False, True, True),
    ArmSpec("RGD w/o A", True, False, True),
    ArmSpec("RGD w/o H", True, True, False),
    ArmSpec("RGD L only", True, False, False),
    ArmSpec("RGD A only", False, True, False),
    ArmSpec("RGD H only", False, False, True),
    ArmSpec("RGD need only", False, False, False),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def component_scores(
    components: Mapping[str, Any],
    spec: ArmSpec,
) -> Dict[str, Any]:
    """Evaluate one factorial arm using the frozen RGD thresholds."""
    latency = float(components["latency_survival"]) if spec.use_l else 1.0
    alternatives = (
        float(components["admissible_alternative_fraction"]) if spec.use_a else 1.0
    )
    headroom = float(components["recovery_headroom"]) if spec.use_h else 1.0
    need = float(components["need_score"])
    opportunity = float(latency * math.sqrt(max(0.0, alternatives * headroom)))
    priority = float(opportunity * need)
    raw_feasibility_ok = bool(components.get("raw_feasibility_valid", False))
    support_evidence_check_active = bool(spec.use_a)
    support_evidence_ok = (
        bool(components.get("support_cost_complete", False))
        if support_evidence_check_active
        else True
    )
    hard_alternative_ok = (
        int(components["alternative_count"]) >= 1 if spec.use_a else True
    )
    absolute_alternative_ok = int(components["absolute_alternative_count"]) >= 1
    eligible = bool(
        raw_feasibility_ok
        and support_evidence_ok
        and absolute_alternative_ok
        and
        hard_alternative_ok
        and opportunity >= RGD_FLOOR
        and priority >= RGD_THRESHOLD
    )
    return {
        "ablated_latency_survival": latency,
        "ablated_admissible_alternative_fraction": alternatives,
        "ablated_recovery_headroom": headroom,
        "ablated_opportunity": opportunity,
        "ablated_priority": priority,
        "hard_alternative_check_active": bool(spec.use_a),
        "hard_alternative_ok": bool(hard_alternative_ok),
        "raw_feasibility_ok": bool(raw_feasibility_ok),
        "support_evidence_check_active": bool(support_evidence_check_active),
        "support_evidence_ok": bool(support_evidence_ok),
        "absolute_alternative_ok": bool(absolute_alternative_ok),
        "eligible": eligible,
    }


def _validate_protocol_contract(
    protocol_path: Path,
    *,
    seeds: Sequence[int],
    delay_s: float,
    horizon: int,
    gamma: float,
    epsilon: float,
    bootstrap_draws: int,
) -> Dict[str, Any]:
    """Require the fixed component-ablation protocol before a replay starts."""
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8-sig")) or {}
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    component = dict(submission.get("component_ablation", {}) or {})
    expected_range = {
        "start": int(min(seeds)),
        "end": int(max(seeds)),
        "count": len(seeds),
    }
    required = {
        "delay_s": float(delay_s),
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "corrective_margin": float(epsilon),
        "opportunity_floor": RGD_FLOOR,
        "priority_threshold": RGD_THRESHOLD,
        "bootstrap_draws": int(bootstrap_draws),
        "removed_component_value": 1.0,
        "remove_support_hard_gate_when_A_removed": True,
        "absolute_alternative_feasibility_non_ablatable": True,
        "alternative_metric_source": "action_support_ranking_costs",
        "headroom_metric_source": "action_recovery_costs",
        "viable_cost_threshold": 0.55,
    }
    if str(submission.get("rgd_method_version", "")) != "support_breadth_v11":
        raise ValueError("component ablation method version drift")
    if component.get("seed_range") != expected_range:
        raise ValueError("component ablation seed range drift")
    for field, expected in required.items():
        if component.get(field) != expected:
            raise ValueError(f"component ablation contract drift: {field}")
    runtime = dict(protocol.get("runtime_config", {}) or {})
    if runtime.get("rgd_decision_threshold") != RGD_THRESHOLD:
        raise ValueError("component ablation runtime threshold drift")
    return component


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    trace_root: Path,
    protocol_path: Path,
    seeds: Sequence[int],
    delay_s: float,
    horizon: int,
    gamma: float,
    epsilon: float,
    bootstrap_draws: int,
) -> None:
    """Validate that release rollouts were produced by the locked source arm."""
    expected = {
        "trace_root": str(trace_root.resolve()),
        "protocol": str(protocol_path.resolve()),
        "seeds": [int(seed) for seed in seeds],
        "seed_is_experimental_unit": True,
        "method_version": "support_breadth_v11",
        "alternative_metric_source": "action_support_ranking_costs",
        "headroom_metric_source": "action_recovery_costs",
        "absolute_alternative_feasibility_non_ablatable": True,
        "viable_cost_threshold": 0.55,
        "delays_s": [0.7, float(delay_s), 2.7],
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
        "policy_frequency_hz": 10,
        "bootstrap_draws": int(bootstrap_draws),
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"component ablation source manifest drift: {field}")


def _paper_acceptance(
    summary: Sequence[Mapping[str, Any]],
    *,
    legal_action_provenance: str,
) -> Dict[str, Any]:
    """State the manuscript gate only when full RGD is strictly superior."""
    rates = {
        str(row.get("arm", "")): float(row.get("corrective_set_fraction"))
        for row in summary
    }
    full = rates.get("RGD")
    comparators = []
    for label in ("RGD w/o L", "RGD w/o A", "RGD w/o H"):
        if full is None or label not in rates:
            raise ValueError(f"missing component-ablation arm: {label}")
        margin = float(full - rates[label])
        comparators.append(
            {
                "arm": label,
                "full_rgd_fraction": float(full),
                "comparator_fraction": float(rates[label]),
                "margin_fraction": margin,
                "strictly_superior": bool(margin > 0.0),
            }
        )
    metric_passed = bool(all(item["strictly_superior"] for item in comparators))
    exact_actions = str(legal_action_provenance) == "exact"
    return {
        "metric_passed": metric_passed,
        "legal_action_provenance": str(legal_action_provenance),
        "exact_action_provenance": exact_actions,
        "comparators": comparators,
        "passed": bool(metric_passed and exact_actions),
    }


def _select_frames(
    records: Sequence[Dict[str, Any]],
    spec: ArmSpec,
    delay_s: float,
) -> List[int]:
    def predicate(record: Dict[str, Any]) -> bool:
        components = _query_gate_components(record, delay_s)
        return bool(component_scores(components, spec)["eligible"])

    return scheduled_frames(records, predicate)


def _branch_outcome(
    rows: Sequence[Dict[str, Any]],
    epsilon: float,
) -> Dict[str, Any]:
    baselines = [row for row in rows if str(row["raw_action"]) == "fast"]
    if len(baselines) != 1:
        raise ValueError(f"expected one matched-fast branch, found {len(baselines)}")
    baseline = baselines[0]
    candidates = [row for row in rows if str(row["raw_action"]) != "fast"]
    baseline_identity = _effective_identity(baseline)
    distinct_by_effect: Dict[Tuple[int, float], Dict[str, Any]] = {}
    for candidate in candidates:
        identity = _effective_identity(candidate)
        current = distinct_by_effect.get(identity)
        if current is None or float(candidate["utility"]) > float(current["utility"]):
            distinct_by_effect[identity] = candidate
    alternatives = [
        row for identity, row in distinct_by_effect.items() if identity != baseline_identity
    ]
    best_row = max(alternatives, key=lambda row: float(row["utility"]), default=None)
    best_advantage = (
        float(best_row["utility"]) - float(baseline["utility"])
        if best_row is not None
        else float("-inf")
    )
    return {
        "baseline": baseline,
        "distinct_alternatives": len(alternatives),
        "best_row": best_row,
        "best_advantage": best_advantage,
        "corrective": bool(best_row is not None and best_advantage >= float(epsilon)),
    }


def _run_missing_seed(
    seed: int,
    release_frames: Sequence[int],
    trace_root: str,
    protocol_path: str,
    scratch_root: str,
    horizon: int,
    gamma: float,
) -> Dict[str, Any]:
    if not release_frames:
        return {"seed": int(seed), "branches": []}
    root = Path(trace_root)
    scratch = Path(scratch_root) / f"seed_{seed}"
    cfg = _build_fast_config(Path(protocol_path), int(seed), scratch)
    cfg["_recorded_snapshot_path"] = str(
        root / "always_fast" / "highway" / f"seed_{seed}" / "snapshots.pkl"
    )
    snapshots, max_position_error = _capture_release_snapshots(
        cfg,
        int(seed),
        sorted({int(value) for value in release_frames}),
        [],
        scratch,
    )
    branches: List[Dict[str, Any]] = []
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        for release_frame in sorted(snapshots):
            snapshot = snapshots[release_frame]
            baseline = _run_branch(snapshot, cfg, int(seed), None, int(horizon), float(gamma))
            legal_actions = {
                int(value)
                for value in str(baseline["legal_actions"]).split(";")
                if value != ""
            }
            branches.append({**baseline, "branch_role": "matched_fast"})
            for action in RAW_ACTIONS:
                if int(action) not in legal_actions:
                    continue
                candidate = _run_branch(
                    snapshot, cfg, int(seed), int(action), int(horizon), float(gamma)
                )
                branches.append({**candidate, "branch_role": "candidate"})
    return {
        "seed": int(seed),
        "branches": branches,
        "max_position_error_m": float(max_position_error),
    }


def _arm_seed_counts(
    rows: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> Dict[str, Dict[int, Tuple[int, int]]]:
    """Pre-aggregate corrective numerator and release denominator once."""
    counts: Dict[str, Dict[int, Tuple[int, int]]] = {
        spec.label: {int(seed): (0, 0) for seed in seeds} for spec in ARM_SPECS
    }
    for row in rows:
        arm = str(row["arm"])
        seed = int(row["seed"])
        numerator, denominator = counts[arm][seed]
        counts[arm][seed] = (
            numerator + int(row["corrective_set_nonempty"]),
            denominator + 1,
        )
    return counts


def _pooled_rate(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    arm: str,
    sampled_seeds: Sequence[int],
) -> float:
    numerator = 0
    denominator = 0
    for seed in sampled_seeds:
        current = counts[arm].get(int(seed), (0, 0))
        numerator += int(current[0])
        denominator += int(current[1])
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _bootstrap_arm_summary(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    arm: str,
    seeds: Sequence[int],
    draws: int,
) -> Tuple[float, float, float, int]:
    point = _pooled_rate(counts, arm, seeds)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: List[float] = []
    population = np.asarray([int(seed) for seed in seeds])
    for _ in range(int(draws)):
        sampled = rng.choice(population, size=len(population), replace=True).tolist()
        value = _pooled_rate(counts, arm, sampled)
        if math.isfinite(value):
            samples.append(value)
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(point), float(low), float(high), len(samples)


def _bootstrap_arm_difference(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    comparator: str,
    seeds: Sequence[int],
    draws: int,
) -> Tuple[float, float, float, int]:
    point = _pooled_rate(counts, "RGD", seeds) - _pooled_rate(counts, comparator, seeds)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: List[float] = []
    population = np.asarray([int(seed) for seed in seeds])
    for _ in range(int(draws)):
        sampled = rng.choice(population, size=len(population), replace=True).tolist()
        full = _pooled_rate(counts, "RGD", sampled)
        other = _pooled_rate(counts, comparator, sampled)
        if math.isfinite(full) and math.isfinite(other):
            samples.append(full - other)
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(point), float(low), float(high), len(samples)


def _factorial_main_effects(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    seeds: Sequence[int],
    draws: int,
) -> List[Dict[str, Any]]:
    arm_by_label = {spec.label: spec for spec in ARM_SPECS}

    def effect(component: str, sampled: Sequence[int]) -> float:
        field = {"L": "use_l", "A": "use_a", "H": "use_h"}[component]
        on = [
            _pooled_rate(counts, label, sampled)
            for label, spec in arm_by_label.items()
            if bool(getattr(spec, field))
        ]
        off = [
            _pooled_rate(counts, label, sampled)
            for label, spec in arm_by_label.items()
            if not bool(getattr(spec, field))
        ]
        on = [value for value in on if math.isfinite(value)]
        off = [value for value in off if math.isfinite(value)]
        return float(np.mean(on) - np.mean(off)) if on and off else float("nan")

    population = np.asarray([int(seed) for seed in seeds])
    output: List[Dict[str, Any]] = []
    for component in ("L", "A", "H"):
        point = effect(component, seeds)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = []
        for _ in range(int(draws)):
            sampled = rng.choice(population, size=len(population), replace=True).tolist()
            value = effect(component, sampled)
            if math.isfinite(value):
                samples.append(value)
        low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
        output.append(
            {
                "component": component,
                "main_effect_fraction": float(point),
                "main_effect_pp": 100.0 * float(point),
                "ci_low_fraction": float(low),
                "ci_high_fraction": float(high),
                "ci_low_pp": 100.0 * float(low),
                "ci_high_pp": 100.0 * float(high),
                "definition": "mean factorial contrast: component on minus component off",
                "cluster_unit": "simulator_seed",
                "bootstrap_draws": int(draws),
                "bootstrap_valid_draws": len(samples),
            }
        )
    return output


def _summaries(
    rows: Sequence[Dict[str, Any]],
    accounting: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    draws: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    counts = _arm_seed_counts(rows, seeds)
    totals = {
        spec.label: {
            "scheduled": sum(
                int(row["scheduled_count"])
                for row in accounting
                if row["arm"] == spec.label
            ),
            "excluded": sum(
                int(row["excluded_count"])
                for row in accounting
                if row["arm"] == spec.label
            ),
            "evaluated": sum(
                int(row["evaluated_count"])
                for row in accounting
                if row["arm"] == spec.label
            ),
        }
        for spec in ARM_SPECS
    }
    summary: List[Dict[str, Any]] = []
    for spec in ARM_SPECS:
        arm_rows = [row for row in rows if row["arm"] == spec.label]
        corrective = sum(int(row["corrective_set_nonempty"]) for row in arm_rows)
        rate, low, high, valid = _bootstrap_arm_summary(
            counts, spec.label, seeds, draws
        )
        if spec.label == "RGD":
            difference = (0.0, 0.0, 0.0, int(draws))
        else:
            difference = _bootstrap_arm_difference(
                counts, spec.label, seeds, draws
            )
        summary.append(
            {
                "arm": spec.label,
                "L": int(spec.use_l),
                "A": int(spec.use_a),
                "H": int(spec.use_h),
                "scheduled_queries": totals[spec.label]["scheduled"],
                "excluded_queries": totals[spec.label]["excluded"],
                "evaluated_releases": totals[spec.label]["evaluated"],
                "corrective_releases": int(corrective),
                "corrective_yield_per_seed": float(corrective / len(seeds)),
                "corrective_set_fraction": rate,
                "ci_low": low,
                "ci_high": high,
                "rgd_minus_arm_fraction": difference[0],
                "rgd_minus_arm_ci_low": difference[1],
                "rgd_minus_arm_ci_high": difference[2],
                "cluster_unit": "simulator_seed",
                "bootstrap_draws": int(draws),
                "bootstrap_valid_draws": int(valid),
                "difference_bootstrap_valid_draws": int(difference[3]),
            }
        )

    by_seed: List[Dict[str, Any]] = []
    accounting_map = {
        (str(row["arm"]), int(row["seed"])): row for row in accounting
    }
    for spec in ARM_SPECS:
        for seed in seeds:
            arm_rows = [
                row
                for row in rows
                if row["arm"] == spec.label and int(row["seed"]) == int(seed)
            ]
            acc = accounting_map[(spec.label, int(seed))]
            corrective = sum(int(row["corrective_set_nonempty"]) for row in arm_rows)
            by_seed.append(
                {
                    "arm": spec.label,
                    "L": int(spec.use_l),
                    "A": int(spec.use_a),
                    "H": int(spec.use_h),
                    "seed": int(seed),
                    "scheduled_queries": int(acc["scheduled_count"]),
                    "excluded_queries": int(acc["excluded_count"]),
                    "evaluated_releases": len(arm_rows),
                    "corrective_releases": int(corrective),
                    "corrective_set_fraction": (
                        float(corrective / len(arm_rows)) if arm_rows else ""
                    ),
                }
            )
    return summary, by_seed


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=160)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--delay", type=float, default=1.7)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TEMP", ".")) / "rgd_component_ablation_scratch",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.delay < 0.0 or args.horizon <= 0 or args.epsilon < 0.0:
        raise ValueError("delay, horizon, or epsilon is invalid")
    seeds = [int(args.seed_start) + offset for offset in range(int(args.seeds))]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)

    source_manifest_path = args.source_analysis / "release_rollout_manifest.json"
    source_branches_path = args.source_analysis / "release_rollout_branches.csv"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if [int(value) for value in source_manifest.get("seeds", [])] != seeds:
        raise ValueError("source release-rollout seed block does not match ablation seeds")
    for field, expected in (
        ("horizon_steps", int(args.horizon)),
        ("gamma", float(args.gamma)),
        ("epsilon", float(args.epsilon)),
    ):
        actual = source_manifest.get(field)
        if not math.isclose(float(actual), float(expected), abs_tol=1e-12):
            raise ValueError(f"source manifest {field}={actual!r}, expected {expected!r}")

    accounting: List[Dict[str, Any]] = []
    event_specs: List[Tuple[int, ArmSpec, int, int, Dict[str, Any], Dict[str, Any]]] = []
    required_release_keys = set()
    delay_steps = int(math.ceil(float(args.delay) * 10.0))
    for seed in seeds:
        records, _ = _load_trace(args.trace_root, int(seed))
        for spec in ARM_SPECS:
            selected = _select_frames(records, spec, float(args.delay))
            evaluated = [
                int(frame)
                for frame in selected
                if int(frame) + delay_steps + int(args.horizon) <= len(records)
            ]
            accounting.append(
                {
                    "arm": spec.label,
                    "L": int(spec.use_l),
                    "A": int(spec.use_a),
                    "H": int(spec.use_h),
                    "seed": int(seed),
                    "scheduled_count": len(selected),
                    "excluded_count": len(selected) - len(evaluated),
                    "evaluated_count": len(evaluated),
                    "eligibility_rule": "release_frame_plus_full_rollout_horizon_within_trace",
                }
            )
            for query_frame in evaluated:
                release_frame = int(query_frame + delay_steps)
                components = _query_gate_components(records[query_frame], float(args.delay))
                ablated = component_scores(components, spec)
                if not bool(ablated["eligible"]):
                    raise RuntimeError("scheduled query failed its own factorial gate")
                event_specs.append(
                    (
                        int(seed),
                        spec,
                        int(query_frame),
                        release_frame,
                        components,
                        ablated,
                    )
                )
                required_release_keys.add((int(seed), release_frame))

    source_branches = _read_csv(source_branches_path)
    branch_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in source_branches:
        key = (int(row["seed"]), int(row["release_frame"]))
        branch_groups.setdefault(key, []).append(row)
    missing_by_seed = {
        seed: sorted(
            release for row_seed, release in required_release_keys
            if row_seed == seed and (row_seed, release) not in branch_groups
        )
        for seed in seeds
    }
    new_branch_rows: List[Dict[str, Any]] = []
    max_position_error = 0.0
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(
                _run_missing_seed,
                int(seed),
                frames,
                str(args.trace_root.resolve()),
                str(args.protocol.resolve()),
                str(args.scratch_root.resolve()),
                int(args.horizon),
                float(args.gamma),
            ): int(seed)
            for seed, frames in missing_by_seed.items()
            if frames
        }
        for future in as_completed(futures):
            result = future.result()
            new_branch_rows.extend(result["branches"])
            max_position_error = max(
                max_position_error,
                float(result.get("max_position_error_m", 0.0) or 0.0),
            )
            print(
                f"seed={result['seed']} new_branches={len(result['branches'])}",
                flush=True,
            )
    for row in new_branch_rows:
        key = (int(row["seed"]), int(row["release_frame"]))
        branch_groups.setdefault(key, []).append(row)

    missing = sorted(key for key in required_release_keys if key not in branch_groups)
    if missing:
        raise RuntimeError(f"component ablation still misses release branches: {missing[:5]}")
    retained_branches = [
        row for key in sorted(required_release_keys) for row in branch_groups[key]
    ]
    retained_branches.sort(
        key=lambda row: (int(row["seed"]), int(row["release_frame"]), str(row["raw_action"]))
    )

    outcomes = {
        key: _branch_outcome(branch_groups[key], float(args.epsilon))
        for key in required_release_keys
    }
    event_rows: List[Dict[str, Any]] = []
    for seed, spec, query_frame, release_frame, components, ablated in event_specs:
        outcome = outcomes[(seed, release_frame)]
        best_row = outcome["best_row"]
        event_rows.append(
            {
                "seed": int(seed),
                "arm": spec.label,
                "L": int(spec.use_l),
                "A": int(spec.use_a),
                "H": int(spec.use_h),
                "query_frame": int(query_frame),
                "release_frame": int(release_frame),
                "delay_s": float(args.delay),
                "candidate_state_id": f"{seed}:{query_frame}:{delay_steps}",
                "release_state_id": f"{seed}:{release_frame}",
                "need_score": float(components["need_score"]),
                "latency_survival": float(components["latency_survival"]),
                "admissible_alternative_fraction": float(
                    components["admissible_alternative_fraction"]
                ),
                "recovery_headroom": float(components["recovery_headroom"]),
                "alternative_count": int(components["alternative_count"]),
                "absolute_alternative_count": int(
                    components["absolute_alternative_count"]
                ),
                **ablated,
                "release_distinct_alternatives": int(outcome["distinct_alternatives"]),
                "corrective_set_nonempty": int(outcome["corrective"]),
                "best_advantage": (
                    float(outcome["best_advantage"])
                    if math.isfinite(float(outcome["best_advantage"]))
                    else ""
                ),
                "baseline_utility": float(outcome["baseline"]["utility"]),
                "baseline_collision": int(outcome["baseline"]["collision"]),
                "best_effective_action": (
                    "" if best_row is None else int(best_row["effective_action"])
                ),
                "best_target_speed_after": (
                    "" if best_row is None else float(best_row["target_speed_after"])
                ),
                "best_collision": "" if best_row is None else int(best_row["collision"]),
                "horizon_steps": int(args.horizon),
                "gamma": float(args.gamma),
                "epsilon": float(args.epsilon),
                "outcome_definition": "normalized simulator return minus collision indicator",
            }
        )
    event_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            next(i for i, spec in enumerate(ARM_SPECS) if spec.label == row["arm"]),
            int(row["query_frame"]),
        )
    )

    summary, by_seed = _summaries(
        event_rows, accounting, seeds, int(args.bootstrap_draws)
    )
    main_effects = _factorial_main_effects(
        _arm_seed_counts(event_rows, seeds), seeds, int(args.bootstrap_draws)
    )
    _write_csv(args.output_dir / "component_ablation_events.csv", event_rows)
    _write_csv(args.output_dir / "component_ablation_branches.csv", retained_branches)
    _write_csv(args.output_dir / "component_ablation_selection_accounting.csv", accounting)
    _write_csv(args.output_dir / "component_ablation_by_seed.csv", by_seed)
    _write_csv(args.output_dir / "component_ablation_summary.csv", summary)
    _write_csv(args.output_dir / "component_ablation_main_effects.csv", main_effects)

    manifest = {
        "analysis": "RGD internal 2^3 component ablation on matched release-state rollouts",
        "design": "full factorial with leave-one-out arms listed first",
        "trace_root": str(args.trace_root.resolve()),
        "source_analysis": str(args.source_analysis.resolve()),
        "protocol": str(args.protocol.resolve()),
        "seeds": seeds,
        "seed_is_experimental_unit": True,
        "query_events_nested_within_seed": True,
        "delay_s": float(args.delay),
        "horizon_steps": int(args.horizon),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "opportunity_floor": float(RGD_FLOOR),
        "priority_threshold": float(RGD_THRESHOLD),
        "budget": 6,
        "cooldown_frames": 20,
        "ablation_rule": (
            "a removed multiplicative component is set to 1; removing A also "
            "removes the support-breadth hard condition; absolute alternative "
            "feasibility remains fail-closed in every arm"
        ),
        "component_definition": {
            "L": "latency survival from the frozen release delay",
            "A": "legal non-hold action support from action-specific ranking costs",
            "H": "absolute recovery-cost headroom",
        },
        "threshold_policy": "fixed across arms; no arm-specific retuning",
        "claim_scope": "functional component ablation, not compute-matched endpoint causality",
        "arms": [spec.__dict__ for spec in ARM_SPECS],
        "required_unique_release_states": len(required_release_keys),
        "source_release_states_reused": sum(
            1 for key in required_release_keys if key in {
                (int(row["seed"]), int(row["release_frame"])) for row in source_branches
            }
        ),
        "new_release_states_evaluated": sum(len(value) for value in missing_by_seed.values()),
        "max_fast_replay_position_error_m": float(max_position_error),
        "bootstrap_draws": int(args.bootstrap_draws),
        "bootstrap_seed": int(BOOTSTRAP_SEED),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_branches_sha256": _sha256(source_branches_path),
        "summary": summary,
        "factorial_main_effects": main_effects,
    }
    manifest_path = args.output_dir / "component_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
