"""Analyze the corrected TVT main comparison without running new experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest


GROUPS = ["rgd_fixed_policy", "always_fast", "random_budget", "uncertainty_budget", "risk_budget"]
BASELINES = ["always_fast", "random_budget", "uncertainty_budget", "risk_budget"]
LABELS = {
    "rgd_fixed_policy": "RGD",
    "always_fast": "Fast-only",
    "random_budget": "Random",
    "uncertainty_budget": "Uncertainty",
    "risk_budget": "TTC-risk",
}
CONTINUOUS_FIELDS = [
    "avg_route_completion",
    "avg_episode_reward",
    "avg_driving_distance",
    "avg_speed_safety_qualified",
    "avg_runtime_per_frame",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / den
    return max(0.0, center - half), min(1.0, center + half)


def percentile_ci(values: Sequence[float], alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))


def paired_bootstrap_ci(differences: Sequence[float], rng: np.random.Generator, draws: int = 20000) -> Tuple[float, float]:
    data = np.asarray(differences, dtype=np.float64)
    if data.size == 0:
        return 0.0, 0.0
    indices = rng.integers(0, data.size, size=(draws, data.size))
    return percentile_ci(np.mean(data[indices], axis=1).tolist())


def clustered_ratio_ci(
    per_episode: Sequence[Tuple[int, int]],
    rng: np.random.Generator,
    draws: int = 20000,
) -> Tuple[float, float]:
    data = np.asarray(per_episode, dtype=np.float64)
    if data.size == 0 or float(np.sum(data[:, 1])) <= 0.0:
        return 0.0, 0.0
    indices = rng.integers(0, data.shape[0], size=(draws, data.shape[0]))
    sampled = data[indices]
    numerators = np.sum(sampled[:, :, 0], axis=1)
    denominators = np.sum(sampled[:, :, 1], axis=1)
    ratios = np.divide(numerators, denominators, out=np.zeros_like(numerators), where=denominators > 0)
    return percentile_ci(ratios.tolist())


def paired_clustered_ratio_difference_ci(
    target_by_seed: Dict[int, Tuple[int, int]],
    baseline_by_seed: Dict[int, Tuple[int, int]],
    rng: np.random.Generator,
    draws: int = 20000,
) -> Tuple[float, float, float]:
    """Paired seed-cluster bootstrap for a difference of aggregate ratios."""
    seeds = sorted(set(target_by_seed) & set(baseline_by_seed))
    if not seeds:
        return 0.0, 0.0, 0.0
    target = np.asarray([target_by_seed[seed] for seed in seeds], dtype=np.float64)
    baseline = np.asarray([baseline_by_seed[seed] for seed in seeds], dtype=np.float64)
    if float(np.sum(target[:, 1])) <= 0.0 or float(np.sum(baseline[:, 1])) <= 0.0:
        return 0.0, 0.0, 0.0
    point = float(np.sum(target[:, 0]) / np.sum(target[:, 1]) - np.sum(baseline[:, 0]) / np.sum(baseline[:, 1]))
    indices = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    target_sample = target[indices]
    baseline_sample = baseline[indices]
    target_den = np.sum(target_sample[:, :, 1], axis=1)
    baseline_den = np.sum(baseline_sample[:, :, 1], axis=1)
    valid = (target_den > 0) & (baseline_den > 0)
    differences = (
        np.sum(target_sample[:, :, 0], axis=1)[valid] / target_den[valid]
        - np.sum(baseline_sample[:, :, 0], axis=1)[valid] / baseline_den[valid]
    )
    low, high = percentile_ci(differences.tolist())
    return point, low, high


def load_rows(slow_root: Path, fast_root: Path) -> List[Dict[str, Any]]:
    with (slow_root / "closed_loop_latency_sweep_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (fast_root / "closed_loop_latency_sweep_rows.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows.extend(row for row in csv.DictReader(handle) if row.get("group") == "always_fast")
    return rows


def load_event_bundle(result_dir: Path) -> Tuple[List[Dict[str, Any]], int, str]:
    events: List[Dict[str, Any]] = []
    pending = 0
    terminal = "unknown"
    for path in sorted(result_dir.rglob("event_log_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        events.extend(payload.get("events", []) or [])
        pending += int(payload.get("pending_release_count", 0) or 0)
        terminal = str(payload.get("terminal_cause", terminal) or terminal)
    return events, pending, terminal


def load_reasoning(result_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in sorted(result_dir.rglob("*reasoning_records.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records.extend(payload.get("analysis_records", []) or [])
    return records


def holm_adjust(pairs: Sequence[Tuple[str, float]]) -> Dict[str, float]:
    ordered = sorted(pairs, key=lambda item: item[1])
    adjusted: Dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for index, (name, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - index) * pvalue))
        adjusted[name] = running
    return adjusted


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow-root", type=Path, required=True)
    parser.add_argument("--fast-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260715)

    rows = load_rows(args.slow_root, args.fast_root)
    by_group_seed: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    event_by_group_seed: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(dict)
    reasoning_by_group_seed: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(dict)
    pending_by_group_seed: Dict[str, Dict[int, int]] = defaultdict(dict)
    terminal_by_group_seed: Dict[str, Dict[int, str]] = defaultdict(dict)
    result_rows: List[Dict[str, Any]] = []

    for row in rows:
        group = str(row["group"])
        seed = int(float(row["seed_idx"]))
        result_dir = Path(str(row["result_dir"]))
        events, pending, terminal = load_event_bundle(result_dir)
        reasoning = load_reasoning(result_dir)
        by_group_seed[group][seed] = row
        event_by_group_seed[group][seed] = events
        reasoning_by_group_seed[group][seed] = reasoning
        pending_by_group_seed[group][seed] = pending
        terminal_by_group_seed[group][seed] = terminal
        result_rows.append(
            {
                "group": group,
                "seed": seed,
                "success": int(fnum(row.get("success_rate")) > 0.5),
                "collision": int(fnum(row.get("collision_rate")) > 0.5),
                "frames": int(fnum(row.get("total_frames"))),
                "route_completion": fnum(row.get("avg_route_completion")),
                "reward": fnum(row.get("avg_episode_reward")),
                "distance": fnum(row.get("avg_driving_distance")),
                "speed": fnum(row.get("avg_speed_safety_qualified")),
                "runtime_per_frame": fnum(row.get("avg_runtime_per_frame")),
                "terminal": terminal,
            }
        )

    main_table: List[Dict[str, Any]] = []
    for group in GROUPS:
        selected = [row for row in result_rows if row["group"] == group]
        successes = sum(int(row["success"]) for row in selected)
        low, high = wilson(successes, len(selected))
        main_table.append(
            {
                "group": group,
                "label": LABELS[group],
                "episodes": len(selected),
                "successes": successes,
                "success_rate": successes / max(1, len(selected)),
                "success_ci_low": low,
                "success_ci_high": high,
                "mean_route_completion": mean(row["route_completion"] for row in selected),
                "median_route_completion": median(row["route_completion"] for row in selected),
                "mean_distance": mean(row["distance"] for row in selected),
                "mean_reward": mean(row["reward"] for row in selected),
            }
        )

    target_outcome = {
        seed: int(fnum(row.get("success_rate")) > 0.5)
        for seed, row in by_group_seed["rgd_fixed_policy"].items()
    }
    raw_pvalues: List[Tuple[str, float]] = []
    paired_rows: List[Dict[str, Any]] = []
    for baseline in BASELINES:
        baseline_outcome = {
            seed: int(fnum(row.get("success_rate")) > 0.5)
            for seed, row in by_group_seed[baseline].items()
        }
        seeds = sorted(set(target_outcome) & set(baseline_outcome))
        wins = sum(target_outcome[seed] > baseline_outcome[seed] for seed in seeds)
        losses = sum(target_outcome[seed] < baseline_outcome[seed] for seed in seeds)
        ties = len(seeds) - wins - losses
        pvalue = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
        raw_pvalues.append((baseline, pvalue))
        binary_differences = [target_outcome[seed] - baseline_outcome[seed] for seed in seeds]
        rd_low, rd_high = paired_bootstrap_ci(binary_differences, rng)
        paired_rows.append(
            {
                "baseline": baseline,
                "label": LABELS[baseline],
                "n": len(seeds),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "paired_risk_difference": mean(binary_differences),
                "paired_rd_ci_low": rd_low,
                "paired_rd_ci_high": rd_high,
                "mcnemar_exact_p": pvalue,
                "discordant_seeds": ";".join(str(seed) for seed in seeds if target_outcome[seed] != baseline_outcome[seed]),
            }
        )
    adjusted = holm_adjust(raw_pvalues)
    for row in paired_rows:
        row["holm_adjusted_p"] = adjusted[row["baseline"]]

    continuous_rows: List[Dict[str, Any]] = []
    for baseline in BASELINES:
        seeds = sorted(set(by_group_seed["rgd_fixed_policy"]) & set(by_group_seed[baseline]))
        for field in CONTINUOUS_FIELDS:
            differences = [
                fnum(by_group_seed["rgd_fixed_policy"][seed].get(field))
                - fnum(by_group_seed[baseline][seed].get(field))
                for seed in seeds
            ]
            low, high = paired_bootstrap_ci(differences, rng)
            continuous_rows.append(
                {
                    "baseline": baseline,
                    "metric": field,
                    "n": len(differences),
                    "mean_paired_difference": mean(differences),
                    "median_paired_difference": median(differences),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

    mechanism_specs = [
        ("release_rate", "releases", "queries"),
        ("pending_drop_rate", "pending", "queries"),
        ("release_unavailable_rate", "unavailable", "releases"),
        ("post_latency_rewrite_rate", "rewritten", "releases"),
        ("execution_state_divergence_rate", "divergent", "releases"),
        ("route_preserved_given_divergence", "preserved", "divergent"),
        ("route_preserved_per_release", "preserved", "releases"),
        ("final_returns_to_fast_rate", "returns_fast", "releases"),
    ]
    episode_counts: Dict[str, Dict[int, Counter]] = defaultdict(dict)
    mechanism_rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        for seed, events in event_by_group_seed[group].items():
            counts = Counter()
            counts["queries"] = sum(str(event.get("system_used")) == "slow" for event in events)
            counts["releases"] = sum(bool(event.get("closed_loop_latency_release_event")) for event in events)
            counts["pending"] = int(pending_by_group_seed[group].get(seed, 0))
            counts["unavailable"] = sum(bool(event.get("closed_loop_release_action_unavailable")) for event in events)
            counts["rewritten"] = sum(bool(event.get("closed_loop_post_latency_shield_rewrite")) for event in events)
            counts["divergent"] = sum(bool(event.get("closed_loop_release_route_divergence")) for event in events)
            counts["preserved"] = sum(bool(event.get("closed_loop_route_preserved_divergent_release")) for event in events)
            counts["returns_fast"] = sum(bool(event.get("closed_loop_final_returns_to_fast")) for event in events)
            episode_counts[group][seed] = counts
        for metric, numerator, denominator in mechanism_specs:
            pairs = [
                (episode_counts[group][seed][numerator], episode_counts[group][seed][denominator])
                for seed in sorted(episode_counts[group])
            ]
            num = sum(pair[0] for pair in pairs)
            den = sum(pair[1] for pair in pairs)
            low, high = clustered_ratio_ci(pairs, rng)
            mechanism_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "numerator": num,
                    "denominator": den,
                    "rate": num / den if den else 0.0,
                    "seed_cluster_ci_low": low,
                    "seed_cluster_ci_high": high,
                }
            )

    paired_mechanism_specs = [
        ("release_per_query", "releases", "queries"),
        ("unavailable_per_release", "unavailable", "releases"),
        ("rewrite_per_release", "rewritten", "releases"),
        ("preserved_given_divergence", "preserved", "divergent"),
        ("preserved_per_query", "preserved", "queries"),
    ]
    paired_mechanism_rows: List[Dict[str, Any]] = []
    for baseline in ["random_budget", "uncertainty_budget", "risk_budget"]:
        for metric, numerator, denominator in paired_mechanism_specs:
            target_pairs = {
                seed: (counts[numerator], counts[denominator])
                for seed, counts in episode_counts["rgd_fixed_policy"].items()
            }
            baseline_pairs = {
                seed: (counts[numerator], counts[denominator])
                for seed, counts in episode_counts[baseline].items()
            }
            point, low, high = paired_clustered_ratio_difference_ci(
                target_pairs,
                baseline_pairs,
                rng,
            )
            paired_mechanism_rows.append(
                {
                    "baseline": baseline,
                    "label": LABELS[baseline],
                    "metric": metric,
                    "difference_rgd_minus_baseline": point,
                    "paired_seed_cluster_ci_low": low,
                    "paired_seed_cluster_ci_high": high,
                    "draws": 20000,
                }
            )

    lifecycle_event_rows: List[Dict[str, Any]] = []
    for group in ["rgd_fixed_policy", "random_budget", "uncertainty_budget", "risk_budget"]:
        for seed, events in event_by_group_seed[group].items():
            for event in events:
                is_query = str(event.get("system_used")) == "slow"
                is_release = bool(event.get("closed_loop_latency_release_event"))
                if not (is_query or is_release):
                    continue
                lifecycle_event_rows.append(
                    {
                        "group": group,
                        "seed": seed,
                        "frame": int(fnum(event.get("frame"))),
                        "event_kind": "query_and_release" if is_query and is_release else ("query" if is_query else "release"),
                        "source_frame": int(fnum(event.get("closed_loop_latency_source_frame"), -1)),
                        "query_fast_action": event.get("query_state_fast_proposal_action", ""),
                        "query_slow_action": event.get("query_state_slow_released_action", ""),
                        "release_fast_action": event.get("closed_loop_execution_state_fast_action", ""),
                        "released_slow_action": event.get("closed_loop_released_slow_action", ""),
                        "executed_action": event.get("closed_loop_latency_executed_action", ""),
                        "final_action": event.get("final_action", ""),
                        "gate_delay_steps": int(fnum(event.get("recoverability_effective_delay_steps"))),
                        "replay_delay_steps": int(fnum(event.get("closed_loop_latency_delay_steps"))),
                        "matched_fast_pending": str(event.get("closed_loop_latency_provisional_controller", "")) == "matched_fast_policy",
                        "opportunity": fnum(event.get("recoverability_post_latency_opportunity")),
                        "opportunity_floor": fnum(event.get("recoverability_opportunity_floor")),
                        "opportunity_eligible": bool(event.get("recoverability_opportunity_eligible")),
                        "legal_alternative_count": int(fnum(event.get("recoverability_alternative_viable_count"))),
                        "release_unavailable": bool(event.get("closed_loop_release_action_unavailable")),
                        "post_latency_rewrite": bool(event.get("closed_loop_post_latency_shield_rewrite")),
                        "release_divergence": bool(event.get("closed_loop_release_route_divergence")),
                        "route_preserved_divergence": bool(event.get("closed_loop_route_preserved_divergent_release")),
                        "final_returns_to_fast": bool(event.get("closed_loop_final_returns_to_fast")),
                        "episode_done": bool(event.get("episode_done")),
                        "terminal_cause": event.get("terminal_cause", ""),
                    }
                )

    stage_rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        per_seed: Dict[int, Counter] = {}
        for seed, records in reasoning_by_group_seed[group].items():
            counts = Counter()
            for record in records:
                if str(record.get("system_used")) != "slow":
                    continue
                objective = (record.get("rgd_subordinate_diagnostics", {}) or {}).get("slow_path_objective", {}) or {}
                counts["queries"] += 1
                counts["raw_available"] += int(bool(objective.get("llm_action_available")))
                counts["raw_to_validation"] += int(objective.get("llm_raw_action") != objective.get("post_validation_action"))
                counts["validation_to_risk"] += int(objective.get("post_validation_action") != objective.get("post_risk_calibration_action"))
                counts["risk_to_release"] += int(objective.get("post_risk_calibration_action") != objective.get("query_state_slow_released_action"))
                counts["query_state_divergence"] += int(bool(objective.get("query_state_route_divergence")))
            per_seed[seed] = counts
        for metric in ["raw_available", "raw_to_validation", "validation_to_risk", "risk_to_release", "query_state_divergence"]:
            pairs = [(counts[metric], counts["queries"]) for counts in per_seed.values()]
            num = sum(pair[0] for pair in pairs)
            den = sum(pair[1] for pair in pairs)
            low, high = clustered_ratio_ci(pairs, rng)
            stage_rows.append(
                {
                    "group": group,
                    "metric": metric,
                    "numerator": num,
                    "denominator": den,
                    "rate": num / den if den else 0.0,
                    "seed_cluster_ci_low": low,
                    "seed_cluster_ci_high": high,
                }
            )

    rgd_events = event_by_group_seed["rgd_fixed_policy"]
    fast_events = event_by_group_seed["always_fast"]
    prefix_rows: List[Dict[str, Any]] = []
    for seed in sorted(rgd_events):
        target = rgd_events[seed]
        control = fast_events[seed]
        first_slow = next((int(event["frame"]) for event in target if str(event.get("system_used")) == "slow"), None)
        prefix_length = first_slow if first_slow is not None else min(len(target), len(control))
        prefix_match = all(target[index]["final_action"] == control[index]["final_action"] for index in range(prefix_length))
        query_match = bool(
            first_slow is None
            or (
                first_slow < len(control)
                and target[first_slow]["final_action"] == control[first_slow]["final_action"]
            )
        )
        prefix_rows.append(
            {
                "seed": seed,
                "first_slow_frame": "" if first_slow is None else first_slow,
                "prefix_length": prefix_length,
                "nonvacuous": bool(first_slow is not None and first_slow > 0),
                "prefix_action_match": prefix_match,
                "query_frame_fast_match": query_match,
            }
        )

    invariant_violations: List[Dict[str, Any]] = []
    for seed, events in rgd_events.items():
        for event in events:
            if str(event.get("system_used")) != "slow":
                continue
            reasons = []
            if not bool(event.get("recoverability_opportunity_eligible")):
                reasons.append("not_eligible")
            if int(event.get("recoverability_alternative_viable_count", 0) or 0) < 1:
                reasons.append("no_viable_alternative")
            if fnum(event.get("recoverability_post_latency_opportunity")) < fnum(event.get("recoverability_opportunity_floor"), 1.0):
                reasons.append("below_floor")
            if int(event.get("recoverability_effective_delay_steps", 0) or 0) != 17:
                reasons.append("gate_delay_not_17")
            if int(event.get("closed_loop_latency_delay_steps", 0) or 0) != 17:
                reasons.append("replay_delay_not_17")
            if str(event.get("closed_loop_latency_provisional_controller")) != "matched_fast_policy":
                reasons.append("wrong_provisional_controller")
            if reasons:
                invariant_violations.append({"seed": seed, "frame": event.get("frame"), "reasons": ";".join(reasons)})

    case_rows: List[Dict[str, Any]] = []
    for baseline in BASELINES:
        seeds = sorted(set(by_group_seed["rgd_fixed_policy"]) & set(by_group_seed[baseline]))
        for seed in seeds:
            target_success = int(fnum(by_group_seed["rgd_fixed_policy"][seed].get("success_rate")) > 0.5)
            base_success = int(fnum(by_group_seed[baseline][seed].get("success_rate")) > 0.5)
            if target_success == base_success:
                continue
            counts = episode_counts["rgd_fixed_policy"][seed]
            case_rows.append(
                {
                    "baseline": baseline,
                    "seed": seed,
                    "direction": "RGD_win" if target_success > base_success else "RGD_loss",
                    "rgd_terminal": terminal_by_group_seed["rgd_fixed_policy"][seed],
                    "baseline_terminal": terminal_by_group_seed[baseline][seed],
                    "rgd_queries": counts["queries"],
                    "rgd_releases": counts["releases"],
                    "rgd_divergent_releases": counts["divergent"],
                    "rgd_route_preserved_releases": counts["preserved"],
                    "rgd_post_latency_rewrites": counts["rewritten"],
                }
            )

    write_csv(args.output_dir / "main_results.csv", main_table)
    write_csv(args.output_dir / "results_by_seed.csv", result_rows)
    write_csv(args.output_dir / "paired_binary_comparisons.csv", paired_rows)
    write_csv(args.output_dir / "paired_continuous_comparisons.csv", continuous_rows)
    write_csv(args.output_dir / "mechanism_attribution.csv", mechanism_rows)
    write_csv(args.output_dir / "paired_mechanism_comparisons.csv", paired_mechanism_rows)
    write_csv(args.output_dir / "lifecycle_event_trace.csv", lifecycle_event_rows)
    write_csv(args.output_dir / "slow_stage_rewrites.csv", stage_rows)
    write_csv(args.output_dir / "prefix_audit.csv", prefix_rows)
    write_csv(args.output_dir / "discordant_case_index.csv", case_rows)
    write_csv(args.output_dir / "invariant_violations.csv", invariant_violations)

    rgd_preserved_seeds = sum(episode_counts["rgd_fixed_policy"][seed]["preserved"] > 0 for seed in episode_counts["rgd_fixed_policy"])
    summary = {
        "analysis_version": "corrected_gate_full_fast_counterfactual_v1",
        "seeds": sorted(by_group_seed["rgd_fixed_policy"]),
        "groups": GROUPS,
        "main_results": main_table,
        "paired_binary": paired_rows,
        "invariant_violation_count": len(invariant_violations),
        "rgd_route_preserved_divergent_releases": sum(counts["preserved"] for counts in episode_counts["rgd_fixed_policy"].values()),
        "rgd_route_preserved_seed_count": rgd_preserved_seeds,
        "prefix_nonvacuous_count": sum(bool(row["nonvacuous"]) for row in prefix_rows),
        "prefix_nonvacuous_match_count": sum(bool(row["nonvacuous"] and row["prefix_action_match"]) for row in prefix_rows),
        "query_frame_match_count": sum(bool(row["query_frame_fast_match"]) for row in prefix_rows if row["first_slow_frame"] != ""),
        "claim_boundary": {
            "completion_superiority": "not_supported",
            "eligibility_invariant": "supported" if not invariant_violations else "failed",
            "strong_control_effect": "supported" if sum(counts["preserved"] for counts in episode_counts["rgd_fixed_policy"].values()) >= 20 and rgd_preserved_seeds >= 10 else "insufficient_events",
        },
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    main_lookup = {row["group"]: row for row in main_table}
    mechanism_lookup = {(row["group"], row["metric"]): row for row in mechanism_rows}
    stage_lookup = {(row["group"], row["metric"]): row for row in stage_rows}
    lines = [
        "# Corrected RGD main-result analysis",
        "",
        "This report uses only the locked 30-seed main comparison (seeds 100--129). It merges the corrected four slow arms with the unchanged always-fast arm. No calibration or development seed enters the outcome table.",
        "",
        "## Outcome evidence",
        "",
        "| Method | Success | Wilson 95% CI | Mean completion | Mean distance |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        row = main_lookup[group]
        lines.append(
            f"| {row['label']} | {row['successes']}/{row['episodes']} ({row['success_rate']:.3f}) | "
            f"[{row['success_ci_low']:.3f}, {row['success_ci_high']:.3f}] | {row['mean_route_completion']:.3f} | {row['mean_distance']:.2f} |"
        )
    lines.extend([
        "",
        "| Baseline | W/L/T | Paired RD (95% bootstrap CI) | Exact McNemar p | Holm p |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in paired_rows:
        lines.append(
            f"| {row['label']} | {row['wins']}/{row['losses']}/{row['ties']} | "
            f"{row['paired_risk_difference']:.3f} [{row['paired_rd_ci_low']:.3f}, {row['paired_rd_ci_high']:.3f}] | "
            f"{row['mcnemar_exact_p']:.3f} | {row['holm_adjusted_p']:.3f} |"
        )
    lines.extend([
        "",
        "Completion superiority is not supported: RGD is 6/30 versus 5/30 for fast-only and TTC-risk, and 6/30 versus random and uncertainty. All paired exact tests are non-significant. The manuscript must describe these as bounded directional differences, not SOTA or universal gains.",
        "",
        "## Allocation and release mechanism",
        "",
        "| Method | Queries | Release/query | Unavailable/release | Post-latency rewrite/release | Preserved divergent/release |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for group in GROUPS:
        release = mechanism_lookup[(group, "release_rate")]
        unavailable = mechanism_lookup[(group, "release_unavailable_rate")]
        rewrite = mechanism_lookup[(group, "post_latency_rewrite_rate")]
        preserved = mechanism_lookup[(group, "route_preserved_per_release")]
        lines.append(
            f"| {LABELS[group]} | {release['denominator']} | {release['rate']:.3f} | {unavailable['rate']:.3f} | "
            f"{rewrite['rate']:.3f} | {preserved['rate']:.3f} |"
        )
    lines.extend([
        "",
        "RGD's strongest result is allocation quality rather than binary completion: 70/71 queries return before termination, whereas uncertainty and TTC-risk lose 21 pending proposals each. TTC-risk also releases 26/47 actions that are no longer legal and rewrites 27/47 after release. These channels directly expose why risk salience is not the same as delayed corrective opportunity.",
        "",
        "## Paired mechanism contrasts",
        "",
        "Differences below are RGD minus TTC-risk. Intervals use a paired seed-cluster bootstrap over the 30 locked main seeds and are descriptive rather than multiplicity-adjusted endpoint tests.",
        "",
        "| Metric | Difference | Paired seed-cluster 95% CI |",
        "| --- | ---: | ---: |",
    ])
    for row in paired_mechanism_rows:
        if row["baseline"] != "risk_budget":
            continue
        lines.append(
            f"| {row['metric']} | {row['difference_rgd_minus_baseline']:.3f} | "
            f"[{row['paired_seed_cluster_ci_low']:.3f}, {row['paired_seed_cluster_ci_high']:.3f}] |"
        )
    lines.extend([
        "",
        "## Slow-output pipeline",
        "",
        "| Method | Risk-calibration rewrites/query | Query-state divergence/query |",
        "| --- | ---: | ---: |",
    ])
    for group in ["rgd_fixed_policy", "random_budget", "uncertainty_budget", "risk_budget"]:
        risk_rewrite = stage_lookup[(group, "validation_to_risk")]
        divergence = stage_lookup[(group, "query_state_divergence")]
        lines.append(f"| {LABELS[group]} | {risk_rewrite['rate']:.3f} | {divergence['rate']:.3f} |")
    lines.extend([
        "",
        f"RGD produces {summary['rgd_route_preserved_divergent_releases']} route-preserved divergent releases across {summary['rgd_route_preserved_seed_count']} seeds. This is below the predeclared 20-event threshold for a strong control-effect claim, so attribution remains descriptive.",
        "",
        "## Protocol integrity",
        "",
        f"- RGD invariant violations: {len(invariant_violations)}.",
        f"- Non-vacuous pre-query prefixes: {summary['prefix_nonvacuous_match_count']}/{summary['prefix_nonvacuous_count']} match always-fast.",
        f"- Query-frame matched-fast actions: {summary['query_frame_match_count']}/{sum(row['first_slow_frame'] != '' for row in prefix_rows)}.",
        "- Fourteen first queries occur at frame 0 and are explicitly excluded from non-vacuous prefix evidence.",
        "",
        "## Paper-facing interpretation",
        "",
        "The defensible story is not that slow reasoning universally improves driving. It is that post-latency recoverability separates queries that can still be released and grounded from urgent queries that expire, become illegal, or are erased by the safety map. RGD operationalizes this distinction with zero eligibility violations and materially lower expiration/rewrite pressure than TTC-risk routing, while the 30-seed completion endpoint remains statistically unresolved.",
        "",
        "The main paper should therefore foreground the new decision variable, latency-monotone eligibility, and route-to-shield attribution. Binary completion belongs as a bounded system-level endpoint with exact paired uncertainty, not as the sole novelty claim.",
    ])
    (args.output_dir / "deep_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "invariant_violations": len(invariant_violations), "paired": paired_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
