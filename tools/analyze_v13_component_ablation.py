"""Fixed-horizon analysis helpers for v13 component ablations."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Arm:
    label: str
    use_l: bool
    use_a: bool
    use_h: bool


ARMS = (
    Arm("Full RGD", True, True, True),
    Arm("w/o L", False, True, True),
    Arm("w/o A", True, False, True),
    Arm("w/o H", True, True, False),
)


def _discount_sum(gamma: float, steps: int) -> float:
    return sum(float(gamma) ** index for index in range(max(0, int(steps))))


def normalize_fixed_horizon_branch(
    row: Mapping[str, Any], *, horizon: int, gamma: float
) -> dict[str, Any]:
    """Normalize a realized branch against the shared fixed horizon.

    Terminal states contribute zero future increment, so a branch ending early
    cannot gain an advantage from a shorter denominator.  Collision remains a
    binary penalty independent of the realized rollout length.
    """
    if horizon <= 0 or not 0.0 < gamma <= 1.0:
        raise ValueError("horizon and gamma must define a valid discount schedule")
    trajectory = json.loads(str(row.get("branch_trajectory_json", "[]")) or "[]")
    if not isinstance(trajectory, list):
        raise ValueError("branch trajectory must be a JSON list")
    realized_steps = min(int(row.get("steps_completed", len(trajectory))), len(trajectory), int(horizon))
    if realized_steps < 0:
        raise ValueError("steps_completed must be nonnegative")
    realized_return = float(row.get("normalized_return", row.get("utility", 0.0)))
    denominator = _discount_sum(float(gamma), int(horizon))
    numerator = _discount_sum(float(gamma), realized_steps)
    normalized_return = realized_return * numerator / denominator if denominator else 0.0
    collision = int(bool(row.get("collision", False)))
    output = dict(row)
    output.update(
        {
            "realized_normalized_return": realized_return,
            "normalized_return": normalized_return,
            "utility": normalized_return - collision,
            "return_denominator_steps": int(horizon),
            "post_terminal_reward_convention": "zero_increment_absorbing_state",
        }
    )
    return output


def _gate_domain(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    domains = {
        tuple(
            int(token)
            for token in str(row.get("runtime_gate_action_universe", "")).split(";")
            if token != ""
        )
        for row in rows
    }
    if len(domains) != 1 or not domains or not next(iter(domains)):
        raise ValueError("branches must declare one nonempty gate action domain")
    return set(next(iter(domains)))


def branch_outcome(rows: Sequence[Mapping[str, Any]], epsilon: float) -> dict[str, Any]:
    baseline_rows = [row for row in rows if str(row.get("raw_action")) == "fast"]
    if len(baseline_rows) != 1:
        raise ValueError("expected exactly one fast branch")
    baseline = dict(baseline_rows[0])
    candidates = [dict(row) for row in rows if str(row.get("raw_action")) != "fast"]
    expected_actions = _gate_domain(rows)
    raw_actions = {int(row["raw_action"]) for row in candidates}
    if raw_actions != expected_actions:
        raise ValueError("candidate raw actions must exactly cover gate domain")
    effective_to_raw: dict[int, int] = {}
    for candidate in candidates:
        raw = int(candidate["raw_action"])
        effective = int(candidate["effective_action"])
        previous = effective_to_raw.get(effective)
        if previous is not None and previous != raw:
            raise ValueError("raw-action alias maps multiple candidates to one effective action")
        effective_to_raw[effective] = raw
    baseline_effective = int(baseline["effective_action"])
    alternatives = [
        candidate
        for candidate in candidates
        if int(candidate["effective_action"]) != baseline_effective
    ]
    best = max(alternatives, key=lambda candidate: float(candidate["utility"]), default=None)
    advantage = float(best["utility"]) - float(baseline["utility"]) if best else float("-inf")
    return {
        "baseline": baseline,
        "best_row": best,
        "best_advantage": advantage,
        "distinct_alternatives": len(alternatives),
        "corrective": bool(best is not None and advantage >= float(epsilon)),
    }


def _pooled_rate(
    counts: Mapping[str, Mapping[int, tuple[int, int]]], arm: str, seeds: Sequence[int]
) -> float:
    numerator = denominator = 0
    for seed in seeds:
        current_numerator, current_denominator = counts[str(arm)].get(int(seed), (0, 0))
        numerator += int(current_numerator)
        denominator += int(current_denominator)
    if denominator <= 0:
        raise ValueError(f"{arm} has no evaluated releases")
    return numerator / denominator


def bootstrap_rate(
    counts: Mapping[str, Mapping[int, tuple[int, int]]],
    arm: str,
    seeds: Sequence[int],
    *,
    draws: int,
    bootstrap_seed: int,
) -> tuple[float, float, float, int]:
    point = _pooled_rate(counts, arm, seeds)
    if draws <= 0 or not seeds:
        raise ValueError("bootstrap requires positive draws and a nonempty seed cohort")
    rng = np.random.default_rng(int(bootstrap_seed))
    population = np.asarray([int(seed) for seed in seeds])
    samples = [
        _pooled_rate(counts, arm, rng.choice(population, len(population), replace=True).tolist())
        for _ in range(int(draws))
    ]
    low, high = np.quantile(np.asarray(samples), (0.025, 0.975))
    return float(point), float(low), float(high), len(samples)


def summarize(
    events: Sequence[Mapping[str, Any]],
    accounting: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    draws: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    event_index: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for event in events:
        event_index.setdefault((str(event["arm"]), int(event["seed"])), []).append(event)
    accounting_index = {(str(row["arm"]), int(row["seed"])): row for row in accounting}
    counts: dict[str, dict[int, tuple[int, int]]] = {}
    by_seed: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_counts: dict[int, tuple[int, int]] = {}
        for seed in seeds:
            key = (arm.label, int(seed))
            if key not in accounting_index:
                raise ValueError(f"missing selection accounting for {key}")
            selected = event_index.get(key, [])
            expected = int(accounting_index[key]["evaluated_count"])
            if len(selected) != expected:
                raise ValueError("event/accounting denominator drift")
            numerator = sum(int(event.get("corrective_set_nonempty", 0)) for event in selected)
            arm_counts[int(seed)] = (numerator, expected)
            by_seed.append(
                {
                    "arm": arm.label,
                    "seed": int(seed),
                    "evaluated_releases": expected,
                    "corrective_releases": numerator,
                    "corrective_set_fraction": numerator / expected if expected else "",
                }
            )
        counts[arm.label] = arm_counts
    summary = []
    for arm in ARMS:
        point, low, high, valid = bootstrap_rate(
            counts, arm.label, seeds, draws=draws, bootstrap_seed=bootstrap_seed
        )
        summary.append(
            {
                "arm": arm.label,
                "corrective_set_fraction": point,
                "ci_low": low,
                "ci_high": high,
                "bootstrap_valid_draws": valid,
            }
        )
    return summary, by_seed
