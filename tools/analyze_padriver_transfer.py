"""Build PaDriver-compatible and lane-density transfer tables from raw runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


GROUP_LABELS = {
    "rgd_fixed_policy": "RGD",
    "risk_budget": "TTC-risk",
    "always_fast": "Fast-only",
}

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260718

# This order is fixed before inspecting endpoint values.  A lower tier is
# consulted only when every metric in the preceding tier is tied.  Conflicting
# directions within a tier remain Pareto-incomparable; no scalar score is used.
SAFETY_FIRST_RANKING_TIERS = (
    {
        "tier": 1,
        "name": "safety",
        "metrics": (
            {"field": "collision_rate_macro", "direction": "lower"},
            {"field": "success_rate_macro", "direction": "higher"},
        ),
    },
    {
        "tier": 2,
        "name": "mobility",
        "metrics": (
            {"field": "distance_all_episode_macro_m", "direction": "higher"},
            {"field": "speed_all_realized_frames_macro_kmh", "direction": "higher"},
        ),
    },
    {
        "tier": 3,
        "name": "resources",
        "metrics": (
            {"field": "slow_call_rate_macro", "direction": "lower"},
            {"field": "runtime_s_per_frame_macro", "direction": "lower"},
        ),
    },
)

PRIMARY_ENDPOINTS = (
    "success_rate",
    "collision_rate",
    "distance_all_episode_m",
    "speed_all_realized_frames_kmh",
)

TRANSFER_CELLS = tuple(
    (lanes, density)
    for lanes in (4, 5, 6)
    for density in (2.0, 3.0)
)

# The transfer protocol fixes three policies, six lane-density cells, and one
# matched episode for each of thirty seeds.
EXPECTED_TRANSFER_KEYS = frozenset(
    (group, int(lanes), float(density), int(seed))
    for group in GROUP_LABELS
    for lanes, density in TRANSFER_CELLS
    for seed in range(30)
)

PADRIVER_ROWS = [
    {
        "evaluation": "PADriver (normal)",
        "distance_m": 603.0,
        "speed_kmh": 72.47,
        "safe_distance_rate": 0.91,
        "keep_rate": 0.92,
        "success_count": 25,
        "runtime_s_per_frame": 1.4,
    },
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_source_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate the complete fixed transfer factorial before aggregation."""
    source_rows = [dict(row) for row in rows]
    keys = {
        (
            str(row.get("group", "")),
            int(row.get("transfer_lanes_count")),
            float(row.get("transfer_vehicles_density")),
            int(row.get("seed_idx")),
        )
        for row in source_rows
    }
    hashes = {str(row.get("source_hash", "")).strip() for row in source_rows}
    if len(hashes) != 1 or not next(iter(hashes), ""):
        raise RuntimeError("cannot mix executable source hashes in transfer inputs")
    if keys != EXPECTED_TRANSFER_KEYS:
        raise RuntimeError(
            "expected the complete 540-run factorial: "
            f"observed {len(keys)} unique transfer keys"
        )
    return {
        "source_rows": len(source_rows),
        "observed_unique_keys": len(keys),
        "overlapping_rows": len(source_rows) - len(keys),
        "source_hash": next(iter(hashes)),
        "source_hash_row_count": len(source_rows),
    }


def load_physical_frames(result_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(result_dir.rglob("*physical_frames.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one physical-frame file under {result_dir}, found {len(files)}")
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"empty physical-frame trace: {files[0]}")
    return frames


def load_events(result_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in sorted(result_dir.rglob("event_log_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        events.extend(payload.get("events", []) or [])
    return events


def per_episode_metrics(row: Dict[str, str]) -> Dict[str, Any]:
    result_dir = Path(row["result_dir"])
    frames = load_physical_frames(result_dir)
    events = load_events(result_dir)
    manifest_path = result_dir / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    manifest_config = dict(manifest.get("config", {}) or {})
    manifest_replay = dict(manifest_config.get("closed_loop_latency_replay", {}) or {})
    safe_flags = [fnum(frame.get("closest_vehicle_distance"), float("inf")) >= 5.0 for frame in frames]
    keep_flags = [int(fnum(frame.get("action_id"), -1)) == 1 for frame in frames]
    speeds = [fnum(frame.get("speed")) * 3.6 for frame in frames]
    initial_forward = sorted(
        fnum(neighbor.get("longitudinal"))
        for neighbor in list(frames[0].get("neighbor_snapshots", []) or [])
        if fnum(neighbor.get("longitudinal")) > 0.0
    )
    initial_forward_gaps = [
        initial_forward[index + 1] - initial_forward[index]
        for index in range(len(initial_forward) - 1)
    ]
    counts = Counter()
    counts["queries"] = sum(str(event.get("system_used")) == "slow" for event in events)
    counts["immediate_returns"] = sum(
        str(event.get("system_used")) == "slow"
        and bool(event.get("llm_backed_execution_available", False))
        and not str(event.get("slow_reasoning_failure_reason", "") or "")
        for event in events
    )
    counts["releases"] = sum(bool(event.get("closed_loop_latency_release_event")) for event in events)
    counts["unavailable"] = sum(bool(event.get("closed_loop_release_action_unavailable")) for event in events)
    counts["rewritten"] = sum(bool(event.get("closed_loop_post_latency_shield_rewrite")) for event in events)
    counts["divergent"] = sum(bool(event.get("closed_loop_release_route_divergence")) for event in events)
    counts["preserved"] = sum(bool(event.get("closed_loop_route_preserved_divergent_release")) for event in events)
    counts["slow_fallback"] = sum(str(event.get("system_used")) == "fast_after_slow_failure" for event in events)
    replay_enabled = any(bool(frame.get("closed_loop_latency_replay_enabled")) for frame in frames)
    replay_delay_positive = any(fnum(frame.get("closed_loop_latency_extra_s")) > 0.0 for frame in frames)
    bridge_cfg = dict(manifest_config.get("hidden_slower_bridge", {}) or {})
    collision = int(fnum(row.get("collision_rate")) > 0.5)
    return {
        "group": row["group"],
        "seed": int(fnum(row.get("seed_idx"))),
        "lanes_count": int(fnum(row.get("transfer_lanes_count"))),
        "vehicles_density": fnum(row.get("transfer_vehicles_density")),
        "success": int(fnum(row.get("success_rate")) > 0.5),
        "collision": collision,
        "distance_m": fnum(row.get("avg_driving_distance")),
        "speed_kmh": mean(speeds),
        "speed_sum_kmh": sum(speeds),
        "safe_distance_rate": mean(safe_flags),
        "safe_distance_frames": sum(safe_flags),
        "keep_rate": mean(keep_flags),
        "keep_frames": sum(keep_flags),
        "runtime_s_per_frame": fnum(row.get("avg_runtime_per_frame")),
        "slow_call_rate": fnum(row.get("slow_call_rate")),
        "frames": len(frames),
        "collision_events_per_1000_frames": 1000.0 * collision / len(frames),
        "first_step_collision": int(collision and len(frames) <= 2),
        "result_dir": result_dir.as_posix(),
        "vehicle_count": int(fnum(row.get("transfer_vehicle_count"))),
        "observed_max_lane_id": max(int(fnum(frame.get("lane_id"), -1)) for frame in frames),
        "replay_enabled": replay_enabled,
        "replay_delay_positive": replay_delay_positive,
        "initial_nearest_forward_m": initial_forward[0] if initial_forward else float("nan"),
        "initial_forward_gap_mean_m": mean(initial_forward_gaps) if initial_forward_gaps else float("nan"),
        "manifest_lanes_count": int(fnum(manifest_config.get("lanes_count"), -1)),
        "manifest_vehicles_density": fnum(manifest_config.get("vehicles_density"), -1.0),
        "manifest_extra_latency_s": fnum(manifest_replay.get("extra_latency_s"), -1.0),
        "hidden_slower_bridge": int(bool(bridge_cfg.get("enable", False))),
        "target_speed_min_mps": fnum(manifest_config.get("target_speed_min"), -1.0),
        **counts,
    }


def episode_key(row: Dict[str, Any]) -> Tuple[str, int, float, int]:
    return (
        str(row["group"]),
        int(row["lanes_count"]),
        float(row["vehicles_density"]),
        int(row["seed"]),
    )


def deduplicate_episodes(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Merge execution shards while failing on conflicting deterministic results."""
    canonical: Dict[Tuple[str, int, float, int], Dict[str, Any]] = {}
    duplicate_count = 0
    nondeterministic_fields = {
        "runtime_s_per_frame",
        "result_dir",
        "manifest_lanes_count",
        "manifest_vehicles_density",
        "manifest_extra_latency_s",
    }
    for row in rows:
        key = episode_key(row)
        previous = canonical.get(key)
        if previous is None:
            canonical[key] = row
            continue
        duplicate_count += 1
        for field, value in row.items():
            if field in nondeterministic_fields or field in {"group", "seed", "lanes_count", "vehicles_density"}:
                continue
            prior = previous[field]
            if isinstance(value, float) or isinstance(prior, float):
                if not math.isclose(float(value), float(prior), rel_tol=1e-9, abs_tol=1e-9):
                    raise RuntimeError(f"conflicting duplicate {key} for {field}: {prior!r} != {value!r}")
            elif value != prior:
                raise RuntimeError(f"conflicting duplicate {key} for {field}: {prior!r} != {value!r}")
    return [canonical[key] for key in sorted(canonical)], duplicate_count


def exact_two_sided_binomial(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    observed = math.comb(n, wins) / (2.0**n)
    return min(
        1.0,
        sum(
            math.comb(n, k) / (2.0**n)
            for k in range(n + 1)
            if math.comb(n, k) / (2.0**n) <= observed + 1e-15
        ),
    )


def percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated sample percentile without extra dependencies."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"invalid percentile probability: {probability}")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_interval(values: Sequence[float]) -> Tuple[float, float]:
    return percentile(values, 0.025), percentile(values, 0.975)


def pooled_realized_frame_speed(rows: Sequence[Dict[str, Any]]) -> float:
    frames = sum(int(row["frames"]) for row in rows)
    if frames <= 0:
        raise ValueError("realized-frame speed requires at least one frame")
    return sum(float(row["speed_sum_kmh"]) for row in rows) / frames


def endpoint_value(rows: Sequence[Dict[str, Any]], metric: str) -> float:
    """Compute one endpoint using its pre-specified analysis population."""
    if not rows:
        raise ValueError(f"cannot compute {metric} from no episodes")
    if metric == "success_rate":
        return mean(float(row["success"]) for row in rows)
    if metric == "collision_rate":
        return mean(float(row["collision"]) for row in rows)
    if metric == "distance_all_episode_m":
        return mean(float(row["distance_m"]) for row in rows)
    if metric == "speed_all_realized_frames_kmh":
        return pooled_realized_frame_speed(rows)
    raise KeyError(f"unknown endpoint: {metric}")


def paired_bootstrap_difference(
    target: Sequence[Dict[str, Any]],
    control: Sequence[Dict[str, Any]],
    metric: str,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap a method contrast while retaining matched seed clusters."""
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    if len(target) != len(control) or not target:
        raise ValueError("paired bootstrap requires equally sized nonempty samples")
    target_by_seed = {int(row["seed"]): row for row in target}
    control_by_seed = {int(row["seed"]): row for row in control}
    seeds = sorted(target_by_seed)
    if seeds != sorted(control_by_seed) or len(seeds) != len(target):
        raise ValueError("paired bootstrap requires one matched row per seed")
    point = endpoint_value(target, metric) - endpoint_value(control, metric)
    rng = random.Random(seed)
    estimates: List[float] = []
    for _ in range(draws):
        sampled = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_target = [target_by_seed[value] for value in sampled]
        sampled_control = [control_by_seed[value] for value in sampled]
        estimates.append(
            endpoint_value(sampled_target, metric)
            - endpoint_value(sampled_control, metric)
        )
    low, high = bootstrap_interval(estimates)
    return point, low, high


def summarize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, float], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["group"], row["lanes_count"], row["vehicles_density"])].append(row)
    output: List[Dict[str, Any]] = []
    for (group, lanes, density), selected in sorted(grouped.items()):
        totals = Counter()
        for row in selected:
            for key in ("queries", "immediate_returns", "releases", "unavailable", "rewritten", "divergent", "preserved", "slow_fallback"):
                totals[key] += int(row[key])
        if any(row["replay_enabled"] or row["replay_delay_positive"] for row in selected):
            raise RuntimeError(f"zero-added-latency transfer contains replayed actions for {group}, lanes={lanes}, density={density}")
        if any(int(row["observed_max_lane_id"]) >= lanes for row in selected):
            raise RuntimeError(f"lane index exceeds configured lane count for {group}, lanes={lanes}, density={density}")
        output.append(
            {
                "group": group,
                "label": GROUP_LABELS.get(group, group),
                "lanes_count": lanes,
                "vehicles_density": density,
                "seeds": len(selected),
                "distance_all_episode_m": endpoint_value(selected, "distance_all_episode_m"),
                "speed_all_realized_frames_kmh": endpoint_value(selected, "speed_all_realized_frames_kmh"),
                "safe_distance_all_realized_frames_rate": (
                    sum(int(row["safe_distance_frames"]) for row in selected)
                    / sum(int(row["frames"]) for row in selected)
                ),
                "keep_all_realized_frames_rate": (
                    sum(int(row["keep_frames"]) for row in selected)
                    / sum(int(row["frames"]) for row in selected)
                ),
                "collisions": sum(row["collision"] for row in selected),
                "collision_rate": mean(row["collision"] for row in selected),
                "collision_events_per_1000_frames": sum(row["collision"] for row in selected) / sum(row["frames"] for row in selected) * 1000.0,
                "first_step_collisions": sum(row["first_step_collision"] for row in selected),
                "success_count": sum(row["success"] for row in selected),
                "success_rate": mean(row["success"] for row in selected),
                "runtime_s_per_frame": (
                    sum(float(row["runtime_s_per_frame"]) * int(row["frames"]) for row in selected)
                    / sum(int(row["frames"]) for row in selected)
                ),
                "slow_call_rate": totals["queries"] / sum(int(row["frames"]) for row in selected),
                "slow_fallback_events": totals["slow_fallback"],
                "observed_max_lane_id": max(int(row["observed_max_lane_id"]) for row in selected),
                "queries": totals["queries"],
                "immediate_return_per_query": totals["immediate_returns"] / totals["queries"] if totals["queries"] else 0.0,
                "release_per_query": totals["releases"] / totals["queries"] if totals["queries"] else 0.0,
                "unavailable_per_release": totals["unavailable"] / totals["releases"] if totals["releases"] else 0.0,
                "rewrite_per_release": totals["rewritten"] / totals["releases"] if totals["releases"] else 0.0,
                "preserved_per_query": totals["preserved"] / totals["queries"] if totals["queries"] else 0.0,
            }
        )
    return output


def _paired_seed_ids(
    rows: Sequence[Dict[str, Any]],
    seed_ids: Iterable[int] | None,
) -> List[int]:
    if seed_ids is not None:
        seeds = sorted(int(seed) for seed in seed_ids)
    else:
        seeds = sorted(
            int(row["seed"])
            for row in rows
            if row["group"] == "rgd_fixed_policy"
            and int(row["lanes_count"]) == TRANSFER_CELLS[0][0]
            and math.isclose(float(row["vehicles_density"]), TRANSFER_CELLS[0][1])
        )
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed IDs must be a nonempty unique set")
    return seeds


def paired_endpoints(
    rows: Sequence[Dict[str, Any]],
    *,
    seed_ids: Iterable[int] | None = None,
    draws: int = BOOTSTRAP_DRAWS,
) -> List[Dict[str, Any]]:
    """Return within-cell, seed-paired RGD contrasts for all primary endpoints."""
    by_key = {(row["group"], row["lanes_count"], row["vehicles_density"], row["seed"]): row for row in rows}
    seeds = _paired_seed_ids(rows, seed_ids)
    output: List[Dict[str, Any]] = []
    comparison_index = 0
    for lanes, density in TRANSFER_CELLS:
        for baseline in ("risk_budget", "always_fast"):
            target = [by_key[("rgd_fixed_policy", lanes, density, seed)] for seed in seeds]
            control = [by_key[(baseline, lanes, density, seed)] for seed in seeds]
            success_pairs = [(row_t["success"], row_c["success"]) for row_t, row_c in zip(target, control)]
            wins = sum(value_t > value_c for value_t, value_c in success_pairs)
            losses = sum(value_t < value_c for value_t, value_c in success_pairs)
            result: Dict[str, Any] = {
                "lanes_count": lanes,
                "vehicles_density": density,
                "baseline": baseline,
                "baseline_label": GROUP_LABELS[baseline],
                "n": len(success_pairs),
                "wins": wins,
                "losses": losses,
                "ties": len(success_pairs) - wins - losses,
                "exact_discordant_p": exact_two_sided_binomial(wins, losses),
                "cluster_unit": "seed_within_cell",
                "bootstrap_draws": draws,
                "ci_method": "paired_seed_percentile_bootstrap_95",
            }
            for metric_index, metric in enumerate(PRIMARY_ENDPOINTS):
                point, low, high = paired_bootstrap_difference(
                    target,
                    control,
                    metric,
                    draws=draws,
                    seed=BOOTSTRAP_SEED + comparison_index * len(PRIMARY_ENDPOINTS) + metric_index,
                )
                stem = {
                    "success_rate": "paired_success_difference",
                    "collision_rate": "paired_collision_difference",
                    "distance_all_episode_m": "paired_distance_all_episode_difference_m",
                    "speed_all_realized_frames_kmh": "paired_speed_all_realized_frames_difference_kmh",
                }[metric]
                result[stem] = point
                result[f"{stem}_ci_low"] = low
                result[f"{stem}_ci_high"] = high
            output.append(result)
            comparison_index += 1
    return output


def _cell_rows(
    by_key: Dict[Tuple[str, int, float, int], Dict[str, Any]],
    group: str,
    cell: Tuple[int, float],
    sampled_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    lanes, density = cell
    return [by_key[(group, lanes, density, seed)] for seed in sampled_seeds]


def _macro_endpoint(
    by_key: Dict[Tuple[str, int, float, int], Dict[str, Any]],
    group: str,
    metric: str,
    sampled_seeds: Sequence[int],
) -> float:
    # Cells receive equal weight; frame weighting is confined within each cell.
    return mean(
        endpoint_value(_cell_rows(by_key, group, cell, sampled_seeds), metric)
        for cell in TRANSFER_CELLS
    )


def macro_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    seed_ids: Iterable[int] | None = None,
    draws: int = BOOTSTRAP_DRAWS,
) -> List[Dict[str, Any]]:
    """Summarize six cells equally and bootstrap seed clusters across all cells."""
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    by_key = {(row["group"], row["lanes_count"], row["vehicles_density"], row["seed"]): row for row in rows}
    seeds = _paired_seed_ids(rows, seed_ids)
    output: List[Dict[str, Any]] = []
    for group_index, group in enumerate(GROUP_LABELS):
        result: Dict[str, Any] = {
            "group": group,
            "label": GROUP_LABELS[group],
            "cells": len(TRANSFER_CELLS),
            "seed_clusters": len(seeds),
            "episodes": len(TRANSFER_CELLS) * len(seeds),
            "cluster_unit": "seed_spanning_all_six_cells",
            "cell_weighting": "equal_macro",
            "bootstrap_draws": draws,
            "ci_method": "seed_cluster_percentile_bootstrap_95",
        }
        for metric_index, metric in enumerate(PRIMARY_ENDPOINTS):
            point = _macro_endpoint(by_key, group, metric, seeds)
            rng = random.Random(BOOTSTRAP_SEED + 10_000 + group_index * len(PRIMARY_ENDPOINTS) + metric_index)
            estimates = []
            for _ in range(draws):
                sampled = [seeds[rng.randrange(len(seeds))] for _ in seeds]
                estimates.append(_macro_endpoint(by_key, group, metric, sampled))
            low, high = bootstrap_interval(estimates)
            field = {
                "success_rate": "success_rate_macro",
                "collision_rate": "collision_rate_macro",
                "distance_all_episode_m": "distance_all_episode_macro_m",
                "speed_all_realized_frames_kmh": "speed_all_realized_frames_macro_kmh",
            }[metric]
            result[field] = point
            result[f"{field}_ci_low"] = low
            result[f"{field}_ci_high"] = high
        result["slow_call_rate_macro"] = mean(
            sum(int(row["queries"]) for row in _cell_rows(by_key, group, cell, seeds))
            / sum(int(row["frames"]) for row in _cell_rows(by_key, group, cell, seeds))
            for cell in TRANSFER_CELLS
        )
        result["runtime_s_per_frame_macro"] = mean(
            sum(
                float(row["runtime_s_per_frame"]) * int(row["frames"])
                for row in _cell_rows(by_key, group, cell, seeds)
            )
            / sum(int(row["frames"]) for row in _cell_rows(by_key, group, cell, seeds))
            for cell in TRANSFER_CELLS
        )
        output.append(result)
    return output


def macro_paired_endpoints(
    rows: Sequence[Dict[str, Any]],
    *,
    seed_ids: Iterable[int] | None = None,
    draws: int = BOOTSTRAP_DRAWS,
) -> List[Dict[str, Any]]:
    """Return paired RGD contrasts with each seed retained as a six-cell cluster."""
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    by_key = {(row["group"], row["lanes_count"], row["vehicles_density"], row["seed"]): row for row in rows}
    seeds = _paired_seed_ids(rows, seed_ids)
    output: List[Dict[str, Any]] = []
    for baseline_index, baseline in enumerate(("risk_budget", "always_fast")):
        for metric_index, metric in enumerate(PRIMARY_ENDPOINTS):
            point = (
                _macro_endpoint(by_key, "rgd_fixed_policy", metric, seeds)
                - _macro_endpoint(by_key, baseline, metric, seeds)
            )
            rng = random.Random(BOOTSTRAP_SEED + 20_000 + baseline_index * len(PRIMARY_ENDPOINTS) + metric_index)
            estimates = []
            for _ in range(draws):
                sampled = [seeds[rng.randrange(len(seeds))] for _ in seeds]
                estimates.append(
                    _macro_endpoint(by_key, "rgd_fixed_policy", metric, sampled)
                    - _macro_endpoint(by_key, baseline, metric, sampled)
                )
            low, high = bootstrap_interval(estimates)
            output.append(
                {
                    "baseline": baseline,
                    "baseline_label": GROUP_LABELS[baseline],
                    "metric": metric,
                    "difference_rgd_minus_baseline": point,
                    "paired_cluster_ci_low": low,
                    "paired_cluster_ci_high": high,
                    "favors_rgd_when_difference_is": "negative" if metric == "collision_rate" else "positive",
                    "cells": len(TRANSFER_CELLS),
                    "seed_clusters": len(seeds),
                    "cluster_unit": "seed_spanning_all_six_cells",
                    "cell_weighting": "equal_macro",
                    "bootstrap_draws": draws,
                    "ci_method": "paired_seed_cluster_percentile_bootstrap_95",
                }
            )
    return output


def safety_first_pairwise_ranking(macro_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the fixed tiered Pareto rule without constructing a weighted score."""
    output: List[Dict[str, Any]] = []
    for left_index, left in enumerate(macro_rows):
        for right in macro_rows[left_index + 1 :]:
            relation = "tied_all_tiers"
            decisive_tier: int | str = ""
            decisive_name = ""
            for tier in SAFETY_FIRST_RANKING_TIERS:
                left_better = False
                right_better = False
                for spec in tier["metrics"]:
                    left_value = float(left[spec["field"]])
                    right_value = float(right[spec["field"]])
                    if math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12):
                        continue
                    if (spec["direction"] == "higher" and left_value > right_value) or (
                        spec["direction"] == "lower" and left_value < right_value
                    ):
                        left_better = True
                    else:
                        right_better = True
                if left_better and right_better:
                    relation = f"pareto_incomparable_at_tier_{tier['tier']}"
                elif left_better:
                    relation = f"left_dominates_at_tier_{tier['tier']}"
                elif right_better:
                    relation = f"right_dominates_at_tier_{tier['tier']}"
                else:
                    continue
                decisive_tier = tier["tier"]
                decisive_name = tier["name"]
                break
            output.append(
                {
                    "left_group": left["group"],
                    "left_label": left["label"],
                    "right_group": right["group"],
                    "right_label": right["label"],
                    "relation": relation,
                    "decisive_tier": decisive_tier,
                    "decisive_tier_name": decisive_name,
                    "weighted_score_used": False,
                }
            )
    return output


def build_padriver_table(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target = [
        row
        for row in episodes
        if row["group"] == "rgd_fixed_policy"
        and row["lanes_count"] == 4
        and row["vehicles_density"] == 2.0
    ]
    successful = [row for row in target if row["success"]]
    if not target or not successful:
        raise RuntimeError("PaDriver-format comparison requires successful RGD 4-lane/density-2 episodes")
    successful_frames = sum(int(row["frames"]) for row in successful)
    rows: List[Dict[str, Any]] = []
    for source in PADRIVER_ROWS:
        rows.append(
            {
                **source,
                "collision_rate": "",
                "source": "PADriver Table 1; collision rate not reported and not inferred from non-completion",
                "analysis_population": "PADriver-reported normal-setting convention",
            }
        )
    rows.append(
        {
            "evaluation": "RGD (ours)",
            "distance_m": mean(row["distance_m"] for row in successful),
            "speed_kmh": pooled_realized_frame_speed(successful),
            "safe_distance_rate": sum(int(row["safe_distance_frames"]) for row in successful) / successful_frames,
            "keep_rate": sum(int(row["keep_frames"]) for row in successful) / successful_frames,
            "success_count": sum(row["success"] for row in target),
            "runtime_s_per_frame": (
                sum(float(row["runtime_s_per_frame"]) * int(row["frames"]) for row in target)
                / sum(int(row["frames"]) for row in target)
            ),
            "collision_rate": mean(row["collision"] for row in target),
            "source": "Current zero-added-latency 4-lane/density-2.0 run, seeds 0-29",
            "analysis_population": "successful episodes for Dis./Spe./Saf./Kep.; all episodes for Coll./Suc./Runtime",
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-draws",
        type=int,
        default=BOOTSTRAP_DRAWS,
        help=(
            "Number of percentile-bootstrap draws used for derived confidence intervals. "
            f"The publication protocol default is {BOOTSTRAP_DRAWS}."
        ),
    )
    args = parser.parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap-draws must be positive")

    source_rows: List[Dict[str, str]] = []
    input_paths: List[Path] = []
    inputs: List[Dict[str, str]] = []
    for bundle in args.bundle:
        path = bundle / "padriver_transfer_sweep_rows.csv"
        rows = read_csv(path)
        source_rows.extend(rows)
        input_paths.append(path)
        inputs.append(
            {
                "path": (Path("canonical_input") / path.name).as_posix(),
                "sha256": sha256(path),
                "rows": len(rows),
            }
        )
    source_audit = audit_source_rows(source_rows)
    expected = len(EXPECTED_TRANSFER_KEYS)
    loaded_episodes = [per_episode_metrics(row) for row in source_rows]
    episodes, duplicate_rows = deduplicate_episodes(loaded_episodes)
    keys = {episode_key(row) for row in episodes}
    expected_keys = EXPECTED_TRANSFER_KEYS
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)[:5]
        unexpected = sorted(keys - expected_keys)[:5]
        raise RuntimeError(
            f"expected the complete {expected}-run factorial, found {len(keys)} unique keys "
            f"after loading {len(source_rows)} source rows; missing={missing}, unexpected={unexpected}"
        )

    summary = summarize(episodes)
    paired = paired_endpoints(episodes, seed_ids=range(30), draws=args.bootstrap_draws)
    macro = macro_summary(episodes, seed_ids=range(30), draws=args.bootstrap_draws)
    macro_paired = macro_paired_endpoints(episodes, seed_ids=range(30), draws=args.bootstrap_draws)
    safety_ranking = safety_first_pairwise_ranking(macro)
    comparison = build_padriver_table(episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = args.output_dir / "canonical_input"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    for index, source_path in enumerate(input_paths):
        destination_name = (
            source_path.name
            if len(input_paths) == 1
            else f"{index:02d}_{source_path.name}"
        )
        destination = canonical_dir / destination_name
        shutil.copy2(source_path, destination)
        inputs[index]["path"] = destination.relative_to(args.output_dir).as_posix()
        inputs[index]["sha256"] = sha256(destination)
    compact_episodes = [
        {key: value for key, value in row.items() if key != "result_dir"}
        for row in episodes
    ]
    episode_path = args.output_dir / "lane_density_episode_metrics.csv"
    write_csv(episode_path, compact_episodes)
    write_csv(args.output_dir / "lane_density_transfer_summary.csv", summary)
    write_csv(args.output_dir / "lane_density_paired_endpoints.csv", paired)
    write_csv(args.output_dir / "lane_density_macro_summary.csv", macro)
    write_csv(args.output_dir / "lane_density_macro_paired_endpoints.csv", macro_paired)
    write_csv(args.output_dir / "safety_first_pairwise_ranking.csv", safety_ranking)
    write_csv(args.output_dir / "padriver_style_comparison.csv", comparison)
    effective_audit = [
        {
            "group": row["group"],
            "lanes_count": row["lanes_count"],
            "vehicles_density": row["vehicles_density"],
            "seed": row["seed"],
            "vehicle_count": row["vehicle_count"],
            "added_latency_s": 0.0,
            "observed_max_lane_id": row["observed_max_lane_id"],
            "replay_enabled_in_frames": row["replay_enabled"],
            "positive_replay_delay_in_frames": row["replay_delay_positive"],
            "slow_fallback_events": row["slow_fallback"],
            "source_hash": row["source_hash"],
            "protocol_hash": row["protocol_hash"],
            "config_hash": row["config_hash"],
            "transfer_scope": row["transfer_scope"],
            "hidden_slower_bridge": row["hidden_slower_bridge"],
            "target_speed_min_mps": row["target_speed_min_mps"],
            "legacy_manifest_lanes_count": row["manifest_lanes_count"],
            "legacy_manifest_vehicles_density": row["manifest_vehicles_density"],
            "legacy_manifest_extra_latency_s": row["manifest_extra_latency_s"],
            "legacy_manifest_matches_transfer_overrides": (
                row["manifest_lanes_count"] == row["lanes_count"]
                and math.isclose(row["manifest_vehicles_density"], row["vehicles_density"])
                and math.isclose(row["manifest_extra_latency_s"], 0.0)
            ),
            "result_dir": row["result_dir"],
        }
        for row in episodes
    ]
    write_csv(args.output_dir / "effective_transfer_config_audit.csv", effective_audit)
    spacing_rows: List[Dict[str, Any]] = []
    fast_episodes = [row for row in episodes if row["group"] == "always_fast"]
    for lanes in (4, 5, 6):
        setting_values: Dict[float, Tuple[float, float]] = {}
        for density in (2.0, 3.0):
            selected = [
                row for row in fast_episodes
                if row["lanes_count"] == lanes and math.isclose(row["vehicles_density"], density)
            ]
            setting_values[density] = (
                median(row["initial_nearest_forward_m"] for row in selected),
                median(row["initial_forward_gap_mean_m"] for row in selected),
            )
        gap_ratio = setting_values[2.0][1] / setting_values[3.0][1]
        for density in (2.0, 3.0):
            nearest, gap = setting_values[density]
            spacing_rows.append(
                {
                    "lanes_count": lanes,
                    "vehicles_density": density,
                    "episodes": 30,
                    "initial_nearest_forward_median_m": nearest,
                    "initial_forward_gap_mean_median_m": gap,
                    "density_times_gap_median": density * gap,
                    "gap_ratio_density2_over_density3": gap_ratio,
                    "source": "frame-zero physical neighbor snapshots from Fast-only matched seeds",
                }
            )
    write_csv(args.output_dir / "traffic_spacing_audit.csv", spacing_rows)

    manifest = {
        "analysis": "padriver_compatible_lane_density_transfer_v3",
        "inputs": inputs,
        "compact_episode_input": {
            "path": episode_path.name,
            "sha256": sha256(episode_path),
            "rows": len(compact_episodes),
        },
        "run_count": len(episodes),
        "source_row_count": len(source_rows),
        "deduplicated_execution_rows": duplicate_rows,
        "source_provenance_audit": source_audit,
        "groups": list(GROUP_LABELS),
        "lanes": [4, 5, 6],
        "densities": [2.0, 3.0],
        "seeds": [0, 29],
        "added_latency_s": 0.0,
        "effective_config_audit": "effective_transfer_config_audit.csv",
        "traffic_spacing_audit": "traffic_spacing_audit.csv",
        "statistical_outputs": {
            "cell_summary": "lane_density_transfer_summary.csv",
            "within_cell_seed_paired_contrasts": "lane_density_paired_endpoints.csv",
            "six_cell_macro_summary": "lane_density_macro_summary.csv",
            "six_cell_seed_cluster_paired_contrasts": "lane_density_macro_paired_endpoints.csv",
            "safety_first_pairwise_ranking": "safety_first_pairwise_ranking.csv",
        },
        "legacy_snapshot_note": (
            "The execution runner applied lane/density/latency overrides after constructing its embedded "
            "protocol snapshot. The audit retains those legacy fields, verifies lanes and zero replay from "
            "physical frames, and independently checks the density override through frame-zero spacing."
        ),
        "padriver_table_columns": ["Evaluation", "Dis.", "Spe.", "Saf.", "Kep.", "Coll.", "Suc.", "Runtime"],
        "padriver_density_column_included": False,
        "metric_contract": {
            "table_vii_distance_all_episode_m": "arithmetic mean episode distance over the complete 30-seed denominator, including failed episodes",
            "table_vii_speed_all_realized_frames_kmh": "pooled ego speed over every realized physical frame from all 30 episodes, converted from m/s",
            "table_vii_safe_distance_all_realized_frames_rate": "pooled fraction of all realized frames with closest_vehicle_distance >= 5 m",
            "table_vii_keep_all_realized_frames_rate": "pooled fraction of all realized frames with action_id=1 (IDLE/KEEP)",
            "padriver_distance_speed_safe_keep": "successful-episode subset, used only in padriver_style_comparison.csv to preserve that table's reporting convention",
            "collision_rate": "observed collision episodes / 30 for current runs; not inferred for PADriver",
            "runtime_s_per_frame": "pooled runtime seconds per realized frame over all episodes",
        },
        "inference_contract": {
            "within_cell": "paired seed bootstrap; methods retain the same resampled seed IDs within each lane-density cell",
            "across_cells": "equal-weight macro average over the six fixed cells; bootstrap cluster is seed and retains all six cells for each resampled seed ID",
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_interval": "two-sided 95% percentile bootstrap",
        },
        "ranking_contract": {
            "complete_factorial_required": True,
            "tiers": SAFETY_FIRST_RANKING_TIERS,
            "rule": "Consult the next tier only when all metrics in the preceding tier tie; conflicting directions within a tier are Pareto-incomparable.",
            "weighted_score": None,
            "post_hoc_weighting_permitted": False,
        },
    }
    (args.output_dir / "lane_density_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(episodes)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
