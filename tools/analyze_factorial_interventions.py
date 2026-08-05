"""Matched local rollouts for every factorial release event.

Each saved pre-release snapshot is branched into (i) the Fast action selected at
that release state and (ii) the returned slow action, followed by the same
deterministic Fast controller.  Traffic is interactive in both branches; only
the starting state and exogenous simulator seed are matched.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.release_snapshot import (  # noqa: E402
    RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
    ReleaseSnapshot,
    validate_release_snapshot_policy_state,
)
from tools.analyze_query_release_factorial import (  # noqa: E402
    ARM_NAMES,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    _read_csv,
    _read_json,
    _sha256_file,
    seed_bootstrap_indices,
    validate_bundle_contract,
)
from tools.analyze_release_state_rollouts import _run_branch  # noqa: E402
from tools.run_query_release_factorial import (  # noqa: E402
    _release_execution_is_distinct,
)


ANALYSIS_SCHEMA = "rgd_factorial_intervention_rollout_v1"
DEFAULT_HORIZON = 20
DEFAULT_GAMMA = 0.99
DEFAULT_EPSILON = 0.02


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _single(paths: Iterable[Path], *, context: str) -> Path:
    values = list(paths)
    require(len(values) == 1, f"{context}: expected one artifact, found {len(values)}")
    return values[0]


def _fast_continuation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    cfg["protocol_name"] = "factorial_intervention_fast_continuation"
    cfg["system_routing"] = {"simple": "fast", "complex": "fast"}
    replay = copy.deepcopy(dict(cfg.get("closed_loop_latency_replay", {}) or {}))
    replay["enable"] = False
    replay["target_systems"] = []
    cfg["closed_loop_latency_replay"] = replay
    cfg["capture_release_snapshots_online"] = False
    cfg["require_release_snapshot_on_release"] = False
    return cfg


def _load_snapshot_bundle(seed_dir: Path) -> Dict[str, ReleaseSnapshot]:
    root = seed_dir / "release_snapshots"
    manifest_path = _single(root.glob("release_snapshots_*.json"), context=str(root))
    manifest = _read_json(manifest_path)
    require(
        manifest.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
        f"{seed_dir}: snapshot bundle schema drift",
    )
    bundle_name = str(manifest.get("bundle_file", "") or "")
    require(bool(bundle_name), f"{seed_dir}: snapshot manifest omits bundle file")
    bundle_path = root / bundle_name
    require(bundle_path.is_file(), f"{seed_dir}: snapshot bundle missing")
    require(
        _sha256_file(bundle_path) == str(manifest.get("bundle_sha256", "") or ""),
        f"{seed_dir}: snapshot bundle SHA256 mismatch",
    )
    payload = pickle.loads(bundle_path.read_bytes())
    require(
        isinstance(payload, Mapping)
        and payload.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
        f"{seed_dir}: invalid snapshot pickle payload",
    )
    snapshots = dict(payload.get("snapshots", {}) or {})
    require(
        len(snapshots) == int(manifest.get("snapshot_count", -1)),
        f"{seed_dir}: snapshot count mismatch",
    )
    manifest_rows = {
        str(row.get("request_id", "") or ""): dict(row)
        for row in list(manifest.get("snapshots", []) or [])
    }
    require(set(manifest_rows) == set(snapshots), f"{seed_dir}: snapshot key drift")
    for request_id, snapshot in snapshots.items():
        require(isinstance(snapshot, ReleaseSnapshot), f"{seed_dir}: invalid snapshot type")
        validate_release_snapshot_policy_state(
            snapshot,
            context=f"{seed_dir.name} request {request_id}",
        )
        row = manifest_rows[request_id]
        require(
            str(snapshot.snapshot_identity_sha256)
            == str(row.get("snapshot_identity_sha256", "") or ""),
            f"{seed_dir}: snapshot identity manifest mismatch",
        )
    return snapshots


def _load_release_events(seed_dir: Path) -> Dict[str, Dict[str, Any]]:
    event_path = _single(
        (seed_dir / "event_logs").glob("event_log_*.json"),
        context=f"{seed_dir} event log",
    )
    payload = _read_json(event_path)
    events: Dict[str, Dict[str, Any]] = {}
    for raw in list(payload.get("events", []) or []):
        event = dict(raw)
        if not bool(event.get("closed_loop_latency_release_event", False)):
            continue
        request_id = str(event.get("closed_loop_latency_request_id", "") or "")
        require(bool(request_id), f"{seed_dir}: release event has no request ID")
        require(request_id not in events, f"{seed_dir}: duplicate release request ID")
        require(
            bool(event.get("closed_loop_release_snapshot_captured", False)),
            f"{seed_dir}: release event lacks snapshot",
        )
        events[request_id] = event
    return events


def _same_effective_action(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if int(left["effective_action"]) != int(right["effective_action"]):
        return False
    left_target = float(left["target_speed_after"])
    right_target = float(right["target_speed_after"])
    if math.isfinite(left_target) != math.isfinite(right_target):
        return False
    return not math.isfinite(left_target) or math.isclose(
        left_target, right_target, rel_tol=0.0, abs_tol=1e-6
    )


def _finite_or_blank(value: float) -> Any:
    return float(value) if math.isfinite(float(value)) else ""


def _event_rollout_row(
    *,
    arm: str,
    seed: int,
    request_id: str,
    event: Mapping[str, Any],
    snapshot: ReleaseSnapshot,
    cfg: Mapping[str, Any],
    horizon: int,
    gamma: float,
    epsilon: float,
) -> Dict[str, Any]:
    frame = int(event.get("frame", -1))
    source_frame = int(event.get("closed_loop_latency_source_frame", -1))
    require(frame == int(snapshot.frame), f"{arm}/{seed}/{request_id}: release frame drift")
    require(
        source_frame == int(snapshot.source_frame),
        f"{arm}/{seed}/{request_id}: source frame drift",
    )
    require(
        request_id == str(snapshot.request_id),
        f"{arm}/{seed}/{request_id}: snapshot request mismatch",
    )
    require(
        str(event.get("closed_loop_release_snapshot_identity_sha256", "") or "")
        == str(snapshot.snapshot_identity_sha256),
        f"{arm}/{seed}/{request_id}: event/snapshot identity mismatch",
    )

    baseline = _run_branch(snapshot, dict(cfg), seed, None, horizon, gamma)
    slow_action = int(event.get("closed_loop_released_slow_action", -1))
    require(slow_action in range(5), f"{arm}/{seed}/{request_id}: invalid slow action")
    candidate = _run_branch(snapshot, dict(cfg), seed, slow_action, horizon, gamma)

    runtime_fast = int(event.get("closed_loop_execution_state_fast_action", -1))
    executed_action = int(event.get("closed_loop_latency_executed_action", -1))
    require(
        int(baseline["fast_action"]) == runtime_fast,
        f"{arm}/{seed}/{request_id}: matched Fast proposal drift",
    )
    require(
        int(baseline["effective_action"]) == runtime_fast,
        f"{arm}/{seed}/{request_id}: matched Fast effective action drift",
    )
    rejected = bool(event.get("closed_loop_release_opportunity_rejected", False))
    unavailable = bool(event.get("closed_loop_release_action_unavailable", False))
    expected_actual = baseline if rejected or unavailable else candidate
    require(
        int(expected_actual["effective_action"]) == executed_action,
        f"{arm}/{seed}/{request_id}: branch does not reproduce executed action",
    )
    executed_distinct = bool(_release_execution_is_distinct(event))
    candidate_distinct = not _same_effective_action(baseline, candidate)
    require(
        not executed_distinct or candidate_distinct,
        f"{arm}/{seed}/{request_id}: distinct lifecycle lacks distinct branch",
    )

    utility_delta = float(candidate["utility"]) - float(baseline["utility"])
    return_delta = float(candidate["normalized_return"]) - float(
        baseline["normalized_return"]
    )
    progress_delta = float(candidate["progress_m"]) - float(baseline["progress_m"])
    collision_delta = int(candidate["collision"]) - int(baseline["collision"])
    baseline_ttc = float(baseline["min_ttc"])
    candidate_ttc = float(candidate["min_ttc"])
    ttc_delta = (
        candidate_ttc - baseline_ttc
        if math.isfinite(candidate_ttc) and math.isfinite(baseline_ttc)
        else float("nan")
    )
    if not candidate_distinct:
        classification = "effect_equivalent"
    elif utility_delta >= float(epsilon):
        classification = "beneficial"
    elif utility_delta <= -float(epsilon):
        classification = "harmful"
    else:
        classification = "neutral"
    return {
        "arm": str(arm),
        "seed": int(seed),
        "request_id": str(request_id),
        "source_frame": source_frame,
        "release_frame": frame,
        "fast_action": runtime_fast,
        "slow_action": slow_action,
        "candidate_effective_action": int(candidate["effective_action"]),
        "executed_action": executed_action,
        "release_guard_rejected": int(rejected),
        "release_action_unavailable": int(unavailable),
        "candidate_effect_distinct": int(candidate_distinct),
        "executed_distinct": int(executed_distinct),
        "classification": classification,
        "baseline_utility": float(baseline["utility"]),
        "candidate_utility": float(candidate["utility"]),
        "utility_delta": utility_delta,
        "normalized_return_delta": return_delta,
        "collision_delta": collision_delta,
        "progress_delta_m": progress_delta,
        "baseline_min_ttc_s": _finite_or_blank(baseline_ttc),
        "candidate_min_ttc_s": _finite_or_blank(candidate_ttc),
        "min_ttc_delta_s": _finite_or_blank(ttc_delta),
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
    }


def _process_cell(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    arm = str(task["arm"])
    seed = int(task["seed"])
    seed_dir = Path(str(task["seed_dir"]))
    snapshot_payload = _read_json(seed_dir / "experiment_snapshot.json")
    cfg = _fast_continuation_config(dict(snapshot_payload.get("config", {}) or {}))
    events = _load_release_events(seed_dir)
    snapshots = _load_snapshot_bundle(seed_dir)
    require(set(events) == set(snapshots), f"{arm}/{seed}: release/snapshot key mismatch")
    require(
        len(events) == int(task["expected_releases"]),
        f"{arm}/{seed}: release count disagrees with factorial row",
    )
    previous_disable = int(logging.root.manager.disable)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            logging.disable(logging.CRITICAL)
            try:
                return [
                    _event_rollout_row(
                        arm=arm,
                        seed=seed,
                        request_id=request_id,
                        event=events[request_id],
                        snapshot=snapshots[request_id],
                        cfg=cfg,
                        horizon=int(task["horizon"]),
                        gamma=float(task["gamma"]),
                        epsilon=float(task["epsilon"]),
                    )
                    for request_id in sorted(
                        events,
                        key=lambda key: (
                            int(events[key].get("frame", -1)),
                            str(key),
                        ),
                    )
                ]
            finally:
                logging.disable(previous_disable)


def _bootstrap_pooled(
    values_by_seed: Mapping[int, Sequence[float]],
    seeds: Sequence[int],
    indices: np.ndarray,
) -> Tuple[float, float, float, int]:
    pooled = [float(value) for seed in seeds for value in values_by_seed[int(seed)]]
    if not pooled:
        return float("nan"), float("nan"), float("nan"), 0
    samples: List[float] = []
    for draw in indices:
        values = [
            float(value)
            for position in draw
            for value in values_by_seed[int(seeds[int(position)])]
        ]
        if values:
            samples.append(float(np.mean(values)))
    require(bool(samples), "pooled seed bootstrap produced no valid draws")
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(np.mean(pooled)), float(low), float(high), len(samples)


def summarize_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    draws: int,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    ordered_seeds = tuple(sorted(int(seed) for seed in seeds))
    indices = seed_bootstrap_indices(
        len(ordered_seeds), draws=int(draws), bootstrap_seed=int(bootstrap_seed)
    )
    summaries: List[Dict[str, Any]] = []
    for arm in ARM_NAMES:
        arm_rows = [dict(row) for row in rows if str(row["arm"]) == arm]
        by_seed = {
            seed: [row for row in arm_rows if int(row["seed"]) == seed]
            for seed in ordered_seeds
        }
        per_seed_metrics = {
            "release_events_per_seed": [float(len(by_seed[seed])) for seed in ordered_seeds],
            "candidate_distinct_per_seed": [
                float(sum(int(row["candidate_effect_distinct"]) for row in by_seed[seed]))
                for seed in ordered_seeds
            ],
            "executed_distinct_per_seed": [
                float(sum(int(row["executed_distinct"]) for row in by_seed[seed]))
                for seed in ordered_seeds
            ],
            "selected_utility_gain_per_seed": [
                float(
                    sum(
                        float(row["utility_delta"])
                        for row in by_seed[seed]
                        if int(row["executed_distinct"])
                    )
                )
                for seed in ordered_seeds
            ],
            "rejected_beneficial_per_seed": [
                float(
                    sum(
                        str(row["classification"]) == "beneficial"
                        and int(row["release_guard_rejected"])
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
        }
        for metric, values in per_seed_metrics.items():
            samples = np.mean(np.asarray(values, dtype=float)[indices], axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
            summaries.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "estimand": "mean_per_simulator_seed",
                    "estimate": float(np.mean(values)),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "numerator": "",
                    "denominator": len(ordered_seeds),
                    "n_seed_blocks": len(ordered_seeds),
                    "bootstrap_draws": int(draws),
                    "valid_bootstrap_draws": int(draws),
                }
            )

        selected_values = {
            seed: [
                float(row["utility_delta"])
                for row in by_seed[seed]
                if int(row["executed_distinct"])
            ]
            for seed in ordered_seeds
        }
        point, low, high, valid = _bootstrap_pooled(
            selected_values, ordered_seeds, indices
        )
        selected = [row for row in arm_rows if int(row["executed_distinct"])]
        summaries.append(
            {
                "arm": arm,
                "metric": "utility_delta_per_executed_intervention",
                "estimand": "event_conditional_cluster_bootstrap",
                "estimate": _finite_or_blank(point),
                "ci_low": _finite_or_blank(low),
                "ci_high": _finite_or_blank(high),
                "numerator": "",
                "denominator": len(selected),
                "n_seed_blocks": len(ordered_seeds),
                "bootstrap_draws": int(draws),
                "valid_bootstrap_draws": int(valid),
            }
        )
        benefit_values = {
            seed: [
                float(str(row["classification"]) == "beneficial")
                for row in by_seed[seed]
                if int(row["executed_distinct"])
            ]
            for seed in ordered_seeds
        }
        point, low, high, valid = _bootstrap_pooled(
            benefit_values, ordered_seeds, indices
        )
        summaries.append(
            {
                "arm": arm,
                "metric": "beneficial_fraction_of_executed_interventions",
                "estimand": "event_conditional_cluster_bootstrap",
                "estimate": _finite_or_blank(point),
                "ci_low": _finite_or_blank(low),
                "ci_high": _finite_or_blank(high),
                "numerator": sum(
                    str(row["classification"]) == "beneficial" for row in selected
                ),
                "denominator": len(selected),
                "n_seed_blocks": len(ordered_seeds),
                "bootstrap_draws": int(draws),
                "valid_bootstrap_draws": int(valid),
            }
        )
    return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    require(args.horizon > 0, "horizon must be positive")
    require(0.0 < args.gamma <= 1.0, "gamma must lie in (0, 1]")
    require(args.epsilon >= 0.0, "epsilon must be nonnegative")
    require(args.draws > 0 and args.workers > 0, "draws and workers must be positive")
    selected_arms = tuple(dict.fromkeys(str(arm) for arm in args.arms))

    result_rows = _read_csv(args.bundle / "factorial_episode_results.csv")
    run_manifest = _read_json(args.bundle / "factorial_run_manifest.json")
    proposal_manifest = _read_json(args.bundle / "proposal_bank_manifest.json")
    contract = validate_bundle_contract(result_rows, run_manifest, proposal_manifest)
    seeds = tuple(int(seed) for seed in contract["seeds"])
    factorial_rows = {
        (int(row["seed"]), str(row["arm"])): row for row in result_rows
    }
    tasks = [
        {
            "arm": arm,
            "seed": seed,
            "seed_dir": str((args.bundle / arm / f"seed_{seed}").resolve()),
            "expected_releases": int(float(factorial_rows[(seed, arm)]["release_events"])),
            "horizon": int(args.horizon),
            "gamma": float(args.gamma),
            "epsilon": float(args.epsilon),
        }
        for seed in seeds
        for arm in selected_arms
        if int(float(factorial_rows[(seed, arm)]["release_events"])) > 0
    ]
    event_rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(_process_cell, task): task for task in tasks}
        for future in as_completed(futures):
            event_rows.extend(future.result())
    event_rows.sort(
        key=lambda row: (
            ARM_NAMES.index(str(row["arm"])),
            int(row["seed"]),
            int(row["release_frame"]),
            str(row["request_id"]),
        )
    )
    require(bool(event_rows), "selected arms contain no release events")
    summaries = summarize_events(
        event_rows,
        seeds=seeds,
        draws=int(args.draws),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "factorial_intervention_events.csv"
    summary_path = args.output_dir / "factorial_intervention_summary.csv"
    _write_csv(events_path, event_rows)
    _write_csv(summary_path, summaries)
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "accepted": True,
        "source_bundle": str(args.bundle.resolve()),
        "proposal_bank_sha256": str(contract["proposal_bank_sha256"]),
        "arms": list(selected_arms),
        "seeds": list(seeds),
        "independent_unit": "simulator_seed",
        "release_snapshot_stage": "pre_release_frame_policy_decision",
        "branch_design": "matched_release_state_first_action_then_shared_fast_continuation",
        "traffic_model": "interactive_deterministic_policy_from_cloned_release_state",
        "fixed_traffic_replay_claimed": False,
        "horizon_steps": int(args.horizon),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "event_count": len(event_rows),
        "bootstrap": {
            "unit": "simulator_seed",
            "draws": int(args.draws),
            "seed": int(args.bootstrap_seed),
        },
        "artifacts": {
            events_path.name: _sha256_file(events_path),
            summary_path.name: _sha256_file(summary_path),
        },
        "summary": summaries,
    }
    manifest_path = args.output_dir / "factorial_intervention_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(manifest_path))
    print(json.dumps({"accepted": True, "events": len(event_rows), "output": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
