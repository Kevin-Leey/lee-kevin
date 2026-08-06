"""Analyze endpoint effects from the v13 L/A/H/N component-ablation replay.

The tool keeps two estimands separate: release-event effects are matched from
the exact saved release snapshot, whereas episode effects compare paired seed
outcomes across arms. It never substitutes one for the other.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    COMPONENT_ABLATION_ARMS,
    FACTORIAL_REPLAY_VERSION,
)
from tools.analyze_factorial_interventions import (  # noqa: E402
    DEFAULT_EPSILON,
    DEFAULT_GAMMA,
    DEFAULT_HORIZON,
    EVENT_ROW_FIELDS,
    _process_cell,
    _read_csv,
    _read_json,
    _sha256_file,
    _write_csv,
    require,
    summarize_events,
)
from tools.analyze_query_release_factorial import (  # noqa: E402
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    seed_bootstrap_indices,
)
from tools.run_v13_component_ablation import (  # noqa: E402
    COMPONENT_ABLATION_RUN_SCHEMA,
    COMPONENT_ABLATION_VERSION,
)


ANALYSIS_SCHEMA = "rgd_v13_component_ablation_intervention_analysis_v1"
EPISODE_METRICS = (
    "collision",
    "route_completion",
    "episode_reward",
    "driving_distance",
    "avg_speed",
)
PAIRWISE_EFFECTS = (
    ("full_minus_without_l", ("full", "without_l"), (1.0, -1.0)),
    ("full_minus_without_a", ("full", "without_a"), (1.0, -1.0)),
    ("full_minus_without_h", ("full", "without_h"), (1.0, -1.0)),
    ("full_minus_without_n", ("full", "without_n"), (1.0, -1.0)),
    (
        "h_x_n_interaction",
        ("full", "without_h", "without_n", "without_h_and_n"),
        (1.0, -1.0, -1.0, 1.0),
    ),
)


def _finite_metric(row: Mapping[str, Any], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"component-ablation row has invalid {field}") from exc
    if not np.isfinite(value):
        raise ValueError(f"component-ablation row has non-finite {field}")
    return value


def _load_contract(bundle: Path) -> tuple[tuple[int, ...], tuple[str, ...], list[Dict[str, Any]], Dict[str, Any]]:
    manifest = _read_json(bundle / "component_ablation_run_manifest.json")
    require(manifest.get("schema") == COMPONENT_ABLATION_RUN_SCHEMA, "component-ablation run schema drift")
    require(
        manifest.get("component_ablation_version") == COMPONENT_ABLATION_VERSION,
        "component-ablation version drift",
    )
    require(
        manifest.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
        "component-ablation replay version drift",
    )
    expected_arms = tuple(spec.name for spec in COMPONENT_ABLATION_ARMS)
    manifest_arms = tuple(str(dict(row).get("name", "")) for row in manifest.get("arms", []))
    require(manifest_arms == expected_arms, "component-ablation arm contract drift")
    seed_start = int(manifest["seed_start"])
    seed_count = int(manifest["seed_count"])
    require(seed_count > 0, "component-ablation seed cohort is empty")
    seeds = tuple(range(seed_start, seed_start + seed_count))
    rows = _read_csv(bundle / "component_ablation_episode_results.csv")
    require(
        len(rows) == len(seeds) * len(expected_arms),
        "component-ablation episode result count drift",
    )
    matrix = {(int(row["seed"]), str(row["arm"])): dict(row) for row in rows}
    expected_cells = {(seed, arm) for seed in seeds for arm in expected_arms}
    require(set(matrix) == expected_cells, "component-ablation episode matrix drift")
    proposal_hash = str(manifest.get("proposal_bank_sha256", "") or "")
    require(len(proposal_hash) == 64, "component-ablation proposal-bank hash is invalid")
    for (seed, arm), row in matrix.items():
        require(
            str(row.get("proposal_bank_sha256", "") or "") == proposal_hash,
            f"{arm}/{seed}: proposal-bank binding drift",
        )
        require(
            str(row.get("component_ablation_arm", "") or "") == arm,
            f"{arm}/{seed}: arm label drift",
        )
    return seeds, expected_arms, rows, manifest


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
    for name, effect_arms, coefficients in PAIRWISE_EFFECTS:
        require(set(effect_arms).issubset(set(arms)), f"component effect {name} has unknown arm")
        for metric in EPISODE_METRICS:
            values = np.asarray(
                [
                    sum(
                        coefficient
                        * _finite_metric(matrix[(int(seed), arm)], metric)
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.horizon <= 0 or not 0.0 < args.gamma <= 1.0 or args.epsilon < 0.0:
        raise ValueError("invalid intervention rollout parameters")
    if args.draws <= 0 or args.workers <= 0:
        raise ValueError("bootstrap draws and workers must be positive")
    bundle = Path(args.bundle).resolve()
    output_dir = Path(args.output_dir).resolve()
    seeds, arms, episode_rows, run_manifest = _load_contract(bundle)
    matrix = {(int(row["seed"]), str(row["arm"])): row for row in episode_rows}
    tasks = [
        {
            "arm": arm,
            "seed": seed,
            "seed_dir": str((bundle / arm / f"seed_{seed}").resolve()),
            "expected_releases": int(float(matrix[(seed, arm)]["release_events"])),
            "horizon": int(args.horizon),
            "gamma": float(args.gamma),
            "epsilon": float(args.epsilon),
            "legacy_v2": False,
        }
        for seed in seeds
        for arm in arms
        if int(float(matrix[(seed, arm)]["release_events"])) > 0
    ]
    event_rows: list[Dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=min(int(args.workers), len(tasks))) as pool:
            futures = [pool.submit(_process_cell, task) for task in tasks]
            for future in as_completed(futures):
                event_rows.extend(future.result())
    event_rows.sort(
        key=lambda row: (arms.index(str(row["arm"])), int(row["seed"]), int(row["release_frame"]), str(row["request_id"]))
    )
    event_summary = summarize_events(
        event_rows,
        seeds=seeds,
        draws=int(args.draws),
        bootstrap_seed=int(args.bootstrap_seed),
        arms=arms,
        allow_custom_arms=True,
    )
    episode_effects = _episode_effects(
        episode_rows,
        seeds=seeds,
        arms=arms,
        draws=int(args.draws),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "component_ablation_intervention_events.csv"
    summary_path = output_dir / "component_ablation_intervention_summary.csv"
    effects_path = output_dir / "component_ablation_episode_effects.csv"
    _write_csv(events_path, event_rows, fieldnames=EVENT_ROW_FIELDS)
    _write_csv(summary_path, event_summary)
    _write_csv(effects_path, episode_effects)
    manifest_path = output_dir / "component_ablation_intervention_manifest.json"
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "accepted": True,
        "source_bundle": str(bundle),
        "source_run_manifest_sha256": _sha256_file(
            bundle / "component_ablation_run_manifest.json"
        ),
        "proposal_bank_sha256": str(run_manifest["proposal_bank_sha256"]),
        "arms": list(arms),
        "seeds": list(seeds),
        "independent_unit": "simulator_seed",
        "release_snapshot_stage": "pre_release_frame_policy_decision",
        "branch_design": "matched_release_state_first_action_then_shared_fast_continuation",
        "horizon_steps": int(args.horizon),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "event_count": len(event_rows),
        "bootstrap": {"draws": int(args.draws), "seed": int(args.bootstrap_seed)},
        "outputs": [events_path.name, summary_path.name, effects_path.name],
        "output_sha256": {
            events_path.name: _sha256_file(events_path),
            summary_path.name: _sha256_file(summary_path),
            effects_path.name: _sha256_file(effects_path),
        },
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(manifest_path))
    print(json.dumps({"accepted": True, "events": len(event_rows), "output": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
