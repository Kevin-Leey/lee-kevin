"""Independently verify the frozen v13 matched component-ablation artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_main_table_runtime import load_formal_protocol  # noqa: E402


SYSTEM_VERSION = "action_aligned_release_gate_v13"
QUERY_GATE_VERSION = "identifiable_gate_v12"
RELEASE_CONTRACT_VERSION = "action_cost_alignment_v2"
ACTION_UNIVERSE_SOURCE = "driving_state.effective_action_universe"
SOURCE_HASH = "ddb21f275ce0379ef844ef6fb29843b8db6150e97ceabd53b284bf43148011e7"
ARMS = (
    ("Full RGD", "none"),
    ("w/o L", "latency_survival"),
    ("w/o A", "relative_support_maneuver_breadth"),
    ("w/o H", "corrective_recovery_headroom"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    require(path.is_file(), f"missing CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return dict(payload)


def close(left: Any, right: Any, message: str, tol: float = 1e-12) -> None:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    matches = (
        left_value == right_value
        if math.isinf(left_value) or math.isinf(right_value)
        else math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=tol)
    )
    require(matches, f"{message}: {left_value!r} != {right_value!r}")


def action_list(value: Any, field: str, *, canonical: bool = True) -> Tuple[int, ...]:
    try:
        actions = tuple(int(item) for item in str(value or "").split(";") if item != "")
    except ValueError as exc:
        raise ValueError(f"invalid {field}") from exc
    require(actions, f"{field} is empty")
    if canonical:
        require(actions == tuple(sorted(actions)), f"{field} is unordered")
    require(len(actions) == len(set(actions)), f"{field} contains duplicates")
    require(set(actions).issubset(set(range(5))), f"{field} contains unknown actions")
    return actions


def effective_identity(row: Mapping[str, Any]) -> Tuple[int, float]:
    target = float(row["target_speed_after"])
    require(math.isfinite(target), "nonfinite effective target speed")
    return int(row["effective_action"]), round(target, 6)


def equivalent(left: Mapping[str, Any], right: Mapping[str, Any], context: str) -> None:
    require(effective_identity(left) == effective_identity(right), f"{context}: identity drift")
    for field in ("normalized_return", "utility", "progress_m", "target_speed_after", "min_ttc"):
        close(left[field], right[field], f"{context}: {field} drift")
    for field in ("collision", "steps_completed", "effective_action", "fast_action"):
        require(int(left[field]) == int(right[field]), f"{context}: {field} drift")
    require(left["branch_trajectory_json"] == right["branch_trajectory_json"], f"{context}: trajectory drift")


def validate_branch(row: Mapping[str, Any], *, horizon: int, gamma: float) -> None:
    require(int(row["horizon_steps"]) == horizon, "branch horizon drift")
    close(row["gamma"], gamma, "branch gamma drift")
    steps = int(row["steps_completed"])
    require(0 < steps <= horizon, "invalid realized branch length")
    require(int(row["realized_steps_completed"]) == steps, "realized branch length drift")
    collision = int(row["collision"])
    require(collision in (0, 1), "invalid collision label")
    realized_return = float(row["realized_normalized_return"])
    realized_utility = float(row["realized_utility"])
    fixed_return = float(row["normalized_return"])
    fixed_utility = float(row["utility"])
    require(
        all(
            math.isfinite(float(row[field]))
            for field in ("normalized_return", "utility", "progress_m", "target_speed_after")
        ),
        "nonfinite branch metric",
    )
    close(realized_utility, realized_return - collision, "realized utility derivation drift")
    realized_weight = sum(gamma**offset for offset in range(steps))
    fixed_weight = sum(gamma**offset for offset in range(horizon))
    close(fixed_return, realized_return * realized_weight / fixed_weight, "fixed-H return derivation drift")
    close(fixed_utility, fixed_return - collision, "fixed-H utility derivation drift")
    require(int(row["return_denominator_steps"]) == horizon, "return denominator drift")
    require(row["post_terminal_reward_convention"] == "zero_increment_absorbing_state", "terminal convention drift")
    require(row["candidate_action_domain"] == "runtime_gate_action_universe", "candidate domain label drift")
    require(row["runtime_gate_action_universe_source"] == ACTION_UNIVERSE_SOURCE, "gate source drift")
    require(row["runtime_fast_action_universe_source"] == ACTION_UNIVERSE_SOURCE, "Fast source drift")
    effective = action_list(row["runtime_effective_action_universe"], "effective action universe")
    gate = action_list(row["runtime_gate_action_universe"], "gate action universe")
    legal = action_list(row["legal_actions"], "legal action universe", canonical=False)
    require(effective == gate and set(gate).issubset(set(legal)), "runtime action-domain relation drift")
    trajectory = json.loads(row["branch_trajectory_json"])
    require(isinstance(trajectory, list) and len(trajectory) == steps, "branch trajectory length drift")
    release_frame = int(row["release_frame"])
    for offset, step in enumerate(trajectory):
        require(int(step["frame"]) == release_frame + offset, "branch trajectory frame drift")


def derive_outcome(rows: Sequence[Mapping[str, Any]], epsilon: float) -> Dict[str, Any]:
    baselines = [row for row in rows if row["raw_action"] == "fast"]
    require(len(baselines) == 1 and baselines[0]["branch_role"] == "matched_fast", "matched Fast branch drift")
    baseline = baselines[0]
    gate = action_list(baseline["runtime_gate_action_universe"], "release gate universe")
    candidates = [row for row in rows if row["raw_action"] != "fast"]
    require(
        tuple(sorted(int(row["raw_action"]) for row in candidates)) == gate,
        "candidate rows do not exactly cover release gate universe",
    )
    require(all(row["branch_role"] == "candidate" for row in candidates), "candidate branch role drift")
    predicted = [row for row in candidates if int(row["raw_action"]) == int(baseline["fast_action"])]
    require(len(predicted) == 1, "forced Fast-action candidate missing")
    equivalent(baseline, predicted[0], "forced Fast-action candidate")
    baseline_identity = effective_identity(baseline)
    distinct: Dict[Tuple[int, float], Mapping[str, Any]] = {}
    for row in sorted(candidates, key=lambda item: int(item["raw_action"])):
        identity = effective_identity(row)
        if identity in distinct:
            equivalent(distinct[identity], row, f"raw-action alias {identity}")
        else:
            distinct[identity] = row
    alternatives = [row for identity, row in distinct.items() if identity != baseline_identity]
    best = max(alternatives, key=lambda item: float(item["utility"]), default=None)
    advantage = float(best["utility"]) - float(baseline["utility"]) if best else -math.inf
    return {
        "baseline": baseline,
        "best": best,
        "best_advantage": advantage,
        "distinct_alternatives": len(alternatives),
        "corrective": bool(best is not None and advantage >= epsilon),
        "gate": gate,
    }


def pooled_rate(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    arm: str,
    sampled_seeds: Sequence[int],
) -> float:
    numerator = sum(counts[arm][int(seed)][0] for seed in sampled_seeds)
    denominator = sum(counts[arm][int(seed)][1] for seed in sampled_seeds)
    require(denominator > 0, f"{arm}: undefined pooled rate")
    return numerator / denominator


def bootstrap_rate(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    arm: str,
    seeds: Sequence[int],
    *,
    draws: int,
    bootstrap_seed: int,
) -> Tuple[float, float, float]:
    point = pooled_rate(counts, arm, seeds)
    population = np.asarray(seeds, dtype=np.int64)
    rng = np.random.default_rng(bootstrap_seed)
    values = [
        pooled_rate(counts, arm, rng.choice(population, size=len(population), replace=True).tolist())
        for _ in range(draws)
    ]
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return point, float(low), float(high)


def bootstrap_difference(
    counts: Mapping[str, Mapping[int, Tuple[int, int]]],
    comparator: str,
    seeds: Sequence[int],
    *,
    draws: int,
    bootstrap_seed: int,
) -> Tuple[float, float, float]:
    point = pooled_rate(counts, "Full RGD", seeds) - pooled_rate(counts, comparator, seeds)
    population = np.asarray(seeds, dtype=np.int64)
    rng = np.random.default_rng(bootstrap_seed)
    values = []
    for _ in range(draws):
        sampled = rng.choice(population, size=len(population), replace=True).tolist()
        values.append(
            pooled_rate(counts, "Full RGD", sampled)
            - pooled_rate(counts, comparator, sampled)
        )
    low, high = np.quantile(np.asarray(values), [0.025, 0.975])
    return point, float(low), float(high)


def verify_input_hashes(manifest: Mapping[str, Any]) -> int:
    roots = {
        "frozen_trace_bundle": Path(str(manifest["trace_root"])),
        "snapshot_bundle": Path(str(manifest["snapshot_root"])),
    }
    total = 0
    inputs = dict(manifest.get("input_sha256", {}) or {})
    require(set(inputs) == set(roots), "input hash registry drift")
    for label, root in roots.items():
        for relative, expected in dict(inputs[label]).items():
            path = root / Path(relative)
            require(path.is_file(), f"missing authenticated input: {path}")
            require(sha256(path) == expected, f"authenticated input hash drift: {path}")
            total += 1
    return total


def verify(artifact: Path, protocol_path: Path) -> Dict[str, Any]:
    manifest = load_json(artifact / "component_ablation_manifest.json")
    protocol = load_formal_protocol(protocol_path)
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    contract = dict(submission.get("component_ablation", {}) or {})
    registry = dict((submission.get("evidence_artifacts", {}) or {}).get("artifacts", {}) or {})
    registry_spec = dict(registry.get("component_ablation", {}) or {})

    require(manifest.get("accepted") is True, "analysis manifest is not accepted")
    require(manifest.get("method_version") == SYSTEM_VERSION, "system version drift")
    require(manifest.get("query_gate_method_version") == QUERY_GATE_VERSION, "query-gate version drift")
    require(manifest.get("release_contract_version") == RELEASE_CONTRACT_VERSION, "release-contract version drift")
    require(manifest.get("design") == "matched four-arm leave-one-component-out", "design drift")
    require(manifest.get("source_hash") == SOURCE_HASH, "runtime source drift")
    require(sha256(protocol_path) == manifest.get("protocol_sha256"), "protocol file hash drift")
    require(Path(str(manifest["protocol"])).resolve() == protocol_path.resolve(), "protocol path drift")
    analysis_source = Path(str(manifest.get("analysis_source_path", "") or ""))
    require(analysis_source.is_file(), "analysis source missing")
    require(sha256(analysis_source) == manifest.get("analysis_source_sha256"), "analysis source hash drift")

    seed_range = dict(contract.get("seed_range", {}) or {})
    seeds = list(range(int(seed_range["start"]), int(seed_range["end"]) + 1))
    require(manifest.get("seeds") == seeds, "mechanism seed cohort drift")
    require(manifest.get("seed_is_experimental_unit") is True, "seed unit missing")
    require(manifest.get("query_events_nested_within_seed") is True, "event nesting missing")
    require(manifest.get("bootstrap_unit") == "simulator_seed", "bootstrap unit drift")
    require(manifest.get("legal_action_provenance") == "exact", "action provenance drift")
    require(manifest.get("candidate_action_domain") == "runtime_gate_action_universe", "candidate domain drift")
    require(manifest.get("claim_scope") == "descriptive mechanism analysis only", "claim scope drift")
    require(manifest.get("armwise_identical_release_state_panels") is False, "arm-panel semantics drift")
    require(manifest.get("latency_survival_floor") == "calibration_locked", "L floor status drift")
    require(manifest.get("maneuver_breadth_floor") == "calibration_locked", "A floor status drift")
    require(manifest.get("corrective_headroom_floor") == "calibration_locked", "H floor status drift")
    floors = dict(manifest.get("calibration_locked_floor_values", {}) or {})
    for key, contract_key in (
        ("latency_survival", "latency_survival_floor"),
        ("maneuver_breadth", "maneuver_breadth_floor"),
        ("corrective_headroom", "corrective_headroom_floor"),
        ("state_need", "state_need_floor"),
    ):
        close(floors.get(key), contract.get(contract_key), f"{key} floor value drift")
    close(manifest.get("state_need_floor"), contract.get("state_need_floor"), "state need floor drift")
    require(int(manifest["budget"]) == int(contract["budget"]), "budget drift")
    require(int(manifest["cooldown_minimum_query_frame_gap"]) == int(contract["cooldown_minimum_query_frame_gap"]), "cooldown gap drift")
    require(manifest.get("delay_s") == [float(contract["delay_s"][0])], "delay drift")
    horizon = int(manifest["horizon_steps"])
    gamma = float(manifest["gamma"])
    epsilon = float(manifest["epsilon"])
    require(horizon == int(contract["horizon_steps"]), "horizon drift")
    close(gamma, contract["gamma"], "gamma drift")
    close(epsilon, contract["corrective_margin"], "epsilon drift")
    draws = int(manifest["bootstrap_draws"])
    bootstrap_seed = int(manifest["bootstrap_seed"])
    require(draws == int(contract["bootstrap_draws"]), "bootstrap draw drift")
    require(bootstrap_seed == int(contract["bootstrap_seed"]), "bootstrap seed drift")
    for key, expected in dict(registry_spec.get("required_manifest_values", {}) or {}).items():
        require(manifest.get(key) == expected, f"registry requirement drift at {key}")

    output_names = list(manifest.get("outputs", []) or [])
    output_hashes = dict(manifest.get("output_sha256", {}) or {})
    require(set(output_names) == set(output_hashes), "output hash inventory drift")
    for name in output_names:
        require(sha256(artifact / name) == output_hashes[name], f"output hash drift: {name}")
    authenticated_inputs = verify_input_hashes(manifest)

    summary = read_csv(artifact / "component_ablation_summary.csv")
    events = read_csv(artifact / "component_ablation_events.csv")
    by_seed = read_csv(artifact / "component_ablation_by_seed.csv")
    effects = read_csv(artifact / "component_ablation_main_effects.csv")
    branches = read_csv(artifact / "component_ablation_branches.csv")
    accounting = read_csv(artifact / "component_ablation_selection_accounting.csv")
    expected_arms = [label for label, _ in ARMS]
    removed = dict(ARMS)
    require([row["arm"] for row in summary] == expected_arms, "summary arm order drift")

    event_keys = set()
    panels: Dict[str, set[Tuple[int, int]]] = {arm: set() for arm in expected_arms}
    for row in events:
        arm = row["arm"]
        seed = int(row["seed"])
        query = int(row["query_frame"])
        release = int(row["release_frame"])
        require(arm in expected_arms and seed in seeds, "event arm/seed drift")
        require(row["removed_component"] == removed[arm], "event removed component drift")
        require(release == query + 17, "event delay-step drift")
        require(row["candidate_state_id"] == f"{seed}:{query}:17", "candidate state ID drift")
        require(row["release_state_id"] == f"{seed}:{release}", "release state ID drift")
        key = (arm, seed, query)
        require(key not in event_keys, "duplicate ablation event")
        event_keys.add(key)
        panels[arm].add((seed, release))
        require(int(row["absolute_alternative_count"]) > 0, "event lost absolute feasibility")
        action_list(row["release_gate_action_universe"], "event release action universe")
        require(int(row["horizon_steps"]) == horizon, "event horizon drift")
        close(row["gamma"], gamma, "event gamma drift")
        close(row["epsilon"], epsilon, "event epsilon drift")

    accounting_map = {(row["arm"], int(row["seed"])): row for row in accounting}
    expected_cells = {(arm, seed) for arm in expected_arms for seed in seeds}
    require(set(accounting_map) == expected_cells, "selection accounting matrix drift")
    events_by_cell: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {
        cell: [] for cell in expected_cells
    }
    for row in events:
        events_by_cell[(row["arm"], int(row["seed"]))].append(row)
    for cell, row in accounting_map.items():
        scheduled = int(row["scheduled_count"])
        excluded = int(row["excluded_count"])
        evaluated = int(row["evaluated_count"])
        require(0 <= evaluated <= scheduled <= int(manifest["budget"]), f"{cell}: invalid selection counts")
        require(scheduled - evaluated == excluded, f"{cell}: selection accounting does not close")
        require(evaluated == len(events_by_cell[cell]), f"{cell}: event/accounting denominator drift")
        queries = sorted(int(event["query_frame"]) for event in events_by_cell[cell])
        require(
            all(right - left >= int(manifest["cooldown_minimum_query_frame_gap"]) for left, right in zip(queries, queries[1:])),
            f"{cell}: evaluated query cooldown drift",
        )

    branch_groups: Dict[Tuple[int, int], List[Mapping[str, Any]]] = {}
    branch_keys = set()
    for row in branches:
        validate_branch(row, horizon=horizon, gamma=gamma)
        key = (int(row["seed"]), int(row["release_frame"]))
        raw = row["raw_action"]
        branch_key = (*key, raw)
        require(branch_key not in branch_keys, "duplicate release-state/raw-action branch")
        branch_keys.add(branch_key)
        branch_groups.setdefault(key, []).append(row)
    required_release_states = {(int(row["seed"]), int(row["release_frame"])) for row in events}
    require(set(branch_groups) == required_release_states, "branch release-state union drift")
    require(len(branch_groups) == int(manifest["required_unique_release_states"]), "unique release-state count drift")
    outcomes = {key: derive_outcome(rows, epsilon) for key, rows in branch_groups.items()}
    for row in events:
        outcome = outcomes[(int(row["seed"]), int(row["release_frame"]))]
        baseline = outcome["baseline"]
        best = outcome["best"]
        require(int(row["corrective_set_nonempty"]) == int(outcome["corrective"]), "event corrective label drift")
        require(int(row["release_distinct_alternatives"]) == int(outcome["distinct_alternatives"]), "event distinct-alternative count drift")
        require(action_list(row["release_gate_action_universe"], "event gate universe") == outcome["gate"], "event gate universe differs from branch")
        close(row["baseline_utility"], baseline["utility"], "event baseline utility drift")
        require(int(row["baseline_collision"]) == int(baseline["collision"]), "event baseline collision drift")
        require(int(row["baseline_steps_completed"]) == int(baseline["steps_completed"]), "event baseline length drift")
        if best is None:
            require(row["best_advantage"] == "" and row["best_effective_action"] == "", "event records nonexistent best branch")
        else:
            close(row["best_advantage"], outcome["best_advantage"], "event best advantage drift")
            require(int(row["best_effective_action"]) == int(best["effective_action"]), "event best action drift")
            close(row["best_target_speed_after"], best["target_speed_after"], "event best target speed drift")
            require(int(row["best_collision"]) == int(best["collision"]), "event best collision drift")
            require(int(row["best_steps_completed"]) == int(best["steps_completed"]), "event best length drift")

    counts: Dict[str, Dict[int, Tuple[int, int]]] = {
        arm: {seed: (0, 0) for seed in seeds} for arm in expected_arms
    }
    for row in events:
        arm = row["arm"]
        seed = int(row["seed"])
        numerator, denominator = counts[arm][seed]
        counts[arm][seed] = (numerator + int(row["corrective_set_nonempty"]), denominator + 1)
    summary_map = {row["arm"]: row for row in summary}
    for arm in expected_arms:
        row = summary_map[arm]
        scheduled = sum(int(accounting_map[(arm, seed)]["scheduled_count"]) for seed in seeds)
        excluded = sum(int(accounting_map[(arm, seed)]["excluded_count"]) for seed in seeds)
        corrective = sum(counts[arm][seed][0] for seed in seeds)
        evaluated = sum(counts[arm][seed][1] for seed in seeds)
        require(int(row["scheduled_queries"]) == scheduled, f"{arm}: scheduled summary drift")
        require(int(row["excluded_queries"]) == excluded, f"{arm}: excluded summary drift")
        require(int(row["evaluated_releases"]) == evaluated, f"{arm}: evaluated summary drift")
        require(int(row["corrective_releases"]) == corrective, f"{arm}: corrective summary drift")
        point, low, high = bootstrap_rate(
            counts, arm, seeds, draws=draws, bootstrap_seed=bootstrap_seed
        )
        for field, expected in (("corrective_set_fraction", point), ("ci_low", low), ("ci_high", high)):
            close(row[field], expected, f"{arm}: {field} drift")
        require(int(row["valid_bootstrap_draws"]) == draws, f"{arm}: valid draw count drift")

    effect_map = {row["comparison"]: row for row in effects}
    require(len(effect_map) == 3, "leave-one-out effect row count drift")
    for arm in expected_arms[1:]:
        comparison = f"Full RGD - {arm}"
        point, low, high = bootstrap_difference(
            counts, arm, seeds, draws=draws, bootstrap_seed=bootstrap_seed
        )
        effect = effect_map[comparison]
        for field, expected in (("corrective_fraction_difference", point), ("ci_low", low), ("ci_high", high)):
            close(effect[field], expected, f"{comparison}: {field} drift")
        summary_row = summary_map[arm]
        close(summary_row["full_minus_arm_fraction"], point, f"{comparison}: summary point drift")
        close(summary_row["full_minus_arm_ci_low"], low, f"{comparison}: summary low drift")
        close(summary_row["full_minus_arm_ci_high"], high, f"{comparison}: summary high drift")

    by_seed_map = {(row["arm"], int(row["seed"])): row for row in by_seed}
    require(set(by_seed_map) == expected_cells, "by-seed matrix drift")
    for cell in expected_cells:
        arm, seed = cell
        row = by_seed_map[cell]
        numerator, denominator = counts[arm][seed]
        require(int(row["evaluated_releases"]) == denominator, f"{cell}: by-seed denominator drift")
        require(int(row["corrective_releases"]) == numerator, f"{cell}: by-seed numerator drift")
        if denominator:
            close(row["corrective_set_fraction"], numerator / denominator, f"{cell}: by-seed fraction drift")

    full_panel = panels["Full RGD"]
    overlaps = {}
    for arm in expected_arms[1:]:
        panel = panels[arm]
        union = full_panel | panel
        overlaps[arm] = {
            "full_states": len(full_panel),
            "arm_states": len(panel),
            "intersection": len(full_panel & panel),
            "union": len(union),
            "jaccard": len(full_panel & panel) / len(union) if union else 1.0,
            "arm_only": len(panel - full_panel),
            "full_only": len(full_panel - panel),
        }

    require(float(manifest["max_fast_replay_position_error_m"]) <= 1e-6, "snapshot position error exceeds tolerance")
    require(float(manifest["max_snapshot_speed_error_mps"]) <= 1e-6, "snapshot speed error exceeds tolerance")
    return {
        "schema_version": "rgd_v13_component_ablation_verification_v1",
        "accepted": True,
        "artifact": str(artifact.resolve()),
        "manifest_sha256": sha256(artifact / "component_ablation_manifest.json"),
        "analysis_source_sha256": manifest["analysis_source_sha256"],
        "authenticated_input_files": authenticated_inputs,
        "output_files": len(output_names),
        "seeds": seeds,
        "arms": expected_arms,
        "events": len(events),
        "unique_release_states": len(branch_groups),
        "branch_rows": len(branches),
        "bootstrap_draws": draws,
        "panel_overlap": overlaps,
        "claim_scope": manifest["claim_scope"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol_v13.yaml"))
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify(args.artifact, args.protocol)
    report_path = args.report or args.artifact / "v13_component_ablation_verification.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
