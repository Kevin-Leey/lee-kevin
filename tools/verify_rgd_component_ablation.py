"""Derivational checks for the RGD component-ablation artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_common_trajectory_allocators import RGD_FLOOR, RGD_THRESHOLD
from tools.analyze_rgd_component_ablation import ARM_SPECS, _branch_outcome


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: Any, right: Any, message: str, tol: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{message}: {left!r} != {right!r}")


def event_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["seed"]), int(row["release_frame"])


def verify(root: Path, source_analysis: Path) -> None:
    manifest = json.loads(
        (root / "component_ablation_manifest.json").read_text(encoding="utf-8")
    )
    events = read_csv(root / "component_ablation_events.csv")
    branches = read_csv(root / "component_ablation_branches.csv")
    accounting = read_csv(root / "component_ablation_selection_accounting.csv")
    summary = read_csv(root / "component_ablation_summary.csv")
    by_seed = read_csv(root / "component_ablation_by_seed.csv")
    effects = read_csv(root / "component_ablation_main_effects.csv")

    seeds = [int(value) for value in manifest["seeds"]]
    require(seeds == list(range(160, 190)), "unexpected ablation seed block")
    require(manifest["seed_is_experimental_unit"] is True, "seed unit not declared")
    require(manifest["query_events_nested_within_seed"] is True, "query nesting omitted")
    require(len(ARM_SPECS) == 8, "factorial design must contain eight cells")
    require(
        len({(spec.use_l, spec.use_a, spec.use_h) for spec in ARM_SPECS}) == 8,
        "factorial masks are not unique",
    )
    require(len(accounting) == len(seeds) * len(ARM_SPECS), "accounting grid incomplete")
    require(len(by_seed) == len(seeds) * len(ARM_SPECS), "per-seed grid incomplete")
    require(len(summary) == len(ARM_SPECS), "summary arm count mismatch")
    require({row["component"] for row in effects} == {"L", "A", "H"}, "main effects missing")

    spec_by_label = {spec.label: spec for spec in ARM_SPECS}
    for row in events:
        spec = spec_by_label[row["arm"]]
        l_value = float(row["latency_survival"]) if spec.use_l else 1.0
        a_value = float(row["admissible_alternative_fraction"]) if spec.use_a else 1.0
        h_value = float(row["recovery_headroom"]) if spec.use_h else 1.0
        opportunity = l_value * math.sqrt(max(0.0, a_value * h_value))
        priority = opportunity * float(row["need_score"])
        close(row["ablated_latency_survival"], l_value, "L neutralization drift")
        close(row["ablated_admissible_alternative_fraction"], a_value, "A neutralization drift")
        close(row["ablated_recovery_headroom"], h_value, "H neutralization drift")
        close(row["ablated_opportunity"], opportunity, "opportunity drift")
        close(row["ablated_priority"], priority, "priority drift")
        require(opportunity + 1e-12 >= RGD_FLOOR, "selected event fails opportunity floor")
        require(priority + 1e-12 >= RGD_THRESHOLD, "selected event fails priority threshold")
        if spec.use_a:
            require(int(row["alternative_count"]) >= 1, "active A hard gate bypassed")

    branch_groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in branches:
        branch_groups.setdefault(event_key(row), []).append(row)
    require(set(map(event_key, events)) <= set(branch_groups), "event lacks action branches")
    epsilon = float(manifest["epsilon"])
    outcomes = {
        key: _branch_outcome(rows, epsilon) for key, rows in branch_groups.items()
    }
    for row in events:
        outcome = outcomes[event_key(row)]
        require(
            int(row["corrective_set_nonempty"]) == int(outcome["corrective"]),
            "corrective label does not derive from branches",
        )
        require(
            int(row["release_distinct_alternatives"]) == int(outcome["distinct_alternatives"]),
            "distinct-alternative count does not derive from branches",
        )

    summary_by_arm = {row["arm"]: row for row in summary}
    for spec in ARM_SPECS:
        arm_rows = [row for row in events if row["arm"] == spec.label]
        acc_rows = [row for row in accounting if row["arm"] == spec.label]
        out = summary_by_arm[spec.label]
        scheduled = sum(int(row["scheduled_count"]) for row in acc_rows)
        evaluated = sum(int(row["evaluated_count"]) for row in acc_rows)
        corrective = sum(int(row["corrective_set_nonempty"]) for row in arm_rows)
        require(int(out["scheduled_queries"]) == scheduled, "scheduled Q summary drift")
        require(int(out["evaluated_releases"]) == evaluated == len(arm_rows), "release summary drift")
        require(int(out["corrective_releases"]) == corrective, "corrective summary drift")
        close(
            out["corrective_set_fraction"],
            corrective / evaluated,
            "corrective-set fraction drift",
        )

    source_events = read_csv(source_analysis / "release_rollout_events.csv")
    source_rgd = {
        (int(row["seed"]), int(row["query_frame"])): int(row["corrective_set_nonempty"])
        for row in source_events
        if row["allocator"] == "RGD" and math.isclose(float(row["delay_s"]), 1.7)
    }
    ablation_rgd = {
        (int(row["seed"]), int(row["query_frame"])): int(row["corrective_set_nonempty"])
        for row in events
        if row["arm"] == "RGD"
    }
    require(ablation_rgd == source_rgd, "full RGD arm does not reproduce locked source events")

    print(
        "PASS: eight RGD component cells, seed-cluster accounting, gate scores, "
        "matched-action labels, summaries, and the locked full-RGD arm are derivationally verified."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-analysis", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    verify(args.artifact, args.source_analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
