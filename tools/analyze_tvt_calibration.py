"""Reconstruct frozen allocator cutoffs from the 30 calibration trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, List, Sequence, Tuple


BUDGET = 6
COOLDOWN = 20
DELAY_STEPS = 17
UNCERTAINTY_CUTOFF = 1.00


def stable_unit_interval(*items: Any) -> float:
    digest = hashlib.md5("|".join(str(item) for item in items).encode("utf-8")).hexdigest()
    return float(int(digest[:8], 16) / 0xFFFFFFFF)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(record: Dict[str, Any]) -> Dict[str, Any]:
    return dict(record.get("rgd_subordinate_diagnostics", {}) or {})


def load_frames(fast_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in range(30, 60):
        patterns = (
            f"always_fast_latency_1p7s_seed_{seed}/ep_{seed}/highway_{seed}_reasoning_records.json",
            f"always_fast/highway/seed_{seed}/ep_{seed}/highway_{seed}_reasoning_records.json",
        )
        matches = [path for pattern in patterns for path in fast_root.glob(pattern)]
        if len(matches) != 1:
            raise RuntimeError(f"expected one calibration reasoning file for seed {seed}, found {len(matches)}")
        payload = json.loads(matches[0].read_text(encoding="utf-8-sig"))
        records = list(payload.get("analysis_records", []) or [])
        episode_frames = len(records)
        for record in records:
            frame = int(record.get("frame_id", 0) or 0)
            diag = diagnostics(record)
            gate = dict((diag.get("recoverability_signal", {}) or {}).get("recoverability_gate", {}) or {})
            ambiguity = dict(diag.get("ambiguity_and_conflict", {}) or {})
            baseline = dict(diag.get("baseline_trigger_scores", {}) or {})
            entropy = float(ambiguity.get("route_ambiguity_entropy", 0.0) or 0.0)
            disagreement = float(ambiguity.get("route_ambiguity_disagreement", 0.0) or 0.0)
            rows.append(
                {
                    "seed": seed,
                    "frame": frame,
                    "episode_frames": episode_frames,
                    "query_progress": frame / max(1, episode_frames),
                    "within_final_17_frames": frame >= max(0, episode_frames - DELAY_STEPS),
                    "rgd_opportunity_eligible": bool(gate.get("opportunity_eligible", False)),
                    "rgd_priority": float(gate.get("score", 0.0) or 0.0),
                    "absolute_alternative_feasible": bool(gate.get("absolute_alternative_feasible", False)),
                    "alternative_metric_source": str(gate.get("alternative_metric_source", "unknown") or "unknown"),
                    "headroom_metric_source": str(gate.get("headroom_metric_source", "unknown") or "unknown"),
                    "viable_cost_threshold": float(gate.get("viable_cost_threshold", 0.0) or 0.0),
                    "uncertainty_score": max(entropy, disagreement),
                    "ttc_score": float(baseline.get("ttc_route_score", 0.0) or 0.0),
                    "random_draw": stable_unit_interval("random_budget_v2", seed, frame),
                    "uncertainty_exposure_draw": stable_unit_interval("baseline_exposure_v1", "uncertainty", seed, frame),
                }
            )
    return rows


def selected_rows(
    rows: Sequence[Dict[str, Any]],
    predicate: Callable[[Dict[str, Any]], bool],
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for seed in range(30, 60):
        last = -10_000
        calls = 0
        for row in (item for item in rows if int(item["seed"]) == seed):
            frame = int(row["frame"])
            if calls >= BUDGET or frame - last < COOLDOWN or not predicate(row):
                continue
            selected.append(row)
            last = frame
            calls += 1
    return selected


def exposure_summary(name: str, parameter: float, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    progresses = [float(row["query_progress"]) for row in rows]
    frames = [int(row["frame"]) for row in rows]
    return {
        "allocator": name,
        "selected_parameter": parameter,
        "calibration_calls": len(rows),
        "calls_per_seed": len(rows) / 30.0,
        "median_query_frame": median(frames) if frames else "",
        "median_query_progress": median(progresses) if progresses else "",
        "final_17_frame_calls": sum(bool(row["within_final_17_frames"]) for row in rows),
        "budget": BUDGET,
        "cooldown_frames": COOLDOWN,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    frames = load_frames(args.fast_root.resolve())
    if {row["alternative_metric_source"] for row in frames} != {"action_support_ranking_costs"}:
        raise RuntimeError("calibration traces do not use the support-breadth A definition")
    if {row["headroom_metric_source"] for row in frames} != {"action_recovery_costs"}:
        raise RuntimeError("calibration traces do not use raw-cost H")
    if any(abs(float(row["viable_cost_threshold"]) - 0.55) > 1e-12 for row in frames):
        raise RuntimeError("calibration trace cost threshold drift")
    grid = [index / 100.0 for index in range(101)]

    rgd_grid: List[Tuple[float, List[Dict[str, Any]]]] = [
        (
            value,
            selected_rows(frames, lambda row, value=value: bool(row["rgd_opportunity_eligible"]) and float(row["rgd_priority"]) >= value),
        )
        for value in grid
    ]
    maximum_eligible_exposure = max(len(rows) for _, rows in rgd_grid)
    rgd_candidates = [item for item in rgd_grid if len(item[1]) >= 0.98 * maximum_eligible_exposure]
    rgd_parameter, rgd_selected = max(rgd_candidates, key=lambda item: item[0])
    target_calls = len(rgd_selected)

    random_grid = [
        (value, selected_rows(frames, lambda row, value=value: float(row["random_draw"]) <= value))
        for value in grid
    ]
    random_parameter, random_selected = min(
        (item for item in random_grid if len(item[1]) >= target_calls),
        key=lambda item: item[0],
    )

    uncertainty_grid = [
        (
            value,
            selected_rows(
                frames,
                lambda row, value=value: float(row["uncertainty_score"]) >= UNCERTAINTY_CUTOFF
                and float(row["uncertainty_exposure_draw"]) <= value,
            ),
        )
        for value in grid
    ]
    uncertainty_parameter, uncertainty_selected = min(
        (item for item in uncertainty_grid if len(item[1]) >= target_calls),
        key=lambda item: item[0],
    )

    ttc_grid = [
        (value, selected_rows(frames, lambda row, value=value: float(row["ttc_score"]) >= value))
        for value in grid
    ]
    ttc_parameter, ttc_selected = max(
        (item for item in ttc_grid if len(item[1]) >= target_calls),
        key=lambda item: item[0],
    )

    selection_rows = [
        {
            **exposure_summary("RGD", rgd_parameter, rgd_selected),
            "parameter_name": "priority_threshold",
            "selection_rule": "highest_0.01_threshold_retaining_at_least_98pct_of_maximum_eligible_exposure",
            "target_calls": target_calls,
            "maximum_eligible_exposure": maximum_eligible_exposure,
        },
        {
            **exposure_summary("Random", random_parameter, random_selected),
            "parameter_name": "per_frame_probability",
            "selection_rule": "smallest_0.01_probability_reaching_RGD_target_calls",
            "target_calls": target_calls,
            "maximum_eligible_exposure": "",
        },
        {
            **exposure_summary("Uncertainty", uncertainty_parameter, uncertainty_selected),
            "parameter_name": "exposure_probability_after_uncertainty_cutoff_1.00",
            "selection_rule": "smallest_0.01_probability_reaching_RGD_target_calls",
            "target_calls": target_calls,
            "maximum_eligible_exposure": "",
        },
        {
            **exposure_summary("TTC-risk", ttc_parameter, ttc_selected),
            "parameter_name": "ttc_score_cutoff",
            "selection_rule": "largest_0.01_cutoff_reaching_RGD_target_calls",
            "target_calls": target_calls,
            "maximum_eligible_exposure": "",
        },
    ]

    ttc_sensitivity: List[Dict[str, Any]] = []
    for value, selected in ttc_grid:
        if round(value, 2) not in {round(ttc_parameter - 0.02, 2), round(ttc_parameter - 0.01, 2), round(ttc_parameter, 2), round(ttc_parameter + 0.01, 2), round(ttc_parameter + 0.02, 2)}:
            continue
        ttc_sensitivity.append(
            {
                **exposure_summary("TTC-risk", value, selected),
                "is_selected": value == ttc_parameter,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "calibration_frame_scores.csv", frames)
    write_csv(args.output_dir / "calibration_selection.csv", selection_rows)
    write_csv(args.output_dir / "ttc_cutoff_sensitivity.csv", ttc_sensitivity)
    (args.output_dir / "calibration_summary.json").write_text(
        json.dumps(
            {
                "seeds": list(range(30, 60)),
                "grid_step": 0.01,
                "selection": selection_rows,
                "ttc_local_sensitivity": ttc_sensitivity,
                "source_boundary": "complete Fast-only calibration trajectories; no main seed used",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"target_calls": target_calls, "selection": selection_rows}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
