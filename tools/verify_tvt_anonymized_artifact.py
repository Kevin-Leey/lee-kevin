"""Verify the compact, anonymized TVT artifact without raw model transcripts."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


GROUPS = {
    "RGD": "rgd_fixed_policy",
    "Fast-only": "always_fast",
    "Random": "random_budget",
    "Uncertainty": "uncertainty_budget",
    "TTC-risk": "risk_budget",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(actual: str, expected: float, *, tol: float = 1e-9) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tol)


def verify(root: Path) -> None:
    analysis = root / "analysis"

    results = {row["group"]: row for row in read_rows(analysis / "main_results.csv")}
    expected_completion = {
        "RGD": 6,
        "Fast-only": 5,
        "Random": 6,
        "Uncertainty": 6,
        "TTC-risk": 5,
    }
    for label, successes in expected_completion.items():
        row = results[GROUPS[label]]
        require(int(row["episodes"]) == 30, f"{label}: expected 30 episodes")
        require(int(row["successes"]) == successes, f"{label}: completion mismatch")

    mechanism_rows = read_rows(analysis / "mechanism_attribution.csv")
    mechanism = {(row["group"], row["metric"]): row for row in mechanism_rows}
    expected_lifecycle = {
        "rgd_fixed_policy": (70, 71, 7, 10, 29, 19),
        "random_budget": (75, 81, 13, 17, 34, 17),
        "uncertainty_budget": (55, 76, 6, 11, 23, 12),
        "risk_budget": (47, 68, 26, 27, 35, 8),
    }
    metric_names = (
        "release_rate",
        "release_unavailable_rate",
        "post_latency_rewrite_rate",
        "execution_state_divergence_rate",
        "route_preserved_per_release",
    )
    for group, (released, queries, unavailable, rewrite, divergent, preserved) in expected_lifecycle.items():
        expected = ((released, queries), (unavailable, released), (rewrite, released),
                    (divergent, released), (preserved, released))
        for metric, (numerator, denominator) in zip(metric_names, expected):
            row = mechanism[(group, metric)]
            require(int(row["numerator"]) == numerator, f"{group}/{metric}: numerator mismatch")
            require(int(row["denominator"]) == denominator, f"{group}/{metric}: denominator mismatch")

    timing = {row["group"]: row for row in read_rows(analysis / "query_timing_summary.csv")}
    expected_timing = {
        "rgd_fixed_policy": (71, 25.0, 0.15671641791044777, 1),
        "random_budget": (81, 71.0, 0.47333333333333333, 6),
        "uncertainty_budget": (76, 76.0, 0.6166666666666667, 21),
        "risk_budget": (68, 64.0, 0.6382995821279331, 21),
    }
    for group, (queries, frame, progress, final_window) in expected_timing.items():
        row = timing[group]
        require(int(row["queries"]) == queries, f"{group}: query count mismatch")
        require(close(row["median_query_frame"], frame), f"{group}: median frame mismatch")
        require(close(row["median_query_progress"], progress), f"{group}: median progress mismatch")
        require(int(row["final_17_frame_queries"]) == final_window, f"{group}: final-window mismatch")

    components = {row["component"]: row for row in read_rows(analysis / "rgd_proxy_component_summary.csv")}
    require(int(components["legal_alternative_ratio"]["unique_values"]) == 1,
            "RGD selected calls should have a constant legal-alternative ratio")
    require(close(components["legal_alternative_ratio"]["median"], 1.0),
            "RGD legal-alternative median mismatch")
    require(close(components["opportunity"]["minimum"], 0.20136150961478633),
            "RGD opportunity minimum mismatch")
    require(close(components["opportunity"]["maximum"], 0.5292934112111307),
            "RGD opportunity maximum mismatch")

    strata = read_rows(analysis / "rgd_proxy_outcome_strata.csv")
    require([int(row["release_per_query_numerator"]) for row in strata] == [24, 22, 24],
            "Selected-E release numerators mismatch")
    require([int(row["preserved_per_query_numerator"]) for row in strata] == [7, 5, 7],
            "Selected-E preservation numerators mismatch")

    trace = read_rows(analysis / "lifecycle_event_trace.csv")
    required_columns = {
        "group", "seed", "frame", "event_kind", "source_frame",
        "source_frame_recorded", "source_frame_repaired", "gate_delay_steps",
        "replay_delay_steps", "matched_fast_pending", "opportunity",
        "opportunity_floor", "release_unavailable", "post_latency_rewrite",
        "release_divergence", "route_preserved_divergence", "episode_done",
        "terminal_cause",
    }
    require(required_columns.issubset(trace[0]), "Sanitized trace schema is incomplete")
    require(len(trace) == 543, "Sanitized trace row count mismatch")
    counts = Counter((row["group"], row["event_kind"]) for row in trace)
    for group, (released, queries, *_rest) in expected_lifecycle.items():
        require(counts[(group, "query")] == queries, f"{group}: trace query count mismatch")
        require(counts[(group, "release")] == released, f"{group}: trace release count mismatch")
    require(sum(row["source_frame_repaired"] == "True" for row in trace) == 20,
            "Expected 20 explicitly marked historical frame-zero repairs")

    print("PASS: compact TVT artifact tables and trace schema are internally consistent.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    verify(args.artifact_root.resolve())


if __name__ == "__main__":
    main()
