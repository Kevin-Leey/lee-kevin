"""Generate the zero-delay policy and response-delay figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvt_figure_utils import OKABE_ITO, apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
MAIN_ANALYSIS = ROOT / "results" / "tvt_zero_delay_20260728" / "main_analysis"
LATENCY_ANALYSIS = ROOT / "results" / "tvt_zero_delay_20260728" / "latency_analysis"
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

METHODS = [
    "RGD",
    "Fast-only",
    "Always-trigger Slow",
    "Random trigger",
    "Uncertainty trigger",
    "Risk trigger",
]
BASELINES = METHODS[1:]
SHORT_METHODS = [
    "RGD",
    "Fast-only",
    "Always Slow",
    "Random",
    "Uncertainty",
    "TTC-risk",
]
DISPLAY_LABELS = {"Risk trigger": "TTC-risk trigger"}
ARMS = ["Full RGD", "w/o L", "w/o A", "w/o H"]
METHOD_COLORS = {
    "RGD": "#0072B2",
    "Fast-only": "#777777",
    "Always-trigger Slow": "#D55E00",
    "Random trigger": "#009E73",
    "Uncertainty trigger": "#56B4E9",
    "Risk trigger": "#E69F00",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def clean_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.5, width=0.6)
    axis.grid(axis=grid_axis, color="#E4E4E4", linewidth=0.45)
    axis.set_axisbelow(True)


def generate_policy_and_latency_figure(
    summary_rows: list[dict[str, str]],
    lifecycle_rows: list[dict[str, str]],
    latency_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    summary = {(row["label"], row["metric"]): row for row in summary_rows}
    latency = {
        (float(row["latency_s"]), row["metric"]): row for row in latency_rows
    }
    queries = {
        row["label"]: int(row["numerator"])
        for row in lifecycle_rows
        if row["metric"] == "attempt_rate_per_frame"
    }

    required_metrics = [
        "collision_rate",
        "success_rate",
        "avg_driving_distance",
        "avg_runtime_per_frame",
    ]
    missing = [
        (method, metric)
        for method in METHODS
        for metric in required_metrics
        if (method, metric) not in summary
    ]
    if missing:
        raise RuntimeError(f"Missing policy-summary data: {missing}")

    expected_queries = {
        "RGD": 161,
        "Fast-only": 0,
        "Always-trigger Slow": 180,
        "Random trigger": 133,
        "Uncertainty trigger": 3,
        "Risk trigger": 106,
    }
    if queries != expected_queries:
        raise RuntimeError(f"Unexpected request counts: {queries}")
    for method in METHODS:
        if float(summary[(method, "success_rate")]["mean"]) != 1.0:
            raise RuntimeError(f"Unexpected incomplete episode for {method}")
        if float(summary[(method, "collision_rate")]["mean"]) != 0.0:
            raise RuntimeError(f"Unexpected collision for {method}")

    delays = [0.0, 0.7, 1.7, 2.7]
    delay_metrics = [
        "success_rate",
        "collision_rate",
        "avg_driving_distance",
        "latency_slow_attempt_exposure_count",
    ]
    missing_delay = [
        (delay, metric)
        for delay in delays
        for metric in delay_metrics
        if (delay, metric) not in latency
    ]
    if missing_delay:
        raise RuntimeError(f"Missing delay-sweep data: {missing_delay}")
    for delay in delays:
        if int(latency[(delay, "success_rate")]["n_seeds"]) != 30:
            raise RuntimeError(f"Unexpected seed count at {delay:.1f} s")
        if float(latency[(delay, "success_rate")]["mean"]) != 1.0:
            raise RuntimeError(f"Unexpected incomplete episode at {delay:.1f} s")
        if float(latency[(delay, "collision_rate")]["mean"]) != 0.0:
            raise RuntimeError(f"Unexpected collision at {delay:.1f} s")

    delay_requests = np.array(
        [
            int(float(latency[(delay, "latency_slow_attempt_exposure_count")]["total"]))
            for delay in delays
        ],
        dtype=int,
    )
    if not np.array_equal(delay_requests, np.array([161, 162, 143, 109])):
        raise RuntimeError(f"Unexpected delay-sweep request counts: {delay_requests}")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.55),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.50},
    )

    colors = [METHOD_COLORS[method] for method in METHODS]
    method_y = np.arange(len(METHODS), dtype=float)
    query_axis = axes[0]
    query_values = [queries[method] for method in METHODS]
    query_bars = query_axis.barh(
        method_y,
        query_values,
        height=0.62,
        color=colors,
        edgecolor="#333333",
        linewidth=0.45,
    )
    for bar, value in zip(query_bars, query_values):
        query_axis.text(
            value + 4.0,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=6.6,
        )
    query_axis.set_yticks(method_y, SHORT_METHODS)
    query_axis.invert_yaxis()
    query_axis.set_xlim(0, 205)
    query_axis.set_xticks([0, 50, 100, 150, 200])
    query_axis.set_xlabel("Slow-path requests")
    query_axis.tick_params(axis="y", labelsize=6.1)
    clean_axis(query_axis)

    runtime_axis = axes[1]
    runtime_means = 1000.0 * np.array(
        [float(summary[(method, "avg_runtime_per_frame")]["mean"]) for method in METHODS]
    )
    runtime_lows = 1000.0 * np.array(
        [float(summary[(method, "avg_runtime_per_frame")]["ci_low"]) for method in METHODS]
    )
    runtime_highs = 1000.0 * np.array(
        [float(summary[(method, "avg_runtime_per_frame")]["ci_high"]) for method in METHODS]
    )
    runtime_bars = runtime_axis.barh(
        method_y,
        runtime_means,
        height=0.62,
        color=colors,
        edgecolor="#333333",
        linewidth=0.45,
        xerr=np.vstack([runtime_means - runtime_lows, runtime_highs - runtime_means]),
        error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65},
    )
    for bar, value, high in zip(runtime_bars, runtime_means, runtime_highs):
        if value >= 12.0:
            x_text, alignment, text_color = value - 0.8, "right", "white"
        else:
            x_text, alignment, text_color = high + 1.6, "left", "#222222"
        runtime_axis.text(
            x_text,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha=alignment,
            va="center",
            fontsize=6.6,
            color=text_color,
        )
    runtime_axis.set_yticks(method_y, SHORT_METHODS)
    runtime_axis.invert_yaxis()
    runtime_axis.set_xlim(0, 41)
    runtime_axis.set_xticks([0, 10, 20, 30, 40])
    runtime_axis.set_xlabel("Runtime (ms/frame)")
    runtime_axis.tick_params(axis="y", labelsize=6.1)
    clean_axis(runtime_axis)

    delay_axis = axes[2]
    delay_means = np.array(
        [float(latency[(delay, "avg_driving_distance")]["mean"]) for delay in delays]
    )
    delay_lows = np.array(
        [float(latency[(delay, "avg_driving_distance")]["ci_low"]) for delay in delays]
    )
    delay_highs = np.array(
        [float(latency[(delay, "avg_driving_distance")]["ci_high"]) for delay in delays]
    )
    delay_axis.errorbar(
        delays,
        delay_means,
        yerr=np.vstack([delay_means - delay_lows, delay_highs - delay_means]),
        color=METHOD_COLORS["RGD"],
        marker="o",
        markersize=4.0,
        markeredgecolor="#173A5E",
        markeredgewidth=0.55,
        linewidth=1.15,
        elinewidth=0.75,
        capsize=2.0,
        zorder=3,
    )
    for delay, distance, requests in zip(delays, delay_means, delay_requests):
        delay_axis.annotate(
            f"{requests} req.",
            (delay, distance),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4},
        )
    delay_axis.set_xticks(delays, ["0", "0.7", "1.7", "2.7"])
    delay_axis.set_xlim(-0.18, 2.88)
    delay_axis.set_ylim(575, 635)
    delay_axis.set_yticks([580, 590, 600, 610, 620, 630])
    delay_axis.set_xlabel("Added response delay (s)")
    delay_axis.set_ylabel("Mean distance (m)")
    clean_axis(delay_axis, grid_axis="y")

    figure.subplots_adjust(left=0.105, right=0.992, bottom=0.28, top=0.96)
    for axis, panel_label in zip(
        (query_axis, runtime_axis, delay_axis),
        (
            "(a) Slow-path requests",
            "(b) Runtime",
            "(c) Delay sweep",
        ),
    ):
        axis.text(
            0.5,
            -0.34,
            panel_label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
        )
    paths, _ = save_figure_triplet(
        figure,
        output_dir / "fig_zero_delay_latency_extension",
        crop=False,
    )
    return paths


def generate_component_figure(
    ablation_rows: list[dict[str, str]],
    by_seed_rows: list[dict[str, str]],
    verification: dict,
    output_dir: Path,
) -> list[Path]:
    indexed = {row["arm"]: row for row in ablation_rows}
    expected_counts = {
        "Full RGD": (117, 116, 4),
        "w/o L": (117, 116, 4),
        "w/o A": (120, 120, 4),
        "w/o H": (120, 120, 4),
    }
    if set(indexed) != set(expected_counts):
        raise RuntimeError(f"Unexpected component arms: {sorted(indexed)}")
    for arm, expected in expected_counts.items():
        row = indexed[arm]
        observed = (
            int(row["scheduled_queries"]),
            int(row["evaluated_releases"]),
            int(row["corrective_releases"]),
        )
        if observed != expected:
            raise RuntimeError(f"Unexpected component counts for {arm}: {observed}")

    seed_counts = {arm: set() for arm in ARMS}
    for row in by_seed_rows:
        if row["arm"] in seed_counts:
            seed_counts[row["arm"]].add(int(row["seed"]))
    if any(len(seeds) != 20 for seeds in seed_counts.values()):
        raise RuntimeError(f"Unexpected component seed coverage: {seed_counts}")

    if verification.get("accepted") is not True:
        raise RuntimeError("Component verification was not accepted")
    overlap = verification.get("panel_overlap")
    expected_overlap = {
        "w/o L": (116, 0, 0, 1.0),
        "w/o A": (49, 71, 67, 0.2620320855614973),
        "w/o H": (26, 94, 90, 0.12380952380952381),
    }
    if not isinstance(overlap, dict) or set(overlap) != set(expected_overlap):
        raise RuntimeError("Missing or unexpected release-state overlap")
    for arm, expected in expected_overlap.items():
        row = overlap[arm]
        observed = (
            int(row["intersection"]),
            int(row["arm_only"]),
            int(row["full_only"]),
            float(row["jaccard"]),
        )
        if any(abs(a - b) > 1e-10 for a, b in zip(observed, expected)):
            raise RuntimeError(f"Unexpected overlap for {arm}: {observed}")

    arm_colors = [
        "#2F5597",
        OKABE_ITO["blue"],
        OKABE_ITO["green"],
        OKABE_ITO["vermillion"],
    ]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.75),
        gridspec_kw={"width_ratios": [0.92, 1.10, 1.08], "wspace": 0.48},
    )

    query_axis = axes[0]
    arm_y = np.arange(len(ARMS), dtype=float)
    query_values = np.array([expected_counts[arm][0] for arm in ARMS], dtype=int)
    bars = query_axis.barh(
        arm_y,
        query_values,
        height=0.58,
        color=arm_colors,
        edgecolor="#333333",
        linewidth=0.45,
    )
    for bar, total in zip(bars, query_values):
        query_axis.text(
            total + 1.5,
            bar.get_y() + bar.get_height() / 2,
            str(total),
            ha="left",
            va="center",
            fontsize=6.4,
        )
    query_axis.set_yticks(arm_y, ARMS)
    query_axis.invert_yaxis()
    query_axis.set_xlim(0, 132)
    query_axis.set_xticks([0, 30, 60, 90, 120])
    query_axis.set_xlabel("Scheduled slow-path queries")
    clean_axis(query_axis)

    overlap_axis = axes[1]
    ablations = ARMS[1:]
    overlap_y = np.arange(len(ablations), dtype=float)
    shared = np.array([int(overlap[arm]["intersection"]) for arm in ablations])
    arm_only = np.array([int(overlap[arm]["arm_only"]) for arm in ablations])
    overlap_axis.barh(
        overlap_y,
        shared,
        height=0.58,
        color=OKABE_ITO["blue"],
        edgecolor="#333333",
        linewidth=0.45,
    )
    overlap_axis.barh(
        overlap_y,
        arm_only,
        left=shared,
        height=0.58,
        color="#BDBDBD",
        edgecolor="#333333",
        linewidth=0.45,
    )
    for yy, arm, shared_count, only_count, total in zip(
        overlap_y, ablations, shared, arm_only, shared + arm_only
    ):
        overlap_axis.text(
            shared_count / 2,
            yy,
            "shared",
            ha="center",
            va="center",
            fontsize=5.7,
            color="white",
        )
        if only_count:
            overlap_axis.text(
                shared_count + only_count / 2,
                yy,
                "ablation only",
                ha="center",
                va="center",
                fontsize=5.7,
                color="#222222",
            )
        overlap_axis.text(
            total + 2.0,
            yy,
            f"J={float(overlap[arm]['jaccard']):.3f}",
            ha="left",
            va="center",
            fontsize=6.2,
        )
    overlap_axis.set_yticks(overlap_y, ablations)
    overlap_axis.invert_yaxis()
    overlap_axis.set_xlim(0, 136)
    overlap_axis.set_xticks([0, 30, 60, 90, 120])
    overlap_axis.set_xlabel("Release states")
    clean_axis(overlap_axis)

    corrective_axis = axes[2]
    fractions = 100.0 * np.array(
        [float(indexed[arm]["corrective_set_fraction"]) for arm in ARMS]
    )
    lows = 100.0 * np.array([float(indexed[arm]["ci_low"]) for arm in ARMS])
    highs = 100.0 * np.array([float(indexed[arm]["ci_high"]) for arm in ARMS])
    for yy, arm, value, low, high, color in zip(
        arm_y, ARMS, fractions, lows, highs, arm_colors
    ):
        corrective_axis.hlines(yy, low, high, color=color, linewidth=1.4, zorder=2)
        corrective_axis.scatter(
            value,
            yy,
            s=27,
            marker="o",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.45,
            zorder=3,
        )
        corrective_axis.text(
            8.15,
            yy,
            f"{int(indexed[arm]['corrective_releases'])}/{int(indexed[arm]['evaluated_releases'])}",
            ha="right",
            va="center",
            fontsize=6.1,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        )
    corrective_axis.set_yticks(arm_y, ARMS)
    corrective_axis.invert_yaxis()
    corrective_axis.set_xlim(0, 8.4)
    corrective_axis.set_xticks([0, 2, 4, 6, 8])
    corrective_axis.set_xlabel("Corrective releases (%)")
    clean_axis(corrective_axis)

    figure.subplots_adjust(left=0.085, right=0.992, bottom=0.17, top=0.88)
    for axis, title in zip(
        axes,
        (
            "(a) Query admission",
            "(b) Release-state overlap",
            "(c) Corrective fraction",
        ),
    ):
        axis.set_title(title, loc="left", pad=3.5, fontsize=7.2, fontweight="bold")
    paths, _ = save_figure_triplet(
        figure, output_dir / "fig_zero_delay_component_extension"
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-summary", type=Path, default=MAIN_ANALYSIS / "main_arm_summary.csv"
    )
    parser.add_argument(
        "--lifecycle", type=Path, default=MAIN_ANALYSIS / "lifecycle_summary.csv"
    )
    parser.add_argument(
        "--latency-summary", type=Path, default=LATENCY_ANALYSIS / "latency_summary.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inputs = [
        args.main_summary,
        args.lifecycle,
        args.latency_summary,
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_tvt_style()

    outputs = generate_policy_and_latency_figure(
        read_rows(args.main_summary),
        read_rows(args.lifecycle),
        read_rows(args.latency_summary),
        args.output_dir,
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
