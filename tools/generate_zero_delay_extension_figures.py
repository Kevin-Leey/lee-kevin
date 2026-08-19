"""Generate the additive paper-facing formal main and component figures.

The figures are deliberately derived from the locked formal-analysis tables.
They visualize task endpoints and resource/selectivity endpoints separately so
that a request-efficiency result is not confused with a closed-loop gain.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvt_figure_utils import OKABE_ITO, apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
MAIN_ANALYSIS = ROOT / "results" / "highway_result" / "formal_run" / "main_analysis"
COMPONENT_ANALYSIS = (
    ROOT
    / "results"
    / "highway_result"
    / "formal_run"
    / "component_ablation_analysis"
)
COMPONENT_BUNDLE = (
    ROOT
    / "results"
    / "highway_result"
    / "formal_run"
    / "component_ablation"
)
GUARDED_INTERVENTIONS = (
    ROOT
    / "results"
    / "highway_result"
    / "formal_run"
    / "query_release_factorial_interventions"
)
NO_GUARD_INTERVENTIONS = (
    ROOT
    / "results"
    / "highway_result"
    / "formal_run"
    / "query_release_factorial_interventions_no_guard"
)
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

MAIN_GROUPS = [
    "rgd_fixed_policy",
    "always_fast",
    "always_slow",
    "random_budget",
    "uncertainty_budget",
    "risk_budget",
]
MAIN_LABELS = ["RGD", "Fast-only", "Always Slow", "Random", "Uncertainty", "Risk"]
MAIN_COLORS = [
    "#0072B2",
    "#777777",
    "#D55E00",
    "#009E73",
    "#56B4E9",
    "#E69F00",
]
COMPONENT_ARMS = [
    "full",
    "without_l",
    "without_a",
    "without_h",
    "without_n",
    "without_h_and_n",
]
COMPONENT_LABELS = ["Full", "w/o L", "w/o A", "w/o H", "w/o N", "w/o H/N"]
COMPONENT_COLORS = [
    "#2F5597",
    OKABE_ITO["blue"],
    OKABE_ITO["green"],
    OKABE_ITO["vermillion"],
    "#CC79A7",
    "#8C8C8C",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def clean_axis(axis: plt.Axes, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.5, width=0.6)
    axis.grid(axis=grid_axis, color="#E4E4E4", linewidth=0.45)
    axis.set_axisbelow(True)


def _mean_ci(rows: dict[str, dict[str, str]], group: str, metric: str) -> tuple[float, float, float]:
    row = rows[group]
    return (
        float(row[f"{metric}_mean"]),
        float(row[f"{metric}_ci_low"]),
        float(row[f"{metric}_ci_high"]),
    )


def _horizontal_interval(
    axis: plt.Axes,
    values: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    labels: list[str],
    colors: list[str],
    xlabel: str,
    xmax: float,
    annotations: list[str] | None = None,
) -> None:
    y = np.arange(len(labels), dtype=float)
    axis.barh(y, values, height=0.58, color=colors, edgecolor="#333333", linewidth=0.45)
    axis.errorbar(
        values,
        y,
        xerr=np.vstack((values - lows, highs - values)),
        fmt="none",
        ecolor="#333333",
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        zorder=3,
    )
    for index, (yy, value, high) in enumerate(zip(y, values, highs)):
        annotation = annotations[index] if annotations is not None else f"{value:.2f}"
        axis.text(
            max(value, high) + xmax * 0.018,
            yy,
            annotation,
            va="center",
            fontsize=5.9,
            linespacing=1.05,
        )
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, xmax)
    axis.set_xlabel(xlabel)
    axis.tick_params(axis="y", labelsize=6.0)
    clean_axis(axis)


def _bootstrap_mean_interval(values: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    """Return a deterministic 95% seed-bootstrap interval for a mean."""
    if values.ndim != 1 or values.size != 20:
        raise RuntimeError("Component intervals require 20 paired seed blocks")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, values.size, size=(20000, values.size))].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def generate_policy_and_latency_figure(
    main_rows: list[dict[str, str]],
    contrast_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    indexed = {row["group"]: row for row in main_rows}
    if set(indexed) != set(MAIN_GROUPS):
        raise RuntimeError(f"Unexpected formal main groups: {sorted(indexed)}")
    for group in MAIN_GROUPS:
        if int(indexed[group]["n_seeds"]) != 30:
            raise RuntimeError(f"Unexpected seed count for {group}")

    requests = np.array([_mean_ci(indexed, group, "request_count") for group in MAIN_GROUPS])
    success = np.array([_mean_ci(indexed, group, "success_rate") for group in MAIN_GROUPS])
    route = np.array([_mean_ci(indexed, group, "route_completion") for group in MAIN_GROUPS])
    if not np.allclose(success[:, 0], 28.0 / 30.0) or not np.allclose(route[:, 0], route[0, 0]):
        raise RuntimeError("Formal task endpoints are not aligned across groups")

    request_contrasts = {
        row["baseline_group"]: row
        for row in contrast_rows
        if row["reference_group"] == "rgd_fixed_policy"
        and row["endpoint"] == "request_count"
        and row["baseline_group"] in {"always_slow", "random_budget"}
    }
    if set(request_contrasts) != {"always_slow", "random_budget"}:
        raise RuntimeError("Missing prespecified request-count contrasts")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.62),
        gridspec_kw={"width_ratios": [1.02, 0.92, 1.06], "wspace": 0.52},
    )
    _horizontal_interval(
        axes[0], requests[:, 0], requests[:, 1], requests[:, 2], MAIN_LABELS, MAIN_COLORS,
        "Requests per episode", 4.6,
    )
    contrast_labels = ["Always Slow", "Random"]
    contrast_groups = ["always_slow", "random_budget"]
    contrast_values = np.array(
        [float(request_contrasts[group]["estimate_rgd_minus_baseline"]) for group in contrast_groups]
    )
    contrast_lows = np.array(
        [float(request_contrasts[group]["ci_low"]) for group in contrast_groups]
    )
    contrast_highs = np.array(
        [float(request_contrasts[group]["ci_high"]) for group in contrast_groups]
    )
    contrast_y = np.arange(len(contrast_labels), dtype=float)
    axes[1].axvline(0.0, color="#555555", linewidth=0.7)
    axes[1].errorbar(
        contrast_values,
        contrast_y,
        xerr=np.vstack((contrast_values - contrast_lows, contrast_highs - contrast_values)),
        fmt="o",
        color=OKABE_ITO["blue"],
        markersize=4.2,
        capsize=2.2,
        elinewidth=0.85,
    )
    reductions = {"always_slow": 90.5, "random_budget": 84.5}
    for yy, group, value in zip(contrast_y, contrast_groups, contrast_values):
        row = request_contrasts[group]
        effect = float(row["paired_standardized_effect_dz"])
        axes[1].text(
            value + 0.22,
            yy,
            f"{reductions[group]:.1f}% fewer\n$d_z$={effect:.2f}",
            fontsize=5.5,
            va="center",
            ha="left",
        )
    axes[1].set_yticks(contrast_y, contrast_labels)
    axes[1].invert_yaxis()
    axes[1].set_ylim(1.35, -0.35)
    axes[1].set_xlim(-4.5, 0.25)
    axes[1].set_xlabel("Paired request difference")
    axes[1].tick_params(axis="y", labelsize=6.0)
    clean_axis(axes[1])

    y = np.arange(len(MAIN_LABELS), dtype=float)
    axes[2].errorbar(
        success[:, 0] * 100.0, y + 0.12,
        xerr=np.vstack(((success[:, 0] - success[:, 1]) * 100.0, (success[:, 2] - success[:, 0]) * 100.0)),
        fmt="o", color="#0072B2", markersize=4.0, capsize=1.8, elinewidth=0.75,
        label="Success",
    )
    axes[2].errorbar(
        route[:, 0] * 100.0, y - 0.12,
        xerr=np.vstack(((route[:, 0] - route[:, 1]) * 100.0, (route[:, 2] - route[:, 0]) * 100.0)),
        fmt="s", color="#D55E00", markersize=3.5, capsize=1.8, elinewidth=0.75,
        label="Route completion",
    )
    axes[2].set_yticks(y, MAIN_LABELS)
    axes[2].invert_yaxis()
    axes[2].set_xlim(70, 102)
    axes[2].set_xticks([70, 80, 90, 100])
    axes[2].set_xlabel("Task endpoint (%)")
    axes[2].legend(frameon=False, fontsize=5.8, loc="lower left", handlelength=1.2)
    axes[2].tick_params(axis="y", labelsize=6.0)
    clean_axis(axes[2], grid_axis="x")

    figure.subplots_adjust(left=0.105, right=0.995, bottom=0.28, top=0.94)
    for axis, label in zip(
        axes,
        ("(a) Query allocation", "(b) Paired request effect", "(c) Task preservation"),
    ):
        axis.text(0.5, -0.34, label, transform=axis.transAxes, ha="center", va="top", fontsize=7.2, fontweight="bold")
    paths, _ = save_figure_triplet(figure, output_dir / "fig_formal_main_allocation")
    return paths


def generate_component_figure(
    summary_rows: list[dict[str, str]],
    by_seed_rows: list[dict[str, str]],
    verification: dict,
    output_dir: Path,
) -> list[Path]:
    if verification.get("accepted") is not True:
        raise RuntimeError("Component verification was not accepted")
    rows = {(row["arm"], row["metric"]): row for row in summary_rows}
    required = [
        "release_events_per_seed",
        "first_step_actuator_distinct_per_seed",
        "executed_first_step_actuator_distinct_per_seed",
    ]
    if any((arm, metric) not in rows for arm in COMPONENT_ARMS for metric in required):
        raise RuntimeError("Missing formal component metrics")
    arrays = []
    for metric in required:
        arrays.append(np.array([
            [float(rows[(arm, metric)]["estimate"]), float(rows[(arm, metric)]["ci_low"]), float(rows[(arm, metric)]["ci_high"])]
            for arm in COMPONENT_ARMS
        ]))

    n_seeds = int(rows[(COMPONENT_ARMS[0], required[0])]["n_seed_blocks"])
    if n_seeds != 20:
        raise RuntimeError(f"Unexpected component seed count: {n_seeds}")

    seed_rows = {(row["arm"], int(row["seed"])): row for row in by_seed_rows}
    seeds = sorted({int(row["seed"]) for row in by_seed_rows})
    expected_cells = {(arm, seed) for arm in COMPONENT_ARMS for seed in seeds}
    if len(seeds) != n_seeds or set(seed_rows) != expected_cells:
        raise RuntimeError("Unexpected component arm--seed matrix")
    issued = np.array(
        [
            _bootstrap_mean_interval(
                np.array(
                    [float(seed_rows[(arm, seed)]["issued_queries"]) for seed in seeds],
                    dtype=float,
                ),
                seed=20260808 + index,
            )
            for index, arm in enumerate(COMPONENT_ARMS)
        ]
    )

    # Totals expose the workload induced by each ablation; intervals remain
    # seed-bootstrap intervals, scaled from the per-seed estimand.
    issued_total = issued * n_seeds
    release_total = arrays[0] * n_seeds
    distinct_total = arrays[1] * n_seeds
    executed_total = arrays[2] * n_seeds
    issued_annotations = [
        f"{int(round(value))}\n{value / n_seeds:.2f}/seed"
        for value in issued_total[:, 0]
    ]
    release_annotations = [
        f"{int(round(value))}\n{value / n_seeds:.2f}/seed"
        for value in release_total[:, 0]
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.68),
        gridspec_kw={"width_ratios": [1.03, 1.0, 1.02], "wspace": 0.62},
    )
    _horizontal_interval(
        axes[0],
        issued_total[:, 0],
        issued_total[:, 1],
        issued_total[:, 2],
        COMPONENT_LABELS,
        COMPONENT_COLORS,
        "Issued requests (20 seeds)",
        57.0,
        issued_annotations,
    )
    _horizontal_interval(
        axes[1],
        release_total[:, 0],
        release_total[:, 1],
        release_total[:, 2],
        COMPONENT_LABELS,
        COMPONENT_COLORS,
        "Release events (20 seeds)",
        40.0,
        release_annotations,
    )

    mapped_counts = np.rint(distinct_total[:, 0]).astype(int)
    executed_counts = np.rint(executed_total[:, 0]).astype(int)
    containment_labels = COMPONENT_LABELS
    containment_y = np.arange(len(containment_labels), dtype=float)
    axes[2].barh(
        containment_y,
        mapped_counts,
        height=0.58,
        color=COMPONENT_COLORS,
        edgecolor="#333333",
        linewidth=0.45,
        label="Before projection",
    )
    axes[2].scatter(
        executed_counts,
        containment_y,
        marker="D",
        s=18,
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.75,
        zorder=4,
        label="After projection",
    )
    for yy, before, after in zip(containment_y, mapped_counts, executed_counts):
        axes[2].text(
            max(float(before), 0.0) + 0.55,
            yy,
            f"{before} $\\rightarrow$ {after}",
            ha="left",
            va="center",
            fontsize=6.0,
        )
    axes[2].set_yticks(containment_y, containment_labels)
    axes[2].invert_yaxis()
    axes[2].set_xlim(-1.0, 20.0)
    axes[2].set_xticks([0, 4, 8, 12, 16, 20])
    axes[2].set_xlabel("Distinct commands (events)")
    axes[2].tick_params(axis="y", labelsize=6.0)
    handles, legend_labels = axes[2].get_legend_handles_labels()
    legend_order = [legend_labels.index("Before projection"), legend_labels.index("After projection")]
    axes[2].legend(
        [handles[index] for index in legend_order],
        [legend_labels[index] for index in legend_order],
        frameon=False,
        fontsize=5.4,
        loc="upper right",
        handlelength=1.1,
        borderaxespad=0.2,
    )
    clean_axis(axes[2])

    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.28, top=0.94)
    for axis, label in zip(
        axes,
        ("(a) Query admission", "(b) Delayed release", "(c) Projection containment"),
    ):
        axis.text(0.5, -0.34, label, transform=axis.transAxes, ha="center", va="top", fontsize=7.2, fontweight="bold")
    paths, _ = save_figure_triplet(figure, output_dir / "fig_formal_component_release")
    return paths


def _component_stage_counts(component_bundle: Path) -> tuple[dict[str, list[int]], dict[str, dict[str, int]]]:
    removed = {
        "full": set(),
        "without_l": {"L"},
        "without_a": {"A"},
        "without_h": {"H"},
        "without_n": {"N"},
        "without_h_and_n": {"H", "N"},
    }
    fields = {
        "L": "component_ablation_latency_survival_pass",
        "A": "component_ablation_maneuver_breadth_pass",
        "H": "component_ablation_corrective_headroom_pass",
        "N": "component_ablation_state_need_pass",
    }
    stagewise: dict[str, list[int]] = {}
    lifecycle = {name: {} for name in ("issued", "released", "timeout")}
    predicate_totals = {name: [] for name in fields}
    for arm in COMPONENT_ARMS:
        event_paths = sorted((component_bundle / arm).rglob("event_log_*.json"))
        if len(event_paths) != 20:
            raise RuntimeError(f"Expected 20 component event logs for {arm}")
        events = []
        for path in event_paths:
            events.extend(read_json(path).get("events", []))
        candidates = [event for event in events if event.get("factorial_candidate_query") is True]
        if len(candidates) != 70:
            raise RuntimeError(f"Expected 70 synchronized candidates for {arm}")
        for name, field in fields.items():
            predicate_totals[name].append(sum(event.get(field) is True for event in candidates))
        survivors = list(candidates)
        retained = [len(survivors)]
        for name, field in fields.items():
            if name not in removed[arm]:
                survivors = [event for event in survivors if event.get(field) is True]
            retained.append(len(survivors))
        stagewise[arm] = retained
        lifecycle["issued"][arm] = sum(
            event.get("closed_loop_latency_issuance_event") is True for event in events
        )
        lifecycle["released"][arm] = sum(
            event.get("closed_loop_latency_release_event") is True for event in events
        )
        lifecycle["timeout"][arm] = sum(
            event.get("closed_loop_latency_timeout_event") is True for event in events
        )
    expected = {"L": 49, "A": 61, "H": 13, "N": 8}
    for name, totals in predicate_totals.items():
        if totals != [expected[name]] * len(COMPONENT_ARMS):
            raise RuntimeError(f"Unexpected {name} predicate totals: {totals}")
    return stagewise, lifecycle


def generate_release_validation_figure(
    component_bundle: Path,
    guarded_dir: Path,
    no_guard_dir: Path,
    output_dir: Path,
) -> list[Path]:
    stagewise, lifecycle = _component_stage_counts(component_bundle)
    guarded_manifest = read_json(guarded_dir / "actual_intervention_manifest.json")
    no_guard_manifest = read_json(no_guard_dir / "actual_intervention_manifest.json")
    if guarded_manifest.get("accepted") is not True or no_guard_manifest.get("accepted") is not True:
        raise RuntimeError("Matched intervention analysis was not accepted")
    guarded_rows = read_rows(guarded_dir / "actual_intervention_effects.csv")
    no_guard_rows = sorted(
        read_rows(no_guard_dir / "actual_intervention_effects.csv"),
        key=lambda row: (int(row["seed"]), int(row["release_frame"])),
    )
    guarded_keys = {(int(row["seed"]), int(row["release_frame"])) for row in guarded_rows}
    no_guard_keys = {(int(row["seed"]), int(row["release_frame"])) for row in no_guard_rows}
    if len(guarded_keys) != 1 or len(no_guard_keys) != 3 or not guarded_keys < no_guard_keys:
        raise RuntimeError("Unexpected guarded/no-guard intervention sets")
    if sum(str(row["actual_slow_corrective"]).lower() == "true" for row in guarded_rows) != 1:
        raise RuntimeError("Guarded intervention is not corrective")
    if sum(str(row["actual_slow_corrective"]).lower() == "true" for row in no_guard_rows) != 1:
        raise RuntimeError("Unexpected no-guard corrective count")

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.68),
        gridspec_kw={"width_ratios": [1.30, 0.96, 1.02], "wspace": 0.58},
    )

    matrix = np.array([stagewise[arm] for arm in COMPONENT_ARMS], dtype=float)
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=70, aspect="auto", interpolation="nearest")
    axes[0].set_xticks(
        np.arange(5),
        ["Candidates", "after $L$", "after $A$", "after $H$", "after $N$"],
        rotation=25,
        ha="right",
    )
    axes[0].set_yticks(np.arange(len(COMPONENT_ARMS)), COMPONENT_LABELS)
    axes[0].tick_params(axis="x", labelsize=5.5, length=0)
    axes[0].tick_params(axis="y", labelsize=5.8, length=0)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            axes[0].text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                fontsize=5.9,
                color="white" if value >= 36 else "#222222",
            )
    for row, columns in {1: [1], 2: [2], 3: [3], 4: [4], 5: [3, 4]}.items():
        for column in columns:
            axes[0].add_patch(
                plt.Rectangle(
                    (column - 0.49, row - 0.49),
                    0.98,
                    0.98,
                    fill=False,
                    edgecolor=OKABE_ITO["orange"],
                    linewidth=1.15,
                    linestyle=(0, (3, 2)),
                )
            )
    axes[0].set_xlabel("Serial admission stage")
    for spine in axes[0].spines.values():
        spine.set_visible(False)

    active_arms = ["without_h", "without_n", "without_h_and_n"]
    active_labels = ["w/o H", "w/o N", "w/o H/N"]
    x = np.arange(len(active_arms), dtype=float)
    width = 0.22
    for offset, name, color, label in (
        (-width, "issued", OKABE_ITO["blue"], "Issued"),
        (0.0, "released", OKABE_ITO["green"], "Released"),
        (width, "timeout", OKABE_ITO["vermillion"], "Timeout"),
    ):
        values = np.array([lifecycle[name][arm] for arm in active_arms], dtype=float)
        axes[1].bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="#333333",
            linewidth=0.45,
            label=label,
        )
        for xx, value in zip(x + offset, values):
            axes[1].text(xx, value + 0.8, str(int(value)), ha="center", va="bottom", fontsize=5.4)
    axes[1].set_xticks(x, active_labels)
    axes[1].set_ylim(0, 45)
    axes[1].set_yticks([0, 10, 20, 30, 40])
    axes[1].set_ylabel("Event count")
    axes[1].set_xlabel("Ablation arm")
    axes[1].legend(frameon=False, fontsize=5.7, loc="upper left", handlelength=1.0)
    axes[1].tick_params(axis="x", labelsize=5.6)
    clean_axis(axes[1], grid_axis="y")

    advantages = np.array([float(row["actual_slow_advantage"]) for row in no_guard_rows])
    fast_values = np.array([float(row["fast_utility"]) for row in no_guard_rows])
    retained = np.array(
        [(int(row["seed"]), int(row["release_frame"])) in guarded_keys for row in no_guard_rows]
    )
    colors = [OKABE_ITO["green"] if keep else OKABE_ITO["vermillion"] for keep in retained]
    bars = axes[2].bar(
        np.arange(len(no_guard_rows)),
        advantages,
        color=colors,
        edgecolor="#333333",
        linewidth=0.5,
        width=0.62,
    )
    axes[2].axhline(0.0, color="#333333", linewidth=0.65)
    for index, (bar, advantage, fast_value, keep) in enumerate(
        zip(bars, advantages, fast_values, retained)
    ):
        relative = 100.0 * advantage / fast_value
        label = f"{advantage:+.4f}"
        if keep:
            label += f"\n({relative:+.1f}%)"
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            advantage + (0.003 if advantage >= 0 else -0.003),
            label,
            ha="center",
            va="bottom" if advantage >= 0 else "top",
            fontsize=5.7,
        )
    axes[2].set_xticks(
        np.arange(len(no_guard_rows)),
        [
            f"{row['seed']}\n{'Retained' if keep else 'Filtered'}"
            for row, keep in zip(no_guard_rows, retained)
        ],
    )
    axes[2].set_ylim(-0.045, 0.075)
    axes[2].set_yticks([-0.04, 0.00, 0.04])
    axes[2].set_ylabel("Matched utility advantage")
    axes[2].tick_params(axis="x", labelsize=5.5)
    clean_axis(axes[2], grid_axis="y")

    figure.subplots_adjust(left=0.10, right=0.995, bottom=0.28, top=0.94)
    for axis, label in zip(
        axes,
        ("(a) Stagewise retention", "(b) Slow-path lifecycle", "(c) Release validation"),
    ):
        axis.text(
            0.5,
            -0.34,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.2,
            fontweight="bold",
        )
    paths, _ = save_figure_triplet(
        figure, output_dir / "fig_formal_component_release_aligned"
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-analysis", type=Path, default=MAIN_ANALYSIS)
    parser.add_argument("--component-analysis", type=Path, default=COMPONENT_ANALYSIS)
    parser.add_argument("--component-bundle", type=Path, default=COMPONENT_BUNDLE)
    parser.add_argument("--guarded-interventions", type=Path, default=GUARDED_INTERVENTIONS)
    parser.add_argument("--no-guard-interventions", type=Path, default=NO_GUARD_INTERVENTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-figure-only", action="store_true")
    args = parser.parse_args()

    main_path = args.main_analysis / "main_results.csv"
    contrast_path = args.main_analysis / "main_results_paired_contrasts.csv"
    component_path = args.component_analysis / "component_ablation_summary.csv"
    component_seed_path = args.component_analysis / "component_ablation_by_seed.csv"
    verification_path = args.component_analysis / "v13_component_ablation_verification.json"
    for path in (
        main_path,
        contrast_path,
        component_path,
        component_seed_path,
        verification_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_tvt_style()
    if args.release_figure_only:
        outputs = generate_release_validation_figure(
            args.component_bundle,
            args.guarded_interventions,
            args.no_guard_interventions,
            args.output_dir,
        )
        print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))
        return 0
    outputs = generate_policy_and_latency_figure(
        read_rows(main_path), read_rows(contrast_path), args.output_dir
    )
    outputs.extend(
        generate_component_figure(
            read_rows(component_path),
            read_rows(component_seed_path),
            read_json(verification_path),
            args.output_dir,
        )
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
