"""Generate the paper-facing query--release factorial figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvt_figure_utils import apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = (
    ROOT
    / "results"
    / "rgd_factorial_confirmatory_20260731"
    / "frozen_v7_repro_20260801"
    / "analysis"
)
DEFAULT_OUTPUT_DIR = ROOT / "paper" / "figures"

ARMS = ("full", "query_only", "release_only", "neither")
ARM_LABELS = ("Full\nRGD", "Query\nonly", "Release\nonly", "Neither")
ARM_COLORS = ("#1F77B4", "#009E73", "#E69F00", "#CC79A7")
PANEL_LABELS = (
    "(a) Slow-path calls",
    "(b) Selections differing from Fast",
    "(c) Driving distance",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_summary(rows: list[dict[str, str]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indexed = {(row["arm"], row["metric"]): row for row in rows}
    selected = []
    for arm in ARMS:
        row = indexed.get((arm, metric))
        if row is None:
            raise RuntimeError(f"Missing factorial summary for {arm}/{metric}")
        selected.append(row)
    mean = np.asarray([float(row["mean"]) for row in selected], dtype=float)
    low = np.asarray([float(row["ci_low"]) for row in selected], dtype=float)
    high = np.asarray([float(row["ci_high"]) for row in selected], dtype=float)
    return mean, low, high


def clean_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D7DBE0", linewidth=0.45, alpha=0.85)
    axis.set_axisbelow(True)


def add_panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.5,
        -0.34,
        label,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        fontweight="bold",
    )


def draw_bar_panel(
    axis: plt.Axes,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    ylabel: str,
    upper: float,
) -> None:
    positions = np.arange(len(ARMS))
    bars = axis.bar(
        positions,
        mean,
        width=0.70,
        color=ARM_COLORS,
        edgecolor="#333333",
        linewidth=0.45,
        yerr=np.vstack([mean - low, high - mean]),
        error_kw={"elinewidth": 0.65, "capsize": 1.8, "capthick": 0.65},
    )
    for bar, value, upper_bound in zip(bars, mean, high):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            min(upper_bound + 0.11, upper - 0.12),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    axis.set_xticks(positions, ARM_LABELS)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0.0, upper)
    clean_axis(axis)


def generate_figure(summary_rows: list[dict[str, str]], effect_rows: list[dict[str, str]], output_dir: Path) -> list[Path]:
    call_mean, call_low, call_high = metric_summary(summary_rows, "issued_queries")
    selection_mean, selection_low, selection_high = metric_summary(
        summary_rows, "primitive_distinct_selections"
    )
    distance_mean, distance_low, distance_high = metric_summary(
        summary_rows, "driving_distance"
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    figure.subplots_adjust(
        left=0.088,
        right=0.975,
        bottom=0.28,
        top=0.96,
        wspace=0.34,
    )

    draw_bar_panel(
        axes[0],
        call_mean,
        call_low,
        call_high,
        ylabel="Slow calls / episode",
        upper=7.0,
    )
    draw_bar_panel(
        axes[1],
        selection_mean,
        selection_low,
        selection_high,
        ylabel="Selections differing\nfrom Fast / episode",
        upper=3.65,
    )

    positions = np.arange(len(ARMS))
    axes[2].errorbar(
        positions,
        distance_mean,
        yerr=np.vstack([distance_mean - distance_low, distance_high - distance_mean]),
        linestyle="none",
        marker="o",
        markersize=5.4,
        color="#1F77B4",
        markeredgecolor="#333333",
        markeredgewidth=0.55,
        elinewidth=0.85,
        capsize=2.2,
        zorder=3,
    )
    axes[2].set_xticks(positions, ARM_LABELS)
    axes[2].set_ylabel("Driving distance (m)")
    axes[2].set_ylim(535, 650)
    axes[2].set_yticks([540, 560, 580, 600, 620, 640])
    clean_axis(axes[2])

    for axis, panel_label in zip(axes, PANEL_LABELS):
        add_panel_label(axis, panel_label)

    paths, _ = save_figure_triplet(
        figure,
        output_dir / "fig_query_release_factorial",
        crop=False,
    )
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary_path = args.analysis_dir / "factorial_arm_summary.csv"
    effects_path = args.analysis_dir / "factorial_paired_effects.csv"
    if not summary_path.is_file() or not effects_path.is_file():
        raise FileNotFoundError("Factorial analysis summaries are unavailable")

    apply_tvt_style()
    paths = generate_figure(read_rows(summary_path), read_rows(effects_path), args.output_dir)
    print("\n".join(str(path) for path in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
