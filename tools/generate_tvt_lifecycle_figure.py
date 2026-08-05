"""Generate the query-to-release lifecycle diagnostics figure for the TVT paper."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import struct

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "tvt_final_20260718" / "main_analysis_v12"
MECHANISM_PATH = ANALYSIS_DIR / "mechanism_attribution.csv"
PAIRED_PATH = ANALYSIS_DIR / "paired_mechanism_comparisons.csv"
OUTPUT_STEM = ROOT / "paper" / "figures" / "fig4_lifecycle_diagnostics"

GROUPS = {
    "rgd_fixed_policy": "RGD",
    "random_budget": "Random",
    "uncertainty_budget": "Uncertainty",
    "risk_budget": "TTC-risk",
}
BASELINES = {
    "random_budget": "Random",
    "uncertainty_budget": "Uncertainty",
    "risk_budget": "TTC-risk",
}
COLORS = {
    "RGD": "#0072B2",
    "Random": "#E69F00",
    "Uncertainty": "#009E73",
    "TTC-risk": "#D55E00",
}
MARKERS = {"RGD": "o", "Random": "s", "Uncertainty": "^", "TTC-risk": "D"}


@dataclass(frozen=True)
class RateEstimate:
    numerator: int
    denominator: int
    value: float
    low: float
    high: float


@dataclass(frozen=True)
class DifferenceEstimate:
    value: float
    low: float
    high: float
    draws: int


RATE_SPECS = (
    ("release_per_attempt", "Released\n/ query"),
    ("release_unavailable_rate", "Unavailable\n/ release"),
    ("post_latency_rewrite_rate", "Rewritten\n/ release"),
    ("kept_distinct_per_release", "Distinct final\n/ release\N{DAGGER}"),
)
FOREST_SPECS = (
    ("release_per_attempt", "Released / query"),
    ("unavailable_per_release", "Unavailable / release"),
    ("rewrite_per_release", "Rewritten / release"),
    ("kept_distinct_per_attempt", "Distinct final / query\N{DAGGER}"),
)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )


def require_columns(reader: csv.DictReader, required: set[str], path: Path) -> None:
    columns = set(reader.fieldnames or ())
    missing = required - columns
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")


def load_mechanism() -> dict[tuple[str, str], RateEstimate]:
    required = {
        "group",
        "metric",
        "numerator",
        "denominator",
        "rate",
        "seed_cluster_ci_low",
        "seed_cluster_ci_high",
    }
    rows: dict[tuple[str, str], RateEstimate] = {}
    with MECHANISM_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader, required, MECHANISM_PATH)
        for row in reader:
            if row["group"] not in GROUPS:
                continue
            key = (row["group"], row["metric"])
            if key in rows:
                raise RuntimeError(f"Duplicate mechanism row: {key}")
            estimate = RateEstimate(
                numerator=int(row["numerator"]),
                denominator=int(row["denominator"]),
                value=float(row["rate"]),
                low=float(row["seed_cluster_ci_low"]),
                high=float(row["seed_cluster_ci_high"]),
            )
            rows[key] = estimate

    needed_metrics = {metric for metric, _ in RATE_SPECS} | {
        "valid_return_per_attempt",
        "release_per_attempt",
        "release_unavailable_rate",
        "post_latency_rewrite_rate",
        "kept_distinct_per_release",
    }
    for group in GROUPS:
        for metric in needed_metrics:
            key = (group, metric)
            if key not in rows:
                raise RuntimeError(f"Missing mechanism row: {key}")
            item = rows[key]
            if item.denominator <= 0 or not 0 <= item.numerator <= item.denominator:
                raise RuntimeError(f"Invalid mechanism counts for {key}: {item}")
            expected = item.numerator / item.denominator
            if abs(item.value - expected) > 1e-12:
                raise RuntimeError(f"Rate/count mismatch for {key}: {item}")
            if not 0 <= item.low <= item.value <= item.high <= 1:
                raise RuntimeError(f"Invalid 95% CI for {key}: {item}")
    return rows


def load_paired() -> dict[tuple[str, str], DifferenceEstimate]:
    required = {
        "baseline",
        "metric",
        "difference_rgd_minus_baseline",
        "paired_seed_cluster_ci_low",
        "paired_seed_cluster_ci_high",
        "draws",
    }
    rows: dict[tuple[str, str], DifferenceEstimate] = {}
    with PAIRED_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader, required, PAIRED_PATH)
        for row in reader:
            if row["baseline"] not in BASELINES:
                continue
            key = (row["baseline"], row["metric"])
            if key in rows:
                raise RuntimeError(f"Duplicate paired row: {key}")
            estimate = DifferenceEstimate(
                value=float(row["difference_rgd_minus_baseline"]),
                low=float(row["paired_seed_cluster_ci_low"]),
                high=float(row["paired_seed_cluster_ci_high"]),
                draws=int(row["draws"]),
            )
            rows[key] = estimate

    for baseline in BASELINES:
        for metric, _ in FOREST_SPECS:
            key = (baseline, metric)
            if key not in rows:
                raise RuntimeError(f"Missing paired row: {key}")
            item = rows[key]
            if item.draws <= 0 or not -1 <= item.low <= item.value <= item.high <= 1:
                raise RuntimeError(f"Invalid paired estimate for {key}: {item}")
    return rows


def funnel_counts(
    mechanism: dict[tuple[str, str], RateEstimate], group: str
) -> tuple[list[str], list[int]]:
    released = mechanism[(group, "release_per_attempt")]
    valid = mechanism[(group, "valid_return_per_attempt")]
    unavailable = mechanism[(group, "release_unavailable_rate")]
    rewritten = mechanism[(group, "post_latency_rewrite_rate")]
    distinct = mechanism[(group, "kept_distinct_per_release")]

    query_count = released.denominator
    release_count = released.numerator
    release_rows = (unavailable, rewritten, distinct)
    if valid.denominator != query_count or valid.numerator != query_count:
        raise RuntimeError(f"Expected one valid return per query for {group}")
    if any(item.denominator != release_count for item in release_rows):
        raise RuntimeError(f"Release-denominator mismatch for {group}")

    available_count = release_count - unavailable.numerator
    unrewritten_count = available_count - rewritten.numerator
    if not 0 <= unrewritten_count <= available_count <= release_count <= query_count:
        raise RuntimeError(f"Non-nested aggregate attrition counts for {group}")
    if distinct.numerator > release_count:
        raise RuntimeError(f"Distinct count exceeds releases for {group}")

    labels = ["Query", "Released", "Available", "Unrewritten", "Distinct final\N{DAGGER}"]
    counts = [
        query_count,
        release_count,
        available_count,
        unrewritten_count,
        distinct.numerator,
    ]
    return labels, counts


def style_axis(axis: mpl.axes.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_rate_panel(
    axis: mpl.axes.Axes, mechanism: dict[tuple[str, str], RateEstimate]
) -> None:
    x = np.arange(len(RATE_SPECS), dtype=float)
    offsets = np.linspace(-0.24, 0.24, len(GROUPS))
    for offset, (group, label) in zip(offsets, GROUPS.items()):
        items = [mechanism[(group, metric)] for metric, _ in RATE_SPECS]
        values = np.array([item.value for item in items])
        lower = values - np.array([item.low for item in items])
        upper = np.array([item.high for item in items]) - values
        axis.errorbar(
            x + offset,
            values,
            yerr=np.vstack([lower, upper]),
            fmt=MARKERS[label],
            markersize=4.7,
            markerfacecolor="white",
            markeredgewidth=1.1,
            color=COLORS[label],
            ecolor=COLORS[label],
            elinewidth=1.0,
            capsize=2.3,
            capthick=0.9,
            label=label,
            zorder=3,
        )

    axis.set_xticks(x, [label for _, label in RATE_SPECS])
    axis.set_xlim(-0.55, len(RATE_SPECS) - 0.45)
    axis.set_ylim(-0.02, 1.04)
    axis.set_ylabel("Empirical rate (95% CI)")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.yaxis.set_major_locator(MultipleLocator(0.2))
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_title("Slow-proposal lifecycle rates", pad=20)
    axis.text(
        0.5,
        1.035,
        "Denominator is shown under each endpoint",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.8,
        color="#555555",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[label],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS[label],
            markeredgewidth=1.1,
            markersize=4.7,
            label=label,
        )
        for label in GROUPS.values()
    ]
    axis.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.4,
    )
    style_axis(axis)


def draw_funnel_panel(
    axis: mpl.axes.Axes, mechanism: dict[tuple[str, str], RateEstimate]
) -> None:
    groups = ("rgd_fixed_policy", "risk_budget")
    labels, _ = funnel_counts(mechanism, groups[0])
    y = np.arange(len(labels), dtype=float)
    offsets = (-0.19, 0.19)
    hatches = ("", "////")

    for offset, hatch, group in zip(offsets, hatches, groups):
        method = GROUPS[group]
        _, counts = funnel_counts(mechanism, group)
        shares = np.array(counts, dtype=float) / counts[0]
        bars = axis.barh(
            y + offset,
            shares,
            height=0.32,
            facecolor=mpl.colors.to_rgba(COLORS[method], 0.18),
            edgecolor=COLORS[method],
            linewidth=0.9,
            hatch=hatch,
            label=method,
            zorder=2,
        )
        for bar, count, share in zip(bars, counts, shares):
            inside = share >= 0.40
            axis.text(
                share - 0.015 if inside else share + 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{count} ({share:.0%})",
                ha="right" if inside else "left",
                va="center",
                fontsize=6.5,
                color=COLORS[method],
            )

    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.08)
    axis.xaxis.set_major_locator(MultipleLocator(0.25))
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Share of admitted queries")
    axis.grid(axis="x", color="#D7D7D7", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.axhline(3.5, color="#777777", linewidth=0.8, linestyle=(0, (2, 2)))
    axis.set_title("Query-normalized attrition", pad=8)
    axis.legend(loc="lower right", frameon=False, borderaxespad=0.1)
    style_axis(axis)


def draw_forest_panel(
    axis: mpl.axes.Axes, paired: dict[tuple[str, str], DifferenceEstimate]
) -> None:
    centers = np.arange(len(FOREST_SPECS) - 1, -1, -1, dtype=float)
    baseline_offsets = (0.20, 0.0, -0.20)
    for index, (metric, _) in enumerate(FOREST_SPECS):
        center = centers[index]
        if index % 2 == 0:
            axis.axhspan(center - 0.47, center + 0.47, color="#F3F3F3", zorder=0)
        for offset, (baseline, baseline_label) in zip(baseline_offsets, BASELINES.items()):
            item = paired[(baseline, metric)]
            value, low, high = 100 * item.value, 100 * item.low, 100 * item.high
            axis.errorbar(
                value,
                center + offset,
                xerr=np.array([[value - low], [high - value]]),
                fmt=MARKERS[baseline_label],
                markersize=4.4,
                markerfacecolor="white",
                markeredgewidth=1.0,
                color=COLORS[baseline_label],
                ecolor=COLORS[baseline_label],
                elinewidth=1.05,
                capsize=2.0,
                zorder=3,
            )

    axis.axvline(0, color="#333333", linewidth=0.85, zorder=1)
    axis.set_yticks(centers, [label for _, label in FOREST_SPECS])
    axis.set_ylim(-0.55, len(FOREST_SPECS) - 0.45)
    axis.set_xlim(-55, 65)
    axis.xaxis.set_major_locator(MultipleLocator(20))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.0f}"))
    axis.set_xlabel("RGD minus baseline (percentage points)")
    axis.grid(axis="x", color="#D7D7D7", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.set_title("Paired lifecycle differences", pad=8)
    handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[label],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=COLORS[label],
            markeredgewidth=1.0,
            markersize=4.4,
            label=label,
        )
        for label in BASELINES.values()
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        ncol=1,
        frameon=False,
        borderaxespad=0.2,
        handletextpad=0.3,
    )
    style_axis(axis)


def draw_figure(
    mechanism: dict[tuple[str, str], RateEstimate],
    paired: dict[tuple[str, str], DifferenceEstimate],
) -> mpl.figure.Figure:
    figure = plt.figure(figsize=(7.16, 5.9))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(0.86, 1.36),
        width_ratios=(0.92, 1.28),
        hspace=0.48,
        wspace=0.46,
    )
    rate_axis = figure.add_subplot(grid[0, :])
    funnel_axis = figure.add_subplot(grid[1, 0])
    forest_axis = figure.add_subplot(grid[1, 1])

    draw_rate_panel(rate_axis, mechanism)
    draw_funnel_panel(funnel_axis, mechanism)
    draw_forest_panel(forest_axis, paired)

    for label, axis in zip(("(a)", "(b)", "(c)"), (rate_axis, funnel_axis, forest_axis)):
        axis.text(
            -0.075,
            1.08,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )

    figure.text(
        0.5,
        0.012,
        "\N{DAGGER}Dashed break: distinct is a separate final-action predicate, diagnostic only and not evidence of benefit. Paired CIs use 20,000 seed-cluster draws.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#333333",
    )
    figure.subplots_adjust(left=0.105, right=0.985, top=0.91, bottom=0.105)
    return figure


def save_figure(figure: mpl.figure.Figure) -> list[Path]:
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png", "svg"):
        path = OUTPUT_STEM.with_suffix(f".{suffix}")
        kwargs = {"dpi": 600} if suffix == "png" else {}
        figure.savefig(path, bbox_inches="tight", pad_inches=0.04, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def check_outputs(outputs: list[Path]) -> None:
    by_suffix = {path.suffix: path for path in outputs}
    if set(by_suffix) != {".pdf", ".png", ".svg"}:
        raise RuntimeError(f"Unexpected output set: {sorted(by_suffix)}")
    for path in outputs:
        if not path.is_file() or path.stat().st_size < 10_000:
            raise RuntimeError(f"Missing or unexpectedly small output: {path}")

    if not by_suffix[".pdf"].read_bytes().startswith(b"%PDF"):
        raise RuntimeError("PDF signature check failed")
    png_header = by_suffix[".png"].read_bytes()[:24]
    if png_header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("PNG signature check failed")
    width, height = struct.unpack(">II", png_header[16:24])
    if width < 3000 or height < 2500:
        raise RuntimeError(f"PNG resolution too small: {width}x{height}")
    if "<svg" not in by_suffix[".svg"].read_text(encoding="utf-8")[:1000]:
        raise RuntimeError("SVG signature check failed")


def main() -> None:
    set_style()
    mechanism = load_mechanism()
    paired = load_paired()
    for group in ("rgd_fixed_policy", "risk_budget"):
        funnel_counts(mechanism, group)
    outputs = save_figure(draw_figure(mechanism, paired))
    check_outputs(outputs)
    print("Validated 16 lifecycle rates and 12 paired estimates.")
    for path in outputs:
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
