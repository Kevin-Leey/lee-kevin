"""Generate the paper's central post-latency recoverability evidence figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rollout = rows(args.rollout)
    common_rows = [
        row for row in rollout if row["allocator"] in {"RGD", "TTC-delay", "TTC-risk"}
    ]
    common = {row["allocator"]: row for row in common_rows}
    if len(common_rows) != 3 or set(common) != {"RGD", "TTC-delay", "TTC-risk"}:
        raise RuntimeError("Expected one row for each allocator in the nominal-delay comparison")
    latency = sorted(
        [row for row in rollout if row["allocator"] == "RGD-fixed"],
        key=lambda row: float(row["delay_s"]),
    )
    if [float(row["delay_s"]) for row in latency] != [0.7, 1.7, 2.7]:
        raise RuntimeError("Expected the fixed-query delay grid 0.7/1.7/2.7 s")
    if len({int(row["release_count"]) for row in latency}) != 1:
        raise RuntimeError("Fixed-query rows do not share one common denominator")
    rgd, ttc_delay, ttc = common["RGD"], common["TTC-delay"], common["TTC-risk"]
    expected_counts = {
        "RGD": (12, 96),
        "TTC-delay": (0, 47),
        "TTC-risk": (4, 80),
    }
    for allocator, (corrective, released) in expected_counts.items():
        row = common[allocator]
        observed = (int(row["corrective_count"]), int(row["release_count"]))
        if observed != (corrective, released):
            raise RuntimeError(
                f"Locked allocator counts changed for {allocator}: {observed}"
            )
    fixed_counts = [
        (int(row["corrective_count"]), int(row["release_count"])) for row in latency
    ]
    if fixed_counts != [(12, 92), (12, 92), (5, 92)]:
        raise RuntimeError(f"Locked fixed-query counts changed: {fixed_counts}")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.45, 2.20),
        sharey=True,
        gridspec_kw={"width_ratios": [1.08, 1.0], "wspace": 0.30},
    )
    blue, neutral, orange = "#0072B2", "#777777", "#D55E00"

    ax = axes[0]
    values = [
        float(rgd["corrective_fraction"]),
        float(ttc_delay["corrective_fraction"]),
        float(ttc["corrective_fraction"]),
    ]
    lows = [float(rgd["ci_low"]), float(ttc_delay["ci_low"]), float(ttc["ci_low"])]
    highs = [float(rgd["ci_high"]), float(ttc_delay["ci_high"]), float(ttc["ci_high"])]
    yerr = [
        [value - low for value, low in zip(values, lows)],
        [high - value for value, high in zip(values, highs)],
    ]
    bars = ax.bar(
        [0, 1, 2],
        values,
        width=0.58,
        color=[blue, "#D9D9D9", "#F0A35A"],
        edgecolor=["#004A75", neutral, "#9B3D00"],
        linewidth=0.8,
        hatch=["", "", "///"],
        yerr=yerr,
        capsize=3,
        error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#333333"},
    )
    counts = [
        f"{rgd['corrective_count']}/{rgd['release_count']}",
        f"{ttc_delay['corrective_count']}/{ttc_delay['release_count']}",
        f"{ttc['corrective_count']}/{ttc['release_count']}",
    ]
    for bar, high, count in zip(bars, highs, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            high + 0.016,
            count,
            ha="center",
            va="bottom",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
    ax.set_xticks([0, 1, 2], ["RGD", "TTC-\ndelay", "TTC-\nrisk"])
    ax.set_ylim(0, 0.25)
    ax.set_ylabel("Corrective-set fraction")
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_title("(a) Allocators, 1.7 s", loc="left", pad=3.0)

    ax = axes[1]
    delays = [float(row["delay_s"]) for row in latency]
    retained = [float(row["corrective_fraction"]) for row in latency]
    latency_lows = [float(row["ci_low"]) for row in latency]
    latency_highs = [float(row["ci_high"]) for row in latency]
    latency_yerr = [
        [value - low for value, low in zip(retained, latency_lows)],
        [high - value for value, high in zip(retained, latency_highs)],
    ]
    counts = [f"{row['corrective_count']}/{row['release_count']}" for row in latency]
    ax.errorbar(
        delays,
        retained,
        yerr=latency_yerr,
        color=blue,
        marker="o",
        markersize=4.2,
        linewidth=1.25,
        elinewidth=0.8,
        capsize=3,
    )
    for x, high, count in zip(delays, latency_highs, counts):
        ax.text(
            x,
            high + 0.015,
            count,
            ha="center",
            va="bottom",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6},
        )
    ax.set_xlim(min(delays) - 0.18, max(delays) + 0.18)
    ax.set_ylim(0.0, 0.25)
    ax.set_xticks(delays)
    ax.set_xlabel("Release delay (s)")
    ax.set_yticks([0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.grid(axis="y", which="major", color="#D9D9D9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_title("(b) RGD delay sweep", loc="left", pad=3.0)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2.5, width=0.6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.88)
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(
            args.output_dir / f"fig3_post_latency_evidence.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
            **kwargs,
        )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
