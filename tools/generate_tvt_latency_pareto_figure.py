"""Generate the submission-grade allocator--latency / Pareto / yield figure."""

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("results/exploratory_figures/fig_allocator_latency_pareto.pdf")
N = 30
POLICIES = ["RGD", "Fast-only", "Random", "TTC-risk", "Uncertainty"]
ZERO = [7, 6, 8, 10, 9]
DELAYED = [7, 6, 5, 6, 6]
CALLS = [4.33, 0.00, 4.67, 5.03, 5.13]
SPEED = [22.59, 22.50, 22.40, 22.25, 22.27]
# Completions per 100 slow attempts under measured latency (fast-only omitted).
YIELD_LABELS = ["RGD", "Random", "TTC-risk", "Uncertainty"]
YIELD_VALS = [5.38, 3.57, 3.97, 3.90]
COLORS = {
    "RGD": "#0072B2",
    "Fast-only": "#4D4D4D",
    "Random": "#E69F00",
    "TTC-risk": "#D55E00",
    "Uncertainty": "#009E73",
}
MARKERS = {
    "RGD": "o",
    "Fast-only": "s",
    "Random": "^",
    "TTC-risk": "D",
    "Uncertainty": "v",
}


def wilson(k, n=N, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, center - half, center + half


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 2.05),
        gridspec_kw={"wspace": 0.36, "width_ratios": [1.05, 1.0, 0.95]},
    )

    # (a) Latency interaction
    ax = axes[0]
    for name, zc, dc in zip(POLICIES, ZERO, DELAYED):
        color, marker = COLORS[name], MARKERS[name]
        values, lowers, uppers = [], [], []
        for count in (zc, dc):
            p, lo, hi = wilson(count)
            values.append(p)
            lowers.append(p - lo)
            uppers.append(hi - p)
        ax.plot(
            [0, 1], values, color=color, marker=marker, linewidth=1.35,
            markersize=4.6, label=name, zorder=3,
        )
        ax.errorbar(
            [0, 1], values, yerr=[lowers, uppers], fmt="none",
            ecolor=color, elinewidth=0.75, capsize=2.0, alpha=0.85, zorder=2,
        )
    ax.set_xticks([0, 1], ["Frozen state\n(0 s)", "Closed-loop motion\n(1.7 s)"])
    ax.set_ylabel("Collision-free completion")
    ax.set_ylim(0.0, 0.56)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.legend(
        frameon=False, ncol=2, loc="upper right", handlelength=1.6,
        columnspacing=0.75, borderaxespad=0.1,
    )
    ax.text(-0.18, 1.05, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.text(0.02, 0.03, "n = 30 paired seeds", transform=ax.transAxes, fontsize=6.5)

    # (b) Calls--speed Pareto
    ax = axes[1]
    for name, calls, speed in zip(POLICIES, CALLS, SPEED):
        color, marker = COLORS[name], MARKERS[name]
        size = 68 if name == "RGD" else 44
        ax.scatter(
            calls, speed, s=size, marker=marker, facecolor=color,
            edgecolor="white", linewidth=0.7, zorder=3,
        )
        offsets = {
            "RGD": (0.08, 0.015),
            "Fast-only": (0.10, -0.028),
            "Random": (-0.55, 0.018),
            "TTC-risk": (-0.58, -0.048),
            "Uncertainty": (-0.22, 0.028),
        }
        dx, dy = offsets[name]
        ax.text(calls + dx, speed + dy, name, color=color, fontsize=6.6)
    # Soft dominance region for RGD among online allocators
    ax.axvspan(-0.2, 4.45, ymin=0.72, ymax=0.98, color="#0072B2", alpha=0.06, zorder=0)
    ax.annotate(
        "fewer calls / higher speed",
        xy=(4.33, 22.59), xytext=(1.55, 22.30),
        arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#0072B2"},
        color="#0072B2", fontsize=6.5,
    )
    ax.set_xlabel("Slow attempts per episode")
    ax.set_ylabel("Mean speed (m/s)")
    ax.set_xlim(-0.25, 5.70)
    ax.set_ylim(22.14, 22.68)
    ax.grid(color="#D9D9D9", linewidth=0.45)
    ax.text(-0.18, 1.05, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (c) Compute-normalized yield under measured latency
    ax = axes[2]
    y_colors = [COLORS[n] for n in YIELD_LABELS]
    x = np.arange(len(YIELD_LABELS))
    bars = ax.bar(
        x, YIELD_VALS, color=y_colors, width=0.68, edgecolor="white",
        linewidth=0.6, zorder=3,
    )
    for bar, val, name in zip(bars, YIELD_VALS, YIELD_LABELS):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val + 0.08, f"{val:.2f}",
            ha="center", va="bottom", fontsize=6.6, color=COLORS[name],
            fontweight="bold" if name == "RGD" else "normal",
        )
    ax.set_xticks(x, YIELD_LABELS, rotation=18, ha="right")
    ax.set_ylabel("Completions / 100 calls")
    ax.set_ylim(0.0, 6.4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45)
    ax.text(-0.20, 1.05, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    ax.text(
        0.02, 0.03, "measured 1.7 s latency",
        transform=ax.transAxes, fontsize=6.5,
    )

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    # Also write PNG for visual QA
    fig.savefig(OUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
