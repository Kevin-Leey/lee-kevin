"""Generate Fig. 1: risk vs post-latency recoverability mismatch."""

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np


OUT = Path("paper2/generated_figures/fig1_risk_recoverability_mismatch")


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_ab(ax, title, risk_high=True, recoverable=True, label="(a)"):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(-0.02, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)

    # Timeline
    ax.plot([0.08, 0.92], [0.42, 0.42], color="#444444", lw=1.0, zorder=1)
    ax.scatter([0.18, 0.72], [0.42, 0.42], s=28, color="#222222", zorder=3)
    ax.text(0.18, 0.30, "query t", ha="center", fontsize=6.6)
    ax.text(0.72, 0.30, "return t+l", ha="center", fontsize=6.6)
    ax.annotate(
        "", xy=(0.68, 0.48), xytext=(0.22, 0.48),
        arrowprops=dict(arrowstyle="->", color="#6a7180", lw=0.8),
    )
    ax.text(0.45, 0.53, "latency l", ha="center", fontsize=6.3, color="#6a7180")

    # Risk badge
    risk_color = "#D55E00" if risk_high else "#009E73"
    risk_text = "HIGH risk" if risk_high else "MODERATE risk"
    ax.add_patch(FancyBboxPatch(
        (0.05, 0.72), 0.38, 0.18, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=risk_color, edgecolor=risk_color, alpha=0.15, lw=0.9,
    ))
    ax.text(0.24, 0.81, risk_text, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=risk_color)

    # Recoverability badge
    rec_color = "#0072B2" if recoverable else "#999999"
    rec_text = "R set nonempty" if recoverable else "R set empty"
    ax.add_patch(FancyBboxPatch(
        (0.57, 0.72), 0.38, 0.18, boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=rec_color, edgecolor=rec_color, alpha=0.15, lw=0.9,
    ))
    ax.text(0.76, 0.81, rec_text, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=rec_color)

    # Verdict
    if risk_high and not recoverable:
        verdict = "Risk trigger: call  |  Recoverability: skip"
        vcol = "#B00020"
    elif (not risk_high) and recoverable:
        verdict = "Risk trigger: skip | Recoverability: call"
        vcol = "#0072B2"
    else:
        verdict = "Triggers agree"
        vcol = "#444444"
    ax.text(0.50, 0.12, verdict, ha="center", va="center",
            fontsize=6.5, color=vcol, fontweight="bold")
    ax.text(0.50, 0.98, title, ha="center", va="top", fontsize=7.2)


def panel_c(ax):
    ax.text(-0.08, 1.05, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    # 2x2 schematic: risk high/low vs recoverability yes/no
    # Counts are illustrative locked-log contrast from paper narrative, not new stats.
    # We show qualitative regions + early-call fractions from Table timing.
    regions = [
        # x0, y0, color, title, note
        (0.0, 0.52, "#F4C7B0", "High risk\nR empty", "risk FP zone"),
        (0.52, 0.52, "#C6DBEF", "High risk\nR nonempty", "joint zone"),
        (0.0, 0.0, "#E8E8E8", "Low risk\nR empty", "skip both"),
        (0.52, 0.0, "#B8E0D2", "Low/mod risk\nR nonempty", "missed by risk"),
    ]
    for x0, y0, color, title, note in regions:
        ax.add_patch(Rectangle(
            (x0, y0), 0.48, 0.48, facecolor=color, edgecolor="#555555", lw=0.6,
        ))
        ax.text(x0 + 0.24, y0 + 0.30, title, ha="center", va="center",
                fontsize=6.4, fontweight="bold", color="#222222")
        ax.text(x0 + 0.24, y0 + 0.12, note, ha="center", va="center",
                fontsize=5.8, color="#444444")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axis("off")
    ax.text(0.50, -0.08, "Risk salience  vs  post-latency recoverability",
            ha="center", va="top", fontsize=6.6, color="#333333")
    # Axis labels
    ax.text(0.24, 1.01, "R empty", ha="center", fontsize=6.0, color="#555")
    ax.text(0.76, 1.01, "R nonempty", ha="center", fontsize=6.0, color="#555")
    ax.text(-0.05, 0.76, "High\nrisk", ha="right", va="center", fontsize=6.0, color="#555")
    ax.text(-0.05, 0.24, "Low\nrisk", ha="right", va="center", fontsize=6.0, color="#555")


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(
        1, 3, figsize=(7.16, 1.95),
        gridspec_kw={"wspace": 0.26, "width_ratios": [1.0, 1.0, 1.05]},
    )
    panel_ab(
        axes[0],
        "Salient but no longer correctable",
        risk_high=True, recoverable=False, label="(a)",
    )
    panel_ab(
        axes[1],
        "Moderate risk, still correctable",
        risk_high=False, recoverable=True, label="(b)",
    )
    panel_c(axes[2])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(OUT.with_suffix(f".{ext}"), bbox_inches="tight", pad_inches=0.03, **kwargs)
    plt.close(fig)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
