"""Generate the TVT-grade RGD system architecture figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper2" / "generated_figures" / "fig2_system_architecture_rgd_tvt"


def rounded_box(ax, x, y, w, h, edge, face, alpha=0.10, lw=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.016",
            linewidth=lw,
            edgecolor=edge,
            facecolor=face,
            alpha=alpha,
            mutation_aspect=1.0,
        )
    )


def item_chip(ax, x, y, w, h, edge, text, fontsize=5.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0.65,
            edgecolor=edge,
            facecolor="white",
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#2a2f35")


def arrow(ax, x1, y1, x2, y2, color, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=9,
            linewidth=1.05,
            color=color,
            shrinkA=0, shrinkB=0,
        )
    )


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.50, 1.72))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, orange, purple, green, gray = (
        "#1f5a9d", "#c85a12", "#5c4d9a", "#1f7a4d", "#5a626a",
    )

    # Column 1: State
    rounded_box(ax, 0.012, 0.28, 0.168, 0.58, blue, blue, alpha=0.09)
    ax.text(0.096, 0.80, "STATE", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=blue)
    for i, lab in enumerate(["ego kinematics", "neighbor traffic", "action mask"]):
        item_chip(ax, 0.028, 0.66 - i * 0.12, 0.136, 0.085, blue, lab, 5.3)

    # Column 2: Recoverability estimator
    rounded_box(ax, 0.215, 0.28, 0.275, 0.58, orange, orange, alpha=0.09)
    ax.text(0.352, 0.80, "RECOVERABILITY", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=orange)
    for i, lab in enumerate([
        "recovery window Wt",
        "corrective affordance Ft",
        "reversibility Mt",
        "priority score Pt",
    ]):
        item_chip(ax, 0.232, 0.675 - i * 0.105, 0.241, 0.078, orange, lab, 5.2)

    # Column 3: Route decision
    rounded_box(ax, 0.525, 0.28, 0.175, 0.58, purple, purple, alpha=0.09)
    ax.text(0.612, 0.80, "RGD ROUTE", ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=purple)
    for i, lab in enumerate(["Pt >= tau or Qt", "budget bt > 0", "fast / slow"]):
        item_chip(ax, 0.540, 0.655 - i * 0.12, 0.145, 0.085, purple, lab, 5.2)

    # Column 4: Execution
    rounded_box(ax, 0.735, 0.28, 0.250, 0.58, green, green, alpha=0.09)
    ax.text(0.860, 0.80, "EXECUTION", ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=green)
    for i, lab in enumerate([
        "provisional control",
        "slow proposal at t+l",
        "shared safety map",
        "final action at",
    ]):
        item_chip(ax, 0.752, 0.675 - i * 0.105, 0.216, 0.078, green, lab, 5.2)

    # Flow arrows
    arrow(ax, 0.180, 0.57, 0.215, 0.57, blue)
    arrow(ax, 0.490, 0.57, 0.525, 0.57, orange)
    arrow(ax, 0.700, 0.57, 0.735, 0.57, purple)

    # Latency annotation under execution
    ax.annotate(
        "", xy=(0.92, 0.22), xytext=(0.76, 0.22),
        arrowprops=dict(arrowstyle="<->", color=gray, lw=0.8),
    )
    ax.text(0.84, 0.155, "latency l_t", ha="center", va="center",
            fontsize=5.4, color=gray)

    # Audit bar
    rounded_box(ax, 0.08, 0.03, 0.84, 0.105, "#8a9299", "#f4f6f8", alpha=1.0, lw=0.75)
    ax.text(
        0.50, 0.082,
        "AUDIT: score · route · proposal · shield rewrite · final action",
        ha="center", va="center", fontsize=5.5, fontweight="bold", color="#3f464d",
    )

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 300} if suffix == "png" else {}
        fig.savefig(OUT.with_suffix(f".{suffix}"), bbox_inches="tight", pad_inches=0.015, **kwargs)
    plt.close(fig)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
