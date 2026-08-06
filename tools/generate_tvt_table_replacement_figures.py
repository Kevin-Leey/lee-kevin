from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT = Path("results/legacy_figures")
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    }
)

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
GRAY = "#6F7782"
RED = "#D55E00"


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", color="#d9dde3", linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)


def paired_support():
    labels = ["fast", "slow", "random", "safety", "risk"]
    collision = np.array([0.175, 0.033, 0.150, 0.175, 0.158])
    collision_lo = np.array([0.108, -0.042, 0.083, 0.108, 0.067])
    collision_hi = np.array([0.250, 0.117, 0.217, 0.250, 0.250])
    task = np.array([0.175, 0.208, 0.150, 0.175, 0.167])
    task_lo = np.array([0.100, 0.133, 0.083, 0.108, 0.075])
    task_hi = np.array([0.250, 0.292, 0.217, 0.250, 0.258])

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 1.62))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.errorbar(
        collision,
        y + 0.12,
        xerr=[collision - collision_lo, collision_hi - collision],
        fmt="o",
        markersize=3.0,
        capsize=2.2,
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1.0,
        label="collision delta",
    )
    ax.errorbar(
        task,
        y - 0.12,
        xerr=[task - task_lo, task_hi - task],
        fmt="s",
        markersize=3.0,
        capsize=2.2,
        color=GREEN,
        ecolor=GREEN,
        elinewidth=1.0,
        label="task delta",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("paired delta (positive favors RGD)")
    ax.set_xlim(-0.06, 0.32)
    ax.set_ylim(-0.55, len(labels) - 0.45)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.58, 1.01),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    clean_axes(ax)
    save(fig, "fig_paired_seed_support")


def mismatch_audit():
    rows = ["RGD", "risk", "random", "fast"]
    cols = ["high\nrec.", "high\ncoll.", "mod.\nrec.", "easy\nfast"]
    occupancy = np.array(
        [
            [0.287, 0.199, 0.049, 0.222],
            [0.196, 0.167, 0.032, 0.290],
            [0.012, 0.199, 0.004, 0.574],
            [0.019, 0.236, 0.022, 0.565],
        ]
    )
    slow_rate = np.array(
        [
            [0.295, 0.000, 0.099, 0.000],
            [1.000, 0.011, 1.000, 0.000],
            [0.000, 0.253, 0.000, 0.288],
            [0.000, 0.000, 0.000, 0.000],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.55), constrained_layout=True)
    panels = [
        (axes[0], occupancy, "frame occupancy", 0.0, 0.60, "Blues"),
        (axes[1], slow_rate, "slow-call rate", 0.0, 1.00, "YlGnBu"),
    ]
    for panel_label, (ax, data, title, vmin, vmax, cmap) in zip(["a", "b"], panels):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
        ax.set_title(title, pad=3)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
        ax.tick_params(length=0)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] > (vmax * 0.55) else "#222222"
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", color=color, fontsize=6.5)
        ax.text(
            -0.10,
            1.05,
            panel_label,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.ax.tick_params(labelsize=6, length=2)
    save(fig, "fig_risk_recoverability_mismatch_audit")


def action_flow():
    labels = [
        "slow frames",
        "LLM action kept",
        "risk calibration",
        "raw to final",
        "safety override",
        "route preserved",
    ]
    values = np.array([0.111, 0.906, 0.094, 0.373, 0.448, 0.674])
    colors = [BLUE, GREEN, ORANGE, GRAY, RED, "#56B4E9"]
    y = np.arange(len(labels))[::-1]

    fig, ax = plt.subplots(figsize=(3.35, 1.60))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("rate")
    for yi, v in zip(y, values):
        ax.text(min(v + 0.025, 0.96), yi, f"{v:.3f}", va="center", ha="left", fontsize=6.5)
    clean_axes(ax)
    save(fig, "fig_action_flow_attribution")


if __name__ == "__main__":
    paired_support()
    mismatch_audit()
    action_flow()
