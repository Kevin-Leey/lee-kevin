"""Create the paper-facing query/release factorial and latency-stress figure.

The figure is descriptive: simulator seeds are aggregated before plotting and
request-level records are never treated as independent error bars.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


ARMS = ("full", "query_only", "release_only", "neither")
ARM_LABELS = {
    "full": "Full",
    "query_only": "Query\nonly",
    "release_only": "Release\nonly",
    "neither": "Neither",
}
ARM_COLORS = {
    "full": "#1f6f8b",
    "query_only": "#5b9aa0",
    "release_only": "#d98c3f",
    "neither": "#8a8f98",
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def build_figure(factorial_root: Path, latency_root: Path, output: Path) -> None:
    episodes = read_csv(factorial_root / "factorial_episode_results.csv")
    lifecycle = defaultdict(lambda: {"candidates": 0.0, "issued": 0.0, "released": 0.0, "timeouts": 0.0})
    for row in episodes:
        arm = row["arm"]
        lifecycle[arm]["candidates"] += float(row["candidate_queries"])
        lifecycle[arm]["issued"] += float(row["issued_queries"])
        lifecycle[arm]["released"] += float(row["release_events"])
        lifecycle[arm]["timeouts"] += float(row["timeouts"])
    n_seed = len({row["seed"] for row in episodes})

    strata = read_csv(latency_root / "stress_latency_stratified_summary.csv")
    release_only = {
        float(row["scheduled_latency_s"]): row
        for row in strata
        if row["arm"] == "release_only"
    }
    latency_values = sorted(release_only)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)
    ax = axes[0]
    x = list(range(len(ARMS)))
    width = 0.62
    released = [lifecycle[a]["released"] / n_seed for a in ARMS]
    timeouts = [lifecycle[a]["timeouts"] / n_seed for a in ARMS]
    rejected = [
        (lifecycle[a]["candidates"] - lifecycle[a]["issued"]) / n_seed
        for a in ARMS
    ]
    ax.bar(x, rejected, width, color="#d9dde2", label="gate rejected")
    ax.bar(x, timeouts, width, bottom=rejected, color="#c65d5d", label="timeout")
    bottoms = [r + t for r, t in zip(rejected, timeouts)]
    ax.bar(x, released, width, bottom=bottoms, color="#4d9f78", label="released")
    for i, arm in enumerate(ARMS):
        total = lifecycle[arm]["candidates"] / n_seed
        ax.text(i, total + 0.12, f"{total:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, [ARM_LABELS[a] for a in ARMS])
    ax.set_ylabel("Events per seed")
    ax.set_title("(a) Shared-opportunity factorial")
    ax.set_ylim(0, max(lifecycle[a]["candidates"] / n_seed for a in ARMS) + 1.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d7dbe0", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.99),
        ncol=3,
        columnspacing=0.8,
        handlelength=1.2,
        borderaxespad=0.0,
    )

    ax = axes[1]
    x = list(range(len(latency_values)))
    release_rate = [
        float(release_only[v]["release_rate_given_issue"]) * 100.0
        for v in latency_values
    ]
    timeout_rate = [
        float(release_only[v]["timeout_rate_given_issue"]) * 100.0
        for v in latency_values
    ]
    ax.plot(x, release_rate, marker="o", markersize=4.2, linewidth=1.7, color="#2f7d59", label="released")
    ax.plot(x, timeout_rate, marker="s", markersize=3.8, linewidth=1.5, color="#b34b4b", label="timeout")
    ax.axvline(2.4, color="#59636e", linestyle=(0, (3, 2)), linewidth=0.9, alpha=0.85)
    ax.text(2.4, 98.0, "gate prediction\n1.7 s", ha="center", va="top", fontsize=7.1, color="#59636e")
    ax.set_xticks(x, [f"{v:g}" for v in latency_values])
    ax.set_xlabel("Scheduled latency (s)")
    ax.set_ylabel("Rate among issued (%)")
    ax.set_title("(b) Synthetic latency/timeout stress")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d7dbe0", linewidth=0.45, alpha=0.75)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.2)

    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorial-root", type=Path, default=Path("results/rgd_factorial_confirmatory_20260731/stress_v5"))
    parser.add_argument("--latency-root", type=Path, default=Path("results/rgd_factorial_confirmatory_20260731/latency_error_analysis"))
    parser.add_argument("--output", type=Path, default=Path("paper/figures/fig_factorial_latency_analysis.pdf"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_figure(args.factorial_root, args.latency_root, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
