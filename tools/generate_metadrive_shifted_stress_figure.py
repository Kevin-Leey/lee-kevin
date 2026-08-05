"""Generate the deterministic MetaDrive shifted-stress figure for the paper.

The script reads the locked per-seed run rows. It never infers missing
environments or rewrites measured values with a generative model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = (
    ROOT
    / "results"
    / "metadrive_result"
    / "formal_run"
    / "2026-07-15"
    / "metadrive_stress"
)
OUTPUT = ROOT / "paper" / "figures" / "fig6_metadrive_shifted_stress"

GROUPS = (
    "always_fast",
    "random_budget",
    "uncertainty_budget",
    "risk_budget",
    "rgd_fixed_policy",
)
LABELS = {
    "always_fast": "Fast-only",
    "random_budget": "Random",
    "uncertainty_budget": "Uncertainty",
    "risk_budget": "TTC-risk",
    "rgd_fixed_policy": "RGD",
}
ENVS = ("metadrive-highway-v0", "metadrive-merge-v0")
ENV_LABELS = {
    "metadrive-highway-v0": "Highway",
    "metadrive-merge-v0": "Merge",
}
ENV_COLORS = {
    "metadrive-highway-v0": "#0072B2",
    "metadrive-merge-v0": "#D55E00",
}
ENV_MARKERS = {
    "metadrive-highway-v0": "o",
    "metadrive-merge-v0": "s",
}


@dataclass(frozen=True)
class Run:
    group: str
    env: str
    seed: int
    success: int
    collision: int
    route_completion: float
    frames: int
    slow_calls: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing {key}")
    return float(value)


def load_runs() -> list[Run]:
    runs: list[Run] = []
    for group in GROUPS:
        path = BUNDLE / group / f"{group}_run_rows.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in _read_csv(path):
            env = str(row.get("env", ""))
            if env not in ENVS:
                continue
            frames = int(round(_as_float(row, "total_frames")))
            slow_rate = _as_float(row, "slow_call_rate")
            slow_calls_float = slow_rate * frames
            slow_calls = int(round(slow_calls_float))
            if abs(slow_calls_float - slow_calls) > 1e-6:
                raise ValueError(
                    f"non-integral slow-call count for {group}/{env}/"
                    f"{row.get('seed_idx')}: {slow_calls_float}"
                )
            runs.append(
                Run(
                    group=group,
                    env=env,
                    seed=int(row["seed_idx"]),
                    success=int(round(_as_float(row, "success_rate"))),
                    collision=int(round(_as_float(row, "collision_rate"))),
                    route_completion=_as_float(row, "avg_route_completion"),
                    frames=frames,
                    slow_calls=slow_calls,
                )
            )

    keys = [(run.group, run.env, run.seed) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate group/environment/seed rows")
    for env in ENVS:
        expected: set[int] | None = None
        for group in GROUPS:
            seeds = {run.seed for run in runs if run.group == group and run.env == env}
            if len(seeds) != 30:
                raise ValueError(f"expected 30 seeds for {group}/{env}, found {len(seeds)}")
            if expected is None:
                expected = seeds
            elif seeds != expected:
                raise ValueError(f"unmatched seeds for {group}/{env}")
    if len(runs) != len(GROUPS) * len(ENVS) * 30:
        raise ValueError(f"unexpected run count: {len(runs)}")
    return runs


def _bootstrap_mean_ci(
    values: Iterable[float], *, seed: int, draws: int = 20_000
) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        raise ValueError("empty bootstrap sample")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(draws, data.size))
    estimates = data[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(data.mean()), float(low), float(high)


def _subset(runs: list[Run], group: str, env: str) -> list[Run]:
    return sorted(
        (run for run in runs if run.group == group and run.env == env),
        key=lambda run: run.seed,
    )


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean(axis: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.45)
    axis.set_axisbelow(True)


def generate(runs: list[Run]) -> None:
    _style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.55),
        gridspec_kw={"width_ratios": [1.05, 1.0, 1.15], "wspace": 0.38},
    )

    # (a) Route completion with seed-bootstrap intervals.
    axis = axes[0]
    x = np.arange(len(GROUPS), dtype=float)
    offsets = (-0.11, 0.11)
    for env, offset in zip(ENVS, offsets):
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for group_index, group in enumerate(GROUPS):
            values = [run.route_completion for run in _subset(runs, group, env)]
            mean, low, high = _bootstrap_mean_ci(
                values, seed=20260719 + 100 * group_index + ENVS.index(env)
            )
            means.append(mean)
            lows.append(low)
            highs.append(high)
        values_arr = np.asarray(means)
        axis.errorbar(
            x + offset,
            values_arr,
            yerr=np.vstack([values_arr - lows, np.asarray(highs) - values_arr]),
            fmt=ENV_MARKERS[env],
            markersize=4.6,
            markerfacecolor="white",
            markeredgewidth=1.0,
            capsize=2.1,
            linewidth=1.0,
            color=ENV_COLORS[env],
            label=ENV_LABELS[env],
            zorder=3,
        )
    axis.set_xticks(x, [LABELS[group] for group in GROUPS], rotation=24, ha="right")
    axis.set_ylabel("Mean route completion")
    axis.set_ylim(0.30, 0.76)
    axis.legend(frameon=False, loc="upper right", handletextpad=0.4)
    _clean(axis)

    # (b) Exact micro slow-call rate; numerator is calls and denominator is frames.
    axis = axes[1]
    width = 0.34
    for env, offset in zip(ENVS, (-width / 2, width / 2)):
        rates = []
        for group in GROUPS:
            selected = _subset(runs, group, env)
            rates.append(sum(run.slow_calls for run in selected) / sum(run.frames for run in selected))
        axis.bar(
            x + offset,
            rates,
            width=width,
            color=ENV_COLORS[env],
            alpha=0.90,
            edgecolor="white",
            linewidth=0.45,
            label=ENV_LABELS[env],
        )
    axis.set_xticks(x, [LABELS[group] for group in GROUPS], rotation=24, ha="right")
    axis.set_ylabel("Slow calls per control step")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_ylim(0.0, 0.058)
    _clean(axis)

    # (c) Paired route-completion differences on the common seeds.
    axis = axes[2]
    baselines = GROUPS[:-1]
    y = np.arange(len(baselines), dtype=float)
    for env, offset in zip(ENVS, (-0.09, 0.09)):
        values: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        rgd_by_seed = {run.seed: run.route_completion for run in _subset(runs, "rgd_fixed_policy", env)}
        for baseline_index, baseline in enumerate(baselines):
            base_by_seed = {run.seed: run.route_completion for run in _subset(runs, baseline, env)}
            differences = [rgd_by_seed[seed] - base_by_seed[seed] for seed in sorted(rgd_by_seed)]
            mean, low, high = _bootstrap_mean_ci(
                differences, seed=20261719 + 100 * baseline_index + ENVS.index(env)
            )
            values.append(mean)
            lows.append(low)
            highs.append(high)
        values_arr = np.asarray(values)
        axis.errorbar(
            values_arr,
            y + offset,
            xerr=np.vstack([values_arr - lows, np.asarray(highs) - values_arr]),
            fmt=ENV_MARKERS[env],
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            capsize=2.0,
            linewidth=1.0,
            color=ENV_COLORS[env],
            label=ENV_LABELS[env],
            zorder=3,
        )
    axis.axvline(0.0, color="#555555", linewidth=0.75, linestyle="--", zorder=1)
    axis.set_yticks(y, [LABELS[group] for group in baselines])
    axis.invert_yaxis()
    axis.set_xlabel("RGD minus baseline route completion")
    axis.set_xlim(-0.018, 0.072)
    _clean(axis, grid_axis="x")

    panel_labels = (
        "(a) Route completion",
        "(b) Slow-path exposure",
        "(c) Paired RGD differences",
    )
    for axis, label in zip(axes, panel_labels):
        axis.text(
            0.5,
            -0.42,
            label,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=7.4,
            fontweight="bold",
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=0.33, top=0.96, left=0.065, right=0.995)
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=360)
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def print_audit(runs: list[Run]) -> None:
    for env in ENVS:
        for group in GROUPS:
            selected = _subset(runs, group, env)
            frames = sum(run.frames for run in selected)
            calls = sum(run.slow_calls for run in selected)
            print(
                env,
                group,
                f"n={len(selected)}",
                f"success={sum(run.success for run in selected)}/{len(selected)}",
                f"collision={sum(run.collision for run in selected)}/{len(selected)}",
                f"route={np.mean([run.route_completion for run in selected]):.6f}",
                f"slow={calls}/{frames}",
            )


def main() -> None:
    runs = load_runs()
    print_audit(runs)
    generate(runs)
    print(OUTPUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
