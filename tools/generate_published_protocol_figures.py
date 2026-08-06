"""Generate source-style figures for the three published-protocol evaluations."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev

import matplotlib.pyplot as plt
import numpy as np

from tvt_figure_utils import apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"

POLICY_CSV = (
    RESULTS
    / "published_protocol_exact_policy_transfer_n1000_20260729.csv"
)
BUA_CSV = (
    RESULTS
    / "published_protocol_exact_bua_n1000_20260729.csv"
)
CTRAIL_CSV = (
    RESULTS
    / "published_protocol_v13_ctrail_qwen3_8b_eval_n100pt_20260726_r1_seed_metrics_v7.csv"
)

POLICY_TASKS = [
    "A: Pedestrian Crossing",
    "B: Uncontrolled Intersection",
    "C: Integrated Traffic",
]
BUA_TASKS = ["Left-Turn", "Right-Turn", "Going-Straight", "Merge", "Roundabout"]
CTRAIL_TASKS = ["Highway", "Merge", "Roundabout", "Intersection"]

# Values transcribed from the source papers' quantitative tables. They are
# plotted as reported and are never used to manufacture local uncertainty.
POLICY_SOURCE = {
    "safety": [99.2, 98.3, 97.9],
    "reward": [0.66, 0.77, 0.56],
    "reward_error": [0.03, 0.02, 0.03],
    "speed": [7.1, 8.6, 6.8],
}
BUA_SOURCE_FULL = {
    "tasks": [0.966, 0.991, 0.984, 0.987, 0.974],
    "density": [0.975, 0.961, 0.948],
}
CTRAIL_SOURCE = {
    "seen": [88.3, 87.1, 85.2, 84.2],
    "seen_ci": [1.6, 1.8, 2.2, 3.2],
    "unseen": [86.8, 85.5, 84.3, 82.1],
    "unseen_ci": [6.0, 4.0, 3.8, 3.2],
}


def read_rows(path: Path, expected: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected:
        raise RuntimeError(f"{path.name}: expected {expected} rows, observed {len(rows)}")
    if any(int(row["failed_returns"]) != 0 for row in rows):
        raise RuntimeError(f"{path.name}: slow-return failures are present")
    return rows


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError((successes, n))
    z = 1.959963984540054
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    radius /= denominator
    return p, center - radius, center + radius


def mean_t_interval(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 1000:
        raise RuntimeError(f"The source-matched mean-CI contract expects n=1000, got {len(values)}")
    mean = fmean(values)
    # t(0.975, 999), fixed to avoid introducing a SciPy runtime dependency.
    radius = 1.962341461133 * stdev(values) / math.sqrt(len(values))
    return mean, mean - radius, mean + radius


def grouped(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[row[key]].append(row)
    return dict(output)


def interval_errors(
    values: list[float], lows: list[float], highs: list[float]
) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    return np.vstack(
        [values_array - np.asarray(lows), np.asarray(highs) - values_array]
    )


def style_source_axis(axis: plt.Axes, *, grid_axis: str = "both") -> None:
    axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.55, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.5, width=0.55)
    for spine in axis.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.6)


def generate_policy_figure(rows: list[dict[str, str]]) -> list[Path]:
    by_task = grouped(rows, "task")
    if list(sorted(by_task)) != list(sorted(POLICY_TASKS)):
        raise RuntimeError(f"Unexpected Policy Transfer tasks: {sorted(by_task)}")

    local: dict[str, list[tuple[float, float, float]]] = {
        "safety": [],
        "reward": [],
        "speed": [],
    }
    for task in POLICY_TASKS:
        selected = by_task[task]
        if len(selected) != 1000:
            raise RuntimeError(f"{task}: expected 1000 episodes")
        safety = wilson(sum(int(row["collision_free"]) for row in selected), 1000)
        local["safety"].append(tuple(100.0 * value for value in safety))
        local["reward"].append(
            mean_t_interval([float(row["published_mean_reward_per_step"]) for row in selected])
        )
        local["speed"].append(
            mean_t_interval([float(row["mean_episode_speed_mps"]) for row in selected])
        )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.16, 2.55),
        gridspec_kw={"wspace": 0.34},
    )
    x = np.arange(3, dtype=float)
    specs = [
        ("safety", "Collision-free safety (%)", (0.0, 105.0)),
        ("reward", "Average reward", (0.0, 0.90)),
        ("speed", "Average speed (m/s)", (0.0, 11.0)),
    ]
    source_color = "#C00000"
    rgd_color = "#008000"
    for panel, (axis, (metric, ylabel, limits)) in enumerate(zip(axes, specs)):
        source = np.asarray(POLICY_SOURCE[metric], dtype=float)
        if metric == "reward":
            axis.errorbar(
                x,
                source,
                yerr=np.asarray(POLICY_SOURCE["reward_error"]),
                color=source_color,
                linestyle="-",
                marker="o",
                markersize=3.7,
                capsize=2.2,
                label="ASR-RL",
                zorder=3,
            )
        else:
            axis.plot(
                x,
                source,
                color=source_color,
                linestyle="-",
                marker="o",
                markersize=3.7,
                label="ASR-RL",
                zorder=3,
            )

        local_values = [entry[0] for entry in local[metric]]
        local_lows = [entry[1] for entry in local[metric]]
        local_highs = [entry[2] for entry in local[metric]]
        axis.errorbar(
            x,
            local_values,
            yerr=interval_errors(local_values, local_lows, local_highs),
            color=rgd_color,
            linestyle="--",
            marker="o",
            markersize=3.7,
            capsize=2.2,
            label="RGD",
            zorder=4,
        )
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        style_source_axis(axis)

        axis.set_xticks(x, ["A", "B", "C"])
        axis.set_xlabel("Test scenario")
    legend = axes[0].legend(
        loc="lower left",
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=2.1,
        columnspacing=1.0,
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.29)
    for axis, title in zip(
        axes,
        (
            "(a) Collision-free safety",
            "(b) Average reward",
            "(c) Average speed",
        ),
    ):
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            0.035,
            title,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )
    outputs, _ = save_figure_triplet(
        figure, FIGURES / "published_policy_transfer_profile", png_dpi=600
    )
    return outputs


def generate_bua_figure(rows: list[dict[str, str]]) -> list[Path]:
    task_rows = [row for row in rows if row["split"] == "scenario"]
    density_rows = [row for row in rows if row["split"] == "density"]
    by_task = grouped(task_rows, "task")
    by_setting = grouped(density_rows, "setting")
    if sorted(by_task) != sorted(BUA_TASKS) or len(by_setting) != 3:
        raise RuntimeError("Unexpected BUA task or density registry")

    def rates(selected: list[dict[str, str]], field: str) -> tuple[float, float, float]:
        return wilson(sum(int(row[field]) for row in selected), len(selected))

    task_success = [rates(by_task[task], "destination_success") for task in BUA_TASKS]
    density_keys = ["traffic density 1.0", "traffic density 2.0", "traffic density 3.0"]
    density_success = [rates(by_setting[key], "destination_success") for key in density_keys]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.50, 3.50),
        gridspec_kw={"hspace": 0.58},
    )
    rgd_color = "#0072B2"
    panels = [
        (
            axes[0],
            np.arange(5, dtype=float),
            BUA_SOURCE_FULL["tasks"],
            task_success,
            ["Left", "Right", "Straight", "Merge", "Round."],
            "(a) Traffic scenario",
        ),
        (
            axes[1],
            np.arange(3, dtype=float),
            BUA_SOURCE_FULL["density"],
            density_success,
            ["1.0", "2.0", "3.0"],
            "(b) Left-turn traffic density",
        ),
    ]
    for axis, x, source, rgd, labels, title in panels:
        axis.plot(
            x,
            source,
            color="#000080",
            linestyle="-",
            marker="o",
            markersize=3.8,
            label="Behavioral Attention",
            zorder=4,
        )
        values = [entry[0] for entry in rgd]
        lows = [entry[1] for entry in rgd]
        highs = [entry[2] for entry in rgd]
        axis.errorbar(
            x,
            values,
            yerr=interval_errors(values, lows, highs),
            color=rgd_color,
            linestyle="-",
            marker="s",
            markersize=3.5,
            capsize=2.0,
            label="RGD",
            zorder=3,
        )
        axis.set_xticks(x, labels)
        if title.startswith("(b)"):
            # The density-3.0 RGD estimate is 0.601 (Wilson 95% CI extends
            # below 0.60), so the density panel needs a lower bound that
            # displays the estimate and its uncertainty without clipping.
            axis.set_ylim(0.55, 1.025)
            axis.set_yticks([0.60, 0.70, 0.80, 0.90, 1.00])
        else:
            axis.set_ylim(0.64, 1.025)
            axis.set_yticks([0.70, 0.80, 0.90, 1.00])
        style_source_axis(axis)
    axes[0].set_ylabel("Success rate")
    axes[0].legend(
        loc="lower left",
        frameon=False,
        fontsize=6.5,
        handlelength=2.0,
        borderaxespad=0.2,
    )
    figure.subplots_adjust(left=0.18, right=0.98, top=0.985, bottom=0.13)
    for axis, title in ((panel[0], panel[-1]) for panel in panels):
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            position.y0 - 0.085,
            title,
            ha="center",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
    outputs, _ = save_figure_triplet(
        figure, FIGURES / "published_bua_protocol_profile", png_dpi=600
    )
    return outputs


def generate_ctrail_figure(rows: list[dict[str, str]]) -> list[Path]:
    by_task_split: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_task_split[(row["task"], row["split"])].append(row)
    expected = {(task, split) for task in CTRAIL_TASKS for split in ("seen", "unseen")}
    if set(by_task_split) != expected:
        raise RuntimeError(f"Unexpected C-TRAIL registry: {sorted(by_task_split)}")

    local: dict[str, list[tuple[float, float, float]]] = {"seen": [], "unseen": []}
    for split in ("seen", "unseen"):
        for task in CTRAIL_TASKS:
            selected = by_task_split[(task, split)]
            local[split].append(
                wilson(sum(int(row["safety_survival"]) for row in selected), len(selected))
            )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(3.50, 3.45),
        gridspec_kw={"hspace": 0.60},
    )
    x = np.arange(4, dtype=float)
    seen_color = "#0072B2"
    unseen_color = "#C00000"
    for split, color, linestyle, marker, label in (
        ("seen", seen_color, "-", "o", "Seen"),
        ("unseen", unseen_color, "--", "^", "Unseen"),
    ):
        axes[0].errorbar(
            x,
            CTRAIL_SOURCE[split],
            yerr=CTRAIL_SOURCE[f"{split}_ci"],
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.0,
            capsize=2.2,
            label=label,
            zorder=3,
        )
        values = [100.0 * entry[0] for entry in local[split]]
        lows = [100.0 * entry[1] for entry in local[split]]
        highs = [100.0 * entry[2] for entry in local[split]]
        axes[1].errorbar(
            x,
            values,
            yerr=interval_errors(values, lows, highs),
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4.0,
            capsize=2.2,
            label=label,
            zorder=3,
        )

    for axis, title, ylabel, limits in (
        (axes[0], "(a) C-TRAIL (GPT-4o)", "Trajectory SR (%)", (78, 91)),
        (axes[1], "(b) RGD (ours)", "Completion (%)", (72, 103)),
    ):
        axis.set_xticks(x, ["Highway", "Merge", "Round.", "Inter."])
        axis.set_ylabel(ylabel)
        axis.set_ylim(*limits)
        style_source_axis(axis)
    axes[0].legend(frameon=False, loc="upper right", ncol=2, handlelength=2.2)
    figure.subplots_adjust(left=0.18, right=0.98, top=0.985, bottom=0.13)
    for axis, title in zip(
        axes,
        ("(a) C-TRAIL (GPT-4o)", "(b) RGD (ours)"),
    ):
        position = axis.get_position()
        figure.text(
            (position.x0 + position.x1) / 2,
            position.y0 - 0.085,
            title,
            ha="center",
            va="top",
            fontsize=7.0,
            fontweight="bold",
        )
    outputs, _ = save_figure_triplet(
        figure, FIGURES / "published_ctrail_transfer_profile", png_dpi=600
    )
    return outputs


def main() -> int:
    for path in (POLICY_CSV, BUA_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not FIGURES.is_dir():
        raise RuntimeError(f"Existing figure directory is missing: {FIGURES}")

    apply_tvt_style()
    policy_rows = read_rows(POLICY_CSV, 3000)
    bua_rows = read_rows(BUA_CSV, 8000)

    outputs: list[Path] = []
    outputs.extend(generate_policy_figure(policy_rows))
    outputs.extend(generate_bua_figure(bua_rows))
    for path in outputs:
        print(path.relative_to(ROOT))
    print("validated episodes: policy=3000, bua=8000, total=11000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
