"""Generate the paired cross-platform effect forest for the TVT paper."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

from tvt_figure_utils import OKABE_ITO, apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAIRED = (
    ROOT / "results" / "cross_platform_qwen3_zero_delay_final" / "paired_contrasts.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "results" / "cross_platform_qwen3_zero_delay_final" / "cross_platform_summary.csv"
)
DEFAULT_AUDIT = (
    ROOT / "results" / "cross_platform_qwen3_zero_delay_final" / "execution_audit.json"
)
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "fig5_cross_platform_forest"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    summary_column: str
    title: str
    xlabel: str
    scale: float = 1.0
    reverse: bool = False


@dataclass(frozen=True)
class SuccessEstimate:
    value: float
    low: float
    high: float
    rgd_successes: int
    fast_successes: int
    seeds: int
    draws: int


PLOT_METRICS = (
    MetricSpec("success_rate", "", "(a) Suc.", "RGD - Fast-only\n(pp)", scale=100.0),
    MetricSpec(
        "collision_rate",
        "collision_rate",
        "(b) Coll.",
        "Fast-only - RGD\n(pp)",
        scale=100.0,
        reverse=True,
    ),
    MetricSpec("distance_m", "distance_m", "(c) Dis.", "RGD - Fast-only\n(m)"),
    MetricSpec("speed_mps", "speed_all_frames_mps", "(d) Spd.", "RGD - Fast-only\n(m/s)"),
)
PAIRED_METRICS = tuple(spec for spec in PLOT_METRICS if spec.key != "success_rate")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"No rows in {path}")
        return rows


def require_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")


def load_summary(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    rows = read_rows(path)
    require_columns(
        path,
        rows,
        {
            "platform",
            "scenario",
            "method",
            "seeds",
            "distance_m",
            "speed_all_frames_mps",
            "collision_rate",
        },
    )
    summary: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["platform"], row["scenario"], row["method"])
        if key in summary:
            raise RuntimeError(f"Duplicate summary row: {key}")
        if int(row["seeds"]) <= 0:
            raise RuntimeError(f"Invalid seed count for {key}: {row['seeds']}")
        summary[key] = row
    return summary


def load_paired(path: Path) -> tuple[list[tuple[str, str]], dict[tuple[str, str, str], dict[str, str]]]:
    rows = read_rows(path)
    require_columns(
        path,
        rows,
        {
            "platform",
            "scenario",
            "metric",
            "rgd",
            "fast_only",
            "rgd_minus_fast",
            "ci_low",
            "ci_high",
            "bootstrap_unit",
            "bootstrap_draws",
        },
    )
    order: list[tuple[str, str]] = []
    paired: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        task = (row["platform"], row["scenario"])
        if task not in order:
            order.append(task)
        key = (*task, row["metric"])
        if key in paired:
            raise RuntimeError(f"Duplicate paired row: {key}")
        value = float(row["rgd_minus_fast"])
        low, high = float(row["ci_low"]), float(row["ci_high"])
        if not low <= value <= high:
            raise RuntimeError(f"Invalid confidence interval for {key}: {low}, {value}, {high}")
        if abs(value - (float(row["rgd"]) - float(row["fast_only"]))) > 2e-6:
            raise RuntimeError(f"Paired difference mismatch for {key}")
        if int(row["bootstrap_draws"]) <= 0:
            raise RuntimeError(f"Invalid bootstrap draw count for {key}")
        paired[key] = row
    return order, paired


def validate_sources(
    tasks: list[tuple[str, str]],
    paired: dict[tuple[str, str, str], dict[str, str]],
    summary: dict[tuple[str, str, str], dict[str, str]],
) -> int | None:
    seed_counts: set[int] = set()
    for platform, scenario in tasks:
        rgd_key = (platform, scenario, "RGD")
        fast_key = (platform, scenario, "Fast-only")
        if rgd_key not in summary or fast_key not in summary:
            raise RuntimeError(f"Missing RGD/Fast-only summary rows for {(platform, scenario)}")
        rgd_summary, fast_summary = summary[rgd_key], summary[fast_key]
        rgd_seeds, fast_seeds = int(rgd_summary["seeds"]), int(fast_summary["seeds"])
        if rgd_seeds != fast_seeds:
            raise RuntimeError(f"Unpaired seed counts for {(platform, scenario)}")
        seed_counts.add(rgd_seeds)
        for spec in PAIRED_METRICS:
            key = (platform, scenario, spec.key)
            if key not in paired:
                raise RuntimeError(f"Missing paired contrast row: {key}")
            row = paired[key]
            if abs(float(row["rgd"]) - float(rgd_summary[spec.summary_column])) > 2e-6:
                raise RuntimeError(f"RGD summary mismatch for {key}")
            if abs(float(row["fast_only"]) - float(fast_summary[spec.summary_column])) > 2e-6:
                raise RuntimeError(f"Fast-only summary mismatch for {key}")
    expected = len(tasks) * len(PAIRED_METRICS)
    if len(paired) != expected:
        raise RuntimeError(f"Expected {expected} paired rows, found {len(paired)}")
    return next(iter(seed_counts)) if len(seed_counts) == 1 else None


def transform(row: dict[str, str], spec: MetricSpec) -> tuple[float, float, float]:
    value = float(row["rgd_minus_fast"])
    low, high = float(row["ci_low"]), float(row["ci_high"])
    if spec.reverse:
        value, low, high = -value, -high, -low
    return spec.scale * value, spec.scale * low, spec.scale * high


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_success_rows(path: Path, expected_rows: int, expected_hash: str) -> dict[int, float]:
    if not path.is_file():
        raise RuntimeError(f"Missing audited run rows: {path}")
    if sha256(path).lower() != expected_hash.lower():
        raise RuntimeError(f"Run-row hash mismatch: {path}")
    rows = read_rows(path)
    require_columns(path, rows, {"seed_idx", "episodes_run", "success_rate"})
    success_by_seed: dict[int, float] = {}
    for row in rows:
        seed = int(row["seed_idx"])
        if seed in success_by_seed:
            raise RuntimeError(f"Duplicate seed {seed} in {path}")
        if int(row["episodes_run"]) != 1:
            raise RuntimeError(f"Expected one episode for seed {seed} in {path}")
        value = float(row["success_rate"])
        if value not in {0.0, 1.0}:
            raise RuntimeError(f"Non-binary success value for seed {seed} in {path}: {value}")
        success_by_seed[seed] = value
    if len(success_by_seed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} audited seed rows in {path}, found {len(success_by_seed)}"
        )
    return success_by_seed


def load_success_effects(
    audit_path: Path,
    paired_path: Path,
    summary_path: Path,
    tasks: list[tuple[str, str]],
) -> dict[tuple[str, str], SuccessEstimate]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    artifacts = audit.get("artifacts", {})
    expected_paired_hash = artifacts.get("paired_contrasts", {}).get("sha256")
    expected_summary_hash = artifacts.get("summary", {}).get("sha256")
    if not expected_paired_hash or sha256(paired_path).lower() != expected_paired_hash.lower():
        raise RuntimeError("Paired-contrast artifact does not match execution_audit.json")
    if not expected_summary_hash or sha256(summary_path).lower() != expected_summary_hash.lower():
        raise RuntimeError("Summary artifact does not match execution_audit.json")

    bootstrap = audit.get("bootstrap", {})
    draws = int(bootstrap.get("draws", 0))
    random_seed = int(bootstrap.get("random_seed", -1))
    if draws <= 0 or random_seed < 0:
        raise RuntimeError("Invalid bootstrap contract in execution_audit.json")
    seed_contract = audit.get("seed_contract", {})
    expected_rows = int(seed_contract.get("unique_seeds_per_method_environment", 0))
    if expected_rows <= 0 or int(seed_contract.get("episodes_per_seed", 0)) != 1:
        raise RuntimeError("Invalid seed contract in execution_audit.json")

    audited_runs: dict[tuple[str, str, str], dict[int, float]] = {}
    for run in audit.get("runs", []):
        method = run.get("method")
        if method not in {"RGD", "Fast-only"}:
            continue
        key = (str(run["platform"]), str(run["scenario"]), str(method))
        if key in audited_runs:
            raise RuntimeError(f"Duplicate audited run: {key}")
        run_rows_path = ROOT / Path(str(run["run_rows"]))
        audited_runs[key] = load_success_rows(
            run_rows_path,
            expected_rows,
            str(run["run_rows_sha256"]),
        )

    rng = np.random.default_rng(random_seed)
    estimates: dict[tuple[str, str], SuccessEstimate] = {}
    for platform, scenario in tasks:
        rgd_key = (platform, scenario, "RGD")
        fast_key = (platform, scenario, "Fast-only")
        if rgd_key not in audited_runs or fast_key not in audited_runs:
            raise RuntimeError(f"Missing audited RGD/Fast-only run for {(platform, scenario)}")
        rgd_rows, fast_rows = audited_runs[rgd_key], audited_runs[fast_key]
        if set(rgd_rows) != set(fast_rows):
            raise RuntimeError(f"Unpaired seed sets for {(platform, scenario)}")
        seeds = sorted(rgd_rows)
        paired_success = np.array(
            [rgd_rows[seed] - fast_rows[seed] for seed in seeds], dtype=float
        )
        bootstrap_draws = rng.choice(
            paired_success,
            size=(draws, len(seeds)),
            replace=True,
        ).mean(axis=1)
        low, high = np.quantile(bootstrap_draws, [0.025, 0.975])
        estimate = SuccessEstimate(
            value=float(paired_success.mean()),
            low=float(low),
            high=float(high),
            rgd_successes=int(sum(rgd_rows.values())),
            fast_successes=int(sum(fast_rows.values())),
            seeds=len(seeds),
            draws=draws,
        )
        if not estimate.low <= estimate.value <= estimate.high:
            raise RuntimeError(f"Invalid success CI for {(platform, scenario)}: {estimate}")
        estimates[(platform, scenario)] = estimate
    if len(audited_runs) != len(tasks) * 2:
        raise RuntimeError(
            f"Expected {len(tasks) * 2} audited method-task runs, found {len(audited_runs)}"
        )
    return estimates


def draw_figure(
    tasks: list[tuple[str, str]],
    paired: dict[tuple[str, str, str], dict[str, str]],
    success: dict[tuple[str, str], SuccessEstimate],
    common_seeds: int | None,
) -> mpl.figure.Figure:
    figure, axes = plt.subplots(1, 4, figsize=(7.16, 3.55), sharey=True)
    y = np.arange(len(tasks) - 1, -1, -1, dtype=float)
    platforms = list(dict.fromkeys(platform for platform, _ in tasks))
    palette = (OKABE_ITO["blue"], OKABE_ITO["vermillion"], OKABE_ITO["green"])
    markers = ("o", "s", "D")
    platform_style = {
        platform: (palette[index % len(palette)], markers[index % len(markers)])
        for index, platform in enumerate(platforms)
    }

    for axis, spec in zip(axes, PLOT_METRICS):
        if spec.key == "success_rate":
            transformed = [
                (
                    spec.scale * success[task].value,
                    spec.scale * success[task].low,
                    spec.scale * success[task].high,
                )
                for task in tasks
            ]
        else:
            transformed = [transform(paired[(*task, spec.key)], spec) for task in tasks]
        extrema = [bound for _, low, high in transformed for bound in (low, high)]
        limit = max(abs(value) for value in extrema)
        limit = limit * 1.14 if limit > 0 else 1.0
        for row_index, (task, position, estimate) in enumerate(zip(tasks, y, transformed)):
            platform, _ = task
            value, low, high = estimate
            color, marker = platform_style[platform]
            if row_index % 2 == 0:
                axis.axhspan(position - 0.46, position + 0.46, color="#F4F4F4", zorder=0)
            axis.errorbar(
                value,
                position,
                xerr=np.array([[value - low], [high - value]]),
                fmt=marker,
                markersize=4.6,
                markerfacecolor="white",
                markeredgewidth=1.0,
                color=color,
                ecolor=color,
                elinewidth=1.0,
                capsize=2.0,
                zorder=3,
            )
        axis.axvline(0, color="#333333", linewidth=0.85, zorder=1)
        axis.set_xlim(-limit, limit)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+g}"))
        axis.grid(axis="x", color="#D9D9D9", linewidth=0.45)
        axis.set_axisbelow(True)
        axis.set_title(spec.title, fontweight="bold", pad=6)
        axis.set_xlabel(spec.xlabel)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_yticks(y, [scenario for _, scenario in tasks])
    axes[0].set_ylim(-0.65, len(tasks) - 0.35)
    for index in range(1, len(tasks)):
        if tasks[index][0] != tasks[index - 1][0]:
            separator = (y[index] + y[index - 1]) / 2
            for axis in axes:
                axis.axhline(separator, color="#777777", linewidth=0.8, linestyle=(0, (2, 2)))

    handles = [
        Line2D(
            [0],
            [0],
            marker=platform_style[platform][1],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=platform_style[platform][0],
            markeredgewidth=1.0,
            markersize=4.8,
            label=platform,
        )
        for platform in platforms
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(handles)),
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.4,
    )
    if common_seeds is not None:
        figure.text(
            0.995,
            0.985,
            f"n = {common_seeds} paired seeds/task",
            ha="right",
            va="top",
            fontsize=6.7,
            color="#444444",
        )
    figure.subplots_adjust(left=0.15, right=0.992, bottom=0.22, top=0.84, wspace=0.42)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired", type=Path, default=DEFAULT_PAIRED)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_tvt_style()
    tasks, paired = load_paired(args.paired)
    summary = load_summary(args.summary)
    common_seeds = validate_sources(tasks, paired, summary)
    success = load_success_effects(args.audit, args.paired, args.summary, tasks)
    if common_seeds is not None and any(
        estimate.seeds != common_seeds for estimate in success.values()
    ):
        raise RuntimeError("Success run-row seed counts do not match the summary")
    outputs, png_size = save_figure_triplet(
        draw_figure(tasks, paired, success, common_seeds), args.output_stem
    )
    print(
        f"Validated {len(tasks)} tasks, {len(paired)} summary effects, "
        f"and {len(success)} run-row success effects "
        f"({next(iter(success.values())).draws} paired-seed bootstrap draws each)."
    )
    for platform, scenario in tasks:
        estimate = success[(platform, scenario)]
        print(
            f"Suc. {platform} {scenario}: RGD {estimate.rgd_successes}/{estimate.seeds}, "
            f"Fast-only {estimate.fast_successes}/{estimate.seeds}; "
            f"delta={100 * estimate.value:+.1f} pp, "
            f"95% CI [{100 * estimate.low:+.1f}, {100 * estimate.high:+.1f}]"
        )
    for path in outputs:
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print(f"PNG resolution: {png_size[0]}x{png_size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
