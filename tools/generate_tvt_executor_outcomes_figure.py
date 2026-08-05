"""Generate a request-outcome composition figure for reasoning executors."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
import numpy as np

from tvt_figure_utils import OKABE_ITO, apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results"
    / "tvt_final_20260718"
    / "multi_llm_rerun2_20260718"
    / "analysis"
    / "executor_diagnostics.csv"
)
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "fig7_executor_outcomes"


@dataclass(frozen=True)
class OutcomeSpec:
    column: str
    label: str
    color: str
    hatch: str


OUTCOMES = (
    OutcomeSpec("valid", "Valid", OKABE_ITO["blue"], ""),
    OutcomeSpec("parse_schema", "Format/schema", OKABE_ITO["orange"], "///"),
    OutcomeSpec("missing_fields", "Missing fields", OKABE_ITO["yellow"], ".."),
    OutcomeSpec("service_transport", "Service/transport", OKABE_ITO["vermillion"], "xx"),
    OutcomeSpec("other", "Other", OKABE_ITO["gray"], "\\\\"),
)


def read_and_validate(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            raise RuntimeError(f"No rows in {path}")
        required = {"label", "attempts", *(outcome.column for outcome in OUTCOMES)}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")

    labels: set[str] = set()
    for row in rows:
        label = row["label"]
        if label in labels:
            raise RuntimeError(f"Duplicate executor row: {label}")
        labels.add(label)
        attempts = int(row["attempts"])
        counts = [int(row[outcome.column]) for outcome in OUTCOMES]
        if attempts <= 0 or any(count < 0 for count in counts):
            raise RuntimeError(f"Invalid request counts for {label}")
        if sum(counts) != attempts:
            raise RuntimeError(
                f"Outcome categories do not sum to attempts for {label}: {sum(counts)} != {attempts}"
            )
    return rows


def draw_figure(rows: list[dict[str, str]]) -> mpl.figure.Figure:
    present = [
        outcome
        for outcome in OUTCOMES
        if any(int(row[outcome.column]) > 0 for row in rows)
    ]
    figure_height = max(2.45, 0.42 * len(rows) + 0.95)
    figure, axis = plt.subplots(figsize=(3.45, figure_height))
    y = np.arange(len(rows) - 1, -1, -1, dtype=float)
    left = np.zeros(len(rows), dtype=float)
    small_callouts = np.zeros(len(rows), dtype=int)
    small_label_threshold = 0.12

    for outcome in present:
        counts = np.array([int(row[outcome.column]) for row in rows], dtype=float)
        attempts = np.array([int(row["attempts"]) for row in rows], dtype=float)
        shares = counts / attempts
        bars = axis.barh(
            y,
            shares,
            left=left,
            height=0.58,
            color=outcome.color,
            edgecolor="white" if not outcome.hatch else "#444444",
            linewidth=0.7,
            hatch=outcome.hatch,
            zorder=3,
        )
        for row_index, (bar, count, share) in enumerate(
            zip(bars, counts.astype(int), shares)
        ):
            if count == 0:
                continue
            center = bar.get_x() + bar.get_width() / 2
            if share < small_label_threshold:
                callout_index = small_callouts[row_index]
                direction = 1 if callout_index % 2 == 0 else -1
                text_x = 1.035 + 0.055 * callout_index
                text_y = bar.get_y() + bar.get_height() / 2 + direction * 0.36
                axis.annotate(
                    str(count),
                    xy=(center, bar.get_y() + bar.get_height() / 2),
                    xytext=(text_x, text_y),
                    ha="left",
                    va="center",
                    fontsize=6.5,
                    color="#333333",
                    arrowprops={
                        "arrowstyle": "-",
                        "color": "#555555",
                        "linewidth": 0.6,
                        "shrinkA": 0,
                        "shrinkB": 1,
                    },
                    annotation_clip=False,
                )
                small_callouts[row_index] += 1
                continue
            label = f"{count} ({share:.0%})"
            axis.text(
                center,
                bar.get_y() + bar.get_height() / 2,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if outcome.column in {"valid", "service_transport"} else "#111111",
                fontweight="bold",
            )
        left += shares

    for position, row in zip(y, rows):
        axis.text(
            1.235,
            position,
            f"n={int(row['attempts'])}",
            ha="right",
            va="center",
            fontsize=6.5,
            color="#444444",
        )

    axis.set_yticks(y, [row["label"] for row in rows])
    axis.set_xlim(0, 1.25)
    axis.set_ylim(-0.75, len(rows) - 0.25)
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.set_xlabel("Share of slow requests")
    axis.set_title("RGD slow-request outcomes", loc="left", fontsize=8.0, pad=8)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles = [
        Patch(
            facecolor=outcome.color,
            edgecolor="white" if not outcome.hatch else "#444444",
            hatch=outcome.hatch,
            label=outcome.label,
        )
        for outcome in present
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.45,
    )
    figure.subplots_adjust(left=0.29, right=0.97, bottom=0.19, top=0.73)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_tvt_style()
    rows = read_and_validate(args.input)
    outputs, png_size = save_figure_triplet(draw_figure(rows), args.output_stem)
    total_attempts = sum(int(row["attempts"]) for row in rows)
    print(f"Validated {len(rows)} executors and {total_attempts} mutually exclusive request outcomes.")
    for path in outputs:
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print(f"PNG resolution: {png_size[0]}x{png_size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
