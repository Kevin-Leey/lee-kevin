"""Generate the absolute traffic-stress heatmap used by the TVT manuscript.

The traffic grid is an exploratory six-cell summary.  This figure intentionally
plots the three observed methods in absolute units rather than plotting paired
differences: the RGD row must remain visible in every panel and the reader can
inspect the outcome level before considering any contrast.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

try:  # Works both when called as a script and when imported by pytest.
    from tvt_figure_utils import apply_tvt_style, save_figure_triplet
except ModuleNotFoundError:  # pragma: no cover - exercised by package imports.
    from tools.tvt_figure_utils import apply_tvt_style, save_figure_triplet


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = ROOT / "results" / "tvt_final_20260718"
DEFAULT_CANDIDATES = (
    ANALYSIS_ROOT
    / "table_vii_progress_guard_v2_analysis_full"
    / "lane_density_transfer_summary.csv",
    ANALYSIS_ROOT
    / "table_vii_progress_guard_v2_analysis"
    / "lane_density_transfer_summary.csv",
)
DEFAULT_OUTPUT = ROOT / "paper" / "figures" / "fig6_traffic_stress_heatmap"

GROUP_ORDER = ("rgd_fixed_policy", "risk_budget", "always_fast")
DISPLAY_LABELS = {
    "rgd_fixed_policy": "RGD (ours)",
    "risk_budget": "TTC-risk",
    "always_fast": "Fast-only",
}
SOURCE_LABELS = {
    "rgd_fixed_policy": "RGD",
    "risk_budget": "TTC-risk",
    "always_fast": "Fast-only",
}
EXPECTED_CELLS = tuple(
    (lanes, density)
    for lanes in (4, 5, 6)
    for density in (2.0, 3.0)
)
EXPECTED_SEEDS = 30
RGD_COLOR = "#0072B2"


@dataclass(frozen=True)
class MetricSpec:
    """Description of one absolute cell metric."""

    key: str
    title: str
    scale: float
    decimals: int
    cmap: str
    domain_min: float
    domain_max: float | None
    tick_step: float


METRICS = (
    MetricSpec(
        "success_rate",
        r"(a) Suc. (%) $\uparrow$",
        100.0,
        1,
        "viridis",
        0.0,
        100.0,
        10.0,
    ),
    MetricSpec(
        "collision_rate",
        r"(b) Coll. (%) $\downarrow$",
        100.0,
        1,
        "magma_r",
        0.0,
        100.0,
        10.0,
    ),
    MetricSpec(
        "distance_all_episode_m",
        r"(c) Dis. (m) $\uparrow$",
        1.0,
        1,
        "cividis",
        0.0,
        None,
        50.0,
    ),
)


REQUIRED_COLUMNS = {
    "group",
    "label",
    "lanes_count",
    "vehicles_density",
    "seeds",
    "distance_all_episode_m",
    "success_count",
    "success_rate",
    "collisions",
    "collision_rate",
}


def default_input() -> Path:
    """Return the canonical summary, with an older analysis path as fallback."""

    for path in DEFAULT_CANDIDATES:
        if path.is_file():
            return path
    return DEFAULT_CANDIDATES[0]


def _finite_float(row: dict[str, str], key: str, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid numeric value for {key} at {context}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite value for {key} at {context}: {value!r}")
    return value


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read and minimally validate the authoritative cell-summary CSV."""

    if not path.is_file():
        raise FileNotFoundError(f"Traffic summary does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def validate_rows(
    rows: list[dict[str, str]],
) -> tuple[
    list[tuple[int, float]],
    list[str],
    dict[tuple[int, float, str], dict[str, str]],
]:
    """Validate the complete 3-method by 6-cell factorial summary.

    The returned index is keyed by ``(lanes, density, group)``.  Validation is
    deliberately strict so a partial or mixed analysis cannot silently become
    a manuscript figure.
    """

    indexed: dict[tuple[int, float, str], dict[str, str]] = {}
    observed_groups: set[str] = set()
    observed_cells: set[tuple[int, float]] = set()
    for row_number, row in enumerate(rows, start=2):
        group = row.get("group", "")
        context = f"row {row_number} ({group or 'unknown group'})"
        if group not in GROUP_ORDER:
            raise RuntimeError(f"Unexpected method group at {context}: {group!r}")
        if row.get("label") != SOURCE_LABELS[group]:
            raise RuntimeError(
                f"Method label/group mismatch at {context}: "
                f"label={row.get('label')!r}, expected={SOURCE_LABELS[group]!r}"
            )
        try:
            lanes = int(row["lanes_count"])
            density = float(row["vehicles_density"])
            seeds = int(row["seeds"])
            success_count = int(row["success_count"])
            collision_count = int(row["collisions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid integer/cell metadata at {context}") from exc
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(
                f"Expected {EXPECTED_SEEDS} seeds at {context}, observed {seeds}"
            )
        if lanes <= 0 or density <= 0:
            raise RuntimeError(f"Invalid traffic cell at {context}: {lanes}, {density}")
        if not 0 <= success_count <= seeds:
            raise RuntimeError(f"Invalid success count at {context}: {success_count}")
        if not 0 <= collision_count <= seeds:
            raise RuntimeError(f"Invalid collision count at {context}: {collision_count}")

        success_rate = _finite_float(row, "success_rate", context)
        collision_rate = _finite_float(row, "collision_rate", context)
        distance = _finite_float(row, "distance_all_episode_m", context)
        if not 0 <= success_rate <= 1 or not 0 <= collision_rate <= 1:
            raise RuntimeError(
                f"Rates must lie in [0, 1] at {context}: "
                f"success={success_rate}, collision={collision_rate}"
            )
        if distance < 0:
            raise RuntimeError(f"Distance must be non-negative at {context}: {distance}")
        if not math.isclose(success_rate, success_count / seeds, abs_tol=1e-10):
            raise RuntimeError(
                f"Success count/rate mismatch at {context}: "
                f"count={success_count}, rate={success_rate}"
            )
        if not math.isclose(collision_rate, collision_count / seeds, abs_tol=1e-10):
            raise RuntimeError(
                f"Collision count/rate mismatch at {context}: "
                f"count={collision_count}, rate={collision_rate}"
            )

        key = (lanes, density, group)
        if key in indexed:
            raise RuntimeError(f"Duplicate traffic summary row: {key}")
        indexed[key] = row
        observed_groups.add(group)
        observed_cells.add((lanes, density))

    expected_groups = set(GROUP_ORDER)
    if observed_groups != expected_groups:
        raise RuntimeError(
            f"Expected exactly methods {sorted(expected_groups)}, "
            f"found {sorted(observed_groups)}"
        )
    if observed_cells != set(EXPECTED_CELLS):
        raise RuntimeError(
            f"Expected traffic cells {list(EXPECTED_CELLS)}, "
            f"found {sorted(observed_cells)}"
        )
    expected_rows = len(EXPECTED_CELLS) * len(GROUP_ORDER)
    if len(indexed) != expected_rows:
        raise RuntimeError(
            f"Expected complete {len(EXPECTED_CELLS)}x{len(GROUP_ORDER)} summary, "
            f"found {len(indexed)} rows"
        )
    return list(EXPECTED_CELLS), list(GROUP_ORDER), indexed


def metric_matrix(
    cells: list[tuple[int, float]],
    groups: list[str],
    indexed: dict[tuple[int, float, str], dict[str, str]],
    spec: MetricSpec,
) -> np.ndarray:
    """Return one method-by-cell matrix in the display unit for ``spec``."""

    return np.asarray(
        [
            [
                _finite_float(indexed[(lanes, density, group)], spec.key, "indexed row")
                * spec.scale
                for lanes, density in cells
            ]
            for group in groups
        ],
        dtype=float,
    )


def _limits(values: np.ndarray, spec: MetricSpec) -> tuple[float, float]:
    """Choose stable, padded limits while retaining the metric's natural domain."""

    low = float(np.min(values))
    high = float(np.max(values))
    if math.isclose(low, high):
        pad = max(abs(low) * 0.05, spec.tick_step)
    else:
        pad = max((high - low) * 0.05, spec.tick_step * 0.2)
    lower = math.floor((low - pad) / spec.tick_step) * spec.tick_step
    upper = math.ceil((high + pad) / spec.tick_step) * spec.tick_step
    lower = max(spec.domain_min, lower)
    if spec.domain_max is not None:
        upper = min(spec.domain_max, upper)
    if upper <= lower:
        upper = lower + spec.tick_step
    return lower, upper


def _text_color(image: mpl.image.AxesImage, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#111111" if luminance > 0.56 else "white"


def draw_figure(
    cells: list[tuple[int, float]],
    groups: list[str],
    indexed: dict[tuple[int, float, str], dict[str, str]],
) -> mpl.figure.Figure:
    """Draw three absolute-value panels with an explicit RGD row."""

    figure, axes = plt.subplots(1, len(METRICS), figsize=(7.16, 2.90), squeeze=False)
    axes = list(axes[0])
    cell_labels = [f"L{lanes}/D{density:g}" for lanes, density in cells]
    method_labels = [DISPLAY_LABELS[group] for group in groups]

    for axis, spec in zip(axes, METRICS):
        values = metric_matrix(cells, groups, indexed, spec)
        lower, upper = _limits(values, spec)
        image = axis.imshow(
            values,
            cmap=spec.cmap,
            norm=Normalize(vmin=lower, vmax=upper),
            aspect="auto",
            interpolation="none",
        )

        # White cell dividers preserve the table-like reading order at column size.
        axis.set_xticks(np.arange(-0.5, len(cells), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xticks(np.arange(len(cells)))
        axis.set_xticklabels(cell_labels, rotation=35, ha="right", rotation_mode="anchor")
        axis.set_yticks(np.arange(len(groups)))
        axis.set_yticklabels(method_labels)
        axis.tick_params(axis="both", which="major", length=2.5, width=0.6, pad=2.2)
        axis.tick_params(axis="x", labelsize=6.4)
        axis.tick_params(axis="y", labelsize=6.8)
        axis.set_title(spec.title, fontweight="bold", pad=5.5)
        axis.spines[:].set_visible(False)

        # The blue outline and bold label make the proposed method visible even
        # when the figure is printed in grayscale or read without its caption.
        for column_index in range(len(cells)):
            axis.add_patch(
                Rectangle(
                    (column_index - 0.5, -0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=RGD_COLOR,
                    linewidth=1.55,
                    zorder=4,
                )
            )
        axis.get_yticklabels()[0].set_fontweight("bold")
        axis.get_yticklabels()[0].set_color(RGD_COLOR)

        for row_index in range(len(groups)):
            for column_index in range(len(cells)):
                value = float(values[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.{spec.decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=_text_color(image, value),
                    fontweight="bold" if row_index == 0 else "normal",
                    zorder=5,
                )

        colorbar = figure.colorbar(
            image,
            ax=axis,
            orientation="horizontal",
            fraction=0.065,
            pad=0.24,
            aspect=26,
        )
        colorbar.set_ticks(np.linspace(lower, upper, 3))
        colorbar.ax.tick_params(labelsize=6.0, length=2.0, width=0.5)
        colorbar.outline.set_linewidth(0.45)

    figure.subplots_adjust(left=0.145, right=0.995, bottom=0.29, top=0.90, wspace=0.47)
    return figure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        "--input",
        "--paired",  # Backward-compatible alias for the former generator CLI.
        dest="summary",
        type=Path,
        default=default_input(),
        help="authoritative lane_density_transfer_summary.csv",
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    apply_tvt_style()
    cells, groups, indexed = validate_rows(read_rows(args.summary))
    outputs, png_size = save_figure_triplet(
        draw_figure(cells, groups, indexed), args.output_stem
    )
    print(
        f"Validated {len(cells)} traffic cells x {len(groups)} methods "
        "from the absolute summary."
    )
    print(f"Input: {args.summary.relative_to(ROOT)}")
    for path in outputs:
        print(f"{path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
    print(f"PNG resolution: {png_size[0]}x{png_size[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
