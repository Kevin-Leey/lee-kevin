"""Redraw the Hu et al. protocol comparison as a compact TVT figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tvt_figure_utils import OKABE_ITO, apply_tvt_style


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures" / "fig3_hu_protocol_comparison_compact"

METHODS = ("Hu et al. (GPT-4)", "GRAD", "DeepSeek-R1", "RGD (ours)")
SETTINGS = ("4 lanes / 2.0", "4 lanes / 2.5", "5 lanes / 3.0")
SUCCESS_RATES = np.asarray(
    [
        [68.6, 56.2, 33.1],
        [63.2, 49.8, 11.2],
        [67.3, 54.7, 30.5],
        [100.0, 100.0, 80.0],
    ],
    dtype=float,
)
COLORS = (
    OKABE_ITO["vermillion"],
    OKABE_ITO["orange"],
    OKABE_ITO["sky"],
    OKABE_ITO["blue"],
)


def main() -> int:
    apply_tvt_style()
    figure, axis = plt.subplots(figsize=(4.5694, 2.4809))

    x = np.arange(len(SETTINGS), dtype=float)
    width = 0.19
    offsets = (np.arange(len(METHODS), dtype=float) - 1.5) * width

    for method_index, (method, color, offset) in enumerate(
        zip(METHODS, COLORS, offsets)
    ):
        bars = axis.bar(
            x + offset,
            SUCCESS_RATES[method_index],
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=method,
            zorder=3,
        )
        for bar, value in zip(bars, SUCCESS_RATES[method_index]):
            label = f"{value:.1f}%"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.2,
                label,
                ha="center",
                va="bottom",
                color="#222222",
                fontsize=6.1,
                zorder=4,
            )

    axis.set_xlim(-0.58, len(SETTINGS) - 0.42)
    axis.set_ylim(0.0, 110.0)
    axis.set_xticks(x, SETTINGS)
    axis.set_yticks(np.arange(0.0, 101.0, 20.0))
    axis.set_xlabel("Traffic setting")
    axis.set_ylabel("Success rate (%)")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(length=2.5, width=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in (axis.spines["left"], axis.spines["bottom"]):
        spine.set_color("#555555")
        spine.set_linewidth(0.6)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=False,
        fontsize=6.1,
        handlelength=1.3,
        handletextpad=0.45,
        borderaxespad=0.0,
        columnspacing=1.2,
        labelspacing=0.25,
    )

    figure.subplots_adjust(left=0.115, right=0.985, top=0.79, bottom=0.22)
    svg_path = OUTPUT.with_suffix(".svg")
    pdf_path = OUTPUT.with_suffix(".pdf")
    png_path = OUTPUT.with_suffix(".png")
    figure.savefig(
        svg_path,
        metadata={"Creator": "RGD deterministic TVT figure generator", "Date": None},
    )
    figure.savefig(
        pdf_path,
        metadata={
            "Creator": "RGD deterministic TVT figure generator",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        png_path,
        dpi=600,
        metadata={"Software": "RGD deterministic TVT figure generator"},
    )
    plt.close(figure)
    for path in (svg_path, pdf_path, png_path):
        if not path.is_file() or path.stat().st_size < 5_000:
            raise RuntimeError(f"Missing or unexpectedly small output: {path}")
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
