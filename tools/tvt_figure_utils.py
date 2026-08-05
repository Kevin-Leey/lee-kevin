"""Shared publication style and output validation for deterministic TVT figures."""

from __future__ import annotations

from pathlib import Path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt


OKABE_ITO = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#777777",
}


def apply_tvt_style() -> None:
    """Apply one colorblind-safe, IEEE-sized style across deterministic figures."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "rgd-tvt-deterministic-figures",
            "savefig.transparent": False,
        }
    )


def save_figure_triplet(
    figure: mpl.figure.Figure,
    output_stem: Path,
    *,
    png_dpi: int = 600,
) -> tuple[list[Path], tuple[int, int]]:
    """Save PDF/PNG/SVG outputs and validate signatures and PNG resolution."""

    width_in, height_in = figure.get_size_inches()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "png", "svg"):
        path = output_stem.with_suffix(f".{suffix}")
        if suffix == "pdf":
            kwargs = {
                "metadata": {
                    "Creator": "RGD deterministic TVT figure generator",
                    "CreationDate": None,
                    "ModDate": None,
                }
            }
        elif suffix == "svg":
            kwargs = {
                "metadata": {
                    "Creator": "RGD deterministic TVT figure generator",
                    "Date": None,
                }
            }
        else:
            kwargs = {
                "dpi": png_dpi,
                "metadata": {"Software": "RGD deterministic TVT figure generator"},
            }
        figure.savefig(path, bbox_inches="tight", pad_inches=0.035, **kwargs)
        outputs.append(path)
    plt.close(figure)

    by_suffix = {path.suffix: path for path in outputs}
    if set(by_suffix) != {".pdf", ".png", ".svg"}:
        raise RuntimeError(f"Unexpected output set: {sorted(by_suffix)}")
    for path in outputs:
        if not path.is_file() or path.stat().st_size < 5_000:
            raise RuntimeError(f"Missing or unexpectedly small output: {path}")

    if not by_suffix[".pdf"].read_bytes().startswith(b"%PDF"):
        raise RuntimeError("PDF signature check failed")
    png_header = by_suffix[".png"].read_bytes()[:24]
    if png_header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError("PNG signature check failed")
    width_px, height_px = struct.unpack(">II", png_header[16:24])
    min_width = int(width_in * png_dpi * 0.70)
    min_height = int(height_in * png_dpi * 0.70)
    if width_px < min_width or height_px < min_height:
        raise RuntimeError(
            "PNG resolution too small: "
            f"{width_px}x{height_px}; expected at least {min_width}x{min_height}"
        )
    svg_head = by_suffix[".svg"].read_text(encoding="utf-8")[:2_000]
    if "<svg" not in svg_head:
        raise RuntimeError("SVG signature check failed")
    return outputs, (width_px, height_px)
