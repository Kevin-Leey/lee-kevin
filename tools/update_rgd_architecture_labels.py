"""Update terminology in the RGD architecture figure."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SVG_PATH = ROOT / "paper" / "figures" / "sturcture.svg"
PDF_PATH = ROOT / "paper" / "figures" / "sturcture.pdf"
EDGE_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
XLINK_NAMESPACE = "http://www.w3.org/1999/xlink"

TEXT_REPLACEMENTS = {
    "RGD allocator": "Query admission",
    "Latency": "Delay",
    "survival": "feasibility",
    "Maneuver": "Action",
    "breadth": "support",
    "Recovery": "Cost",
    "headroom": "advantage",
    "State need": "Scene demand",
    "AND + resource ready": "All conditions + resources",
    "Release authority": "Release validation",
    "Legality": "State checks",
    "Feasibility": "State checks",
    "Distinct from Fast": "Distinct from fast",
    "Recovery advantage": "Cost advantage",
    "Final admissibility": "Validated action",
    "Matched Fast action": "Fast action",
    "VoD corrective set": "VoD labels",
    "Slow executor": "Slow reasoner",
}

POSITIONAL_REPLACEMENTS = {
    ("matrix(1 0 0 1 1693.71 1855)", "Matched"): "Rollout",
    ("matrix(1 0 0 1 1972.72 1855)", "-"): "",
    ("matrix(1 0 0 1 1995.63 1855)", "action comparison"): "comparison",
}

TRANSFORM_REPLACEMENTS = {
    "matrix(1 0 0 1 751.498 724)": "matrix(1 0 0 1 785 724)",
    "matrix(1 0 0 1 785.3 744)": "matrix(1 0 0 1 819 744)",
}


def render_pdf(svg_text: str, width: float, height: float) -> None:
    edge_path = next((path for path in EDGE_PATHS if path.is_file()), None)
    if edge_path is None:
        raise RuntimeError("Microsoft Edge is required to render the architecture PDF")

    page_width = 16.0
    page_height = page_width * height / width
    svg_body = svg_text.split("?>", 1)[-1]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
@page {{ size: {page_width:.4f}in {page_height:.4f}in; margin: 0; }}
html, body {{ width: {page_width:.4f}in; height: {page_height:.4f}in; margin: 0; overflow: hidden; }}
svg {{ display: block; width: {page_width:.4f}in; height: {page_height:.4f}in; }}
</style></head><body>{svg_body}</body></html>"""
    with tempfile.TemporaryDirectory(prefix="rgd-figure-") as directory:
        directory_path = Path(directory)
        html_path = directory_path / "architecture.html"
        profile_path = directory_path / "edge-profile"
        html_path.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                str(edge_path),
                "--headless=new",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--user-data-dir={profile_path}",
                f"--print-to-pdf={PDF_PATH}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )


def main() -> None:
    ET.register_namespace("", SVG_NAMESPACE)
    ET.register_namespace("xlink", XLINK_NAMESPACE)
    tree = ET.parse(SVG_PATH)
    root = tree.getroot()
    width = float(root.get("width", "0"))
    height = float(root.get("height", "0"))
    if width <= 0 or height <= 0:
        raise RuntimeError("Architecture SVG is missing valid width and height")
    root.set("viewBox", f"0 0 {width:g} {height:g}")

    observed_targets: set[str] = set()
    for element in root.iter(f"{{{SVG_NAMESPACE}}}text"):
        transform = element.get("transform", "")
        if transform in TRANSFORM_REPLACEMENTS:
            element.set("transform", TRANSFORM_REPLACEMENTS[transform])
        current = (element.text or "").strip()
        positional_key = (element.get("transform", ""), current)
        if positional_key in POSITIONAL_REPLACEMENTS:
            element.text = POSITIONAL_REPLACEMENTS[positional_key]
            current = (element.text or "").strip()
        elif current in TEXT_REPLACEMENTS:
            element.text = TEXT_REPLACEMENTS[current]
            current = element.text
        observed_targets.add(current)

    missing = sorted(set(TEXT_REPLACEMENTS.values()) - observed_targets)
    if missing:
        raise RuntimeError(f"Architecture labels were not updated: {missing}")

    tree.write(SVG_PATH, encoding="utf-8", xml_declaration=True)
    render_pdf(SVG_PATH.read_text(encoding="utf-8"), width, height)


if __name__ == "__main__":
    main()
