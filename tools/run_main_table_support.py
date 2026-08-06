import csv
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from dilu.evaluation.formal_surface import COMPARISON_HEADLINE_FIELDS
from tools.protocol_io import dump_json


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return float(default)
    if isinstance(value, bool):
        return float(int(value))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _unique_fieldnames(fieldnames: Sequence[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for fieldname in fieldnames:
        key = str(fieldname)
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def write_rows_csv(output_path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    resolved = _unique_fieldnames(list(fieldnames or (rows[0].keys() if rows else [])))
    if not resolved:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: Sequence[Dict[str, Any]], field: str) -> float:
    values = [_safe_float(row.get(field), 0.0) for row in rows if row.get(field) not in (None, "")]
    return float(mean(values)) if values else 0.0


def _overall_rows(
    group_rows: Dict[str, Sequence[Dict[str, Any]]],
    group_order: Sequence[str],
    env_order: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metrics = list(COMPARISON_HEADLINE_FIELDS)
    for env_name in env_order:
        for group_name in group_order:
            selected = [row for row in group_rows.get(group_name, []) if str(row.get("env", "") or "") == env_name]
            if not selected:
                continue
            row: Dict[str, Any] = {
                "env": env_name,
                "group": group_name,
                "runs": len(selected),
                "seed_count": len({str(item.get("seed_idx", "") or "") for item in selected}),
            }
            for metric in metrics:
                row[metric] = _aggregate(selected, metric)
            rows.append(row)
    return rows


def _write_markdown(path: Path, rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> None:
    header = ["env", "group", "runs", *metrics]
    lines = [
        "# Overall group comparison",
        "",
        f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        values = [str(row.get("env", "")), str(row.get("group", "")), str(row.get("runs", 0))]
        values.extend(f"{_safe_float(row.get(metric), 0.0):.6f}" for metric in metrics)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overall_comparison_assets(
    bundle_root: Path,
    group_rows: Dict[str, Sequence[Dict[str, Any]]],
    group_order: Sequence[str],
    env_order: Sequence[str],
) -> Dict[str, str]:
    metrics = list(COMPARISON_HEADLINE_FIELDS)
    rows = _overall_rows(group_rows, group_order, env_order)
    csv_path = bundle_root / "overall_group_comparison.csv"
    json_path = bundle_root / "overall_group_comparison.json"
    markdown_path = bundle_root / "overall_group_comparison.md"
    write_rows_csv(csv_path, rows, ["env", "group", "runs", "seed_count", *metrics])
    dump_json(
        json_path,
        {
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "metrics": metrics,
            "rows": rows,
        },
    )
    _write_markdown(markdown_path, rows, metrics)
    return {"csv": str(csv_path), "json": str(json_path), "markdown": str(markdown_path)}
