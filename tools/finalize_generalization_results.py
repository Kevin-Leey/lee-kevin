"""Finalize generalization tables/figures from completed run rows (real data only)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except Exception:
        return 0.0


def mean_rows(rows: List[Dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(f(r, key) for r in rows) / len(rows)


def summarise_group_rows(rows_path: Path) -> List[Dict[str, Any]]:
    rows = load_csv(rows_path)
    by = defaultdict(list)
    for row in rows:
        by[row.get("env", "")].append(row)
    out = []
    for env, rs in sorted(by.items()):
        out.append(
            {
                "group": rs[0].get("group", rows_path.parent.name),
                "env": env,
                "n": len(rs),
                "success_rate": mean_rows(rs, "success_rate"),
                "collision_free": 1.0 - mean_rows(rs, "collision_rate"),
                "route_completion": mean_rows(rs, "avg_route_completion"),
                "distance": mean_rows(rs, "avg_driving_distance"),
                "speed": mean_rows(rs, "avg_speed_all_frames"),
                "slow_call_rate": mean_rows(rs, "slow_call_rate"),
                "route_preserve": mean_rows(rs, "route_action_preservation_rate"),
                "safety_override": mean_rows(rs, "safety_override_rate"),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # MetaDrive formal
    md_root = Path("results/metadrive_result/formal_run/2026-07-15/metadrive_stress")
    md_rows: List[Dict[str, Any]] = []
    for path in sorted(md_root.glob("*/*_run_rows.csv")):
        md_rows.extend(summarise_group_rows(path))
    write_csv(Path("results/metadrive_result/analysis/metadrive_summary.csv"), md_rows)
    Path("results/metadrive_result/analysis/metadrive_summary.json").write_text(
        json.dumps(md_rows, indent=2), encoding="utf-8"
    )

    # Multi-LLM highway aggregation (qwen3 old + new highway models)
    ml_rows: List[Dict[str, Any]] = []
    # prior qwen3
    q3 = Path(
        "results/multi_llm_probe/formal_run/2026-07-15/multi_llm_generalization/qwen3_8b"
    )
    for path in sorted(q3.glob("*/*_run_rows.csv")):
        for item in summarise_group_rows(path):
            item["model"] = "Qwen3-8B"
            if item["env"] == "highway-v0":
                ml_rows.append(item)
    # new highway multi model
    hw = Path("results/multi_llm_probe/formal_run/2026-07-15/multi_llm_highway")
    label_map = {
        "qwen2_5_7b": "Qwen2.5-7B",
        "qwen3_5_4b": "Qwen3.5-4B",
        "grok_4_5": "Grok-4.5",
        "gpt_5_6_sol": "GPT-5.6-sol",
        "qwen3_8b": "Qwen3-8B",
    }
    if hw.is_dir():
        for model_dir in sorted([p for p in hw.iterdir() if p.is_dir() and p.name != "_logs"]):
            for path in sorted(model_dir.glob("*/*_run_rows.csv")):
                for item in summarise_group_rows(path):
                    item["model"] = label_map.get(model_dir.name, model_dir.name)
                    ml_rows.append(item)

    write_csv(Path("results/multi_llm_probe/analysis/multi_llm_highway_summary.csv"), ml_rows)
    Path("results/multi_llm_probe/analysis/multi_llm_highway_summary.json").write_text(
        json.dumps(ml_rows, indent=2), encoding="utf-8"
    )

    print(json.dumps({"metadrive": md_rows, "multi_llm": ml_rows}, indent=2))


if __name__ == "__main__":
    main()
