import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_BUNDLE = Path("results/highway_result/formal_run/2026-06-04/01-39-31")
DEFAULT_OUTPUT_DIR = Path("results/tvt_revision_round5/executor_probe_analysis")
GROUPS = [
    "rgd_fixed_policy",
    "kinematic_risk_slow_path",
]
METRICS = [
    "collision_rate",
    "success_rate",
    "slow_call_rate",
    "avg_runtime_per_frame",
    "safety_override_rate",
    "route_action_preservation_rate",
    "independent_selective_routing_gain",
]
EXECUTOR_LABELS = {
    "rgd_fixed_policy": "online_llm",
    "kinematic_risk_slow_path": "kinematic_risk",
}
EXECUTOR_ROLES = {
    "rgd_fixed_policy": "schema-constrained online language executor",
    "kinematic_risk_slow_path": "deterministic kinematic-risk executor",
}


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: Iterable[Dict[str, Any]], metric: str) -> float:
    values = [_safe_float(row.get(metric)) for row in rows]
    return float(mean(values)) if values else 0.0


def _load_group_rows(bundle_root: Path, group: str) -> List[Dict[str, str]]:
    path = bundle_root / group / f"{group}_run_rows.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing run-row file: {path}")
    return _read_csv(path)


def summarise(bundle_root: Path) -> Dict[str, Any]:
    per_group: List[Dict[str, Any]] = []
    per_env: List[Dict[str, Any]] = []
    for group in GROUPS:
        rows = _load_group_rows(bundle_root, group)
        envs = sorted({str(row.get("env", "")) for row in rows})
        gate_consistent = all(
            str(row.get("rgd_enable_corridor_gate", "")).lower() == "true"
            and str(row.get("rgd_enable_budget_gate", "")).lower() == "true"
            and str(row.get("rgd_enable_margin_gate", "")).lower() == "true"
            and str(row.get("rgd_enable_heuristic_gate", "")).lower() == "true"
            for row in rows
        )
        group_summary: Dict[str, Any] = {
            "group": group,
            "executor": EXECUTOR_LABELS[group],
            "role": EXECUTOR_ROLES[group],
            "runs": len(rows),
            "env_count": len(envs),
            "seed_count": len({str(row.get("seed_idx", "")) for row in rows}),
            "same_asro_gate": bool(gate_consistent),
        }
        for metric in METRICS:
            group_summary[metric] = _mean(rows, metric)
        per_group.append(group_summary)
        for env in envs:
            selected = [row for row in rows if str(row.get("env", "")) == env]
            env_row: Dict[str, Any] = {
                "env": env,
                "group": group,
                "executor": EXECUTOR_LABELS[group],
                "runs": len(selected),
                "same_asro_gate": bool(gate_consistent),
            }
            for metric in METRICS:
                env_row[metric] = _mean(selected, metric)
            per_env.append(env_row)
    return {
        "bundle_root": str(bundle_root),
        "design": {
            "question": "Whether the RGD route layer remains executable when the slow executor is swapped.",
            "unit": "closed-loop episode seed",
            "fixed_factor": "ASRO recoverability gate and downstream safety arbitration",
            "varied_factor": "slow executor implementation",
            "scope": "executor-swap diagnostic, not a public leaderboard or model-training result",
        },
        "per_group": per_group,
        "per_environment": per_env,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarise the executor-agnostic RGD probe from locked run rows.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarise(args.bundle_root)
    output_dir = args.output_dir
    group_path = output_dir / "executor_agnostic_probe_summary.csv"
    env_path = output_dir / "executor_agnostic_probe_by_env.csv"
    json_path = output_dir / "executor_agnostic_probe.json"
    group_fields = ["group", "executor", "role", "runs", "env_count", "seed_count", "same_asro_gate", *METRICS]
    env_fields = ["env", "group", "executor", "runs", "same_asro_gate", *METRICS]
    _write_csv(group_path, summary["per_group"], group_fields)
    _write_csv(env_path, summary["per_environment"], env_fields)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"group_csv": str(group_path), "env_csv": str(env_path), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
