"""Verify exact Fast-policy replay from versioned v12 snapshot bundles."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_release_state_rollouts import (
    _build_fast_config,
    _load_trace,
    _run_branch,
)
from tools.run_v12_branch_labels import _load_snapshots


def _close(left: Any, right: Any, tolerance: float) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(left_value)
        and math.isfinite(right_value)
        and abs(left_value - right_value) <= tolerance
    )


def verify_seed(
    *,
    trace_root: Path,
    protocol: Path,
    scratch_root: Path,
    seed: int,
    horizon: int,
    gamma: float,
    tolerance: float,
) -> Dict[str, Any]:
    records, physical = _load_trace(trace_root, seed)
    snapshot_path = (
        trace_root / "always_fast" / "highway" / f"seed_{seed}" / "snapshots.pkl"
    )
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"seed {seed}: missing snapshot bundle {snapshot_path}")

    import pickle

    with snapshot_path.open("rb") as handle:
        raw_snapshots = pickle.load(handle)
    if not isinstance(raw_snapshots, dict) or not raw_snapshots:
        raise ValueError(f"seed {seed}: empty or invalid snapshot bundle")
    targets = sorted(int(frame) for frame in raw_snapshots)
    snapshots = _load_snapshots(snapshot_path, targets, seed)
    cfg = _build_fast_config(
        protocol,
        seed,
        scratch_root / f"seed_{seed}",
    )

    proposal_mismatches: List[int] = []
    action_mismatches: List[int] = []
    state_mismatches: List[int] = []
    horizon_mismatches: List[int] = []
    for frame, snapshot in snapshots.items():
        branch = _run_branch(snapshot, cfg, seed, None, horizon, gamma)
        trajectory = json.loads(str(branch["branch_trajectory_json"]))
        if int(branch["fast_action"]) != int(records[frame]["predicted_action_id"]):
            proposal_mismatches.append(frame)
        expected_steps = min(horizon, len(physical) - frame)
        if len(trajectory) != expected_steps:
            horizon_mismatches.append(frame)
        for row in trajectory:
            source_frame = int(row["frame"])
            source = physical[source_frame]
            if int(row["effective_action"]) != int(source["action_id"]):
                action_mismatches.append(source_frame)
            state_matches = (
                _close(row["position_x"], source["position_x"], tolerance)
                and _close(row["position_y"], source["position_y"], tolerance)
                and _close(row["speed"], source["speed"], tolerance)
                and int(row["lane_id"]) == int(source["lane_id"])
            )
            if not state_matches:
                state_mismatches.append(source_frame)

    report = {
        "seed": int(seed),
        "snapshot_count": len(snapshots),
        "horizon_steps": int(horizon),
        "proposal_mismatches": sorted(set(proposal_mismatches)),
        "action_mismatches": sorted(set(action_mismatches)),
        "state_mismatches": sorted(set(state_mismatches)),
        "horizon_mismatches": sorted(set(horizon_mismatches)),
    }
    report["passed"] = not any(
        report[field]
        for field in (
            "proposal_mismatches",
            "action_mismatches",
            "state_mismatches",
            "horizon_mismatches",
        )
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--protocol", default=REPO_ROOT / "formal_protocol_v12.yaml", type=Path)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--seeds", required=True, type=int)
    parser.add_argument("--horizon", default=20, type=int)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--tolerance", default=1e-6, type=float)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.seeds < 1 or args.horizon < 1 or args.tolerance < 0.0:
        raise ValueError("seeds and horizon must be positive; tolerance must be nonnegative")
    reports = [
        verify_seed(
            trace_root=args.trace_root.resolve(),
            protocol=args.protocol.resolve(),
            scratch_root=args.scratch_root.resolve(),
            seed=seed,
            horizon=int(args.horizon),
            gamma=float(args.gamma),
            tolerance=float(args.tolerance),
        )
        for seed in range(args.seed_start, args.seed_start + args.seeds)
    ]
    payload = {
        "schema": "v12_snapshot_replay_verification_v1",
        "passed": all(bool(report["passed"]) for report in reports),
        "seed_start": int(args.seed_start),
        "seed_count": int(args.seeds),
        "reports": reports,
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
