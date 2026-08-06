"""Run proposal-blind discrepancy query gates beside the frozen factorial.

This sibling bundle never alters the four-arm factorial matrix. Both baseline
arms reuse the evaluation proposal bank, request identities, latencies, Fast
controller, safety stack, budget, and cooldown. Only release authorization
differs between the two baseline arms.
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.discrepancy_query_gate import (  # noqa: E402
    DISCREPANCY_GATE_VERSION,
    FEATURE_SCHEMA_VERSION,
    DiscrepancyQueryGate,
    load_discrepancy_artifact,
)
from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_REPLAY_VERSION,
    FactorialArm,
)
from tools.run_main_table_runtime import (  # noqa: E402
    load_formal_base_config,
    load_formal_protocol,
)
from tools.run_query_release_factorial import (  # noqa: E402
    DEFAULT_PROPOSAL_SOURCE_POLICY,
    DEFAULT_SOURCE,
    _proposal_manifest,
    _run_arm_seed,
    _sha256_file,
    _write_csv,
    _write_json,
    load_proposal_bank,
)


DISCREPANCY_BASELINE_VERSION = "rgd_discrepancy_query_baseline_v1"
DISCREPANCY_BASELINE_RUN_SCHEMA = "rgd_discrepancy_query_baseline_run_v1"
DISCREPANCY_ARMS = (
    FactorialArm("discrepancy_only", True, False),
    FactorialArm("discrepancy_release", True, True),
)


def _baseline_arm_order(seed: int) -> list[FactorialArm]:
    arms = list(DISCREPANCY_ARMS)
    random.Random(20260731 + int(seed)).shuffle(arms)
    return arms


def _run_baseline_seed_block(
    task: Mapping[str, Any],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    seed = int(task["seed"])
    rows = []
    run_order = []
    for order_index, arm in enumerate(_baseline_arm_order(seed)):
        # Constructing from the authenticated JSON for each arm prevents hidden
        # mutable state from crossing paired executions.
        gate = DiscrepancyQueryGate(task["model_artifact"])
        row = _run_arm_seed(
            protocol=task["protocol"],
            base_cfg=task["base_cfg"],
            protocol_path=task["protocol_path"],
            result_root=task["result_root"],
            seed=seed,
            arm=arm,
            proposals=task["proposals"],
            bank_sha256=str(task["bank_sha256"]),
            proposal_source_policy=str(task["proposal_source_policy"]),
            verbose=bool(task["verbose"]),
            query_admission_policy=gate,
        )
        row.update(
            {
                "discrepancy_baseline_version": DISCREPANCY_BASELINE_VERSION,
                "discrepancy_gate_version": DISCREPANCY_GATE_VERSION,
                "discrepancy_gate_artifact_sha256": str(
                    task["model_artifact"]["artifact_sha256"]
                ),
                "model_artifact_sha256": str(
                    task["model_artifact"]["artifact_sha256"]
                ),
                "discrepancy_gate_feature_schema_version": FEATURE_SCHEMA_VERSION,
            }
        )
        rows.append(row)
        run_order.append(
            {"seed": seed, "order": int(order_index), "arm": arm.name}
        )
    return rows, run_order


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument(
        "--latency-profile", choices=("frozen", "stress"), default="frozen"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.seeds <= 0 or args.seed_start < 0 or args.workers <= 0:
        raise ValueError("seed range must be nonnegative and nonempty")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    artifact = load_discrepancy_artifact(args.model_artifact)
    split = dict(artifact["seed_split"])
    training_seeds = {
        int(seed)
        for seed in list(split["fit_seeds"]) + list(split["calibration_seeds"])
    }
    overlap = sorted(set(seeds) & training_seeds)
    if overlap:
        raise ValueError(
            f"evaluation seeds overlap discrepancy fit/calibration blocks: {overlap}"
        )

    protocol = load_formal_protocol(args.protocol)
    base_cfg = load_formal_base_config(protocol, REPO_ROOT / "config.yaml")
    bank = load_proposal_bank(
        args.source_root,
        seeds,
        latency_profile=args.latency_profile,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    proposal_manifest = _proposal_manifest(
        bank,
        source_root=args.source_root,
        latency_profile=args.latency_profile,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    bank_digest = str(proposal_manifest["bank_sha256"])
    args.result_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.result_root / "proposal_bank_manifest.json", proposal_manifest)
    bundled_model_path = args.result_root / "discrepancy_query_gate_model.json"
    _write_json(bundled_model_path, artifact)

    tasks = [
        {
            "protocol": protocol,
            "base_cfg": base_cfg,
            "protocol_path": args.protocol,
            "result_root": args.result_root,
            "seed": int(seed),
            "proposals": bank[seed],
            "bank_sha256": bank_digest,
            "proposal_source_policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
            "model_artifact": artifact,
            "verbose": bool(args.verbose),
        }
        for seed in seeds
    ]
    if int(args.workers) == 1:
        block_results = map(_run_baseline_seed_block, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(int(args.workers), len(tasks)))
        block_results = pool.map(_run_baseline_seed_block, tasks)

    rows = []
    run_order = []
    try:
        for seed_rows, seed_order in block_results:
            rows.extend(seed_rows)
            run_order.extend(seed_order)
            for row in seed_rows:
                print(
                    f"seed={row['seed']} arm={row['arm']} "
                    f"queries={row['issued_queries']} "
                    f"releases={row['release_events']} "
                    f"distinct={row['distinct_actuations']}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    rows.sort(key=lambda row: (int(row["seed"]), str(row["arm"])))
    _write_csv(args.result_root / "discrepancy_episode_results.csv", rows)
    _write_json(
        args.result_root / "discrepancy_run_manifest.json",
        {
            "schema": DISCREPANCY_BASELINE_RUN_SCHEMA,
            "discrepancy_baseline_version": DISCREPANCY_BASELINE_VERSION,
            "discrepancy_gate_version": DISCREPANCY_GATE_VERSION,
            "discrepancy_gate_feature_schema_version": FEATURE_SCHEMA_VERSION,
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "model_artifact_sha256": artifact["artifact_sha256"],
            "protocol_path": str(args.protocol.resolve()),
            "protocol_sha256": _sha256_file(args.protocol),
            "proposal_bank_sha256": bank_digest,
            "latency_profile": args.latency_profile,
            "candidate_source_policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
            "candidate_source_gate_independent": True,
            "seed_start": int(args.seed_start),
            "seed_count": int(args.seeds),
            "evaluation_seeds": seeds,
            "evaluation_training_seed_disjoint": True,
            "arms": [asdict(arm) for arm in DISCREPANCY_ARMS],
            "randomized_block_run_order": run_order,
            "result_rows": len(rows),
            "model_artifact": {
                "path": bundled_model_path.name,
                "file_sha256": _sha256_file(bundled_model_path),
                "source_path": str(args.model_artifact.resolve()),
                "source_file_sha256": _sha256_file(args.model_artifact),
                "artifact_sha256": artifact["artifact_sha256"],
                "training_source_content_sha256": artifact["training_source"][
                    "source_content_sha256"
                ],
                "seed_split": artifact["seed_split"],
                "fit_class_counts": artifact["fit"]["class_counts"],
                "calibration": artifact["calibration"],
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
