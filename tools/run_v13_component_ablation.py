"""Run the v13 L/A/H/N query-gate ablation on one frozen proposal bank.

Every arm replays the same gate-independent proposal stream, latency profile,
Fast controller, release guard, safety stack, and seed block. Only the named
query-admission predicates are removed. The ``w/o H,N`` arm supplies the
prespecified H x N interaction check without multiplying the runtime by a
full 2^4 sweep.
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    COMPONENT_ABLATION_ARMS,
    FACTORIAL_REPLAY_VERSION,
    ComponentAblationArm,
    ComponentAblationQueryPolicy,
    FactorialArm,
)
from tools.run_main_table_runtime import (  # noqa: E402
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
    resolve_policy_execution_horizon,
    validate_policy_execution_horizon,
)
from tools.run_query_release_factorial import (  # noqa: E402
    DEFAULT_PROPOSAL_SOURCE_POLICY,
    V13_PROTOCOL_NAME,
    _contract_seed_block,
    _factorial_group_config,
    _proposal_manifest,
    _run_arm_seed,
    _sha256_file,
    _write_csv,
    _write_json,
    load_proposal_bank,
)


COMPONENT_ABLATION_RUN_SCHEMA = "rgd_v13_component_ablation_run_v1"
COMPONENT_ABLATION_VERSION = "v13_serial_gate_lahn_v1"
COMPONENT_ABLATION_DESIGN = (
    "matched_six_arm_lahn_leave_one_out_with_hxn_interaction"
)


def _factorial_arm(spec: ComponentAblationArm) -> FactorialArm:
    return FactorialArm(
        name=str(spec.name),
        query_gate_enabled=True,
        release_guard_enabled=True,
    )


def _arm_order(seed: int) -> list[ComponentAblationArm]:
    arms = list(COMPONENT_ABLATION_ARMS)
    random.Random(20260806 + int(seed)).shuffle(arms)
    return arms


def _run_seed_block(task: Mapping[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    seed = int(task["seed"])
    rows: list[Dict[str, Any]] = []
    order: list[Dict[str, Any]] = []
    for index, spec in enumerate(_arm_order(seed)):
        row = _run_arm_seed(
            protocol=task["protocol"],
            base_cfg=task["base_cfg"],
            protocol_path=Path(task["protocol_path"]),
            result_root=Path(task["result_root"]),
            seed=seed,
            arm=_factorial_arm(spec),
            proposals=task["proposals"],
            bank_sha256=str(task["bank_sha256"]),
            proposal_source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
            verbose=bool(task["verbose"]),
            query_admission_policy=ComponentAblationQueryPolicy(spec),
        )
        row.update(
            {
                "component_ablation_arm": str(spec.name),
                "component_ablation_display_name": str(spec.display_name),
                "component_ablation_removed_components": ";".join(
                    spec.removed_components
                ),
            }
        )
        rows.append(row)
        order.append(
            {
                "seed": seed,
                "order": int(index),
                "arm": str(spec.name),
                "removed_components": list(spec.removed_components),
            }
        )
    return rows, order


def _validate_formal_component_preflight(
    *,
    protocol: Mapping[str, Any],
    seeds: Sequence[int],
    latency_profile: str,
    fixed_latency_steps: int | None,
    predicted_latency_s: float | None,
) -> Mapping[str, Any] | None:
    if str(protocol.get("protocol_name", "") or "") != V13_PROTOCOL_NAME:
        return None
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    contract = dict(submission.get("component_ablation", {}) or {})
    if contract.get("design") != COMPONENT_ABLATION_DESIGN:
        raise ValueError("formal component-ablation design drift")
    expected_arms = tuple(spec.name for spec in COMPONENT_ABLATION_ARMS)
    if tuple(str(value) for value in contract.get("arms", ())) != expected_arms:
        raise ValueError("formal component-ablation arm order drift")
    if tuple(int(seed) for seed in seeds) != _contract_seed_block(
        contract, "seed_range"
    ):
        raise ValueError("formal component-ablation seed cohort drift")
    if str(latency_profile) != str(contract.get("latency_profile", "") or ""):
        raise ValueError("formal component-ablation latency profile drift")
    if fixed_latency_steps != int(contract.get("fixed_delay_steps", -1)):
        raise ValueError("formal component-ablation fixed-delay drift")
    expected_prediction = float(contract.get("predicted_latency_s", -1.0))
    if predicted_latency_s is None or float(predicted_latency_s) != expected_prediction:
        raise ValueError("formal component-ablation predicted-latency drift")
    if contract.get("candidate_source_policy") != DEFAULT_PROPOSAL_SOURCE_POLICY:
        raise ValueError("formal component-ablation candidate-source drift")
    return contract


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=6000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument(
        "--latency-profile",
        choices=(
            "frozen",
            "fixed",
            "jitter",
            "burst",
            "drop",
            "out_of_order",
            "stress",
        ),
        default="fixed",
    )
    parser.add_argument("--fixed-delay-steps", type=int, default=17)
    parser.add_argument(
        "--predicted-latency-s",
        type=float,
        default=1.7,
        help="Gate-visible service-time prediction in seconds.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed_start < 0 or args.seeds <= 0 or args.workers <= 0:
        raise ValueError("seed range and worker count must be positive")
    if args.latency_profile == "fixed" and (
        args.fixed_delay_steps is None or args.fixed_delay_steps < 0
    ):
        raise ValueError("fixed latency replay requires nonnegative --fixed-delay-steps")
    if args.latency_profile != "fixed" and args.fixed_delay_steps is not None:
        raise ValueError("--fixed-delay-steps requires --latency-profile fixed")
    if args.predicted_latency_s is not None and args.predicted_latency_s < 0.0:
        raise ValueError("predicted latency must be nonnegative")
    protocol_path = Path(args.protocol).resolve()
    source_root = Path(args.source_root).resolve()
    result_root = Path(args.result_root).resolve()
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    protocol = load_formal_protocol(protocol_path)
    base_cfg = load_formal_base_config(protocol, REPO_ROOT / "config.yaml")
    component_contract = _validate_formal_component_preflight(
        protocol=protocol,
        seeds=seeds,
        latency_profile=str(args.latency_profile),
        fixed_latency_steps=args.fixed_delay_steps,
        predicted_latency_s=args.predicted_latency_s,
    )
    bank = load_proposal_bank(
        source_root,
        seeds,
        latency_profile=str(args.latency_profile),
        fixed_latency_steps=args.fixed_delay_steps,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    profile_steps = [
        int(record.latency_steps)
        for records in bank.values()
        for record in records.values()
    ]
    if not profile_steps:
        raise ValueError("component ablation requires a nonempty latency profile")
    predicted_latency_s = (
        float(args.predicted_latency_s)
        if args.predicted_latency_s is not None
        else float(median(profile_steps)) / 10.0
    )
    base_cfg = dict(base_cfg)
    base_cfg["rgd_predicted_slow_latency_s"] = predicted_latency_s
    base_cfg["rgd_predicted_slow_latency_source"] = (
        "cli_override"
        if args.predicted_latency_s is not None
        else "proposal_bank_median"
    )
    sample_arm = _factorial_arm(COMPONENT_ABLATION_ARMS[0])
    sample_group_cfg = _factorial_group_config(
        protocol,
        sample_arm,
        predicted_latency_s=predicted_latency_s,
    )
    sample_cfg = build_group_config(
        base_cfg,
        "component_ablation_full",
        sample_group_cfg,
        "highway-v0",
        1,
        result_root / "full" / f"seed_{int(seeds[0])}",
        protocol,
    )
    if component_contract is not None:
        execution_horizon = validate_policy_execution_horizon(
            sample_cfg,
            dict(component_contract.get("execution_contract", {}) or {}),
            context="formal component ablation",
        )
    else:
        execution_horizon = resolve_policy_execution_horizon(
            sample_cfg, context="component ablation"
        )
    proposal_manifest = _proposal_manifest(
        bank,
        source_root=source_root,
        latency_profile=str(args.latency_profile),
        fixed_latency_steps=args.fixed_delay_steps,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    bank_sha256 = str(proposal_manifest["bank_sha256"])
    result_root.mkdir(parents=True, exist_ok=True)
    _write_json(result_root / "proposal_bank_manifest.json", proposal_manifest)
    tasks = [
        {
            "protocol": protocol,
            "base_cfg": base_cfg,
            "protocol_path": str(protocol_path),
            "result_root": str(result_root),
            "seed": seed,
            "proposals": bank[seed],
            "bank_sha256": bank_sha256,
            "verbose": bool(args.verbose),
        }
        for seed in seeds
    ]
    if int(args.workers) == 1:
        blocks = map(_run_seed_block, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(int(args.workers), len(tasks)))
        blocks = pool.map(_run_seed_block, tasks)

    rows: list[Dict[str, Any]] = []
    order: list[Dict[str, Any]] = []
    try:
        for seed_rows, seed_order in blocks:
            rows.extend(seed_rows)
            order.extend(seed_order)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    rows.sort(key=lambda row: (int(row["seed"]), str(row["arm"])))
    _write_csv(result_root / "component_ablation_episode_results.csv", rows)
    _write_json(
        result_root / "component_ablation_run_manifest.json",
        {
            "schema": COMPONENT_ABLATION_RUN_SCHEMA,
            "component_ablation_version": COMPONENT_ABLATION_VERSION,
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "method_version": str(
                dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                    "rgd_method_version", ""
                )
                or ""
            ),
            "query_gate_method_version": str(
                dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                    "query_gate_method_version", ""
                )
                or ""
            ),
            "release_contract_version": str(
                dict(protocol.get("tvt_submission_contract", {}) or {}).get(
                    "release_contract_version", ""
                )
                or ""
            ),
            "design": COMPONENT_ABLATION_DESIGN,
            "protocol_path": str(protocol_path),
            "protocol_sha256": _sha256_file(protocol_path),
            "source_root": str(source_root),
            "proposal_bank_sha256": bank_sha256,
            "latency_profile": str(args.latency_profile),
            "fixed_latency_steps": (
                int(args.fixed_delay_steps)
                if args.fixed_delay_steps is not None
                else None
            ),
            "predicted_latency_s": predicted_latency_s,
            "predicted_latency_source": base_cfg[
                "rgd_predicted_slow_latency_source"
            ],
            "delay_s": [
                float(args.fixed_delay_steps or 0)
                / float(execution_horizon.policy_frequency_hz)
            ],
            "candidate_source_policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
            "candidate_source_gate_independent": True,
            "seed_start": int(args.seed_start),
            "seed_count": int(args.seeds),
            "arms": [asdict(spec) for spec in COMPONENT_ABLATION_ARMS],
            "query_gate_enabled": True,
            "release_guard_enabled": True,
            "randomized_block_run_order": order,
            "result_rows": len(rows),
            **execution_horizon.as_manifest(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
