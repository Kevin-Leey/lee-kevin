"""Validate and analyze the paired query-gate x release-guard factorial.

Simulator seed is the independent unit.  All four arms from a seed are kept
together both when forming contrasts and when drawing percentile-bootstrap
samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_ARMS,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
    FACTORIAL_RUN_SCHEMA,
)
from tools.audit_query_release_factorial import (  # noqa: E402
    AUDIT_SCHEMA,
    audit_bundle,
)


ANALYSIS_SCHEMA = "rgd_query_release_factorial_analysis_v5"
DEFAULT_BOOTSTRAP_DRAWS = 20000
DEFAULT_BOOTSTRAP_SEED = 20260730
DISTINCT_ACTION_METRIC_STAGE = (
    "post_release_guard_and_frame_safety_pre_actuator_bridge"
)
PROPOSAL_RECORD_FIELDS = frozenset(
    {
        "seed",
        "source_frame",
        "request_id",
        "raw_slow_action",
        "latency_steps",
        "outcome",
        "response_text",
        "response_sha256",
    }
)
PROPOSAL_OUTCOMES = frozenset({"valid", "timeout", "failure"})

OUTCOME_METRICS: Tuple[str, ...] = (
    "collision",
    "success_rate",
    "route_completion",
    "episode_reward",
    "driving_distance",
    "avg_speed",
)
LIFECYCLE_METRICS: Tuple[str, ...] = (
    "candidate_queries",
    "issued_queries",
    "query_gate_rejections",
    "scheduled_timeouts",
    "timeouts",
    "failure_events",
    "release_events",
    "primitive_distinct_selections",
    "pending_at_episode_end",
    "pending_timeouts_at_episode_end",
    "snapshot_count",
)
METRICS = OUTCOME_METRICS + LIFECYCLE_METRICS
AUXILIARY_VALIDATED_METRICS: Tuple[str, ...] = ("runtime_per_frame",)
INTEGER_METRICS = frozenset(("collision",) + LIFECYCLE_METRICS)

ARM_BY_NAME = {arm.name: arm for arm in FACTORIAL_ARMS}
ARM_NAMES = tuple(arm.name for arm in FACTORIAL_ARMS)

# Main effects are average enabled-minus-disabled differences over the other
# factor.  The interaction is the unscaled difference-in-differences.
EFFECT_COEFFICIENTS: Mapping[str, Mapping[str, float]] = {
    "query_main_effect": {
        "full": 0.5,
        "query_only": 0.5,
        "release_only": -0.5,
        "neither": -0.5,
    },
    "release_main_effect": {
        "full": 0.5,
        "query_only": -0.5,
        "release_only": 0.5,
        "neither": -0.5,
    },
    "query_x_release_interaction": {
        "full": 1.0,
        "query_only": -1.0,
        "release_only": -1.0,
        "neither": 1.0,
    },
}
EFFECT_DEFINITIONS = {
    "query_main_effect": "0.5*((full-release_only)+(query_only-neither))",
    "release_main_effect": "0.5*((full-query_only)+(release_only-neither))",
    "query_x_release_interaction": "full-query_only-release_only+neither",
}

# These within-seed contrasts answer the operational questions obscured by
# averaged factorial main effects: what each gate adds in the full system and
# what either gate does relative to unconditional proposal execution.
PAIRWISE_CONTRASTS: Mapping[str, Tuple[str, str]] = {
    "full_minus_neither": ("full", "neither"),
    "full_minus_query_only": ("full", "query_only"),
    "full_minus_release_only": ("full", "release_only"),
    "query_only_minus_neither": ("query_only", "neither"),
    "release_only_minus_neither": ("release_only", "neither"),
}

# A result row may contain fewer candidate queries than its seed's proposal
# bank only when the request-level auditor has authenticated terminal right
# censoring for that exact seed-by-arm cell.  Keep this policy literal tied to
# the versioned audit schema so a semantic audit change fails closed here.
RIGHT_CENSORING_AUDIT_POLICY = (
    "proposal records after a verified terminal frame are reported as "
    "right-censored; records at or before that frame require candidate coverage"
)
REQUIRED_AUDIT_CONTRACT_CLAIMS = (
    "proposal_bank_hash_authenticated",
    "proposal_source_files_authenticated",
    "all_candidate_records_bound_to_bank",
    "all_request_ids_lifecycle_closed",
    "release_iff_one_authenticated_snapshot",
    "timeout_failure_pending_snapshot_forbidden",
    "cross_arm_candidate_identity_authenticated",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: Any, field: str) -> float:
    require(not isinstance(value, bool), f"boolean is not a numeric {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    require(math.isfinite(number), f"non-finite {field}: {value!r}")
    return number


def _integer(value: Any, field: str, *, nonnegative: bool = False) -> int:
    number = _finite_number(value, field)
    integer = int(number)
    require(number == integer, f"non-integral {field}: {value!r}")
    if nonnegative:
        require(integer >= 0, f"negative {field}: {value!r}")
    return integer


def _json_integer(value: Any, field: str, *, nonnegative: bool = False) -> int:
    require(type(value) is int, f"invalid integer {field}: {value!r}")
    if nonnegative:
        require(value >= 0, f"negative {field}: {value!r}")
    return value


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"invalid boolean {field}: {value!r}")


def _sha256_json(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal bank is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any, field: str) -> str:
    digest = str(value or "")
    require(
        re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
        f"invalid {field}: {value!r}",
    )
    return digest


def _metric_family(metric: str) -> str:
    return "outcome" if metric in OUTCOME_METRICS else "lifecycle"


def validate_factorial_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Optional[Sequence[int]] = None,
    expected_bank_sha256: Optional[str] = None,
    expected_replay_version: str = FACTORIAL_REPLAY_VERSION,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Validate and normalize a complete seed-by-arm result matrix.

    This function is intentionally independent of filesystem artifacts so tests
    and downstream verifiers can use the exact same fail-closed contract.
    """

    require(bool(rows), "factorial result table is empty")
    normalized: Dict[Tuple[int, str], Dict[str, Any]] = {}
    row_hashes = set()

    for row_index, raw in enumerate(rows):
        require(isinstance(raw, Mapping), f"row {row_index} is not an object")
        arm_name = str(raw.get("arm", ""))
        require(arm_name in ARM_BY_NAME, f"row {row_index}: unknown arm {arm_name!r}")
        seed = _integer(raw.get("seed"), f"row {row_index} seed", nonnegative=True)
        key = (seed, arm_name)
        require(key not in normalized, f"duplicate factorial arm for seed: {key}")

        version = str(raw.get("factorial_replay_version", ""))
        require(
            version == str(expected_replay_version),
            f"{key}: factorial replay version drift",
        )
        arm = ARM_BY_NAME[arm_name]
        query_enabled = _boolean(
            raw.get("query_gate_enabled"), f"{key} query_gate_enabled"
        )
        release_enabled = _boolean(
            raw.get("release_guard_enabled"), f"{key} release_guard_enabled"
        )
        require(
            query_enabled is arm.query_gate_enabled,
            f"{key}: query-gate arm flag mismatch",
        )
        require(
            release_enabled is arm.release_guard_enabled,
            f"{key}: release-guard arm flag mismatch",
        )

        digest = _valid_sha256(
            raw.get("proposal_bank_sha256"), f"{key} proposal_bank_sha256"
        )
        row_hashes.add(digest)
        item: Dict[str, Any] = {
            "factorial_replay_version": version,
            "arm": arm_name,
            "query_gate_enabled": query_enabled,
            "release_guard_enabled": release_enabled,
            "seed": seed,
            "proposal_bank_sha256": digest,
        }
        for metric in METRICS + AUXILIARY_VALIDATED_METRICS:
            value = _finite_number(raw.get(metric), f"{key} {metric}")
            if metric in INTEGER_METRICS:
                require(value == int(value), f"{key}: non-integral {metric}")
                value = int(value)
                if metric in LIFECYCLE_METRICS:
                    require(value >= 0, f"{key}: negative {metric}")
            item[metric] = value

        distinct_alias = _integer(
            raw.get("distinct_actuations"),
            f"{key} distinct_actuations compatibility alias",
            nonnegative=True,
        )
        aligned_alias = _integer(
            raw.get("aligned_distinct_actuations"),
            f"{key} aligned_distinct_actuations compatibility alias",
            nonnegative=True,
        )
        aligned_alias_stage = str(
            raw.get("aligned_distinct_actuations_stage", "") or ""
        )
        metric_stage = str(raw.get("distinct_action_metric_stage", "") or "")
        effect_distinctness_available = _boolean(
            raw.get("effect_distinctness_available"),
            f"{key} effect_distinctness_available",
        )
        require(
            metric_stage == DISTINCT_ACTION_METRIC_STAGE,
            f"{key}: distinct-action metric stage drift",
        )
        require(
            effect_distinctness_available is False,
            f"{key}: episode rows must not claim effect distinctness",
        )
        require(
            distinct_alias == item["primitive_distinct_selections"],
            f"{key}: distinct_actuations alias disagrees with primitive selections",
        )
        require(
            aligned_alias <= item["primitive_distinct_selections"],
            f"{key}: aligned_distinct_actuations exceed primitive selections",
        )
        require(
            release_enabled or aligned_alias == 0,
            f"{key}: aligned_distinct_actuations require an enabled release guard",
        )
        require(
            not aligned_alias_stage
            or aligned_alias_stage == DISTINCT_ACTION_METRIC_STAGE,
            f"{key}: aligned-distinct compatibility stage drift",
        )
        item.update(
            {
                "distinct_actuations": distinct_alias,
                "aligned_distinct_actuations": aligned_alias,
                "aligned_distinct_actuations_stage": aligned_alias_stage,
                "distinct_action_metric_stage": metric_stage,
                "effect_distinctness_available": effect_distinctness_available,
            }
        )

        require(item["collision"] in (0, 1), f"{key}: collision must be binary")
        require(
            0.0 <= item["success_rate"] <= 1.0,
            f"{key}: success_rate must lie in [0, 1]",
        )
        require(
            item["candidate_queries"]
            == item["issued_queries"] + item["query_gate_rejections"],
            f"{key}: candidate-query accounting mismatch",
        )
        require(
            item["scheduled_timeouts"]
            == item["timeouts"] + item["pending_timeouts_at_episode_end"],
            f"{key}: scheduled/terminal/pending timeout accounting mismatch",
        )
        require(
            item["pending_timeouts_at_episode_end"]
            <= item["pending_at_episode_end"],
            f"{key}: pending timeouts exceed pending requests",
        )
        require(
            item["issued_queries"]
            == item["release_events"]
            + item["timeouts"]
            + item["failure_events"]
            + item["pending_at_episode_end"],
            f"{key}: issued/released/pending lifecycle mismatch",
        )
        require(
            item["primitive_distinct_selections"] <= item["release_events"],
            f"{key}: primitive distinct selections exceed release events",
        )
        require(
            item["snapshot_count"] == item["release_events"],
            f"{key}: release/snapshot coverage mismatch",
        )
        normalized[key] = item

    require(len(row_hashes) == 1, "mixed proposal-bank hashes across factorial rows")
    only_hash = next(iter(row_hashes))
    if expected_bank_sha256 is not None:
        require(
            only_hash == _valid_sha256(
                expected_bank_sha256, "expected proposal_bank_sha256"
            ),
            "factorial rows do not match the declared proposal bank",
        )

    observed_seeds = tuple(sorted({seed for seed, _ in normalized}))
    require(bool(observed_seeds), "factorial result table has no seeds")
    if expected_seeds is None:
        seeds = observed_seeds
    else:
        seeds = tuple(
            sorted(
                _integer(seed, "expected seed", nonnegative=True)
                for seed in expected_seeds
            )
        )
        require(len(set(seeds)) == len(seeds), "duplicate expected seeds")
        require(observed_seeds == seeds, "factorial seed cohort does not match manifest")

    expected_cells = {(seed, arm_name) for seed in seeds for arm_name in ARM_NAMES}
    observed_cells = set(normalized)
    missing = sorted(expected_cells - observed_cells)
    extra = sorted(observed_cells - expected_cells)
    require(
        observed_cells == expected_cells,
        f"incomplete factorial matrix: missing={missing}, extra={extra}",
    )
    return normalized


def factorial_effects(arm_values: Mapping[str, float]) -> Dict[str, float]:
    """Return standard 2x2 main effects and difference-in-differences."""

    require(set(arm_values) == set(ARM_NAMES), "factorial contrast requires four arms")
    values = {
        arm: _finite_number(arm_values[arm], f"contrast value for {arm}")
        for arm in ARM_NAMES
    }
    effects = {
        effect: float(sum(coefficients[arm] * values[arm] for arm in ARM_NAMES))
        for effect, coefficients in EFFECT_COEFFICIENTS.items()
    }
    for effect, value in effects.items():
        require(math.isfinite(value), f"non-finite factorial effect: {effect}")
    return effects


def seed_bootstrap_indices(
    n_seeds: int,
    *,
    draws: int,
    bootstrap_seed: int,
) -> np.ndarray:
    """Draw seed blocks once; callers reuse the matrix for every arm and metric."""

    seed_count = _integer(n_seeds, "bootstrap seed count", nonnegative=True)
    draw_count = _integer(draws, "bootstrap draws", nonnegative=True)
    random_seed = _integer(
        bootstrap_seed, "bootstrap random seed", nonnegative=True
    )
    require(seed_count > 0, "bootstrap requires at least one seed")
    require(draw_count > 0, "bootstrap draws must be positive")
    rng = np.random.default_rng(random_seed)
    return rng.integers(0, seed_count, size=(draw_count, seed_count))


def percentile_bootstrap_mean(
    values: Sequence[float],
    bootstrap_indices: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute a mean and deterministic percentile interval from seed draws."""

    data = np.asarray(values, dtype=np.float64)
    require(
        data.ndim == 1 and data.size > 0,
        "bootstrap values must be nonempty and one-dimensional",
    )
    require(bool(np.all(np.isfinite(data))), "bootstrap values contain non-finite entries")
    indices = np.asarray(bootstrap_indices)
    require(
        indices.ndim == 2 and indices.shape[1] == data.size,
        "bootstrap index matrix does not match the seed cohort",
    )
    require(indices.shape[0] > 0, "bootstrap index matrix has no draws")
    require(np.issubdtype(indices.dtype, np.integer), "bootstrap indices must be integers")
    require(bool(np.all((indices >= 0) & (indices < data.size))), "bootstrap index out of range")
    confidence = _finite_number(confidence_level, "confidence level")
    require(0.0 < confidence < 1.0, "confidence level must lie in (0, 1)")
    bootstrap_means = np.mean(data[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [tail, 1.0 - tail])
    return float(np.mean(data)), float(low), float(high)


def analyze_factorial_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
    expected_seeds: Optional[Sequence[int]] = None,
    expected_bank_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Produce JSON-ready arm summaries and paired seed-level contrasts."""

    matrix = validate_factorial_rows(
        rows,
        expected_seeds=expected_seeds,
        expected_bank_sha256=expected_bank_sha256,
    )
    seeds = tuple(sorted({seed for seed, _ in matrix}))
    bootstrap_indices = seed_bootstrap_indices(
        len(seeds), draws=draws, bootstrap_seed=bootstrap_seed
    )
    confidence = _finite_number(confidence_level, "confidence level")
    require(0.0 < confidence < 1.0, "confidence level must lie in (0, 1)")

    arm_summaries: List[Dict[str, Any]] = []
    for arm_name in ARM_NAMES:
        arm = ARM_BY_NAME[arm_name]
        for metric in METRICS:
            values = [float(matrix[(seed, arm_name)][metric]) for seed in seeds]
            point, low, high = percentile_bootstrap_mean(
                values,
                bootstrap_indices,
                confidence_level=confidence,
            )
            arm_summaries.append(
                {
                    "arm": arm_name,
                    "query_gate_enabled": bool(arm.query_gate_enabled),
                    "release_guard_enabled": bool(arm.release_guard_enabled),
                    "metric": metric,
                    "metric_family": _metric_family(metric),
                    "n_seeds": len(seeds),
                    "mean": point,
                    "ci_low": low,
                    "ci_high": high,
                    "total": float(sum(values)),
                    "confidence_level": confidence,
                    "bootstrap_draws": int(draws),
                }
            )

    paired_effects: List[Dict[str, Any]] = []
    seed_effects: List[Dict[str, Any]] = []
    for metric in METRICS:
        by_effect: Dict[str, List[float]] = {
            effect: [] for effect in EFFECT_COEFFICIENTS
        }
        for seed in seeds:
            effects = factorial_effects(
                {
                    arm_name: float(matrix[(seed, arm_name)][metric])
                    for arm_name in ARM_NAMES
                }
            )
            seed_effects.append(
                {
                    "seed": int(seed),
                    "metric": metric,
                    "metric_family": _metric_family(metric),
                    **effects,
                }
            )
            for effect, value in effects.items():
                by_effect[effect].append(value)

        for effect in EFFECT_COEFFICIENTS:
            point, low, high = percentile_bootstrap_mean(
                by_effect[effect],
                bootstrap_indices,
                confidence_level=confidence,
            )
            paired_effects.append(
                {
                    "effect": effect,
                    "contrast": EFFECT_DEFINITIONS[effect],
                    "metric": metric,
                    "metric_family": _metric_family(metric),
                    "n_seed_blocks": len(seeds),
                    "estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": confidence,
                    "bootstrap_draws": int(draws),
                    "bootstrap_unit": "simulator_seed",
                }
            )

        for contrast_name, (left_arm, right_arm) in PAIRWISE_CONTRASTS.items():
            differences = [
                float(matrix[(seed, left_arm)][metric])
                - float(matrix[(seed, right_arm)][metric])
                for seed in seeds
            ]
            point, low, high = percentile_bootstrap_mean(
                differences,
                bootstrap_indices,
                confidence_level=confidence,
            )
            paired_effects.append(
                {
                    "effect": contrast_name,
                    "contrast": f"{left_arm}-{right_arm}",
                    "metric": metric,
                    "metric_family": _metric_family(metric),
                    "n_seed_blocks": len(seeds),
                    "estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": confidence,
                    "bootstrap_draws": int(draws),
                    "bootstrap_unit": "simulator_seed",
                }
            )

    digest = next(iter({row["proposal_bank_sha256"] for row in matrix.values()}))
    return {
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "proposal_bank_sha256": digest,
        "metrics": list(METRICS),
        "outcome_metrics": list(OUTCOME_METRICS),
        "lifecycle_metrics": list(LIFECYCLE_METRICS),
        "lifecycle_metric_semantics": {
            "primitive_distinct_selections": (
                "primitive difference at the common release-selection stage; "
                "not an effect-distinctness estimate"
            ),
            "distinct_action_metric_stage": DISTINCT_ACTION_METRIC_STAGE,
            "effect_distinctness_available": False,
        },
        "auxiliary_validated_metrics": list(AUXILIARY_VALIDATED_METRICS),
        "arm_summaries": arm_summaries,
        "paired_effects": paired_effects,
        "pairwise_contrasts": {
            name: {"left": arms[0], "right": arms[1]}
            for name, arms in PAIRWISE_CONTRASTS.items()
        },
        "seed_effects": seed_effects,
    }


def _validated_audit_reachable_counts(
    audit_report: Mapping[str, Any],
    *,
    expected_replay_version: str,
    seeds: Sequence[int],
    proposal_bank_sha256: str,
    proposal_counts_by_seed: Mapping[int, int],
    proposal_request_ids_by_seed: Mapping[int, Sequence[str]],
    matrix: Mapping[Tuple[int, str], Mapping[str, Any]],
) -> Tuple[Dict[Tuple[int, str], int], Dict[str, Any]]:
    """Bind audited terminal censoring to the normalized result matrix.

    The report is intentionally checked against the manifests and rows already
    authenticated by this function's caller.  This is not a generic report
    parser: accepting a short candidate count without this exact binding would
    turn a lifecycle assertion into an unchecked analysis override.
    """

    require(isinstance(audit_report, Mapping), "request audit report is not an object")
    report = dict(audit_report)
    require(report.get("schema") == AUDIT_SCHEMA, "unexpected request audit schema")
    require(report.get("accepted") is True, "request audit report was not accepted")
    require(
        str(report.get("factorial_replay_version", ""))
        == str(expected_replay_version),
        "request audit replay-version drift",
    )
    require(
        _valid_sha256(
            report.get("proposal_bank_sha256"), "request audit proposal_bank_sha256"
        )
        == proposal_bank_sha256,
        "request audit proposal-bank hash mismatch",
    )
    require(report.get("errors") == [], "accepted request audit contains errors")

    audit_contract = report.get("audit_contract")
    require(isinstance(audit_contract, Mapping), "request audit contract is missing")
    for claim in REQUIRED_AUDIT_CONTRACT_CLAIMS:
        require(
            audit_contract.get(claim) is True,
            f"request audit contract does not authenticate {claim}",
        )
    require(
        audit_contract.get("cross_arm_comparison_scope")
        == "common_reachable_proposals",
        "request audit cross-arm comparison scope drift",
    )
    require(
        audit_contract.get("right_censoring_policy") == RIGHT_CENSORING_AUDIT_POLICY,
        "request audit right-censoring policy drift",
    )

    expected_seed_set = set(seeds)
    expected_cells = {(seed, arm_name) for seed in seeds for arm_name in ARM_NAMES}
    cells = report.get("cells")
    require(isinstance(cells, list), "request audit cells are missing")
    reachable_by_cell: Dict[Tuple[int, str], int] = {}
    censored_total = 0
    for index, raw_cell in enumerate(cells):
        require(
            isinstance(raw_cell, Mapping),
            f"request audit cell {index} is not an object",
        )
        seed = _json_integer(
            raw_cell.get("seed"), f"request audit cell {index} seed", nonnegative=True
        )
        arm_name = str(raw_cell.get("arm", ""))
        key = (seed, arm_name)
        require(seed in expected_seed_set, f"request audit cell outside seed cohort: {key}")
        require(arm_name in ARM_BY_NAME, f"request audit cell has unknown arm: {key}")
        require(key not in reachable_by_cell, f"duplicate request audit cell: {key}")
        require(
            raw_cell.get("accepted") is True,
            f"request audit cell was not accepted: {key}",
        )

        reported_candidates = _json_integer(
            raw_cell.get("candidate_queries"),
            f"request audit {key} candidate_queries",
            nonnegative=True,
        )
        reachable = _json_integer(
            raw_cell.get("reachable_proposal_count"),
            f"request audit {key} reachable_proposal_count",
            nonnegative=True,
        )
        censored = _json_integer(
            raw_cell.get("right_censored_proposal_count"),
            f"request audit {key} right_censored_proposal_count",
            nonnegative=True,
        )
        censored_ids = raw_cell.get("right_censored_proposal_ids")
        require(
            isinstance(censored_ids, list)
            and all(isinstance(item, str) and bool(item) for item in censored_ids),
            f"request audit {key} right-censored proposal IDs are invalid",
        )
        require(
            censored_ids == sorted(set(censored_ids)),
            f"request audit {key} right-censored proposal IDs are not canonical",
        )
        require(
            set(censored_ids).issubset(set(proposal_request_ids_by_seed[seed])),
            f"request audit {key} right-censored proposal IDs are outside the bank",
        )
        require(
            len(censored_ids) == censored,
            f"request audit {key} right-censored proposal count/ID mismatch",
        )
        require(
            reachable + censored == proposal_counts_by_seed[seed],
            f"request audit {key} reachability does not partition the proposal bank",
        )
        require(
            reported_candidates == reachable,
            f"request audit {key} candidate/reachable count mismatch",
        )
        require(
            matrix[key]["candidate_queries"] == reachable,
            f"request audit {key} candidate count does not match result row",
        )
        reachable_by_cell[key] = reachable
        censored_total += censored

    require(
        set(reachable_by_cell) == expected_cells,
        "request audit cell coverage does not match factorial matrix",
    )
    aggregate = report.get("aggregate")
    require(isinstance(aggregate, Mapping), "request audit aggregate is missing")
    require(
        _json_integer(aggregate.get("seed_count"), "request audit aggregate seed_count", nonnegative=True)
        == len(seeds),
        "request audit aggregate seed cohort mismatch",
    )
    require(
        _json_integer(aggregate.get("arm_count"), "request audit aggregate arm_count", nonnegative=True)
        == len(ARM_NAMES),
        "request audit aggregate arm count mismatch",
    )
    require(
        _json_integer(
            aggregate.get("arm_seed_cells"),
            "request audit aggregate arm_seed_cells",
            nonnegative=True,
        )
        == len(expected_cells),
        "request audit aggregate cell count mismatch",
    )
    require(
        _json_integer(
            aggregate.get("proposal_count"),
            "request audit aggregate proposal_count",
            nonnegative=True,
        )
        == sum(proposal_counts_by_seed.values()),
        "request audit aggregate proposal-bank count mismatch",
    )
    aggregate_checks = {
        "candidate_queries": sum(
            int(matrix[(seed, arm_name)]["candidate_queries"])
            for seed, arm_name in expected_cells
        ),
        "reachable_proposal_count": sum(reachable_by_cell.values()),
        "right_censored_proposal_count": censored_total,
    }
    for field, expected in aggregate_checks.items():
        require(
            _json_integer(
                aggregate.get(field), f"request audit aggregate {field}", nonnegative=True
            )
            == expected,
            f"request audit aggregate {field} mismatch",
        )

    return reachable_by_cell, {
        "schema": AUDIT_SCHEMA,
        "sha256": _sha256_json(report),
        "factorial_replay_version": str(expected_replay_version),
        "proposal_bank_sha256": proposal_bank_sha256,
        "right_censored_proposal_count": censored_total,
        "right_censoring_authorized": bool(censored_total),
    }


def validate_bundle_contract(
    rows: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
    *,
    expected_replay_version: str = FACTORIAL_REPLAY_VERSION,
    audit_report: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate manifests, with audited terminal censoring as an opt-in exception.

    Without ``audit_report``, every factorial cell must account for the full
    proposal bank exactly as earlier releases did.  A supplied report must be
    the accepted, current request audit for this same replay cohort and bank.
    """

    require(isinstance(run_manifest, Mapping), "run manifest is not an object")
    require(isinstance(proposal_manifest, Mapping), "proposal manifest is not an object")
    require(
        run_manifest.get("schema") == FACTORIAL_RUN_SCHEMA,
        "unexpected factorial run-manifest schema",
    )
    require(
        proposal_manifest.get("schema") == FACTORIAL_PROPOSAL_SCHEMA,
        "unexpected proposal-bank manifest schema",
    )
    for label, manifest in (
        ("run", run_manifest),
        ("proposal", proposal_manifest),
    ):
        require(
            manifest.get("factorial_replay_version") == str(expected_replay_version),
            f"{label} manifest factorial replay version drift",
        )

    seed_start = _integer(run_manifest.get("seed_start"), "seed_start", nonnegative=True)
    seed_count = _integer(run_manifest.get("seed_count"), "seed_count", nonnegative=True)
    require(seed_count > 0, "seed_count must be positive")
    seeds = tuple(range(seed_start, seed_start + seed_count))
    require(
        _integer(run_manifest.get("result_rows"), "result_rows", nonnegative=True)
        == len(rows),
        "run-manifest result row count mismatch",
    )
    require(len(rows) == seed_count * len(ARM_NAMES), "factorial result matrix size mismatch")

    run_hash = _valid_sha256(
        run_manifest.get("proposal_bank_sha256"), "run proposal_bank_sha256"
    )
    proposal_hash = _valid_sha256(
        proposal_manifest.get("bank_sha256"), "proposal bank_sha256"
    )
    require(run_hash == proposal_hash, "run and proposal manifests use different banks")
    bank_payload = proposal_manifest.get("bank_payload")
    require(isinstance(bank_payload, list), "proposal manifest bank_payload must be a list")
    require(
        _sha256_json(bank_payload) == proposal_hash,
        "proposal-bank payload hash mismatch",
    )
    payload_by_seed: Dict[int, Mapping[str, Any]] = {}
    proposal_counts_by_seed: Dict[int, int] = {}
    proposal_request_ids_by_seed: Dict[int, List[str]] = {}
    observed_block_seeds: List[int] = []
    global_request_ids = set()
    proposal_count = 0
    for block_index, block in enumerate(bank_payload):
        location = f"proposal bank block {block_index}"
        require(isinstance(block, Mapping), f"{location} is not an object")
        require(
            set(block) == {"seed", "records"},
            f"{location}: seed-block schema mismatch",
        )
        seed = _json_integer(
            block.get("seed"), f"{location} seed", nonnegative=True
        )
        require(seed not in payload_by_seed, f"duplicate proposal bank seed: {seed}")
        observed_block_seeds.append(seed)
        records = block.get("records")
        require(isinstance(records, list), f"seed {seed}: proposal records must be a list")
        source_frames = set()
        observed_source_frames: List[int] = []
        for record_index, record in enumerate(records):
            record_location = f"seed {seed} proposal record {record_index}"
            require(
                isinstance(record, Mapping),
                f"{record_location} is not an object",
            )
            require(
                set(record) == PROPOSAL_RECORD_FIELDS,
                f"{record_location}: proposal-record schema mismatch",
            )
            record_seed = _json_integer(
                record.get("seed"),
                f"{record_location} seed",
                nonnegative=True,
            )
            require(
                record_seed == seed,
                f"seed {seed}: proposal record belongs to another seed",
            )
            source_frame = _json_integer(
                record.get("source_frame"),
                f"{record_location} source_frame",
                nonnegative=True,
            )
            require(
                source_frame not in source_frames,
                f"seed {seed}: duplicate proposal source frame {source_frame}",
            )
            request_id = record.get("request_id")
            require(
                isinstance(request_id, str) and bool(request_id.strip()),
                f"{record_location}: empty or non-string request ID",
            )
            require(
                request_id not in global_request_ids,
                f"duplicate proposal request ID: {request_id}",
            )
            raw_action = _json_integer(
                record.get("raw_slow_action"),
                f"{record_location} raw_slow_action",
            )
            require(
                raw_action in range(5),
                f"{record_location}: action outside discrete action universe",
            )
            _json_integer(
                record.get("latency_steps"),
                f"{record_location} latency_steps",
                nonnegative=True,
            )
            outcome = record.get("outcome")
            require(
                isinstance(outcome, str) and outcome in PROPOSAL_OUTCOMES,
                f"{record_location}: invalid response outcome",
            )
            response_text = record.get("response_text")
            require(
                isinstance(response_text, str),
                f"{record_location}: response_text must be a string",
            )
            response_sha256 = _valid_sha256(
                record.get("response_sha256"),
                f"{record_location} response_sha256",
            )
            require(
                hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                == response_sha256,
                f"{record_location}: response text/hash mismatch",
            )
            source_frames.add(source_frame)
            observed_source_frames.append(source_frame)
            global_request_ids.add(request_id)
        require(
            observed_source_frames == sorted(observed_source_frames),
            f"seed {seed}: proposal records are not in canonical frame order",
        )
        require(bool(records), f"seed {seed}: proposal bank has no candidates")
        proposal_count += len(records)
        proposal_counts_by_seed[seed] = len(records)
        proposal_request_ids_by_seed[seed] = [
            str(record["request_id"]) for record in records
        ]
        payload_by_seed[seed] = block
    require(
        _integer(proposal_manifest.get("proposal_count"), "proposal_count", nonnegative=True)
        == proposal_count,
        "proposal manifest count mismatch",
    )
    require(proposal_count > 0, "proposal bank has no candidates")
    require(
        _integer(proposal_manifest.get("seed_count"), "proposal seed_count", nonnegative=True)
        == seed_count,
        "proposal/run seed-count mismatch",
    )
    require(
        set(payload_by_seed) == set(seeds),
        "proposal bank seed blocks do not match the run cohort",
    )
    require(
        tuple(observed_block_seeds) == seeds,
        "proposal bank seed blocks are not in canonical cohort order",
    )
    require(
        str(run_manifest.get("latency_profile", ""))
        == str(proposal_manifest.get("latency_profile", "")),
        "latency-profile mismatch across manifests",
    )
    require(
        _boolean(
            proposal_manifest.get("candidate_source_gate_independent"),
            "proposal candidate_source_gate_independent",
        ),
        "paper-facing factorial requires a gate-independent candidate source",
    )
    require(
        _boolean(
            run_manifest.get("candidate_source_gate_independent"),
            "run candidate_source_gate_independent",
        ),
        "run manifest does not declare a gate-independent candidate source",
    )
    proposal_source_policy = str(
        proposal_manifest.get("candidate_source_policy", "") or ""
    )
    require(
        proposal_source_policy == "scheduled_always_slow",
        "unexpected proposal candidate-source policy",
    )
    require(
        str(run_manifest.get("candidate_source_policy", "") or "")
        == proposal_source_policy,
        "candidate-source policy mismatch across manifests",
    )

    manifest_arms = run_manifest.get("arms")
    require(isinstance(manifest_arms, list), "run-manifest arms must be a list")
    arm_map: Dict[str, Mapping[str, Any]] = {}
    for item in manifest_arms:
        require(isinstance(item, Mapping), "run-manifest arm entry is not an object")
        name = str(item.get("name", ""))
        require(name in ARM_BY_NAME, f"unknown run-manifest arm {name!r}")
        require(name not in arm_map, f"duplicate run-manifest arm {name!r}")
        arm_map[name] = item
    require(set(arm_map) == set(ARM_NAMES), "run-manifest arm set is incomplete")
    for name, arm in ARM_BY_NAME.items():
        require(
            _boolean(arm_map[name].get("query_gate_enabled"), f"{name} manifest query flag")
            is arm.query_gate_enabled,
            f"{name}: run-manifest query flag mismatch",
        )
        require(
            _boolean(arm_map[name].get("release_guard_enabled"), f"{name} manifest release flag")
            is arm.release_guard_enabled,
            f"{name}: run-manifest release flag mismatch",
        )

    order_rows = run_manifest.get("randomized_block_run_order")
    require(isinstance(order_rows, list), "randomized block run order is missing")
    seen_order: Dict[Tuple[int, str], int] = {}
    for item in order_rows:
        require(isinstance(item, Mapping), "run-order entry is not an object")
        seed = _integer(item.get("seed"), "run-order seed", nonnegative=True)
        arm_name = str(item.get("arm", ""))
        require(
            seed in seeds and arm_name in ARM_BY_NAME,
            "run-order cell outside factorial cohort",
        )
        key = (seed, arm_name)
        require(key not in seen_order, f"duplicate run-order cell: {key}")
        seen_order[key] = _integer(item.get("order"), f"run-order index {key}", nonnegative=True)
    expected_cells = {(seed, arm_name) for seed in seeds for arm_name in ARM_NAMES}
    require(set(seen_order) == expected_cells, "randomized block run order is incomplete")
    for seed in seeds:
        orders = {seen_order[(seed, arm_name)] for arm_name in ARM_NAMES}
        require(orders == set(range(len(ARM_NAMES))), f"seed {seed}: invalid arm run order")

    matrix = validate_factorial_rows(
        rows,
        expected_seeds=seeds,
        expected_bank_sha256=run_hash,
        expected_replay_version=str(expected_replay_version),
    )
    audit_summary: Optional[Dict[str, Any]] = None
    if audit_report is None:
        expected_candidates_by_cell = {
            (seed, arm_name): proposal_counts_by_seed[seed]
            for seed in seeds
            for arm_name in ARM_NAMES
        }
    else:
        expected_candidates_by_cell, audit_summary = _validated_audit_reachable_counts(
            audit_report,
            expected_replay_version=str(expected_replay_version),
            seeds=seeds,
            proposal_bank_sha256=run_hash,
            proposal_counts_by_seed=proposal_counts_by_seed,
            proposal_request_ids_by_seed=proposal_request_ids_by_seed,
            matrix=matrix,
        )
    for seed in seeds:
        for arm_name in ARM_NAMES:
            require(
                matrix[(seed, arm_name)]["candidate_queries"]
                == expected_candidates_by_cell[(seed, arm_name)],
                f"{(seed, arm_name)}: candidate count does not match proposal bank",
            )
    return {
        "seeds": seeds,
        "proposal_bank_sha256": run_hash,
        "matrix": matrix,
        "latency_profile": str(run_manifest.get("latency_profile", "")),
        "candidate_source_policy": proposal_source_policy,
        "candidate_source_gate_independent": True,
        "proposal_counts_by_seed": proposal_counts_by_seed,
        "request_audit": audit_summary,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _read_csv(path: Path) -> List[Dict[str, str]]:
    require(path.is_file(), f"missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write an empty table: {path}")
    fields = list(rows[0])
    require(
        all(set(row) == set(fields) for row in rows),
        f"inconsistent CSV fields for {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    require(args.draws > 0, "--draws must be positive")
    require(0.0 < args.confidence_level < 1.0, "--confidence-level must lie in (0, 1)")

    bundle = args.bundle.resolve()
    result_path = bundle / "factorial_episode_results.csv"
    run_manifest_path = bundle / "factorial_run_manifest.json"
    proposal_manifest_path = bundle / "proposal_bank_manifest.json"
    rows = _read_csv(result_path)
    run_manifest = _read_json(run_manifest_path)
    proposal_manifest = _read_json(proposal_manifest_path)
    audit_report = audit_bundle(bundle)
    require(
        str(audit_report.get("bundle", "")) == str(bundle),
        "request audit bundle path mismatch",
    )
    contract = validate_bundle_contract(
        rows,
        run_manifest,
        proposal_manifest,
        audit_report=audit_report,
    )
    analysis = analyze_factorial_rows(
        rows,
        draws=args.draws,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        expected_seeds=contract["seeds"],
        expected_bank_sha256=contract["proposal_bank_sha256"],
    )

    arm_path = args.output_dir / "factorial_arm_summary.csv"
    effects_path = args.output_dir / "factorial_paired_effects.csv"
    analysis_path = args.output_dir / "factorial_analysis.json"
    _atomic_write_csv(arm_path, analysis["arm_summaries"])
    _atomic_write_csv(effects_path, analysis["paired_effects"])

    payload = {
        "schema": ANALYSIS_SCHEMA,
        "accepted": True,
        "source_bundle": str(bundle),
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "latency_profile": contract["latency_profile"],
        "request_audit": contract["request_audit"],
        "design": {
            "independent_unit": "simulator_seed",
            "paired_by_seed": True,
            "full_2x2_factorial": True,
            "candidate_source_policy": contract["candidate_source_policy"],
            "candidate_source_gate_independent": bool(
                contract["candidate_source_gate_independent"]
            ),
            "arms": [asdict(arm) for arm in FACTORIAL_ARMS],
            "effect_coefficients": EFFECT_COEFFICIENTS,
            "effect_definitions": EFFECT_DEFINITIONS,
        },
        "bootstrap": {
            "method": "paired_seed_block_percentile",
            "unit": "simulator_seed",
            "shared_draw_matrix_across_arms_metrics_and_contrasts": True,
            "draws": int(args.draws),
            "seed": int(args.bootstrap_seed),
            "confidence_level": float(args.confidence_level),
            "interval_scope": "pointwise_descriptive",
        },
        **analysis,
        "input_sha256": {
            result_path.name: _sha256_file(result_path),
            run_manifest_path.name: _sha256_file(run_manifest_path),
            proposal_manifest_path.name: _sha256_file(proposal_manifest_path),
        },
        "output_sha256": {
            arm_path.name: _sha256_file(arm_path),
            effects_path.name: _sha256_file(effects_path),
        },
    }
    _atomic_write_json(analysis_path, payload)
    print(
        json.dumps(
            {
                "accepted": True,
                "seed_count": analysis["seed_count"],
                "output": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
