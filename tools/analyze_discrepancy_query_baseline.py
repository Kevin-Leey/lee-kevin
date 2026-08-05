"""Validate and analyze a deployable discrepancy-query baseline.

The comparison reuses the complete seed blocks from the core query-gate x
release-guard factorial.  Simulator seed is the independent unit, and one
shared bootstrap draw matrix is used for every arm, metric, and contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.discrepancy_query_gate import (  # noqa: E402
    DISCREPANCY_GATE_VERSION,
    FEATURE_SCHEMA_VERSION,
    load_discrepancy_artifact,
)
from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
)
from tools.analyze_query_release_factorial import (  # noqa: E402
    ARM_BY_NAME,
    AUXILIARY_VALIDATED_METRICS,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    DISTINCT_ACTION_METRIC_STAGE,
    INTEGER_METRICS,
    LIFECYCLE_METRICS,
    METRICS,
    OUTCOME_METRICS,
    _atomic_write_csv,
    _atomic_write_json,
    _boolean,
    _finite_number,
    _integer,
    _metric_family,
    _read_csv,
    _read_json,
    _sha256_file,
    _sha256_json,
    _valid_sha256,
    percentile_bootstrap_mean,
    require,
    seed_bootstrap_indices,
    validate_bundle_contract,
)


DISCREPANCY_BASELINE_VERSION = "rgd_discrepancy_query_baseline_v2"
DISCREPANCY_RUN_SCHEMA = "rgd_discrepancy_query_baseline_run_v2"
ANALYSIS_SCHEMA = "rgd_discrepancy_query_baseline_analysis_v2"

BASELINE_ARM_FLAGS: Mapping[str, Mapping[str, bool]] = {
    "discrepancy_only": {
        "query_gate_enabled": True,
        "release_guard_enabled": False,
    },
    "discrepancy_release": {
        "query_gate_enabled": True,
        "release_guard_enabled": True,
    },
}
BASELINE_ARM_NAMES = tuple(BASELINE_ARM_FLAGS)
COMPARISON_ARM_NAMES = (
    "query_only",
    "full",
    "discrepancy_only",
    "discrepancy_release",
)
PAIRED_CONTRASTS: Mapping[str, Tuple[str, str]] = {
    "query_only_minus_discrepancy_only": (
        "query_only",
        "discrepancy_only",
    ),
    "full_minus_discrepancy_release": (
        "full",
        "discrepancy_release",
    ),
    "discrepancy_release_minus_discrepancy_only": (
        "discrepancy_release",
        "discrepancy_only",
    ),
    "full_minus_discrepancy_only": (
        "full",
        "discrepancy_only",
    ),
}


def validate_discrepancy_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_seeds: Sequence[int],
    expected_bank_sha256: str,
    expected_model_sha256: str,
) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Validate and normalize the complete baseline seed-by-arm matrix."""

    require(bool(rows), "discrepancy result table is empty")
    seeds = tuple(
        sorted(
            _integer(seed, "expected seed", nonnegative=True)
            for seed in expected_seeds
        )
    )
    require(bool(seeds), "discrepancy baseline requires at least one seed")
    require(len(set(seeds)) == len(seeds), "duplicate expected seeds")
    bank_sha256 = _valid_sha256(
        expected_bank_sha256, "expected proposal_bank_sha256"
    )
    model_sha256 = _valid_sha256(
        expected_model_sha256, "expected model_artifact_sha256"
    )

    normalized: Dict[Tuple[int, str], Dict[str, Any]] = {}
    row_bank_hashes = set()
    row_model_hashes = set()
    for row_index, raw in enumerate(rows):
        require(isinstance(raw, Mapping), f"row {row_index} is not an object")
        arm_name = str(raw.get("arm", ""))
        require(
            arm_name in BASELINE_ARM_FLAGS,
            f"row {row_index}: unknown discrepancy arm {arm_name!r}",
        )
        seed = _integer(raw.get("seed"), f"row {row_index} seed", nonnegative=True)
        key = (seed, arm_name)
        require(key not in normalized, f"duplicate discrepancy arm for seed: {key}")

        require(
            str(raw.get("factorial_replay_version", ""))
            == FACTORIAL_REPLAY_VERSION,
            f"{key}: factorial replay version drift",
        )
        require(
            str(raw.get("discrepancy_baseline_version", ""))
            == DISCREPANCY_BASELINE_VERSION,
            f"{key}: discrepancy baseline version drift",
        )
        flags = BASELINE_ARM_FLAGS[arm_name]
        query_enabled = _boolean(
            raw.get("query_gate_enabled"), f"{key} query_gate_enabled"
        )
        release_enabled = _boolean(
            raw.get("release_guard_enabled"), f"{key} release_guard_enabled"
        )
        require(query_enabled, f"{key}: discrepancy query gate must be enabled")
        require(
            release_enabled is flags["release_guard_enabled"],
            f"{key}: release-guard arm flag mismatch",
        )

        row_bank_sha256 = _valid_sha256(
            raw.get("proposal_bank_sha256"), f"{key} proposal_bank_sha256"
        )
        row_model_sha256 = _valid_sha256(
            raw.get("model_artifact_sha256"), f"{key} model_artifact_sha256"
        )
        row_bank_hashes.add(row_bank_sha256)
        row_model_hashes.add(row_model_sha256)
        item: Dict[str, Any] = {
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "discrepancy_baseline_version": DISCREPANCY_BASELINE_VERSION,
            "model_artifact_sha256": row_model_sha256,
            "arm": arm_name,
            "query_gate_enabled": query_enabled,
            "release_guard_enabled": release_enabled,
            "seed": seed,
            "proposal_bank_sha256": row_bank_sha256,
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
        item.update(
            {
                "distinct_actuations": distinct_alias,
                "aligned_distinct_actuations": aligned_alias,
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
            item["issued_queries"]
            == item["release_events"]
            + item["timeouts"]
            + item["failure_events"]
            + item["pending_at_episode_end"],
            f"{key}: issued/released/pending lifecycle mismatch",
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
            item["primitive_distinct_selections"] <= item["release_events"],
            f"{key}: primitive distinct selections exceed release events",
        )
        require(
            item["snapshot_count"] == item["release_events"],
            f"{key}: release/snapshot coverage mismatch",
        )
        normalized[key] = item

    require(
        row_bank_hashes == {bank_sha256},
        "discrepancy rows do not match the core proposal bank",
    )
    require(
        row_model_hashes == {model_sha256},
        "discrepancy rows do not match the declared model artifact",
    )
    observed_seeds = tuple(sorted({seed for seed, _ in normalized}))
    require(observed_seeds == seeds, "discrepancy seed cohort does not match core")
    expected_cells = {
        (seed, arm_name) for seed in seeds for arm_name in BASELINE_ARM_NAMES
    }
    missing = sorted(expected_cells - set(normalized))
    extra = sorted(set(normalized) - expected_cells)
    require(
        set(normalized) == expected_cells,
        f"incomplete discrepancy matrix: missing={missing}, extra={extra}",
    )
    return normalized


def _validate_proposal_manifest(
    proposal_manifest: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
    expected_bank_sha256: str,
    expected_latency_profile: str,
    expected_source_policy: str,
) -> None:
    require(isinstance(proposal_manifest, Mapping), "proposal manifest is not an object")
    require(
        proposal_manifest.get("schema") == FACTORIAL_PROPOSAL_SCHEMA,
        "unexpected proposal-bank manifest schema",
    )
    require(
        proposal_manifest.get("factorial_replay_version")
        == FACTORIAL_REPLAY_VERSION,
        "proposal manifest factorial replay version drift",
    )
    seeds = tuple(int(seed) for seed in expected_seeds)
    bank_sha256 = _valid_sha256(
        proposal_manifest.get("bank_sha256"), "proposal bank_sha256"
    )
    require(
        bank_sha256 == _valid_sha256(expected_bank_sha256, "core proposal hash"),
        "baseline proposal bank does not match core",
    )
    require(
        str(proposal_manifest.get("latency_profile", ""))
        == expected_latency_profile,
        "baseline proposal latency profile does not match core",
    )
    require(
        str(proposal_manifest.get("candidate_source_policy", "") or "")
        == expected_source_policy,
        "baseline proposal source policy does not match core",
    )
    require(
        _boolean(
            proposal_manifest.get("candidate_source_gate_independent"),
            "proposal candidate_source_gate_independent",
        ),
        "baseline proposal source is not gate-independent",
    )

    bank_payload = proposal_manifest.get("bank_payload")
    require(isinstance(bank_payload, list), "proposal manifest bank_payload must be a list")
    payload_by_seed: Dict[int, Mapping[str, Any]] = {}
    proposal_count = 0
    for block in bank_payload:
        require(isinstance(block, Mapping), "proposal bank seed block is not an object")
        seed = _integer(block.get("seed"), "proposal bank seed", nonnegative=True)
        require(seed not in payload_by_seed, f"duplicate proposal bank seed: {seed}")
        records = block.get("records")
        require(isinstance(records, list), f"seed {seed}: proposal records must be a list")
        for record in records:
            require(isinstance(record, Mapping), f"seed {seed}: proposal record is not an object")
            require(
                _integer(
                    record.get("seed"),
                    f"seed {seed} proposal identity",
                    nonnegative=True,
                )
                == seed,
                f"seed {seed}: proposal record belongs to another seed",
            )
            require(
                "source_artifact" not in record,
                "portable proposal identity must exclude source paths",
            )
        proposal_count += len(records)
        payload_by_seed[seed] = block
    require(
        set(payload_by_seed) == set(seeds),
        "baseline proposal bank seed blocks do not match core",
    )
    require(
        _integer(
            proposal_manifest.get("seed_count"),
            "proposal seed_count",
            nonnegative=True,
        )
        == len(seeds),
        "baseline proposal seed count does not match core",
    )
    require(
        _integer(
            proposal_manifest.get("proposal_count"),
            "proposal_count",
            nonnegative=True,
        )
        == proposal_count,
        "proposal manifest count mismatch",
    )
    require(
        _sha256_json(bank_payload) == bank_sha256,
        "baseline proposal-bank payload hash mismatch",
    )


def validate_discrepancy_bundle_contract(
    rows: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
    *,
    core_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate the baseline bundle against an already validated core bundle."""

    require(isinstance(core_contract, Mapping), "core contract is not an object")
    core_seeds = tuple(
        _integer(seed, "core seed", nonnegative=True)
        for seed in core_contract.get("seeds", ())
    )
    require(bool(core_seeds), "core contract has no seeds")
    core_bank_sha256 = _valid_sha256(
        core_contract.get("proposal_bank_sha256"), "core proposal_bank_sha256"
    )
    core_latency_profile = str(core_contract.get("latency_profile", ""))
    core_source_policy = str(core_contract.get("candidate_source_policy", "") or "")

    require(isinstance(run_manifest, Mapping), "run manifest is not an object")
    require(
        run_manifest.get("schema") == DISCREPANCY_RUN_SCHEMA,
        "unexpected discrepancy run-manifest schema",
    )
    require(
        str(run_manifest.get("factorial_replay_version", ""))
        == FACTORIAL_REPLAY_VERSION,
        "discrepancy run factorial replay version drift",
    )
    require(
        str(run_manifest.get("discrepancy_baseline_version", ""))
        == DISCREPANCY_BASELINE_VERSION,
        "discrepancy run version drift",
    )
    require(
        str(run_manifest.get("discrepancy_gate_version", ""))
        == DISCREPANCY_GATE_VERSION,
        "discrepancy gate version drift",
    )
    require(
        str(run_manifest.get("discrepancy_gate_feature_schema_version", ""))
        == FEATURE_SCHEMA_VERSION,
        "discrepancy feature schema version drift",
    )
    model_sha256 = _valid_sha256(
        run_manifest.get("model_artifact_sha256"), "model_artifact_sha256"
    )
    seed_start = _integer(run_manifest.get("seed_start"), "seed_start", nonnegative=True)
    seed_count = _integer(run_manifest.get("seed_count"), "seed_count", nonnegative=True)
    require(seed_count > 0, "seed_count must be positive")
    manifest_seeds = tuple(range(seed_start, seed_start + seed_count))
    require(manifest_seeds == core_seeds, "discrepancy seed cohort does not match core")
    evaluation_seeds = tuple(
        _integer(seed, "evaluation seed", nonnegative=True)
        for seed in run_manifest.get("evaluation_seeds", ())
    )
    require(evaluation_seeds == core_seeds, "explicit evaluation seed list drift")
    require(
        _boolean(
            run_manifest.get("evaluation_training_seed_disjoint"),
            "evaluation_training_seed_disjoint",
        ),
        "evaluation/training seed split is not declared disjoint",
    )
    require(
        _integer(run_manifest.get("result_rows"), "result_rows", nonnegative=True)
        == len(rows),
        "discrepancy run-manifest result row count mismatch",
    )
    require(
        len(rows) == seed_count * len(BASELINE_ARM_NAMES),
        "discrepancy result matrix size mismatch",
    )

    run_bank_sha256 = _valid_sha256(
        run_manifest.get("proposal_bank_sha256"), "run proposal_bank_sha256"
    )
    require(
        run_bank_sha256 == core_bank_sha256,
        "discrepancy run proposal bank does not match core",
    )
    require(
        str(run_manifest.get("latency_profile", "")) == core_latency_profile,
        "discrepancy latency profile does not match core",
    )
    require(
        str(run_manifest.get("candidate_source_policy", "") or "")
        == core_source_policy,
        "discrepancy source policy does not match core",
    )
    require(
        _boolean(
            run_manifest.get("candidate_source_gate_independent"),
            "run candidate_source_gate_independent",
        ),
        "discrepancy run source is not gate-independent",
    )
    model_manifest = run_manifest.get("model_artifact")
    require(isinstance(model_manifest, Mapping), "model artifact manifest is missing")
    require(
        _valid_sha256(
            model_manifest.get("artifact_sha256"),
            "nested model artifact_sha256",
        )
        == model_sha256,
        "nested model artifact hash mismatch",
    )
    split = model_manifest.get("seed_split")
    require(isinstance(split, Mapping), "model seed split is missing")
    fit_seeds = tuple(
        _integer(seed, "model fit seed", nonnegative=True)
        for seed in split.get("fit_seeds", ())
    )
    calibration_seeds = tuple(
        _integer(seed, "model calibration seed", nonnegative=True)
        for seed in split.get("calibration_seeds", ())
    )
    require(bool(fit_seeds) and bool(calibration_seeds), "model seed split is empty")
    require(
        not (set(fit_seeds) & set(calibration_seeds)),
        "model fit/calibration seed blocks overlap",
    )
    require(
        not (set(core_seeds) & (set(fit_seeds) | set(calibration_seeds))),
        "evaluation cohort overlaps model fit/calibration seeds",
    )
    require(
        str(split.get("unit", "")) == "simulator_seed"
        and _boolean(split.get("disjoint"), "model split disjoint"),
        "model split is not a disjoint simulator-seed split",
    )
    require(
        _boolean(
            split.get("fit_labels_use_slow_fast_query_discrepancy"),
            "fit label contract",
        )
        and not _boolean(
            split.get("calibration_labels_use_slow_fast_query_discrepancy"),
            "calibration label contract",
        ),
        "model label-access split drift",
    )
    calibration = model_manifest.get("calibration")
    require(isinstance(calibration, Mapping), "model calibration manifest is missing")
    require(
        not _boolean(calibration.get("uses_outcomes"), "calibration uses_outcomes")
        and not _boolean(
            calibration.get("uses_discrepancy_labels"),
            "calibration uses_discrepancy_labels",
        ),
        "model threshold calibration leaks labels or outcomes",
    )
    require(
        _integer(
            calibration.get("target_invocations"),
            "calibration target invocations",
            nonnegative=True,
        )
        == _integer(
            calibration.get("achieved_invocations"),
            "calibration achieved invocations",
            nonnegative=True,
        ),
        "model threshold does not match RGD invocation exposure",
    )
    _validate_proposal_manifest(
        proposal_manifest,
        expected_seeds=core_seeds,
        expected_bank_sha256=core_bank_sha256,
        expected_latency_profile=core_latency_profile,
        expected_source_policy=core_source_policy,
    )

    manifest_arms = run_manifest.get("arms")
    require(isinstance(manifest_arms, list), "run-manifest arms must be a list")
    arm_map: Dict[str, Mapping[str, Any]] = {}
    for item in manifest_arms:
        require(isinstance(item, Mapping), "run-manifest arm entry is not an object")
        name = str(item.get("name", ""))
        require(name in BASELINE_ARM_FLAGS, f"unknown discrepancy manifest arm {name!r}")
        require(name not in arm_map, f"duplicate discrepancy manifest arm {name!r}")
        arm_map[name] = item
    require(
        set(arm_map) == set(BASELINE_ARM_NAMES),
        "discrepancy run-manifest arm set is incomplete",
    )
    for name, flags in BASELINE_ARM_FLAGS.items():
        require(
            _boolean(
                arm_map[name].get("query_gate_enabled"),
                f"{name} manifest query flag",
            )
            is flags["query_gate_enabled"],
            f"{name}: run-manifest query flag mismatch",
        )
        require(
            _boolean(
                arm_map[name].get("release_guard_enabled"),
                f"{name} manifest release flag",
            )
            is flags["release_guard_enabled"],
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
            seed in core_seeds and arm_name in BASELINE_ARM_FLAGS,
            "run-order cell outside discrepancy cohort",
        )
        key = (seed, arm_name)
        require(key not in seen_order, f"duplicate run-order cell: {key}")
        seen_order[key] = _integer(
            item.get("order"), f"run-order index {key}", nonnegative=True
        )
    expected_cells = {
        (seed, arm_name) for seed in core_seeds for arm_name in BASELINE_ARM_NAMES
    }
    require(set(seen_order) == expected_cells, "randomized block run order is incomplete")
    for seed in core_seeds:
        orders = {seen_order[(seed, arm_name)] for arm_name in BASELINE_ARM_NAMES}
        require(
            orders == set(range(len(BASELINE_ARM_NAMES))),
            f"seed {seed}: invalid discrepancy arm run order",
        )

    matrix = validate_discrepancy_rows(
        rows,
        expected_seeds=core_seeds,
        expected_bank_sha256=core_bank_sha256,
        expected_model_sha256=model_sha256,
    )
    return {
        "seeds": core_seeds,
        "proposal_bank_sha256": core_bank_sha256,
        "model_artifact_sha256": model_sha256,
        "matrix": matrix,
        "latency_profile": core_latency_profile,
        "candidate_source_policy": core_source_policy,
        "candidate_source_gate_independent": True,
    }


def validate_bundled_model_artifact(
    baseline_bundle: Path,
    run_manifest: Mapping[str, Any],
) -> Tuple[Path, Dict[str, Any]]:
    """Authenticate the self-contained model and its split/calibration provenance."""

    model_manifest = run_manifest.get("model_artifact")
    require(isinstance(model_manifest, Mapping), "model artifact manifest is missing")
    raw_relative = str(model_manifest.get("path", "") or "")
    relative = Path(raw_relative)
    require(
        bool(raw_relative)
        and not relative.is_absolute()
        and ".." not in relative.parts,
        "model artifact path must remain inside the baseline bundle",
    )
    bundle_root = baseline_bundle.resolve()
    model_path = (baseline_bundle / relative).resolve()
    try:
        model_path.relative_to(bundle_root)
    except ValueError as exc:
        raise ValueError("model artifact escapes the baseline bundle") from exc
    require(model_path.is_file(), f"missing bundled model artifact: {model_path}")
    require(
        _valid_sha256(model_manifest.get("file_sha256"), "model file_sha256")
        == _sha256_file(model_path),
        "bundled model file hash mismatch",
    )
    artifact = load_discrepancy_artifact(model_path)
    artifact_sha256 = _valid_sha256(
        artifact.get("artifact_sha256"), "bundled model artifact_sha256"
    )
    require(
        artifact_sha256
        == _valid_sha256(
            run_manifest.get("model_artifact_sha256"),
            "run model_artifact_sha256",
        )
        == _valid_sha256(
            model_manifest.get("artifact_sha256"),
            "nested model artifact_sha256",
        ),
        "bundled model identity does not match the run manifest",
    )
    for field, artifact_value in (
        ("seed_split", artifact.get("seed_split")),
        ("fit_class_counts", dict(artifact.get("fit", {}) or {}).get("class_counts")),
        ("calibration", artifact.get("calibration")),
    ):
        require(
            model_manifest.get(field) == artifact_value,
            f"bundled model {field} provenance drift",
        )
    require(
        str(model_manifest.get("training_source_content_sha256", ""))
        == str(
            dict(artifact.get("training_source", {}) or {}).get(
                "source_content_sha256", ""
            )
        ),
        "bundled model training-source hash drift",
    )
    return model_path, artifact


def _arm_metadata(arm_name: str) -> Dict[str, Any]:
    if arm_name in BASELINE_ARM_FLAGS:
        flags = BASELINE_ARM_FLAGS[arm_name]
        return {
            "query_gate_enabled": bool(flags["query_gate_enabled"]),
            "release_guard_enabled": bool(flags["release_guard_enabled"]),
            "query_policy": "deployable_discrepancy_predictor",
            "source_bundle": "discrepancy_baseline",
        }
    arm = ARM_BY_NAME[arm_name]
    return {
        "query_gate_enabled": bool(arm.query_gate_enabled),
        "release_guard_enabled": bool(arm.release_guard_enabled),
        "query_policy": "rgd_query_gate",
        "source_bundle": "core_factorial",
    }


def analyze_discrepancy_comparison(
    core_matrix: Mapping[Tuple[int, str], Mapping[str, Any]],
    discrepancy_matrix: Mapping[Tuple[int, str], Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Summarize comparison arms and paired within-seed contrasts."""

    seed_blocks = tuple(_integer(seed, "analysis seed", nonnegative=True) for seed in seeds)
    require(bool(seed_blocks), "analysis requires at least one seed")
    require(len(set(seed_blocks)) == len(seed_blocks), "duplicate analysis seeds")
    required_core = {
        (seed, arm) for seed in seed_blocks for arm in ("query_only", "full")
    }
    required_baseline = {
        (seed, arm) for seed in seed_blocks for arm in BASELINE_ARM_NAMES
    }
    require(
        required_core.issubset(set(core_matrix)),
        "core matrix lacks a required paired comparison cell",
    )
    require(
        set(discrepancy_matrix) == required_baseline,
        "discrepancy matrix is not a complete paired seed block",
    )
    confidence = _finite_number(confidence_level, "confidence level")
    require(0.0 < confidence < 1.0, "confidence level must lie in (0, 1)")
    bootstrap_indices = seed_bootstrap_indices(
        len(seed_blocks), draws=draws, bootstrap_seed=bootstrap_seed
    )

    def row_for(seed: int, arm_name: str) -> Mapping[str, Any]:
        if arm_name in BASELINE_ARM_FLAGS:
            return discrepancy_matrix[(seed, arm_name)]
        return core_matrix[(seed, arm_name)]

    arm_summaries: List[Dict[str, Any]] = []
    for arm_name in COMPARISON_ARM_NAMES:
        metadata = _arm_metadata(arm_name)
        for metric in METRICS:
            values = [float(row_for(seed, arm_name)[metric]) for seed in seed_blocks]
            point, low, high = percentile_bootstrap_mean(
                values,
                bootstrap_indices,
                confidence_level=confidence,
            )
            arm_summaries.append(
                {
                    "arm": arm_name,
                    **metadata,
                    "metric": metric,
                    "metric_family": _metric_family(metric),
                    "n_seeds": len(seed_blocks),
                    "mean": point,
                    "ci_low": low,
                    "ci_high": high,
                    "total": float(sum(values)),
                    "confidence_level": confidence,
                    "bootstrap_draws": int(draws),
                }
            )

    paired_contrasts: List[Dict[str, Any]] = []
    seed_contrasts: List[Dict[str, Any]] = []
    for metric in METRICS:
        for contrast_name, (left_arm, right_arm) in PAIRED_CONTRASTS.items():
            differences = [
                float(row_for(seed, left_arm)[metric])
                - float(row_for(seed, right_arm)[metric])
                for seed in seed_blocks
            ]
            for seed, value in zip(seed_blocks, differences):
                seed_contrasts.append(
                    {
                        "seed": int(seed),
                        "contrast": contrast_name,
                        "metric": metric,
                        "metric_family": _metric_family(metric),
                        "difference": float(value),
                    }
                )
            point, low, high = percentile_bootstrap_mean(
                differences,
                bootstrap_indices,
                confidence_level=confidence,
            )
            paired_contrasts.append(
                {
                    "effect": contrast_name,
                    "contrast": f"{left_arm}-{right_arm}",
                    "metric": metric,
                    "metric_family": _metric_family(metric),
                    "n_seed_blocks": len(seed_blocks),
                    "estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "confidence_level": confidence,
                    "bootstrap_draws": int(draws),
                    "bootstrap_unit": "simulator_seed",
                }
            )

    return {
        "seeds": list(seed_blocks),
        "seed_count": len(seed_blocks),
        "metrics": list(METRICS),
        "outcome_metrics": list(OUTCOME_METRICS),
        "lifecycle_metrics": list(LIFECYCLE_METRICS),
        "auxiliary_validated_metrics": list(AUXILIARY_VALIDATED_METRICS),
        "comparison_arms": list(COMPARISON_ARM_NAMES),
        "arm_summaries": arm_summaries,
        "paired_contrasts": paired_contrasts,
        "contrast_definitions": {
            name: {"left": arms[0], "right": arms[1]}
            for name, arms in PAIRED_CONTRASTS.items()
        },
        "seed_contrasts": seed_contrasts,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-bundle", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    require(args.draws > 0, "--draws must be positive")
    require(0.0 < args.confidence_level < 1.0, "--confidence-level must lie in (0, 1)")

    factorial_result_path = args.factorial_bundle / "factorial_episode_results.csv"
    factorial_run_path = args.factorial_bundle / "factorial_run_manifest.json"
    factorial_proposal_path = args.factorial_bundle / "proposal_bank_manifest.json"
    factorial_rows = _read_csv(factorial_result_path)
    factorial_run = _read_json(factorial_run_path)
    factorial_proposal = _read_json(factorial_proposal_path)
    core_contract = validate_bundle_contract(
        factorial_rows, factorial_run, factorial_proposal
    )

    baseline_result_path = args.baseline_bundle / "discrepancy_episode_results.csv"
    baseline_run_path = args.baseline_bundle / "discrepancy_run_manifest.json"
    baseline_proposal_path = args.baseline_bundle / "proposal_bank_manifest.json"
    baseline_rows = _read_csv(baseline_result_path)
    baseline_run = _read_json(baseline_run_path)
    baseline_proposal = _read_json(baseline_proposal_path)
    baseline_model_path, _ = validate_bundled_model_artifact(
        args.baseline_bundle,
        baseline_run,
    )
    baseline_contract = validate_discrepancy_bundle_contract(
        baseline_rows,
        baseline_run,
        baseline_proposal,
        core_contract=core_contract,
    )

    analysis = analyze_discrepancy_comparison(
        core_contract["matrix"],
        baseline_contract["matrix"],
        seeds=core_contract["seeds"],
        draws=args.draws,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )

    arm_path = args.output_dir / "discrepancy_query_arm_summary.csv"
    contrasts_path = args.output_dir / "discrepancy_query_paired_contrasts.csv"
    analysis_path = args.output_dir / "discrepancy_query_baseline_analysis.json"
    _atomic_write_csv(arm_path, analysis["arm_summaries"])
    _atomic_write_csv(contrasts_path, analysis["paired_contrasts"])
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "accepted": True,
        "source_bundles": {
            "core_factorial": str(args.factorial_bundle.resolve()),
            "discrepancy_baseline": str(args.baseline_bundle.resolve()),
        },
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "discrepancy_baseline_version": DISCREPANCY_BASELINE_VERSION,
        "model_artifact_sha256": baseline_contract["model_artifact_sha256"],
        "proposal_bank_sha256": core_contract["proposal_bank_sha256"],
        "latency_profile": core_contract["latency_profile"],
        "design": {
            "independent_unit": "simulator_seed",
            "paired_by_seed": True,
            "candidate_source_policy": core_contract["candidate_source_policy"],
            "candidate_source_gate_independent": True,
            "comparison_conditions_share_proposal_bank_and_latency_profile": True,
            "baseline_arms": [
                {"name": name, **dict(BASELINE_ARM_FLAGS[name])}
                for name in BASELINE_ARM_NAMES
            ],
            "contrast_definitions": analysis["contrast_definitions"],
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
            "core_factorial": {
                factorial_result_path.name: _sha256_file(factorial_result_path),
                factorial_run_path.name: _sha256_file(factorial_run_path),
                factorial_proposal_path.name: _sha256_file(factorial_proposal_path),
            },
            "discrepancy_baseline": {
                baseline_result_path.name: _sha256_file(baseline_result_path),
                baseline_run_path.name: _sha256_file(baseline_run_path),
                baseline_proposal_path.name: _sha256_file(baseline_proposal_path),
                baseline_model_path.name: _sha256_file(baseline_model_path),
            },
        },
        "output_sha256": {
            arm_path.name: _sha256_file(arm_path),
            contrasts_path.name: _sha256_file(contrasts_path),
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
