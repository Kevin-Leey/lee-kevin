"""Train and calibrate a proposal-blind behavior-discrepancy query gate.

The discrepancy label is read only for fit seeds. Calibration seeds expose only
query-time RGD admission decisions and are used solely to match invocation rate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.discrepancy_query_gate import (  # noqa: E402
    DISCREPANCY_ARTIFACT_SCHEMA,
    DISCREPANCY_GATE_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    canonical_json_sha256,
    extract_query_features,
    feature_schema_payload,
    validate_discrepancy_artifact,
)
from tools.run_query_release_factorial import (  # noqa: E402
    DEFAULT_PROPOSAL_SOURCE_POLICY,
    _query_event,
    _sha256_file,
    _source_paths,
    _validate_proposal_source,
)


DEFAULT_TRAINING_SOURCE = Path(
    "results/tvt_final_20260721/main_identifiable_v12_diagnostic/formal_run/"
    "main_v12_20260721/always_slow/highway"
)
DEFAULT_FIT_SEEDS = tuple(range(4000, 4020))
DEFAULT_CALIBRATION_SEEDS = tuple(range(4020, 4030))
MODEL_RANDOM_SEED = 20260730
CALIBRATION_RULE = "exact_rgd_exposure_count_midpoint_threshold_v1"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _class_counts(labels: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {"agreement": int(counts[0]), "discrepancy": int(counts[1])}


def _load_seed_records(
    source_root: Path,
    seed: int,
    *,
    include_discrepancy_label: bool,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    event_path, reasoning_path, snapshot_path = _source_paths(source_root, seed)
    _validate_proposal_source(
        snapshot_path,
        seed=seed,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    payload = json.loads(event_path.read_text(encoding="utf-8-sig"))
    events = list(payload.get("events", ()) or ())
    records: list[Dict[str, Any]] = []
    for event in events:
        if not _query_event(event):
            continue
        if bool(event.get("closed_loop_latency_release_event", False)):
            raise RuntimeError(f"seed {seed}: release row selected as a query example")
        frame = int(event.get("frame", -1))
        fast_action = int(event.get("query_state_fast_proposal_action", -1))
        features = extract_query_features(event, fast_action=fast_action)
        row: Dict[str, Any] = {
            "seed": int(seed),
            "frame": frame,
            "features": features,
            "rgd_query_exposed": bool(event.get("recoverability_serial_gate_pass", False)),
        }
        if include_discrepancy_label:
            # The slow action is accessed only inside this fit-only branch.
            if "query_state_slow_released_action" not in event:
                raise RuntimeError(f"seed {seed} frame {frame}: discrepancy label missing")
            slow_action = int(event["query_state_slow_released_action"])
            if slow_action not in range(5):
                raise RuntimeError(f"seed {seed} frame {frame}: invalid slow label action")
            row["label"] = int(slow_action != fast_action)
        records.append(row)
    if not records:
        raise RuntimeError(f"seed {seed}: no always-slow query examples")
    provenance = {
        "seed": int(seed),
        "event_log": {
            "path": str(event_path.relative_to(source_root)).replace("\\", "/"),
            "sha256": _sha256_file(event_path),
        },
        "reasoning_trace": {
            "path": str(reasoning_path.relative_to(source_root)).replace("\\", "/"),
            "sha256": _sha256_file(reasoning_path),
        },
        "experiment_snapshot": {
            "path": str(snapshot_path.relative_to(source_root)).replace("\\", "/"),
            "sha256": _sha256_file(snapshot_path),
        },
        "record_count": len(records),
        "discrepancy_label_read": bool(include_discrepancy_label),
    }
    return records, provenance


def _calibrated_threshold(scores: np.ndarray, target_count: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not bool(np.all(np.isfinite(values))):
        raise ValueError("calibration scores must be finite and nonempty")
    target = int(target_count)
    if target < 0 or target > values.size:
        raise ValueError("calibration target count is out of range")
    descending = np.sort(values)[::-1]
    if target == 0:
        threshold = float(np.nextafter(descending[0], math.inf))
    elif target == values.size:
        threshold = float(np.nextafter(descending[-1], -math.inf))
    else:
        upper = float(descending[target - 1])
        lower = float(descending[target])
        if not upper > lower:
            raise RuntimeError(
                "calibration scores tie at the RGD exposure boundary; "
                "a scalar threshold cannot match exposure exactly"
            )
        threshold = upper + (lower - upper) / 2.0
    achieved = int(np.sum(values >= threshold))
    if achieved != target:
        raise RuntimeError(
            f"calibration threshold mismatch: expected {target}, achieved {achieved}"
        )
    return float(threshold)


def _classifier_pipeline(classifier_params: Mapping[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(**dict(classifier_params)),
            ),
        ]
    )


def _seed_block_cross_validation(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    classifier_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return out-of-fold diagnostics without selecting model settings."""

    unique_seeds = tuple(sorted({int(seed) for seed in groups.tolist()}))
    if len(unique_seeds) < 2:
        raise RuntimeError("seed-block validation requires at least two seeds")
    split_count = min(5, len(unique_seeds))
    splitter = GroupKFold(n_splits=split_count)
    scores = np.full(labels.shape, np.nan, dtype=np.float64)
    folds = []
    for fold_index, (fit_index, validation_index) in enumerate(
        splitter.split(features, labels, groups=groups)
    ):
        fit_labels = labels[fit_index]
        if set(fit_labels.tolist()) != {0, 1}:
            raise RuntimeError(
                f"seed-block fold {fold_index} training split lacks one class"
            )
        pipeline = _classifier_pipeline(classifier_params)
        pipeline.fit(features[fit_index], fit_labels)
        fold_scores = pipeline.predict_proba(features[validation_index])[:, 1]
        scores[validation_index] = fold_scores
        validation_labels = labels[validation_index]
        validation_seeds = sorted(
            {int(seed) for seed in groups[validation_index].tolist()}
        )
        fold_auc = (
            float(roc_auc_score(validation_labels, fold_scores))
            if len(set(validation_labels.tolist())) == 2
            else None
        )
        folds.append(
            {
                "fold": int(fold_index),
                "validation_seeds": validation_seeds,
                "record_count": int(validation_index.size),
                "class_counts": _class_counts(validation_labels.tolist()),
                "roc_auc": fold_auc,
            }
        )
    if not bool(np.all(np.isfinite(scores))):
        raise RuntimeError("seed-block validation did not score every fit record")
    predictions = (scores >= 0.5).astype(np.int64)
    return {
        "role": "fit_cohort_diagnostic_only",
        "used_for_model_selection": False,
        "splitter": "GroupKFold",
        "group_unit": "simulator_seed",
        "n_splits": int(split_count),
        "record_count": int(labels.size),
        "seed_count": len(unique_seeds),
        "classification_threshold": 0.5,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "brier_score": float(brier_score_loss(labels, scores)),
        "folds": folds,
    }


def train_discrepancy_artifact(
    source_root: Path,
    *,
    fit_seeds: Sequence[int] = DEFAULT_FIT_SEEDS,
    calibration_seeds: Sequence[int] = DEFAULT_CALIBRATION_SEEDS,
) -> Dict[str, Any]:
    fit = tuple(int(seed) for seed in fit_seeds)
    calibration = tuple(int(seed) for seed in calibration_seeds)
    if not fit or not calibration:
        raise ValueError("fit and calibration seed cohorts must be nonempty")
    if len(set(fit)) != len(fit) or len(set(calibration)) != len(calibration):
        raise ValueError("seed cohorts contain duplicates")
    if set(fit) & set(calibration):
        raise ValueError("fit and calibration seed blocks must be disjoint")

    fit_records: list[Dict[str, Any]] = []
    calibration_records: list[Dict[str, Any]] = []
    provenance = []
    for seed in fit:
        rows, source = _load_seed_records(
            source_root,
            seed,
            include_discrepancy_label=True,
        )
        fit_records.extend(rows)
        provenance.append(source)
    for seed in calibration:
        rows, source = _load_seed_records(
            source_root,
            seed,
            include_discrepancy_label=False,
        )
        calibration_records.extend(rows)
        provenance.append(source)

    x_fit = np.stack([row["features"] for row in fit_records])
    y_fit = np.asarray([row["label"] for row in fit_records], dtype=np.int64)
    if set(y_fit.tolist()) != {0, 1}:
        raise RuntimeError("fit cohort must contain both discrepancy classes")
    x_calibration = np.stack([row["features"] for row in calibration_records])
    target_count = sum(bool(row["rgd_query_exposed"]) for row in calibration_records)

    classifier_params = {
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": MODEL_RANDOM_SEED,
        "solver": "liblinear",
    }
    fit_groups = np.asarray(
        [int(row["seed"]) for row in fit_records], dtype=np.int64
    )
    cross_validation = _seed_block_cross_validation(
        x_fit,
        y_fit,
        fit_groups,
        classifier_params=classifier_params,
    )
    pipeline = _classifier_pipeline(classifier_params)
    pipeline.fit(x_fit, y_fit)
    calibration_scores = pipeline.predict_proba(x_calibration)[:, 1]
    threshold = _calibrated_threshold(calibration_scores, target_count)
    achieved_count = int(np.sum(calibration_scores >= threshold))

    scaler = pipeline.named_steps["standard_scaler"]
    classifier = pipeline.named_steps["logistic_regression"]
    source_records = sorted(provenance, key=lambda row: int(row["seed"]))
    source_content_sha256 = canonical_json_sha256(source_records)
    payload: Dict[str, Any] = {
        "schema": DISCREPANCY_ARTIFACT_SCHEMA,
        "version": DISCREPANCY_GATE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema": feature_schema_payload(),
        "routing_input_contract": {
            "query_time_only": True,
            "proposal_record_visible": False,
            "slow_response_visible": False,
            "slow_action_visible": False,
            "release_outcome_visible": False,
        },
        "training_source": {
            "policy": DEFAULT_PROPOSAL_SOURCE_POLICY,
            "root": str(source_root).replace("\\", "/"),
            "source_content_sha256": source_content_sha256,
            "artifacts": source_records,
        },
        "seed_split": {
            "unit": "simulator_seed",
            "fit_seeds": list(fit),
            "calibration_seeds": list(calibration),
            "disjoint": True,
            "fit_labels_use_slow_fast_query_discrepancy": True,
            "calibration_labels_use_slow_fast_query_discrepancy": False,
        },
        "fit": {
            "record_count": int(len(fit_records)),
            "seed_count": len(fit),
            "class_counts": _class_counts(y_fit.tolist()),
            "label": "query_state_slow_released_action != query_state_fast_proposal_action",
            "seed_block_cross_validation": cross_validation,
        },
        "model": {
            "pipeline": "StandardScaler->LogisticRegression",
            "sklearn_version": str(sklearn.__version__),
            "standard_scaler": {
                "parameters": {"copy": True, "with_mean": True, "with_std": True},
                "mean": [float(value) for value in scaler.mean_],
                "scale": [float(value) for value in scaler.scale_],
                "variance": [float(value) for value in scaler.var_],
                "samples_seen": int(scaler.n_samples_seen_),
            },
            "logistic_regression": {
                "parameters": classifier_params,
                "classes": [int(value) for value in classifier.classes_],
                "coefficients": [float(value) for value in classifier.coef_[0]],
                "intercept": float(classifier.intercept_[0]),
                "iterations": int(classifier.n_iter_[0]),
            },
        },
        "calibration": {
            "rule": CALIBRATION_RULE,
            "record_count": int(len(calibration_records)),
            "seed_count": len(calibration),
            "target_policy": "rgd_serial_query_gate",
            "target_invocations": int(target_count),
            "target_exposure_rate": float(target_count / len(calibration_records)),
            "threshold": float(threshold),
            "comparison_operator": ">=",
            "achieved_invocations": int(achieved_count),
            "achieved_exposure_rate": float(achieved_count / len(calibration_records)),
            "uses_outcomes": False,
            "uses_discrepancy_labels": False,
        },
    }
    payload["artifact_sha256"] = canonical_json_sha256(payload)
    return validate_discrepancy_artifact(payload)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_TRAINING_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    artifact = train_discrepancy_artifact(args.source_root)
    _write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact["artifact_sha256"],
                "fit_records": artifact["fit"]["record_count"],
                "calibration_records": artifact["calibration"]["record_count"],
                "target_invocations": artifact["calibration"]["target_invocations"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

