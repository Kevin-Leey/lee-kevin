"""Deployable, proposal-blind behavior-discrepancy query gate."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from dilu.evaluation.factorial_replay import (
    QueryAdmissionContext,
    QueryAdmissionDecision,
)


DISCREPANCY_GATE_VERSION = "gvlm_style_discrepancy_logistic_v1"
DISCREPANCY_ARTIFACT_SCHEMA = "rgd_discrepancy_query_gate_artifact_v1"
FEATURE_SCHEMA_VERSION = "rgd_query_observables_v1"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    source_paths: Tuple[Tuple[str, ...], ...]
    description: str


# Every source is available before request admission. Alternate paths bridge the
# live nested metadata and the deliberately flattened persisted event schema.
BASE_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec(
        "execution_route_score",
        "float",
        (("rgd_execution_route_score",), ("recoverability_gate", "need_state_hazard")),
        "Query-time state hazard used by the router.",
    ),
    FeatureSpec(
        "recovery_window",
        "float",
        (("recoverability_recovery_window",), ("recoverability_gate", "recovery_window")),
        "Predicted temporal survival during deliberation.",
    ),
    FeatureSpec(
        "post_latency_opportunity",
        "float",
        (("recoverability_post_latency_opportunity",), ("recoverability_gate", "post_latency_opportunity")),
        "Query-time estimate of opportunity remaining after latency.",
    ),
    FeatureSpec(
        "maneuver_family_breadth",
        "float",
        (
            ("recoverability_relative_support_weighted_maneuver_family_breadth",),
            ("recoverability_gate", "relative_support_weighted_maneuver_family_breadth"),
        ),
        "Support-weighted breadth of feasible maneuver families.",
    ),
    FeatureSpec(
        "relative_corrective_headroom",
        "float",
        (("recoverability_relative_corrective_headroom",), ("recoverability_gate", "relative_corrective_headroom")),
        "Cost headroom for corrective action at query time.",
    ),
    FeatureSpec(
        "action_cost_entropy",
        "float",
        (("recoverability_action_cost_entropy",), ("recoverability_gate", "action_cost_entropy")),
        "Entropy of the query-time action-cost distribution.",
    ),
    FeatureSpec(
        "absolute_recovery_depth",
        "float",
        (("recoverability_absolute_recovery_depth",), ("recoverability_gate", "absolute_recovery_depth")),
        "Depth of the feasible corrective set.",
    ),
    FeatureSpec(
        "reasoning_latency_pressure",
        "float",
        (("reasoning_latency_pressure",), ("recoverability_reasoning_latency_pressure",)),
        "Predicted latency pressure before a request is issued.",
    ),
    FeatureSpec(
        "fast_confidence",
        "float",
        (("confidence",), ("decision_confidence",)),
        "Confidence reported by the current Fast controller.",
    ),
    FeatureSpec(
        "pre_screen_trigger",
        "bool",
        (("recoverability_pre_screen_trigger",), ("pre_screen_trigger",)),
        "Query-time hazard pre-screen indicator.",
    ),
    FeatureSpec(
        "gate_domain_valid",
        "bool",
        (("recoverability_gate_domain_valid",), ("recoverability_gate", "gate_domain_valid")),
        "Whether the query-time feature contract is valid.",
    ),
    FeatureSpec(
        "absolute_alternative_feasible",
        "bool",
        (
            ("recoverability_absolute_alternative_feasible",),
            ("recoverability_gate", "absolute_alternative_feasible"),
        ),
        "Whether a non-incumbent action is absolutely feasible.",
    ),
)

FAST_ACTION_FEATURES: Tuple[str, ...] = tuple(
    f"fast_action_is_{action}" for action in range(5)
)
FEATURE_NAMES: Tuple[str, ...] = tuple(spec.name for spec in BASE_FEATURES) + FAST_ACTION_FEATURES

FORBIDDEN_ROUTING_FIELD_MARKERS: Tuple[str, ...] = (
    "slow_action",
    "slow_response",
    "response_sha",
    "response_outcome",
    "release_event",
    "released_action",
    "terminal_outcome",
    "collision",
    "episode_reward",
    "success_rate",
)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_schema_payload() -> list[Dict[str, Any]]:
    rows = [
        {
            "name": spec.name,
            "kind": spec.kind,
            "source_paths": [list(path) for path in spec.source_paths],
            "description": spec.description,
        }
        for spec in BASE_FEATURES
    ]
    rows.extend(
        {
            "name": name,
            "kind": "bool",
            "source_paths": [["fast_action"]],
            "description": "One-hot query-time Fast action.",
        }
        for name in FAST_ACTION_FEATURES
    )
    return rows


def validate_feature_schema_is_proposal_blind() -> None:
    for row in feature_schema_payload():
        serialized = json.dumps(row, sort_keys=True).lower()
        for marker in FORBIDDEN_ROUTING_FIELD_MARKERS:
            if marker in serialized:
                raise RuntimeError(
                    f"discrepancy feature schema contains forbidden marker {marker!r}"
                )


def _path_value(metadata: Mapping[str, Any], paths: Sequence[Sequence[str]], name: str) -> Any:
    for path in paths:
        value: Any = metadata
        found = True
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                found = False
                break
            value = value[key]
        if found and value is not None:
            return value
    raise ValueError(f"missing deployable discrepancy feature: {name}")


def _finite_feature(value: Any, *, kind: str, name: str) -> float:
    if kind == "bool":
        if isinstance(value, (bool, np.bool_)):
            return float(bool(value))
        if value in (0, 1, 0.0, 1.0):
            return float(value)
        raise ValueError(f"invalid boolean discrepancy feature {name}: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"boolean supplied for numeric discrepancy feature {name}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid discrepancy feature {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite discrepancy feature {name}: {value!r}")
    return result


def extract_query_features(
    metadata: Mapping[str, Any],
    *,
    fast_action: int,
) -> np.ndarray:
    """Extract only pre-request observables; proposal fields are never read."""

    validate_feature_schema_is_proposal_blind()
    if not isinstance(metadata, Mapping):
        raise TypeError("query metadata must be a mapping")
    action = int(fast_action)
    if action not in range(5):
        raise ValueError(f"Fast action is outside the discrete universe: {action}")
    values = [
        _finite_feature(
            _path_value(metadata, spec.source_paths, spec.name),
            kind=spec.kind,
            name=spec.name,
        )
        for spec in BASE_FEATURES
    ]
    values.extend(float(action == candidate) for candidate in range(5))
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),) or not bool(np.all(np.isfinite(vector))):
        raise ValueError("invalid discrepancy feature vector")
    return vector


def _artifact_without_hash(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    return payload


def validate_discrepancy_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise TypeError("discrepancy artifact must be an object")
    payload = dict(artifact)
    if payload.get("schema") != DISCREPANCY_ARTIFACT_SCHEMA:
        raise ValueError("unexpected discrepancy artifact schema")
    if payload.get("version") != DISCREPANCY_GATE_VERSION:
        raise ValueError("unexpected discrepancy gate version")
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unexpected discrepancy feature schema version")
    if payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("discrepancy feature order drift")
    if payload.get("feature_schema") != feature_schema_payload():
        raise ValueError("discrepancy feature schema drift")
    expected_hash = canonical_json_sha256(_artifact_without_hash(payload))
    if str(payload.get("artifact_sha256", "")) != expected_hash:
        raise ValueError("discrepancy artifact hash mismatch")

    split = dict(payload.get("seed_split", {}) or {})
    fit_seeds = tuple(int(seed) for seed in split.get("fit_seeds", ()))
    calibration_seeds = tuple(int(seed) for seed in split.get("calibration_seeds", ()))
    if not fit_seeds or not calibration_seeds:
        raise ValueError("discrepancy seed split must be nonempty")
    if len(set(fit_seeds)) != len(fit_seeds) or len(set(calibration_seeds)) != len(calibration_seeds):
        raise ValueError("duplicate seed in discrepancy split")
    if set(fit_seeds) & set(calibration_seeds):
        raise ValueError("fit and calibration seed blocks overlap")
    if split.get("unit") != "simulator_seed" or split.get("disjoint") is not True:
        raise ValueError("invalid discrepancy seed-block split contract")
    if split.get("fit_labels_use_slow_fast_query_discrepancy") is not True:
        raise ValueError("fit-label provenance is missing")
    if split.get("calibration_labels_use_slow_fast_query_discrepancy") is not False:
        raise ValueError("calibration must not use slow/Fast discrepancy labels")

    routing = dict(payload.get("routing_input_contract", {}) or {})
    expected_routing = {
        "query_time_only": True,
        "proposal_record_visible": False,
        "slow_response_visible": False,
        "slow_action_visible": False,
        "release_outcome_visible": False,
    }
    if routing != expected_routing:
        raise ValueError("discrepancy routing-input contract drift")

    source = dict(payload.get("training_source", {}) or {})
    if source.get("policy") != "scheduled_always_slow":
        raise ValueError("discrepancy training source is not gate-independent")
    source_rows = source.get("artifacts")
    if not isinstance(source_rows, list):
        raise ValueError("discrepancy source provenance must be a list")
    source_by_seed: Dict[int, Mapping[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid discrepancy source-provenance row")
        seed = int(row.get("seed", -1))
        if seed in source_by_seed:
            raise ValueError(f"duplicate discrepancy source seed: {seed}")
        expected_label_access = seed in set(fit_seeds)
        if row.get("discrepancy_label_read") is not expected_label_access:
            raise ValueError(f"seed {seed}: discrepancy-label access contract mismatch")
        if int(row.get("record_count", 0)) <= 0:
            raise ValueError(f"seed {seed}: empty discrepancy source block")
        for artifact_name in ("event_log", "reasoning_trace", "experiment_snapshot"):
            artifact_row = row.get(artifact_name)
            if not isinstance(artifact_row, Mapping):
                raise ValueError(f"seed {seed}: missing {artifact_name} provenance")
            digest = str(artifact_row.get("sha256", ""))
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"seed {seed}: invalid {artifact_name} hash")
        source_by_seed[seed] = row
    if set(source_by_seed) != set(fit_seeds) | set(calibration_seeds):
        raise ValueError("source-provenance seeds differ from the frozen split")
    if source.get("source_content_sha256") != canonical_json_sha256(source_rows):
        raise ValueError("discrepancy training-source content hash mismatch")

    fit = dict(payload.get("fit", {}) or {})
    fit_count = int(fit.get("record_count", 0))
    class_counts = dict(fit.get("class_counts", {}) or {})
    agreement = int(class_counts.get("agreement", 0))
    discrepancy = int(class_counts.get("discrepancy", 0))
    if min(agreement, discrepancy) <= 0 or agreement + discrepancy != fit_count:
        raise ValueError("invalid discrepancy fit class counts")
    if fit_count != sum(
        int(source_by_seed[seed]["record_count"]) for seed in fit_seeds
    ):
        raise ValueError("fit record count differs from source seed blocks")
    cross_validation = dict(fit.get("seed_block_cross_validation", {}) or {})
    if (
        cross_validation.get("role") != "fit_cohort_diagnostic_only"
        or cross_validation.get("used_for_model_selection") is not False
        or cross_validation.get("splitter") != "GroupKFold"
        or cross_validation.get("group_unit") != "simulator_seed"
    ):
        raise ValueError("invalid discrepancy seed-block validation contract")
    if (
        int(cross_validation.get("record_count", -1)) != fit_count
        or int(cross_validation.get("seed_count", -1)) != len(fit_seeds)
    ):
        raise ValueError("discrepancy validation cohort size drift")
    split_count = int(cross_validation.get("n_splits", 0))
    if split_count != min(5, len(fit_seeds)):
        raise ValueError("discrepancy validation fold count drift")
    for metric_name in (
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "brier_score",
    ):
        metric = float(cross_validation.get(metric_name, math.nan))
        if not math.isfinite(metric) or not 0.0 <= metric <= 1.0:
            raise ValueError(
                f"invalid discrepancy validation metric: {metric_name}"
            )
    if float(cross_validation.get("classification_threshold", math.nan)) != 0.5:
        raise ValueError("unexpected discrepancy validation threshold")
    folds = list(cross_validation.get("folds", ()) or ())
    if len(folds) != split_count:
        raise ValueError("discrepancy validation fold table is incomplete")
    validation_seed_union = set()
    validation_record_count = 0
    for fold_index, fold_row in enumerate(folds):
        if not isinstance(fold_row, Mapping):
            raise ValueError("invalid discrepancy validation fold row")
        if int(fold_row.get("fold", -1)) != fold_index:
            raise ValueError("discrepancy validation fold ordering drift")
        fold_seeds = {int(seed) for seed in fold_row.get("validation_seeds", ())}
        if not fold_seeds or validation_seed_union & fold_seeds:
            raise ValueError("discrepancy validation seed blocks overlap")
        validation_seed_union.update(fold_seeds)
        validation_record_count += int(fold_row.get("record_count", 0))
    if validation_seed_union != set(fit_seeds) or validation_record_count != fit_count:
        raise ValueError("discrepancy validation coverage drift")

    model = dict(payload.get("model", {}) or {})
    scaler = dict(model.get("standard_scaler", {}) or {})
    classifier = dict(model.get("logistic_regression", {}) or {})
    size = len(FEATURE_NAMES)
    for key in ("mean", "scale", "variance"):
        values = np.asarray(scaler.get(key, ()), dtype=np.float64)
        if values.shape != (size,) or not bool(np.all(np.isfinite(values))):
            raise ValueError(f"invalid scaler {key}")
        if key == "scale" and not bool(np.all(values > 0.0)):
            raise ValueError("scaler scale must be positive")
    coefficients = np.asarray(classifier.get("coefficients", ()), dtype=np.float64)
    if coefficients.shape != (size,) or not bool(np.all(np.isfinite(coefficients))):
        raise ValueError("invalid discrepancy coefficients")
    intercept = float(classifier.get("intercept", math.nan))
    if not math.isfinite(intercept):
        raise ValueError("invalid discrepancy intercept")
    if classifier.get("classes") != [0, 1]:
        raise ValueError("discrepancy classifier class order drift")
    calibration = dict(payload.get("calibration", {}) or {})
    threshold = float(calibration.get("threshold", math.nan))
    if not math.isfinite(threshold):
        raise ValueError("invalid discrepancy invocation threshold")
    calibration_count = int(calibration.get("record_count", 0))
    if calibration_count != sum(
        int(source_by_seed[seed]["record_count"]) for seed in calibration_seeds
    ):
        raise ValueError("calibration record count differs from source seed blocks")
    target = int(calibration.get("target_invocations", -1))
    achieved = int(calibration.get("achieved_invocations", -2))
    if not 0 <= target <= calibration_count or achieved != target:
        raise ValueError("discrepancy calibration does not match RGD exposure count")
    if calibration.get("uses_outcomes") is not False or calibration.get("uses_discrepancy_labels") is not False:
        raise ValueError("calibration leakage contract drift")
    if calibration.get("comparison_operator") != ">=":
        raise ValueError("unexpected discrepancy threshold comparison")
    validate_feature_schema_is_proposal_blind()
    return payload


def load_discrepancy_artifact(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"discrepancy artifact missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return validate_discrepancy_artifact(payload)


class DiscrepancyQueryGate:
    """JSON-backed logistic gate whose input surface excludes proposals."""

    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = validate_discrepancy_artifact(artifact)
        model = dict(self.artifact["model"])
        scaler = dict(model["standard_scaler"])
        classifier = dict(model["logistic_regression"])
        self.mean = np.asarray(scaler["mean"], dtype=np.float64)
        self.scale = np.asarray(scaler["scale"], dtype=np.float64)
        self.coefficients = np.asarray(classifier["coefficients"], dtype=np.float64)
        self.intercept = float(classifier["intercept"])
        self.threshold = float(self.artifact["calibration"]["threshold"])
        self.artifact_sha256 = str(self.artifact["artifact_sha256"])

    def probability(self, metadata: Mapping[str, Any], *, fast_action: int) -> float:
        vector = extract_query_features(metadata, fast_action=fast_action)
        logit = float(np.dot(self.coefficients, (vector - self.mean) / self.scale) + self.intercept)
        if logit >= 0.0:
            probability = 1.0 / (1.0 + math.exp(-logit))
        else:
            exp_logit = math.exp(logit)
            probability = exp_logit / (1.0 + exp_logit)
        return float(probability)

    def decide(self, context: QueryAdmissionContext) -> QueryAdmissionDecision:
        if not isinstance(context, QueryAdmissionContext):
            raise TypeError("discrepancy gate requires QueryAdmissionContext")
        probability = self.probability(
            context.query_metadata,
            fast_action=int(context.fast_action),
        )
        admit = bool(probability >= self.threshold)
        return QueryAdmissionDecision(
            admit=admit,
            audit={
                "discrepancy_gate_version": DISCREPANCY_GATE_VERSION,
                "discrepancy_gate_artifact_sha256": self.artifact_sha256,
                "discrepancy_gate_feature_schema_version": FEATURE_SCHEMA_VERSION,
                "discrepancy_gate_probability": probability,
                "discrepancy_gate_threshold": self.threshold,
                "discrepancy_gate_admit": admit,
                "discrepancy_gate_query_inputs_only": True,
            },
        )
