"""Small protocol-backed surface shared by runtime and result tools."""

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


_FORMAL_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "formal_protocol.yaml"


def _load_formal_protocol() -> Dict[str, Any]:
    with _FORMAL_PROTOCOL_PATH.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle) or {}
    if not isinstance(protocol, dict):
        raise RuntimeError(f"formal_protocol.yaml must decode to a mapping: {_FORMAL_PROTOCOL_PATH}")
    return protocol


def _required_str(mapping: Dict[str, Any], key: str, path: str) -> str:
    value = str(mapping.get(key, "") or "").strip()
    if not value:
        raise RuntimeError(f"formal_protocol.yaml is missing {path}.{key}")
    return value


def _required_tuple(mapping: Dict[str, Any], key: str, path: str) -> Tuple[str, ...]:
    values = mapping.get(key, []) or []
    if not isinstance(values, list) or not values:
        raise RuntimeError(f"formal_protocol.yaml is missing non-empty list {path}.{key}")
    normalized = tuple(str(item or "").strip() for item in values)
    if any(not item for item in normalized):
        raise RuntimeError(f"formal_protocol.yaml list {path}.{key} contains an empty item")
    return normalized


_FORMAL_PROTOCOL = _load_formal_protocol()
_CLAIM_GUARDRAILS = dict(_FORMAL_PROTOCOL.get("claim_guardrails", {}) or {})
_PAPER_BASELINES = dict(_FORMAL_PROTOCOL.get("paper_baselines", {}) or {})

DEFAULT_PRIMARY_EVALUATION_SUBJECT = _required_str(_CLAIM_GUARDRAILS, "primary_evaluation_subject", "claim_guardrails")
DEFAULT_SINGLE_CORE_METHOD_NAME = _required_str(_CLAIM_GUARDRAILS, "single_core_method_name", "claim_guardrails")
DEFAULT_RECOVERABILITY_CORE_VARIABLES = _required_tuple(_CLAIM_GUARDRAILS, "recoverability_core_variables", "claim_guardrails")
PAPER_RECOVERABILITY_OBJECT_STATUS = _required_str(_CLAIM_GUARDRAILS, "recoverability_object_status", "claim_guardrails")

HEADLINE_MAIN_TABLE_METRICS = _required_tuple(_PAPER_BASELINES, "core_metrics", "paper_baselines")
COMPARISON_HEADLINE_FIELDS: Tuple[str, ...] = (
    "collision_rate",
    "success_rate",
    "avg_route_completion",
    "avg_episode_reward",
    "avg_driving_distance",
    "avg_speed_safety_qualified",
    "avg_runtime_per_frame",
    "budget_normalized_independent_high_risk_utility",
    "independent_selective_routing_gain",
)

RECOVERABILITY_EVIDENCE_FIELDS: Tuple[str, ...] = (
    "avg_recovery_window",
    "avg_action_space_affordance",
    "avg_commitment_reversibility",
    "avg_recoverable_deliberation_priority",
    "high_recoverability_collapse_frame_rate",
    "high_recoverability_collapse_slow_rate",
    "independent_high_risk_deliberative_slow_rate",
    "independent_high_risk_slow_call_concentration",
    "budget_normalized_independent_high_risk_utility",
    "independent_selective_routing_gain",
    "gate_snapshot_frame_ratio",
    "near_threshold_frame_rate",
    "near_threshold_slow_rate",
    "near_threshold_route_action_changed_rate",
)
