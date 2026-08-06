"""Frame-level reasoning trace recorder for the RGD runtime."""

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dilu.driver_agent.base.state import ACTIONS_ALL
from dilu.driver_agent.reasoning.rgd_support import (
    build_claim_downgrade_signal_fields,
    build_recoverability_minimal_explanation,
    build_recoverability_public_signal,
    build_route_authority_audit_fields,
)
from dilu.utils.shared import safe_float


def _json_safe(value: Any) -> Any:
    """Convert recorder diagnostics into strict JSON values."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


@dataclass
class ReasoningRecord:
    frame_id: int
    timestamp: float
    scenario_description: str
    available_actions: str
    driving_intention: str
    full_response: str
    predicted_action_id: int
    predicted_action_name: str
    inference_latency: float
    system_used: str
    route_reason: str
    rgd_execution_route_score: float
    fast_rule_name: str
    fast_smoothness_override: bool
    slow_reasoning_mode: str
    slow_reasoning_success: bool
    slow_reasoning_failure_reason: str
    rgd_subordinate_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "rgd_record_v3"
        return deepcopy(_json_safe(payload))


@dataclass
class EpisodeReasoningMetrics:
    episode_id: int
    total_frames: int
    total_inference_time: float
    avg_inference_latency: float
    max_inference_latency: float
    min_inference_latency: float
    slow_frame_count: int
    fast_frame_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReasoningRecorder:
    """Record exactly the fields consumed by metrics and trace exports."""

    def __init__(self, episode_id: int, result_folder: str):
        self.episode_id = int(episode_id)
        self.result_folder = result_folder
        self.records: List[ReasoningRecord] = []

    @staticmethod
    def _action_value(value: Any, default: int) -> int:
        return int(default if value is None else value)

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        return safe_float(value, default)

    def _recoverability_gate(self, decision_meta: Dict[str, Any], public_signal: Dict[str, Any]) -> Dict[str, Any]:
        gate = decision_meta.get("recoverability_gate")
        if isinstance(gate, dict):
            return dict(gate)
        route_policy = str(decision_meta.get("recoverability_route_policy", "") or "").strip()
        route_reason = str(decision_meta.get("route_reason", "") or "").strip()
        if route_policy != "no_rgd_signal" and route_reason != "forced_protocol_no_rgd_signal":
            raise ValueError(f"missing recoverability_gate for route_reason={route_reason or '<unknown>'}")
        return {
            "gate_active": False,
            "score": self._float(decision_meta.get("rgd_execution_route_score"), 0.0),
            "threshold": self._float(decision_meta.get("recoverability_route_boundary"), 0.5),
            "margin": self._float(decision_meta.get("recoverability_route_margin"), 0.0),
            "policy": "no_rgd_signal",
            "public_signal": dict(public_signal),
            "collapse_risk": self._float(decision_meta.get("recoverability_collapse_risk"), 0.0),
            "value_of_computation": self._float(decision_meta.get("recoverability_value_of_computation"), 0.0),
            "pre_screen": {"score": 0.0, "trigger": False, "reason": "no_rgd_signal"},
            "latency": {"predicted_slow_seconds": 0.0, "budget_seconds": 0.0, "pressure": 0.0},
            "selected_system": str(decision_meta.get("system_used", "fast") or "fast"),
            "route_reason": route_reason or "forced_protocol_no_rgd_signal",
            "routing_decision": None,
            "baseline_mode": str(decision_meta.get("paper_baseline_trigger_mode", "none") or "none"),
            "execution_route_score": self._float(decision_meta.get("rgd_execution_route_score"), 0.0),
        }

    def _build_grouped_diagnostics(self, action_id: int, decision_meta: Dict[str, Any]) -> Dict[str, Any]:
        proposed_action = self._action_value(decision_meta.get("proposed_action", action_id), action_id)
        final_action = self._action_value(decision_meta.get("final_action", action_id), action_id)
        safety_override = bool(decision_meta.get("safety_override", False))
        shield_override = bool(decision_meta.get("shield_override", False))
        public_signal = build_recoverability_public_signal(decision_meta)
        recoverability_gate = self._recoverability_gate(decision_meta, public_signal)
        return {
            "recoverability_public_signal": dict(public_signal),
            "recoverability_explanation": build_recoverability_minimal_explanation(decision_meta),
            "route_authority_audit": build_route_authority_audit_fields(
                proposed_action,
                final_action,
                safety_override=safety_override,
                shield_override=shield_override,
            ),
            "claim_downgrade_signals": build_claim_downgrade_signal_fields(
                decision_meta,
                proposed_action,
                final_action,
                safety_override=safety_override,
                shield_override=shield_override,
            ),
            "recoverability_signal": {
                "recoverability_gate": recoverability_gate,
                "recoverability_route_policy": str(decision_meta.get("recoverability_route_policy", "recoverability_closed_object") or "recoverability_closed_object"),
                "recoverability_route_boundary": self._float(decision_meta.get("recoverability_route_boundary"), 0.5),
                "recoverability_route_margin": self._float(decision_meta.get("recoverability_route_margin"), 0.0),
                "recoverability_signed_margin_to_threshold": self._float(decision_meta.get("recoverability_signed_margin_to_threshold"), 0.0),
                "recoverability_near_threshold": bool(decision_meta.get("recoverability_near_threshold", False)),
                "recoverability_near_threshold_band": self._float(decision_meta.get("recoverability_near_threshold_band"), 0.0),
                "recoverability_threshold_source": str(decision_meta.get("recoverability_threshold_source", "unknown") or "unknown"),
                "recoverability_threshold_provenance": deepcopy(decision_meta.get("recoverability_threshold_provenance", {})),
                "recoverability_collapse_severity": str(decision_meta.get("recoverability_collapse_severity", "stable") or "stable"),
                "rad_irreversibility": self._float(decision_meta.get("rad_irreversibility"), 0.0),
                "rad_recovery_cost_target": self._float(decision_meta.get("rad_recovery_cost_target"), 0.0),
                "recovery_window": self._float(public_signal.get("recovery_window"), 0.0),
                "action_space_affordance": self._float(public_signal.get("action_space_affordance"), 0.0),
                "commitment_reversibility": self._float(public_signal.get("commitment_reversibility"), 0.0),
                "soft_recoverability": self._float(public_signal.get("soft_recoverability"), 0.0),
                "recoverable_deliberation_priority": self._float(public_signal.get("recoverable_deliberation_priority"), 0.0),
                "recoverability_score": self._float(decision_meta.get("recoverability_score", decision_meta.get("rgd_execution_route_score")), 0.0),
                "recoverability_collapse_risk": self._float(decision_meta.get("recoverability_collapse_risk"), 0.0),
                "recoverability_slow_gain_likelihood": self._float(decision_meta.get("recoverability_slow_gain_likelihood"), 0.0),
                "recoverability_value_of_computation": self._float(decision_meta.get("recoverability_value_of_computation", decision_meta.get("recoverability_score")), 0.0),
            },
            "slow_path_objective": {
                "llm_action_available": bool(decision_meta.get("llm_action_available", False)),
                "llm_raw_action": self._action_value(decision_meta.get("llm_raw_action", action_id), action_id),
                "post_validation_action": self._action_value(decision_meta.get("post_validation_action", action_id), action_id),
                "post_risk_calibration_action": self._action_value(decision_meta.get("post_risk_calibration_action", action_id), action_id),
                "query_state_fast_proposal_action": self._action_value(decision_meta.get("query_state_fast_proposal_action", action_id), action_id),
                "query_state_slow_released_action": self._action_value(decision_meta.get("query_state_slow_released_action", action_id), action_id),
                "query_state_route_divergence": bool(decision_meta.get("query_state_route_divergence", False)),
                "llm_action_preserved": bool(decision_meta.get("llm_action_preserved", False)),
                "slow_action_risk": self._float(decision_meta.get("slow_action_risk"), 0.0),
                "slow_raw_action_risk": self._float(decision_meta.get("slow_raw_action_risk"), 0.0),
                "slow_min_action": self._action_value(decision_meta.get("slow_min_action", action_id), action_id),
                "slow_min_action_risk": self._float(decision_meta.get("slow_min_action_risk"), 0.0),
                "slow_risk_gap": self._float(decision_meta.get("slow_risk_gap"), 0.0),
                "slow_risk_calibration_threshold": self._float(decision_meta.get("slow_risk_calibration_threshold"), 0.0),
                "slow_risk_calibration_applied": bool(decision_meta.get("slow_risk_calibration_applied", False)),
                "slow_risk_calibration_reason": str(decision_meta.get("slow_risk_calibration_reason", "") or ""),
                "risk_scores_by_action": deepcopy(decision_meta.get("risk_scores_by_action", {})),
                "recoverability_counterfactual_incremental_utility": self._float(decision_meta.get("recoverability_counterfactual_incremental_utility"), 0.0),
                "recoverability_counterfactual_gain": self._float(decision_meta.get("recoverability_counterfactual_gain"), 0.0),
                "memory_retrieval_enabled": bool(decision_meta.get("memory_retrieval_enabled", False)),
                "memory_items_used": int(decision_meta.get("memory_items_used", 0) or 0),
                "few_shot_num": int(decision_meta.get("few_shot_num", 0) or 0),
                "memory_ids": deepcopy(decision_meta.get("memory_ids", [])),
                "memory_actions": deepcopy(decision_meta.get("memory_actions", [])),
                "memory_actions_available": deepcopy(decision_meta.get("memory_actions_available", [])),
                "memory_similarities": deepcopy(decision_meta.get("memory_similarities", [])),
                "trace_cache_enabled": bool(decision_meta.get("trace_cache_enabled", False)),
                "trace_cache_hit": bool(decision_meta.get("trace_cache_hit", False)),
            },
            "safety_envelope": {
                "proposed_action": proposed_action,
                "final_action": final_action,
                "route_action_changed": bool(decision_meta.get("route_action_changed", False)),
                "safety_override": safety_override,
                "shield_override": shield_override,
                "emergency_level": int(decision_meta.get("emergency_level", 0) or 0),
                "risk_event": bool(decision_meta.get("risk_event", False)),
            },
            "baseline_trigger_scores": {
                "paper_baseline_trigger_mode": str(decision_meta.get("paper_baseline_trigger_mode", "none") or "none"),
                "complexity_route_score": self._float(decision_meta.get("complexity_route_score"), 0.0),
                "ttc_route_score": self._float(decision_meta.get("ttc_route_score"), 0.0),
            },
            "ambiguity_and_conflict": {
                "route_ambiguity_profile": deepcopy(decision_meta.get("route_ambiguity_profile")),
                "route_ambiguity_entropy": self._float(decision_meta.get("route_ambiguity_entropy"), 0.0),
                "route_ambiguity_gap": self._float(decision_meta.get("route_ambiguity_gap"), 0.0),
                "route_ambiguity_disagreement": self._float(decision_meta.get("route_ambiguity_disagreement"), 0.0),
            },
        }

    def record_reasoning(
        self,
        frame_id: int,
        scenario_description: str,
        available_actions: str,
        driving_intention: str,
        full_response: str,
        action_id: int,
        inference_start_time: float,
        inference_end_time: float,
        input_text: str = "",
        risk_info: Optional[Dict[str, Any]] = None,
        decision_meta: Optional[Dict[str, Any]] = None,
    ) -> ReasoningRecord:
        del input_text, risk_info
        decision_meta = dict(decision_meta or {})
        latency = max(0.0, float(inference_end_time) - float(inference_start_time))
        record = ReasoningRecord(
            frame_id=int(frame_id),
            timestamp=float(frame_id),
            scenario_description=str(scenario_description),
            available_actions=str(available_actions),
            driving_intention=str(driving_intention),
            full_response=str(full_response),
            predicted_action_id=int(action_id),
            predicted_action_name=ACTIONS_ALL.get(int(action_id), "UNKNOWN"),
            inference_latency=latency,
            system_used=str(decision_meta.get("system_used", "unknown") or "unknown"),
            route_reason=str(decision_meta.get("route_reason", "unknown") or "unknown"),
            rgd_execution_route_score=self._float(decision_meta.get("rgd_execution_route_score", decision_meta.get("rad_recovery_cost_target")), 0.0),
            fast_rule_name=str(decision_meta.get("fast_rule_name", "none") or "none"),
            fast_smoothness_override=bool(decision_meta.get("fast_smoothness_override", False)),
            slow_reasoning_mode=str(decision_meta.get("slow_reasoning_mode", "unknown") or "unknown"),
            slow_reasoning_success=bool(decision_meta.get("slow_reasoning_success", False)),
            slow_reasoning_failure_reason=str(decision_meta.get("slow_reasoning_failure_reason", "") or ""),
            rgd_subordinate_diagnostics=self._build_grouped_diagnostics(int(action_id), decision_meta),
        )
        self.records.append(record)
        return record

    def calculate_episode_metrics(self) -> EpisodeReasoningMetrics:
        if not self.records:
            raise ValueError("No reasoning records available")
        latencies = [float(record.inference_latency) for record in self.records]
        slow_count = sum(str(record.system_used).lower() == "slow" for record in self.records)
        return EpisodeReasoningMetrics(
            episode_id=self.episode_id,
            total_frames=len(self.records),
            total_inference_time=float(sum(latencies)),
            avg_inference_latency=float(sum(latencies) / len(latencies)),
            max_inference_latency=float(max(latencies)),
            min_inference_latency=float(min(latencies)),
            slow_frame_count=int(slow_count),
            fast_frame_count=int(len(self.records) - slow_count),
        )

    def save(self) -> Dict[str, Any]:
        """Persist the self-contained reasoning trace used by aggregators."""
        root = Path(self.result_folder)
        root.mkdir(parents=True, exist_ok=True)
        records = [record.to_dict() for record in self.records]
        payload: Dict[str, Any] = {
            "episode_id": int(self.episode_id),
            "analysis_records": records,
            "record_count": len(records),
        }
        if records:
            payload["metrics"] = self.calculate_episode_metrics().to_dict()
        path = root / f"reasoning_records_{self.episode_id}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return payload
