"""Core Recoverability-Gated Deliberation router."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dilu.driver_agent.base.state import DrivingState
from dilu.driver_agent.base.state import ActionType
from dilu.driver_agent.policy_state import (
    RGD_POLICY_STATE_SCHEMA,
    validate_rgd_policy_state,
)
from dilu.driver_agent.reasoning.decision import RouteAmbiguityProfile, RGDDecision
from dilu.driver_agent.reasoning.fast_thinker import FastThinker
from dilu.driver_agent.reasoning.rad import RADSignalController
from dilu.driver_agent.reasoning.rgd_support import (
    RGDExecutionContract,
    RecoverabilityAssessment,
    RecoverabilityRoutingDecision,
    build_rgd_execution_contract,
    build_failure_pre_screen_config,
    build_slow_path_latency_context,
    bridge_orchestrator_stats_into_decision,
    compute_failure_pre_screen,
    compute_recoverability_assessment,
    compute_recoverability_collapse_risk,
    compute_recoverability_gate_diagnostics,
    export_route_ambiguity_to_decision,
    resolve_closed_recoverability_route,
    resolve_release_dominance_guard,
    resolve_paper_baseline_trigger,
)
from dilu.driver_agent.reasoning.slow_thinker import SlowPathUnavailableError, SlowThinker
from dilu.driver_agent.reasoning.support_memory import StateMemoryRetriever, resolve_memory_path
from dilu.latency_contract import resolve_latency_contract
from dilu.utils.junction_gap import assess_junction_gap
from dilu.utils.shared import float_or_default



logger = logging.getLogger(__name__)

FAST_INCUMBENT_CONTRACT_VERSION = "fast_incumbent_v1"
FAST_INCUMBENT_STAGE = "query_state_complete_fast_stack_pre_route_pre_safety"
FAST_INCUMBENT_SOURCE = "recoverability_provisional_fast_action"


def _action_id_or_default(value: Any, default: int = 1) -> int:
    return int(default if value in (None, "") else value)


class RGDOrchestrator:
    """Recoverability-gated fast/slow route authority for closed-loop driving.

    Sits between the fast executor, slow executor, and downstream safety map.
    A frame is upgraded to the slow path only when the locked recoverability
    estimator (and optional pre-screen) indicates post-latency corrective
    opportunity under the remaining attempt budget. Slow-path failures fall
    back to the identical fast policy; attempt accounting is incremented
    before invocation so failed calls still consume budget.
    """

    def __init__(
        self,
        fast_thinker: FastThinker,
        slow_thinker: SlowThinker,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fast = fast_thinker
        self.slow = slow_thinker
        self.config: Dict[str, Any] = dict(config or {})

        self.stats: Dict[str, Any] = {
            "fast_decisions": 0,
            "slow_decisions": 0,
            "total_latency_ms": 0,
            "decision_count": 0,
        }

        risk_cfg = self.config.get("risk_coupling", {}) or {}
        self._env_type = str(self.config.get("env_type", "") or "").strip()
        self._target_speed_max = self._resolve_env_float("target_speed_max", 31.0)
        self._target_speed_min = self._resolve_env_float("target_speed_min", 18.0)

        self._predictive_route_cfg = {
            "enable": False,
        }
        self._failure_pre_screen_cfg = build_failure_pre_screen_config(
            risk_cfg,
            env_type=self._env_type,
        )
        self._rgd_signal_provider = bool(self.config.get("rgd_signal_provider", True))

        core_story_cfg = risk_cfg.get("core_story", {}) or {}
        self._route_story_mode = "rgd_core"
        self._runtime_contract: RGDExecutionContract = build_rgd_execution_contract(core_story_cfg)  # Build the single canonical runtime contract once so routing logic and result manifests share the same resolved definitions.
        self._recoverability_gate = self._runtime_contract.gate_definition
        self._paper_baseline = self._runtime_contract.paper_baseline
        routing_cfg = self.config.get("system_routing", {}) or {}
        forced_simple = str(routing_cfg.get("simple", "")).strip().lower()
        forced_complex = str(routing_cfg.get("complex", "")).strip().lower()
        self._forced_route_system: Optional[str] = None
        if forced_simple in {"fast", "slow"} and forced_simple == forced_complex:
            self._forced_route_system = forced_simple

        rad_cfg = dict(risk_cfg.get("rad_core", {}) or {})
        self._rad_controller = RADSignalController(rad_cfg)
        self._last_rad_meta: Dict[str, Any] = {}
        self._last_route_feature_vec: Optional[np.ndarray] = None
        self._last_route_ambiguity_profile: Optional[RouteAmbiguityProfile] = None
        self._last_recoverability_context: Dict[str, Any] = {}
        self._current_fast_incumbent_identity: Dict[str, Any] = {}
        self._current_fast_override_context: Dict[str, Any] = {}
        self._support_progress_cooldown = 0
        self._rgd_cruise_progress_cooldown = 0
        self._rgd_cruise_recovery_frames = 0
        budget_raw = self.config.get("slow_call_budget")
        self._slow_call_budget: Optional[int] = None if budget_raw is None else max(0, int(budget_raw))
        self._slow_call_cooldown_frames = max(0, int(self.config.get("slow_call_cooldown_frames", 0) or 0))
        self._rgd_min_observation_frames = max(1, int(self.config.get("rgd_min_observation_frames", 1) or 1))
        self._slow_call_attempts = 0
        self._slow_call_cooldown_remaining = 0
        self._support_memory_retriever: Optional[StateMemoryRetriever] = None
        if bool(self.config.get("enable_memory_retrieval", False)):
            few_shot_num = int(self.config.get("few_shot_num", 0) or 0)
            self._support_memory_retriever = StateMemoryRetriever(
                resolve_memory_path(str(self.config.get("memory_path", "") or "")),
                top_k=max(1, few_shot_num),
            )

        # Higher values require stronger recoverability evidence to escalate to slow path.
        self._rgd_decision_threshold: Optional[float] = None
        _threshold_raw = self.config.get("rgd_decision_threshold")
        _threshold_source = "default_recoverability_score_boundary"
        _base_threshold = None
        _env_threshold = None
        _env_override_applied = False
        if _threshold_raw is not None:
            self._rgd_decision_threshold = float(_threshold_raw)
            _base_threshold = float(_threshold_raw)
            _threshold_source = "runtime_config.rgd_decision_threshold"
        _threshold_by_env_raw = self.config.get("rgd_decision_threshold_by_env", {}) or {}
        if isinstance(_threshold_by_env_raw, dict):
            current_env_name = str(self.config.get("env_type", "") or "").strip()
            env_specific_threshold = _threshold_by_env_raw.get(current_env_name)
            if env_specific_threshold is not None:
                self._rgd_decision_threshold = float(env_specific_threshold)
                _env_threshold = float(env_specific_threshold)
                _env_override_applied = True
                _threshold_source = f"runtime_config.rgd_decision_threshold_by_env.{current_env_name}"
        self._rgd_threshold_audit_band = max(0.0, float_or_default(self.config.get("rgd_threshold_audit_band", 0.03), 0.03))
        self._rgd_threshold_provenance = {
            "source": str(self.config.get("rgd_decision_threshold_source", _threshold_source) or _threshold_source),
            "selection_rule": str(self.config.get("rgd_threshold_selection_rule", "unspecified") or "unspecified"),
            "claim_scope": str(self.config.get("rgd_threshold_claim_scope", "method_route_boundary") or "method_route_boundary"),
            "env_type": self._env_type,
            "base_threshold": _base_threshold,
            "env_override_applied": bool(_env_override_applied),
            "env_threshold": _env_threshold,
            "effective_threshold": self._rgd_decision_threshold,
            "audit_band": float(self._rgd_threshold_audit_band),
        }

    def _resolve_env_float(self, key: str, default: float) -> float:
        by_env = self.config.get(f"{key}_by_env", {}) or {}
        if isinstance(by_env, dict) and self._env_type in by_env:
            return float_or_default(by_env.get(self._env_type), default)
        return float_or_default(self.config.get(key), default)

    def _build_core_junction_pre_screen(self, state: DrivingState) -> Dict[str, Any]:
        """Expose junction reservation loss as RGD compute value, not a safety afterthought."""
        scenario_key = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        if scenario_key not in {"intersection", "roundabout"}:
            return {"pre_screen_score": 0.0, "pre_screen_trigger": False, "pre_screen_reason": "none", "components": {}}

        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        conflict_candidates: List[float] = []
        for raw_value in (
            getattr(state, "cross_traffic_distance", None),
            getattr(state, "closest_vehicle_distance", None),
        ):
            value = float_or_default(raw_value, float("inf"))
            if math.isfinite(value) and value >= 0.0:
                conflict_candidates.append(value)
        if not conflict_candidates:
            return {"pre_screen_score": 0.0, "pre_screen_trigger": False, "pre_screen_reason": "none", "components": {}}

        conflict_distance = float(min(conflict_candidates))
        if scenario_key == "intersection" and speed <= 0.6 and conflict_distance >= 10.0:
            return {
                "pre_screen_score": 0.0,
                "pre_screen_trigger": False,
                "pre_screen_reason": "low_speed_waiting",
                "pre_screen_source": "core_junction_reservation",
                "components": {
                    "junction_reservation": 0.0,
                    "junction_conflict_distance": float(conflict_distance),
                    "junction_required_clearance": 0.0,
                },
            }
        safety_cfg = dict(self.config.get("safety_thresholds", {}) or {})
        rss_cfg = dict(self.config.get("rss_params", {}) or {})
        min_distance = max(6.0, float_or_default(safety_cfg.get("min_front_distance", None), 10.5))
        reaction_time = max(0.0, float_or_default(rss_cfg.get("reaction_time", None), 0.25))
        brake = max(0.1, float_or_default(rss_cfg.get("min_brake_decel", None), 3.5))
        stopping_corridor = speed * reaction_time + (speed * speed) / (2.0 * brake)
        occupancy_corridor = speed * 3.0
        required_clearance = max(stopping_corridor, occupancy_corridor) + max(min_distance, 10.5)
        if conflict_distance >= required_clearance:
            return {
                "pre_screen_score": 0.0,
                "pre_screen_trigger": False,
                "pre_screen_reason": "none",
                "components": {
                    "junction_reservation": 0.0,
                    "junction_conflict_distance": float(conflict_distance),
                    "junction_required_clearance": float(required_clearance),
                },
            }

        deficit = float(required_clearance - conflict_distance)
        speed_pressure = min(1.0, speed / (8.0 if scenario_key == "roundabout" else 12.0))
        score = float(max(0.72, min(0.98, 0.70 + 0.22 * min(1.0, deficit / 4.0) + 0.08 * speed_pressure)))
        return {
            "pre_screen_score": float(score),
            "pre_screen_trigger": True,
            "pre_screen_reason": "cross_traffic",
            "pre_screen_source": "core_junction_reservation",
            "components": {
                "junction_reservation": float(score),
                "junction_conflict_distance": float(conflict_distance),
                "junction_required_clearance": float(required_clearance),
                "junction_clearance_deficit": float(deficit),
            },
        }

    @staticmethod
    def _merge_pre_screen(primary: Dict[str, Any], secondary: Dict[str, Any]) -> Dict[str, Any]:
        """Keep the strongest pre-screen source while preserving component diagnostics."""
        primary = dict(primary or {})
        secondary = dict(secondary or {})
        primary_score = float_or_default(primary.get("pre_screen_score", None), 0.0)
        secondary_score = float_or_default(secondary.get("pre_screen_score", None), 0.0)
        if bool(secondary.get("pre_screen_trigger", False)) and (
            not bool(primary.get("pre_screen_trigger", False)) or secondary_score > primary_score
        ):
            merged = dict(secondary)
        else:
            merged = dict(primary)
        components = dict(primary.get("components", {}) or {})
        components.update(dict(secondary.get("components", {}) or {}))
        merged["components"] = components
        merged["soft_recoverability_floor"] = float_or_default(
            primary.get("soft_recoverability_floor", secondary.get("soft_recoverability_floor", 0.20)),
            0.20,
        )
        return merged

    def _execute_slow_path(
        self,
        state: DrivingState,
        route_score: float,
        route_ambiguity_profile: Optional[RouteAmbiguityProfile],
        recoverability_context: Optional[Dict[str, Any]],
    ) -> RGDDecision:
        """Execute the slow path and return the routed decision."""
        incumbent_bound = bool(self._current_fast_incumbent_identity)
        query_fast = self._peek_full_fast_decision(
            state,
            fast_override_context=(
                self._current_fast_override_context if incumbent_bound else None
            ),
        )
        if incumbent_bound:
            self._assert_current_fast_incumbent_match(
                state=state,
                observed_decision=query_fast,
                observed_source="matched_fast_query",
            )
        slow_decision = self.slow.think(state=state, recoverability_context=recoverability_context)
        slow_meta = dict(getattr(slow_decision, "agent_opinions", {}) or {})  # Collect slow-path diagnostic exports before building the routed decision wrapper.
        pass_meta = self._highway_fast_pass_override(state, int(slow_decision.action))
        pre_guard_action = int(pass_meta.get("rgd_highway_pass_resolved_action", slow_decision.action))
        selected_action = int(pre_guard_action)
        guard_cfg = dict(self.config.get("release_dominance_guard", {}) or {})
        guard_enabled = bool(guard_cfg.get("enable", False))
        latency_contract = resolve_latency_contract(self.config)
        effective_delay_steps = max(
            0, int(latency_contract.get("scheduled_steps", 0) or 0)
        )
        guard_meta: Dict[str, Any] = {
            "release_dominance_guard_enabled": bool(guard_enabled),
            "release_dominance_guard_scheduled_delay_steps": int(
                effective_delay_steps
            ),
            "release_dominance_guard_scope": (
                "query_equals_release_zero_delay" if effective_delay_steps == 0
                else "deferred_positive_delay_release"
            ),
            "release_dominance_guard_applied": False,
            "release_dominance_guard_pre_guard_action": int(pre_guard_action),
        }
        if guard_enabled and effective_delay_steps == 0:
            selected_action, resolved_guard = resolve_release_dominance_guard(
                slow_action=int(pre_guard_action),
                matched_fast_action=int(query_fast.action),
                risk_scores=dict(
                    slow_meta.get("risk_scores_by_action", slow_meta.get("slow_risk_scores", {})) or {}
                ),
                risk_margin=float(guard_cfg.get("risk_margin", 0.0) or 0.0),
                require_strict_improvement=bool(
                    guard_cfg.get("require_strict_improvement", True)
                ),
                progress_guard={
                    "speed": float_or_default(getattr(state, "ego_speed", None), float("nan")),
                    "front_distance": float_or_default(getattr(state, "front_distance", None), float("nan")),
                    "ttc": float_or_default(getattr(state, "ttc", None), float("nan")),
                    "thw": float_or_default(getattr(state, "thw", None), float("nan")),
                },
            )
            guard_meta.update(resolved_guard)
        selected_reasoning = str(slow_decision.reasoning)
        if bool(pass_meta.get("rgd_highway_pass_applied", False)):
            selected_reasoning = f"{selected_reasoning} | rgd_highway_pass: {ActionType.to_english(pre_guard_action)}"
        if bool(guard_meta.get("release_dominance_guard_fallback_to_fast", False)):
            selected_reasoning = (
                f"{selected_reasoning} | release dominance: "
                f"{ActionType.to_english(pre_guard_action)}->{ActionType.to_english(selected_action)}"
            )
        decision = RGDDecision(
            action=selected_action,
            reasoning=selected_reasoning,
            confidence=slow_decision.confidence,
            system_used="slow",
            route_label=self._route_story_mode,
            route_score=float(route_score),
            ambiguity_profile=route_ambiguity_profile,
            stats={
                "depth": 2,
                "thinking_steps": getattr(slow_decision, "thinking_steps", []),
                "query_state_fast_proposal_action": int(query_fast.action),
                "query_state_fast_proposal_rule": str(query_fast.stats.get("rule_name") or "unknown"),
                "query_state_fast_incumbent_identity_sha256": str(
                    self._current_fast_incumbent_identity.get("identity_sha256", "")
                ),
                "query_state_fast_incumbent_identity_match": bool(incumbent_bound),
                "query_state_slow_pre_guard_action": int(pre_guard_action),
                "query_state_slow_released_action": int(selected_action),
                "query_state_route_divergence": bool(int(selected_action) != int(query_fast.action)),
                **slow_meta,
                **pass_meta,
                **guard_meta,
            },
        )
        return decision

    def _peek_full_fast_decision(
        self,
        state: DrivingState,
        fast_override_context: Optional[Dict[str, Any]] = None,
    ) -> RGDDecision:
        """Evaluate the complete fast stack without advancing its state.

        This includes the deterministic fast thinker and every support-layer
        adjustment that an always-fast arm would execute on the same frame.
        """
        fast_snapshot = self.fast.snapshot_runtime_state()
        support_snapshot = (
            int(self._support_progress_cooldown),
            int(self._rgd_cruise_progress_cooldown),
            int(self._rgd_cruise_recovery_frames),
        )
        fast_decision_count_snapshot = int(self.stats.get("fast_decisions", 0) or 0)
        try:
            return self._execute_fast(
                state,
                route_score=0.0,
                fast_override_context=fast_override_context,
            )
        finally:
            self.fast.restore_runtime_state(fast_snapshot)
            (
                self._support_progress_cooldown,
                self._rgd_cruise_progress_cooldown,
                self._rgd_cruise_recovery_frames,
            ) = support_snapshot
            self.stats["fast_decisions"] = int(fast_decision_count_snapshot)

    @staticmethod
    def _fast_override_context_payload(
        override_context: Optional[Dict[str, Any]],
    ) -> Dict[str, float]:
        context = dict(override_context or {})
        gate = dict(context.get("gate_diagnostics", {}) or {})
        assessment = dict(context.get("recoverability_assessment", {}) or {})
        pre_screen = dict(context.get("failure_pre_screen", {}) or {})
        ambiguity = dict(context.get("route_ambiguity_profile", {}) or {})

        def unit(value: Any) -> float:
            return float(max(0.0, min(1.0, float_or_default(value, 0.0))))

        return {
            "collapse_risk": max(
                unit(gate.get("collapse_risk")),
                unit(assessment.get("collapse_risk")),
            ),
            "pre_screen_score": max(
                unit(gate.get("pre_screen_score")),
                unit(pre_screen.get("pre_screen_score")),
            ),
            "ambiguity_intervention_risk": unit(
                ambiguity.get("intervention_risk")
            ),
        }

    @staticmethod
    def _canonical_payload_sha256(payload: Dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_fast_incumbent_identity(
        self,
        *,
        action: int,
        action_universe: Tuple[int, ...],
        override_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context_payload = self._fast_override_context_payload(override_context)
        context_sha = self._canonical_payload_sha256(context_payload)
        identity_payload = {
            "action_id": int(action),
            "action_universe": [int(item) for item in action_universe],
            "override_context_sha256": str(context_sha),
            "stage": FAST_INCUMBENT_STAGE,
            "source": FAST_INCUMBENT_SOURCE,
            "contract_version": FAST_INCUMBENT_CONTRACT_VERSION,
        }
        return {
            **identity_payload,
            "identity_sha256": self._canonical_payload_sha256(identity_payload),
        }

    def _assert_fast_incumbent_match(
        self,
        *,
        expected_identity: Dict[str, Any],
        observed_decision: RGDDecision,
        action_universe: Tuple[int, ...],
        override_context: Optional[Dict[str, Any]],
        observed_source: str,
    ) -> bool:
        expected_identity = dict(expected_identity or {})
        observed_identity = self._build_fast_incumbent_identity(
            action=int(observed_decision.action),
            action_universe=tuple(int(item) for item in action_universe),
            override_context=override_context,
        )
        expected_payload_valid = False
        try:
            expected_payload = {
                "action_id": int(expected_identity["action_id"]),
                "action_universe": [
                    int(item) for item in expected_identity["action_universe"]
                ],
                "override_context_sha256": str(
                    expected_identity["override_context_sha256"]
                ),
                "stage": str(expected_identity["stage"]),
                "source": str(expected_identity["source"]),
                "contract_version": str(expected_identity["contract_version"]),
            }
            expected_payload_valid = bool(
                expected_payload["stage"] == FAST_INCUMBENT_STAGE
                and expected_payload["source"] == FAST_INCUMBENT_SOURCE
                and expected_payload["contract_version"]
                == FAST_INCUMBENT_CONTRACT_VERSION
                and self._canonical_payload_sha256(expected_payload)
                == str(expected_identity.get("identity_sha256", "") or "")
            )
        except (KeyError, TypeError, ValueError):
            expected_payload_valid = False
        matches = bool(
            expected_payload_valid
            and int(observed_decision.action) == int(expected_identity["action_id"])
            and [int(item) for item in action_universe]
            == [int(item) for item in expected_identity["action_universe"]]
            and observed_identity["override_context_sha256"]
            == str(expected_identity["override_context_sha256"])
            and observed_identity["identity_sha256"]
            == str(expected_identity["identity_sha256"])
        )
        stat_key = {
            "actual_fast_execution": "rgd_actual_fast_identity_match",
            "matched_fast_query": "rgd_matched_fast_identity_match",
        }.get(str(observed_source), "rgd_fast_incumbent_identity_match")
        self.stats[stat_key] = bool(matches)
        self.stats["rgd_fast_incumbent_expected_identity_valid"] = bool(
            expected_payload_valid
        )
        self.stats["rgd_fast_incumbent_observed_identity_sha256"] = str(
            observed_identity["identity_sha256"]
        )
        if not matches:
            raise RuntimeError(
                "fast incumbent identity drift; "
                f"source={observed_source}; "
                f"expected={expected_identity.get('identity_sha256')}; "
                f"observed={observed_identity.get('identity_sha256')}"
            )
        return True

    def _assert_current_fast_incumbent_match(
        self,
        *,
        state: DrivingState,
        observed_decision: RGDDecision,
        observed_source: str,
    ) -> bool:
        expected_identity = dict(
            getattr(self, "_current_fast_incumbent_identity", {}) or {}
        )
        if not expected_identity:
            raise RuntimeError("frozen fast incumbent identity is missing")
        frozen_context = dict(
            getattr(self, "_current_fast_override_context", {}) or {}
        )
        if not frozen_context:
            raise RuntimeError("frozen H-independent fast override context is missing")
        self._assert_fast_incumbent_match(
            expected_identity=expected_identity,
            observed_decision=observed_decision,
            action_universe=tuple(int(action) for action in state.get_available_actions()),
            override_context=frozen_context,
            observed_source=observed_source,
        )
        observed_decision.stats.update({
            "fast_incumbent_action_id": int(expected_identity["action_id"]),
            "fast_incumbent_identity_sha256": str(
                expected_identity["identity_sha256"]
            ),
            "fast_incumbent_identity_match": True,
            "fast_incumbent_observed_source": str(observed_source),
        })
        return True

    def _build_h_independent_fast_override_context(
        self,
        *,
        principle: Dict[str, Any],
        rad_meta: Dict[str, Any],
        route_ambiguity_profile: Optional[RouteAmbiguityProfile],
        failure_pre_screen: Dict[str, Any],
    ) -> Dict[str, Any]:
        collapse_risk = compute_recoverability_collapse_risk(
            principle, rad_meta
        )
        return {
            "recoverability_assessment": {
                "collapse_risk": float(collapse_risk),
            },
            "gate_diagnostics": {
                "collapse_risk": float(collapse_risk),
                "pre_screen_score": float_or_default(
                    failure_pre_screen.get("pre_screen_score"), 0.0
                ),
            },
            "route_ambiguity_profile": (
                {
                    "intervention_risk": float(
                        route_ambiguity_profile.intervention_risk
                    )
                }
                if route_ambiguity_profile is not None
                else {}
            ),
            "failure_pre_screen": {
                "pre_screen_score": float_or_default(
                    failure_pre_screen.get("pre_screen_score"), 0.0
                ),
            },
            "context_stage": "current_frame_frozen_h_independent_fast_override",
        }

    @staticmethod
    def _build_final_recoverability_context(
        *,
        recoverability_assessment: RecoverabilityAssessment,
        gate_diagnostics: Dict[str, Any],
        route_ambiguity_profile: Optional[RouteAmbiguityProfile],
        routing_decision: Optional[RecoverabilityRoutingDecision],
        latency_context: Dict[str, Any],
        failure_pre_screen: Dict[str, Any],
        incumbent_identity: Dict[str, Any],
        fast_override_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "recoverability_assessment": recoverability_assessment.to_dict(),
            "gate_diagnostics": dict(gate_diagnostics or {}),
            "route_ambiguity_profile": (
                route_ambiguity_profile.to_dict()
                if route_ambiguity_profile is not None
                else {}
            ),
            "routing_decision": (
                routing_decision.to_dict()
                if routing_decision is not None
                else None
            ),
            "latency_context": dict(latency_context or {}),
            "failure_pre_screen": dict(failure_pre_screen or {}),
            "provisional_action": int(incumbent_identity["action_id"]),
            "fast_incumbent_identity": dict(incumbent_identity),
            "fast_override_context": dict(fast_override_context),
            "context_stage": "current_frame_post_gate_frozen_incumbent",
        }

    def _collect_route_signals(
        self,
        state: DrivingState,
    ) -> Dict[str, Any]:
        """Collect the route evidence bundle used by the RGD kernel."""
        self._current_fast_incumbent_identity = {}
        self._current_fast_override_context = {}
        conflict_score = 0.0
        rad_conflict_score = 0.0
        fast_executor_action_universe, action_universe_provenance = (
            self._resolve_fast_executor_action_universe(state)
        )
        resolved_state_universe = tuple(int(action) for action in state.get_available_actions())
        slow_executor_action_universe = (
            tuple(self.slow.get_available_action_universe(state))
            if hasattr(self.slow, "get_available_action_universe")
            else resolved_state_universe
        )
        if resolved_state_universe != fast_executor_action_universe:
            raise RuntimeError(
                "DrivingState effective universe diverged from fast executor contract; "
                f"state={resolved_state_universe}; fast={fast_executor_action_universe}"
            )
        if slow_executor_action_universe != fast_executor_action_universe:
            raise RuntimeError(
                "slow executor action universe diverged from fast/gate contract; "
                f"slow={slow_executor_action_universe}; fast={fast_executor_action_universe}"
            )
        self.stats["rgd_effective_action_universe"] = list(fast_executor_action_universe)
        self.stats["rgd_fast_executor_action_universe"] = list(fast_executor_action_universe)
        self.stats["rgd_slow_executor_action_universe"] = list(
            slow_executor_action_universe
        )
        self.stats["rgd_effective_action_universe_source"] = str(
            getattr(state, "effective_action_universe_source", "unknown") or "unknown"
        )
        rad_score, rad_meta = self._rad_controller.estimate_signal(
            state,
            rad_conflict_score,
            action_universe=fast_executor_action_universe,
        )
        rad_meta.update({
            "method_version": "identifiable_gate_v12",
            "gate_action_universe": list(fast_executor_action_universe),
            "fast_executor_action_universe": list(fast_executor_action_universe),
            "slow_executor_action_universe": list(slow_executor_action_universe),
            "effective_action_universe": list(resolved_state_universe),
            "gate_action_universe_source": "driving_state.effective_action_universe",
            "fast_executor_action_universe_source": "driving_state.effective_action_universe",
            "fast_executor_action_universe_provenance": dict(action_universe_provenance),
        })
        rad_meta["score"] = float(rad_score)
        rad_meta["recovery_cost_target"] = float(rad_meta.get("recovery_cost_target", rad_score))
        rad_meta["recovery_objective_value"] = float(rad_meta.get("recovery_objective_value", rad_meta["recovery_cost_target"]))
        self._last_rad_meta = rad_meta
        self.stats["complexity_route_score"] = float(max(
            rad_meta.get("ttc_pressure", 0.0) or 0.0,
            rad_meta.get("proximity_complexity", 0.0) or 0.0,
        ))
        self.stats["ttc_route_score"] = float(max(
            rad_meta.get("ttc_pressure", 0.0) or 0.0,
            0.5 * float(rad_meta.get("proximity_complexity", 0.0) or 0.0),
        ))
        self.stats["rad_best_recovery_cost"] = float(rad_meta.get("best_recovery_cost", 0.0) or 0.0)
        self.stats["rad_recovery_regret"] = float(rad_meta.get("recovery_regret", 0.0) or 0.0)
        self.stats["rad_corridor_entropy"] = float(rad_meta.get("corridor_entropy", 0.0) or 0.0)
        self.stats["rad_recovery_margin"] = float(rad_meta.get("recovery_margin", rad_meta.get("recovery_objective_margin", 0.0)) or 0.0)
        self.stats["rad_irreversibility"] = float(rad_score)
        self.stats["rad_recovery_cost_target"] = float(rad_meta.get("recovery_cost_target", rad_score))
        self.stats["rad_recovery_objective_value"] = float(rad_meta.get("recovery_objective_value", rad_meta.get("recovery_cost_target", rad_score)))
        self.stats["rad_recovery_objective_margin"] = float(rad_meta.get("recovery_objective_margin", 0.0))
        if rad_meta.get("best_action") is not None:
            self.stats["rad_best_action"] = int(rad_meta["best_action"])

        principle = rad_meta.get("recoverability_heuristic", {}) or {}
        self.stats["rad_principle_satisfied"] = bool(principle.get("principle_satisfied", True))
        self.stats["rad_principle_stage"] = str(principle.get("stage", "unknown"))
        self.stats["rad_corridor_width"] = int(principle.get("corridor_width", 0))
        self.stats["rad_corridor_width_raw"] = int(principle.get("corridor_width_raw", principle.get("corridor_width", 0)) or 0)
        self.stats["rad_recovery_budget"] = float(principle.get("recovery_budget_remaining", 1.0))
        self.stats["rad_cost_slope"] = float(principle.get("cost_slope", 0.0))
        self.stats["rad_curve_stage"] = str(principle.get("rollout_curve_stage", "no_wm"))
        self.stats["rad_corridor_boundary"] = float(principle.get("corridor_boundary", 0.0) or 0.0)
        self.stats["rad_corridor_boundary_raw"] = float(principle.get("corridor_boundary_raw", principle.get("corridor_boundary", 0.0)) or 0.0)
        self.stats["rad_corridor_boundary_delta"] = float(principle.get("corridor_boundary_delta", 0.0) or 0.0)
        self.stats["rad_corridor_width_delta"] = float(principle.get("corridor_width_delta", 0.0) or 0.0)
        self.stats["rad_corridor_stability_score"] = float(principle.get("corridor_stability_score", 0.0) or 0.0)
        self.stats["rad_near_commitment"] = bool(principle.get("near_commitment", False))
        self.stats["rad_near_commitment_clamped"] = bool(principle.get("near_commitment_clamped", False))
        self.stats["rad_near_commitment_contradiction"] = bool(principle.get("near_commitment_contradiction", False))
        self.stats["rad_principle_corridor_exists"] = float(principle.get("corridor_exists", 0.5))
        self.stats["rad_principle_dominance_margin"] = float(principle.get("dominance_margin", 0.5))
        self.stats["rad_reachable_safe_set_size"] = float(principle.get("reachable_safe_set_size", rad_meta.get("reachable_safe_set_size", 0.0)))
        self.stats["rad_reachable_safe_set_ratio"] = float(principle.get("reachable_safe_set_ratio", rad_meta.get("reachable_safe_set_ratio", 0.0)))
        self.stats["rad_viable_headroom"] = float(principle.get("viable_headroom", rad_meta.get("viable_headroom", 0.0)))
        self.stats["rad_viability_proxy_score"] = float(principle.get("viability_proxy_score", rad_meta.get("viability_proxy_score", 0.0)))
        self.stats["rad_short_horizon_irreversible_risk"] = float(principle.get("short_horizon_irreversible_risk", rad_meta.get("short_horizon_irreversible_risk", 0.0)))
        raw_support_best_action = rad_meta.get("best_support_action", 1)
        self.stats["rad_support_best_action"] = (
            1 if raw_support_best_action in (None, "") else int(raw_support_best_action)
        )
        self.stats["rad_support_best_cost"] = float(rad_meta.get("best_support_ranking_cost", rad_meta.get("best_recovery_cost", 0.0)) or 0.0)
        self.stats["rad_support_only_ranking_disagreement"] = bool(rad_meta.get("support_only_ranking_disagreement", False))

        failure_pre_screen = compute_failure_pre_screen(
            state=state,
            principle=principle,
            rad_meta=rad_meta,
            config=self._failure_pre_screen_cfg,
        )
        failure_pre_screen = self._merge_pre_screen(
            failure_pre_screen,
            self._build_core_junction_pre_screen(state),
        )
        runtime_budget_state = self.slow.get_runtime_budget_state()
        latency_contract = resolve_latency_contract(self.config)
        configured_latency = (
            float(latency_contract["predicted_seconds"])
            if bool(latency_contract.get("prediction_available", False))
            else None
        )
        latency_source = str(
            latency_contract.get("source", "missing_prediction")
            or "missing_prediction"
        )
        predicted_delay_steps = (
            int(latency_contract.get("predicted_steps", 0) or 0)
            if bool(latency_contract.get("prediction_available", False))
            else None
        )
        latency_context = build_slow_path_latency_context(
            llm_available=runtime_budget_state["llm_available"],
            llm_invoke_timeout_s=runtime_budget_state["llm_invoke_timeout_s"],
            short_horizon_seconds=float(max(0.2, float(rad_meta.get("prediction_horizon_s", principle.get("short_horizon_seconds", 0.8)) or 0.8))),
            state=state,
            predicted_slow_latency_s=configured_latency,
            latency_source=latency_source,
            safety_reserve_seconds=self._resolve_env_float("rgd_latency_safety_reserve_s", 0.0),
            policy_frequency=float_or_default(
                latency_contract.get("policy_frequency_hz"), 10.0
            ),
            resolved_delay_steps=predicted_delay_steps,
        )
        fused_route_score = self._compute_unified_route_score(
            lookahead_risk=None,
            irreversibility_score=float(rad_meta.get("recovery_cost_target", rad_score)),
        )
        route_ambiguity_profile = self._build_route_ambiguity_profile(conflict_score, fused_route_score)
        self._last_route_ambiguity_profile = route_ambiguity_profile
        fast_override_context = self._build_h_independent_fast_override_context(
            principle=principle,
            rad_meta=rad_meta,
            route_ambiguity_profile=route_ambiguity_profile,
            failure_pre_screen=failure_pre_screen,
        )
        provisional_fast_decision = self._peek_full_fast_decision(
            state,
            fast_override_context=fast_override_context,
        )
        provisional_fast_action = int(provisional_fast_decision.action)
        if provisional_fast_action not in set(fast_executor_action_universe):
            raise RuntimeError(
                "final fast proposal is outside the effective action universe; "
                f"proposal={provisional_fast_action}; universe={fast_executor_action_universe}"
            )
        incumbent_identity = self._build_fast_incumbent_identity(
            action=provisional_fast_action,
            action_universe=tuple(fast_executor_action_universe),
            override_context=fast_override_context,
        )
        self.stats["recoverability_provisional_fast_action"] = int(provisional_fast_action)
        self.stats["rgd_final_fast_proposal_in_effective_universe"] = True
        recoverability_assessment = compute_recoverability_assessment(
            gate_definition=self._recoverability_gate,
            principle=principle,
            rad_meta=rad_meta,
            route_ambiguity_profile=route_ambiguity_profile,
            latency_context=latency_context,
            pre_screen_context=failure_pre_screen,
            legal_actions=tuple(fast_executor_action_universe),
            hold_action=int(provisional_fast_action),
        )  # Build one unified recoverability object so routing can depend on a single decision scalar rather than multiple parallel gates.
        execution_route_score = float(recoverability_assessment.recoverability_score)
        gate_diagnostics = compute_recoverability_gate_diagnostics(gate_definition=self._recoverability_gate, principle=principle, rad_meta=rad_meta, recoverability_assessment=recoverability_assessment)  # Compute gate diagnostics once here so runtime routing and exported gate traces consume the same unified recoverability verdict.
        hold_incumbent_match = bool(
            int(recoverability_assessment.hold_action)
            == int(incumbent_identity["action_id"])
        )
        if not hold_incumbent_match:
            raise RuntimeError(
                "recoverability hold action diverged from frozen fast incumbent"
            )
        gate_diagnostics.update({
            "fast_incumbent_action_id": int(incumbent_identity["action_id"]),
            "fast_incumbent_identity_sha256": str(incumbent_identity["identity_sha256"]),
            "fast_incumbent_action_universe": list(incumbent_identity["action_universe"]),
            "fast_override_context_sha256": str(incumbent_identity["override_context_sha256"]),
            "fast_incumbent_identity_source": str(incumbent_identity["source"]),
            "fast_incumbent_identity_stage": str(incumbent_identity["stage"]),
            "fast_incumbent_contract_version": str(incumbent_identity["contract_version"]),
            "fast_incumbent_hold_match": bool(hold_incumbent_match),
        })
        route_threshold_override = self._rgd_decision_threshold
        routing_decision = resolve_closed_recoverability_route(
            stats=self.stats,
            recoverability_assessment=recoverability_assessment,
            gate_diagnostics=gate_diagnostics,
            decision_threshold_override=route_threshold_override,
        )  # Resolve the default RGD route from the closed recoverability object once so later stages cannot reintroduce override-style routing. # Closed-object route decision
        final_context = self._build_final_recoverability_context(
            recoverability_assessment=recoverability_assessment,
            gate_diagnostics=gate_diagnostics,
            route_ambiguity_profile=route_ambiguity_profile,
            routing_decision=routing_decision,
            latency_context=latency_context,
            failure_pre_screen=failure_pre_screen,
            incumbent_identity=incumbent_identity,
            fast_override_context=fast_override_context,
        )
        gate_diagnostics["fast_incumbent_single_freeze"] = True
        gate_diagnostics["fast_incumbent_final_context_repeek"] = False
        gate_diagnostics["fast_override_context_h_independent"] = True
        final_context["gate_diagnostics"] = dict(gate_diagnostics)
        self._last_recoverability_context = dict(final_context)
        self._current_fast_incumbent_identity = dict(incumbent_identity)
        self._current_fast_override_context = dict(fast_override_context)
        self.stats.update({
            "rgd_fast_incumbent_action_id": int(incumbent_identity["action_id"]),
            "rgd_fast_incumbent_identity_sha256": str(incumbent_identity["identity_sha256"]),
            "rgd_fast_incumbent_action_universe": list(incumbent_identity["action_universe"]),
            "rgd_fast_override_context_sha256": str(incumbent_identity["override_context_sha256"]),
            "rgd_fast_incumbent_identity_source": str(incumbent_identity["source"]),
            "rgd_fast_incumbent_identity_stage": str(incumbent_identity["stage"]),
            "rgd_fast_incumbent_contract_version": str(incumbent_identity["contract_version"]),
            "rgd_fast_incumbent_hold_match": True,
        })
        return {
            "conflict_score": float(conflict_score),
            "rad_conflict_score": float(rad_conflict_score),
            "rad_score": float(rad_score),
            "rad_meta": rad_meta,
            "principle": principle,
            "fused_route_score": float(fused_route_score),
            "execution_route_score": float(execution_route_score),
            "recoverability_assessment": recoverability_assessment,
            "routing_decision": routing_decision,
            "route_ambiguity_profile": route_ambiguity_profile,
            "gate_diagnostics": gate_diagnostics,
            "failure_pre_screen": failure_pre_screen,
            "latency_context": latency_context,
            "fast_incumbent_identity": dict(incumbent_identity),
            "recoverability_context": dict(final_context),
        }

    def _export_recoverability_gate_snapshot(
        self,
        route_signals: Dict[str, Any],
        selected_system: str,
        route_reason: str,
    ) -> None:
        """Export the paper-facing recoverability gate verdict as one compact runtime object."""
        gate_diagnostics = dict(route_signals.get("gate_diagnostics", {}) or {})
        routing_decision = route_signals.get("routing_decision")
        recoverability_assessment = route_signals.get("recoverability_assessment")
        public_coordinates = (
            recoverability_assessment.to_paper_dict()
            if hasattr(recoverability_assessment, "to_paper_dict")
            else {
                "recovery_window": 0.0,
                "action_space_affordance": 0.0,
                "commitment_reversibility": 0.0,
                "recoverable_deliberation_priority": float(gate_diagnostics.get("recoverability_score", 0.0) or 0.0),
            }
        )
        raw_hold_action = gate_diagnostics.get("hold_action", 1)
        resolved_hold_action = 1 if raw_hold_action in (None, "") else int(raw_hold_action)
        gate_snapshot = {
            "gate_active": bool(gate_diagnostics.get("rgd_gate_active", False)),
            "score": float(gate_diagnostics.get("recoverability_score", 0.0) or 0.0),
            "threshold": float(gate_diagnostics.get("score_boundary", 0.0) or 0.0),
            "margin": float(gate_diagnostics.get("score_gap", 0.0) or 0.0),
            "signed_margin_to_threshold": float(self.stats.get("recoverability_signed_margin_to_threshold", gate_diagnostics.get("score_gap", 0.0)) or 0.0),
            "near_threshold": bool(self.stats.get("recoverability_near_threshold", False)),
            "threshold_audit_band": float(self.stats.get("recoverability_near_threshold_band", self._rgd_threshold_audit_band) or 0.0),
            "threshold_provenance": dict(self.stats.get("recoverability_threshold_provenance", self._rgd_threshold_provenance) or {}),
            "policy": str(gate_diagnostics.get("active_gate_policy", "recoverability_closed_object") or "recoverability_closed_object"),
            "public_signal": dict(public_coordinates),
            "need_score": float(gate_diagnostics.get("need_score", 0.0) or 0.0),
            "post_latency_opportunity": float(gate_diagnostics.get("post_latency_opportunity", 0.0) or 0.0),
            "opportunity_floor": float(gate_diagnostics.get("opportunity_floor", 0.0) or 0.0),
            "opportunity_eligible": bool(gate_diagnostics.get("opportunity_eligible", False)),
            "alternative_viable_count": int(gate_diagnostics.get("alternative_viable_count", 0) or 0),
            "alternative_viable_ratio": float(gate_diagnostics.get("alternative_viable_ratio", 0.0) or 0.0),
            "relative_support_weighted_maneuver_family_breadth": float(gate_diagnostics.get("relative_support_weighted_maneuver_family_breadth", gate_diagnostics.get("alternative_viable_ratio", 0.0)) or 0.0),
            "alternative_support_count": int(gate_diagnostics.get("alternative_viable_count", 0) or 0),
            "alternative_support_ratio": float(gate_diagnostics.get("alternative_viable_ratio", 0.0) or 0.0),
            "method_version": str(gate_diagnostics.get("method_version", "identifiable_gate_v12") or "identifiable_gate_v12"),
            "gate_composition": "explicit_serial_floors",
            "gate_action_universe": list(gate_diagnostics.get("gate_action_universe", []) or []),
            "fast_executor_action_universe": list(gate_diagnostics.get("fast_executor_action_universe", []) or []),
            "gate_action_universe_source": str(gate_diagnostics.get("gate_action_universe_source", "unknown") or "unknown"),
            "fast_executor_action_universe_source": str(gate_diagnostics.get("fast_executor_action_universe_source", "unknown") or "unknown"),
            "gate_domain_valid": bool(gate_diagnostics.get("gate_domain_valid", False)),
            "gate_fail_closed": bool(gate_diagnostics.get("gate_fail_closed", True)),
            "gate_fail_closed_reason": str(gate_diagnostics.get("gate_fail_closed_reason", "unknown") or "unknown"),
            "gate_fail_closed_reasons": list(gate_diagnostics.get("gate_fail_closed_reasons", []) or []),
            "raw_cost_complete": bool(gate_diagnostics.get("raw_cost_complete", False)),
            "missing_raw_cost_actions": list(gate_diagnostics.get("missing_raw_cost_actions", []) or []),
            "nonfinite_raw_cost_actions": list(gate_diagnostics.get("nonfinite_raw_cost_actions", []) or []),
            "alternative_maneuver_family_count": int(gate_diagnostics.get("alternative_maneuver_family_count", 0) or 0),
            "alternative_maneuver_family_total": int(gate_diagnostics.get("alternative_maneuver_family_total", 0) or 0),
            "action_maneuver_family_mapping": {
                str(action): str(family)
                for action, family in dict(gate_diagnostics.get("action_maneuver_family_mapping", {}) or {}).items()
            },
            "raw_feasible_alternative_actions": list(gate_diagnostics.get("raw_feasible_alternative_actions", []) or []),
            "raw_feasible_alternative_families": list(gate_diagnostics.get("raw_feasible_alternative_families", []) or []),
            "support_diagnostic_count": int(gate_diagnostics.get("support_diagnostic_count", 0) or 0),
            "support_diagnostic_effective_mass": float(gate_diagnostics.get("support_diagnostic_effective_mass", 0.0) or 0.0),
            "support_diagnostic_complete": bool(gate_diagnostics.get("support_diagnostic_complete", False)),
            "support_cost_complete": bool(gate_diagnostics.get("support_cost_complete", False)),
            "missing_support_cost_actions": list(gate_diagnostics.get("missing_support_cost_actions", []) or []),
            "nonfinite_support_cost_actions": list(gate_diagnostics.get("nonfinite_support_cost_actions", []) or []),
            "support_family_min_costs": {
                str(family): float(cost)
                for family, cost in dict(gate_diagnostics.get("support_family_min_costs", {}) or {}).items()
            },
            "support_best_family_cost": float(gate_diagnostics.get("support_best_family_cost", 0.0) or 0.0),
            "support_weighted_family_mass": float(gate_diagnostics.get("support_weighted_family_mass", 0.0) or 0.0),
            "support_breadth_formula": str(gate_diagnostics.get("support_breadth_formula", "unknown") or "unknown"),
            "support_breadth_temperature": float(gate_diagnostics.get("support_breadth_temperature", 0.0) or 0.0),
            "support_breadth_temperature_source": str(gate_diagnostics.get("support_breadth_temperature_source", "unknown") or "unknown"),
            "absolute_alternative_count": int(gate_diagnostics.get("absolute_alternative_count", 0) or 0),
            "absolute_alternative_ratio": float(gate_diagnostics.get("absolute_alternative_ratio", 0.0) or 0.0),
            "absolute_alternative_feasible": bool(gate_diagnostics.get("absolute_alternative_feasible", False)),
            "alternative_metric_source": str(gate_diagnostics.get("alternative_metric_source", "relative_support_weighted_maneuver_family_breadth") or "relative_support_weighted_maneuver_family_breadth"),
            "headroom_metric_source": str(gate_diagnostics.get("headroom_metric_source", "incumbent_relative_action_recovery_cost_margin") or "incumbent_relative_action_recovery_cost_margin"),
            "viable_cost_threshold": float(gate_diagnostics.get("viable_cost_threshold", 0.55) or 0.55),
            "cost_headroom": float(gate_diagnostics.get("cost_headroom", 0.0) or 0.0),
            "relative_corrective_headroom": float(gate_diagnostics.get("relative_corrective_headroom", 0.0) or 0.0),
            "corrective_headroom_kappa": float(gate_diagnostics.get("corrective_headroom_kappa", 0.0) or 0.0),
            "corrective_headroom_kappa_source": str(gate_diagnostics.get("corrective_headroom_kappa_source", "unknown") or "unknown"),
            "corrective_advantage_raw": float(gate_diagnostics.get("corrective_advantage_raw", 0.0) or 0.0),
            "absolute_recovery_depth": float(gate_diagnostics.get("absolute_recovery_depth", 0.0) or 0.0),
            "hold_action": int(resolved_hold_action),
            "corrective_gap": float(gate_diagnostics.get("corrective_gap", 0.0) or 0.0),
            "action_cost_entropy": float(gate_diagnostics.get("action_cost_entropy", 0.0) or 0.0),
            "need_state_hazard": float(gate_diagnostics.get("need_state_hazard", 0.0) or 0.0),
            "need_pre_screen_hazard": float(gate_diagnostics.get("need_pre_screen_hazard", 0.0) or 0.0),
            "need_metric_source": str(gate_diagnostics.get("need_metric_source", "unknown") or "unknown"),
            "latency_survival_floor": float(gate_diagnostics.get("latency_survival_floor", 0.0) or 0.0),
            "maneuver_breadth_floor": float(gate_diagnostics.get("maneuver_breadth_floor", 0.0) or 0.0),
            "corrective_headroom_floor": float(gate_diagnostics.get("corrective_headroom_floor", 0.0) or 0.0),
            "state_need_floor": float(gate_diagnostics.get("state_need_floor", 0.0) or 0.0),
            "latency_survival_floor_source": str(gate_diagnostics.get("latency_survival_floor_source", "unknown") or "unknown"),
            "maneuver_breadth_floor_source": str(gate_diagnostics.get("maneuver_breadth_floor_source", "unknown") or "unknown"),
            "corrective_headroom_floor_source": str(gate_diagnostics.get("corrective_headroom_floor_source", "unknown") or "unknown"),
            "state_need_floor_source": str(gate_diagnostics.get("state_need_floor_source", "unknown") or "unknown"),
            "latency_survival_pass": bool(gate_diagnostics.get("latency_survival_pass", False)),
            "maneuver_breadth_pass": bool(gate_diagnostics.get("maneuver_breadth_pass", False)),
            "corrective_headroom_pass": bool(gate_diagnostics.get("corrective_headroom_pass", False)),
            "state_need_pass": bool(gate_diagnostics.get("state_need_pass", False)),
            "domain_contract_pass": bool(gate_diagnostics.get("domain_contract_pass", False)),
            "executor_available_pass": bool(gate_diagnostics.get("executor_available_pass", False)),
            "latency_prediction_pass": bool(gate_diagnostics.get("latency_prediction_pass", False)),
            "absolute_feasibility_pass": bool(gate_diagnostics.get("absolute_feasibility_pass", False)),
            "serial_gate_pass": bool(gate_diagnostics.get("serial_gate_pass", False)),
            "serial_gate_failed_components": list(gate_diagnostics.get("serial_gate_failed_components", []) or []),
            "absolute_alternative_feasibility_non_ablatable": bool(gate_diagnostics.get("absolute_alternative_feasibility_non_ablatable", True)),
            "effective_delay_steps": int(gate_diagnostics.get("effective_delay_steps", 0) or 0),
            "latency_prediction_available": bool(gate_diagnostics.get("latency_prediction_available", False)),
            "policy_frequency": float(gate_diagnostics.get("policy_frequency", 0.0) or 0.0),
            "safety_reserve_seconds": float(gate_diagnostics.get("safety_reserve_seconds", 0.0) or 0.0),
            "llm_backed_execution_available": bool(gate_diagnostics.get("llm_backed_execution_available", False)),
            "latency_source": str(gate_diagnostics.get("latency_source", "unknown") or "unknown"),
            "collapse_risk": float(gate_diagnostics.get("collapse_risk", 0.0) or 0.0),
            "value_of_computation": float(gate_diagnostics.get("value_of_computation", 0.0) or 0.0),
            "pre_screen": {
                "score": float(gate_diagnostics.get("pre_screen_score", 0.0) or 0.0),
                "trigger": bool(gate_diagnostics.get("pre_screen_trigger", False)),
                "reason": str(gate_diagnostics.get("pre_screen_reason", "none") or "none"),
            },
            "latency": {
                "predicted_slow_seconds": float(gate_diagnostics.get("predicted_slow_latency_seconds", 0.0) or 0.0),
                "budget_seconds": float(gate_diagnostics.get("latency_budget_seconds", 0.0) or 0.0),
                "pressure": float(gate_diagnostics.get("reasoning_latency_pressure", 0.0) or 0.0),
                "recovery_window": float(gate_diagnostics.get("recovery_window", 0.0) or 0.0),
                "critical_latency_seconds": float(gate_diagnostics.get("critical_latency_seconds", 0.0) or 0.0),
                "effective_delay_steps": int(gate_diagnostics.get("effective_delay_steps", 0) or 0),
                "policy_frequency": float(gate_diagnostics.get("policy_frequency", 0.0) or 0.0),
                "latency_prediction_available": bool(gate_diagnostics.get("latency_prediction_available", False)),
                "llm_backed_execution_available": bool(gate_diagnostics.get("llm_backed_execution_available", False)),
                "safety_reserve_seconds": float(gate_diagnostics.get("safety_reserve_seconds", 0.0) or 0.0),
                "source": str(gate_diagnostics.get("latency_source", "unknown") or "unknown"),
            },
            "selected_system": str(selected_system),
            "route_reason": str(route_reason),
            "routing_decision": (routing_decision.to_dict() if isinstance(routing_decision, RecoverabilityRoutingDecision) else None),
            "baseline_mode": str(self._paper_baseline.trigger_mode),
            "execution_route_score": float(route_signals.get("execution_route_score", 0.0) or 0.0),
        }
        self.stats["recoverability_gate"] = gate_snapshot

    def _annotate_threshold_audit(self, routing_decision: Optional[RecoverabilityRoutingDecision]) -> None:
        if isinstance(routing_decision, RecoverabilityRoutingDecision):
            margin = float(routing_decision.score_gap)
            threshold = float(routing_decision.decision_threshold)
        else:
            threshold = float(self._rgd_decision_threshold if self._rgd_decision_threshold is not None else 0.50)
            margin = float(float(self.stats.get("recoverability_score", 0.0) or 0.0) - threshold)
        near_threshold = bool(abs(margin) <= self._rgd_threshold_audit_band)
        provenance = dict(self._rgd_threshold_provenance)
        provenance["effective_threshold"] = threshold
        provenance["near_threshold_band"] = float(self._rgd_threshold_audit_band)
        self.stats["recoverability_threshold_provenance"] = provenance
        self.stats["recoverability_threshold_source"] = str(provenance.get("source", "unknown") or "unknown")
        self.stats["recoverability_near_threshold"] = near_threshold
        self.stats["recoverability_near_threshold_band"] = float(self._rgd_threshold_audit_band)
        self.stats["recoverability_signed_margin_to_threshold"] = float(margin)

    def export_runtime_contract(self) -> Dict[str, Any]:
        """Return the canonical runtime contract used by this orchestrator."""
        return self._runtime_contract.to_dict()

    def record_external_slow_request(self) -> None:
        """Account for a replayed slow request without invoking the backend.

        Factorial replay deliberately replaces the remote response after the
        proposal-blind Fast/RGD state update.  Keeping attempt and cooldown
        accounting in the orchestrator makes the replayed policy state identical
        to an online request while leaving the backend call out of the loop.
        """
        if self._slow_call_budget is not None and self._slow_call_attempts >= self._slow_call_budget:
            raise RuntimeError("slow-call budget exhausted during replay")
        if self._slow_call_cooldown_remaining > 0:
            raise RuntimeError("slow-call cooldown active during replay")
        self._slow_call_attempts += 1
        self._slow_call_cooldown_remaining = self._slow_call_cooldown_frames + 1
        self.stats["slow_call_attempts"] = int(self._slow_call_attempts)
        self.stats["slow_call_budget"] = self._slow_call_budget
        self.stats["slow_call_budget_remaining"] = (
            None
            if self._slow_call_budget is None
            else max(0, self._slow_call_budget - self._slow_call_attempts)
        )

    def snapshot_policy_state(self) -> Dict[str, Any]:
        """Return the allowlisted temporal state that can alter later actions."""
        return {
            "schema": RGD_POLICY_STATE_SCHEMA,
            "decision_count": int(self.stats.get("decision_count", 0) or 0),
            "support_progress_cooldown": int(self._support_progress_cooldown),
            "rgd_cruise_progress_cooldown": int(
                self._rgd_cruise_progress_cooldown
            ),
            "rgd_cruise_recovery_frames": int(self._rgd_cruise_recovery_frames),
            "slow_call_attempts": int(self._slow_call_attempts),
            "slow_call_cooldown_remaining": int(
                self._slow_call_cooldown_remaining
            ),
            "rad": self._rad_controller.snapshot_policy_state(),
        }

    def restore_policy_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore a validated temporal policy state without touching config."""
        normalized = validate_rgd_policy_state(
            snapshot,
            slow_call_budget=self._slow_call_budget,
            slow_call_cooldown_frames=self._slow_call_cooldown_frames,
        )
        self.stats["decision_count"] = normalized["decision_count"]
        self._support_progress_cooldown = normalized[
            "support_progress_cooldown"
        ]
        self._rgd_cruise_progress_cooldown = normalized[
            "rgd_cruise_progress_cooldown"
        ]
        self._rgd_cruise_recovery_frames = normalized[
            "rgd_cruise_recovery_frames"
        ]
        self._slow_call_attempts = normalized["slow_call_attempts"]
        self._slow_call_cooldown_remaining = normalized[
            "slow_call_cooldown_remaining"
        ]
        self._rad_controller.restore_policy_state(normalized["rad"])

    def decide(
        self,
        state: DrivingState,
        force_system: Optional[str] = None,
    ) -> RGDDecision:
        """Make one routing and action decision."""
        start_time = time.perf_counter()
        self._current_fast_incumbent_identity = {}
        self._current_fast_override_context = {}

        # Protocol controls must alter only the selected executor.  Compute the
        # same natural RGD route state first so an always-fast baseline does not
        # skip state updates later consumed by the shared fast policy.
        protocol_force = self._forced_route_system if force_system is None else None
        if protocol_force in {"fast", "slow"} and self._rgd_signal_provider:
            self._forced_route_system = None
            try:
                natural_system, route_score, route_ambiguity_profile = self._route_system(state, None)
            finally:
                self._forced_route_system = protocol_force
            system = protocol_force
            self.stats["natural_route_system_before_protocol_force"] = natural_system
            self.stats["route_reason"] = "forced_protocol_after_matched_route_state"
        else:
            system, route_score, route_ambiguity_profile = self._route_system(state, force_system)
        if self._slow_call_cooldown_remaining > 0:
            self._slow_call_cooldown_remaining -= 1
        if system == "slow" and self._slow_call_budget is not None:
            budget_exhausted = self._slow_call_attempts >= self._slow_call_budget
            cooldown_active = self._slow_call_cooldown_remaining > 0
            if budget_exhausted or cooldown_active:
                system = "fast"
                reason = "budget_exhausted" if budget_exhausted else "cooldown_active"
                self.stats["route_reason"] = f"slow_suppressed_{reason}"
                self.stats["slow_call_suppressed"] = int(self.stats.get("slow_call_suppressed", 0) or 0) + 1
            else:
                self._slow_call_attempts += 1
                self._slow_call_cooldown_remaining = self._slow_call_cooldown_frames + 1
        self.stats["slow_call_attempts"] = int(self._slow_call_attempts)
        self.stats["slow_call_budget"] = self._slow_call_budget
        self.stats["slow_call_budget_remaining"] = (
            None if self._slow_call_budget is None else max(0, self._slow_call_budget - self._slow_call_attempts)
        )
        logger.info(
            "[Router] Routing -> %s (route_score=%.3f, rad=%.3f)",
            system,
            route_score,
            float(self._last_rad_meta.get("score", 0.0) or 0.0),
        )

        if system == "fast":
            decision = self._execute_fast(
                state,
                route_score,
                fast_override_context=(
                    self._current_fast_override_context
                    if self._current_fast_incumbent_identity
                    else None
                ),
            )
            if self._current_fast_incumbent_identity:
                self._assert_current_fast_incumbent_match(
                    state=state,
                    observed_decision=decision,
                    observed_source="actual_fast_execution",
                )
        else:
            try:
                decision = self._execute_slow_path(
                    state=state,
                    route_score=route_score,
                    route_ambiguity_profile=route_ambiguity_profile,
                    recoverability_context=dict(self._last_recoverability_context),
                )
            except SlowPathUnavailableError as exc:
                decision = self._execute_slow_unavailable_recovery(
                    state=state,
                    route_score=route_score,
                    route_ambiguity_profile=route_ambiguity_profile,
                    failure_reason=str(getattr(exc, "failure_reason", exc)),
                )

        if decision.ambiguity_profile is not None:
            export_route_ambiguity_to_decision(decision)

        if self._last_route_feature_vec is not None:
            decision.stats["route_feature_vec"] = self._last_route_feature_vec.tolist()

        bridge_orchestrator_stats_into_decision(self.stats, decision)

        decision.latency_ms = (time.perf_counter() - start_time) * 1000
        self.stats["total_latency_ms"] += decision.latency_ms
        self.stats["decision_count"] += 1
        return decision

    def _execute_slow_unavailable_recovery(
        self,
        state: DrivingState,
        route_score: float,
        route_ambiguity_profile: Optional[RouteAmbiguityProfile],
        failure_reason: str,
    ) -> RGDDecision:
        """Fall back to the identical fast policy after a failed slow request.

        A slow-executor outage must not activate a second, unpublished junction
        controller: doing so confounds routing with a hidden policy change.  The
        normal safety layer remains downstream of this decision, exactly as it
        does for the always-fast control.
        """
        decision = self._execute_fast(
            state,
            route_score,
            fast_override_context=(
                self._current_fast_override_context
                if self._current_fast_incumbent_identity
                else None
            ),
        )
        if self._current_fast_incumbent_identity:
            self._assert_current_fast_incumbent_match(
                state=state,
                observed_decision=decision,
                observed_source="actual_fast_execution",
            )
        self.stats["slow_unavailable_recoveries"] = int(self.stats.get("slow_unavailable_recoveries", 0) or 0) + 1
        decision.reasoning = f"{decision.reasoning} | slow unavailable -> identical fast fallback"
        decision.system_used = "fast_after_slow_failure"
        decision.ambiguity_profile = route_ambiguity_profile
        decision.stats.update({
            "depth": 0,
            "slow_reasoning_mode": "fast_fallback",
            "slow_reasoning_success": False,
            "slow_reasoning_failure_reason": str(failure_reason),
            "llm_backed_execution_available": False,
            "slow_unavailable_recovery": True,
            "slow_failure_fallback_policy": "identical_fast_policy",
            "post_risk_calibration_action": int(decision.action),
        })
        return decision
    def _route_system(
        self,
        state: DrivingState,
        force_system: Optional[str],
    ) -> Tuple[str, float, Optional[RouteAmbiguityProfile]]:
        """Choose fast or slow according to RAD plus support-layer evidence."""
        if force_system:
            self.stats["route_reason"] = "forced_runtime"
            return force_system, 1.0 if force_system == "slow" else 0.0, None
        if self._forced_route_system in {"fast", "slow"} and not self._rgd_signal_provider:
            self.stats["forced_route_system"] = self._forced_route_system
            self.stats["route_reason"] = "forced_protocol_no_rgd_signal"
            self.stats["rgd_signal_provider"] = False
            self._last_recoverability_context = {
                "recoverability_assessment": {},
                "gate_diagnostics": {},
                "route_ambiguity_profile": {},
                "routing_decision": None,
                "latency_context": {},
                "failure_pre_screen": {},
                "provisional_action": 1,
            }
            return self._forced_route_system, 1.0 if self._forced_route_system == "slow" else 0.0, None
        if self._forced_route_system in {"fast", "slow"}:
            self.stats["forced_route_system"] = self._forced_route_system
            self.stats["route_reason"] = "forced_protocol"
            route_signals = self._collect_route_signals(state)
            route_ambiguity_profile = route_signals["route_ambiguity_profile"]
            self._last_recoverability_context = dict(
                route_signals.get("recoverability_context", {}) or {}
            )
            self._annotate_threshold_audit(route_signals.get("routing_decision"))
            self._export_recoverability_gate_snapshot(route_signals, self._forced_route_system, self.stats["route_reason"])
            return self._forced_route_system, 1.0 if self._forced_route_system == "slow" else 0.0, route_ambiguity_profile  # Apply experiment-level force route after collecting the same RGD evidence bundle so baselines stay controlled without getting a hidden latency shortcut.
        self.stats["route_story_mode"] = self._route_story_mode
        route_signals = self._collect_route_signals(state)
        route_ambiguity_profile = route_signals["route_ambiguity_profile"]
        baseline_route = None
        if self._paper_baseline.trigger_mode != "none" and route_ambiguity_profile is not None:
            baseline_route = resolve_paper_baseline_trigger(
                self.config,
                self.stats,
                self._paper_baseline,
                route_ambiguity_profile,
                route_signals["rad_meta"],
                route_signals["fused_route_score"],
            )  # Explicit paper baselines may replace the default RGD route while preserving the same recoverability evidence bundle for later comparison.
        if baseline_route is not None:
            system, execution_route_score, _ = baseline_route
            self._last_recoverability_context = dict(
                route_signals.get("recoverability_context", {}) or {}
            )
            self._annotate_threshold_audit(route_signals.get("routing_decision"))
            self._export_recoverability_gate_snapshot(route_signals, system, self.stats.get("route_reason", "baseline_trigger"))  # Preserve the same recoverability evidence object even when routing is delegated to a paper baseline.
            logger.debug("[Route-Baseline] mode=%s exec=%.3f fused=%.3f -> %s", self._paper_baseline.trigger_mode, execution_route_score, route_signals["fused_route_score"], system)
            return system, execution_route_score, route_ambiguity_profile

        routing_decision: RecoverabilityRoutingDecision = route_signals["routing_decision"]
        self._last_recoverability_context = dict(
            route_signals.get("recoverability_context", {}) or {}
        )
        system = str(routing_decision.selected_system)
        route_reason = str(routing_decision.route_reason)
        observed_frames = int(self.stats.get("decision_count", 0) or 0) + 1
        temporal_evidence_valid = observed_frames >= self._rgd_min_observation_frames
        self.stats["rgd_observed_frames"] = observed_frames
        self.stats["rgd_min_observation_frames"] = int(self._rgd_min_observation_frames)
        self.stats["rgd_temporal_evidence_valid"] = bool(temporal_evidence_valid)
        if system == "slow" and not temporal_evidence_valid:
            system = "fast"
            route_reason = "recoverability_temporal_evidence_warmup"
            self.stats["rgd_temporal_warmup_suppressions"] = int(
                self.stats.get("rgd_temporal_warmup_suppressions", 0) or 0
            ) + 1
        # Junction conflict evidence is already part of need_score.  It cannot
        # override a failed post-latency opportunity test; imminent conflicts
        # must remain on the immediate fast/safety path.
        self.stats["route_ambiguity_entropy"] = float(route_ambiguity_profile.ambiguity_entropy) if route_ambiguity_profile is not None else 0.0  # Keep ambiguity diagnostics explicit whenever auxiliary routing analysis is intentionally enabled.
        self.stats["route_ambiguity_gap"] = float(route_ambiguity_profile.ambiguity_gap) if route_ambiguity_profile is not None else 0.0  # Gap remains diagnostic-only outside the absolute paper-facing surface.
        self.stats["route_ambiguity_disagreement"] = float(route_ambiguity_profile.evidence_disagreement) if route_ambiguity_profile is not None else 0.0  # Disagreement remains diagnostic-only outside the absolute paper-facing surface.
        self.stats["rgd_execution_route_score"] = float(max(0.0, min(1.0, routing_decision.route_score)))  # Need ranks frames only after the explicit serial floors establish eligibility.
        self.stats["unified_route_score"] = float(route_signals["fused_route_score"])  # Preserve the fused score as a diagnostic metric rather than the primary execution authority.
        self.stats["route_reason"] = route_reason
        self._annotate_threshold_audit(routing_decision)
        self._export_recoverability_gate_snapshot(route_signals, system, self.stats["route_reason"])  # Publish one compact gate object so downstream surfaces foreground recoverability rather than auxiliary machinery.
        logger.debug(
            "[Route-Vote] story=%s recoverability=%.3f threshold=%.3f fused=%.3f -> %s",
            self._route_story_mode,
            routing_decision.route_score,
            routing_decision.decision_threshold,
            route_signals["fused_route_score"],
            system,
        )
        return system, float(max(0.0, min(1.0, routing_decision.route_score))), route_ambiguity_profile

    def _execute_fast(
        self,
        state: DrivingState,
        route_score: float,
        fast_override_context: Optional[Dict[str, Any]] = None,
    ) -> RGDDecision:
        """Execute the fast policy directly once the RGD router keeps the frame on fast."""
        fast_decision = self.fast.think(state)
        lane_hold_meta = self._highway_lane_change_hold_override(state, fast_decision.action)
        lane_hold_action = int(lane_hold_meta.get("rgd_highway_lane_hold_resolved_action", fast_decision.action))
        pass_meta = self._highway_fast_pass_override(state, lane_hold_action)
        pass_action = int(pass_meta.get("rgd_highway_pass_resolved_action", lane_hold_action))
        cruise_meta = self._rgd_cruise_progress_override(
            state,
            pass_action,
            fast_override_context=fast_override_context,
        )
        cruise_action = int(cruise_meta.get("rgd_cruise_progress_resolved_action", pass_action))
        support_meta = self._support_progress_override(
            state,
            cruise_action,
            fast_override_context=fast_override_context,
        )
        selected_action = int(support_meta.get("support_progress_resolved_action", cruise_action))
        effective_universe = {int(action) for action in state.get_available_actions()}
        if selected_action not in effective_universe:
            raise RuntimeError(
                "fast override emitted an action outside the effective universe; "
                f"action={selected_action}; universe={sorted(effective_universe)}"
            )
        selected_reasoning = str(fast_decision.reasoning)
        if bool(lane_hold_meta.get("rgd_highway_lane_hold_applied", False)):
            selected_reasoning = f"{selected_reasoning} | rgd_highway_lane_hold: {ActionType.to_english(lane_hold_action)}"
        if bool(pass_meta.get("rgd_highway_pass_applied", False)):
            selected_reasoning = f"{selected_reasoning} | rgd_highway_pass: {ActionType.to_english(pass_action)}"
        if bool(cruise_meta.get("rgd_cruise_progress_applied", False)):
            selected_reasoning = f"{selected_reasoning} | rgd_cruise_progress: {ActionType.to_english(cruise_action)}"
        if bool(support_meta.get("support_progress_control_applied", False)):
            selected_reasoning = f"{selected_reasoning} | support_progress_restart: {ActionType.to_english(selected_action)}"
        self.stats["fast_decisions"] += 1
        rad_score = float(self._last_rad_meta.get("score", self.stats.get("rad_irreversibility", route_score)) or 0.0)
        calibration_context = dict(getattr(fast_decision, "calibration_context", {}) or {})
        return RGDDecision(
            action=selected_action,
            reasoning=selected_reasoning,
            confidence=float(fast_decision.confidence),
            system_used="fast",
            route_label=self._route_story_mode,
            route_score=float(route_score),
            ambiguity_profile=self._last_route_ambiguity_profile,
            stats={
                "depth": 0,
                "rule_name": getattr(fast_decision, "rule_name", None),
                "fast_smoothness_override": bool(getattr(fast_decision, "smoothness_override", False)),  # Per-frame smoothness control flag for downstream metrics.
                "fast_rad_bias": False,  # Fast execution currently does not post-adjust the rule action via a separate RAD bias stage, so the audit flag must stay explicit and stable.
                "fast_rad_score": rad_score,  # Export the recoverability signal that kept the frame on fast so route audits can still compare fast decisions against the governing RGD score.
                "fast_decision_mode": str(getattr(fast_decision, "decision_mode", "hard_rule_shell") or "hard_rule_shell"),
                "fast_abstention_applied": bool(getattr(fast_decision, "abstention_applied", False)),
                "fast_midlayer_score_gap": float(getattr(fast_decision, "top_score_gap", 0.0) or 0.0),
                "fast_rule_calibration_profile": str(calibration_context.get("scenario_profile", "generic") or "generic"),
                "fast_rule_distance_buffer_m": float(calibration_context.get("distance_buffer_m", 0.0) or 0.0),
                "fast_rule_ttc_buffer_s": float(calibration_context.get("ttc_buffer_s", 0.0) or 0.0),
                "fast_rule_cross_traffic_buffer_m": float(calibration_context.get("cross_traffic_buffer_m", 0.0) or 0.0),
                "fast_rule_caution_level": float(calibration_context.get("caution_level", 0.0) or 0.0),
                "fast_abstention_band": float(calibration_context.get("abstention_band", 0.0) or 0.0),
                "fast_shell_minimal": bool(str(getattr(fast_decision, "decision_mode", "hard_rule_shell") or "hard_rule_shell") == "hard_rule_shell"),
                "fast_midlayer_best_action": int(calibration_context.get("midlayer_best_action", fast_decision.action) or fast_decision.action),
                "fast_midlayer_runner_up_action": int(calibration_context.get("midlayer_runner_up_action", fast_decision.action) or fast_decision.action),
                "fast_midlayer_best_score": float(calibration_context.get("best_score", 0.0) or 0.0),
                "fast_midlayer_runner_up_score": float(calibration_context.get("runner_up_score", 0.0) or 0.0),
                "fast_midlayer_shared_context": dict(calibration_context.get("shared_context", {}) or {}),
                "fast_midlayer_selected_parts": dict(calibration_context.get("selected_action_parts", {}) or {}),
                **lane_hold_meta,
                **pass_meta,
                **cruise_meta,
                **support_meta,
            },
        )

    def _resolve_fast_executor_action_universe(
        self,
        state: DrivingState,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        """Resolve and install the single effective per-frame action domain."""
        state.clear_effective_action_universe()
        base_actions = {int(action) for action in state.get_available_actions()}
        controlled_actions = {
            int(action) for action in self._highway_fast_pass_available_actions(state)
        }
        if hasattr(self.fast, "peek"):
            base_fast_action = int(self.fast.peek(state).action)
        else:
            base_fast_action = int(ActionType.IDLE)
            if base_fast_action not in base_actions:
                base_fast_action = int(sorted(base_actions)[0])

        hold_meta = self._highway_lane_change_hold_override(state, base_fast_action)
        hold_action = int(
            hold_meta.get("rgd_highway_lane_hold_resolved_action", base_fast_action)
        )
        hold_actions = (
            {hold_action}
            if bool(hold_meta.get("rgd_highway_lane_hold_applied", False))
            else set()
        )
        pass_meta = self._highway_fast_pass_override(state, hold_action)
        pass_actions = (
            {
                int(action)
                for action in list(pass_meta.get("rgd_highway_pass_candidate_actions", []) or [])
            }
            if bool(pass_meta.get("rgd_highway_pass_applied", False))
            else set()
        )

        resolved = tuple(sorted(base_actions | hold_actions | pass_actions))
        if not resolved:
            raise ValueError("fast executor action universe is empty")
        state.set_effective_action_universe(
            list(resolved),
            source="rgd_identifiable_gate_v12",
        )
        provenance = {
            "base_actions": sorted(base_actions),
            "controlled_gap_candidates": sorted(controlled_actions - base_actions),
            "highway_pass_actions": sorted(pass_actions),
            "highway_hold_actions": sorted(hold_actions),
            "base_fast_proposal_action": int(base_fast_action),
            "hold_condition_reason": str(hold_meta.get("rgd_highway_lane_hold_reason", "unknown") or "unknown"),
            "pass_condition_reason": str(pass_meta.get("rgd_highway_pass_reason", "unknown") or "unknown"),
            "includes_highway_pass_override": bool(pass_actions - base_actions),
            "includes_highway_hold_override": bool(hold_actions - base_actions),
            "effective_action_universe": list(resolved),
            "effective_action_universe_source": str(state.effective_action_universe_source),
        }
        return resolved, provenance

    def _highway_lane_change_hold_override(self, state: DrivingState, action: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "rgd_highway_lane_hold_enabled": True,
            "rgd_highway_lane_hold_applied": False,
            "rgd_highway_lane_hold_resolved_action": int(action),
        }
        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        if scenario != "highway":
            meta["rgd_highway_lane_hold_reason"] = "non_highway"
            return meta
        history = [int(item) for item in list(getattr(self.fast, "action_history", []) or [])]
        if not history or history[-1] not in {int(ActionType.LANE_LEFT), int(ActionType.LANE_RIGHT)}:
            meta["rgd_highway_lane_hold_reason"] = "no_recent_lane_change"
            return meta
        if int(action) == int(history[-1]):
            meta["rgd_highway_lane_hold_reason"] = "already_holding_lane_change"
            return meta
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        if speed > 24.0:
            meta["rgd_highway_lane_hold_reason"] = "speed_too_high"
            return meta
        if abs(float_or_default(getattr(state, "front_heading_delta", None), 0.0)) > math.radians(18.0):
            meta["rgd_highway_lane_hold_reason"] = "front_heading_unstable"
            return meta
        previous_lane_action = int(history[-1])
        available = self._highway_fast_pass_available_actions(state)
        if previous_lane_action not in available:
            meta["rgd_highway_lane_hold_reason"] = "lane_action_unavailable"
            return meta
        target_side = "left" if previous_lane_action == int(ActionType.LANE_LEFT) else "right"
        target_front = float_or_default(getattr(state, f"{target_side}_front_distance", None), float("inf"))
        target_rear = float_or_default(getattr(state, f"{target_side}_rear_distance", None), float("inf"))
        if math.isfinite(target_front) and target_front < max(10.0, speed * 0.50):
            meta["rgd_highway_lane_hold_reason"] = "target_front_tight"
            return meta
        if math.isfinite(target_rear) and target_rear < 0.8:
            meta["rgd_highway_lane_hold_reason"] = "target_rear_overlap"
            return meta
        if int(action) in {int(ActionType.SLOWER), int(ActionType.IDLE), int(ActionType.FASTER)}:
            meta["rgd_highway_lane_hold_applied"] = True
            meta["rgd_highway_lane_hold_reason"] = "complete_lane_change_commitment"
            meta["rgd_highway_lane_hold_resolved_action"] = previous_lane_action
            return meta
        meta["rgd_highway_lane_hold_reason"] = "base_action_not_longitudinal"
        return meta

    def _highway_fast_pass_override(self, state: DrivingState, action: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "rgd_highway_pass_enabled": True,
            "rgd_highway_pass_applied": False,
            "rgd_highway_pass_resolved_action": int(action),
        }
        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        if scenario != "highway":
            meta["rgd_highway_pass_reason"] = "non_highway"
            return meta
        action_id = int(action)
        if action_id not in {int(ActionType.IDLE), int(ActionType.SLOWER), int(ActionType.FASTER)}:
            meta["rgd_highway_pass_reason"] = "base_action_not_pass_candidate"
            return meta
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        front = float_or_default(getattr(state, "front_distance", None), float("inf"))
        front_speed = float_or_default(getattr(state, "front_speed", None), speed)
        ttc = float_or_default(getattr(state, "ttc", None), float("inf"))
        thw = float_or_default(getattr(state, "thw", None), float("inf"))
        meta["rgd_highway_pass_action_id"] = int(action_id)
        meta["rgd_highway_pass_probe_speed"] = float(speed)
        meta["rgd_highway_pass_probe_front"] = float(front) if math.isfinite(front) else float("inf")
        meta["rgd_highway_pass_probe_front_speed"] = float(front_speed)
        meta["rgd_highway_pass_probe_ttc"] = float(ttc) if math.isfinite(ttc) else float("inf")
        meta["rgd_highway_pass_probe_thw"] = float(thw) if math.isfinite(thw) else float("inf")
        if action_id == int(ActionType.FASTER) and not (
            math.isfinite(front)
            and front <= max(24.0, speed * 1.15)
            and speed - front_speed >= 2.0
            and (not math.isfinite(ttc) or ttc >= 3.0)
            and (not math.isfinite(thw) or thw >= 0.72)
        ):
            meta["rgd_highway_pass_reason"] = "base_action_progress_candidate"
            return meta
        if action_id == int(ActionType.IDLE) and not (
            speed - front_speed >= 2.0
            or (
                speed >= 18.0
                and math.isfinite(front)
                and front <= max(18.0, speed * 0.76)
                and (not math.isfinite(ttc) or ttc >= 3.0)
                and (not math.isfinite(thw) or thw >= 0.58)
            )
        ):
            meta["rgd_highway_pass_reason"] = "no_slow_lead_pressure"
            return meta
        if not (
            11.0 <= speed <= 25.2
            and math.isfinite(front)
            and (6.0 if action_id == int(ActionType.SLOWER) else 8.0) <= front <= max(30.0, speed * 1.45)
            and (speed - front_speed >= 1.2 or action_id == int(ActionType.IDLE))
            and (not math.isfinite(ttc) or ttc >= (2.8 if action_id != int(ActionType.SLOWER) else 3.5))
            and (not math.isfinite(thw) or thw >= (0.62 if action_id != int(ActionType.SLOWER) else 0.38))
        ):
            meta["rgd_highway_pass_reason"] = "no_slow_lead_pressure"
            return meta
        available = self._highway_fast_pass_available_actions(state)
        meta["rgd_highway_pass_available"] = sorted(int(item) for item in available)
        candidates: List[Tuple[float, int]] = []
        for candidate, front_name, rear_name, speed_name, rear_speed_name, lane_bias in (
            (int(ActionType.LANE_LEFT), "left_front_distance", "left_rear_distance", "left_front_speed", "left_rear_speed", 0.2),
            (int(ActionType.LANE_RIGHT), "right_front_distance", "right_rear_distance", "right_front_speed", "right_rear_speed", 0.0),
        ):
            if candidate not in available:
                continue
            target_front = float_or_default(getattr(state, front_name, None), float("inf"))
            target_rear = float_or_default(getattr(state, rear_name, None), float("inf"))
            target_speed = float_or_default(getattr(state, speed_name, None), speed)
            target_rear_speed_raw = getattr(state, rear_speed_name, None)
            target_rear_speed = float_or_default(target_rear_speed_raw, speed)
            if math.isfinite(target_front) and target_front < max(13.5, speed * 0.70):
                continue
            controlled_rear_cutin = bool(
                math.isfinite(target_rear)
                and max(2.8, speed * 0.14) <= target_rear < max(13.0, speed * 0.60)
                and target_rear_speed <= speed + (0.4 if target_rear < 5.0 else 0.0)
                and (not math.isfinite(target_front) or target_front >= max(26.0, speed * 1.35))
            )
            if math.isfinite(target_rear) and target_rear < max(8.0, speed * 0.38) and not controlled_rear_cutin:
                continue
            target_rel = target_speed - speed
            if target_rel < -4.0:
                continue
            front_gain = (80.0 if not math.isfinite(target_front) else target_front) - front
            if front_gain < 6.0 and target_rel < 0.8 and not controlled_rear_cutin:
                continue
            score = front_gain + max(0.0, target_rel) * 2.5 + lane_bias
            candidates.append((float(score), int(candidate)))
        if not candidates:
            meta["rgd_highway_pass_reason"] = "no_safe_pass_gap"
            return meta
        meta["rgd_highway_pass_candidate_actions"] = sorted({
            int(candidate) for _, candidate in candidates
        })
        selected = int(max(candidates, key=lambda item: (item[0], -item[1]))[1])
        meta["rgd_highway_pass_applied"] = True
        meta["rgd_highway_pass_reason"] = "slow_lead_overtake_gap"
        meta["rgd_highway_pass_resolved_action"] = selected
        meta["rgd_highway_pass_speed"] = float(speed)
        meta["rgd_highway_pass_front"] = float(front)
        return meta

    def _highway_fast_pass_available_actions(self, state: DrivingState) -> set:
        available = {int(item) for item in state.get_available_actions()}
        if list(getattr(state, "effective_action_universe", []) or []):
            return available
        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        if scenario != "highway":
            return available
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        if not (10.0 <= speed <= 23.0):
            return available
        for candidate, front_name, rear_name, rear_speed_name in (
            (int(ActionType.LANE_LEFT), "left_front_distance", "left_rear_distance", "left_rear_speed"),
            (int(ActionType.LANE_RIGHT), "right_front_distance", "right_rear_distance", "right_rear_speed"),
        ):
            if candidate in available:
                continue
            simulator_legal = {
                int(action) for action in list(getattr(state, "legal_actions", []) or [])
            }
            if simulator_legal and candidate not in simulator_legal:
                continue
            target_front = float_or_default(getattr(state, front_name, None), float("inf"))
            target_rear = float_or_default(getattr(state, rear_name, None), float("inf"))
            rear_speed_raw = getattr(state, rear_speed_name, None)
            if rear_speed_raw is None or not math.isfinite(target_rear):
                continue
            rear_speed = max(0.0, float_or_default(rear_speed_raw, speed))
            controlled_rear_gap = bool(
                7.0 <= target_rear < max(12.0, speed * 0.58)
                and rear_speed <= speed - (2.8 if target_rear < 7.0 else 1.4)
                and (not math.isfinite(target_front) or target_front >= max(24.0, speed * 1.15))
            )
            if controlled_rear_gap:
                available.add(candidate)
        return available

    def _rgd_cruise_progress_override(
        self,
        state: DrivingState,
        action: int,
        fast_override_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "rgd_cruise_progress_enabled": True,
            "rgd_cruise_progress_applied": False,
            "rgd_cruise_progress_resolved_action": int(action),
        }
        closed_gate = (
            self._recoverability_gate.enable_corridor_gate
            and self._recoverability_gate.enable_budget_gate
            and self._recoverability_gate.enable_margin_gate
            and self._recoverability_gate.enable_heuristic_gate
        )
        if not closed_gate:
            meta["rgd_cruise_progress_reason"] = "closed_recoverability_gate_disabled"
            self._decay_rgd_cruise_progress_cooldown()
            return meta
        if int(action) != int(ActionType.IDLE):
            meta["rgd_cruise_progress_reason"] = "base_action_not_idle"
            self._decay_rgd_cruise_progress_cooldown()
            return meta
        available = self._highway_fast_pass_available_actions(state)
        if int(ActionType.FASTER) not in available:
            meta["rgd_cruise_progress_reason"] = "faster_unavailable"
            self._decay_rgd_cruise_progress_cooldown()
            return meta
        if self._rgd_cruise_progress_cooldown > 0:
            meta["rgd_cruise_progress_reason"] = "cruise_progress_cooldown"
            meta["rgd_cruise_progress_cooldown"] = int(self._rgd_cruise_progress_cooldown)
            self._decay_rgd_cruise_progress_cooldown()
            return meta

        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        if scenario not in {"highway", "merge"}:
            meta["rgd_cruise_progress_reason"] = "non_cruise_scenario"
            self._decay_rgd_cruise_progress_cooldown()
            return meta
        if not self._rgd_cruise_progress_is_safe(state):
            meta["rgd_cruise_progress_reason"] = "progress_risk_not_clear"
            self._decay_rgd_cruise_progress_cooldown()
            return meta

        corridor_action = self._rgd_corridor_progress_action(state, available)
        if corridor_action is not None:
            meta["rgd_cruise_progress_applied"] = True
            meta["rgd_cruise_progress_reason"] = "closed_loop_corridor_progress"
            meta["rgd_cruise_progress_resolved_action"] = int(corridor_action)
            meta["rgd_cruise_progress_speed"] = float(max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0)))
            self._rgd_cruise_progress_cooldown = 2
            return meta

        context = (
            fast_override_context
            if fast_override_context is not None
            else self._last_recoverability_context
        )
        collapse = float(
            self._fast_override_context_payload(context)["collapse_risk"]
        )
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        front = float_or_default(getattr(state, "front_distance", None), float("inf"))
        target_speed = 31.0 if scenario == "highway" else 23.0
        if self._support_memory_retriever is not None:
            target_speed += 3.0
        progress_deficit = max(0.0, min(1.0, (target_speed - speed) / max(target_speed, 1e-6)))
        if progress_deficit <= 0.02:
            meta["rgd_cruise_progress_reason"] = "target_speed_reached"
            self._decay_rgd_cruise_progress_cooldown()
            return meta
        recovery_commitment = self._rgd_cruise_recovery_commitment(state, target_speed)
        if recovery_commitment:
            meta["rgd_cruise_progress_applied"] = True
            meta["rgd_cruise_progress_reason"] = "closed_loop_safe_speed_recovery"
            meta["rgd_cruise_progress_resolved_action"] = int(ActionType.FASTER)
            meta["rgd_cruise_progress_speed"] = float(speed)
            meta["rgd_cruise_progress_target_speed"] = float(target_speed)
            meta["rgd_cruise_recovery_frames"] = int(self._rgd_cruise_recovery_frames)
            self._rgd_cruise_progress_cooldown = 0
            return meta
        if collapse >= 0.48:
            meta["rgd_cruise_progress_reason"] = "recoverability_pressure_too_high"
            meta["rgd_cruise_progress_collapse_risk"] = float(collapse)
            self._decay_rgd_cruise_progress_cooldown()
            return meta

        meta["rgd_cruise_progress_applied"] = True
        meta["rgd_cruise_progress_reason"] = "closed_loop_low_risk_progress"
        meta["rgd_cruise_progress_resolved_action"] = int(ActionType.FASTER)
        meta["rgd_cruise_progress_collapse_risk"] = float(collapse)
        meta["rgd_cruise_progress_speed"] = float(speed)
        meta["rgd_cruise_progress_target_speed"] = float(target_speed)
        meta["rgd_cruise_progress_front_distance"] = float(front) if math.isfinite(front) else float("inf")
        self._rgd_cruise_progress_cooldown = 1 if self._support_memory_retriever is None else 0
        return meta

    def _rgd_corridor_progress_action(self, state: DrivingState, available: set) -> Optional[int]:
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        front = float_or_default(getattr(state, "front_distance", None), float("inf"))
        ttc = float_or_default(getattr(state, "ttc", None), float("inf"))
        thw = float_or_default(getattr(state, "thw", None), float("inf"))
        support_enabled = self._support_memory_retriever is not None
        if speed < (12.0 if support_enabled else 14.0):
            return None
        closest = float_or_default(getattr(state, "closest_vehicle_distance", None), float("inf"))
        closest_long = float_or_default(getattr(state, "closest_vehicle_longitudinal", None), float("inf"))
        closest_lat = abs(float_or_default(getattr(state, "closest_vehicle_lateral", None), float("inf")))
        if math.isfinite(closest) and closest < max(11.0, speed * 0.62) and -5.0 <= closest_long <= 12.0 and closest_lat <= 11.0:
            return None
        corridor_pressure = bool(
            math.isfinite(front)
            and (
                front < max(30.0 if support_enabled else 32.0, speed * (1.70 if support_enabled else 1.85))
                or (math.isfinite(thw) and thw < (1.65 if support_enabled else 1.55))
                or (math.isfinite(ttc) and ttc < (5.6 if support_enabled else 5.2))
            )
        )
        if speed > (30.0 if support_enabled else 28.0) or not corridor_pressure:
            return None
        candidates: List[Tuple[float, int]] = []
        pass_candidates: List[Tuple[float, int]] = []
        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        front_speed = float_or_default(getattr(state, "front_speed", None), speed)
        front_rel = front_speed - speed
        pass_pressure = bool(
            scenario == "highway"
            and math.isfinite(front)
            and front <= max(46.0 if support_enabled else 44.0, speed * (2.10 if support_enabled else 2.0))
            and front_rel <= (-2.0 if support_enabled else -2.5)
            and (not math.isfinite(ttc) or ttc >= (4.7 if support_enabled else 5.0))
            and (not math.isfinite(thw) or thw >= (1.45 if support_enabled else 1.55))
        )
        current_lane = int(float_or_default(getattr(state, "ego_lane", None), 0.0))
        for action, front_name, rear_name, speed_name, lane_delta in (
            (int(ActionType.LANE_LEFT), "left_front_distance", "left_rear_distance", "left_front_speed", -1),
            (int(ActionType.LANE_RIGHT), "right_front_distance", "right_rear_distance", "right_front_speed", 1),
        ):
            if int(action) not in available:
                continue
            target_front = float_or_default(getattr(state, front_name, None), float("inf"))
            target_rear = float_or_default(getattr(state, rear_name, None), float("inf"))
            target_speed_raw = getattr(state, speed_name, None)
            target_rel = float_or_default(target_speed_raw, speed) - speed if target_speed_raw is not None else 0.0
            adaptive_pass = bool(
                speed <= (22.0 if support_enabled else 21.0)
                and target_rel >= (0.6 if support_enabled else 1.0)
                and (not math.isfinite(target_front) or target_front >= max(10.5 if support_enabled else 12.0, speed * (0.65 if support_enabled else 0.75)))
            )
            if math.isfinite(target_front) and target_front < max(24.0 if support_enabled else 26.0, speed * (1.05 if support_enabled else 1.15)) and not adaptive_pass:
                continue
            if math.isfinite(target_rear) and target_rear < max(16.0 if support_enabled else 18.0, speed * (0.72 if support_enabled else 0.78)):
                continue
            if target_rel < -7.0:
                continue
            front_gain = 80.0 if not math.isfinite(target_front) else target_front - (front if math.isfinite(front) else 0.0)
            if pass_pressure:
                pass_front_ok = bool(
                    not math.isfinite(target_front)
                    or target_front >= max(13.5 if support_enabled else 15.0, speed * (0.70 if support_enabled else 0.78))
                )
                pass_rear_ok = bool(
                    not math.isfinite(target_rear)
                    or target_rear >= max(13.0 if support_enabled else 15.0, speed * (0.58 if support_enabled else 0.64))
                )
                if pass_front_ok and pass_rear_ok and target_rel >= -5.0:
                    pass_score = front_gain + max(0.0, target_rel + 2.0) * (2.5 if support_enabled else 2.0) - 0.08 * max(0, current_lane + lane_delta)
                    pass_candidates.append((float(pass_score), int(action)))
            if front_gain < (6.0 if support_enabled else 8.0) and not adaptive_pass:
                continue
            lane_bias = -0.05 * max(0, current_lane + lane_delta)
            pass_bonus = max(0.0, target_rel) * (4.0 if support_enabled else 3.0) if adaptive_pass else 0.0
            candidates.append((float(front_gain + pass_bonus + lane_bias), int(action)))
        if pass_candidates:
            return int(max(pass_candidates, key=lambda item: (item[0], -item[1]))[1])
        if not candidates:
            return None
        return int(max(candidates, key=lambda item: (item[0], -item[1]))[1])

    def _rgd_cruise_recovery_commitment(self, state: DrivingState, target_speed: float) -> bool:
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        support_enabled = self._support_memory_retriever is not None
        if speed >= min(float(target_speed) - 1.0, 24.0 if support_enabled else 23.6):
            self._rgd_cruise_recovery_frames = 0
            return False
        front = float_or_default(getattr(state, "front_distance", None), float("inf"))
        ttc = float_or_default(getattr(state, "ttc", None), float("inf"))
        thw = float_or_default(getattr(state, "thw", None), float("inf"))
        clear_gap = bool(
            (not math.isfinite(front) or front >= max(21.5 if support_enabled else 22.0, speed * (1.10 if support_enabled else 1.16)))
            and (not math.isfinite(ttc) or ttc >= (4.2 if support_enabled else 4.35))
            and (not math.isfinite(thw) or thw >= (1.05 if support_enabled else 1.08))
        )
        if not clear_gap:
            self._rgd_cruise_recovery_frames = 0
            return False
        if self._rgd_cruise_recovery_frames <= 0:
            self._rgd_cruise_recovery_frames = 6 if support_enabled else 4
        self._rgd_cruise_recovery_frames = max(0, int(self._rgd_cruise_recovery_frames) - 1)
        return True

    def _rgd_cruise_progress_is_safe(self, state: DrivingState) -> bool:
        recent = list(getattr(state, "history_frames", []) or [])[-4:]
        if any(int(frame.get("action", -1) or -1) == int(ActionType.SLOWER) for frame in recent):
            speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
            front = float_or_default(getattr(state, "front_distance", None), float("inf"))
            ttc = float_or_default(getattr(state, "ttc", None), float("inf"))
            thw = float_or_default(getattr(state, "thw", None), float("inf"))
            support_enabled = self._support_memory_retriever is not None
            recent_slower_count = sum(1 for frame in recent if int(frame.get("action", -1) or -1) == int(ActionType.SLOWER))
            closest = float_or_default(getattr(state, "closest_vehicle_distance", None), float("inf"))
            closest_long = float_or_default(getattr(state, "closest_vehicle_longitudinal", None), float("inf"))
            closest_lat = abs(float_or_default(getattr(state, "closest_vehicle_lateral", None), float("inf")))
            closest_closing = float_or_default(getattr(state, "closest_vehicle_closing_speed", None), float("-inf"))
            side_pressure = bool(
                math.isfinite(closest)
                and math.isfinite(closest_long)
                and math.isfinite(closest_lat)
                and math.isfinite(closest_closing)
                and 1.25 < closest_lat <= 5.8
                and -1.5 <= closest_long <= max(15.0, speed * 0.72)
                and closest_closing >= max(2.5, speed * 0.10)
            )
            if (
                recent_slower_count <= 1
                and speed < (18.5 if support_enabled else 18.0)
                and not side_pressure
                and (not math.isfinite(front) or front >= max(16.0, speed * 0.88))
                and (not math.isfinite(ttc) or ttc >= 3.2)
                and (not math.isfinite(thw) or thw >= 0.82)
            ):
                return True
            recovered_gap = bool(
                speed < (20.5 if support_enabled else 19.5)
                and (not math.isfinite(front) or front >= max(22.0 if support_enabled else 23.0, speed * (1.10 if support_enabled else 1.20)))
                and (not math.isfinite(ttc) or ttc >= (4.4 if support_enabled else 4.8))
                and (not math.isfinite(thw) or thw >= (1.10 if support_enabled else 1.20))
            )
            if recovered_gap:
                return True
            if speed >= 20.5 or (math.isfinite(front) and front < 20.0) or (math.isfinite(ttc) and ttc < 4.5):
                self._rgd_cruise_progress_cooldown = max(int(self._rgd_cruise_progress_cooldown), 3)
                return False
            return False
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        front = float_or_default(getattr(state, "front_distance", None), float("inf"))
        ttc = float_or_default(getattr(state, "ttc", None), float("inf"))
        thw = float_or_default(getattr(state, "thw", None), float("inf"))
        rel_speed = float_or_default(getattr(state, "front_relative_speed", None), 0.0)
        closest = float_or_default(getattr(state, "closest_vehicle_distance", None), float("inf"))
        closest_long = float_or_default(getattr(state, "closest_vehicle_longitudinal", None), float("inf"))
        closest_lat = abs(float_or_default(getattr(state, "closest_vehicle_lateral", None), float("inf")))
        closest_closing = float_or_default(getattr(state, "closest_vehicle_closing_speed", None), float("-inf"))
        side_cutin_pressure = bool(
            math.isfinite(closest)
            and math.isfinite(closest_long)
            and math.isfinite(closest_lat)
            and math.isfinite(closest_closing)
            and 1.25 < closest_lat <= 5.8
            and -1.5 <= closest_long <= max(15.0, speed * 0.72)
            and closest_closing >= max(2.5, speed * 0.10)
        )
        if side_cutin_pressure:
            return False
        stopping_room = max(14.0, speed * 0.85)
        if math.isfinite(front) and front < stopping_room:
            return False
        if math.isfinite(ttc) and ttc < 4.5:
            return False
        if math.isfinite(thw) and thw < 0.75:
            return False
        if rel_speed < -8.0 and math.isfinite(front) and front < max(18.0, speed):
            return False
        return True

    def _decay_rgd_cruise_progress_cooldown(self) -> None:
        self._rgd_cruise_progress_cooldown = max(0, int(self._rgd_cruise_progress_cooldown) - 1)

    def _support_progress_override(
        self,
        state: DrivingState,
        action: int,
        fast_override_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._support_memory_retriever is None:
            return {
                "support_progress_control_enabled": False,
                "support_progress_control_applied": False,
                "support_memory_retrieval_count": 0,
            }
        meta: Dict[str, Any] = {
            "support_progress_control_enabled": True,
            "support_progress_control_applied": False,
            "support_memory_retrieval_count": 0,
            "support_progress_resolved_action": int(action),
        }
        if int(action) != int(ActionType.IDLE):
            meta["support_progress_control_reason"] = "base_action_not_idle"
            self._decay_support_progress_cooldown()
            return meta
        if self._support_progress_cooldown > 0:
            meta["support_progress_control_reason"] = "support_progress_cooldown"
            meta["support_progress_control_cooldown"] = int(self._support_progress_cooldown)
            self._decay_support_progress_cooldown()
            return meta
        pressure = self._support_control_pressure(fast_override_context)
        gap = assess_junction_gap(state, int(ActionType.FASTER))
        junction_clear_gap = bool(gap.get("should_clear", False))
        if self._structured_progress_gate_blocks_support(state, gap, junction_clear_gap):
            meta["support_progress_control_reason"] = "structured_junction_requires_clearance_pressure"
            meta["support_progress_control_pressure"] = float(pressure)
            meta["support_progress_junction_clear_gap"] = bool(junction_clear_gap)
            self._decay_support_progress_cooldown()
            return meta
        if pressure >= 0.58 and not junction_clear_gap:
            meta["support_progress_control_reason"] = "pressure_too_high"
            meta["support_progress_control_pressure"] = float(pressure)
            return meta
        available = [int(item) for item in state.get_available_actions()]
        if int(ActionType.FASTER) not in set(available):
            meta["support_progress_control_reason"] = "faster_unavailable"
            meta["support_progress_control_pressure"] = float(pressure)
            return meta
        if not self._support_progress_is_safe(state):
            meta["support_progress_control_reason"] = "progress_risk_not_clear"
            meta["support_progress_control_pressure"] = float(pressure)
            meta["support_progress_junction_clear_gap"] = bool(junction_clear_gap)
            self._decay_support_progress_cooldown()
            return meta
        examples = self._support_memory_retriever.retrieve(state, available)
        meta["support_memory_retrieval_count"] = 1
        meta["support_progress_memory_ids"] = [int(example.memory_id) for example in examples]
        meta["support_progress_memory_actions"] = [int(example.action) for example in examples]
        meta["support_progress_memory_similarities"] = [float(example.similarity) for example in examples]
        faster_vote = sum(
            max(0.25, float(example.similarity))
            for example in examples
            if int(example.action) == int(ActionType.FASTER) and bool(example.action_available)
        )
        total_vote = sum(max(0.25, float(example.similarity)) for example in examples if bool(example.action_available))
        vote_share = float(faster_vote / max(total_vote, 1e-6))
        meta["support_progress_faster_vote_share"] = float(vote_share)
        meta["support_progress_control_pressure"] = float(pressure)
        meta["support_progress_junction_clear_gap"] = bool(junction_clear_gap)
        if (faster_vote > 0.0 and vote_share >= 0.50) or junction_clear_gap:
            meta["support_progress_control_applied"] = True
            meta["support_progress_control_reason"] = (
                "support_junction_gap_clearance" if junction_clear_gap else "memory_supported_safe_progress"
            )
            meta["support_progress_resolved_action"] = int(ActionType.FASTER)
            self._support_progress_cooldown = max(int(self._support_progress_cooldown), 3)
        else:
            meta["support_progress_control_reason"] = "memory_did_not_support_progress"
            self._decay_support_progress_cooldown()
        return meta

    def _structured_progress_gate_blocks_support(
        self,
        state: DrivingState,
        gap: Dict[str, Any],
        junction_clear_gap: bool,
    ) -> bool:
        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        if scenario not in {"intersection", "roundabout"}:
            return False
        if bool(junction_clear_gap):
            return False
        if bool(gap.get("should_yield", False)) or bool(gap.get("front_blocked", False)):
            return True
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        recent_wait = int(gap.get("recent_low_speed_wait", 0) or 0)
        waiting_to_clear = bool(
            speed <= 1.2
            and recent_wait >= 2
            and (
                bool(gap.get("clearance_pressure", False))
                or bool(gap.get("in_junction", False))
                or math.isfinite(float_or_default(gap.get("min_ego_distance", None), float("inf")))
            )
        )
        return not waiting_to_clear

    def _decay_support_progress_cooldown(self) -> None:
        self._support_progress_cooldown = max(0, int(self._support_progress_cooldown) - 1)

    def _support_control_pressure(
        self,
        fast_override_context: Optional[Dict[str, Any]] = None,
    ) -> float:
        context = (
            fast_override_context
            if fast_override_context is not None
            else self._last_recoverability_context
        )
        payload = self._fast_override_context_payload(context)
        return float(max(payload.values()))

    def _support_progress_is_safe(self, state: DrivingState) -> bool:
        recent = list(getattr(state, "history_frames", []) or [])[-4:]
        if any(int(frame.get("action", -1) or -1) == int(ActionType.SLOWER) for frame in recent):
            self._support_progress_cooldown = max(int(self._support_progress_cooldown), 4)
            return False
        scenario = str(getattr(state, "scenario_type", "") or "").split("-")[0].strip().lower()
        speed = max(0.0, float_or_default(getattr(state, "ego_speed", None), 0.0))
        front = float_or_default(getattr(state, "front_distance", None), float("inf"))
        ttc = float_or_default(getattr(state, "ttc", None), float("inf"))
        if ttc <= 0.0:
            ttc = float("inf")
        thw = float_or_default(getattr(state, "thw", None), float("inf"))
        if thw <= 0.0:
            thw = float("inf")
        cross = float_or_default(getattr(state, "cross_traffic_distance", None), float("inf"))
        if cross <= 0.0:
            cross = float("inf")
        gap = assess_junction_gap(state, int(ActionType.FASTER))
        in_exit_zone = bool(gap.get("in_exit_zone", False))
        if bool(gap.get("should_clear", False)) and bool(gap.get("in_junction", False)) and not in_exit_zone:
            conflict_risk = float(gap.get("conflict_risk", 0.0) or 0.0)
            if conflict_risk >= 0.90:
                return bool(speed <= 1.2 and (not math.isfinite(front) or front >= 8.0))
            return bool(
                speed <= 4.0
                and (not math.isfinite(front) or front >= 5.0)
                and (not math.isfinite(ttc) or ttc >= 2.0)
            )
        if in_exit_zone:
            if speed > 10.0:
                return False
            if scenario == "roundabout" and math.isfinite(cross) and cross < 20.0:
                return False
        if bool(gap.get("should_clear", False)) and speed <= 0.5:
            recent_wait = int(gap.get("recent_low_speed_wait", 0) or 0)
            if recent_wait >= 3 and (not math.isfinite(cross) or cross >= 8.0):
                return True
        if bool(gap.get("front_blocked", False)):
            return False
        if bool(gap.get("should_yield", False)) and not bool(gap.get("should_clear", False)):
            return False
        if scenario == "highway" and speed > 12.0:
            return False
        should_clear_junction = bool(gap.get("should_clear", False)) and bool(gap.get("in_junction", False)) and not in_exit_zone
        if not should_clear_junction:
            if in_exit_zone:
                exit_cross_threshold = 6.0 if scenario == "intersection" else 10.0
                if math.isfinite(cross) and cross < exit_cross_threshold:
                    return False
            else:
                if math.isfinite(cross) and cross < 16.0:
                    return False
            if math.isfinite(ttc) and ttc < 8.0:
                return False
            if math.isfinite(thw) and thw < (2.4 if scenario in {"highway", "merge"} else 2.0):
                return False
        front_multiplier = 3.2 if scenario in {"highway", "merge"} else 2.2
        if math.isfinite(front) and front < max(36.0, speed * front_multiplier):
            return False
        return True

    def _compute_unified_route_score(
        self,
        lookahead_risk: Optional[float],
        irreversibility_score: float,
    ) -> float:
        """Fuse diagnostic routing evidence without hand-authored weights.

        The execution route is decided by the closed recoverability object.
        This fused score is diagnostic-only, so it uses a threshold-free
        probabilistic union over available risk signals rather than a manually
        weighted mixture.
        """
        irreversibility = float(max(0.0, min(1.0, irreversibility_score)))
        components = [
            float(max(0.0, min(1.0, irreversibility))),
        ]
        if lookahead_risk is not None:
            components.append(float(max(0.0, min(1.0, lookahead_risk))))
        residual_mass = 1.0
        for component in components:
            residual_mass *= (1.0 - component)
        score = float(max(0.0, min(1.0, 1.0 - residual_mass)))
        self._last_route_feature_vec = np.array(
            [
                float(max(0.0, min(1.0, irreversibility))),  # RAD irreversibility signal.
                1.0,  # Base dimension for learnable routing.
            ],
            dtype=np.float64,
        )
        self.stats["route_components"] = {
            "irreversibility": float(max(0.0, min(1.0, irreversibility))),
            "lookahead": float(max(0.0, min(1.0, lookahead_risk))) if lookahead_risk is not None else 0.0,
        }
        self.stats["route_irreversibility_score"] = float(max(0.0, min(1.0, irreversibility)))
        return float(max(0.0, min(1.0, score)))

    def _build_route_ambiguity_profile(
        self,
        conflict_score: float,
        route_score: float,
    ) -> RouteAmbiguityProfile:
        """Build a structured routing ambiguity profile from the current heuristic landscape.

        This is intentionally a scaffold rather than the final learned update rule:
        action probabilities are derived from the exported RAD action-cost landscape.
        """
        raw_action_costs = dict(self._last_rad_meta.get("action_recovery_costs", {}) or {})
        support_action_costs = dict(self._last_rad_meta.get("action_support_ranking_costs", {}) or {})
        valid_raw_costs = {
            int(action): float(cost)
            for action, cost in raw_action_costs.items()
            if np.isfinite(float(cost))
        }
        valid_support_costs = {
            int(action): float(cost)
            for action, cost in support_action_costs.items()
            if np.isfinite(float(cost))
        }
        support_complete = bool(
            valid_raw_costs
            and set(valid_raw_costs).issubset(valid_support_costs)
        )
        probability_costs = valid_support_costs if support_complete else valid_raw_costs
        probability_cost_source = (
            "action_support_ranking_costs"
            if support_complete
            else "action_recovery_costs_fallback"
        )
        ordered_items = sorted(
            probability_costs.items(),
            key=lambda item: item[1],
        )
        min_cost = float(ordered_items[0][1]) if ordered_items else 0.0
        temperature = max(
            1e-6,
            float_or_default(self._last_rad_meta.get("support_breadth_temperature"), 0.10),
        )
        logits = np.array([-(cost - min_cost) / temperature for _, cost in ordered_items], dtype=np.float64)  # Convert recovery costs into relative recoverability evidence before softmax normalization.
        logits -= float(np.max(logits)) if logits.size else 0.0  # Stabilize softmax numerically so large negative costs do not underflow.
        probs = np.exp(logits)
        probs /= float(np.sum(probs)) if float(np.sum(probs)) > 0.0 else 1.0  # Normalize into a valid action distribution over the currently evaluated actions.
        action_probabilities = {action_id: float(prob) for (action_id, _), prob in zip(ordered_items, probs)}

        top_probs = sorted(action_probabilities.values(), reverse=True)
        ambiguity_gap = float(top_probs[0] - top_probs[1]) if len(top_probs) > 1 else float(top_probs[0] if top_probs else 1.0)  # The top-2 gap is the simplest ambiguity signal for deciding whether falsification or deeper reasoning should activate.
        ambiguity_entropy = 0.0
        if top_probs:
            entropy_raw = float(-sum(prob * np.log(max(prob, 1e-8)) for prob in action_probabilities.values()))  # Shannon entropy quantifies routing ambiguity directly from the action distribution rather than proxying with many separate heuristics.
            ambiguity_entropy = float(entropy_raw / np.log(max(2, len(action_probabilities))))  # Normalize entropy to [0,1] so later thresholds remain interpretable across action subsets.
        hypothesis_beliefs_raw: Dict[str, float] = {}
        hypothesis_total = float(sum(max(0.0, value) for value in hypothesis_beliefs_raw.values()))
        hypothesis_beliefs = {name: float(max(0.0, value) / hypothesis_total) for name, value in hypothesis_beliefs_raw.items()} if hypothesis_total > 0.0 else {}

        source_priors = {
            "rule": float(max(0.0, min(1.0, conflict_score))),
            "conflict": float(max(0.0, min(1.0, conflict_score))),
            "rad": float(max(0.0, min(1.0, self._last_rad_meta.get("recovery_cost_target", route_score) or route_score))),
            "safe": float(max(0.0, min(1.0, self._last_rad_meta.get("best_recovery_cost", route_score) or route_score))),
        }
        ambiguity_best_action = int(max(action_probabilities.items(), key=lambda item: item[1])[0]) if action_probabilities else _action_id_or_default(self._last_rad_meta.get("best_action", 1))
        evidence_disagreement = float(source_priors.get("conflict", 0.0))
        intervention_risk = float(max(0.0, min(1.0, route_score)))  # Route score is still the best proxy for imminent intervention pressure until intervention-calibrated updates are implemented.
        return RouteAmbiguityProfile(
            action_probabilities=action_probabilities,
            action_recovery_costs=valid_raw_costs,
            action_support_ranking_costs=valid_support_costs,
            probability_cost_source=probability_cost_source,
            method_version=str(self._last_rad_meta.get("method_version", "identifiable_gate_v12") or "identifiable_gate_v12"),
            gate_action_universe=list(self._last_rad_meta.get("gate_action_universe", []) or []),
            fast_executor_action_universe=list(self._last_rad_meta.get("fast_executor_action_universe", []) or []),
            action_recovery_cost_parts={
                int(action): dict(parts)
                for action, parts in dict(self._last_rad_meta.get("action_recovery_cost_parts", {}) or {}).items()
            },
            raw_cost_source=str(self._last_rad_meta.get("raw_cost_source", "unknown") or "unknown"),
            raw_cost_complete=bool(self._last_rad_meta.get("raw_cost_complete", False)),
            missing_raw_cost_actions=list(self._last_rad_meta.get("missing_raw_cost_actions", []) or []),
            nonfinite_raw_cost_actions=list(self._last_rad_meta.get("nonfinite_raw_cost_actions", []) or []),
            ambiguity_best_action=ambiguity_best_action,
            selected_probability=float(action_probabilities.get(ambiguity_best_action, 0.0)),
            ambiguity_entropy=ambiguity_entropy,
            ambiguity_gap=ambiguity_gap,
            evidence_disagreement=evidence_disagreement,
            intervention_risk=intervention_risk,
            source_priors=source_priors,
            hypothesis_beliefs=hypothesis_beliefs,
        )

