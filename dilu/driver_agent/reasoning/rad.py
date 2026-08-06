"""Action-level recoverability signals used by the RGD route gate."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.policy_state import (
    RAD_POLICY_STATE_SCHEMA,
    validate_rad_policy_state,
)


METHOD_VERSION = "identifiable_gate_v12"
_LATERAL_ACTIONS = {int(ActionType.LANE_LEFT), int(ActionType.LANE_RIGHT)}


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(lower)
    if not math.isfinite(resolved):
        return float(lower)
    return float(min(upper, max(lower, resolved)))


def _finite(value: Any, default: float = math.inf) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    return resolved if math.isfinite(resolved) else float(default)


def _stable_actions(actions: Iterable[Any]) -> Tuple[int, ...]:
    return tuple(sorted({int(action) for action in actions}))


class RADSignalController:
    """Derive an identifiable action-cost surface from the current state.

    Raw recovery costs preserve the action-specific safety surface.  A separate
    support surface is exported for relative maneuver-breadth diagnostics; it
    never repairs a missing raw cost.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.target_lane_projection_enable = bool(
            self.config.get("target_lane_projection_enable", False)
        )
        self.support_breadth_temperature = 0.10
        self._corridor_boundary_ema: Optional[float] = None
        self._corridor_width_ema: Optional[float] = None
        self._last_corridor_stage: Optional[str] = None

    @staticmethod
    def _action_semantic_penalty(action: int) -> float:
        penalties = {
            int(ActionType.IDLE): 0.00,
            int(ActionType.SLOWER): 0.06,
            int(ActionType.FASTER): 0.18,
            int(ActionType.LANE_LEFT): 0.16,
            int(ActionType.LANE_RIGHT): 0.16,
        }
        return float(penalties.get(int(action), 1.0))

    @staticmethod
    def _target_lane_values(state: DrivingState, action: int) -> Dict[str, float]:
        if int(action) == int(ActionType.LANE_LEFT):
            return {
                "available": 1.0 if bool(state.can_change_left) else 0.0,
                "front_distance": _finite(state.left_front_distance),
                "rear_distance": _finite(state.left_rear_distance),
                "front_speed": _finite(state.left_front_speed, state.ego_speed),
                "rear_speed": _finite(state.left_rear_speed, state.ego_speed),
            }
        if int(action) == int(ActionType.LANE_RIGHT):
            return {
                "available": 1.0 if bool(state.can_change_right) else 0.0,
                "front_distance": _finite(state.right_front_distance),
                "rear_distance": _finite(state.right_rear_distance),
                "front_speed": _finite(state.right_front_speed, state.ego_speed),
                "rear_speed": _finite(state.right_rear_speed, state.ego_speed),
            }
        raise ValueError("target-lane values require a lateral action")

    @staticmethod
    def _state_pressures(state: DrivingState, conflict_score: float) -> Dict[str, float]:
        speed = max(0.0, _finite(state.ego_speed, 0.0))
        front = _finite(state.front_distance)
        ttc = _finite(state.ttc)
        thw = _finite(state.thw)
        closing = max(0.0, -_finite(state.front_relative_speed, 0.0))
        if not math.isfinite(ttc) and math.isfinite(front) and closing > 1e-6:
            ttc = front / closing
        ttc_pressure = (
            _clamp((4.0 - ttc) / 4.0) if math.isfinite(ttc) else 0.0
        )
        headway_pressure = (
            _clamp((1.2 - thw) / 1.2) if math.isfinite(thw) else 0.0
        )
        desired_gap = max(10.0, speed * 1.2)
        proximity = (
            _clamp((desired_gap - front) / desired_gap)
            if math.isfinite(front)
            else 0.0
        )
        cross_distance = _finite(state.cross_traffic_distance)
        cross_pressure = (
            _clamp((18.0 - cross_distance) / 18.0)
            if math.isfinite(cross_distance)
            else 0.0
        )
        conflict = _clamp(conflict_score)
        hazard = max(ttc_pressure, headway_pressure, proximity, cross_pressure, conflict)
        return {
            "speed": speed,
            "ttc_pressure": ttc_pressure,
            "headway_pressure": headway_pressure,
            "proximity_complexity": max(proximity, cross_pressure),
            "cross_traffic_pressure": cross_pressure,
            "conflict_pressure": conflict,
            "state_hazard_score": hazard,
        }

    def estimate_action_recovery_cost(
        self,
        state: DrivingState,
        action: int,
        risk_payload: Optional[Mapping[str, Any]] = None,
        *,
        ttc_pressure: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Estimate a bounded raw cost for one legal action.

        The direct estimator is intentionally deterministic.  If target-lane
        projection is disabled, lateral actions retain their commit penalty.
        With projection enabled, a lateral action is evaluated against the
        destination-lane lead and rear vehicles and fails closed when that
        maneuver is unavailable or has an urgent rear conflict.
        """

        action_id = int(action)
        payload = dict(risk_payload or {})
        common_risk = _clamp(
            payload.get(
                "coupled_risk",
                payload.get("rollout_risk", self._state_pressures(state, 0.0)["state_hazard_score"]),
            )
        )
        local_ttc_pressure = _clamp(
            ttc_pressure
            if ttc_pressure is not None
            else self._state_pressures(state, 0.0)["ttc_pressure"]
        )
        parts: Dict[str, float] = {
            "common_risk": common_risk,
            "coupled_risk": common_risk,
            "ttc_pressure": local_ttc_pressure,
            "residual_action_penalty": self._action_semantic_penalty(action_id),
            "raw_cost_formula": "common_risk_plus_residual_action_penalty",
            "residual_action_penalty_source": "rad_action_semantics",
            "lane_commit_penalty": 0.0,
            "dominant_term_name": "common_risk",
        }

        if action_id not in _LATERAL_ACTIONS or not self.target_lane_projection_enable:
            cost = _clamp(common_risk + parts["residual_action_penalty"])
            if parts["residual_action_penalty"] > common_risk:
                parts["dominant_term_name"] = "residual_action_penalty"
            return cost, parts

        target = self._target_lane_values(state, action_id)
        legal = set(_stable_actions(state.get_available_actions()))
        target_available = bool(target["available"]) and action_id in legal
        parts.update(
            {
                "target_lane_projection_applied": 1.0,
                "target_lane_available": 1.0 if target_available else 0.0,
                "target_front_distance": target["front_distance"],
                "target_rear_distance": target["rear_distance"],
            }
        )
        if not target_available:
            parts["lane_commit_penalty"] = 1.0
            parts["dominant_term_name"] = "lane_commit_penalty"
            return 1.0, parts

        target_front = target["front_distance"]
        target_front_speed = target["front_speed"]
        ego_speed = max(0.0, _finite(state.ego_speed, 0.0))
        front_closing = max(0.0, ego_speed - target_front_speed)
        target_front_ttc = (
            target_front / front_closing
            if math.isfinite(target_front) and front_closing > 1e-6
            else math.inf
        )
        target_gap = max(10.0, ego_speed * 1.1)
        target_front_pressure = (
            max(
                _clamp((target_gap - target_front) / target_gap),
                _clamp((3.5 - target_front_ttc) / 3.5)
                if math.isfinite(target_front_ttc)
                else 0.0,
            )
            if math.isfinite(target_front)
            else 0.0
        )
        target_rear = target["rear_distance"]
        rear_closing = max(0.0, target["rear_speed"] - ego_speed)
        target_rear_ttc = (
            target_rear / rear_closing
            if math.isfinite(target_rear) and rear_closing > 1e-6
            else math.inf
        )
        target_rear_urgent = (
            math.isfinite(target_rear_ttc) and target_rear_ttc < 3.0
        )
        target_rear_pressure = (
            _clamp((3.0 - target_rear_ttc) / 3.0)
            if math.isfinite(target_rear_ttc)
            else 0.0
        )
        target_risk = max(target_front_pressure, target_rear_pressure)
        parts.update(
            {
                "target_lane_risk": target_risk,
                "coupled_risk": target_risk,
                "target_front_ttc": target_front_ttc,
                "target_rear_ttc": target_rear_ttc,
                "target_rear_urgent": 1.0 if target_rear_urgent else 0.0,
            }
        )
        if target_rear_urgent:
            parts["lane_commit_penalty"] = 1.0
            parts["dominant_term_name"] = "target_rear_urgent_penalty"
            return 1.0, parts

        lane_penalty = 0.10
        parts["lane_commit_penalty"] = lane_penalty
        parts["residual_action_penalty"] = lane_penalty
        parts["dominant_term_name"] = (
            "target_lane_risk" if target_risk >= lane_penalty else "lane_commit_penalty"
        )
        return _clamp(target_risk + lane_penalty), parts

    def _safety_decomposition_costs(
        self,
        state: DrivingState,
        actions: Tuple[int, ...],
        common_risk: float,
    ) -> Tuple[Dict[int, float], Dict[int, Dict[str, Any]], list[int], list[int]]:
        decomposition = state.__dict__.get("_safety_cost_decomposition")
        if not isinstance(decomposition, Mapping):
            return {}, {}, [], []
        totals: Dict[int, float] = {}
        missing: list[int] = []
        nonfinite: list[int] = []
        for action in actions:
            value = decomposition.get(action)
            if not isinstance(value, Mapping) or "total" not in value:
                missing.append(action)
                continue
            try:
                total = float(value["total"])
            except (TypeError, ValueError):
                nonfinite.append(action)
                continue
            if not math.isfinite(total):
                nonfinite.append(action)
                continue
            totals[action] = total
        if not totals:
            return {}, {}, missing, nonfinite
        domain_min = min(totals.values())
        costs: Dict[int, float] = {}
        parts: Dict[int, Dict[str, Any]] = {}
        for action, total in totals.items():
            residual = max(0.0, total - domain_min)
            costs[action] = _clamp(common_risk + residual)
            parts[action] = {
                "common_risk": common_risk,
                "coupled_risk": common_risk,
                "raw_safety_total": total,
                "domain_min_safety_total": domain_min,
                "residual_action_penalty": residual,
                "raw_cost_formula": "common_risk_plus_residual_action_penalty",
                "residual_action_penalty_source": "safety_cost_decomposition.total_minus_domain_min",
                "dominant_term_name": (
                    "residual_action_penalty" if residual > common_risk else "common_risk"
                ),
            }
        return costs, parts, missing, nonfinite

    def _recoverability_heuristic(
        self,
        actions: Tuple[int, ...],
        costs: Mapping[int, float],
        hazard: float,
    ) -> Dict[str, Any]:
        finite_costs = [float(cost) for cost in costs.values() if math.isfinite(float(cost))]
        viable = [action for action, cost in costs.items() if cost <= 0.55]
        width = len(viable)
        best = min(finite_costs) if finite_costs else 1.0
        worst = max(finite_costs) if finite_costs else 1.0
        boundary = _clamp(best)
        if self._corridor_boundary_ema is None:
            self._corridor_boundary_ema = boundary
        else:
            self._corridor_boundary_ema = 0.75 * self._corridor_boundary_ema + 0.25 * boundary
        if self._corridor_width_ema is None:
            self._corridor_width_ema = float(width)
        else:
            self._corridor_width_ema = 0.75 * self._corridor_width_ema + 0.25 * float(width)
        if hazard >= 0.80:
            stage = "critical"
        elif hazard >= 0.55:
            stage = "decisive"
        elif width <= 1:
            stage = "committed"
        else:
            stage = "open"
        self._last_corridor_stage = stage
        ratio = float(width / len(actions)) if actions else 0.0
        return {
            "stage": stage,
            "corridor_width": width,
            "corridor_width_raw": width,
            "corridor_exists": bool(width > 0),
            "recovery_budget_remaining": _clamp(1.0 - best),
            "dominance_margin": _clamp(worst - best),
            "principle_satisfied": bool(width > 0),
            "reachable_safe_set_size": width,
            "reachable_safe_set_ratio": ratio,
            "viable_headroom": _clamp(1.0 - best),
            "viability_proxy_score": ratio,
            "short_horizon_irreversible_risk": _clamp(hazard),
            "short_horizon_seconds": 0.8,
            "prediction_horizon_s": 0.8,
            "cost_slope": _clamp(worst - best),
            "rollout_curve_stage": stage,
            "corridor_boundary": self._corridor_boundary_ema,
            "corridor_boundary_raw": boundary,
            "corridor_boundary_delta": abs(boundary - self._corridor_boundary_ema),
            "corridor_width_delta": abs(float(width) - self._corridor_width_ema),
            "corridor_stability_score": _clamp(1.0 - abs(boundary - self._corridor_boundary_ema)),
            "near_commitment": bool(width <= 1),
            "near_commitment_clamped": bool(width <= 1 and hazard >= 0.8),
            "near_commitment_contradiction": False,
        }

    def estimate_signal(
        self,
        state: DrivingState,
        conflict_score: float = 0.0,
        *,
        action_universe: Optional[Iterable[Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Return the current irreversibility signal and action-cost evidence."""

        actions = _stable_actions(
            state.get_available_actions() if action_universe is None else action_universe
        )
        if not actions:
            raise ValueError("RAD requires a non-empty action universe")
        pressures = self._state_pressures(state, conflict_score)
        common_risk = pressures["state_hazard_score"]
        injected_costs, parts, missing, nonfinite = self._safety_decomposition_costs(
            state, actions, common_risk
        )
        raw_source = "safety_cost_decomposition" if state.__dict__.get("_safety_cost_decomposition") is not None else "rad_action_semantics"
        if raw_source == "rad_action_semantics":
            costs: Dict[int, float] = {}
            parts = {}
            for action in actions:
                cost, action_parts = self.estimate_action_recovery_cost(
                    state,
                    action,
                    {
                        "coupled_risk": common_risk,
                        "rollout_risk": common_risk,
                        "rollout_utility": 1.0 - common_risk,
                    },
                    ttc_pressure=pressures["ttc_pressure"],
                )
                costs[action] = cost
                parts[action] = action_parts
        else:
            costs = injected_costs
        raw_complete = not missing and not nonfinite and set(costs) == set(actions)
        finite_costs = {action: cost for action, cost in costs.items() if math.isfinite(cost)}
        best_action = min(finite_costs, key=finite_costs.get) if finite_costs else None
        ordered_costs = sorted(finite_costs.values())
        best_cost = ordered_costs[0] if ordered_costs else 1.0
        next_cost = ordered_costs[1] if len(ordered_costs) > 1 else best_cost
        margin = max(0.0, next_cost - best_cost)
        heuristic = self._recoverability_heuristic(actions, finite_costs, common_risk)
        meta: Dict[str, Any] = {
            "method_version": METHOD_VERSION,
            "action_recovery_costs": dict(costs),
            "action_support_ranking_costs": dict(costs),
            "action_recovery_cost_parts": dict(parts),
            "raw_cost_source": raw_source,
            "raw_cost_complete": bool(raw_complete),
            "missing_raw_cost_actions": list(missing),
            "nonfinite_raw_cost_actions": list(nonfinite),
            "best_action": best_action,
            "best_support_action": best_action,
            "best_recovery_cost": best_cost,
            "best_support_ranking_cost": best_cost,
            "recovery_regret": margin,
            "recovery_margin": margin,
            "recovery_objective_margin": margin,
            "recovery_cost_target": _clamp(common_risk),
            "recovery_objective_value": _clamp(common_risk),
            "support_only_ranking_disagreement": False,
            "support_breadth_temperature": self.support_breadth_temperature,
            "support_breadth_temperature_source": "identifiable_gate_v12.fixed_T_A",
            "corridor_entropy": 0.0,
            "reachable_safe_set_size": heuristic["reachable_safe_set_size"],
            "reachable_safe_set_ratio": heuristic["reachable_safe_set_ratio"],
            "viable_headroom": heuristic["viable_headroom"],
            "short_horizon_irreversible_risk": common_risk,
            "prediction_horizon_s": 0.8,
            "recoverability_heuristic": heuristic,
            **pressures,
        }
        return _clamp(common_risk), meta

    def snapshot_policy_state(self) -> Dict[str, Any]:
        return {
            "schema": RAD_POLICY_STATE_SCHEMA,
            "corridor_boundary_ema": self._corridor_boundary_ema,
            "corridor_width_ema": self._corridor_width_ema,
            "last_corridor_stage": self._last_corridor_stage,
        }

    def restore_policy_state(self, snapshot: Dict[str, Any]) -> None:
        normalized = validate_rad_policy_state(snapshot)
        self._corridor_boundary_ema = normalized["corridor_boundary_ema"]
        self._corridor_width_ema = normalized["corridor_width_ema"]
        self._last_corridor_stage = normalized["last_corridor_stage"]
