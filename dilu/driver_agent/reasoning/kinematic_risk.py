"""Deterministic kinematic-risk executor for the declared executor-swap arm."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.decision import RGDDecision


_TIE_BREAK_ORDER = {
    int(ActionType.SLOWER): 0,
    int(ActionType.IDLE): 1,
    int(ActionType.LANE_LEFT): 2,
    int(ActionType.LANE_RIGHT): 3,
    int(ActionType.FASTER): 4,
}


def _finite_cost(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return resolved if math.isfinite(resolved) else float("inf")


class KinematicRiskActionProvider:
    """Select the legal action with the lowest runtime safety cost.

    The provider consumes the action-cost decomposition already attached to the
    current ``DrivingState`` by the shared safety system. It therefore changes
    the slow executor only; action legality and the downstream safety stack
    remain identical to the RGD arm.
    """

    def __call__(
        self,
        state: DrivingState,
        recoverability_context: Dict[str, Any],
    ) -> RGDDecision:
        del recoverability_context
        available = tuple(int(action) for action in state.get_available_actions())
        decomposition = getattr(state, "_safety_cost_decomposition", None)
        if not isinstance(decomposition, Mapping):
            raise ValueError("kinematic-risk executor requires a safety cost decomposition")
        costs = {int(action): dict(parts or {}) for action, parts in decomposition.items()}
        if set(costs) != set(available):
            raise ValueError("kinematic-risk action costs do not cover the effective action universe")

        def rank(action: int) -> tuple[float, float, int]:
            parts = costs[action]
            return (
                _finite_cost(parts.get("safety")),
                _finite_cost(parts.get("total")),
                _TIE_BREAK_ORDER.get(int(action), int(action)),
            )

        selected = min(available, key=rank)
        risk_scores = {int(action): float(rank(int(action))[0]) for action in available}
        return RGDDecision(
            action=int(selected),
            reasoning="kinematic-risk executor: minimum projected safety cost",
            confidence=1.0,
            system_used="slow",
            route_label="kinematic_risk",
            route_score=0.0,
            stats={
                "slow_reasoning_mode": "kinematic_risk",
                "slow_reasoning_success": True,
                "kinematic_risk_selected_action": int(selected),
            },
            agent_opinions={"risk_scores_by_action": risk_scores},
        )


__all__ = ["KinematicRiskActionProvider"]
