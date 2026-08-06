"""Minimal deterministic safety adapter for the RGD runtime."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from dilu.utils.shared import float_or_default

from .rss_calculator import FASTER, IDLE, LANE_LEFT, LANE_RIGHT, SLOWER, RSSCalculator


@dataclass
class FrameSafetyResult:
    final_action: int
    proposed_action: int
    state_action: int
    shield_action: int
    safety_override: bool = False
    shield_override: bool = False
    emergency_level: int = 0
    safety_reason: str = ""
    shield_reason: str = ""
    emergency_reason: str = ""
    bsrc_diag: Optional[Dict[str, Any]] = None


class UnifiedSafetySystem:
    """Single RSS/DCBF safety path with the public runtime hooks."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self._rss = RSSCalculator(self.config)
        self.rss = self._rss

    def apply_action_safety_stack(
        self,
        action: int,
        available_actions: List[int],
        state: Any,
        frame: int = 0,
    ) -> FrameSafetyResult:
        available = self._normalise_available_actions(available_actions)
        proposed = int(action)
        if proposed not in available:
            raise ValueError(
                f"proposed action {proposed} is not available at frame {int(frame)}; "
                f"available={available}"
            )
        runtime_state = self._rss._normalise_state(state)
        final_action, reason, overridden = self._rss.filter_action(
            proposed, available, runtime_state
        )
        final_action = int(final_action)
        if final_action not in available:
            raise RuntimeError(
                f"RSS returned unavailable action {final_action} at frame {int(frame)}; "
                f"available={available}; reason={reason}"
            )

        emergency_level = self._emergency_level(runtime_state)
        emergency_reason = "" if emergency_level == 0 else "RSS emergency boundary crossed"
        stage = "rss_dcbf" if overridden else ("emergency" if emergency_level else "none")
        return FrameSafetyResult(
            final_action=final_action,
            proposed_action=proposed,
            state_action=final_action,
            shield_action=final_action,
            safety_override=bool(overridden),
            shield_override=False,
            emergency_level=emergency_level,
            safety_reason=str(reason),
            shield_reason="",
            emergency_reason=emergency_reason,
            bsrc_diag={
                "safety_stack_mode": "rss_dcbf",
                "stack_mode": "rss_dcbf",
                "intervention_stage": stage,
            },
        )

    def get_action_cost_decomposition(
        self, state: Any, available_actions: Iterable[int]
    ) -> Dict[int, Dict[str, float]]:
        available = self._normalise_available_actions(list(available_actions))
        runtime_state = self._rss._normalise_state(state)
        result: Dict[int, Dict[str, float]] = {}
        for action in available:
            assessment = self._rss.assess_safety(
                self._project_state(runtime_state, action)
            )
            safety = self._safety_cost(assessment)
            comfort = 0.05 if action in {LANE_LEFT, LANE_RIGHT} else 0.0
            efficiency = 0.0 if action == FASTER else 0.04
            result[action] = {
                "total": float(safety + comfort + efficiency),
                "safety": float(safety),
                "comfort": float(comfort),
                "efficiency": float(efficiency),
            }
        return result

    def filter_action(
        self, proposed_action: int, available_actions: List[int], state: Any
    ):
        """Expose the RSS three-value result for direct callers."""
        return self._rss.filter_action(proposed_action, available_actions, state)

    def assess_safety(self, state: Any) -> Dict[str, Any]:
        return self._rss.assess_safety(state)

    @staticmethod
    def _normalise_available_actions(available_actions: Iterable[int]) -> List[int]:
        available = sorted({int(item) for item in list(available_actions or [])})
        invalid = [action for action in available if action not in {LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER}]
        if invalid:
            raise ValueError(f"invalid action ids in available_actions: {invalid}")
        if not available:
            raise ValueError("available_actions is empty")
        return available

    def _project_state(self, state: Dict[str, Any], action: int) -> Dict[str, Any]:
        projected = dict(state)
        speed = max(0.0, float_or_default(projected.get("speed"), 0.0))
        front_dist = float_or_default(projected.get("front_dist"), 100.0)
        cross_dist = float_or_default(projected.get("cross_traffic_dist"), float("inf"))
        if action == FASTER:
            speed += 0.48
            front_dist -= speed * self._projection_dt()
        elif action == SLOWER:
            speed = max(0.0, speed - 0.48)
            front_dist += min(0.5, speed * self._projection_dt() * 0.25)
        elif action in {LANE_LEFT, LANE_RIGHT}:
            cross_dist -= max(0.0, speed * min(0.15, self._projection_dt()))
        projected["speed"] = speed
        projected["front_dist"] = front_dist
        projected["cross_traffic_dist"] = cross_dist
        return projected

    def _projection_dt(self) -> float:
        return 1.0 / max(float(self.config.get("policy_frequency", 1.0) or 1.0), 1.0)

    @staticmethod
    def _safety_cost(assessment: Dict[str, Any]) -> float:
        lon_margin = float_or_default(assessment.get("lon_margin"), 0.0)
        cross_margin = float_or_default(assessment.get("cross_margin"), 0.0)
        lon_cost = 0.0 if assessment.get("lon_safe", False) else min(1.0, abs(lon_margin) / 25.0)
        cross_cost = 0.0 if assessment.get("cross_safe", False) else min(1.0, abs(cross_margin) / 25.0)
        return float(max(lon_cost, cross_cost))

    @staticmethod
    def _emergency_level(state: Dict[str, Any]) -> int:
        ttc = float_or_default(state.get("ttc"), float("inf"))
        front_dist = float_or_default(state.get("front_dist"), float("inf"))
        cross_dist = float_or_default(state.get("cross_traffic_dist"), float("inf"))
        if ttc < 1.0 or front_dist < 3.0 or cross_dist < 3.0:
            return 2
        if ttc < 1.8 or front_dist < 5.0 or cross_dist < 5.0:
            return 1
        return 0
