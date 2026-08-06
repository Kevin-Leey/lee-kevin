"""Deterministic RSS/DCBF action filter for the discrete driving interface."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dilu.utils.junction_gap import assess_junction_gap


LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4

ACTION_SPEED_DELTA = {
    LANE_LEFT: 0.0,
    IDLE: 0.0,
    LANE_RIGHT: 0.0,
    FASTER: 0.48,
    SLOWER: -0.48,
}


@dataclass(frozen=True)
class RSSParams:
    reaction_time: float = 0.25
    max_accel: float = 1.0
    min_brake_decel: float = 4.5
    max_brake_decel: float = 4.5
    safety_margin: float = 1.0
    cross_traffic_factor: float = 0.65
    slack: float = 0.15


class RSSCalculator:
    """RSS safe-distance calculator and action-level safety filter."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(config or {})
        rss_cfg = dict(config.get("rss_params", {}) or {})
        safety_cfg = dict(config.get("safety_thresholds", {}) or {})
        self.params = RSSParams(
            reaction_time=float(rss_cfg.get("reaction_time", RSSParams.reaction_time)),
            max_accel=float(rss_cfg.get("max_accel", RSSParams.max_accel)),
            min_brake_decel=float(rss_cfg.get("min_brake_decel", RSSParams.min_brake_decel)),
            max_brake_decel=float(rss_cfg.get("max_brake_decel", RSSParams.max_brake_decel)),
            safety_margin=float(rss_cfg.get("safety_margin", RSSParams.safety_margin)),
            cross_traffic_factor=float(rss_cfg.get("cross_traffic_factor", RSSParams.cross_traffic_factor)),
            slack=float(rss_cfg.get("slack", RSSParams.slack)),
        )
        self._min_lane_gap = float(safety_cfg.get("min_lane_gap", 10.0))
        self._max_safe_speed = float(safety_cfg.get("max_safe_speed", 32.0))
        frequency = max(float(config.get("policy_frequency", 1.0) or 1.0), 1.0)
        self._projection_dt = 1.0 / frequency
        self._lane_change_duration = max(1.0, float(config.get("lane_change_duration", 1.2) or 1.2))

    def longitudinal_safe_distance(self, ego_speed: float, front_speed: float) -> float:
        """Return the longitudinal RSS stopping distance for one lead vehicle."""
        params = self.params
        ego = max(0.0, float(ego_speed))
        front = max(0.0, float(front_speed))
        distance = (
            ego * params.reaction_time
            + ego * ego / (2.0 * max(params.min_brake_decel, 0.1))
            - front * front / (2.0 * max(params.max_brake_decel, 0.1))
            + ego * params.max_accel * params.reaction_time / 2.0
        )
        return max(2.0, distance) * params.safety_margin

    def cross_traffic_safe_distance(self, ego_speed: float, scenario: str = "") -> float:
        """Return a conservative cross-traffic clearance for the current scene."""
        ego = max(0.0, float(ego_speed))
        params = self.params
        distance = ego * params.reaction_time + ego * ego / (2.0 * max(params.min_brake_decel, 0.1))
        scenario_text = str(scenario or "").lower()
        if "intersection" in scenario_text:
            floor = min(10.0, 3.0 + ego * 0.7)
        elif "roundabout" in scenario_text:
            floor = 12.0
        else:
            floor = 3.0
        return max(floor, distance * params.cross_traffic_factor) * params.safety_margin

    def assess_safety(self, state: Any) -> Dict[str, Any]:
        """Evaluate the current longitudinal and cross-traffic margins."""
        state = self._normalise_state(state)
        speed = self._number(state.get("speed"), 0.0)
        front_speed = self._front_speed(state, speed)
        front = self._distance(state.get("front_dist"), float("inf"))
        scenario = self._scenario(state)
        cross = self._effective_cross_distance(state, IDLE)
        d_lon = self.longitudinal_safe_distance(speed, front_speed)
        d_cross = self.cross_traffic_safe_distance(speed, scenario)
        return {
            "lon_safe": bool(front >= d_lon),
            "cross_safe": bool(cross >= d_cross),
            "d_safe_lon": float(d_lon),
            "d_safe_cross": float(d_cross),
            "lon_margin": float(front - d_lon),
            "cross_margin": float(cross - d_cross),
        }

    def filter_action(
        self,
        proposed_action: int,
        available_actions: List[int],
        state: Any,
    ) -> Tuple[int, str, bool]:
        """Return a legal action after applying RSS and target-lane constraints."""
        proposed = int(proposed_action)
        available = [int(action) for action in available_actions]
        if proposed not in available:
            raise ValueError(f"proposed action {proposed} is unavailable; available={available}")
        state = self._normalise_state(state)

        if proposed == FASTER and self._highway_speed_cap_override(state, available):
            replacement = SLOWER if SLOWER in available else (IDLE if IDLE in available else proposed)
            return replacement, "RSS_DCBF_HIGHWAY_SPEED_CAP", replacement != proposed
        if proposed == FASTER and self.highway_adjacent_acceleration_risk(state):
            replacement = IDLE if IDLE in available else (SLOWER if SLOWER in available else proposed)
            return replacement, "RSS_DCBF_ADJACENT_ACCELERATION_HOLD", replacement != proposed

        if proposed in {LANE_LEFT, LANE_RIGHT}:
            if self._highway_target_lane_escape_is_safe(state, proposed, self._number(state.get("speed"), 0.0)):
                return proposed, "RSS_DCBF_TARGET_LANE_ESCAPE", False
            replacement = self._braking_alternative(available, proposed)
            return replacement, "RSS_DCBF_TARGET_LANE_BLOCKED", replacement != proposed

        if self._action_is_safe(state, proposed):
            return proposed, "RSS_DCBF_PROPOSED_PROJECTS_SAFE", False

        for candidate in self._override_order(proposed, available):
            if candidate == proposed:
                continue
            if self._action_is_safe(state, candidate):
                return candidate, f"RSS_DCBF_OVERRIDE_{candidate}", True
        replacement = self._braking_alternative(available, proposed)
        return replacement, "RSS_DCBF_EMERGENCY_BRAKE", replacement != proposed

    def get_action_cost_decomposition(
        self, state: Any, available_actions: Iterable[int]
    ) -> Dict[int, Dict[str, float]]:
        """Expose comparable finite safety costs for every runtime-legal action."""
        state = self._normalise_state(state)
        costs: Dict[int, Dict[str, float]] = {}
        for raw_action in available_actions:
            action = int(raw_action)
            safety = self._action_risk(state, action)
            comfort = 0.03 if action in {LANE_LEFT, LANE_RIGHT} else 0.01 if action == SLOWER else 0.0
            efficiency = 0.08 if action == SLOWER else 0.03 if action == IDLE else 0.0
            costs[action] = {
                "total": float(safety + comfort + efficiency),
                "safety": float(safety),
                "comfort": float(comfort),
                "efficiency": float(efficiency),
            }
        return costs

    @staticmethod
    def highway_adjacent_acceleration_risk(state: Any) -> bool:
        """Hold acceleration while a close adjacent vehicle occupies the merge corridor."""
        state = RSSCalculator._normalise_state(state)
        scenario = str(state.get("scenario_type", state.get("env_type", "")) or "").lower()
        if "highway" not in scenario:
            return False
        distance = RSSCalculator._distance(state.get("closest_vehicle_distance"), float("inf"))
        longitudinal = RSSCalculator._number(state.get("closest_vehicle_longitudinal"), float("inf"))
        lateral = abs(RSSCalculator._number(state.get("closest_vehicle_lateral"), float("inf")))
        if not all(math.isfinite(value) for value in (distance, longitudinal, lateral)):
            return False
        return bool(distance <= 8.0 and 2.0 <= lateral <= 7.0 and -4.0 <= longitudinal <= 14.0)

    def _action_is_safe(self, state: Dict[str, Any], action: int) -> bool:
        if action in {LANE_LEFT, LANE_RIGHT}:
            return self._highway_target_lane_escape_is_safe(state, action, self._number(state.get("speed"), 0.0))
        projected = self._project_state(state, action)
        speed = self._number(state.get("speed"), 0.0)
        front_speed = self._front_speed(state, speed)
        safe_lon = self.longitudinal_safe_distance(speed, front_speed) * (1.0 - self.params.slack)
        safe_cross = self.cross_traffic_safe_distance(speed, self._scenario(state)) * (1.0 - self.params.slack)
        return bool(projected["front_dist"] >= safe_lon and projected["cross_dist"] >= safe_cross)

    def _action_risk(self, state: Dict[str, Any], action: int) -> float:
        if action in {LANE_LEFT, LANE_RIGHT}:
            return 0.0 if self._highway_target_lane_escape_is_safe(state, action, self._number(state.get("speed"), 0.0)) else 1.0
        projected = self._project_state(state, action)
        speed = self._number(state.get("speed"), 0.0)
        front_speed = self._front_speed(state, speed)
        safe_lon = max(1.0, self.longitudinal_safe_distance(speed, front_speed) * (1.0 - self.params.slack))
        safe_cross = max(1.0, self.cross_traffic_safe_distance(speed, self._scenario(state)) * (1.0 - self.params.slack))
        lon = max(0.0, (safe_lon - projected["front_dist"]) / safe_lon)
        cross = max(0.0, (safe_cross - projected["cross_dist"]) / safe_cross)
        return float(min(1.0, max(lon, cross)))

    def _highway_target_lane_escape_is_safe(
        self, state: Dict[str, Any], action: int, ego_speed: float
    ) -> bool:
        """Require source and target corridor margins through lane-change completion."""
        action = int(action)
        if action not in {LANE_LEFT, LANE_RIGHT}:
            return False
        side = "left" if action == LANE_LEFT else "right"
        ego = max(0.0, float(ego_speed))
        target_front = self._distance(state.get(f"{side}_front"), float("inf"))
        target_rear = self._distance(state.get(f"{side}_rear"), float("inf"))
        target_front_speed = self._lane_speed(state.get(f"{side}_front_speed"), ego)
        target_rear_speed = self._lane_speed(state.get(f"{side}_rear_speed"), ego)
        duration = self._lane_change_duration
        target_front_completion = target_front + (target_front_speed - ego) * duration
        rear_closing = max(0.0, target_rear_speed - ego)
        target_rear_completion = target_rear - rear_closing * duration
        target_floor = max(
            self._min_lane_gap + 5.0 + 1.5 * max(0.0, ego - target_front_speed),
            self.longitudinal_safe_distance(ego, target_front_speed),
        )
        rear_floor = max(self._min_lane_gap, 4.0 + 0.5 * rear_closing)
        if target_front_completion < target_floor or target_rear_completion < rear_floor:
            return False

        source_front = self._distance(state.get("front_dist"), float("inf"))
        source_speed = self._front_speed(state, ego)
        source_completion = source_front + min(0.0, source_speed - ego) * min(duration, 1.0)
        source_floor = max(3.0, 0.45 * self._min_lane_gap)
        return bool(source_completion >= source_floor)

    def _highway_speed_cap_override(self, state: Dict[str, Any], available: List[int]) -> bool:
        scenario = self._scenario(state)
        return bool("highway" in scenario and FASTER in available and self._number(state.get("speed"), 0.0) >= self._max_safe_speed)

    def _effective_cross_distance(self, state: Dict[str, Any], action: int) -> float:
        raw = self._distance(state.get("cross_traffic_dist", state.get("cross_traffic_distance")), float("inf"))
        gap = assess_junction_gap(state, action)
        if gap.get("occupied_corridor", False):
            return 0.0
        if gap.get("should_yield", False):
            return min(raw, self._number(gap.get("effective_cross_distance"), raw))
        if gap.get("should_clear", False) and gap.get("in_junction", False):
            return max(raw, self._number(gap.get("effective_cross_distance"), raw))
        return raw

    def _project_state(self, state: Dict[str, Any], action: int) -> Dict[str, float]:
        speed = self._number(state.get("speed"), 0.0)
        front_speed = self._front_speed(state, speed)
        front = self._distance(state.get("front_dist"), float("inf"))
        next_speed = max(0.0, speed + ACTION_SPEED_DELTA.get(int(action), 0.0))
        if not math.isfinite(front):
            projected_front = float("inf")
        else:
            projected_front = max(0.0, front + (front_speed - next_speed) * self._projection_dt)
        return {
            "front_dist": float(projected_front),
            "cross_dist": float(self._effective_cross_distance(state, int(action))),
        }

    def _braking_alternative(self, available: List[int], proposed: int) -> int:
        if SLOWER in available:
            return SLOWER
        if IDLE in available:
            return IDLE
        return int(proposed)

    @staticmethod
    def _override_order(proposed: int, available: List[int]) -> List[int]:
        if proposed == FASTER:
            return [IDLE, LANE_LEFT, LANE_RIGHT, SLOWER]
        return [SLOWER, IDLE, LANE_LEFT, LANE_RIGHT, FASTER]

    @staticmethod
    def _number(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) else float(default)

    @classmethod
    def _distance(cls, value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return float(default)
        return result if math.isfinite(result) and result >= 0.0 else float(default)

    @classmethod
    def _front_speed(cls, state: Dict[str, Any], ego_speed: float) -> float:
        return cls._lane_speed(state.get("front_speed"), ego_speed)

    @classmethod
    def _lane_speed(cls, value: Any, fallback: float) -> float:
        try:
            speed = float(value)
        except (TypeError, ValueError):
            return float(fallback)
        return max(0.0, speed) if math.isfinite(speed) else float(fallback)

    @staticmethod
    def _scenario(state: Dict[str, Any]) -> str:
        return str(state.get("scenario_type", state.get("env_type", "")) or "").lower()

    @staticmethod
    def _normalise_state(state: Any) -> Dict[str, Any]:
        """Convert the runtime's structured state into the RSS field schema."""
        if isinstance(state, Mapping):
            normalized = dict(state)
        elif state is None:
            normalized = {}
        elif callable(getattr(state, "to_dict", None)):
            normalized = dict(state.to_dict())
        else:
            normalized = dict(vars(state))

        aliases = {
            "speed": "ego_speed",
            "lane": "ego_lane",
            "front_dist": "front_distance",
            "left_front": "left_front_distance",
            "left_rear": "left_rear_distance",
            "right_front": "right_front_distance",
            "right_rear": "right_rear_distance",
            "cross_traffic_dist": "cross_traffic_distance",
            "history": "history_frames",
        }
        for canonical, runtime_name in aliases.items():
            if canonical not in normalized and runtime_name in normalized:
                normalized[canonical] = normalized[runtime_name]
        if "ego_x" not in normalized or "ego_y" not in normalized:
            position = normalized.get("position", normalized.get("pos"))
            try:
                x, y = position[0], position[1]
            except (IndexError, KeyError, TypeError):
                pass
            else:
                normalized.setdefault("ego_x", x)
                normalized.setdefault("ego_y", y)
        return normalized
