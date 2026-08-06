"""Deterministic high-frequency controller used as the RGD incumbent."""

from __future__ import annotations

import collections
import math
import time
from typing import Any, Dict, Iterable, Optional

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.policy_state import FAST_POLICY_STATE_SCHEMA, validate_fast_policy_state
from dilu.driver_agent.reasoning.decision import RGDDecision
from dilu.driver_agent.reasoning.lane_change_support import (
    build_lane_change_guard,
    resolve_lane_change_guard_config,
)


class FastThinker:
    """A transparent rule controller over the shared five-action domain.

    The controller is intentionally stateless with respect to observations.  Its
    only temporal state is the history of *executed* commands, which is updated
    by :meth:`record_executed_action` after the actuator and safety layers have
    resolved the frame.
    """

    def __init__(self, lane_change_config: Optional[Dict[str, Any]] = None) -> None:
        config = dict(lane_change_config or {})
        self.config = resolve_lane_change_guard_config(config, profile="fast")
        self.target_speed = float(config.get("target_speed", 27.0) or 27.0)
        self.following_ttc = float(config.get("following_ttc", 2.5) or 2.5)
        self.following_headway = float(config.get("following_headway", 0.9) or 0.9)
        self.lane_change_cooldown = max(
            0, int(config.get("lane_change_cooldown", 4) or 0)
        )
        self.action_history_capacity = max(
            1, int(config.get("action_history_capacity", 12) or 12)
        )
        self.action_history: collections.deque[int] = collections.deque(
            maxlen=self.action_history_capacity
        )
        self.stats: Dict[str, Any] = {"decisions": 0}
        self.last_rule_match: Optional[str] = None

    @staticmethod
    def _finite(value: Any, default: float = math.inf) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        return parsed if math.isfinite(parsed) else float(default)

    def _lane_change_available(self, state: DrivingState, action: int) -> bool:
        available = set(state.get_available_actions())
        if int(action) not in available:
            return False
        if int(action) == int(ActionType.LANE_LEFT) and not state.can_change_left:
            return False
        if int(action) == int(ActionType.LANE_RIGHT) and not state.can_change_right:
            return False
        if self.lane_change_cooldown <= 0:
            return True
        recent = list(self.action_history)[-self.lane_change_cooldown :]
        return not any(
            command in {int(ActionType.LANE_LEFT), int(ActionType.LANE_RIGHT)}
            for command in recent
        )

    def _candidate_lane_change(self, state: DrivingState) -> Optional[int]:
        candidates = []
        for action in (int(ActionType.LANE_LEFT), int(ActionType.LANE_RIGHT)):
            if not self._lane_change_available(state, action):
                continue
            guard = build_lane_change_guard(
                state, action, float(state.ego_speed), self.config
            )
            if guard is None or bool(guard.get("blocked", True)):
                continue
            clearance = min(
                float(guard["front_distance"]), float(guard["rear_distance"])
            )
            candidates.append((clearance, action))
        if not candidates:
            return None
        return int(max(candidates, key=lambda item: item[0])[1])

    def _evaluate(self, state: DrivingState) -> RGDDecision:
        started = time.perf_counter()
        available = set(state.get_available_actions())
        if not available:
            raise ValueError("fast controller received an empty action universe")

        speed = max(0.0, float(state.ego_speed))
        front_distance = self._finite(state.front_distance)
        ttc = self._finite(state.ttc)
        thw = self._finite(state.thw)
        dynamic_gap = max(8.0, speed * self.following_headway)
        severe_following = (
            (math.isfinite(ttc) and ttc <= self.following_ttc)
            or (math.isfinite(front_distance) and front_distance <= dynamic_gap)
            or (math.isfinite(thw) and thw <= self.following_headway)
        )
        moderate_following = (
            math.isfinite(front_distance)
            and front_distance <= max(24.0, speed * 1.35)
        )

        lane_action = self._candidate_lane_change(state) if moderate_following else None
        if severe_following and int(ActionType.SLOWER) in available:
            action = int(ActionType.SLOWER)
            rule = "following_brake"
            confidence = 0.98
        elif lane_action is not None:
            action = int(lane_action)
            rule = "clear_adjacent_escape"
            confidence = 0.90
        elif moderate_following and int(ActionType.IDLE) in available:
            action = int(ActionType.IDLE)
            rule = "following_hold"
            confidence = 0.82
        elif (
            int(ActionType.FASTER) in available
            and speed + 0.25 < self.target_speed
            and (not math.isfinite(front_distance) or front_distance > max(32.0, speed * 1.65))
        ):
            action = int(ActionType.FASTER)
            rule = "open_road_progress"
            confidence = 0.76
        elif int(ActionType.IDLE) in available:
            action = int(ActionType.IDLE)
            rule = "steady_cruise"
            confidence = 0.70
        else:
            action = int(sorted(available)[0])
            rule = "only_available_action"
            confidence = 0.60

        selected_parts = {
            "front_distance": front_distance,
            "ttc": ttc,
            "thw": thw,
            "dynamic_following_gap": dynamic_gap,
            "severe_following": severe_following,
            "moderate_following": moderate_following,
        }
        return RGDDecision(
            action=action,
            reasoning=f"fast rule: {rule}",
            confidence=confidence,
            system_used="fast",
            route_label="fast_rule",
            route_score=0.0,
            stats={
                "rule_name": rule,
                "decision_mode": "hard_rule_shell",
                "smoothness_override": False,
                "abstention_applied": False,
                "top_score_gap": 1.0,
                "calibration_context": {
                    "scenario_profile": str(state.scenario_type or "generic"),
                    "distance_buffer_m": dynamic_gap,
                    "ttc_buffer_s": self.following_ttc,
                    "cross_traffic_buffer_m": 0.0,
                    "caution_level": 1.0 if severe_following else 0.0,
                    "abstention_band": 0.0,
                    "midlayer_best_action": action,
                    "midlayer_runner_up_action": action,
                    "best_score": confidence,
                    "runner_up_score": 0.0,
                    "shared_context": selected_parts,
                    "selected_action_parts": selected_parts,
                },
            },
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def think(self, state: DrivingState) -> RGDDecision:
        decision = self._evaluate(state)
        self.stats["decisions"] = int(self.stats.get("decisions", 0)) + 1
        self.last_rule_match = str(decision.stats.get("rule_name", ""))
        return decision

    def peek(self, state: DrivingState) -> RGDDecision:
        return self._evaluate(state)

    def record_executed_action(self, action: int) -> None:
        action_id = int(action)
        if action_id not in {int(value) for value in ActionType}:
            raise ValueError(f"unsupported executed action: {action}")
        self.action_history.append(action_id)

    def snapshot_runtime_state(self) -> Dict[str, Any]:
        return {
            "action_history": collections.deque(
                self.action_history, maxlen=self.action_history_capacity
            ),
            "stats": dict(self.stats),
            "last_rule_match": self.last_rule_match,
        }

    def restore_runtime_state(self, snapshot: Dict[str, Any]) -> None:
        history = snapshot.get("action_history", [])
        values = [int(action) for action in list(history)]
        if len(values) > self.action_history_capacity:
            values = values[-self.action_history_capacity :]
        self.action_history = collections.deque(
            values, maxlen=self.action_history_capacity
        )
        self.stats = dict(snapshot.get("stats", {}) or {})
        self.last_rule_match = snapshot.get("last_rule_match")

    def snapshot_policy_state(self) -> Dict[str, Any]:
        return {
            "schema": FAST_POLICY_STATE_SCHEMA,
            "action_history": list(self.action_history),
            "action_history_capacity": self.action_history_capacity,
        }

    def restore_policy_state(self, snapshot: Dict[str, Any]) -> None:
        normalized = validate_fast_policy_state(
            snapshot, expected_capacity=self.action_history_capacity
        )
        self.action_history = collections.deque(
            normalized["action_history"], maxlen=self.action_history_capacity
        )
