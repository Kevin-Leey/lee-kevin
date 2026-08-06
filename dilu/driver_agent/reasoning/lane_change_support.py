"""Shared lane-change support used by both fast and slow reasoners."""

from typing import Any, Dict, Optional

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.utils.shared import float_or_default


LANE_CHANGE_GUARD_KEYS = (
    "lane_change_min_front_gap",
    "lane_change_min_rear_gap",
    "lane_change_front_headway_seconds",
    "lane_change_rear_headway_seconds",
    "lane_change_high_speed_threshold",
    "lane_change_high_speed_rear_gap",
    "lane_change_target_front_speed_bias",
    "lane_change_target_rear_speed_margin",
)

LANE_CHANGE_GUARD_PROFILES = {
    "slow": {
        "lane_change_min_front_gap": 22.0,
        "lane_change_min_rear_gap": 14.0,
        "lane_change_front_headway_seconds": 0.90,
        "lane_change_rear_headway_seconds": 0.65,
        "lane_change_high_speed_threshold": 22.0,
        "lane_change_high_speed_rear_gap": 20.0,
        "lane_change_target_front_speed_bias": 4.0,
        "lane_change_target_rear_speed_margin": 6.0,
    },
    "fast": {
        "lane_change_min_front_gap": 16.0,
        "lane_change_min_rear_gap": 10.0,
        "lane_change_front_headway_seconds": 0.65,
        "lane_change_rear_headway_seconds": 0.55,
        "lane_change_high_speed_threshold": 22.0,
        "lane_change_high_speed_rear_gap": 18.0,
        "lane_change_target_front_speed_bias": 3.0,
        "lane_change_target_rear_speed_margin": 6.0,
    },
}


def _resolve_guard_numeric(source: Dict[str, Any], key: str, default: float) -> float:
    """Resolve one numeric guard value without silently discarding explicit zero-like inputs."""
    value = source.get(key, default)
    if value in {None, ""}:
        return float(default)
    return float(value)


def resolve_lane_change_guard_config(config: Optional[Dict[str, Any]] = None, profile: str = "slow") -> Dict[str, float]:
    """Resolve one shared lane-change guard config from runtime overrides plus a named default profile."""
    profile_name = str(profile or "slow").strip().lower()
    resolved = dict(LANE_CHANGE_GUARD_PROFILES.get(profile_name, LANE_CHANGE_GUARD_PROFILES["slow"]))
    source = dict(config or {})
    nested_source = dict(source.get("lane_change_guard", {}) or {})
    for key in LANE_CHANGE_GUARD_KEYS:
        if key in nested_source:
            resolved[key] = _resolve_guard_numeric(nested_source, key, resolved[key])
        elif key in source:
            resolved[key] = _resolve_guard_numeric(source, key, resolved[key])
    return resolved


def lane_change_direction(action: int) -> Optional[str]:
    """Return the lane-change direction label for the given action."""
    if int(action) == ActionType.LANE_LEFT:
        return "left"
    if int(action) == ActionType.LANE_RIGHT:
        return "right"
    return None


def lane_change_snapshot(state: DrivingState, action: int) -> Optional[Dict[str, Any]]:
    """Collect the target-lane distances, speeds, and base safety for one action."""
    direction = lane_change_direction(action)
    if direction is None:
        return None
    if direction == "left":
        return {
            "direction": direction,
            "front_distance": float(state.left_front_distance),
            "rear_distance": float(state.left_rear_distance),
            "front_speed": state.left_front_speed,
            "rear_speed": state.left_rear_speed,
            "lane_safe": bool(state.lane_change_safety.get(direction, False)),
        }
    return {
        "direction": direction,
        "front_distance": float(state.right_front_distance),
        "rear_distance": float(state.right_rear_distance),
        "front_speed": state.right_front_speed,
        "rear_speed": state.right_rear_speed,
        "lane_safe": bool(state.lane_change_safety.get(direction, False)),
    }


def _estimate_front_speed(state: DrivingState, snapshot: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Resolve target-lane lead speed, falling back to a conservative estimate when missing."""
    observed_speed = snapshot.get("front_speed", None)
    if observed_speed is not None:
        return max(0.0, float(observed_speed))

    ego_speed = max(0.0, float(state.ego_speed or 0.0))
    target_front_distance = _distance_or_inf(snapshot.get("front_distance", float("inf")))
    if target_front_distance == float("inf"):
        return ego_speed

    current_front_speed = max(0.0, float(state.front_speed or 0.0))
    closing_bias = float(config.get("lane_change_target_front_speed_bias", 4.0) or 4.0)
    reference_gap = max(float(config.get("lane_change_min_front_gap", 22.0) or 22.0), 1e-6)
    gap_pressure = float(max(0.0, min(1.0, 1.0 - target_front_distance / reference_gap)))
    conservative_speed = max(0.0, ego_speed - closing_bias * (0.55 + 0.45 * gap_pressure))
    if current_front_speed > 0.0:
        conservative_speed = min(conservative_speed, current_front_speed)
    return conservative_speed


def _estimate_rear_speed(state: DrivingState, snapshot: Dict[str, Any], config: Dict[str, Any]) -> float:
    """Resolve target-lane rear speed, falling back to a conservative closing-speed estimate."""
    observed_speed = snapshot.get("rear_speed", None)
    if observed_speed is not None:
        return max(0.0, float(observed_speed))

    ego_speed = max(0.0, float(state.ego_speed or 0.0))
    target_rear_distance = _distance_or_inf(snapshot.get("rear_distance", float("inf")))
    if target_rear_distance == float("inf"):
        return ego_speed

    rear_speed_margin = float(config.get("lane_change_target_rear_speed_margin", 6.0) or 6.0)
    reference_gap = max(float(config.get("lane_change_min_rear_gap", 14.0) or 14.0), 1e-6)
    gap_pressure = float(max(0.0, min(1.0, 1.0 - target_rear_distance / reference_gap)))
    return max(ego_speed, ego_speed + rear_speed_margin * (0.45 + 0.55 * gap_pressure))


def _distance_or_inf(value: Any) -> float:
    """Normalize an optional distance while preserving a real zero-distance observation."""
    if value is None:
        return float("inf")
    return float(value)


def build_lane_change_guard(
    state: DrivingState,
    action: int,
    ego_speed: float,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a shared dynamic guard so fast and slow paths reason over the same target lane state."""
    snapshot = lane_change_snapshot(state, action)
    if snapshot is None:
        return None

    scenario_key = str(state.scenario_type or "").split("-")[0].strip().lower()
    scenario_factor = {
        "highway": 1.20,
        "merge": 1.10,
        "roundabout": 0.95,
        "intersection": 0.95,
    }.get(scenario_key, 1.0)

    dynamic_front_gap = max(
        float(config.get("lane_change_min_front_gap", 22.0) or 22.0),
        float(ego_speed or 0.0) * float(config.get("lane_change_front_headway_seconds", 0.90) or 0.90),
    ) * scenario_factor
    dynamic_rear_gap = max(
        float(config.get("lane_change_min_rear_gap", 14.0) or 14.0),
        float(ego_speed or 0.0) * float(config.get("lane_change_rear_headway_seconds", 0.65) or 0.65),
    ) * scenario_factor
    if float(ego_speed or 0.0) >= float(config.get("lane_change_high_speed_threshold", 22.0) or 22.0):
        dynamic_rear_gap = max(
            dynamic_rear_gap,
            float(config.get("lane_change_high_speed_rear_gap", 20.0) or 20.0) * scenario_factor,
        )

    front_distance = _distance_or_inf(snapshot.get("front_distance", float("inf")))
    rear_distance = _distance_or_inf(snapshot.get("rear_distance", float("inf")))
    front_gap_pressure = 0.0 if front_distance == float("inf") else float(max(0.0, min(1.0, 1.0 - front_distance / max(dynamic_front_gap, 1e-6))))
    rear_gap_pressure = 0.0 if rear_distance == float("inf") else float(max(0.0, min(1.0, 1.0 - rear_distance / max(dynamic_rear_gap, 1e-6))))
    estimated_front_speed = _estimate_front_speed(state, snapshot, config)
    estimated_rear_speed = _estimate_rear_speed(state, snapshot, config)
    rear_closing_speed = max(0.0, float(estimated_rear_speed) - max(0.0, float(ego_speed or 0.0)))
    rear_time_to_contact = float("inf")
    if rear_distance != float("inf") and rear_closing_speed > 1e-6:
        rear_time_to_contact = float(rear_distance / rear_closing_speed)

    closest_blocked = _closest_vehicle_lane_change_blocked(state, action, ego_speed)
    guard_trigger = (not bool(snapshot.get("lane_safe", False))) or front_gap_pressure > 0.0 or rear_gap_pressure > 0.0
    blocked = (not bool(snapshot.get("lane_safe", False))) or front_gap_pressure >= 1.0 or rear_gap_pressure >= 1.0 or closest_blocked
    return {
        **snapshot,
        "dynamic_front_gap": float(dynamic_front_gap),
        "dynamic_rear_gap": float(dynamic_rear_gap),
        "front_gap_pressure": float(front_gap_pressure),
        "rear_gap_pressure": float(rear_gap_pressure),
        "pressure": float(max(front_gap_pressure, rear_gap_pressure)),
        "guard_trigger": bool(guard_trigger),
        "blocked": bool(blocked),
        "estimated_front_speed": float(estimated_front_speed),
        "estimated_rear_speed": float(estimated_rear_speed),
        "rear_closing_speed": float(rear_closing_speed),
        "rear_time_to_contact": float(rear_time_to_contact),
        "closest_vehicle_blocked": bool(closest_blocked),
    }


def _closest_vehicle_lane_change_blocked(state: DrivingState, action: int, ego_speed: float) -> bool:
    lateral = float_or_default(getattr(state, "closest_vehicle_lateral", None), float("inf"))
    longitudinal = float_or_default(getattr(state, "closest_vehicle_longitudinal", None), float("inf"))
    closest = float_or_default(getattr(state, "closest_vehicle_distance", None), float("inf"))
    if not (lateral != float("inf") and longitudinal != float("inf") and closest != float("inf")):
        return False
    target_side = _lateral_on_target_side(state, action, 1.4)
    if not target_side:
        return False
    lateral_abs = abs(lateral)
    if lateral_abs > 10.5:
        return False
    speed = max(0.0, float(ego_speed or 0.0))
    if longitudinal > 0.0 and lateral_abs <= 9.8 and longitudinal <= max(18.0, speed * 0.82):
        return True
    forward_limit = max(10.0, speed * 0.50)
    rear_limit = max(7.5, speed * 0.34)
    return bool(-rear_limit <= longitudinal <= forward_limit)


def _lateral_on_target_side(state: DrivingState, action: int, threshold: float) -> bool:
    lateral = float_or_default(getattr(state, "closest_vehicle_lateral", None), float("inf"))
    env_type = str(getattr(state, "env_type", "") or "").lower()
    left_is_negative = "metadrive" in env_type
    if int(action) == ActionType.LANE_LEFT:
        return bool(lateral < -float(threshold) if left_is_negative else lateral > float(threshold))
    if int(action) == ActionType.LANE_RIGHT:
        return bool(lateral > float(threshold) if left_is_negative else lateral < -float(threshold))
    return False
