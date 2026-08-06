"""Frame-local state extraction and execution-history helpers.

The runtime keeps the current observation separate from the executed-action
history.  This makes the policy state at a frame unambiguous: the history
contains completed frames only, and the current action is recorded after the
environment step succeeds.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableSequence
from typing import Any, Dict, Iterable, Optional, Sequence

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.utils.shared import float_or_default


def _mapping_value(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _finite_or_inf(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return resolved if math.isfinite(resolved) else float("inf")


def _heading(vehicle: Any) -> float:
    value = _mapping_value(vehicle, "heading_theta", "heading", default=0.0)
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(math.atan2(float(value.y), float(value.x)))
    try:
        vector = list(value)
    except TypeError:
        return 0.0
    return float(math.atan2(float(vector[1]), float(vector[0]))) if len(vector) >= 2 else 0.0


def _relative_geometry(ego: Any, other: Any) -> tuple[float, float, float]:
    try:
        ex, ey = float(ego.position[0]), float(ego.position[1])
        ox, oy = float(other.position[0]), float(other.position[1])
    except (AttributeError, IndexError, TypeError, ValueError):
        return float("inf"), float("inf"), float("inf")
    heading = _heading(ego)
    dx, dy = ox - ex, oy - ey
    longitudinal = dx * math.cos(heading) + dy * math.sin(heading)
    lateral = -dx * math.sin(heading) + dy * math.cos(heading)
    return float(math.hypot(dx, dy)), float(longitudinal), float(lateral)


def _lane_identifier(vehicle: Any, fallback: int = 0) -> int:
    lane_index = _mapping_value(vehicle, "lane_index")
    try:
        return int(lane_index[2])
    except (IndexError, TypeError, ValueError):
        return int(fallback)


def _nearby_vehicles(sce: Any, env: Any, limit: int = 12) -> Sequence[Any]:
    for name in ("getSurroundingVehicles", "getSurrendVehicles"):
        getter = getattr(sce, name, None)
        if callable(getter):
            try:
                return list(getter(limit) or [])
            except (AttributeError, TypeError, ValueError):
                pass
    unwrapped = getattr(env, "unwrapped", env)
    road = getattr(unwrapped, "road", None)
    vehicles = list(getattr(road, "vehicles", []) or [])
    ego = getattr(unwrapped, "vehicle", None)
    return [vehicle for vehicle in vehicles if vehicle is not ego][:limit]


def extract_runtime_state(env: Any, sce: Any, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Return backend-neutral kinematic evidence for the current frame."""
    unwrapped = getattr(env, "unwrapped", env)
    ego = getattr(unwrapped, "vehicle", None)
    if ego is None:
        raise RuntimeError("runtime environment does not expose an ego vehicle")

    env_type = str(cfg.get("env_type", getattr(getattr(env, "spec", None), "id", "")) or "")
    scenario_type = str(
        cfg.get("scenario_type", getattr(sce, "scenario_type", "highway")) or "highway"
    )
    lane = _lane_identifier(ego)
    total_lanes = int(
        cfg.get("lanes_count", getattr(unwrapped, "config", {}).get("lanes_count", 1))
        or 1
    )
    get_actions = getattr(unwrapped, "get_available_actions", None)
    raw_actions = get_actions() if callable(get_actions) else [int(item) for item in ActionType]
    legal_actions = sorted({int(action) for action in list(raw_actions or [])})

    ego_speed = max(0.0, float_or_default(getattr(ego, "speed", 0.0), 0.0))
    front_distance = float("inf")
    front_speed: Optional[float] = None
    left_front_distance = left_rear_distance = float("inf")
    right_front_distance = right_rear_distance = float("inf")
    left_front_speed = left_rear_speed = right_front_speed = right_rear_speed = None
    closest_distance = closest_longitudinal = closest_lateral = float("inf")
    closest_closing_speed = 0.0
    cross_distance = float("inf")
    nearby = _nearby_vehicles(sce, env)
    for vehicle in nearby:
        if vehicle is ego:
            continue
        distance, longitudinal, lateral = _relative_geometry(ego, vehicle)
        if not math.isfinite(distance):
            continue
        other_speed = max(0.0, float_or_default(getattr(vehicle, "speed", 0.0), 0.0))
        if distance < closest_distance:
            closest_distance = distance
            closest_longitudinal = longitudinal
            closest_lateral = lateral
            closest_closing_speed = ego_speed - other_speed
        other_lane = _lane_identifier(vehicle, lane)
        lane_delta = other_lane - lane
        if lane_delta == 0 and longitudinal > 0.0 and distance < front_distance:
            front_distance, front_speed = distance, other_speed
        elif lane_delta < 0:
            if longitudinal >= 0.0 and distance < left_front_distance:
                left_front_distance, left_front_speed = distance, other_speed
            elif longitudinal < 0.0 and distance < left_rear_distance:
                left_rear_distance, left_rear_speed = distance, other_speed
        elif lane_delta > 0:
            if longitudinal >= 0.0 and distance < right_front_distance:
                right_front_distance, right_front_speed = distance, other_speed
            elif longitudinal < 0.0 and distance < right_rear_distance:
                right_rear_distance, right_rear_speed = distance, other_speed
        if scenario_type in {"intersection", "roundabout", "merge"} and abs(lateral) > 1.5:
            cross_distance = min(cross_distance, distance)

    relative_speed = ego_speed - float(front_speed) if front_speed is not None else 0.0
    ttc = front_distance / relative_speed if relative_speed > 1e-6 else float("inf")
    thw = front_distance / ego_speed if ego_speed > 1e-6 else float("inf")
    return {
        "speed": ego_speed,
        "front_speed": front_speed,
        "front_dist": front_distance,
        "front_relative_speed": relative_speed,
        "lane": lane,
        "total_lanes": max(1, total_lanes),
        "nearby_vehicle_count": len(nearby),
        "pos": tuple(getattr(ego, "position", (0.0, 0.0))),
        "heading": _heading(ego),
        "ttc": ttc,
        "thw": thw,
        "left_front_dist": left_front_distance,
        "left_rear_dist": left_rear_distance,
        "left_front_speed": left_front_speed,
        "left_rear_speed": left_rear_speed,
        "right_front_dist": right_front_distance,
        "right_rear_dist": right_rear_distance,
        "right_front_speed": right_front_speed,
        "right_rear_speed": right_rear_speed,
        "closest_vehicle_distance": closest_distance,
        "closest_vehicle_longitudinal": closest_longitudinal,
        "closest_vehicle_lateral": closest_lateral,
        "closest_vehicle_closing_speed": closest_closing_speed,
        "cross_traffic_distance": cross_distance,
        "legal_actions": legal_actions,
        "env_type": env_type,
        "scenario_type": scenario_type,
    }


def build_frame_driving_state(
    frame_state: Mapping[str, Any],
    scenario_type: str,
    cfg: Mapping[str, Any],
    history_buffer: Iterable[Mapping[str, Any]],
    *,
    prev_action: int,
    env_available_actions: Optional[Iterable[int]] = None,
) -> DrivingState:
    """Build the immutable policy input for a frame without appending history."""
    state = dict(frame_state or {})
    actions = (
        list(env_available_actions)
        if env_available_actions is not None
        else list(state.get("legal_actions", []) or [])
    )
    if not actions:
        actions = [int(ActionType.IDLE)]
    lane = int(state.get("lane", state.get("ego_lane", 0)) or 0)
    total_lanes = int(state.get("total_lanes", cfg.get("lanes_count", 1)) or 1)
    can_left = int(ActionType.LANE_LEFT) in actions and lane > 0
    can_right = int(ActionType.LANE_RIGHT) in actions and lane < total_lanes - 1
    driving_state = DrivingState(
        ego_speed=float_or_default(state.get("speed", state.get("ego_speed", 0.0)), 0.0),
        ego_lane=lane,
        total_lanes=total_lanes,
        position=state.get("pos", state.get("position")),
        heading=float_or_default(state.get("heading"), 0.0),
        scenario_type=str(scenario_type or state.get("scenario_type", "highway")),
        env_type=str(cfg.get("env_type", state.get("env_type", "")) or ""),
        front_distance=_finite_or_inf(state.get("front_dist", state.get("front_distance"))),
        front_speed=state.get("front_speed"),
        front_relative_speed=float_or_default(state.get("front_relative_speed"), 0.0),
        ttc=_finite_or_inf(state.get("ttc")),
        thw=_finite_or_inf(state.get("thw")),
        left_front_distance=_finite_or_inf(state.get("left_front_dist", state.get("left_front_distance"))),
        left_rear_distance=_finite_or_inf(state.get("left_rear_dist", state.get("left_rear_distance"))),
        left_front_speed=state.get("left_front_speed"),
        left_rear_speed=state.get("left_rear_speed"),
        right_front_distance=_finite_or_inf(state.get("right_front_dist", state.get("right_front_distance"))),
        right_rear_distance=_finite_or_inf(state.get("right_rear_dist", state.get("right_rear_distance"))),
        right_front_speed=state.get("right_front_speed"),
        right_rear_speed=state.get("right_rear_speed"),
        can_change_left=can_left,
        can_change_right=can_right,
        closest_vehicle_distance=_finite_or_inf(state.get("closest_vehicle_distance")),
        closest_vehicle_longitudinal=_finite_or_inf(state.get("closest_vehicle_longitudinal")),
        closest_vehicle_lateral=_finite_or_inf(state.get("closest_vehicle_lateral")),
        closest_vehicle_closing_speed=float_or_default(state.get("closest_vehicle_closing_speed"), 0.0),
        cross_traffic_distance=_finite_or_inf(state.get("cross_traffic_distance")),
        nearby_vehicle_count=int(state.get("nearby_vehicle_count", 0) or 0),
        legal_actions=[int(action) for action in actions],
        history_frames=[dict(item) for item in list(history_buffer or [])],
    )
    driving_state.__dict__["previous_action"] = int(prev_action)
    return driving_state


def record_executed_history_frame(
    history_buffer: MutableSequence[Dict[str, Any]],
    frame_state: Mapping[str, Any],
    action: int,
) -> None:
    """Append a compact record of the action actually accepted by the environment."""
    state = dict(frame_state or {})
    history_buffer.append(
        {
            "speed": float_or_default(state.get("speed", state.get("ego_speed", 0.0)), 0.0),
            "ttc": _finite_or_inf(state.get("ttc")),
            "thw": _finite_or_inf(state.get("thw")),
            "action": int(action),
        }
    )


def inject_safety_cost_bridge(
    driving_state: DrivingState,
    env: Any,
    runtime_state: Mapping[str, Any],
    safety_system: Any,
) -> None:
    """Attach the safety decomposition over the authoritative action universe."""
    del env, runtime_state
    action_universe = driving_state.get_available_actions()
    getter = getattr(safety_system, "get_action_cost_decomposition", None)
    if not callable(getter):
        return
    safety_state = driving_state.to_dict()
    decomposition = getter(safety_state, action_universe)
    if not isinstance(decomposition, Mapping):
        raise TypeError("safety cost decomposition must be a mapping")
    normalized = {int(action): dict(parts or {}) for action, parts in decomposition.items()}
    if set(normalized) != set(action_universe):
        raise ValueError("safety cost decomposition does not cover the action universe")
    driving_state.__dict__["_safety_cost_decomposition"] = normalized


__all__ = [
    "build_frame_driving_state",
    "extract_runtime_state",
    "inject_safety_cost_bridge",
    "record_executed_history_frame",
]
