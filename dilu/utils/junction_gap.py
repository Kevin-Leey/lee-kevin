"""Junction gap acceptance shared by routing and safety filtering."""

import math
from typing import Any, Dict, Iterable, List

from .shared import float_or_default


LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4


def assess_junction_gap(state: Any, action: int = FASTER) -> Dict[str, Any]:
    """Summarize occupancy, priority, and clearance conditions at a junction."""
    scenario = _scenario_key(_get(state, "scenario_type", "env_type", default=""))
    if scenario not in {"intersection", "roundabout"}:
        return _empty_decision(scenario)

    speed = max(0.0, float_or_default(_get(state, "ego_speed", "speed", default=0.0), 0.0))
    front = float_or_default(_get(state, "front_distance", "front_dist", default=float("inf")), float("inf"))
    ttc = float_or_default(_get(state, "ttc", default=float("inf")), float("inf"))
    if ttc <= 0.0:
        ttc = float("inf")
    ego_x = float_or_default(_get(state, "ego_x", default=None), float("nan"))
    ego_y = float_or_default(_get(state, "ego_y", default=None), float("nan"))
    in_junction = _in_junction_box(scenario, ego_x, ego_y)
    in_exit_zone = in_junction and not _in_core_junction_box(scenario, ego_x, ego_y)
    recent_wait = _recent_low_speed_wait(_get(state, "history_frames", default=[]))
    front_blocked = bool(
        (math.isfinite(front) and front < max(6.0, speed * 1.35))
        or (math.isfinite(ttc) and ttc < 2.6)
    )

    active_conflicts: List[Dict[str, Any]] = []
    yield_conflicts: List[Dict[str, Any]] = []
    min_ego_distance = float("inf")
    min_vehicle_time = float("inf")
    min_vehicle_distance = float("inf")
    min_time_gap = float("inf")
    conflict_risk = 0.0
    horizon = 24.0 if scenario == "intersection" else 18.0
    for conflict in _normalise_conflicts(_get(state, "junction_conflicts", default=[])):
        ego_distance = float_or_default(conflict.get("ego_distance"), float("inf"))
        vehicle_distance = float_or_default(conflict.get("vehicle_distance"), float("inf"))
        vehicle_speed = max(0.0, float_or_default(conflict.get("vehicle_speed"), 0.0))
        euclidean = float_or_default(conflict.get("euclidean_distance"), float("inf"))
        if not math.isfinite(ego_distance) or not  -1.0 <= ego_distance <= horizon:
            continue
        if not math.isfinite(vehicle_distance) or vehicle_distance < -4.0:
            continue

        vehicle_time = vehicle_distance / max(vehicle_speed, 0.5)
        ego_time = _ego_time_to_conflict(ego_distance, speed, int(action))
        time_gap = vehicle_time - ego_time
        item = dict(conflict, vehicle_time=float(vehicle_time), ego_time=float(ego_time), time_gap=float(time_gap))
        active_conflicts.append(item)
        min_ego_distance = min(min_ego_distance, ego_distance)
        min_vehicle_time = min(min_vehicle_time, vehicle_time)
        min_vehicle_distance = min(min_vehicle_distance, max(0.0, vehicle_distance))
        min_time_gap = min(min_time_gap, time_gap)

        immediate = bool(
            math.isfinite(euclidean)
            and euclidean <= (8.5 if scenario == "intersection" else 7.0)
            and vehicle_distance <= 6.0
        )
        overlap = abs(time_gap) <= (2.0 if scenario == "intersection" else 1.6)
        claims_gap = bool(
            0.0 <= vehicle_time <= (3.4 if scenario == "intersection" else 2.6)
            and ego_time >= vehicle_time - 0.4
        )
        if immediate or overlap or claims_gap:
            yield_conflicts.append(item)
            conflict_risk = max(
                conflict_risk,
                0.72,
                1.0 - min(1.0, max(0.0, vehicle_time) / (4.0 if scenario == "intersection" else 3.2)),
            )
        else:
            conflict_risk = max(conflict_risk, max(0.0, 0.42 - max(0.0, time_gap) * 0.05))

    occupied = _occupied_corridor(state, scenario)
    scalar_cross = float_or_default(
        _get(state, "cross_traffic_distance", "cross_traffic_dist", default=float("inf")),
        float("inf"),
    )
    scalar_pressure = bool(math.isfinite(scalar_cross) and scalar_cross < (4.2 if scenario == "intersection" else 3.5))
    reservation_yield = bool((yield_conflicts or occupied or scalar_pressure) and not front_blocked)
    creep_allowed = bool(
        reservation_yield
        and not occupied
        and not scalar_pressure
        and speed <= (0.8 if scenario == "intersection" else 1.0)
        and recent_wait >= 2
        and math.isfinite(min_ego_distance)
        and min_ego_distance >= (12.0 if scenario == "intersection" else 10.0)
        and (not math.isfinite(min_vehicle_distance) or min_vehicle_distance >= 3.0)
    )
    should_yield = bool(reservation_yield and not creep_allowed)
    if occupied:
        conflict_risk = max(conflict_risk, 0.90)
    if scalar_pressure:
        conflict_risk = max(conflict_risk, 0.70)

    clearance_pressure = _clearance_pressure(state, speed)
    should_clear = bool(
        speed <= (10.0 if scenario == "intersection" else 4.5)
        and not front_blocked
        and (
            in_junction
            or recent_wait >= 2
            or clearance_pressure
            or (math.isfinite(min_ego_distance) and min_ego_distance <= 12.0)
        )
    )
    return {
        "scenario": scenario,
        "has_conflicts": bool(active_conflicts),
        "active_conflict_count": len(active_conflicts),
        "yield_conflict_count": len(yield_conflicts),
        "reservation_yield": reservation_yield,
        "creep_allowed": creep_allowed,
        "should_yield": should_yield,
        "can_accept_gap": bool(not should_yield and not front_blocked),
        "should_clear": should_clear,
        "front_blocked": front_blocked,
        "occupied_corridor": occupied,
        "clearance_pressure": clearance_pressure,
        "recent_low_speed_wait": recent_wait,
        "in_junction": in_junction,
        "in_exit_zone": in_exit_zone,
        "min_ego_distance": float(min_ego_distance),
        "min_vehicle_time": float(min_vehicle_time),
        "min_vehicle_distance": float(min_vehicle_distance),
        "min_time_gap": float(min_time_gap),
        "conflict_risk": float(min(1.0, max(0.0, conflict_risk))),
        "effective_cross_distance": _effective_cross_distance(
            scalar_cross, should_yield, occupied, min_ego_distance
        ),
    }


def _normalise_conflicts(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, dict)):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _ego_time_to_conflict(distance: float, speed: float, action: int) -> float:
    distance = max(0.0, float(distance))
    if distance == 0.0:
        return 0.0
    speed = max(0.0, float(speed))
    if action == FASTER:
        acceleration = 2.0
        speed = max(speed, 0.35)
    elif action == SLOWER:
        acceleration = -1.4
    else:
        acceleration = 0.35 if speed < 1.0 else 0.0
    if acceleration == 0.0:
        return distance / max(speed, 0.35)
    if acceleration < 0.0 and speed * speed / (2.0 * abs(acceleration)) <= distance:
        return float("inf")
    discriminant = speed * speed + 2.0 * acceleration * distance
    return float("inf") if discriminant < 0.0 else max(0.0, (-speed + math.sqrt(discriminant)) / acceleration)


def _effective_cross_distance(scalar_cross: float, should_yield: bool, occupied: bool, min_ego_distance: float) -> float:
    if occupied:
        return 0.0
    if should_yield:
        reference = min_ego_distance if math.isfinite(min_ego_distance) else scalar_cross
        return min(1.0, max(0.0, reference))
    return float(scalar_cross)


def _occupied_corridor(state: Any, scenario: str) -> bool:
    closest = float_or_default(_get(state, "closest_vehicle_distance", default=float("inf")), float("inf"))
    longitudinal = float_or_default(_get(state, "closest_vehicle_longitudinal", default=float("inf")), float("inf"))
    lateral = abs(float_or_default(_get(state, "closest_vehicle_lateral", default=float("inf")), float("inf")))
    if not all(math.isfinite(value) for value in (closest, longitudinal, lateral)):
        return False
    return bool(-1.0 <= longitudinal <= 9.0 and lateral <= (2.8 if scenario == "intersection" else 2.4) and closest <= 9.5)


def _clearance_pressure(state: Any, speed: float) -> bool:
    closest = float_or_default(_get(state, "closest_vehicle_distance", default=float("inf")), float("inf"))
    longitudinal = float_or_default(_get(state, "closest_vehicle_longitudinal", default=float("inf")), float("inf"))
    lateral = abs(float_or_default(_get(state, "closest_vehicle_lateral", default=float("inf")), float("inf")))
    return bool(math.isfinite(closest) and closest <= 9.0 and (longitudinal <= 1.0 or lateral <= 4.5) and speed <= 2.2)


def _recent_low_speed_wait(history: Any) -> int:
    if not isinstance(history, Iterable) or isinstance(history, (str, bytes, dict)):
        return 0
    return sum(
        1
        for frame in list(history)[-6:]
        if isinstance(frame, dict)
        and max(0.0, float_or_default(frame.get("speed"), 0.0)) <= 1.0
        and int(frame.get("action", -1) or -1) in {IDLE, SLOWER}
    )


def _in_junction_box(scenario: str, x: float, y: float) -> bool:
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    limit = 35.0 if scenario == "intersection" else 28.0 if scenario == "roundabout" else 0.0
    return bool(limit and -limit <= x <= limit and -limit <= y <= limit)


def _in_core_junction_box(scenario: str, x: float, y: float) -> bool:
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    limit = 22.0 if scenario == "intersection" else 18.0 if scenario == "roundabout" else 0.0
    return bool(limit and -limit <= x <= limit and -limit <= y <= limit)


def _scenario_key(value: Any) -> str:
    text = str(value or "").lower()
    if "roundabout" in text:
        return "roundabout"
    if "intersection" in text:
        return "intersection"
    if "merge" in text:
        return "merge"
    if "highway" in text:
        return "highway"
    return text.split("-")[0]


def _get(state: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = state.get(key, None) if isinstance(state, dict) else getattr(state, key, None)
        if value is not None:
            return value
    return default


def _empty_decision(scenario: str) -> Dict[str, Any]:
    return {
        "scenario": scenario,
        "has_conflicts": False,
        "active_conflict_count": 0,
        "yield_conflict_count": 0,
        "reservation_yield": False,
        "creep_allowed": False,
        "should_yield": False,
        "can_accept_gap": True,
        "should_clear": False,
        "front_blocked": False,
        "occupied_corridor": False,
        "clearance_pressure": False,
        "recent_low_speed_wait": 0,
        "in_junction": False,
        "in_exit_zone": False,
        "min_ego_distance": float("inf"),
        "min_vehicle_time": float("inf"),
        "min_vehicle_distance": float("inf"),
        "min_time_gap": float("inf"),
        "conflict_risk": 0.0,
        "effective_cross_distance": float("inf"),
    }
