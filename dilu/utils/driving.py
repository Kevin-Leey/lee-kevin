"""Common action labels and vehicle kinematic helpers."""

from typing import Any, Dict

from .shared import safe_float


ACTIONS_ALL: Dict[int, str] = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

ACTIONS_DESCRIPTION: Dict[int, str] = {
    0: "Turn-left - change lane to the left of the current lane",
    1: "IDLE - remain in the current lane with current speed",
    2: "Turn-right - change lane to the right of the current lane",
    3: "Acceleration - accelerate the vehicle",
    4: "Deceleration - decelerate the vehicle",
}


def safe_accel(vehicle: Any, default: float = 0.0) -> float:
    """Read the longitudinal acceleration from highway-env or MetaDrive state."""
    action = getattr(vehicle, "action", None)
    if isinstance(action, dict) and "acceleration" in action:
        return safe_float(action["acceleration"], default)
    if hasattr(vehicle, "acceleration"):
        return safe_float(getattr(vehicle, "acceleration"), default)
    return float(default)
