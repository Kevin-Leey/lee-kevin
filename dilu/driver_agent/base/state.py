"""Shared action and observation types for the RGD driving stack.

The runtime deliberately uses a small discrete action domain.  Every
controller, safety layer, and delayed-response gate receives the same
``DrivingState`` object so an action is interpreted consistently at query and
release time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional


class ActionType(IntEnum):
    """The common five-action driving interface."""

    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

    @classmethod
    def to_english(cls, action: int) -> str:
        return ACTION_NAMES.get(int(action), "UNKNOWN")


ACTIONS_ALL: Dict[int, str] = {
    int(ActionType.LANE_LEFT): "LANE_LEFT",
    int(ActionType.IDLE): "IDLE",
    int(ActionType.LANE_RIGHT): "LANE_RIGHT",
    int(ActionType.FASTER): "FASTER",
    int(ActionType.SLOWER): "SLOWER",
}

ACTION_NAMES: Dict[int, str] = {
    int(ActionType.LANE_LEFT): "lane left",
    int(ActionType.IDLE): "keep lane",
    int(ActionType.LANE_RIGHT): "lane right",
    int(ActionType.FASTER): "accelerate",
    int(ActionType.SLOWER): "decelerate",
}


def _finite_or_inf(value: Any, default: float = math.inf) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@dataclass
class DrivingState:
    """Structured per-frame observation consumed by the RGD stack.

    Distances are expressed in metres and speeds in metres per second.  Missing
    neighbour observations use ``inf`` for distance-like values, which encodes
    an unobserved remote vehicle rather than a fabricated close obstacle.
    """

    ego_speed: float = 0.0
    ego_lane: int = 0
    total_lanes: int = 1
    position: Optional[Any] = None
    heading: float = 0.0
    scenario_type: str = "highway"
    env_type: str = ""

    front_distance: float = math.inf
    front_speed: Optional[float] = None
    front_relative_speed: float = 0.0
    ttc: float = math.inf
    thw: float = math.inf

    left_front_distance: float = math.inf
    left_rear_distance: float = math.inf
    left_front_speed: Optional[float] = None
    left_rear_speed: Optional[float] = None
    right_front_distance: float = math.inf
    right_rear_distance: float = math.inf
    right_front_speed: Optional[float] = None
    right_rear_speed: Optional[float] = None
    can_change_left: bool = False
    can_change_right: bool = False

    closest_vehicle_distance: float = math.inf
    closest_vehicle_longitudinal: float = math.inf
    closest_vehicle_lateral: float = math.inf
    closest_vehicle_closing_speed: float = 0.0
    closest_vehicle_heading_delta: float = 0.0
    front_heading_delta: float = 0.0
    cross_traffic_distance: float = math.inf
    junction_gap: Dict[str, Any] = field(default_factory=dict)

    nearby_vehicle_count: int = 0
    legal_actions: List[int] = field(
        default_factory=lambda: [int(action) for action in ActionType]
    )
    lane_change_safety: Dict[str, bool] = field(default_factory=dict)
    history_frames: List[Dict[str, Any]] = field(default_factory=list)
    effective_action_universe: List[int] = field(default_factory=list)
    effective_action_universe_source: str = "legal_actions"

    def __post_init__(self) -> None:
        self.ego_speed = max(0.0, _finite_or_inf(self.ego_speed, 0.0))
        self.total_lanes = max(1, int(self.total_lanes or 1))
        self.ego_lane = max(0, min(int(self.ego_lane or 0), self.total_lanes - 1))
        self.front_distance = _finite_or_inf(self.front_distance)
        self.left_front_distance = _finite_or_inf(self.left_front_distance)
        self.left_rear_distance = _finite_or_inf(self.left_rear_distance)
        self.right_front_distance = _finite_or_inf(self.right_front_distance)
        self.right_rear_distance = _finite_or_inf(self.right_rear_distance)
        self.closest_vehicle_distance = _finite_or_inf(self.closest_vehicle_distance)
        self.closest_vehicle_longitudinal = _finite_or_inf(
            self.closest_vehicle_longitudinal
        )
        self.closest_vehicle_lateral = _finite_or_inf(self.closest_vehicle_lateral)
        self.cross_traffic_distance = _finite_or_inf(self.cross_traffic_distance)
        self.ttc = _finite_or_inf(self.ttc)
        self.thw = _finite_or_inf(self.thw)
        self.legal_actions = self._normalize_actions(self.legal_actions)
        self.effective_action_universe = self._normalize_actions(
            self.effective_action_universe
        )
        supplied_safety = dict(self.lane_change_safety or {})
        self.lane_change_safety = {
            "left": bool(supplied_safety.get("left", self.can_change_left)),
            "right": bool(supplied_safety.get("right", self.can_change_right)),
        }
        self.history_frames = [dict(frame) for frame in list(self.history_frames or [])]
        self.junction_gap = dict(self.junction_gap or {})

    @staticmethod
    def _normalize_actions(actions: Optional[Iterable[Any]]) -> List[int]:
        if not actions:
            return []
        normalized = sorted({int(action) for action in actions})
        invalid = [action for action in normalized if action not in ACTIONS_ALL]
        if invalid:
            raise ValueError(f"unsupported action identifiers: {invalid}")
        return normalized

    @property
    def lane(self) -> int:
        return int(self.ego_lane)

    @property
    def speed(self) -> float:
        return float(self.ego_speed)

    def get_available_actions(self) -> List[int]:
        """Return the action universe frozen for the current frame."""
        return list(self.effective_action_universe or self.legal_actions)

    def set_effective_action_universe(
        self, actions: Iterable[Any], *, source: str
    ) -> None:
        resolved = self._normalize_actions(actions)
        if not resolved:
            raise ValueError("effective action universe must not be empty")
        self.effective_action_universe = resolved
        self.effective_action_universe_source = str(source or "runtime")

    def clear_effective_action_universe(self) -> None:
        self.effective_action_universe = []
        self.effective_action_universe_source = "legal_actions"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def update(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            if not hasattr(self, key):
                raise KeyError(f"unknown driving-state field: {key}")
            setattr(self, key, value)
        self.__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)
