"""MetaDrive adapter exposing the project's five-action RGD interface."""

import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dilu.driver_agent.base.state import ActionType
from dilu.metadrive.published_tasks import PUBLISHED_TASK_MODES, build_published_task_env
from dilu.scenario.env_ids import infer_scenario_type, require_supported_env
from dilu.utils.driving import ACTIONS_DESCRIPTION, safe_accel
from dilu.utils.shared import float_or_default

logger = logging.getLogger(__name__)


METADRIVE_MAPS = {
    "metadrive-highway-v0": "SS",
    "metadrive-merge-v0": "r",
    "metadrive-intersection-v0": "X",
    "metadrive-roundabout-v0": "O",
    "metadrive-pedestrian-crossing-v0": "S",
    "metadrive-uncontrolled-intersection-v0": "X",
    "metadrive-integrated-traffic-v0": "X",
}

METADRIVE_LANES = {
    "metadrive-highway-v0": 4,
    "metadrive-merge-v0": 3,
    "metadrive-intersection-v0": 2,
    "metadrive-roundabout-v0": 2,
    "metadrive-pedestrian-crossing-v0": 2,
    "metadrive-uncontrolled-intersection-v0": 2,
    "metadrive-integrated-traffic-v0": 2,
}

_ACTION_TO_CONTROL = {
    int(ActionType.LANE_LEFT): (0.55, 0.15),
    int(ActionType.IDLE): (0.0, 0.0),
    int(ActionType.LANE_RIGHT): (-0.55, 0.15),
    int(ActionType.FASTER): (0.0, 0.85),
    int(ActionType.SLOWER): (0.0, -0.85),
}

NO_NEARBY_VEHICLES = "There are no other vehicles driving near you, so you can drive completely according to your own ideas.\n"
VEHICLE_PREFIX = "There are other vehicles driving around you, and below is their basic information:\n"


def build_metadrive_env_config(cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Build the MetaDriveEnv config for one formal experiment row."""
    env_type = require_supported_env(str(cfg.get("env_type", "") or ""))
    if env_type not in METADRIVE_MAPS:
        raise ValueError(f"not a MetaDrive env id: {env_type!r}")

    md_cfg = dict(cfg.get("metadrive_eval", {}) or {})
    scenario_seed_base = int(md_cfg.get("metadrive_start_seed_base", 0) or 0)
    scenario_seed = scenario_seed_base + int(seed)
    num_scenarios = max(int(md_cfg.get("metadrive_num_scenarios", 30) or 30), int(seed) + 1)
    horizon = int(md_cfg.get("metadrive_horizon", cfg.get("simulation_duration", 120)) or 120)
    lane_num = int(md_cfg.get("metadrive_lane_num", cfg.get("lanes_count", METADRIVE_LANES[env_type])) or METADRIVE_LANES[env_type])
    density_value = md_cfg.get("metadrive_traffic_density", cfg.get("vehicles_density", 0.1))
    traffic_density = 0.1 if density_value is None else float(density_value)
    spawn_speed = float(md_cfg.get("metadrive_spawn_speed", cfg.get("initial_speed", 8.0)) or 8.0)

    env_config = {
        "use_render": bool(str(cfg.get("render_mode", "") or "").strip().lower() not in {"", "none", "off", "false"}),
        "force_destroy": True,
        "log_level": int(md_cfg.get("metadrive_log_level", 50) or 50),
        "discrete_action": False,
        "action_check": True,
        "horizon": horizon,
        "start_seed": scenario_seed_base,
        "num_scenarios": num_scenarios,
        "map": str(md_cfg.get("metadrive_map", METADRIVE_MAPS[env_type]) or METADRIVE_MAPS[env_type]),
        "traffic_density": traffic_density,
        "random_traffic": bool(md_cfg.get("metadrive_random_traffic", True)),
        "random_spawn_lane_index": bool(md_cfg.get("metadrive_random_spawn_lane_index", True)),
        "traffic_mode": str(md_cfg.get("metadrive_traffic_mode", "trigger") or "trigger"),
        "decision_repeat": int(md_cfg.get("metadrive_decision_repeat", 5) or 5),
        "vehicle_config": {
            "spawn_velocity": [spawn_speed, 0.0],
            "spawn_velocity_car_frame": True,
        },
    }
    if env_type in PUBLISHED_TASK_MODES:
        env_config.pop("map", None)
        env_config.update(
            {
                "map_config": {
                    "type": "block_sequence",
                    "config": METADRIVE_MAPS[env_type],
                    "lane_num": lane_num,
                    "lane_width": float(md_cfg.get("metadrive_lane_width", 3.5) or 3.5),
                    "exit_length": float(md_cfg.get("metadrive_exit_length", 60.0) or 60.0),
                },
            }
        )
    return env_config


def build_metadrive_env(cfg: Dict[str, Any], seed: int):
    """Instantiate the real MetaDrive environment and wrap it with RGD actions."""
    try:
        from metadrive import MetaDriveEnv
    except ImportError as exc:
        raise ImportError("MetaDrive formal rows require the conda env with metadrive installed; use `conda run -n dilu1 ...`.") from exc

    env_config = build_metadrive_env_config(cfg, seed)
    env_type = require_supported_env(str(cfg.get("env_type", "") or ""))
    md_cfg = dict(cfg.get("metadrive_eval", {}) or {})
    scenario_seed_base = int(md_cfg.get("metadrive_start_seed_base", 0) or 0)
    lane_num = int(md_cfg.get("metadrive_lane_num", cfg.get("lanes_count", METADRIVE_LANES[env_type])) or METADRIVE_LANES[env_type])
    adapter_config = dict(env_config)
    adapter_config.update({
        "_dilu_env_type": env_type,
        "_dilu_scenario_seed": scenario_seed_base + int(seed),
        "_dilu_lane_num": lane_num,
        "_dilu_smooth_longitudinal_control": bool(
            md_cfg.get("metadrive_smooth_longitudinal_control", False)
        ),
        "_dilu_longitudinal_target_step_mps": float(
            md_cfg.get("metadrive_longitudinal_target_step_mps", 1.0) or 1.0
        ),
        "_dilu_longitudinal_brake_step_mps": float(
            md_cfg.get("metadrive_longitudinal_brake_step_mps", 2.0) or 2.0
        ),
        "_dilu_longitudinal_throttle_rate_limit": float(
            md_cfg.get("metadrive_longitudinal_throttle_rate_limit", 0.14) or 0.14
        ),
        "_dilu_longitudinal_brake_gain": float(
            md_cfg.get("metadrive_longitudinal_brake_gain", 0.09) or 0.09
        ),
        "_dilu_longitudinal_max_brake_control": float(
            md_cfg.get("metadrive_longitudinal_max_brake_control", 0.22) or 0.22
        ),
        "_dilu_target_speed_min": float(cfg.get("target_speed_min", 0.0) or 0.0),
        "_dilu_target_speed_max": float(cfg.get("target_speed_max", 11.1) or 11.1),
    })
    raw_env = (
        build_published_task_env(env_type, env_config)
        if env_type in PUBLISHED_TASK_MODES
        else MetaDriveEnv(env_config)
    )
    return MetaDriveDiscreteAdapter(raw_env, adapter_config)


class MetaDriveDiscreteAdapter:
    """Gym-like wrapper mapping RGD's 0..4 actions onto MetaDrive continuous control."""

    def __init__(self, env: Any, config: Dict[str, Any]) -> None:
        self.env = env
        self.config = dict(config)
        self.env_type = str(config["_dilu_env_type"])
        self.spec = SimpleNamespace(id=self.env_type)
        self._last_info: Dict[str, Any] = {}
        self._target_speed_mps = 0.0
        self._last_throttle = 0.0

    @property
    def unwrapped(self):
        return self

    @property
    def vehicle(self):
        return self.env.vehicle

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def current_map(self):
        return self.env.current_map

    @property
    def engine(self):
        return self.env.engine

    def reset(self, seed: Optional[int] = None):
        scenario_seed = int(self.config["_dilu_scenario_seed"] if seed is None else self.config["start_seed"] + int(seed))
        obs, info = self.env.reset(seed=scenario_seed)
        self._target_speed_mps = float(getattr(self.vehicle, "speed", 0.0) or 0.0)
        self._last_throttle = 0.0
        info = dict(info or {})
        info["dilu_scenario_seed"] = scenario_seed
        self._last_info = info
        return obs, info

    def step(self, action: int):
        requested_action = int(action)
        effective_action = self.sanitize_action(requested_action)
        control = self._control_for_action(effective_action)
        obs, reward, term, trunc, info = self.env.step(control)
        info = dict(info or {})
        info["dilu_requested_action"] = requested_action
        info["dilu_effective_action"] = effective_action
        info["dilu_action_rewritten"] = bool(requested_action != effective_action)
        info["dilu_control_steering"] = float(control[0])
        info["dilu_control_throttle"] = float(control[1])
        info["dilu_target_speed_mps"] = float(self._target_speed_mps)
        self._last_info = info
        return obs, reward, term, trunc, info

    def close(self) -> None:
        self.env.close()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def get_available_actions(self) -> List[int]:
        actions = [int(ActionType.IDLE), int(ActionType.FASTER), int(ActionType.SLOWER)]
        lane_id, lane_count = self._lane_position()
        if lane_count > 1:
            if lane_id > 0:
                actions.append(int(ActionType.LANE_LEFT))
            if lane_id < lane_count - 1:
                actions.append(int(ActionType.LANE_RIGHT))
        return sorted(actions)

    def sanitize_action(self, action: int) -> int:
        """Return an action legal at the current state without hiding rewrites."""
        action_id = int(action)
        available = set(self.get_available_actions())
        # The runtime's hidden-slower bridge intentionally exposes braking even
        # when MetaDrive's lane-action query omits it; the continuous adapter
        # can execute that command directly.
        if action_id == int(ActionType.SLOWER):
            return action_id
        if action_id in available:
            return action_id
        idle = int(ActionType.IDLE)
        if idle in available:
            return idle
        if not available:
            raise ValueError("MetaDrive exposes no legal RGD action")
        return int(next(iter(sorted(available))))

    def _control_for_action(self, action: int) -> np.ndarray:
        action_id = int(action)
        if action_id not in _ACTION_TO_CONTROL:
            raise ValueError(f"invalid RGD action for MetaDrive: {action!r}")
        # Fall back when a discrete lane-change is requested off the road edge.
        action_id = self.sanitize_action(action_id)
        steering, throttle = _ACTION_TO_CONTROL[action_id]
        speed = float(getattr(self.vehicle, "speed", 0.0))
        env_type = self.env_type
        is_junction = env_type in {"metadrive-intersection-v0", "metadrive-roundabout-v0"}
        # Non-lane-change actions: auto-steer to follow the road
        if action_id in {int(ActionType.FASTER), int(ActionType.IDLE), int(ActionType.SLOWER)}:
            auto_steering = self._lane_follow_steering()
            if auto_steering is not None:
                steering = auto_steering
        # Junction speed control
        if is_junction and action_id == int(ActionType.FASTER) and speed > 10.0:
            throttle = 0.35
        if bool(self.config.get("_dilu_smooth_longitudinal_control", False)):
            target_min = float(self.config.get("_dilu_target_speed_min", 0.0) or 0.0)
            target_max = float(self.config.get("_dilu_target_speed_max", 11.1) or 11.1)
            if action_id == int(ActionType.FASTER):
                self._target_speed_mps = max(self._target_speed_mps, speed) + float(
                    self.config.get("_dilu_longitudinal_target_step_mps", 1.0) or 1.0
                )
            elif action_id == int(ActionType.SLOWER):
                self._target_speed_mps = min(self._target_speed_mps, speed) - float(
                    self.config.get("_dilu_longitudinal_brake_step_mps", 2.0) or 2.0
                )
            elif action_id == int(ActionType.IDLE):
                # The source action ``Zero`` denotes zero requested
                # acceleration, not continued pursuit of a stale target.
                self._target_speed_mps = speed
            self._target_speed_mps = float(
                np.clip(self._target_speed_mps, target_min, target_max)
            )
            speed_error = float(self._target_speed_mps - speed)
            if speed_error < 0.0:
                brake_gain = max(
                    0.01,
                    float(
                        self.config.get("_dilu_longitudinal_brake_gain", 0.09)
                        or 0.09
                    ),
                )
                max_brake_control = max(
                    0.05,
                    float(
                        self.config.get(
                            "_dilu_longitudinal_max_brake_control", 0.22
                        )
                        or 0.22
                    ),
                )
                desired_throttle = float(
                    np.clip(brake_gain * speed_error, -max_brake_control, 0.0)
                )
            else:
                desired_throttle = float(np.clip(0.22 * speed_error, 0.0, 0.45))
            nominal_rate_limit = max(
                0.01,
                float(
                    self.config.get(
                        "_dilu_longitudinal_throttle_rate_limit", 0.14
                    )
                    or 0.14
                ),
            )
            rate_limit = (
                max(nominal_rate_limit, 1.0)
                if action_id == int(ActionType.SLOWER)
                else nominal_rate_limit
            )
            throttle = float(
                np.clip(
                    desired_throttle,
                    self._last_throttle - rate_limit,
                    self._last_throttle + rate_limit,
                )
            )
            self._last_throttle = throttle
        return np.array([steering, throttle], dtype=np.float32)

    def _lane_follow_steering(self) -> Optional[float]:
        """Compute a steering correction that follows the current road geometry."""
        vehicle = self.vehicle
        navigation = getattr(vehicle, "navigation", None)
        if navigation is None:
            return None
        try:
            heading = float(getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0)))
            cl = navigation.current_lane
            if type(cl).__name__ == "CircularLane":
                # Curved roads: aim at a point ahead on the current lane.
                lon, _lat = cl.local_coordinates(vehicle.position)
                target_lon = min(max(0.0, lon) + 15.0, cl.length)
                target_pos = cl.position(target_lon, 0)
                dx = float(target_pos[0]) - float(vehicle.position[0])
                dy = float(target_pos[1]) - float(vehicle.position[1])
                if abs(dx) < 0.1 and abs(dy) < 0.1:
                    return None
                target_heading = math.atan2(dy, dx)
                angle_diff = target_heading - heading
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                return max(-0.55, min(0.55, angle_diff * 0.7))
            else:
                # Straight roads: align with the current lane heading.
                lon, _lat = cl.local_coordinates(vehicle.position)
                lane_heading = cl.heading_theta_at(max(0.0, lon))
                angle_diff = lane_heading - heading
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                return max(-0.55, min(0.55, angle_diff))
        except (ValueError, TypeError, AttributeError): return None

    def _lane_position(self) -> Tuple[int, int]:
        lane_index = getattr(self.vehicle, "lane_index", None)
        lane_count = int(self.config.get("_dilu_lane_num", 1) or 1)
        if lane_index is None or len(lane_index) < 3:
            return 0, max(lane_count, 1)
        lane_id = int(lane_index[2])
        navigation = getattr(self.vehicle, "navigation", None)
        if navigation is not None and hasattr(navigation, "get_current_lane_num"):
            lane_count = int(max(1, round(float(navigation.get_current_lane_num()))))
        return max(0, min(lane_id, lane_count - 1)), max(lane_count, 1)


class MetaDriveScenario:
    """Scenario wrapper for MetaDrive scene descriptions and geometry probes."""

    def __init__(self, env: MetaDriveDiscreteAdapter, env_type: str, seed: int, database: str = None) -> None:
        self.env = env
        self.envType = require_supported_env(env_type)
        self.seed = int(seed)
        self.database = database
        self.ego = env.vehicle
        self.scenario_type = infer_scenario_type(env_type)
        self.network = getattr(env.current_map, "road_network", None)

    def getSurroundingVehicles(self, vehicles_count: int) -> List[Any]:
        count = max(0, int(vehicles_count))
        traffic_mgr = getattr(self.env.engine, "traffic_manager", None)
        all_vehicles = list(getattr(traffic_mgr, "vehicles", []) if traffic_mgr is not None else [])
        task_mgr = getattr(self.env.engine, "published_task_manager", None)
        all_vehicles.extend(list(getattr(task_mgr, "participants", []) if task_mgr is not None else []))
        all_vehicles = list({id(vehicle): vehicle for vehicle in all_vehicles}.values())
        ego = self.ego
        candidates = [v for v in all_vehicles if v is not ego and hasattr(v, "position") and hasattr(v, "speed")]
        candidates.sort(key=self.getVehDis)
        vehicles = candidates[:count]
        interactive_envs = {
            "metadrive-intersection-v0",
            "metadrive-roundabout-v0",
            *PUBLISHED_TASK_MODES,
        }
        if self.envType not in interactive_envs:
            return vehicles
        threshold = 35.0 if self.envType == "metadrive-roundabout-v0" else 45.0
        seen = {id(v) for v in vehicles}
        cross: List[Tuple[float, Any]] = []
        for v in candidates:
            if id(v) in seen:
                continue
            distance = self.getVehDis(v)
            if distance > threshold:
                continue
            if self.getCollisionPoint(v) is not None or distance <= threshold * 0.6:
                cross.append((distance, v))
        cross.sort(key=lambda item: item[0])
        vehicles.extend(v for _, v in cross[: max(2, min(4, count // 2))])
        vehicles.sort(key=self.getVehDis)
        return vehicles[:count]

    def availableActionsDescription(self) -> str:
        lines = ["Your available actions are: "]
        lines.extend(f"{ACTIONS_DESCRIPTION[int(action)]} Action_id: {int(action)}" for action in self.env.get_available_actions())
        return "\n".join(lines) + "\n"

    def describe(self, decisionFrame: int, surroundVehicles: Optional[List[Any]] = None) -> str:
        vehicles = self.getSurroundingVehicles(10) if surroundVehicles is None else list(surroundVehicles)
        lane_id, lane_count = self._lane_position(self.ego)
        lane_text = self._lane_text(lane_id, lane_count)
        road = (
            f"You are driving in a MetaDrive {self.scenario_type} scenario. "
            f"{lane_text}Your current position is `({self.ego.position[0]:.2f}, {self.ego.position[1]:.2f})`, "
            f"speed is {float_or_default(getattr(self.ego, 'speed', 0.0), 0.0):.2f} m/s, "
            f"acceleration is {safe_accel(self.ego):.2f} m/s^2.\n"
        )
        return road + self._describe_vehicles(vehicles)

    def getVehDis(self, veh: Any) -> float:
        return float(np.linalg.norm(np.array(self.ego.position, dtype=float) - np.array(veh.position, dtype=float)))

    def getUnitVector(self, radian: float) -> Tuple[float, float]:
        return math.cos(float(radian)), math.sin(float(radian))

    def getCollisionPoint(self, sv: Any) -> Optional[Tuple[float, float]]:
        ex, ey = self.ego.position
        sx, sy = sv.position
        edx, edy = self.getUnitVector(self._heading(self.ego))
        sdx, sdy = self.getUnitVector(self._heading(sv))
        denom = edx * (-sdy) - edy * (-sdx)
        if abs(denom) < 1e-8:
            return None
        dx = sx - ex
        dy = sy - ey
        t = (dx * (-sdy) - dy * (-sdx)) / denom
        u = (edx * dy - edy * dx) / denom
        if t < 0.0 or u < 0.0:
            return None
        return ex + edx * t, ey + edy * t

    def _describe_vehicles(self, vehicles: Sequence[Any]) -> str:
        if not vehicles:
            return NO_NEARBY_VEHICLES
        ego_heading = self._heading(self.ego)
        ego_dir = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
        rows = []
        for participant_index, vehicle in enumerate(vehicles[:10]):
            rel = np.array(vehicle.position, dtype=float) - np.array(self.ego.position, dtype=float)
            longitudinal = float(np.dot(rel, ego_dir))
            lateral = float(rel[0] * ego_dir[1] - rel[1] * ego_dir[0])
            relation = "ahead of you" if longitudinal >= 0.0 else "behind you"
            lane_relation = self._lane_relation(vehicle, lateral)
            participant = "Pedestrian" if type(vehicle).__name__.lower().startswith("pedestrian") else "Vehicle"
            raw_identifier = getattr(vehicle, "id", None)
            stable_identifier = (
                str(raw_identifier)
                if raw_identifier is not None
                else f"scene-{participant_index}"
            )
            rows.append(
                f"- {participant} `{stable_identifier}` is {lane_relation} and {relation}. "
                f"The position of it is `({vehicle.position[0]:.2f}, {vehicle.position[1]:.2f})`, "
                f"speed is {float_or_default(getattr(vehicle, 'speed', 0.0), 0.0):.2f} m/s, "
                f"acceleration is {safe_accel(vehicle):.2f} m/s^2.\n"
            )
        return VEHICLE_PREFIX + "".join(rows)

    def _lane_relation(self, vehicle: Any, lateral: float) -> str:
        ego_lane = getattr(self.ego, "lane_index", None)
        vehicle_lane = getattr(vehicle, "lane_index", None)
        if (
            ego_lane is not None
            and vehicle_lane is not None
            and len(ego_lane) >= 3
            and len(vehicle_lane) >= 3
            and tuple(ego_lane[:2]) == tuple(vehicle_lane[:2])
        ):
            diff = int(vehicle_lane[2]) - int(ego_lane[2])
            if diff == 0:
                return "driving on the same lane as you"
            if diff < 0:
                return "driving on the lane to your left"
            if diff > 0:
                return "driving on the lane to your right"
        if lateral < -2.0:
            return "to your left"
        if lateral > 2.0:
            return "to your right"
        return "near your driving corridor"

    def _lane_text(self, lane_id: int, lane_count: int) -> str:
        if lane_count <= 1:
            return "You are driving on a road with only one lane, you can't change lane. "
        if lane_id == 0:
            return f"You are driving on a road with {lane_count} lanes, and you are currently driving in the leftmost lane. "
        if lane_id == lane_count - 1:
            return f"You are driving on a road with {lane_count} lanes, and you are currently driving in the rightmost lane. "
        return f"You are driving on a road with {lane_count} lanes, and you are currently driving in lane {lane_id}. "

    def _lane_position(self, vehicle: Any) -> Tuple[int, int]:
        lane_index = getattr(vehicle, "lane_index", None)
        lane_count = int(getattr(self.env, "config", {}).get("_dilu_lane_num", 1) or 1)
        navigation = getattr(vehicle, "navigation", None)
        if navigation is not None and hasattr(navigation, "get_current_lane_num"):
            lane_count = int(max(1, round(float(navigation.get_current_lane_num()))))
        if lane_index is None or len(lane_index) < 3:
            return 0, max(lane_count, 1)
        return max(0, min(int(lane_index[2]), lane_count - 1)), max(lane_count, 1)

    @staticmethod
    def _heading(vehicle: Any) -> float:
        if hasattr(vehicle, "heading_theta"):
            value = float_or_default(getattr(vehicle, "heading_theta", None), float("nan"))
            if math.isfinite(value):
                return value
        heading = getattr(vehicle, "heading", None)
        if isinstance(heading, (int, float, np.floating)):
            return float(heading)
        if hasattr(heading, "x") and hasattr(heading, "y"):
            return float(math.atan2(float(heading.y), float(heading.x)))
        if heading is not None:
            values = list(heading)
            if len(values) >= 2:
                return float(math.atan2(float(values[1]), float(values[0])))
        raise ValueError(f"cannot resolve MetaDrive vehicle heading from {heading!r}")
