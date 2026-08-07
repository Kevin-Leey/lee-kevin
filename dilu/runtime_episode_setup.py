"""Episode setup helpers for the narrowed RGD runtime."""

import atexit
import hashlib
import json
import logging
import math
import os
import random
from typing import Any, Dict, List

import gymnasium as gym
import highway_env
import numpy as np

from dilu.evaluation.physical_metrics import PhysicalMetricsRecorder
from dilu.metadrive import build_metadrive_env
from dilu.evaluation.reasoning_recorder import ReasoningRecorder
from dilu.runtime_frame_trace import collect_runtime_integrity
from dilu.runtime_support import _resolve_render_mode
from dilu.scenario.env_ids import infer_env_label, is_metadrive_env, require_highway_env
from dilu.scenario.env_builder import build_env_config, get_env_defaults, resolve_env_value
from dilu.scenario.published_protocol_envs import (
    build_published_highway_env,
    validate_published_protocol_initial_state,
)


logger = logging.getLogger(__name__)
_HIGHWAY_ENV_CACHE: Dict[str, Any] = {}


def _close_cached_highway_envs() -> None:
    """Release cached highway-env instances when the process exits."""
    for env in list(_HIGHWAY_ENV_CACHE.values()):
        env.close()
    _HIGHWAY_ENV_CACHE.clear()


atexit.register(_close_cached_highway_envs)


def _heading_value(vehicle: Any) -> float:
    heading = getattr(vehicle, "heading", None)
    if isinstance(heading, (int, float)):
        return float(heading)
    if hasattr(heading, "x") and hasattr(heading, "y"):
        return float(math.atan2(float(heading.y), float(heading.x)))
    if heading is not None:
        values = list(heading)
        if len(values) >= 2:
            return float(math.atan2(float(values[1]), float(values[0])))
    raise ValueError(f"cannot resolve vehicle heading from {heading!r}")


def _apply_intersection_close_front_start_clamp(env: Any, env_type: str, cfg: Dict[str, Any]) -> None:
    clamp_cfg = dict(cfg.get("intersection_close_front_start_clamp", {}) or {})
    if env_type != "intersection-v0" or not bool(clamp_cfg.get("enable", False)):
        return
    ego = env.unwrapped.vehicle
    vehicles = list(getattr(env.unwrapped.road, "vehicles", []) or [])
    if ego is None:
        return

    ego_x, ego_y = float(ego.position[0]), float(ego.position[1])
    heading = _heading_value(ego)
    dir_x, dir_y = math.cos(heading), math.sin(heading)
    lateral_limit = float(clamp_cfg.get("lateral_m", 2.8) or 2.8)
    min_forward = float("inf")
    for vehicle in vehicles:
        if vehicle is ego or not hasattr(vehicle, "position"):
            continue
        rel_x = float(vehicle.position[0]) - ego_x
        rel_y = float(vehicle.position[1]) - ego_y
        forward = rel_x * dir_x + rel_y * dir_y
        lateral = abs(rel_x * dir_y - rel_y * dir_x)
        if forward <= 0.0 or lateral > lateral_limit:
            continue
        min_forward = min(min_forward, float(math.hypot(rel_x, rel_y)))

    if min_forward < float(clamp_cfg.get("distance_m", 8.0) or 8.0):
        ego.speed = min(float(getattr(ego, "speed", 0.0) or 0.0), float(clamp_cfg.get("speed_mps", 4.0) or 4.0))


def _rebuild_highway_traffic_if_needed(env: Any, env_type: str, cfg: Dict[str, Any]) -> None:
    if env_type != "highway-v0":
        return
    if not bool(cfg.get("highway_rebuild_traffic_if_needed", False)):
        return
    target_count = int(resolve_env_value(cfg, "vehicle_count", env_type, get_env_defaults(env_type)["vehicle_count"]) or 0)
    road = getattr(env.unwrapped, "road", None)
    ego = getattr(env.unwrapped, "vehicle", None)
    if road is None or ego is None or target_count <= 0:
        return
    vehicles = list(getattr(road, "vehicles", []) or [])
    other_count = max(0, len([vehicle for vehicle in vehicles if vehicle is not ego]))
    if other_count >= target_count:
        road.vehicles = [ego] + [vehicle for vehicle in vehicles if vehicle is not ego][:target_count]
        if hasattr(env.unwrapped, "road"):
            env.unwrapped.road.vehicles = road.vehicles
        return
    other_vehicle_type_path = str(getattr(env.unwrapped, "config", {}).get("other_vehicles_type", "highway_env.vehicle.behavior.IDMVehicle"))
    module_name, class_name = other_vehicle_type_path.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    other_vehicle_type = getattr(module, class_name)
    density = max(0.1, float(resolve_env_value(cfg, "vehicles_density", env_type, get_env_defaults(env_type)["vehicles_density"]) or 0.1))
    for _ in range(target_count - other_count):
        vehicle = other_vehicle_type.create_random(road, spacing=1.0 / density)
        if hasattr(vehicle, "randomize_behavior"):
            vehicle.randomize_behavior()
        road.vehicles.append(vehicle)


def _apply_highway_initial_spacing(env: Any, env_type: str, cfg: Dict[str, Any]) -> None:
    if env_type != "highway-v0":
        return
    if not bool(cfg.get("highway_initial_spacing_normalization", False)):
        return
    spacing = float(resolve_env_value(cfg, "ego_spacing", env_type, 4.0) or 4.0)
    if spacing >= 3.5:
        return
    road = getattr(env.unwrapped, "road", None)
    ego = getattr(env.unwrapped, "vehicle", None)
    if road is None or ego is None:
        return
    try:
        ego_lane = road.network.get_lane(ego.lane_index)
        ego_long, _ = ego_lane.local_coordinates(ego.position)
    except (ValueError, TypeError, KeyError, AttributeError):
        return
    min_same_front = 64.0
    min_adjacent_front = 64.0
    lane_fronts: Dict[Any, Dict[str, Any]] = {}
    for vehicle in list(getattr(road, "vehicles", []) or []):
        if vehicle is ego or not hasattr(vehicle, "position") or getattr(vehicle, "lane_index", None) is None:
            continue
        try:
            veh_lane = road.network.get_lane(vehicle.lane_index)
            ego_relative_long, _ = ego_lane.local_coordinates(vehicle.position)
            veh_long, veh_lat = veh_lane.local_coordinates(vehicle.position)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        longitudinal = float(ego_relative_long - ego_long)
        if longitudinal <= 0.0:
            continue
        lane_delta = int(vehicle.lane_index[2]) - int(ego.lane_index[2])
        if lane_delta != 0 and abs(lane_delta) != 1:
            continue
        target_front = min_same_front if lane_delta == 0 else min_adjacent_front
        lane_key = tuple(vehicle.lane_index)
        lane_front = lane_fronts.get(lane_key)
        if lane_front is None or longitudinal < float(lane_front["longitudinal"]):
            lane_fronts[lane_key] = {
                "longitudinal": longitudinal,
                "target_front": target_front,
            }

    lane_offsets = {
        lane_key: float(lane_front["target_front"]) - float(lane_front["longitudinal"])
        for lane_key, lane_front in lane_fronts.items()
        if float(lane_front["longitudinal"]) < float(lane_front["target_front"])
    }
    if not lane_offsets:
        return

    for vehicle in list(getattr(road, "vehicles", []) or []):
        if vehicle is ego or not hasattr(vehicle, "position") or getattr(vehicle, "lane_index", None) is None:
            continue
        lane_key = tuple(vehicle.lane_index)
        offset = lane_offsets.get(lane_key)
        if offset is None:
            continue
        try:
            veh_lane = road.network.get_lane(vehicle.lane_index)
            ego_relative_long, _ = ego_lane.local_coordinates(vehicle.position)
            veh_long, veh_lat = veh_lane.local_coordinates(vehicle.position)
        except (ValueError, TypeError, KeyError, AttributeError):
            continue
        if float(ego_relative_long - ego_long) <= 0.0:
            continue
        vehicle.position = veh_lane.position(veh_long + offset, veh_lat)


def _sync_controlled_vehicle_speed_target(vehicle: Any) -> None:
    if vehicle is None:
        return
    target_speeds = getattr(vehicle, "target_speeds", None)
    if target_speeds is None or len(target_speeds) == 0:
        return
    speed = float(getattr(vehicle, "speed", 0.0) or 0.0)
    if hasattr(vehicle, "speed_to_index") and hasattr(vehicle, "index_to_speed"):
        speed_index = int(vehicle.speed_to_index(speed))
        vehicle.speed_index = speed_index
        vehicle.target_speed = float(vehicle.index_to_speed(speed_index))
        return
    nearest_index = min(range(len(target_speeds)), key=lambda idx: abs(float(target_speeds[idx]) - speed))
    vehicle.speed_index = int(nearest_index)
    vehicle.target_speed = float(target_speeds[nearest_index])


def _highway_env_cache_key(env_type: str, cfg: Dict[str, Any]) -> str:
    """Build a stable cache key for headless highway-env instances."""
    payload = {
        "env_type": str(env_type or "highway-v0"),
        "env_config": build_env_config(cfg),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _acquire_highway_env(env_type: str, cfg: Dict[str, Any]):
    """Reuse headless highway-env instances across episodes with identical config."""
    render_mode = _resolve_render_mode(cfg)
    env_config = build_env_config(cfg)
    use_protocol_variant = bool(cfg.get("published_protocol_env")) and env_type in {
        "intersection-v0",
        "merge-v0",
        "roundabout-v0",
    }
    if render_mode is not None:
        if use_protocol_variant:
            env = build_published_highway_env(
                env_type, env_config, render_mode=render_mode
            )
        else:
            env = gym.make(env_type, render_mode=render_mode)
            env.unwrapped.configure(env_config)
        return env, True

    cache_key = _highway_env_cache_key(env_type, cfg)
    cached_env = _HIGHWAY_ENV_CACHE.get(cache_key)
    if cached_env is None:
        if use_protocol_variant:
            cached_env = build_published_highway_env(env_type, env_config)
        else:
            cached_env = gym.make(env_type)
            cached_env.unwrapped.configure(env_config)
        _HIGHWAY_ENV_CACHE[cache_key] = cached_env
    return cached_env, False


def create_episode_env(ep: int, cfg: Dict[str, Any], result_dir: str, seed_pool: List[int]):
    """Create and reset the gym environment for one episode."""
    if not seed_pool and cfg.get("fixed_seed_override") is None:
        raise ValueError("seed_pool must not be empty when fixed_seed_override is unset")
    requested_env_type = str(cfg.get("env_type", "highway-v0") or "highway-v0").strip()
    fixed_seed_override = cfg.get("fixed_seed_override")
    seed = int(
        fixed_seed_override
        if fixed_seed_override is not None
        else seed_pool[ep % len(seed_pool)]
    )
    if is_metadrive_env(requested_env_type):
        env_type = requested_env_type
        env = build_metadrive_env(cfg, seed)
        close_env_after_episode = True
    else:
        env_type = require_highway_env(requested_env_type)
        env, close_env_after_episode = _acquire_highway_env(env_type, cfg)
    ep_dir = os.path.join(result_dir, f"ep_{ep}")
    os.makedirs(ep_dir, exist_ok=True)
    prefix = f"{infer_env_label(env_type)}_{ep}"

    # highway-env's traffic reconstruction calls ``create_random`` and
    # ``randomize_behavior`` in a few versions that draw from module-level
    # Python/NumPy RNGs rather than the environment's Generator.  Seed those
    # streams at the episode boundary so a fixed simulator seed determines the
    # complete initial traffic state and can be replayed by the mechanism
    # analyzer in a separate process.
    random.seed(int(seed))
    np.random.seed(int(seed))
    obs, _ = env.reset(seed=seed)
    if (not is_metadrive_env(env_type)) and hasattr(env.unwrapped, "vehicle") and env.unwrapped.vehicle:
        _rebuild_highway_traffic_if_needed(env, env_type, cfg)
        _apply_highway_initial_spacing(env, env_type, cfg)
        env_defaults = get_env_defaults(env_type)
        env_default_speed = env_defaults["initial_speed"]
        cfg_speed = float(resolve_env_value(cfg, "initial_speed", env_type, env_default_speed))
        exact_protocol_speed = bool(cfg.get("protocol_exact_initial_speed", False))
        effective_speed = (
            cfg_speed
            if exact_protocol_speed
            else (
                min(cfg_speed, env_default_speed)
                if env_type in ("intersection-v0", "roundabout-v0")
                else cfg_speed
            )
        )
        if not bool(cfg.get("protocol_randomize_initial_conditions", False)):
            env.unwrapped.vehicle.speed = effective_speed
        _sync_controlled_vehicle_speed_target(env.unwrapped.vehicle)
        _apply_intersection_close_front_start_clamp(env, env_type, cfg)
        if bool(cfg.get("protocol_validate_initial_state", False)):
            validate_published_protocol_initial_state(
                env.unwrapped,
                require_randomized_idm=bool(
                    cfg.get("protocol_randomize_idm_parameters", False)
                ),
            )
    return env, obs, ep_dir, prefix, seed, close_env_after_episode


def _slow_executor_is_required(cfg: Dict[str, Any]) -> bool:
    """Return whether the configured route can invoke the slow executor."""
    if str(cfg.get("agent_type", "llm") or "llm").strip().lower() in {
        "random",
        "idm_only",
    }:
        return False
    routing = dict(cfg.get("system_routing", {}) or {})
    simple = str(routing.get("simple", "") or "").strip().lower()
    complex_route = str(routing.get("complex", "") or "").strip().lower()
    if simple == complex_route == "fast":
        return False
    budget = cfg.get("slow_call_budget")
    if budget is not None and int(budget) <= 0:
        return False
    return True


def _slow_executor_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the declared slow executor only when a route can use it."""
    if not _slow_executor_is_required(cfg):
        return {}
    slow_cfg = dict(cfg.get("slow_thinking", {}) or {})
    executor = str(slow_cfg.get("executor", "online_llm") or "online_llm").strip().lower()
    if executor == "online_llm":
        from dilu.driver_agent.base.llm_factory import LLMFactory
        from dilu.utils.config import setup_api

        try:
            setup_api(cfg)
            max_tokens = int(slow_cfg.get("max_tokens", LLMFactory.PRESETS["slow"]["max_tokens"]) or LLMFactory.PRESETS["slow"]["max_tokens"])
            model = LLMFactory.create(
                "slow", max_tokens=max_tokens, force_new=True
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "online slow executor configuration is incomplete; set the provider model, base URL, and API key"
            ) from exc
        return {"slow_model": model}
    if executor == "kinematic_risk":
        from dilu.driver_agent.reasoning.kinematic_risk import KinematicRiskActionProvider

        return {"slow_action_provider": KinematicRiskActionProvider()}
    if executor in {"disabled", "none"}:
        raise ValueError("the selected route can invoke slow reasoning, but slow_thinking.executor is disabled")
    raise ValueError(f"unsupported slow_thinking.executor: {executor!r}")


def create_episode_agent(sce, cfg: Dict[str, Any], result_dir: str):
    """Create the driver agent for one episode."""
    del result_dir
    try:
        from dilu.driver_agent.driverAgentV2 import DriverAgentV2
    except ImportError as import_err:
        raise ImportError(
            "DriverAgentV2 could not be imported. Check dilu/driver_agent/driverAgentV2.py and its dependencies."
        ) from import_err

    agent = DriverAgentV2(
        sce=sce,
        config=cfg,
        **_slow_executor_kwargs(cfg),
    )
    agent._runtime_integrity_start = collect_runtime_integrity(agent)
    return agent


def create_episode_recorders(ep: int, seed: int, ep_dir: str, cfg: Dict[str, Any]):
    """Create optional per-episode recorders based on the runtime config."""
    env_type = str(cfg.get("env_type", "highway-v0") or "highway-v0")
    policy_frequency = max(
        float(
            resolve_env_value(
                cfg,
                "policy_frequency",
                env_type,
                get_env_defaults(env_type)["policy_frequency"],
            )
            or 1.0
        ),
        1.0,
    )
    step_seconds = 1.0 / policy_frequency
    expected_total_frames = None
    if cfg.get("enable_physical_metrics", False):
        explicit_expected_frames = cfg.get(
            "expected_total_frames", cfg.get("expected_policy_steps")
        )
        if explicit_expected_frames is not None:
            if isinstance(explicit_expected_frames, bool):
                raise ValueError("expected_total_frames must be a positive integer")
            numeric_expected_frames = float(explicit_expected_frames)
            if not math.isfinite(numeric_expected_frames) or not numeric_expected_frames.is_integer():
                raise ValueError("expected_total_frames must be a positive integer")
            expected_total_frames = int(numeric_expected_frames)
        else:
            duration = float(
                resolve_env_value(cfg, "simulation_duration", env_type, 0.0) or 0.0
            )
            expected_total_frames = (
                int(math.ceil(duration - 1e-12))
                if env_type.startswith("metadrive-")
                else int(math.ceil(duration * policy_frequency - 1e-12))
            )
        if expected_total_frames <= 0:
            raise ValueError("physical metrics require a positive expected_total_frames")
    threshold_value = cfg.get("success_completion_threshold", 0.95)
    success_completion_threshold = float(
        0.95 if threshold_value is None else threshold_value
    )
    success_metric_mode = str(cfg.get("success_metric_mode", "completion_threshold") or "completion_threshold")
    physical_recorder = PhysicalMetricsRecorder(
        ep,
        seed,
        ep_dir,
        step_seconds=step_seconds,
        expected_total_frames=expected_total_frames,
        success_completion_threshold=success_completion_threshold,
        success_metric_mode=success_metric_mode,
        env_type=env_type,
        published_reward_vmax_mps=float(
            cfg.get("published_reward_vmax_mps", 0.0) or 0.0
        ),
    ) if cfg.get("enable_physical_metrics", False) else None
    reasoning_recorder = ReasoningRecorder(ep, ep_dir) if cfg.get("enable_reasoning_recording", False) else None
    return physical_recorder, reasoning_recorder
