"""Environment configuration helpers for highway-env scenarios."""

from typing import Any, Dict

import numpy as np


_ENV_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "highway-v0": {"lanes_count": 4, "initial_speed": 26.0, "vehicle_count": 30, "vehicles_density": 2.0, "policy_frequency": 10},
    "merge-v0": {"lanes_count": 4, "initial_speed": 22.0, "vehicle_count": 8, "vehicles_density": 1.8, "policy_frequency": 1},
    "roundabout-v0": {"lanes_count": 2, "initial_speed": 18.0, "vehicle_count": 8, "vehicles_density": 1.5, "policy_frequency": 1},
    "intersection-v0": {"lanes_count": 2, "initial_speed": 8.0, "vehicle_count": 8, "vehicles_density": 1.2, "policy_frequency": 2},
}


def get_env_defaults(env_type: str) -> Dict[str, Any]:
    """Return a copy of the defaults for a supported highway environment."""
    return dict(_ENV_DEFAULTS.get(str(env_type), _ENV_DEFAULTS["highway-v0"]))


def resolve_env_value(cfg: Dict[str, Any], key: str, env_type: str, default: Any) -> Any:
    """Resolve a setting, preferring explicit per-environment overrides."""
    by_env = cfg.get(f"{key}_by_env", {}) or {}
    if isinstance(by_env, dict) and env_type in by_env:
        return by_env[env_type]
    scoped = cfg.get(str(env_type).replace("-", "_") + "_env", {}) or {}
    if isinstance(scoped, dict) and key in scoped:
        return scoped[key]
    return cfg.get(key, default)


def build_env_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a concrete highway-env configuration from an experiment row."""
    env_type = str(cfg.get("env_type", "highway-v0") or "highway-v0")
    defaults = get_env_defaults(env_type)
    vehicle_count = int(resolve_env_value(cfg, "vehicle_count", env_type, defaults["vehicle_count"]))
    speed_min_default = 0.0 if env_type == "intersection-v0" else 10.0
    speed_min = float(resolve_env_value(cfg, "target_speed_min", env_type, speed_min_default))
    speed_max = float(resolve_env_value(cfg, "target_speed_max", env_type, 28.0))
    speed_count = max(2, int(resolve_env_value(cfg, "target_speed_count", env_type, 9) or 9))
    config: Dict[str, Any] = {
        "observation": {
            "type": "Kinematics",
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "vehicles_count": vehicle_count,
            "see_behind": True,
        },
        "action": {"type": "DiscreteMetaAction", "target_speeds": np.linspace(speed_min, speed_max, speed_count)},
        "duration": int(cfg.get("simulation_duration", 40) or 40),
        "vehicles_density": float(resolve_env_value(cfg, "vehicles_density", env_type, defaults["vehicles_density"])),
        "vehicles_count": vehicle_count,
        "controlled_vehicles": 1,
        "real_time_rendering": True,
        "show_trajectories": True,
        "render_agent": True,
        "scaling": 5,
        "ego_spacing": float(resolve_env_value(cfg, "ego_spacing", env_type, 2.0)),
        "initial_speed": float(resolve_env_value(cfg, "initial_speed", env_type, defaults["initial_speed"])),
        "policy_frequency": int(resolve_env_value(cfg, "policy_frequency", env_type, defaults["policy_frequency"])),
        "simulation_frequency": int(resolve_env_value(cfg, "simulation_frequency", env_type, 15) or 15),
    }
    if env_type in {"highway-v0", "merge-v0"}:
        config["lanes_count"] = int(resolve_env_value(cfg, "lanes_count", env_type, defaults["lanes_count"]))
    if env_type == "intersection-v0":
        for key in ("destination", "initial_vehicle_count", "spawn_probability"):
            value = resolve_env_value(cfg, key, env_type, None)
            if value is not None:
                config[key] = value
    for key in ("terminate_on_arrival",):
        value = resolve_env_value(cfg, key, env_type, None)
        if value is not None:
            config[key] = value
    return config
