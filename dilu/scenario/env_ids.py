"""Canonical environment identifiers used by the RGD runtime."""

HIGHWAY_SCENARIO_MAP = {
    "highway-v0": "highway",
    "merge-v0": "merge",
    "roundabout-v0": "roundabout",
    "intersection-v0": "intersection",
}

METADRIVE_SCENARIO_MAP = {
    "metadrive-highway-v0": "highway",
    "metadrive-merge-v0": "merge",
    "metadrive-roundabout-v0": "roundabout",
    "metadrive-intersection-v0": "intersection",
    "metadrive-pedestrian-crossing-v0": "intersection",
    "metadrive-uncontrolled-intersection-v0": "intersection",
    "metadrive-integrated-traffic-v0": "intersection",
}

SCENARIO_MAP = {**HIGHWAY_SCENARIO_MAP, **METADRIVE_SCENARIO_MAP}


def require_highway_env(env_type: str) -> str:
    env = str(env_type or "").strip()
    if env not in HIGHWAY_SCENARIO_MAP:
        raise ValueError(f"unsupported highway-env id: {env!r}; supported={sorted(HIGHWAY_SCENARIO_MAP)}")
    return env


def is_highway_env(env_type: str) -> bool:
    return str(env_type or "").strip() in HIGHWAY_SCENARIO_MAP


def is_metadrive_env(env_type: str) -> bool:
    return str(env_type or "").strip() in METADRIVE_SCENARIO_MAP


def require_supported_env(env_type: str) -> str:
    env = str(env_type or "").strip()
    if env not in SCENARIO_MAP:
        raise ValueError(f"unsupported env id: {env!r}; supported={sorted(SCENARIO_MAP)}")
    return env


def env_family(env_type: str) -> str:
    env = require_supported_env(env_type)
    return "metadrive" if env in METADRIVE_SCENARIO_MAP else "highway_env"


def infer_scenario_type(env_type: str) -> str:
    return SCENARIO_MAP[require_supported_env(env_type)]


def infer_env_label(env_type: str) -> str:
    env = require_supported_env(env_type)
    if env.startswith("metadrive-") and env.endswith("-v0"):
        return "metadrive_" + env[len("metadrive-") : -len("-v0")].replace("-", "_")
    return infer_scenario_type(env)
