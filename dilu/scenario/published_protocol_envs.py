"""Construction and validation helpers for the fixed highway-env protocol."""

from typing import Any, Dict, Optional


PUBLISHED_PROTOCOL_ENVS = frozenset({"merge-v0", "intersection-v0", "roundabout-v0"})


def build_published_highway_env(env_type: str, env_config: Dict[str, Any], render_mode: Optional[str] = None):
    """Create one configured highway-env instance for a published protocol row."""
    import gymnasium as gym
    import highway_env  # noqa: F401  # Registers highway-env environments.

    kwargs = {} if render_mode is None else {"render_mode": render_mode}
    env = gym.make(str(env_type), **kwargs)
    env.unwrapped.configure(dict(env_config))
    return env


def validate_published_protocol_initial_state(env: Any, require_randomized_idm: bool = False) -> None:
    """Reject an incomplete simulator reset before the policy consumes it."""
    vehicle = getattr(env, "vehicle", None)
    road = getattr(env, "road", None)
    if vehicle is None or road is None:
        raise RuntimeError("published protocol environment did not produce an ego vehicle and road")
    vehicles = list(getattr(road, "vehicles", []) or [])
    if vehicle not in vehicles:
        raise RuntimeError("published protocol ego vehicle is absent from the road")
    if require_randomized_idm:
        background = [item for item in vehicles if item is not vehicle]
        if not any(hasattr(item, "randomize_behavior") for item in background):
            raise RuntimeError("published protocol expected randomized IDM background traffic")
