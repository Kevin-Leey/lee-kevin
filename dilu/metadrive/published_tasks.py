"""MetaDrive task construction for the supplementary interactive scenes."""

from typing import Any, Dict


PUBLISHED_TASK_MODES = frozenset(
    {
        "metadrive-pedestrian-crossing-v0",
        "metadrive-uncontrolled-intersection-v0",
        "metadrive-integrated-traffic-v0",
    }
)


def build_published_task_env(env_type: str, config: Dict[str, Any]):
    """Instantiate a MetaDrive environment for one declared task mode."""
    task = str(env_type or "").strip()
    if task not in PUBLISHED_TASK_MODES:
        raise ValueError(f"unsupported published MetaDrive task: {task!r}")
    try:
        from metadrive import MetaDriveEnv
    except ImportError as exc:
        raise ImportError("MetaDrive task modes require a MetaDrive installation.") from exc
    return MetaDriveEnv(dict(config))
