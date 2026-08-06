"""Scenario factory for the supported highway-env and MetaDrive backends."""

from __future__ import annotations

from typing import Any, Optional

from dilu.scenario.envScenario import EnvScenario
from dilu.scenario.env_ids import (
    SCENARIO_MAP,
    infer_scenario_type,
    is_metadrive_env,
    require_supported_env,
)


def create_scenario(
    env: Any,
    env_type: str,
    seed: int,
    database: Optional[str] = None,
) -> Any:
    """Build the backend-specific scenario view used by the driver runtime."""
    resolved_env_type = require_supported_env(env_type)
    if is_metadrive_env(resolved_env_type):
        # Importing this adapter at module scope creates a cycle because the
        # adapter imports the environment-id helpers from this package.
        from dilu.metadrive.adapter import MetaDriveScenario

        return MetaDriveScenario(env, resolved_env_type, int(seed), database)

    backend_env = getattr(env, "unwrapped", env)
    scenario = EnvScenario(backend_env, resolved_env_type, int(seed), database)
    scenario.scenario_type = infer_scenario_type(resolved_env_type)
    return scenario


__all__ = ["SCENARIO_MAP", "create_scenario"]
