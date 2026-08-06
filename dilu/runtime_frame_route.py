"""Single frame action resolver for the RGD runtime."""

import random
from typing import Any, Dict, Tuple


def resolve_agent_action(
    agent,
    cfg: Dict[str, Any],
    driving_state,
) -> Tuple[int, str, Dict[str, Any]]:
    agent_type = str(cfg.get("agent_type", "llm") or "llm")
    if agent_type == "random":
        return random.choice(list(range(5))), "[Random Baseline]", {"system_used": "random", "route_label": "random", "confidence": 0.0}
    if agent_type == "idm_only":
        return 3, "[IDM Baseline]", {"system_used": "idm_only", "route_label": "idm_only", "confidence": 1.0}
    return agent.decide(driving_state)
