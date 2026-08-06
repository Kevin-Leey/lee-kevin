"""Runtime event construction and episode-scoped bookkeeping."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping


EVENT_SCHEMA_VERSION = "rgd_event_log_v3"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def create_episode_runtime_state() -> Dict[str, Any]:
    """Create one isolated episode ledger.

    Request identifiers are retained after terminal events so a delayed
    response cannot be re-used within an episode.
    """
    return {
        "action": 1,
        "prev_image": None,
        "collision_frame": -1,
        "episode_reward": 0.0,
        "event_log": [],
        "latency_replay_queue": [],
        "release_snapshots": {},
        "_latency_request_ids": set(),
        "_latency_terminal_request_ids": set(),
        "_latency_request_sequence": 0,
    }


def build_episode_event(
    frame: int,
    state: Mapping[str, Any],
    decision_meta: Mapping[str, Any],
    terminal_outcome: Mapping[str, Any],
) -> Dict[str, Any]:
    """Merge frame state, decision provenance, and environment termination data."""
    event: Dict[str, Any] = {"frame": int(frame)}
    event.update(_json_safe(dict(state or {})))
    event.update(_json_safe(dict(decision_meta or {})))
    event.update(_json_safe(dict(terminal_outcome or {})))
    event.setdefault("proposed_action", event.get("final_action", 1))
    event.setdefault("final_action", event.get("proposed_action", 1))
    event["proposed_action"] = int(event["proposed_action"])
    event["final_action"] = int(event["final_action"])
    event.setdefault("route_action_changed", bool(event["proposed_action"] != event["final_action"]))
    event.setdefault("route_action_preserved", not bool(event["route_action_changed"]))
    return event


def collect_runtime_integrity(agent: Any) -> Dict[str, Any]:
    """Capture a small, serializable runtime identity at episode creation."""
    payload = {
        "agent_class": f"{agent.__class__.__module__}.{agent.__class__.__name__}",
        "orchestrator_class": (
            f"{agent.orchestrator.__class__.__module__}.{agent.orchestrator.__class__.__name__}"
            if getattr(agent, "orchestrator", None) is not None
            else ""
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {**payload, "runtime_identity_sha256": hashlib.sha256(encoded).hexdigest()}


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "build_episode_event",
    "collect_runtime_integrity",
    "create_episode_runtime_state",
]
