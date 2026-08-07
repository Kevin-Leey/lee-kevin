"""Runtime event construction and episode-scoped bookkeeping."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional


EVENT_SCHEMA_VERSION = "rgd_event_log_v3"


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        normalized = [_json_safe(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ),
        )
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


def classify_release_lifecycle(
    *,
    release_event: bool,
    release_fast_action: Any,
    released_slow_action: Any,
    executed_action: Any,
    release_action_unavailable: bool = False,
    release_alignment_evaluated: bool = False,
    release_alignment_passed: bool = False,
    release_selected_action: Optional[Any] = None,
    release_opportunity_rejected: bool = False,
) -> Dict[str, Any]:
    """Derive release labels from primitive action identities.

    Selection distinctness is evaluated before the shared safety projection.
    Final actuation distinctness is retained only when the selected proposal
    survives unchanged to the environment command.
    """
    if not bool(release_event):
        return {
            "release_selection_comparison_available": False,
            "release_selection_distinct": False,
            "closed_loop_release_actuation_distinct": False,
            "closed_loop_kept_distinct_release": False,
            "closed_loop_final_returns_to_fast": False,
            "closed_loop_post_latency_shield_rewrite": False,
        }
    try:
        fast = int(release_fast_action)
        slow = int(released_slow_action)
        executed = int(executed_action)
        selected = slow if release_selected_action is None else int(release_selected_action)
    except (TypeError, ValueError):
        return {
            "release_selection_comparison_available": False,
            "release_selection_distinct": False,
            "closed_loop_release_actuation_distinct": False,
            "closed_loop_kept_distinct_release": False,
            "closed_loop_final_returns_to_fast": False,
            "closed_loop_post_latency_shield_rewrite": False,
        }
    selection_available = bool(
        not release_action_unavailable and not release_opportunity_rejected
    )
    selection_distinct = bool(selection_available and selected != fast)
    alignment_authorized = bool(
        release_alignment_evaluated and release_alignment_passed
    )
    kept = bool(
        selection_distinct
        and alignment_authorized
        and executed == selected
        and executed != fast
    )
    return {
        "release_selection_comparison_available": True,
        "release_selection_distinct": bool(selection_distinct),
        "closed_loop_release_actuation_distinct": bool(kept),
        "closed_loop_kept_distinct_release": bool(kept),
        "closed_loop_final_returns_to_fast": bool(
            selection_distinct and executed == fast
        ),
        "closed_loop_post_latency_shield_rewrite": bool(executed != selected),
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
    # Restore the authoritative temporal identity after merging extensible
    # state mappings that may contain a colliding auxiliary key.
    event["frame"] = int(frame)
    event.setdefault("proposed_action", event.get("final_action", 1))
    event.setdefault("final_action", event.get("proposed_action", 1))
    event["proposed_action"] = int(event["proposed_action"])
    event["final_action"] = int(event["final_action"])
    event.setdefault("route_action_changed", bool(event["proposed_action"] != event["final_action"]))
    event.setdefault("route_action_preserved", not bool(event["route_action_changed"]))
    if bool(event.get("closed_loop_latency_release_event", False)):
        modern = bool(
            event.get("factorial_replay_version")
            or event.get("native_async_slow_path", False)
        )
        release_stage = str(event.get("release_action_comparison_stage", "") or "")
        final_stage = str(event.get("final_actuator_action_stage", "") or "")
        allowed_release_stages = {
            "post_release_guard_and_frame_safety_pre_actuator_bridge",
            "post_release_guard_pre_final_safety_projection",
        }
        allowed_final_stages = {
            "post_shared_actuator_bridge_pre_environment_step",
            "post_environment_action_adapter",
        }
        if modern:
            if not release_stage or not final_stage:
                raise RuntimeError(
                    "modern release event is missing its explicit action-stage contract"
                )
            if release_stage not in allowed_release_stages:
                raise RuntimeError("invalid release action comparison stage")
            if final_stage not in allowed_final_stages:
                raise RuntimeError("invalid final actuator action stage")

        release_fast = event.get(
            "release_fast_comparator_action",
            event.get("closed_loop_execution_state_fast_action"),
        )
        release_selected = event.get("release_selected_action")
        if not release_stage:
            try:
                sentinel = int(release_selected)
            except (TypeError, ValueError):
                sentinel = -1
            if sentinel < 0:
                release_fast = event.get("closed_loop_execution_state_fast_action")
                release_selected = event.get(
                    "final_actuator_action",
                    event.get("closed_loop_latency_executed_action", event["final_action"]),
                )
                event["release_fast_comparator_action"] = release_fast
                event["release_selected_action"] = release_selected
        lifecycle = classify_release_lifecycle(
                release_event=True,
                release_fast_action=release_fast,
                released_slow_action=event.get(
                    "closed_loop_released_slow_action",
                    event.get("release_selected_action"),
                ),
                release_selected_action=release_selected,
                executed_action=event.get(
                    "final_actuator_action",
                    event.get("closed_loop_latency_executed_action", event["final_action"]),
                ),
                release_action_unavailable=bool(
                    event.get("closed_loop_release_action_unavailable", False)
                ),
                release_alignment_evaluated=bool(
                    event.get(
                        "closed_loop_release_action_alignment_evaluated", False
                    )
                ),
                release_alignment_passed=bool(
                    event.get("closed_loop_release_action_alignment_pass", False)
                ),
                release_opportunity_rejected=bool(
                    event.get("closed_loop_release_opportunity_rejected", False)
                ),
            )
        event.update(lifecycle)
        aligned_final_comparison = bool(
            release_stage
            == "post_release_guard_and_frame_safety_pre_actuator_bridge"
            and final_stage
            == "post_shared_actuator_bridge_pre_environment_step"
        )
        event["closed_loop_release_actuation_comparison_available"] = bool(
            aligned_final_comparison
        )
        event["closed_loop_release_actuation_supports_rollout_effect_claim"] = False
        event["closed_loop_release_actuation_semantics"] = (
            "same_stage_command_identity"
            if aligned_final_comparison
            else "unavailable_without_same_stage_final_fast_counterfactual"
        )
        if modern and not aligned_final_comparison:
            event["closed_loop_release_actuation_distinct"] = False
            event["closed_loop_kept_distinct_release"] = False
            event["closed_loop_final_returns_to_fast"] = False
        if str(
            event.get("closed_loop_latency_terminal_response_outcome", "") or ""
        ) == "valid":
            event["closed_loop_latency_terminal_outcome"] = (
                "distinct_actuation"
                if lifecycle["release_selection_distinct"]
                else "fast_equivalent"
            )
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
    "classify_release_lifecycle",
    "collect_runtime_integrity",
    "create_episode_runtime_state",
]
