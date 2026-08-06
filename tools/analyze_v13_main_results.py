"""Shared predicates for the action-aligned v13 result analysis."""

from __future__ import annotations

from typing import Any, Mapping


def canonical_distinct_actuation(event: Mapping[str, Any]) -> bool:
    """Return whether a released slow proposal changed the executed control.

    The predicate deliberately uses the post-safety executed action, rather
    than the raw LLM proposal, because the latter is not the action that
    reaches the simulator.
    """
    if not bool(event.get("closed_loop_latency_release_event", False)):
        return False
    if not bool(event.get("closed_loop_release_action_alignment_evaluated", False)):
        return False
    if not bool(event.get("closed_loop_release_action_alignment_pass", False)):
        return False
    if bool(event.get("closed_loop_release_opportunity_rejected", False)):
        return False
    if bool(event.get("closed_loop_release_action_unavailable", False)):
        return False
    try:
        fast_action = int(event["closed_loop_execution_state_fast_action"])
        executed_action = int(event["closed_loop_latency_executed_action"])
    except (KeyError, TypeError, ValueError):
        return False
    return executed_action != fast_action
