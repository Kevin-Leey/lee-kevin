"""Per-frame action trace helpers for route-preservation audits."""

from typing import Dict


def build_action_trace_fields(proposed_action: int, final_action: int) -> Dict[str, object]:
    """Build the action-preservation fields appended to each persisted decision trace."""
    route_action_changed = bool(int(proposed_action) != int(final_action))
    return {
        "proposed_action": int(proposed_action),
        "final_action": int(final_action),
        "route_action_changed": bool(route_action_changed),
        "route_action_preserved": bool(not route_action_changed),
    }