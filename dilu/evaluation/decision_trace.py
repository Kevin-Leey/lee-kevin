"""Normalize per-frame controller metadata before it enters an event trace."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from dilu.evaluation.action_trace import build_action_trace_fields


def _trace_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_trace_value(item) for item in value]
    return value


def _flatten_recoverability_gate(payload: Dict[str, Any]) -> None:
    gate = payload.get("recoverability_gate")
    if not isinstance(gate, Mapping):
        return
    normalized_gate = _trace_value(gate)
    payload["recoverability_gate"] = normalized_gate
    for key, value in normalized_gate.items():
        flat_key = str(key) if str(key).startswith("recoverability_") else f"recoverability_{key}"
        payload.setdefault(flat_key, value)


def build_decision_meta(
    source: Mapping[str, Any] | None,
    *,
    proposed_action: int,
    final_action: int,
) -> Dict[str, Any]:
    """Preserve decision provenance while adding action-stage identifiers.

    No truthiness defaults are used for action IDs: action ``0`` is a valid
    left-lane command and must survive serialization unchanged.
    """
    payload: Dict[str, Any] = _trace_value(dict(source or {}))
    payload.update(build_action_trace_fields(int(proposed_action), int(final_action)))
    payload.setdefault("system_used", "fast")
    payload.setdefault("route_label", str(payload["system_used"]))
    payload.setdefault("confidence", 0.0)
    payload.setdefault("latency_ms", 0.0)
    _flatten_recoverability_gate(payload)
    profile = payload.get("route_ambiguity_profile")
    if hasattr(profile, "to_dict"):
        payload["route_ambiguity_profile"] = _trace_value(profile.to_dict())
    return payload


__all__ = ["build_decision_meta"]
