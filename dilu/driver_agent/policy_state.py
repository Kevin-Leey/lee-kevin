"""Versioned policy-state contract for exact counterfactual replay.

The contract intentionally contains only mutable state that can change future
policy outputs.  Environment state, LLM clients, executors, diagnostic caches,
and complete runtime objects are outside this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Dict, Optional


DRIVER_POLICY_STATE_SCHEMA = "driver_agent_v2_policy_state_v1"
FAST_POLICY_STATE_SCHEMA = "fast_thinker_policy_state_v1"
RGD_POLICY_STATE_SCHEMA = "rgd_orchestrator_policy_state_v1"
RAD_POLICY_STATE_SCHEMA = "rad_signal_controller_policy_state_v1"

_DRIVER_FIELDS = frozenset({"schema", "fast", "orchestrator"})
_FAST_FIELDS = frozenset(
    {"schema", "action_history", "action_history_capacity"}
)
_RGD_FIELDS = frozenset(
    {
        "schema",
        "decision_count",
        "support_progress_cooldown",
        "rgd_cruise_progress_cooldown",
        "rgd_cruise_recovery_frames",
        "slow_call_attempts",
        "slow_call_cooldown_remaining",
        "rad",
    }
)
_RAD_FIELDS = frozenset(
    {
        "schema",
        "corridor_boundary_ema",
        "corridor_width_ema",
        "last_corridor_stage",
    }
)
_RAD_STAGES = frozenset({"committed", "critical", "decisive", "open"})


def _mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _exact_fields(
    value: Any,
    expected: frozenset[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    payload = _mapping(value, context=context)
    observed = set(payload)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise ValueError(
            f"{context} field drift: missing={missing}, unexpected={unexpected}"
        )
    if not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{context} contains a non-string field name")
    return payload


def _schema(payload: Mapping[str, Any], expected: str, *, context: str) -> None:
    if payload.get("schema") != expected:
        raise ValueError(f"{context} schema drift")


def _integer(
    value: Any,
    *,
    context: str,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    resolved = int(value)
    if resolved < minimum or (maximum is not None and resolved > maximum):
        raise ValueError(f"{context} is outside the permitted range")
    return resolved


def _optional_finite_float(
    value: Any,
    *,
    context: str,
    minimum: float,
    maximum: float,
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number or null")
    resolved = float(value)
    if not math.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise ValueError(f"{context} is outside the permitted range")
    return resolved


def validate_fast_policy_state(
    value: Any,
    *,
    expected_capacity: Optional[int] = None,
) -> Dict[str, Any]:
    payload = _exact_fields(value, _FAST_FIELDS, context="Fast policy state")
    _schema(payload, FAST_POLICY_STATE_SCHEMA, context="Fast policy state")
    capacity = _integer(
        payload["action_history_capacity"],
        context="Fast action-history capacity",
        minimum=1,
    )
    if expected_capacity is not None and capacity != int(expected_capacity):
        raise ValueError("Fast action-history capacity differs from runtime config")
    history_raw = payload["action_history"]
    if not isinstance(history_raw, list):
        raise ValueError("Fast action history must be a list")
    if len(history_raw) > capacity:
        raise ValueError("Fast action history exceeds its declared capacity")
    history = [
        _integer(action, context="Fast action-history item", maximum=4)
        for action in history_raw
    ]
    return {
        "schema": FAST_POLICY_STATE_SCHEMA,
        "action_history": history,
        "action_history_capacity": capacity,
    }


def validate_rad_policy_state(value: Any) -> Dict[str, Any]:
    payload = _exact_fields(value, _RAD_FIELDS, context="RAD policy state")
    _schema(payload, RAD_POLICY_STATE_SCHEMA, context="RAD policy state")
    boundary = _optional_finite_float(
        payload["corridor_boundary_ema"],
        context="RAD corridor-boundary EMA",
        minimum=0.0,
        maximum=1.0,
    )
    width = _optional_finite_float(
        payload["corridor_width_ema"],
        context="RAD corridor-width EMA",
        minimum=0.0,
        maximum=5.0,
    )
    stage = payload["last_corridor_stage"]
    if stage is not None and stage not in _RAD_STAGES:
        raise ValueError("RAD corridor stage is invalid")
    return {
        "schema": RAD_POLICY_STATE_SCHEMA,
        "corridor_boundary_ema": boundary,
        "corridor_width_ema": width,
        "last_corridor_stage": stage,
    }


def validate_rgd_policy_state(
    value: Any,
    *,
    slow_call_budget: Optional[int] = None,
    slow_call_cooldown_frames: Optional[int] = None,
) -> Dict[str, Any]:
    payload = _exact_fields(value, _RGD_FIELDS, context="RGD policy state")
    _schema(payload, RGD_POLICY_STATE_SCHEMA, context="RGD policy state")
    attempts = _integer(
        payload["slow_call_attempts"], context="RGD slow-call attempts"
    )
    if slow_call_budget is not None and attempts > int(slow_call_budget):
        raise ValueError("RGD slow-call attempts exceed the runtime budget")
    cooldown = _integer(
        payload["slow_call_cooldown_remaining"],
        context="RGD slow-call cooldown",
    )
    if (
        slow_call_cooldown_frames is not None
        and cooldown > int(slow_call_cooldown_frames) + 1
    ):
        raise ValueError("RGD slow-call cooldown differs from runtime config")
    return {
        "schema": RGD_POLICY_STATE_SCHEMA,
        "decision_count": _integer(
            payload["decision_count"], context="RGD decision count"
        ),
        "support_progress_cooldown": _integer(
            payload["support_progress_cooldown"],
            context="RGD support-progress cooldown",
        ),
        "rgd_cruise_progress_cooldown": _integer(
            payload["rgd_cruise_progress_cooldown"],
            context="RGD cruise-progress cooldown",
        ),
        "rgd_cruise_recovery_frames": _integer(
            payload["rgd_cruise_recovery_frames"],
            context="RGD cruise-recovery frames",
        ),
        "slow_call_attempts": attempts,
        "slow_call_cooldown_remaining": cooldown,
        "rad": validate_rad_policy_state(payload["rad"]),
    }


def validate_driver_policy_state(value: Any) -> Dict[str, Any]:
    payload = _exact_fields(value, _DRIVER_FIELDS, context="Driver policy state")
    _schema(payload, DRIVER_POLICY_STATE_SCHEMA, context="Driver policy state")
    return {
        "schema": DRIVER_POLICY_STATE_SCHEMA,
        "fast": validate_fast_policy_state(payload["fast"]),
        "orchestrator": validate_rgd_policy_state(payload["orchestrator"]),
    }


def policy_state_sha256(value: Any) -> str:
    """Hash a validated policy-state payload using canonical JSON."""
    payload = validate_driver_policy_state(value)
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
