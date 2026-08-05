"""Canonical latency resolution shared by RGD routing and runtime replay."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


LATENCY_CONTRACT_VERSION = "rgd_latency_contract_v1"


def _parse_nonnegative_float(value: Any) -> Tuple[Optional[float], Optional[str]]:
    """Parse a latency value without turning malformed input into zero delay."""
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None, "invalid_numeric_latency"
    if not math.isfinite(resolved):
        return None, "nonfinite_latency"
    if resolved < 0.0:
        return None, "negative_latency"
    return float(resolved), None


def _nonnegative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved >= 0 else None


def resolve_policy_frequency(
    cfg: Optional[Dict[str, Any]],
    env=None,
) -> Tuple[float, str]:
    """Resolve the policy rate used by the final simulator instance."""
    candidates = []
    seen_owners = set()
    for label, owner in (
        ("env.unwrapped.config.policy_frequency", getattr(env, "unwrapped", None)),
        ("env.config.policy_frequency", env),
    ):
        if owner is None or id(owner) in seen_owners:
            continue
        seen_owners.add(id(owner))
        owner_cfg = getattr(owner, "config", None)
        if hasattr(owner_cfg, "get"):
            candidates.append((label, owner_cfg.get("policy_frequency")))

    runtime_cfg = dict(cfg or {})
    bound_frequency = runtime_cfg.get("_resolved_policy_frequency_hz")
    if bound_frequency is not None:
        candidates.append(
            (
                str(
                    runtime_cfg.get(
                        "_resolved_policy_frequency_source",
                        "runtime.bound_policy_frequency",
                    )
                    or "runtime.bound_policy_frequency"
                ),
                bound_frequency,
            )
        )

    env_type = str(runtime_cfg.get("env_type", "") or "")
    by_env = runtime_cfg.get("policy_frequency_by_env", {}) or {}
    if isinstance(by_env, dict) and env_type in by_env:
        candidates.append(("config.policy_frequency_by_env", by_env.get(env_type)))
    env_scoped = runtime_cfg.get(env_type.replace("-", "_") + "_env", {}) or {}
    if isinstance(env_scoped, dict) and "policy_frequency" in env_scoped:
        candidates.append(
            ("config.environment.policy_frequency", env_scoped.get("policy_frequency"))
        )
    candidates.append(("config.policy_frequency", runtime_cfg.get("policy_frequency")))

    for source, raw_value in candidates:
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value, source
    return 1.0, "default.policy_frequency"


def resolve_latency_contract(
    cfg: Optional[Dict[str, Any]],
    env=None,
) -> Dict[str, Any]:
    """Resolve one delay contract for routing, guards, and replay scheduling.

    Seconds are authoritative when ``extra_latency_s`` is present. The legacy
    ``delay_steps`` input remains supported only when seconds are absent; when
    both are present it is retained as an audit value.
    """
    runtime_cfg = dict(cfg or {})
    if env is None:
        bound = runtime_cfg.get("_resolved_latency_contract")
        if (
            isinstance(bound, dict)
            and bound.get("version") == LATENCY_CONTRACT_VERSION
        ):
            return dict(bound)

    replay_cfg = dict(runtime_cfg.get("closed_loop_latency_replay", {}) or {})
    replay_enabled = bool(replay_cfg.get("enable", False))
    policy_frequency, frequency_source = resolve_policy_frequency(runtime_cfg, env)
    raw_configured_steps = replay_cfg.get("delay_steps")
    configured_steps = _nonnegative_int(raw_configured_steps)
    configured_steps_valid = bool(
        raw_configured_steps is None or configured_steps is not None
    )

    prediction_available = False
    predicted_seconds = 0.0
    predicted_steps = 0
    source = "missing_prediction"

    invalid_reason = None
    if "extra_latency_s" in replay_cfg:
        parsed_seconds, invalid_reason = _parse_nonnegative_float(
            replay_cfg.get("extra_latency_s")
        )
        if parsed_seconds is not None:
            predicted_seconds = parsed_seconds
            predicted_steps = int(math.ceil(predicted_seconds * policy_frequency))
            prediction_available = True
            source = "closed_loop_latency_replay.extra_latency_s"
        else:
            source = "closed_loop_latency_replay.invalid_extra_latency_s"
    elif raw_configured_steps is not None and configured_steps is not None:
        predicted_steps = int(configured_steps)
        predicted_seconds = float(predicted_steps / policy_frequency)
        prediction_available = True
        source = "closed_loop_latency_replay.delay_steps_legacy"
    elif raw_configured_steps is not None:
        invalid_reason = "invalid_delay_steps"
        source = "closed_loop_latency_replay.invalid_delay_steps"
    elif runtime_cfg.get("rgd_predicted_slow_latency_s") is not None:
        parsed_seconds, invalid_reason = _parse_nonnegative_float(
            runtime_cfg.get("rgd_predicted_slow_latency_s")
        )
        if parsed_seconds is not None:
            predicted_seconds = parsed_seconds
            predicted_steps = int(math.ceil(predicted_seconds * policy_frequency))
            prediction_available = True
            source = str(
                runtime_cfg.get(
                    "rgd_predicted_slow_latency_source",
                    "runtime_config.rgd_predicted_slow_latency_s",
                )
                or "runtime_config.rgd_predicted_slow_latency_s"
            )
        else:
            source = "runtime_config.invalid_predicted_slow_latency_s"

    scheduled_steps = int(predicted_steps if replay_enabled and prediction_available else 0)
    scheduled_seconds = float(scheduled_steps / policy_frequency)
    quantized_prediction_seconds = float(predicted_steps / policy_frequency)
    configured_steps_consistent = bool(
        configured_steps_valid
        and (
            configured_steps is None
            or (prediction_available and configured_steps == predicted_steps)
        )
    )
    return {
        "version": LATENCY_CONTRACT_VERSION,
        "replay_enabled": bool(replay_enabled),
        "prediction_available": bool(prediction_available),
        "prediction_invalid_reason": invalid_reason,
        "predicted_seconds": float(predicted_seconds),
        "quantized_prediction_seconds": float(quantized_prediction_seconds),
        "predicted_steps": int(predicted_steps),
        "scheduled_seconds": float(scheduled_seconds),
        "scheduled_steps": int(scheduled_steps),
        "policy_frequency_hz": float(policy_frequency),
        "policy_frequency_source": str(frequency_source),
        "source": str(source),
        "configured_steps": configured_steps,
        "configured_steps_consistent": bool(configured_steps_consistent),
        # Compatibility aliases retained for existing runtime consumers.
        "seconds": float(predicted_seconds),
        "steps": int(scheduled_steps),
    }


def bind_latency_contract(
    cfg: Optional[Dict[str, Any]],
    env=None,
) -> Dict[str, Any]:
    """Return a runtime config bound to the final environment's policy rate."""
    resolved_cfg = dict(cfg or {})
    contract = resolve_latency_contract(resolved_cfg, env)
    resolved_cfg["_resolved_policy_frequency_hz"] = float(
        contract["policy_frequency_hz"]
    )
    resolved_cfg["_resolved_policy_frequency_source"] = str(
        contract["policy_frequency_source"]
    )
    resolved_cfg["_resolved_latency_contract"] = dict(contract)
    return resolved_cfg
