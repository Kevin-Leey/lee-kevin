"""Closed-loop execution helpers for the reconstructed RGD runtime."""

from __future__ import annotations

import copy
import math
import time
from collections import deque
from collections.abc import Mapping, MutableMapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dilu.driver_agent.base.state import ActionType
from dilu.evaluation.decision_trace import build_decision_meta
from dilu.evaluation.release_snapshot import (
    capture_release_snapshot,
    validate_release_snapshot_policy_state,
)
from dilu.latency_contract import resolve_latency_contract
from dilu.runtime_frame_route import resolve_agent_action
from dilu.runtime_frame_state import (
    build_frame_driving_state,
    extract_runtime_state,
    inject_safety_cost_bridge,
    record_executed_history_frame,
)
from dilu.runtime_frame_trace import build_episode_event
from dilu.utils.driving import ACTIONS_DESCRIPTION


_TERMINAL_OUTCOMES = {"valid", "timeout", "failure"}
_NATIVE_ASYNC_POLICY_FREQUENCY_HZ = 10.0


def _pace_native_async_policy_frame(
    *,
    frame: int,
    agent: Any,
    episode_state: MutableMapping[str, Any],
) -> Dict[str, Any]:
    """Wait for an absolute 10 Hz deadline only for the live async agent."""
    capability = getattr(agent, "uses_native_async_policy_pacing", None)
    enabled = bool(capability()) if callable(capability) else False
    episode_state["_last_policy_pacing_sleep_s"] = 0.0
    episode_state.setdefault("_total_policy_pacing_sleep_s", 0.0)

    if enabled:
        now = time.monotonic()
        origin = episode_state.get("_policy_pacing_origin_monotonic")
        if origin is None:
            origin = now
            episode_state["_policy_pacing_origin_monotonic"] = float(origin)
        deadline = float(origin) + (
            max(0, int(frame)) / _NATIVE_ASYNC_POLICY_FREQUENCY_HZ
        )
        remaining = deadline - now
        if remaining > 0.0:
            sleep_started = time.monotonic()
            sleep_duration = max(0.0, deadline - sleep_started)
            if sleep_duration > 0.0:
                time.sleep(sleep_duration)
                actual_sleep = max(0.0, time.monotonic() - sleep_started)
                episode_state["_last_policy_pacing_sleep_s"] = float(actual_sleep)
                episode_state["_total_policy_pacing_sleep_s"] = float(
                    episode_state["_total_policy_pacing_sleep_s"]
                ) + float(actual_sleep)

    return {
        "policy_pacing_enabled": bool(enabled),
        "policy_pacing_frequency_hz": _NATIVE_ASYNC_POLICY_FREQUENCY_HZ,
        "policy_pacing_sleep_s": float(
            episode_state["_last_policy_pacing_sleep_s"]
        ),
        "policy_pacing_total_sleep_s": float(
            episode_state["_total_policy_pacing_sleep_s"]
        ),
    }


def exclude_policy_pacing_sleep(
    wall_elapsed_s: float,
    episode_state: Mapping[str, Any],
) -> float:
    """Return algorithm wall time after removing this frame's pacing sleep."""
    elapsed = max(0.0, float(wall_elapsed_s))
    pacing_sleep = max(
        0.0,
        float(
            dict(episode_state or {}).get("_last_policy_pacing_sleep_s", 0.0)
            or 0.0
        ),
    )
    return max(0.0, elapsed - pacing_sleep)


def _resolve_render_mode(cfg: Mapping[str, Any]) -> Optional[str]:
    value = str(dict(cfg or {}).get("render_mode", "") or "").strip().lower()
    return None if value in {"", "none", "off", "false", "0"} else value


def _resolve_latency_replay_delay(
    cfg: Mapping[str, Any], env: Any = None
) -> Dict[str, Any]:
    """Compatibility facade for the canonical latency contract resolver."""
    return resolve_latency_contract(dict(cfg or {}), env)


def _request_latency_contract(
    contract: Mapping[str, Any], meta: Mapping[str, Any]
) -> Dict[str, Any]:
    """Bind optional scripted jitter to one immutable request contract."""
    resolved = dict(contract or {})
    frequency = max(1e-9, float(resolved.get("policy_frequency_hz", 1.0) or 1.0))
    scripted = "closed_loop_scripted_latency_steps" in dict(meta or {})
    if scripted:
        raw_steps = dict(meta or {}).get("closed_loop_scripted_latency_steps")
        if isinstance(raw_steps, bool) or not isinstance(raw_steps, int) or raw_steps < 0:
            raise ValueError(
                "closed_loop_scripted_latency_steps must be a nonnegative integer"
            )
        scheduled_steps = int(raw_steps)
        source = "decision_meta.closed_loop_scripted_latency_steps"
    else:
        scheduled_steps = max(0, int(resolved.get("scheduled_steps", 0) or 0))
        source = str(resolved.get("source", "latency_contract") or "latency_contract")
    resolved.update(
        {
            "scheduled_steps": int(scheduled_steps),
            "scheduled_seconds": float(scheduled_steps / frequency),
            "scripted_sample": bool(scripted),
            "source": source,
        }
    )
    return resolved


def _raw_available_actions(env: Any) -> List[int]:
    unwrapped = getattr(env, "unwrapped", env)
    getter = getattr(unwrapped, "get_available_actions", None)
    if not callable(getter):
        return [int(item) for item in ActionType]
    actions = sorted({int(action) for action in list(getter() or [])})
    return actions or [int(ActionType.IDLE)]


def _float(value: Any, default: float = float("inf")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _state_value(state: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in state and state[key] is not None:
            return state[key]
    return default


def _hidden_slower_risk(state: Mapping[str, Any]) -> bool:
    speed = max(0.0, _float(_state_value(state, "speed", "ego_speed", default=0.0), 0.0))
    front = _float(_state_value(state, "front_dist", "front_distance"))
    ttc = _float(_state_value(state, "ttc"))
    thw = _float(_state_value(state, "thw"))
    return bool(
        (math.isfinite(front) and front < max(24.0, speed * 1.5))
        or (math.isfinite(ttc) and ttc < 3.0)
        or (math.isfinite(thw) and thw < 1.0)
    )


def _inject_hidden_slower_action(
    actions: Iterable[int],
    runtime_state: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> List[int]:
    """Expose a virtual brake command only while following risk requires it."""
    raw = [int(action) for action in list(actions or [])]
    bridge = dict(dict(cfg or {}).get("hidden_slower_bridge", {}) or {})
    if not bool(bridge.get("enable", False)):
        return raw
    effective = set(raw)
    if _hidden_slower_risk(dict(runtime_state or {})):
        effective.add(int(ActionType.SLOWER))
    return sorted(effective)


def _available_actions_description(actions: Iterable[int]) -> str:
    lines = ["Your available actions are:"]
    for action in sorted({int(item) for item in list(actions or [])}):
        label = ACTIONS_DESCRIPTION.get(action, f"Action {action}")
        lines.append(f"{label} Action_id: {action}")
    return "\n".join(lines) + "\n"


def _hidden_bridge_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool(dict(dict(cfg or {}).get("hidden_slower_bridge", {}) or {}).get("enable", False))


def _is_highway_runtime(env: Any, cfg: Mapping[str, Any]) -> bool:
    env_type = str(dict(cfg or {}).get("env_type", "") or "")
    if "highway" in env_type:
        return True
    return "highway" in str(getattr(getattr(env, "spec", None), "id", "") or "")


def _apply_unavailable_slower_brake_assist(env: Any, action: int, cfg: Mapping[str, Any]) -> int:
    """Map a hidden brake action to a conservative continuous target when needed."""
    requested = int(action)
    if requested != int(ActionType.SLOWER) or not _hidden_bridge_enabled(cfg):
        return requested
    available = _raw_available_actions(env)
    if requested in available:
        return requested
    unwrapped = getattr(env, "unwrapped", env)
    vehicle = getattr(unwrapped, "vehicle", None)
    if vehicle is None:
        return int(ActionType.IDLE) if int(ActionType.IDLE) in available else requested
    raw_target_speeds = getattr(vehicle, "target_speeds", None)
    target_speeds = list(raw_target_speeds) if raw_target_speeds is not None else []
    current = _float(getattr(vehicle, "target_speed", getattr(vehicle, "speed", 0.0)), 0.0)
    floor = min((float(value) for value in target_speeds), default=0.0)
    brake_target = min(current - max(1.0, abs(current) * 0.2), floor - max(1.0, abs(floor) * 0.2))
    brake_target = max(0.0, float(brake_target))
    setattr(vehicle, "target_speed", brake_target)
    if hasattr(vehicle, "speed_index"):
        vehicle.speed_index = 0
    setattr(vehicle, "_dilu_highway_hidden_brake_latched", True)
    setattr(vehicle, "_dilu_highway_hidden_brake_target", brake_target)
    setattr(vehicle, "_dilu_highway_hidden_brake_original_target", current)
    return int(ActionType.IDLE) if int(ActionType.IDLE) in available else requested


def _release_highway_hidden_brake_target(env: Any, action: int, cfg: Mapping[str, Any]) -> int:
    """Keep a virtual brake target until the observed following corridor is safe."""
    requested = int(action)
    if not _hidden_bridge_enabled(cfg) or not _is_highway_runtime(env, cfg):
        return requested
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    if vehicle is None or not bool(getattr(vehicle, "_dilu_highway_hidden_brake_latched", False)):
        return requested
    target = _float(getattr(vehicle, "_dilu_highway_hidden_brake_target", 0.0), 0.0)
    state = dict(dict(cfg or {}).get("_current_runtime_state", {}) or {})
    if requested in {int(ActionType.LANE_LEFT), int(ActionType.LANE_RIGHT)}:
        setattr(vehicle, "target_speed", target)
        return requested
    speed = max(0.0, _float(_state_value(state, "speed", "ego_speed", default=0.0), 0.0))
    front = _float(_state_value(state, "front_dist", "front_distance"))
    ttc = _float(_state_value(state, "ttc"))
    thw = _float(_state_value(state, "thw"))
    adjacent_distance = _float(_state_value(state, "closest_vehicle_distance"))
    adjacent_lateral = abs(_float(_state_value(state, "closest_vehicle_lateral")))
    adjacent_longitudinal = _float(_state_value(state, "closest_vehicle_longitudinal"))
    adjacent_conflict = (
        math.isfinite(adjacent_distance)
        and adjacent_distance <= 8.0
        and 2.0 <= adjacent_lateral <= 7.0
        and -4.0 <= adjacent_longitudinal <= 14.0
    )
    safe = bool(
        (not math.isfinite(front) or front >= max(20.0, speed * 2.0))
        and (not math.isfinite(ttc) or ttc >= 3.0)
        and (not math.isfinite(thw) or thw >= 1.4)
        and not adjacent_conflict
    )
    if safe:
        restored = _float(getattr(vehicle, "_dilu_highway_hidden_brake_original_target", getattr(vehicle, "target_speed", 0.0)), 0.0)
        setattr(vehicle, "target_speed", restored)
        if hasattr(vehicle, "speed_index"):
            vehicle.speed_index = 0
        setattr(vehicle, "_dilu_highway_hidden_brake_latched", False)
        return requested
    setattr(vehicle, "target_speed", target)
    if hasattr(vehicle, "speed_index"):
        vehicle.speed_index = 0
    return int(ActionType.IDLE) if requested == int(ActionType.FASTER) else requested


def _ensure_latency_state(episode_state: MutableMapping[str, Any]) -> None:
    episode_state.setdefault("latency_replay_queue", [])
    episode_state.setdefault("release_snapshots", {})
    episode_state.setdefault("_latency_request_ids", set())
    episode_state.setdefault("_latency_terminal_request_ids", set())
    episode_state.setdefault("_latency_request_sequence", 0)


def _ready_frame(item: Mapping[str, Any]) -> int:
    return int(item.get("available_frame", item.get("release_frame", 0)) or 0)


def _selected_ready_latency_request(
    episode_state: Mapping[str, Any],
    *,
    frame: int,
) -> Optional[Dict[str, Any]]:
    """Select one due request deterministically without mutating the queue."""
    candidates = [
        dict(item)
        for item in list(episode_state.get("latency_replay_queue", []) or [])
        if int(frame) >= _ready_frame(dict(item))
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            _ready_frame(item),
            int(item.get("source_frame", item.get("release_frame", 0)) or 0),
            str(item.get("request_id", "")),
        ),
    )


def _latency_steps(meta: Mapping[str, Any], contract: Mapping[str, Any]) -> int:
    return int(_request_latency_contract(contract, meta)["scheduled_steps"])


def _issue_outcome(meta: Mapping[str, Any]) -> tuple[Optional[str], bool]:
    explicit = meta.get("closed_loop_latency_response_outcome")
    if explicit is not None and str(explicit or ""):
        value = str(explicit).strip().lower()
        if value not in _TERMINAL_OUTCOMES:
            raise RuntimeError(f"unsupported slow response outcome: {value!r}")
        return value, True
    if bool(meta.get("slow_request_valid_return", False)):
        return "valid", False
    if bool(meta.get("slow_request_failed", False)):
        return "failure", False
    return None, False


def _request_is_issuable(meta: Mapping[str, Any], contract: Mapping[str, Any]) -> tuple[bool, Optional[str], int]:
    if not bool(meta.get("slow_request_attempted", False)):
        return False, None, 0
    outcome, explicit = _issue_outcome(meta)
    steps = _latency_steps(meta, contract)
    if outcome is None:
        return False, None, steps
    return bool(explicit or steps > 0), outcome, steps


def _make_request_id(frame: int, episode_state: MutableMapping[str, Any]) -> str:
    sequence = int(episode_state.get("_latency_request_sequence", 0) or 0)
    episode_state["_latency_request_sequence"] = sequence + 1
    return f"slow:{int(frame)}:{sequence:04d}"


def _metadata_defaults(meta: MutableMapping[str, Any]) -> None:
    meta.setdefault("slow_request_attempted", False)
    meta.setdefault("slow_request_valid_return", False)
    meta.setdefault("slow_request_failed", False)
    meta.setdefault("closed_loop_latency_issuance_event", False)
    meta.setdefault("closed_loop_latency_issued_request_id", "")
    meta.setdefault("closed_loop_latency_issued_response_outcome", "")
    meta.setdefault("closed_loop_latency_terminal_event", False)
    meta.setdefault("closed_loop_latency_terminal_request_id", "")
    meta.setdefault("closed_loop_latency_terminal_response_outcome", "")
    meta.setdefault("closed_loop_latency_release_event", False)
    meta.setdefault("closed_loop_latency_timeout_event", False)
    meta.setdefault("closed_loop_latency_failure_event", False)
    meta.setdefault("closed_loop_release_snapshot_captured", False)
    meta.setdefault("closed_loop_release_action_alignment_evaluated", False)
    meta.setdefault("closed_loop_release_action_alignment_pass", False)
    meta.setdefault("closed_loop_release_opportunity_rejected", False)
    meta.setdefault("closed_loop_release_action_unavailable", False)


def _annotate_pending(
    meta: MutableMapping[str, Any], pending: Mapping[str, Any], *, frame: Optional[int] = None
) -> None:
    _metadata_defaults(meta)
    timing = dict(pending.get("latency_contract", {}) or {})
    source_frame = int(pending.get("source_frame", 0) or 0)
    observed_frame = source_frame if frame is None else int(frame)
    meta.update(
        {
            "closed_loop_latency_request_id": str(pending.get("request_id", "") or ""),
            "closed_loop_latency_response_outcome": str(pending.get("response_outcome", "") or ""),
            "closed_loop_latency_terminal_outcome": "pending",
            "closed_loop_latency_source_frame": source_frame,
            "closed_loop_latency_source_system": "slow",
            "closed_loop_latency_scheduled_steps": int(pending.get("scheduled_steps", 0) or 0),
            "closed_loop_latency_delay_steps": int(pending.get("scheduled_steps", 0) or 0),
            "closed_loop_latency_scheduled_release_frame": int(
                pending.get("release_frame", source_frame) or source_frame
            ),
            "closed_loop_latency_predicted_seconds": float(
                timing.get("predicted_seconds", 0.0) or 0.0
            ),
            "closed_loop_latency_predicted_steps": int(
                timing.get("predicted_steps", 0) or 0
            ),
            "closed_loop_latency_scheduled_seconds": float(
                timing.get("scheduled_seconds", 0.0) or 0.0
            ),
            "closed_loop_latency_policy_frequency_hz": float(
                timing.get("policy_frequency_hz", 1.0) or 1.0
            ),
            "closed_loop_latency_contract_version": str(
                timing.get("version", "") or ""
            ),
            "closed_loop_latency_contract_source": str(
                timing.get("source", "") or ""
            ),
            "closed_loop_latency_scripted_sample": bool(
                timing.get("scripted_sample", False)
            ),
            "closed_loop_latency_pending_age_steps": max(
                0, observed_frame - source_frame
            ),
            "closed_loop_latency_realized_steps": -1,
            "closed_loop_latency_realized_seconds": float("nan"),
            "closed_loop_latency_realized_available": False,
            "closed_loop_latency_realized_source": "not_released",
        }
    )


def _apply_terminal_request(
    *,
    frame: int,
    action: int,
    meta: MutableMapping[str, Any],
    episode_state: MutableMapping[str, Any],
    pending: Mapping[str, Any],
    cfg: Mapping[str, Any],
    agent: Any = None,
    driving_state: Any = None,
) -> int:
    request_id = str(pending.get("request_id", "") or "").strip()
    if not request_id:
        raise RuntimeError("selected queued latency request is missing its request ID")
    terminal_ids = episode_state.setdefault("_latency_terminal_request_ids", set())
    if request_id in terminal_ids:
        raise RuntimeError(f"queued request {request_id} was already terminal")
    outcome = str(pending.get("response_outcome", "valid") or "valid").lower()
    if outcome not in _TERMINAL_OUTCOMES:
        raise RuntimeError(f"queued request has invalid terminal outcome: {outcome!r}")
    snapshots = dict(episode_state.get("release_snapshots", {}) or {})
    snapshot = snapshots.get(request_id)
    require_snapshot = bool(dict(cfg or {}).get("require_release_snapshot_on_release", False))
    if outcome == "valid" and require_snapshot and snapshot is None:
        raise RuntimeError(f"queued request {request_id} has no online snapshot")
    if outcome == "valid" and snapshot is not None:
        validate_release_snapshot_policy_state(snapshot, context=f"queued request {request_id}")
        if str(getattr(snapshot, "request_id", "") or "") != request_id:
            raise RuntimeError(
                f"queued request {request_id} has a mismatched online snapshot"
            )

    released: Optional[int] = None
    release_audit: Optional[Dict[str, Any]] = None
    if outcome == "valid":
        released = int(
            pending.get("raw_slow_action", pending.get("released_action", action))
            if pending.get("raw_slow_action", pending.get("released_action")) is not None
            else action
        )
        replay_cfg = dict(dict(cfg or {}).get("closed_loop_latency_replay", {}) or {})
        revalidation_cfg = dict(
            replay_cfg.get("release_opportunity_revalidation", {}) or {}
        )
        revalidation_enabled = bool(revalidation_cfg.get("enable", False))
        orchestrator = getattr(agent, "orchestrator", None)
        evaluator = getattr(orchestrator, "evaluate_release_proposal", None)
        if callable(evaluator) and driving_state is not None:
            release_audit = dict(
                evaluator(
                    state=driving_state,
                    fast_action=int(action),
                    slow_action=int(released),
                    revalidation_enabled=revalidation_enabled,
                )
            )
        else:
            available = None
            getter = getattr(driving_state, "get_available_actions", None)
            if callable(getter):
                available = {int(item) for item in getter()}
            action_available = available is None or int(released) in available
            release_pass = bool(action_available and not revalidation_enabled)
            release_audit = {
                "selected_action": int(released) if release_pass else int(action),
                "mapped_slow_action": int(released),
                "action_available": bool(action_available),
                "revalidation_enabled": bool(revalidation_enabled),
                "a_pass": bool(action_available and not revalidation_enabled),
                "h_pass": bool(not revalidation_enabled),
                "n_pass": bool(not revalidation_enabled),
                "n_required": False,
                "distinct": bool(int(released) != int(action)),
                "alignment_evaluated": False,
                "alignment_pass": bool(not revalidation_enabled),
                "alignment_margin": 0.0,
                "fast_cost": None,
                "slow_cost": None,
                "required_method_version": "",
                "observed_method_version": "",
                "method_version_pass": bool(not revalidation_enabled),
                "required_raw_cost_source": "",
                "observed_raw_cost_source": "",
                "raw_cost_source_pass": bool(not revalidation_enabled),
                "cost_provenance_pass": bool(not revalidation_enabled),
                "raw_cost_complete": False,
                "release_pass": bool(release_pass),
            }

        required_audit_fields = {
            "selected_action",
            "mapped_slow_action",
            "action_available",
            "revalidation_enabled",
            "a_pass",
            "h_pass",
            "n_pass",
            "n_required",
            "distinct",
            "alignment_evaluated",
            "alignment_pass",
            "alignment_margin",
            "fast_cost",
            "slow_cost",
            "required_method_version",
            "observed_method_version",
            "method_version_pass",
            "required_raw_cost_source",
            "observed_raw_cost_source",
            "raw_cost_source_pass",
            "cost_provenance_pass",
            "raw_cost_complete",
            "release_pass",
        }
        missing = sorted(required_audit_fields - set(release_audit))
        if missing:
            raise RuntimeError(
                "release evaluator returned an incomplete audit: " + ", ".join(missing)
            )
        mapped = int(release_audit["mapped_slow_action"])
        selected = int(release_audit["selected_action"])
        release_pass = bool(release_audit["release_pass"])
        if mapped != int(released):
            raise RuntimeError("release evaluator changed the latched slow action identity")
        expected_selected = int(released) if release_pass else int(action)
        if selected != expected_selected:
            raise RuntimeError("release evaluator selected action disagrees with release_pass")
        alignment_margin = float(release_audit["alignment_margin"])
        if not math.isfinite(alignment_margin) or alignment_margin < 0.0:
            raise RuntimeError("release evaluator returned an invalid alignment margin")

    source_frame = pending.get("source_frame", frame)
    if source_frame is None:
        source_frame = frame
    source_frame = int(source_frame)
    if int(frame) < source_frame:
        raise RuntimeError("queued request became terminal before its source frame")
    realized = max(0, int(frame) - source_frame)
    timing = dict(pending.get("latency_contract", {}) or {})
    frequency = float(timing.get("policy_frequency_hz", 1.0) or 1.0)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise RuntimeError("queued request has an invalid policy frequency")
    scheduled_steps = int(pending.get("scheduled_steps", 0) or 0)
    if scheduled_steps < 0:
        raise RuntimeError("queued request has negative scheduled latency")
    release_frame = int(
        pending.get("release_frame", source_frame + scheduled_steps)
    )
    if release_frame < source_frame:
        raise RuntimeError("queued request release frame precedes its source frame")
    if release_frame - source_frame != scheduled_steps:
        raise RuntimeError("queued request release frame disagrees with scheduled latency")
    predicted_seconds = float(timing.get("predicted_seconds", 0.0) or 0.0)
    predicted_steps = int(timing.get("predicted_steps", 0) or 0)
    scheduled_seconds = float(
        timing.get("scheduled_seconds", scheduled_steps / frequency) or 0.0
    )
    if (
        not math.isfinite(predicted_seconds)
        or predicted_seconds < 0.0
        or predicted_steps < 0
        or not math.isfinite(scheduled_seconds)
        or scheduled_seconds < 0.0
    ):
        raise RuntimeError("queued request has invalid latency provenance")
    response_sha256 = str(pending.get("response_sha256", "") or "")
    if response_sha256 and (
        len(response_sha256) != 64
        or any(character not in "0123456789abcdef" for character in response_sha256)
    ):
        raise RuntimeError("queued request has an invalid response SHA256")

    # Commit only after every snapshot, action, latency, and hash field has
    # passed validation so an exception cannot consume request authority.
    queue = list(episode_state.get("latency_replay_queue", []) or [])
    removed = False
    retained = []
    for item in queue:
        if not removed and str(dict(item).get("request_id", "") or "") == request_id:
            removed = True
            continue
        retained.append(item)
    if not removed:
        raise RuntimeError(f"selected queued request {request_id} is no longer pending")
    episode_state["latency_replay_queue"] = retained
    terminal_ids.add(request_id)

    _metadata_defaults(meta)
    meta.update(
        {
            "closed_loop_latency_request_id": request_id,
            "closed_loop_latency_response_outcome": outcome,
            "closed_loop_latency_terminal_outcome": outcome,
            "closed_loop_latency_terminal_event": True,
            "closed_loop_latency_terminal_request_id": request_id,
            "closed_loop_latency_terminal_response_outcome": outcome,
            "closed_loop_latency_source_frame": source_frame,
            "closed_loop_latency_source_system": "slow",
            "closed_loop_latency_scheduled_steps": scheduled_steps,
            "closed_loop_latency_delay_steps": scheduled_steps,
            "closed_loop_latency_scheduled_release_frame": release_frame,
            "closed_loop_latency_predicted_seconds": predicted_seconds,
            "closed_loop_latency_predicted_steps": predicted_steps,
            "closed_loop_latency_scheduled_seconds": scheduled_seconds,
            "closed_loop_latency_policy_frequency_hz": float(frequency),
            "closed_loop_latency_contract_version": str(
                timing.get("version", "") or ""
            ),
            "closed_loop_latency_contract_source": str(
                timing.get("source", "") or ""
            ),
            "closed_loop_latency_scripted_sample": bool(
                timing.get("scripted_sample", False)
            ),
            "closed_loop_latency_realized_steps": realized,
            "closed_loop_latency_realized_seconds": float(realized / frequency),
            "closed_loop_latency_realized_available": True,
            "closed_loop_latency_realized_source": "simulator_frame_delta",
            "closed_loop_release_snapshot_captured": bool(snapshot is not None),
            "closed_loop_latency_response_sha256": response_sha256,
        }
    )
    if snapshot is not None:
        meta["closed_loop_release_snapshot_identity_sha256"] = str(getattr(snapshot, "snapshot_identity_sha256", "") or "")
    if outcome == "valid":
        if released is None or release_audit is None:
            raise RuntimeError("valid queued response has no release audit")
        selected = int(release_audit["selected_action"])
        meta.update(
            {
                "closed_loop_latency_release_event": True,
                "closed_loop_latency_timeout_event": False,
                "closed_loop_latency_failure_event": False,
                "closed_loop_released_slow_action": released,
                "closed_loop_execution_state_fast_action": int(action),
                "closed_loop_latency_executed_action": selected,
                "release_fast_comparator_action": int(action),
                "release_selected_action": selected,
                "release_action_comparison_stage": "post_release_guard_pre_final_safety_projection",
                "closed_loop_release_action_unavailable": not bool(
                    release_audit["action_available"]
                ),
                "closed_loop_release_revalidation_enabled": bool(
                    release_audit["revalidation_enabled"]
                ),
                "closed_loop_release_revalidation_a_pass": bool(
                    release_audit["a_pass"]
                ),
                "closed_loop_release_revalidation_h_pass": bool(
                    release_audit["h_pass"]
                ),
                "closed_loop_release_revalidation_n_pass": bool(
                    release_audit["n_pass"]
                ),
                "closed_loop_release_revalidation_n_required": bool(
                    release_audit["n_required"]
                ),
                "closed_loop_release_revalidation_all_pass": bool(
                    release_audit["a_pass"]
                    and release_audit["h_pass"]
                    and (
                        release_audit["n_pass"]
                        or not release_audit["n_required"]
                    )
                ),
                "closed_loop_release_action_alignment_evaluated": bool(
                    release_audit["alignment_evaluated"]
                ),
                "closed_loop_release_action_alignment_pass": bool(
                    release_audit["alignment_pass"]
                ),
                "closed_loop_release_action_alignment_margin": float(
                    release_audit["alignment_margin"]
                ),
                "closed_loop_release_action_alignment_fast_cost": release_audit[
                    "fast_cost"
                ],
                "closed_loop_release_action_alignment_slow_cost": release_audit[
                    "slow_cost"
                ],
                "closed_loop_release_action_alignment_required_method_version": str(
                    release_audit["required_method_version"]
                ),
                "closed_loop_release_action_alignment_observed_method_version": str(
                    release_audit["observed_method_version"]
                ),
                "closed_loop_release_action_alignment_method_version_pass": bool(
                    release_audit["method_version_pass"]
                ),
                "closed_loop_release_action_alignment_required_raw_cost_source": str(
                    release_audit["required_raw_cost_source"]
                ),
                "closed_loop_release_action_alignment_observed_raw_cost_source": str(
                    release_audit["observed_raw_cost_source"]
                ),
                "closed_loop_release_action_alignment_raw_cost_source_pass": bool(
                    release_audit["raw_cost_source_pass"]
                ),
                "closed_loop_release_action_alignment_cost_provenance_pass": bool(
                    release_audit["cost_provenance_pass"]
                ),
                "closed_loop_release_action_alignment_raw_cost_complete": bool(
                    release_audit["raw_cost_complete"]
                ),
                "closed_loop_release_action_distinct": bool(
                    release_audit["distinct"]
                ),
                "closed_loop_release_opportunity_rejected": not bool(
                    release_audit["release_pass"]
                ),
                "closed_loop_release_guard_pass": bool(
                    release_audit["release_pass"]
                ),
                "closed_loop_release_actuation_distinct": bool(
                    selected != int(action)
                ),
            }
        )
        return selected
    meta.update(
        {
            "closed_loop_latency_release_event": False,
            "closed_loop_latency_timeout_event": outcome == "timeout",
            "closed_loop_latency_failure_event": outcome == "failure",
            "slow_reasoning_failure_reason": outcome,
        }
    )
    return int(action)


def _enqueue_request(
    *,
    frame: int,
    action: int,
    meta: MutableMapping[str, Any],
    episode_state: MutableMapping[str, Any],
    contract: Mapping[str, Any],
    outcome: str,
    scheduled_steps: int,
    issue_source: Optional[Mapping[str, Any]] = None,
) -> int:
    source = dict(issue_source or meta)
    request_id = str(
        source.get("closed_loop_latency_request_id", source.get("slow_request_id", ""))
        or ""
    ).strip()
    if not request_id:
        request_id = _make_request_id(frame, episode_state)
    request_ids = episode_state.setdefault("_latency_request_ids", set())
    if request_id in request_ids:
        raise RuntimeError(f"duplicate episode latency request ID: {request_id}")
    fast_action = int(source.get("query_state_fast_proposal_action", action) if source.get("query_state_fast_proposal_action") is not None else action)
    raw_slow_action = int(
        source.get("query_state_slow_pre_guard_action", action)
        if source.get("query_state_slow_pre_guard_action") is not None
        else action
    )
    released_action = int(source.get("query_state_slow_released_action", action) if source.get("query_state_slow_released_action") is not None else action)
    request_contract = _request_latency_contract(contract, source)
    scheduled_steps = int(request_contract["scheduled_steps"])
    release_frame = int(frame) + scheduled_steps
    queue_item = {
        "request_id": request_id,
        "source_frame": int(frame),
        "release_frame": release_frame,
        "available_frame": max(int(frame) + 1, release_frame),
        "scheduled_steps": int(scheduled_steps),
        "response_outcome": str(outcome),
        "raw_slow_action": raw_slow_action,
        "released_action": released_action,
        "fast_action": fast_action,
        "response_sha256": str(
            source.get("factorial_shared_response_sha256")
            or source.get("closed_loop_latency_response_sha256", "")
            or ""
        ),
        "latency_contract": {
            key: request_contract.get(key)
            for key in (
                "version",
                "predicted_seconds",
                "predicted_steps",
                "scheduled_seconds",
                "scheduled_steps",
                "policy_frequency_hz",
                "source",
                "scripted_sample",
                "prediction_available",
                "configured_steps_consistent",
            )
        },
    }
    response_sha256 = str(queue_item["response_sha256"] or "")
    if response_sha256 and (
        len(response_sha256) != 64
        or any(character not in "0123456789abcdef" for character in response_sha256)
    ):
        raise RuntimeError("slow response has an invalid SHA256")
    request_ids.add(request_id)
    episode_state.setdefault("latency_replay_queue", []).append(queue_item)
    _metadata_defaults(meta)
    meta.update(
        {
            "closed_loop_latency_request_id": request_id,
            "slow_request_id": request_id,
            "closed_loop_latency_issuance_event": True,
            "closed_loop_latency_issued_request_id": request_id,
            "closed_loop_latency_issued_response_outcome": str(outcome),
            "closed_loop_latency_response_outcome": str(outcome),
            "closed_loop_latency_terminal_outcome": "pending",
            "closed_loop_latency_scheduled_steps": int(scheduled_steps),
            "closed_loop_latency_delay_steps": int(scheduled_steps),
            "closed_loop_latency_scheduled_release_frame": int(release_frame),
            "closed_loop_latency_predicted_seconds": float(
                request_contract.get("predicted_seconds", 0.0) or 0.0
            ),
            "closed_loop_latency_predicted_steps": int(
                request_contract.get("predicted_steps", 0) or 0
            ),
            "closed_loop_latency_scheduled_seconds": float(
                request_contract.get("scheduled_seconds", 0.0) or 0.0
            ),
            "closed_loop_latency_policy_frequency_hz": float(
                request_contract.get("policy_frequency_hz", 1.0) or 1.0
            ),
            "closed_loop_latency_contract_version": str(
                request_contract.get("version", "") or ""
            ),
            "closed_loop_latency_contract_source": str(
                request_contract.get("source", "") or ""
            ),
            "closed_loop_latency_scripted_sample": bool(
                request_contract.get("scripted_sample", False)
            ),
            "closed_loop_latency_configured_steps_consistent": bool(
                request_contract.get("configured_steps_consistent", False)
            ),
            "closed_loop_latency_contract_match_available": bool(
                request_contract.get("prediction_available", False)
            ),
            "closed_loop_latency_contract_match": bool(
                request_contract.get("prediction_available", False)
                and int(request_contract.get("predicted_steps", 0) or 0)
                == int(scheduled_steps)
            ),
            "closed_loop_latency_source_frame": int(frame),
            "closed_loop_latency_source_system": "slow",
            "closed_loop_latency_realized_steps": -1,
            "closed_loop_latency_realized_seconds": float("nan"),
            "closed_loop_latency_realized_available": False,
            "closed_loop_latency_realized_source": "not_released",
        }
    )
    return fast_action


def _apply_closed_loop_latency_replay(
    *,
    frame: int,
    action: int,
    decision_meta: MutableMapping[str, Any],
    episode_state: MutableMapping[str, Any],
    cfg: Mapping[str, Any],
    agent: Any = None,
    driving_state: Any = None,
) -> int:
    """Apply one asynchronous request/release transition around a Fast action."""
    _ensure_latency_state(episode_state)
    meta = decision_meta
    _metadata_defaults(meta)
    if bool(meta.get("native_async_slow_path", False)):
        return int(action)
    contract = resolve_latency_contract(dict(cfg or {}))
    if not bool(contract.get("replay_enabled", False)):
        return int(action)

    issue_source = copy.deepcopy(dict(meta))
    issuable, outcome, scheduled_steps = _request_is_issuable(issue_source, contract)
    invalid_prediction = bool(contract.get("prediction_invalid_reason"))

    # A due request is always resolved before a same-frame request is added.
    pending = _selected_ready_latency_request(episode_state, frame=int(frame))
    executed = int(action)
    if pending is not None:
        executed = _apply_terminal_request(
            frame=int(frame),
            action=executed,
            meta=meta,
            episode_state=episode_state,
            pending=pending,
            cfg=cfg,
            agent=agent,
            driving_state=driving_state,
        )
    elif not issuable:
        queued = list(episode_state.get("latency_replay_queue", []) or [])
        if queued:
            earliest = min(
                queued,
                key=lambda item: (
                    _ready_frame(dict(item)),
                    int(dict(item).get("source_frame", 0) or 0),
                    str(dict(item).get("request_id", "")),
                ),
            )
            _annotate_pending(meta, earliest, frame=int(frame))

    if issuable and outcome is not None and pending is None:
        queued = list(episode_state.get("latency_replay_queue", []) or [])
        if queued:
            # The runtime contract admits at most one live slow request.  A
            # scripted proposal observed while that request is pending cannot
            # create a second request or consume a second response.
            _annotate_pending(meta, queued[0], frame=int(frame))
            blocked_fast = issue_source.get("query_state_fast_proposal_action")
            return int(executed if blocked_fast is None else blocked_fast)
        fallback = _enqueue_request(
            frame=int(frame),
            action=int(action),
            meta=meta,
            episode_state=episode_state,
            contract=contract,
            outcome=outcome,
            scheduled_steps=scheduled_steps,
            issue_source=issue_source,
        )
        executed = fallback
    elif (
        bool(issue_source.get("slow_request_attempted", False))
        and bool(contract.get("replay_enabled", False))
        and invalid_prediction
        and pending is None
    ):
        query_fast = issue_source.get("query_state_fast_proposal_action")
        if query_fast is not None:
            executed = int(query_fast)
        meta["closed_loop_latency_invalid_prediction_fallback"] = True
    return int(executed)


def _capture_online_release_snapshot_if_due(
    *,
    frame: int,
    env: Any,
    obs: Any,
    agent: Any,
    history_buffer: deque,
    episode_state: MutableMapping[str, Any],
    cfg: Mapping[str, Any],
    pending_request: Optional[Mapping[str, Any]] = None,
) -> None:
    """Capture exactly the request-bound state before a valid response releases."""
    _ensure_latency_state(episode_state)
    if not bool(dict(cfg or {}).get("capture_release_snapshots_online", False)):
        return
    pending = (
        dict(pending_request)
        if pending_request is not None
        else _selected_ready_latency_request(episode_state, frame=int(frame))
    )
    if pending is None or str(pending.get("response_outcome", "") or "") != "valid":
        return
    request_id = str(pending.get("request_id", "") or "")
    if not request_id:
        raise RuntimeError("selected queued latency request is missing its request ID")
    snapshots = episode_state.setdefault("release_snapshots", {})
    if request_id in snapshots:
        return
    snapshots[request_id] = capture_release_snapshot(
        agent,
        frame=int(frame),
        env=env,
        obs=obs,
        history=history_buffer,
        previous_action=int(episode_state.get("action", int(ActionType.IDLE)) or int(ActionType.IDLE)),
        pending_request=pending,
    )


def _safety_state(runtime_state: Mapping[str, Any]) -> Dict[str, Any]:
    state = dict(runtime_state or {})
    state.setdefault("front_dist", state.get("front_distance", float("inf")))
    state.setdefault("cross_traffic_dist", state.get("cross_traffic_distance", float("inf")))
    for side in ("left", "right"):
        state.setdefault(f"{side}_front", state.get(f"{side}_front_dist", float("inf")))
        state.setdefault(f"{side}_rear", state.get(f"{side}_rear_dist", float("inf")))
    return state


def _scenario_description(sce: Any, frame: int) -> str:
    describe = getattr(sce, "describe", None)
    if callable(describe):
        try:
            return str(describe(int(frame)))
        except (AttributeError, TypeError, ValueError):
            pass
    return ""


def run_frame_protocol(
    *,
    frame: int,
    env: Any,
    sce: Any,
    agent: Any,
    obs: Any,
    prev_action: int,
    cfg: MutableMapping[str, Any],
    safety_system: Any,
    phys_rec: Any,
    reas_rec: Any,
    history_buffer: deque,
    prev_frame_image: Any = None,
) -> Tuple[int, str, str, Dict[str, Any], float, float, Dict[str, Any], Any, Any, str, Dict[str, Any]]:
    """Evaluate the current policy state before release and safety projection."""
    del obs, phys_rec, reas_rec, prev_frame_image
    t0 = time.perf_counter()
    runtime_state = extract_runtime_state(env, sce, cfg)
    cfg["_current_runtime_state"] = dict(runtime_state)
    raw_actions = _raw_available_actions(env)
    actions = _inject_hidden_slower_action(raw_actions, runtime_state, cfg)
    driving_state = build_frame_driving_state(
        runtime_state,
        str(runtime_state.get("scenario_type", "highway")),
        cfg,
        history_buffer,
        prev_action=int(prev_action),
        env_available_actions=actions,
    )
    driving_state.set_effective_action_universe(actions, source="runtime_effective_action_universe")
    setattr(driving_state, "_runtime_frame_id", int(frame))
    inject_safety_cost_bridge(driving_state, env, runtime_state, safety_system)
    description = _scenario_description(sce, frame)
    question = _available_actions_description(actions)
    proposed_action, response, raw_meta = resolve_agent_action(agent, cfg, driving_state)
    proposed_action = int(proposed_action)
    if proposed_action not in actions:
        proposed_action = int(ActionType.IDLE) if int(ActionType.IDLE) in actions else int(actions[0])
    decision_meta = build_decision_meta(raw_meta, proposed_action=proposed_action, final_action=proposed_action)
    decision_meta.update(
        {
            "final_action": proposed_action,
            "_runtime_state": dict(runtime_state),
            "_runtime_available_actions": list(actions),
        }
    )
    t_inf = time.perf_counter()
    image = None
    if _resolve_render_mode(cfg) is not None:
        try:
            image = env.render()
        except (AttributeError, TypeError, ValueError):
            image = None
    return (
        proposed_action,
        question,
        str(response),
        runtime_state,
        t0,
        t_inf,
        dict(raw_meta or {}),
        image,
        driving_state,
        description,
        decision_meta,
    )


def advance_episode_frame(
    *,
    frame: int,
    env: Any,
    sce: Any,
    q: str,
    resp: str,
    action: int,
    t0: float,
    t_inf: float,
    safety_system: Any,
    phys_rec: Any,
    collision_frame: int,
    decision_meta: MutableMapping[str, Any],
    cfg: Mapping[str, Any],
) -> Tuple[Any, int, bool, Dict[str, Any], float]:
    """Apply the selected command and return normalized terminal information."""
    del sce, q, resp, safety_system, t0, t_inf
    selected = int(action)
    executed = _apply_unavailable_slower_brake_assist(env, selected, cfg)
    hidden_brake_assist = bool(executed != selected and selected == int(ActionType.SLOWER))
    executed = _release_highway_hidden_brake_target(env, int(executed), cfg)
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    bridge_target_speed = (
        None
        if vehicle is None
        else _float(getattr(vehicle, "target_speed", None), float("nan"))
    )
    decision_meta.update(
        {
            "pre_actuator_action": selected,
            "final_actuator_action": int(executed),
            "final_actuator_action_stage": (
                "post_shared_actuator_bridge_pre_environment_step"
            ),
            "actuator_action_rewritten": bool(int(executed) != selected),
            "hidden_slower_brake_assist": bool(hidden_brake_assist),
            "hidden_slower_brake_target_speed": bridge_target_speed,
        }
    )
    step = env.step(int(executed))
    if len(step) == 5:
        next_obs, reward, terminated, truncated, info = step
    elif len(step) == 4:
        next_obs, reward, done, info = step
        terminated, truncated = bool(done), False
    else:
        raise RuntimeError("environment step must return four or five values")
    info = dict(info or {})
    crashed = bool(info.get("crashed", info.get("crash", False)))
    done = bool(terminated or truncated)
    if crashed and int(collision_frame) < 0:
        collision_frame = int(frame)
    terminal = {
        "done": done,
        "term": bool(terminated),
        "trunc": bool(truncated),
        "crashed": crashed,
        "terminal_cause": "collision" if crashed else "terminated" if terminated else "truncated" if truncated else "running",
        "reward": float(reward),
    }
    decision_meta["final_action"] = int(executed)
    decision_meta["closed_loop_latency_executed_action"] = int(executed)
    runtime_state = dict(decision_meta.get("_runtime_state", {}) or {})
    if phys_rec is not None:
        phys_rec.record_frame(
            int(frame), runtime_state, action=int(executed), reward=float(reward), crashed=crashed, done=done, info=info
        )
    return next_obs, int(collision_frame), done, terminal, float(reward)


def execute_episode_step(
    *,
    frame: int,
    env: Any,
    sce: Any,
    agent: Any,
    obs: Any,
    cfg: MutableMapping[str, Any],
    safety_system: Any,
    phys_rec: Any,
    reas_rec: Any,
    history_buffer: deque,
    episode_state: MutableMapping[str, Any],
) -> Tuple[Any, bool]:
    """Execute one full closed-loop frame and append its event provenance."""
    pacing_meta = _pace_native_async_policy_frame(
        frame=int(frame), agent=agent, episode_state=episode_state
    )
    _ensure_latency_state(episode_state)
    prepare = getattr(agent, "prepare_frame", None)
    native_terminal = prepare(int(frame)) if callable(prepare) else None
    scripted_terminal = _selected_ready_latency_request(
        episode_state, frame=int(frame)
    )
    if native_terminal is not None and scripted_terminal is not None:
        raise RuntimeError(
            "native asynchronous and scripted latency terminals coexist on one frame"
        )
    if native_terminal is not None:
        if not isinstance(native_terminal, Mapping):
            raise RuntimeError("native asynchronous terminal descriptor must be a mapping")
        native_request_id = str(native_terminal.get("request_id", "") or "").strip()
        native_outcome = str(
            native_terminal.get("response_outcome", "") or ""
        ).lower()
        if not native_request_id:
            raise RuntimeError("native asynchronous terminal is missing its request ID")
        if not bool(native_terminal.get("native_async", False)):
            raise RuntimeError(
                "non-native descriptor was returned by the native async agent"
            )
        if native_outcome not in _TERMINAL_OUTCOMES:
            raise RuntimeError("native asynchronous terminal has an invalid outcome")
    release_descriptor = native_terminal or scripted_terminal
    _capture_online_release_snapshot_if_due(
        frame=int(frame),
        env=env,
        obs=obs,
        agent=agent,
        history_buffer=history_buffer,
        episode_state=episode_state,
        cfg=cfg,
        pending_request=release_descriptor,
    )
    (
        proposed,
        q,
        response,
        frame_state,
        t0,
        t_inf,
        _raw_meta,
        image,
        driving_state,
        description,
        decision_meta,
    ) = run_frame_protocol(
        frame=int(frame), env=env, sce=sce, agent=agent, obs=obs,
        prev_action=int(episode_state.get("action", int(ActionType.IDLE)) or int(ActionType.IDLE)), cfg=cfg,
        safety_system=safety_system, phys_rec=phys_rec, reas_rec=reas_rec,
        history_buffer=history_buffer, prev_frame_image=episode_state.get("prev_image"),
    )
    decision_meta.update(pacing_meta)
    final_action = _apply_closed_loop_latency_replay(
        frame=int(frame),
        action=int(proposed),
        decision_meta=decision_meta,
        episode_state=episode_state,
        cfg=cfg,
        agent=agent,
        driving_state=driving_state,
    )
    if native_terminal is not None:
        request_id = str(native_terminal.get("request_id", "") or "")
        expected = str(
            decision_meta.get("closed_loop_latency_terminal_request_id", "") or ""
        )
        if not expected or request_id != expected:
            raise RuntimeError(
                "native asynchronous terminal request does not match the frame decision"
            )
        if not bool(decision_meta.get("closed_loop_latency_terminal_event", False)):
            raise RuntimeError(
                "native asynchronous terminal descriptor was not consumed by the frame decision"
            )
        observed_outcome = str(
            decision_meta.get("closed_loop_latency_terminal_response_outcome", "")
            or ""
        ).lower()
        expected_outcome = str(
            native_terminal.get("response_outcome", "") or ""
        ).lower()
        if observed_outcome != expected_outcome:
            raise RuntimeError(
                "native asynchronous terminal outcome does not match the frame decision"
            )
        if bool(decision_meta.get("closed_loop_latency_release_event", False)) != (
            expected_outcome == "valid"
        ):
            raise RuntimeError("native asynchronous release marker disagrees with outcome")
        if bool(decision_meta.get("closed_loop_latency_timeout_event", False)) != (
            expected_outcome == "timeout"
        ):
            raise RuntimeError("native asynchronous timeout marker disagrees with outcome")
        if bool(decision_meta.get("closed_loop_latency_failure_event", False)) != (
            expected_outcome == "failure"
        ):
            raise RuntimeError("native asynchronous failure marker disagrees with outcome")
        snapshot = dict(episode_state.get("release_snapshots", {}) or {}).get(
            request_id
        )
        if snapshot is not None and str(getattr(snapshot, "request_id", "") or "") != request_id:
            raise RuntimeError(
                "native asynchronous release snapshot request ID does not match terminal"
            )
        if snapshot is not None:
            validate_release_snapshot_policy_state(
                snapshot, context=f"native asynchronous request {request_id}"
            )
        decision_meta["closed_loop_release_snapshot_captured"] = bool(snapshot)
        if snapshot is not None:
            decision_meta["closed_loop_release_snapshot_identity_sha256"] = str(
                getattr(snapshot, "snapshot_identity_sha256", "") or ""
            )
        elif bool(decision_meta.get("closed_loop_latency_release_event", False)) and bool(
            dict(cfg or {}).get("require_release_snapshot_on_release", False)
        ):
            raise RuntimeError(
                f"native asynchronous request {request_id} has no online release snapshot"
            )

    actions = [
        int(action)
        for action in list(decision_meta.get("_runtime_available_actions", []) or [])
    ]
    safety_result = safety_system.apply_action_safety_stack(
        int(final_action),
        actions,
        _safety_state(frame_state),
        frame=int(frame),
    )
    final_action = int(safety_result.final_action)
    decision_meta.update(
        {
            "safety_override": bool(safety_result.safety_override),
            "shield_override": bool(safety_result.shield_override),
            "emergency_level": int(safety_result.emergency_level),
            "safety_reason": str(safety_result.safety_reason),
            "shield_reason": str(safety_result.shield_reason),
            "risk_event": bool(safety_result.emergency_level > 0),
            "final_action": int(final_action),
        }
    )
    if reas_rec is not None:
        reas_rec.record_reasoning(
            frame_id=int(frame),
            scenario_description=description,
            available_actions=q,
            driving_intention="Drive safely and make progress.",
            full_response=str(response),
            action_id=int(final_action),
            inference_start_time=t0,
            inference_end_time=time.perf_counter(),
            decision_meta=decision_meta,
        )
    next_obs, collision_frame, done, terminal, reward = advance_episode_frame(
        frame=int(frame), env=env, sce=sce, q=q, resp=response, action=int(final_action), t0=t0, t_inf=t_inf,
        safety_system=safety_system, phys_rec=phys_rec, collision_frame=int(episode_state.get("collision_frame", -1) or -1),
        decision_meta=decision_meta, cfg=cfg,
    )
    final_executed = int(decision_meta.get("final_action", final_action))
    agent.record_executed_action(final_executed)
    record_executed_history_frame(history_buffer, frame_state, final_executed)
    episode_state["action"] = final_executed
    episode_state["collision_frame"] = collision_frame
    episode_state["prev_image"] = image
    episode_state["episode_reward"] = float(episode_state.get("episode_reward", 0.0) or 0.0) + reward
    event = build_episode_event(int(frame), frame_state, decision_meta, terminal)
    episode_state.setdefault("event_log", []).append(event)
    return next_obs, bool(done)


__all__ = [
    "_apply_closed_loop_latency_replay",
    "_apply_unavailable_slower_brake_assist",
    "_available_actions_description",
    "_capture_online_release_snapshot_if_due",
    "_inject_hidden_slower_action",
    "_pace_native_async_policy_frame",
    "_release_highway_hidden_brake_target",
    "_resolve_render_mode",
    "_selected_ready_latency_request",
    "advance_episode_frame",
    "execute_episode_step",
    "exclude_policy_pacing_sleep",
    "run_frame_protocol",
]
