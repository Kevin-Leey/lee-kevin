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


def _resolve_render_mode(cfg: Mapping[str, Any]) -> Optional[str]:
    value = str(dict(cfg or {}).get("render_mode", "") or "").strip().lower()
    return None if value in {"", "none", "off", "false", "0"} else value


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
    target_speeds = list(getattr(vehicle, "target_speeds", []) or [])
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
    raw = meta.get("closed_loop_scripted_latency_steps")
    if raw is None:
        raw = contract.get("scheduled_steps", 0)
    try:
        steps = int(raw)
    except (TypeError, ValueError):
        return int(contract.get("scheduled_steps", 0) or 0)
    return max(0, steps)


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


def _annotate_pending(meta: MutableMapping[str, Any], pending: Mapping[str, Any]) -> None:
    _metadata_defaults(meta)
    meta.update(
        {
            "closed_loop_latency_request_id": str(pending.get("request_id", "") or ""),
            "closed_loop_latency_response_outcome": str(pending.get("response_outcome", "") or ""),
            "closed_loop_latency_terminal_outcome": "pending",
            "closed_loop_latency_scheduled_steps": int(pending.get("scheduled_steps", 0) or 0),
            "closed_loop_latency_realized_steps": max(0, int(pending.get("last_observed_frame", pending.get("source_frame", 0)) or 0) - int(pending.get("source_frame", 0) or 0)),
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
) -> int:
    request_id = str(pending.get("request_id", "") or "")
    if not request_id:
        raise RuntimeError("selected queued latency request is missing its request ID")
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

    # Only remove after all validation succeeded so callers can retry safely.
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
    episode_state.setdefault("_latency_terminal_request_ids", set()).add(request_id)

    _metadata_defaults(meta)
    source_frame = pending.get("source_frame", frame)
    if source_frame is None:
        source_frame = frame
    realized = max(0, int(frame) - int(source_frame))
    meta.update(
        {
            "closed_loop_latency_request_id": request_id,
            "closed_loop_latency_response_outcome": outcome,
            "closed_loop_latency_terminal_outcome": outcome,
            "closed_loop_latency_terminal_event": True,
            "closed_loop_latency_terminal_request_id": request_id,
            "closed_loop_latency_terminal_response_outcome": outcome,
            "closed_loop_latency_scheduled_steps": int(pending.get("scheduled_steps", 0) or 0),
            "closed_loop_latency_realized_steps": realized,
            "closed_loop_release_snapshot_captured": bool(snapshot is not None),
        }
    )
    if snapshot is not None:
        meta["closed_loop_release_snapshot_identity_sha256"] = str(getattr(snapshot, "snapshot_identity_sha256", "") or "")
    if outcome == "valid":
        released = int(pending.get("released_action", pending.get("raw_slow_action", action)) or action)
        meta.update(
            {
                "closed_loop_latency_release_event": True,
                "closed_loop_latency_timeout_event": False,
                "closed_loop_latency_failure_event": False,
                "closed_loop_released_slow_action": released,
                "closed_loop_execution_state_fast_action": int(action),
                "closed_loop_latency_executed_action": released,
                "release_fast_comparator_action": int(action),
                "release_selected_action": released,
                "release_action_comparison_stage": "post_release_guard_and_frame_safety_pre_actuator_bridge",
                "closed_loop_release_actuation_distinct": bool(released != int(action)),
            }
        )
        return released
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
    request_id = str(source.get("closed_loop_latency_request_id", source.get("slow_request_id", "")) or "")
    if not request_id:
        request_id = _make_request_id(frame, episode_state)
    request_ids = episode_state.setdefault("_latency_request_ids", set())
    if request_id in request_ids:
        raise RuntimeError(f"duplicate episode latency request ID: {request_id}")
    request_ids.add(request_id)
    fast_action = int(source.get("query_state_fast_proposal_action", action) if source.get("query_state_fast_proposal_action") is not None else action)
    released_action = int(source.get("query_state_slow_released_action", action) if source.get("query_state_slow_released_action") is not None else action)
    release_frame = int(frame) + int(scheduled_steps)
    queue_item = {
        "request_id": request_id,
        "source_frame": int(frame),
        "release_frame": release_frame,
        "available_frame": max(int(frame) + 1, release_frame),
        "scheduled_steps": int(scheduled_steps),
        "response_outcome": str(outcome),
        "raw_slow_action": int(action),
        "released_action": released_action,
        "fast_action": fast_action,
    }
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
            "closed_loop_latency_realized_steps": 0,
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
) -> int:
    """Apply one asynchronous request/release transition around a Fast action."""
    _ensure_latency_state(episode_state)
    meta = decision_meta
    _metadata_defaults(meta)
    contract = resolve_latency_contract(dict(cfg or {}))
    if not bool(contract.get("replay_enabled", False)):
        return int(action)

    issue_source = copy.deepcopy(dict(meta))
    issuable, outcome, scheduled_steps = _request_is_issuable(issue_source, contract)

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
            _annotate_pending(meta, earliest)

    if issuable and outcome is not None:
        # If a release occupied this frame, it is already terminal; a separate
        # request is appended afterward and cannot alter its authentication.
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
        if pending is None:
            executed = fallback
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
) -> None:
    """Capture exactly the request-bound state before a valid response releases."""
    _ensure_latency_state(episode_state)
    if not bool(dict(cfg or {}).get("capture_release_snapshots_online", False)):
        return
    pending = _selected_ready_latency_request(episode_state, frame=int(frame))
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
    """Evaluate one frame up to, but not including, the environment transition."""
    del obs, phys_rec, prev_frame_image
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
    inject_safety_cost_bridge(driving_state, env, runtime_state, safety_system)
    description = _scenario_description(sce, frame)
    question = _available_actions_description(actions)
    proposed_action, response, raw_meta = resolve_agent_action(agent, cfg, driving_state)
    proposed_action = int(proposed_action)
    if proposed_action not in actions:
        proposed_action = int(ActionType.IDLE) if int(ActionType.IDLE) in actions else int(actions[0])
    decision_meta = build_decision_meta(raw_meta, proposed_action=proposed_action, final_action=proposed_action)
    safety_result = safety_system.apply_action_safety_stack(
        proposed_action,
        list(actions),
        _safety_state(runtime_state),
        frame=int(frame),
    )
    selected_action = int(safety_result.final_action)
    decision_meta.update(
        {
            "safety_override": bool(safety_result.safety_override),
            "shield_override": bool(safety_result.shield_override),
            "emergency_level": int(safety_result.emergency_level),
            "safety_reason": str(safety_result.safety_reason),
            "shield_reason": str(safety_result.shield_reason),
            "risk_event": bool(safety_result.emergency_level > 0),
            "final_action": selected_action,
            "_runtime_state": dict(runtime_state),
        }
    )
    if reas_rec is not None:
        reas_rec.record_reasoning(
            frame_id=int(frame),
            scenario_description=description,
            available_actions=question,
            driving_intention="Drive safely and make progress.",
            full_response=str(response),
            action_id=selected_action,
            inference_start_time=t0,
            inference_end_time=time.perf_counter(),
            decision_meta=decision_meta,
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
    executed = _apply_unavailable_slower_brake_assist(env, int(action), cfg)
    executed = _release_highway_hidden_brake_target(env, int(executed), cfg)
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
    _ensure_latency_state(episode_state)
    _capture_online_release_snapshot_if_due(
        frame=int(frame), env=env, obs=obs, agent=agent, history_buffer=history_buffer, episode_state=episode_state, cfg=cfg
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
        _driving_state,
        _description,
        decision_meta,
    ) = run_frame_protocol(
        frame=int(frame), env=env, sce=sce, agent=agent, obs=obs,
        prev_action=int(episode_state.get("action", int(ActionType.IDLE)) or int(ActionType.IDLE)), cfg=cfg,
        safety_system=safety_system, phys_rec=phys_rec, reas_rec=reas_rec,
        history_buffer=history_buffer, prev_frame_image=episode_state.get("prev_image"),
    )
    final_action = _apply_closed_loop_latency_replay(
        frame=int(frame), action=int(decision_meta.get("final_action", proposed)), decision_meta=decision_meta,
        episode_state=episode_state, cfg=cfg,
    )
    decision_meta["final_action"] = int(final_action)
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
    "_release_highway_hidden_brake_target",
    "_resolve_render_mode",
    "_selected_ready_latency_request",
    "advance_episode_frame",
    "execute_episode_step",
    "run_frame_protocol",
]
