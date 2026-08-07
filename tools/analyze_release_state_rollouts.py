"""Outcome-grounded release-state rollouts for the TVT mechanism claim.

The analysis uses fresh Fast-only trajectories.  Query frames are selected by
the frozen RGD and TTC rules, but the target is independent of either routing
score: at each release state, every runtime-legal high-level action is passed
through the shared safety stack and followed by the same fast controller for a
short horizon.  A corrective option must produce a distinct effective action
and improve a prespecified collision-penalized normalized simulator return.

Experimental unit: simulator seed.  Multiple query/release states within a
seed remain clustered for all confidence intervals.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import math
import os
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.driver_agent.reasoning.fast_thinker import FastThinker
from dilu.driver_agent.driverAgentV2 import DriverAgentV2 as OfflineDriverAgent
from dilu.evaluation.release_snapshot import (
    ReleaseSnapshot,
    capture_release_snapshot,
    validate_release_snapshot_policy_state,
)
from dilu.runtime_episode_setup import create_episode_env
from dilu.runtime_frame_trace import create_episode_runtime_state
from dilu.runtime_support import execute_episode_step
from dilu.safety import UnifiedSafetySystem
from dilu.scenario import create_scenario
from tools.analyze_common_trajectory_allocators import (
    BUDGET,
    COOLDOWN,
    RGD_FLOOR,
    RGD_THRESHOLD,
    TTC_CUTOFF,
    gate_component_values,
    gate_values,
    scheduled_frames,
    ttc_score,
)
from tools.run_main_table_runtime import (
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
)
from tools.v12_floor_overlay import VerifiedFloorOverlay, apply_floor_overlay


RAW_ACTIONS = tuple(range(5))
DEFAULT_DELAYS = (0.7, 1.7, 2.7)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260717


def _query_gate_components(record: Dict[str, Any], delay_s: float) -> Dict[str, Any]:
    """Expose the canonical offline gate components for sibling analyses."""
    return gate_component_values(record, delay_s)


class FastBranchAgent:
    """Fast controller with an optional one-frame counterfactual action."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        policy_state: Optional[Dict[str, Any]] = None,
        fast_state: Optional[Dict[str, Any]] = None,
        allow_legacy_fast_state: bool = False,
        force_frame: Optional[int] = None,
        force_action: Optional[int] = None,
        force_actions: Optional[Dict[int, int]] = None,
    ) -> None:
        self.inner = OfflineDriverAgent(config=copy.deepcopy(dict(cfg or {})))
        if policy_state is not None:
            self.inner.restore_policy_state(copy.deepcopy(policy_state))
        elif fast_state is not None:
            if not allow_legacy_fast_state:
                raise ValueError(
                    "legacy fast-state restore requires explicit diagnostic opt-in"
                )
            self.inner.fast_thinker.restore_runtime_state(copy.deepcopy(fast_state))
        self.fast = self.inner.fast_thinker
        self.force_frame = force_frame
        self.force_action = force_action
        self.force_actions = {
            int(frame): int(action)
            for frame, action in dict(force_actions or {}).items()
        }
        self.frame = 0
        self.fast_actions: Dict[int, int] = {}
        self.legal_actions: Dict[int, Tuple[int, ...]] = {}
        self.last_system_used = "fast"

    def decide(self, state) -> Tuple[int, str, Dict[str, Any]]:
        frame = int(self.frame)
        action, reasoning, metadata = self.inner.decide(state)
        self.fast_actions[frame] = int(action)
        self.legal_actions[frame] = tuple(int(action) for action in state.get_available_actions())
        if frame in self.force_actions:
            action = int(self.force_actions[frame])
        elif self.force_frame is not None and frame == int(self.force_frame):
            if self.force_action is None:
                raise ValueError("force_frame requires force_action")
            action = int(self.force_action)
        self.frame += 1
        self.last_system_used = str(metadata.get("system_used", "fast"))
        return (
            action,
            str(reasoning or "[Fast matched rollout]"),
            {
                **dict(metadata or {}),
                "system_used": "fast",
                "route_label": "matched_fast_rollout",
            },
        )

    def record_executed_action(self, action: int) -> None:
        self.inner.record_executed_action(int(action))

    def snapshot_policy_state(self) -> Dict[str, Any]:
        return self.inner.snapshot_policy_state()

    def restore_policy_state(self, snapshot: Dict[str, Any]) -> None:
        self.inner.restore_policy_state(snapshot)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trace_paths(root: Path, seed: int) -> Tuple[Path, Path]:
    layouts = {
        "current": root / "always_fast" / "highway" / f"seed_{seed}" / f"ep_{seed}",
        "legacy": (
            root
            / "always_fast"
            / f"always_fast_latency_1p7s_seed_{seed}"
            / f"ep_{seed}"
        ),
    }
    complete: List[Tuple[Path, Path]] = []
    incomplete: List[str] = []
    for name, episode_dir in layouts.items():
        reasoning = episode_dir / f"highway_{seed}_reasoning_records.json"
        physical = episode_dir / f"highway_{seed}_physical_frames.json"
        has_reasoning = reasoning.is_file()
        has_physical = physical.is_file()
        if has_reasoning and has_physical:
            complete.append((reasoning, physical))
        elif has_reasoning or has_physical:
            incomplete.append(
                f"{name} layout has reasoning={has_reasoning}, physical={has_physical}"
            )
    if len(complete) == 1:
        return complete[0]
    if len(complete) > 1:
        raise RuntimeError(f"seed {seed}: ambiguous complete trace pairs across layouts")
    if incomplete:
        raise RuntimeError(f"seed {seed}: incomplete trace pair; " + "; ".join(incomplete))
    raise RuntimeError(f"seed {seed}: no reasoning/physical trace pair found")


def _load_trace(root: Path, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    reasoning_path, physical_path = _trace_paths(root, seed)
    reasoning_payload = json.loads(reasoning_path.read_text(encoding="utf-8-sig"))
    physical_payload = json.loads(physical_path.read_text(encoding="utf-8-sig"))
    records = list(reasoning_payload.get("analysis_records", []) or [])
    frames = list(physical_payload.get("frames", []) or [])
    expected = list(range(len(records)))
    if [int(row.get("frame_id", -1)) for row in records] != expected:
        raise RuntimeError(f"seed {seed}: non-contiguous reasoning frames")
    if [int(row.get("frame_id", -1)) for row in frames] != list(range(len(frames))):
        raise RuntimeError(f"seed {seed}: non-contiguous physical frames")
    if len(records) != len(frames):
        raise RuntimeError(f"seed {seed}: reasoning/physical length mismatch")
    return records, frames


def _build_fast_config(
    protocol_path: Path,
    seed: int,
    scratch: Path,
    *,
    verified_floor_overlay: Optional[VerifiedFloorOverlay] = None,
) -> Dict[str, Any]:
    protocol = load_formal_protocol(protocol_path)
    base_cfg = load_formal_base_config(protocol)
    if verified_floor_overlay is not None:
        base_cfg = apply_floor_overlay(base_cfg, verified_floor_overlay)
    group_cfg = dict((protocol.get("groups", {}) or {})["always_fast"] or {})
    cfg = build_group_config(
        base_cfg,
        "always_fast",
        group_cfg,
        "highway-v0",
        1,
        scratch,
        protocol,
    )
    cfg.update(
        {
            "env_type": "highway-v0",
            "scenario_type": "highway",
            "simulation_duration": 30,
            "policy_frequency": 10,
            "simulation_frequency": 10,
            "fixed_seed_override": int(seed),
            "highway_rebuild_traffic_if_needed": True,
            "render_mode": "",
            "enable_physical_metrics": False,
            "enable_reasoning_recording": False,
        }
    )
    cfg["closed_loop_latency_replay"] = {
        "enable": False,
        "extra_latency_s": 0.0,
        "delay_steps": 0,
        "target_systems": ["slow"],
    }
    return cfg


def _position_error(env, frame: Dict[str, Any]) -> float:
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    if vehicle is None:
        return float("inf")
    return float(
        math.hypot(
            float(vehicle.position[0]) - float(frame["position_x"]),
            float(vehicle.position[1]) - float(frame["position_y"]),
        )
    )


def _capture_release_snapshots(
    cfg: Dict[str, Any],
    seed: int,
    target_frames: Iterable[int],
    physical_frames: Sequence[Dict[str, Any]],
    scratch: Path,
) -> Tuple[Dict[int, ReleaseSnapshot], float]:
    targets = sorted({int(frame) for frame in target_frames})
    if not targets:
        return {}, 0.0
    env, obs, _, _, _, close_after = create_episode_env(seed, cfg, str(scratch), [seed])
    scenario = create_scenario(env, "highway-v0", seed, None)
    scenario.scenario_type = "highway"
    agent = FastBranchAgent(cfg)
    safety = UnifiedSafetySystem(cfg)
    history = collections.deque(maxlen=int(cfg.get("history_window", 6) or 6))
    state = create_episode_runtime_state()
    snapshots: Dict[int, ReleaseSnapshot] = {}
    max_position_error = 0.0
    try:
        for frame in range(max(targets) + 1):
            if frame >= len(physical_frames):
                raise RuntimeError(f"seed {seed}: target frame {frame} exceeds trace")
            max_position_error = max(max_position_error, _position_error(env, physical_frames[frame]))
            if frame in targets:
                snapshots[frame] = capture_release_snapshot(
                    agent.inner,
                    frame=frame,
                    env=env,
                    obs=obs,
                    history=history,
                    previous_action=int(state["action"]),
                )
            obs, done = execute_episode_step(
                frame=frame,
                env=env,
                sce=scenario,
                agent=agent,
                obs=obs,
                cfg=cfg,
                safety_system=safety,
                phys_rec=None,
                reas_rec=None,
                history_buffer=history,
                episode_state=state,
            )
            expected_action = int(physical_frames[frame]["action_id"])
            if int(state["action"]) != expected_action:
                raise RuntimeError(
                    f"seed {seed}: fast-prefix mismatch at frame {frame}: "
                    f"replay={state['action']} trace={expected_action}"
                )
            if done and frame < max(targets):
                raise RuntimeError(f"seed {seed}: Fast replay terminated before target frame {max(targets)}")
        if set(snapshots) != set(targets):
            raise RuntimeError(f"seed {seed}: missing release snapshots")
        if max_position_error > 1e-6:
            raise RuntimeError(f"seed {seed}: Fast replay position error {max_position_error:.3g} m")
        return snapshots, max_position_error
    finally:
        if close_after:
            env.close()


def _target_speed(env) -> float:
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    value = getattr(vehicle, "target_speed", float("nan")) if vehicle is not None else float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _acceleration(env) -> float:
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    action = getattr(vehicle, "action", {}) if vehicle is not None else {}
    try:
        return float((action or {}).get("acceleration", float("nan")))
    except (TypeError, ValueError, AttributeError):
        return float("nan")


def _finite_or_none(value: Any) -> Optional[float]:
    """Return a JSON-safe finite scalar for a branch trajectory."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _lane_id(env) -> Optional[int]:
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    lane_index = getattr(vehicle, "lane_index", None) if vehicle is not None else None
    if not isinstance(lane_index, (tuple, list)) or not lane_index:
        return None
    try:
        return int(lane_index[-1])
    except (TypeError, ValueError):
        return None


def _trajectory_step(
    env: Any,
    *,
    frame: int,
    effective_action: int,
    ttc: float,
    collision: bool,
    terminal_cause: str,
) -> Dict[str, Any]:
    """Serialize the post-action state needed for an event-level comparison."""
    vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
    position = getattr(vehicle, "position", (float("nan"), float("nan")))
    try:
        position_x, position_y = float(position[0]), float(position[1])
    except (IndexError, TypeError, ValueError):
        position_x = position_y = float("nan")
    return {
        "frame": int(frame),
        "position_x_m": _finite_or_none(position_x),
        "position_y_m": _finite_or_none(position_y),
        "speed_mps": _finite_or_none(getattr(vehicle, "speed", float("nan"))),
        "lane_id": _lane_id(env),
        "effective_action": int(effective_action),
        "acceleration_mps2": _finite_or_none(_acceleration(env)),
        "ttc_s": _finite_or_none(ttc),
        "collision": int(bool(collision)),
        "terminal_cause": str(terminal_cause),
    }


def _run_branch(
    snapshot: ReleaseSnapshot,
    cfg: Dict[str, Any],
    seed: int,
    raw_action: Optional[int],
    horizon: int,
    gamma: float,
) -> Dict[str, Any]:
    database = Path(tempfile.gettempdir()) / (
        f"rgd_release_rollout_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    try:
        return _run_branch_with_database(
            snapshot,
            cfg,
            seed,
            raw_action,
            horizon,
            gamma,
            database,
        )
    finally:
        database.unlink(missing_ok=True)


def _run_branch_with_database(
    snapshot: ReleaseSnapshot,
    cfg: Dict[str, Any],
    seed: int,
    raw_action: Optional[int],
    horizon: int,
    gamma: float,
    database: Path,
) -> Dict[str, Any]:
    env = copy.deepcopy(snapshot.env)
    scenario = create_scenario(env, "highway-v0", seed, str(database))
    scenario.scenario_type = "highway"
    agent = FastBranchAgent(
        cfg,
        policy_state=validate_release_snapshot_policy_state(
            snapshot,
            context=f"release rollout frame {snapshot.frame}",
        ),
        force_frame=snapshot.frame if raw_action is not None else None,
        force_action=raw_action,
    )
    agent.frame = int(snapshot.frame)
    safety = UnifiedSafetySystem(cfg)
    history = copy.deepcopy(snapshot.history)
    state = create_episode_runtime_state()
    state["action"] = int(snapshot.previous_action)
    obs = copy.deepcopy(snapshot.obs)
    discounted = 0.0
    weight = 0.0
    collision = False
    min_ttc = float("inf")
    release_fast_action = None
    release_legal_actions: Tuple[int, ...] = ()
    release_effective_action = None
    release_target_speed = float("nan")
    release_acceleration = float("nan")
    start_x = float(getattr(getattr(env, "unwrapped", env).vehicle, "position")[0])
    completed_steps = 0
    terminal_cause = "horizon"
    trajectory: List[Dict[str, Any]] = []
    acceleration_samples: List[float] = []
    jerk_samples: List[float] = []
    previous_acceleration: Optional[float] = None
    policy_frequency = float(cfg.get("policy_frequency", 10.0) or 10.0)
    if not math.isfinite(policy_frequency) or policy_frequency <= 0.0:
        raise ValueError("branch rollout requires a positive finite policy frequency")
    for offset in range(int(horizon)):
        frame = int(snapshot.frame + offset)
        reward_before = float(state.get("episode_reward", 0.0) or 0.0)
        obs, done = execute_episode_step(
            frame=frame,
            env=env,
            sce=scenario,
            agent=agent,
            obs=obs,
            cfg=cfg,
            safety_system=safety,
            phys_rec=None,
            reas_rec=None,
            history_buffer=history,
            episode_state=state,
        )
        step_reward = float(state.get("episode_reward", 0.0) or 0.0) - reward_before
        discounted += (float(gamma) ** offset) * step_reward
        weight += float(gamma) ** offset
        event = state["event_log"][-1]
        ttc = _finite_or_none(event.get("ttc", float("inf")))
        min_ttc = min(min_ttc, ttc if ttc is not None else float("inf"))
        collision = bool(collision or event.get("crashed", False))
        completed_steps += 1
        step_terminal_cause = str(event.get("terminal_cause", "running") or "running")
        acceleration = _finite_or_none(_acceleration(env))
        if acceleration is not None:
            acceleration_samples.append(abs(acceleration))
            if previous_acceleration is not None:
                jerk_samples.append(abs(acceleration - previous_acceleration) * policy_frequency)
            previous_acceleration = acceleration
        trajectory.append(
            _trajectory_step(
                env,
                frame=frame,
                effective_action=int(state["action"]),
                ttc=float("inf") if ttc is None else ttc,
                collision=bool(event.get("crashed", False)),
                terminal_cause=step_terminal_cause,
            )
        )
        if offset == 0:
            release_fast_action = int(agent.fast_actions[frame])
            release_legal_actions = tuple(agent.legal_actions[frame])
            release_effective_action = int(state["action"])
            release_target_speed = _target_speed(env)
            release_acceleration = _acceleration(env)
        if done:
            terminal_cause = step_terminal_cause
            break
    end_x = float(getattr(getattr(env, "unwrapped", env).vehicle, "position")[0])
    normalized_return = discounted / weight if weight > 0 else float("nan")
    utility = float(normalized_return - (1.0 if collision else 0.0))
    return {
        "seed": int(seed),
        "release_frame": int(snapshot.frame),
        "raw_action": "fast" if raw_action is None else int(raw_action),
        "fast_action": int(release_fast_action),
        "legal_actions": ";".join(str(action) for action in release_legal_actions),
        "effective_action": int(release_effective_action),
        "target_speed_after": float(release_target_speed),
        "acceleration_after": float(release_acceleration),
        "horizon_steps": int(horizon),
        "steps_completed": int(completed_steps),
        "gamma": float(gamma),
        "normalized_return": float(normalized_return),
        "collision": int(collision),
        "min_ttc": float(min_ttc),
        "progress_m": float(end_x - start_x),
        "mean_abs_acceleration_mps2": (
            float(sum(acceleration_samples) / len(acceleration_samples))
            if acceleration_samples
            else float("nan")
        ),
        "mean_abs_jerk_mps3": (
            float(sum(jerk_samples) / len(jerk_samples))
            if jerk_samples
            else float("nan")
        ),
        "terminal_cause": str(terminal_cause),
        "completed_horizon": bool(
            completed_steps == int(horizon) and terminal_cause == "horizon"
        ),
        "branch_trajectory_json": json.dumps(
            trajectory,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        "utility": utility,
    }


def _effective_identity(row: Dict[str, Any]) -> Tuple[int, float]:
    target = float(row["target_speed_after"])
    target_key = round(target, 6) if math.isfinite(target) else float("nan")
    return int(row["effective_action"]), target_key


def _selected_queries(records: Sequence[Dict[str, Any]], delay: float) -> Dict[str, List[int]]:
    return {
        "RGD": scheduled_frames(
            records,
            lambda record: (
                gate_values(record, delay)[2] >= 1
                and gate_values(record, delay)[0] >= RGD_FLOOR
                and gate_values(record, delay)[1] >= RGD_THRESHOLD
            ),
        ),
        "TTC-risk": scheduled_frames(records, lambda record: ttc_score(record) >= TTC_CUTOFF),
    }


def _delay_step_map(delays: Sequence[float], *, policy_frequency_hz: float = 10.0) -> Dict[float, int]:
    """Resolve the fixed discrete release offset for each declared delay."""

    frequency = float(policy_frequency_hz)
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("policy_frequency_hz must be positive and finite")
    steps: Dict[float, int] = {}
    for value in delays:
        delay = float(value)
        if not math.isfinite(delay) or delay < 0.0:
            raise ValueError("delay values must be finite and nonnegative")
        if delay in steps:
            raise ValueError("delay values must be unique")
        steps[delay] = int(math.ceil(frequency * delay))
    if not steps:
        raise ValueError("at least one fixed delay is required")
    return steps


def _selection_contract(
    selected: Dict[str, Sequence[int]],
    *,
    seed: int,
    record_count: int,
    delays: Sequence[float],
    horizon: int,
) -> Tuple[List[Tuple[str, int, float, int]], List[Dict[str, Any]]]:
    """Freeze main and fixed-query release eligibility before branch rollout.

    Main allocator arms are evaluated at the nominal 1.7-s release delay.
    The fixed RGD cohort is intentionally stricter: one query is retained only
    when it has a complete continuation horizon at every prespecified delay.
    This prevents delay-specific trace truncation from changing its denominator.
    """

    if int(record_count) < 0:
        raise ValueError("record_count must be nonnegative")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    delay_steps = _delay_step_map(delays)
    nominal_delay = 1.7
    if nominal_delay not in delay_steps:
        raise ValueError("the allocator comparison requires the nominal 1.7-s delay")

    def normalized_frames(allocator: str) -> List[int]:
        source = selected.get(allocator, ())
        frames = [int(frame) for frame in source]
        if any(frame < 0 for frame in frames):
            raise ValueError(f"{allocator}: query frames must be nonnegative")
        if frames != sorted(frames) or len(set(frames)) != len(frames):
            raise ValueError(f"{allocator}: query frames must be unique and ordered")
        return frames

    release_specs: List[Tuple[str, int, float, int]] = []
    accounting: List[Dict[str, Any]] = []
    nominal_steps = delay_steps[nominal_delay]
    for allocator in ("RGD", "TTC-delay", "TTC-risk"):
        frames = normalized_frames(allocator)
        evaluated = [
            frame
            for frame in frames
            if int(frame + nominal_steps + int(horizon)) <= int(record_count)
        ]
        accounting.append(
            {
                "seed": int(seed),
                "allocator": allocator,
                "delay_s": nominal_delay,
                "delay_steps": int(nominal_steps),
                "scheduled_count": len(frames),
                "excluded_count": len(frames) - len(evaluated),
                "evaluated_count": len(evaluated),
                "eligibility_rule": "release_frame_plus_full_rollout_horizon_within_trace",
            }
        )
        release_specs.extend(
            (allocator, int(frame), nominal_delay, int(frame + nominal_steps))
            for frame in evaluated
        )

    rgd_frames = normalized_frames("RGD")
    max_steps = max(delay_steps.values())
    fixed_frames = [
        frame
        for frame in rgd_frames
        if int(frame + max_steps + int(horizon)) <= int(record_count)
    ]
    for delay, steps in delay_steps.items():
        accounting.append(
            {
                "seed": int(seed),
                "allocator": "RGD-fixed",
                "delay_s": float(delay),
                "delay_steps": int(steps),
                "scheduled_count": len(rgd_frames),
                "excluded_count": len(rgd_frames) - len(fixed_frames),
                "evaluated_count": len(fixed_frames),
                "eligibility_rule": "common_fixed_query_cohort_with_full_rollout_horizon_at_all_delays",
            }
        )
        release_specs.extend(
            ("RGD-fixed", int(frame), float(delay), int(frame + steps))
            for frame in fixed_frames
        )
    return release_specs, accounting


def _process_seed(
    seed: int,
    trace_root: str,
    protocol_path: str,
    delays: Sequence[float],
    horizon: int,
    gamma: float,
    epsilon: float,
    scratch_root: str,
) -> Dict[str, Any]:
    root = Path(trace_root)
    records, physical = _load_trace(root, seed)
    selected = _selected_queries(records, 1.7)
    release_specs, selection_accounting = _selection_contract(
        selected,
        seed=int(seed),
        record_count=len(records),
        delays=delays,
        horizon=int(horizon),
    )
    fixed_queries = {
        int(query_frame)
        for allocator, query_frame, _, _ in release_specs
        if allocator == "RGD-fixed"
    }

    unique_release_frames = sorted({spec[3] for spec in release_specs})
    scratch = Path(scratch_root) / f"seed_{seed}"
    cfg = _build_fast_config(Path(protocol_path), seed, scratch)
    snapshots, max_position_error = _capture_release_snapshots(
        cfg, seed, unique_release_frames, physical, scratch
    )

    branch_rows: List[Dict[str, Any]] = []
    release_results: Dict[int, Dict[str, Any]] = {}
    for release_frame in unique_release_frames:
        snapshot = snapshots[release_frame]
        baseline = _run_branch(snapshot, cfg, seed, None, horizon, gamma)
        legal_actions = tuple(int(value) for value in str(baseline["legal_actions"]).split(";") if value != "")
        candidates = [
            _run_branch(snapshot, cfg, seed, action, horizon, gamma)
            for action in RAW_ACTIONS
            if action in legal_actions
        ]
        baseline_identity = _effective_identity(baseline)
        distinct_by_effect: Dict[Tuple[int, float], Dict[str, Any]] = {}
        for candidate in candidates:
            identity = _effective_identity(candidate)
            current = distinct_by_effect.get(identity)
            if current is None or float(candidate["utility"]) > float(current["utility"]):
                distinct_by_effect[identity] = candidate
        alternatives = [
            row for identity, row in distinct_by_effect.items() if identity != baseline_identity
        ]
        best_advantage = max(
            [float(row["utility"]) - float(baseline["utility"]) for row in alternatives],
            default=float("-inf"),
        )
        best_row = max(alternatives, key=lambda row: float(row["utility"]), default=None)
        corrective = bool(alternatives and best_advantage >= float(epsilon))
        release_results[release_frame] = {
            "baseline": baseline,
            "candidates": candidates,
            "distinct_alternatives": len(alternatives),
            "best_advantage": best_advantage,
            "best_row": best_row,
            "corrective": corrective,
        }
        branch_rows.append({**baseline, "branch_role": "matched_fast"})
        branch_rows.extend({**row, "branch_role": "candidate"} for row in candidates)

    event_rows: List[Dict[str, Any]] = []
    for allocator, query_frame, delay, release_frame in release_specs:
        outcome = release_results[release_frame]
        best_row = outcome["best_row"]
        record = records[query_frame]
        components = _query_gate_components(record, delay)
        query_opportunity = float(components["opportunity"])
        query_priority = float(components["priority"])
        query_alternatives = int(components["alternative_count"])
        delay_steps = int(release_frame - query_frame)
        event_rows.append(
            {
                "seed": int(seed),
                "allocator": allocator,
                "query_frame": int(query_frame),
                "release_frame": int(release_frame),
                "delay_s": float(delay),
                "delay_steps": delay_steps,
                "candidate_state_id": f"{seed}:{query_frame}:{delay_steps}",
                "release_state_id": f"{seed}:{release_frame}",
                "need_score": float(components["need_score"]),
                "latency_survival": float(components["latency_survival"]),
                "admissible_alternative_fraction": float(
                    components["admissible_alternative_fraction"]
                ),
                "recovery_headroom": float(components["recovery_headroom"]),
                "alternative_count": int(components["alternative_count"]),
                "absolute_alternative_count": int(
                    components["absolute_alternative_count"]
                ),
                "query_opportunity": float(query_opportunity),
                "query_priority": float(query_priority),
                "query_legal_alternatives": int(query_alternatives),
                "release_distinct_alternatives": int(outcome["distinct_alternatives"]),
                "corrective_set_nonempty": int(outcome["corrective"]),
                "best_advantage": (
                    float(outcome["best_advantage"])
                    if math.isfinite(float(outcome["best_advantage"]))
                    else ""
                ),
                "baseline_utility": float(outcome["baseline"]["utility"]),
                "baseline_collision": int(outcome["baseline"]["collision"]),
                "best_effective_action": "" if best_row is None else int(best_row["effective_action"]),
                "best_target_speed_after": "" if best_row is None else float(best_row["target_speed_after"]),
                "best_collision": "" if best_row is None else int(best_row["collision"]),
                "horizon_steps": int(horizon),
                "gamma": float(gamma),
                "epsilon": float(epsilon),
                "outcome_definition": "normalized simulator return minus collision indicator",
            }
        )
    return {
        "seed": int(seed),
        "events": event_rows,
        "branches": branch_rows,
        "selected": {key: len(value) for key, value in selected.items()},
        "fixed_queries": len(fixed_queries),
        "selection_accounting": selection_accounting,
        "max_position_error_m": float(max_position_error),
    }


def _cluster_bootstrap_rate(
    rows: Sequence[Dict[str, Any]],
    allocator: str,
    *,
    delay: Optional[float] = None,
    draws: int = BOOTSTRAP_DRAWS,
) -> Tuple[float, float, float, int, int]:
    selected = [
        row
        for row in rows
        if row["allocator"] == allocator
        and (delay is None or math.isclose(float(row["delay_s"]), float(delay)))
    ]
    seeds = sorted({int(row["seed"]) for row in selected})
    by_seed = {
        seed: [row for row in selected if int(row["seed"]) == seed]
        for seed in seeds
    }
    numerator = sum(int(row["corrective_set_nonempty"]) for row in selected)
    denominator = len(selected)
    point = numerator / denominator if denominator else float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: List[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(np.asarray(seeds), size=len(seeds), replace=True).tolist()
        sample_rows = [row for seed in sampled for row in by_seed[int(seed)]]
        if sample_rows:
            samples.append(
                sum(int(row["corrective_set_nonempty"]) for row in sample_rows)
                / len(sample_rows)
            )
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(point), float(low), float(high), int(numerator), int(denominator)


def _cluster_bootstrap_difference(
    rows: Sequence[Dict[str, Any]],
    *,
    delay: float = 1.7,
    draws: int = BOOTSTRAP_DRAWS,
) -> Tuple[float, float, float]:
    selected = [
        row
        for row in rows
        if row["allocator"] in {"RGD", "TTC-risk"}
        and math.isclose(float(row["delay_s"]), float(delay))
    ]
    seeds = sorted({int(row["seed"]) for row in selected})
    by_arm_seed = {
        allocator: {
            seed: [
                row for row in selected
                if row["allocator"] == allocator and int(row["seed"]) == seed
            ]
            for seed in seeds
        }
        for allocator in ("RGD", "TTC-risk")
    }

    def pooled(sampled: Sequence[int], allocator: str) -> float:
        arm_rows = [row for seed in sampled for row in by_arm_seed[allocator][int(seed)]]
        return (
            sum(int(row["corrective_set_nonempty"]) for row in arm_rows) / len(arm_rows)
            if arm_rows else float("nan")
        )

    point = pooled(seeds, "RGD") - pooled(seeds, "TTC-risk")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: List[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(np.asarray(seeds), size=len(seeds), replace=True).tolist()
        rgd = pooled(sampled, "RGD")
        ttc = pooled(sampled, "TTC-risk")
        if math.isfinite(rgd) and math.isfinite(ttc):
            samples.append(rgd - ttc)
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(point), float(low), float(high)


def _cluster_bootstrap_paired_delay_difference(
    rows: Sequence[Dict[str, Any]],
    cohort: Sequence[int],
    *,
    lower_delay: float = 0.7,
    upper_delay: float = 2.7,
    draws: int = BOOTSTRAP_DRAWS,
) -> Tuple[float, float, float, int]:
    """Bootstrap the fixed-query endpoint difference with seed clustering.

    A query must have both endpoint rollouts or neither.  Seeds that schedule
    no fixed query are still part of ``cohort`` and remain available to every
    bootstrap draw, which preserves the preregistered cluster population.
    """

    lower = float(lower_delay)
    upper = float(upper_delay)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower == upper:
        raise ValueError("paired delays must be distinct finite values")
    if int(draws) <= 0:
        raise ValueError("bootstrap draws must be positive")
    seeds = tuple(int(seed) for seed in cohort)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("paired bootstrap cohort must contain unique seeds")
    seed_set = set(seeds)

    def endpoint(value: Any) -> Optional[float]:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            return None
        if math.isclose(delay, lower, rel_tol=0.0, abs_tol=1e-12):
            return lower
        if math.isclose(delay, upper, rel_tol=0.0, abs_tol=1e-12):
            return upper
        return None

    paired: Dict[Tuple[int, int], Dict[float, Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("allocator", "")) != "RGD-fixed":
            continue
        delay = endpoint(row.get("delay_s"))
        if delay is None:
            continue
        seed = int(row["seed"])
        query_frame = int(row["query_frame"])
        if seed not in seed_set:
            raise ValueError("paired delay row lies outside the declared cohort")
        block = paired.setdefault((seed, query_frame), {})
        if delay in block:
            raise ValueError("fixed query does not form a complete paired block")
        block[delay] = row

    required = {lower, upper}
    for key, block in paired.items():
        if set(block) != required:
            raise ValueError(f"fixed query {key} does not form a complete paired block")

    by_seed: Dict[int, Dict[float, List[Dict[str, Any]]]] = {
        seed: {lower: [], upper: []} for seed in seeds
    }
    for (seed, _), block in paired.items():
        by_seed[seed][lower].append(block[lower])
        by_seed[seed][upper].append(block[upper])

    def rate(sampled: Sequence[int], delay: float) -> float:
        selected = [
            row
            for seed in sampled
            for row in by_seed[int(seed)][delay]
        ]
        if not selected:
            return float("nan")
        return float(
            sum(int(row["corrective_set_nonempty"]) for row in selected)
            / len(selected)
        )

    point_lower = rate(seeds, lower)
    point_upper = rate(seeds, upper)
    point = (
        float(point_lower - point_upper)
        if math.isfinite(point_lower) and math.isfinite(point_upper)
        else float("nan")
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: List[float] = []
    for _ in range(int(draws)):
        sampled = rng.choice(seeds, size=len(seeds), replace=True).tolist()
        lower_rate = rate(sampled, lower)
        upper_rate = rate(sampled, upper)
        if math.isfinite(lower_rate) and math.isfinite(upper_rate):
            samples.append(float(lower_rate - upper_rate))
    if not samples:
        return point, float("nan"), float("nan"), 0
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(point), float(low), float(high), len(samples)


def _summarize(rows: Sequence[Dict[str, Any]], delays: Sequence[float]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for allocator, delay_values in (
        ("RGD", (1.7,)),
        ("TTC-risk", (1.7,)),
        ("RGD-fixed", tuple(float(value) for value in delays)),
    ):
        for delay in delay_values:
            point, low, high, numerator, denominator = _cluster_bootstrap_rate(
                rows, allocator, delay=float(delay)
            )
            summary.append(
                {
                    "allocator": allocator,
                    "delay_s": float(delay),
                    "corrective_count": numerator,
                    "release_count": denominator,
                    "corrective_fraction": point,
                    "ci_low": low,
                    "ci_high": high,
                    "cluster_unit": "seed",
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                }
            )
    difference, diff_low, diff_high = _cluster_bootstrap_difference(rows, delay=1.7)
    for row in summary:
        row["rgd_minus_ttc_difference"] = difference if row["allocator"] == "RGD" else ""
        row["difference_ci_low"] = diff_low if row["allocator"] == "RGD" else ""
        row["difference_ci_high"] = diff_high if row["allocator"] == "RGD" else ""
    return summary


def _relabel(rows: Sequence[Dict[str, Any]], epsilon: float) -> List[Dict[str, Any]]:
    relabeled: List[Dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        raw_advantage = row.get("best_advantage", "")
        row["corrective_set_nonempty"] = int(
            raw_advantage not in (None, "") and float(raw_advantage) >= float(epsilon)
        )
        relabeled.append(row)
    return relabeled


def _sensitivity_summary(
    rows: Sequence[Dict[str, Any]],
    delays: Sequence[float],
    epsilons: Sequence[float] = (0.01, 0.02, 0.05),
) -> List[Dict[str, Any]]:
    sensitivity: List[Dict[str, Any]] = []
    for epsilon in epsilons:
        relabeled = _relabel(rows, float(epsilon))
        arm_rows = _summarize(relabeled, delays)
        for row in arm_rows:
            sensitivity.append({"epsilon": float(epsilon), **row})
    return sensitivity


def _fixed_query_transitions(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixed = [row for row in rows if row["allocator"] == "RGD-fixed"]
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in fixed:
        grouped.setdefault((int(row["seed"]), int(row["query_frame"])), []).append(row)
    counts: Dict[str, int] = collections.Counter()
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["delay_s"]))
        if len(ordered) != 3:
            raise RuntimeError(f"fixed query {key} does not have all three delays")
        pattern = "".join(str(int(row["corrective_set_nonempty"])) for row in ordered)
        counts[pattern] += 1
    rows_out: List[Dict[str, Any]] = []
    total = sum(counts.values())
    for pattern in sorted(counts):
        rows_out.append(
            {
                "pattern_0p7_1p7_2p7": pattern,
                "queries": int(counts[pattern]),
                "fraction": float(counts[pattern] / total if total else float("nan")),
                "interpretation": (
                    "monotone_or_stable" if pattern in {"000", "100", "110", "111"}
                    else "nonmonotone_traffic_gap"
                ),
            }
        )
    return rows_out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path("formal_protocol.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=160)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--delays", type=float, nargs="*", default=list(DEFAULT_DELAYS))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="reuse release_rollout_events.csv and regenerate summaries only",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TEMP", ".")) / "rgd_release_rollout_scratch",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if 1.7 not in {float(value) for value in args.delays}:
        raise ValueError("the frozen allocator comparison requires delay 1.7 s")
    if args.horizon <= 0 or not (0.0 < args.gamma <= 1.0) or args.epsilon < 0.0:
        raise ValueError("invalid horizon, gamma, or epsilon")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(args.seed_start) + offset for offset in range(int(args.seeds))]
    results: List[Dict[str, Any]] = []
    if args.summarize_existing:
        events_path = args.output_dir / "release_rollout_events.csv"
        branches_path = args.output_dir / "release_rollout_branches.csv"
        with events_path.open("r", encoding="utf-8-sig", newline="") as handle:
            event_rows = list(csv.DictReader(handle))
        with branches_path.open("r", encoding="utf-8-sig", newline="") as handle:
            branch_rows = list(csv.DictReader(handle))
        max_position_error = 0.0
        manifest_path = args.output_dir / "release_rollout_manifest.json"
        if manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            max_position_error = float(existing_manifest.get("max_fast_replay_position_error_m", 0.0) or 0.0)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {
                pool.submit(
                    _process_seed,
                    seed,
                    str(args.trace_root.resolve()),
                    str(args.protocol.resolve()),
                    tuple(float(value) for value in args.delays),
                    int(args.horizon),
                    float(args.gamma),
                    float(args.epsilon),
                    str(args.scratch_root.resolve()),
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                result = future.result()
                results.append(result)
                print(
                    f"seed={seed} events={len(result['events'])} "
                    f"max_replay_error={result['max_position_error_m']:.3g}m"
                )
        results.sort(key=lambda row: int(row["seed"]))
        event_rows = [row for result in results for row in result["events"]]
        branch_rows = [row for result in results for row in result["branches"]]
        max_position_error = max(float(result["max_position_error_m"]) for result in results)
    event_rows.sort(
        key=lambda row: (
            int(row["seed"]), str(row["allocator"]), int(row["query_frame"]), float(row["delay_s"])
        )
    )
    branch_rows.sort(
        key=lambda row: (int(row["seed"]), int(row["release_frame"]), str(row["raw_action"]))
    )
    summary = _summarize(event_rows, args.delays)
    sensitivity = _sensitivity_summary(event_rows, args.delays)
    transitions = _fixed_query_transitions(event_rows)
    if not args.summarize_existing:
        _write_csv(args.output_dir / "release_rollout_events.csv", event_rows)
        _write_csv(args.output_dir / "release_rollout_branches.csv", branch_rows)
    _write_csv(args.output_dir / "release_rollout_summary.csv", summary)
    _write_csv(args.output_dir / "release_rollout_sensitivity.csv", sensitivity)
    _write_csv(args.output_dir / "fixed_query_transitions.csv", transitions)
    manifest = {
        "analysis": "fresh-holdout outcome-grounded matched-action release rollouts",
        "trace_root": str(args.trace_root.resolve()),
        "protocol": str(args.protocol.resolve()),
        "seeds": seeds,
        "seed_is_experimental_unit": True,
        "delays_s": [float(value) for value in args.delays],
        "horizon_steps": int(args.horizon),
        "policy_frequency_hz": 10,
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "utility": "discounted normalized simulator return minus collision indicator",
        "action_identity": "post-safety discrete command plus post-bridge target speed",
        "continuation": "same complete deterministic fast controller",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "max_fast_replay_position_error_m": float(max_position_error),
        "summary": summary,
        "epsilon_sensitivity": sensitivity,
        "fixed_query_transitions": transitions,
    }
    (args.output_dir / "release_rollout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
