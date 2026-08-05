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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.driver_agent.reasoning.fast_thinker import FastThinker
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
    gate_values,
    scheduled_frames,
    ttc_score,
)
from tools.run_main_table_runtime import (
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
)


RAW_ACTIONS = tuple(range(5))
DEFAULT_DELAYS = (0.7, 1.7, 2.7)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260717


@dataclass
class ReleaseSnapshot:
    frame: int
    env: Any
    obs: Any
    fast_state: Dict[str, Any]
    history: collections.deque
    previous_action: int


class FastBranchAgent:
    """Fast controller with an optional one-frame counterfactual action."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        fast_state: Optional[Dict[str, Any]] = None,
        force_frame: Optional[int] = None,
        force_action: Optional[int] = None,
    ) -> None:
        fast_cfg = dict(cfg.get("fast_thinking", {}) or {})
        if "lane_change_cooldown" in cfg:
            fast_cfg["lane_change_cooldown"] = int(cfg.get("lane_change_cooldown", 0) or 0)
        self.fast = FastThinker(lane_change_config=fast_cfg)
        if fast_state is not None:
            self.fast.restore_runtime_state(fast_state)
        self.force_frame = force_frame
        self.force_action = force_action
        self.frame = 0
        self.fast_actions: Dict[int, int] = {}
        self.legal_actions: Dict[int, Tuple[int, ...]] = {}
        self.last_system_used = "fast"

    def decide(self, state) -> Tuple[int, str, Dict[str, Any]]:
        frame = int(self.frame)
        decision = self.fast.think(state)
        self.fast_actions[frame] = int(decision.action)
        self.legal_actions[frame] = tuple(int(action) for action in state.get_available_actions())
        action = int(decision.action)
        if self.force_frame is not None and frame == int(self.force_frame):
            if self.force_action is None:
                raise ValueError("force_frame requires force_action")
            action = int(self.force_action)
        self.frame += 1
        return (
            action,
            "[Fast matched rollout]",
            {
                "system_used": "fast",
                "route_label": "matched_fast_rollout",
                "route_score": 0.0,
                "confidence": float(decision.confidence),
                "latency_ms": float(decision.latency_ms),
            },
        )


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trace_paths(root: Path, seed: int) -> Tuple[Path, Path]:
    setting = root / "always_fast" / f"always_fast_latency_1p7s_seed_{seed}"
    reasoning = list(setting.glob(f"ep_{seed}/highway_{seed}_reasoning_records.json"))
    physical = list(setting.glob(f"ep_{seed}/highway_{seed}_physical_frames.json"))
    if len(reasoning) != 1 or len(physical) != 1:
        raise RuntimeError(
            f"seed {seed}: expected one reasoning and physical trace, found "
            f"{len(reasoning)} and {len(physical)}"
        )
    return reasoning[0], physical[0]


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


def _build_fast_config(protocol_path: Path, seed: int, scratch: Path) -> Dict[str, Any]:
    protocol = load_formal_protocol(protocol_path)
    base_cfg = load_formal_base_config(protocol)
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
                snapshots[frame] = ReleaseSnapshot(
                    frame=frame,
                    env=copy.deepcopy(env),
                    obs=copy.deepcopy(obs),
                    fast_state=agent.fast.snapshot_runtime_state(),
                    history=copy.deepcopy(history),
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


def _run_branch(
    snapshot: ReleaseSnapshot,
    cfg: Dict[str, Any],
    seed: int,
    raw_action: Optional[int],
    horizon: int,
    gamma: float,
) -> Dict[str, Any]:
    env = copy.deepcopy(snapshot.env)
    scenario = create_scenario(env, "highway-v0", seed, None)
    scenario.scenario_type = "highway"
    agent = FastBranchAgent(
        cfg,
        fast_state=snapshot.fast_state,
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
        min_ttc = min(min_ttc, float(event.get("ttc", float("inf"))))
        collision = bool(collision or event.get("crashed", False))
        completed_steps += 1
        if offset == 0:
            release_fast_action = int(agent.fast_actions[frame])
            release_legal_actions = tuple(agent.legal_actions[frame])
            release_effective_action = int(state["action"])
            release_target_speed = _target_speed(env)
            release_acceleration = _acceleration(env)
        if done:
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
    delay_steps = {float(delay): int(math.ceil(10.0 * float(delay))) for delay in delays}

    release_specs: List[Tuple[str, int, float, int]] = []
    for allocator in ("RGD", "TTC-risk"):
        for query_frame in selected[allocator]:
            release_frame = int(query_frame + delay_steps[1.7])
            if release_frame + horizon <= len(records):
                release_specs.append((allocator, int(query_frame), 1.7, release_frame))
    # Fixed-query latency erosion: reuse only the RGD queries that have a full
    # rollout horizon at every prespecified delay.
    max_delay_steps = max(delay_steps.values())
    fixed_queries = [
        int(frame)
        for frame in selected["RGD"]
        if int(frame + max_delay_steps + horizon) <= len(records)
    ]
    for query_frame in fixed_queries:
        for delay in delays:
            release_specs.append(
                ("RGD-fixed", query_frame, float(delay), int(query_frame + delay_steps[float(delay)]))
            )

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
        query_opportunity, query_priority, query_alternatives = gate_values(record, delay)
        event_rows.append(
            {
                "seed": int(seed),
                "allocator": allocator,
                "query_frame": int(query_frame),
                "release_frame": int(release_frame),
                "delay_s": float(delay),
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
