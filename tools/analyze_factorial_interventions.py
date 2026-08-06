"""Matched local rollouts for every factorial release event.

Each saved pre-release snapshot is branched into (i) the Fast action selected at
that release state and (ii) the returned slow action, followed by the same
deterministic Fast controller.  Traffic is interactive in both branches; only
the starting state and exogenous simulator seed are matched.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import logging
import math
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.release_snapshot import (  # noqa: E402
    RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
    ReleaseSnapshot,
    validate_release_snapshot_policy_state,
)
from dilu.evaluation.factorial_replay import FACTORIAL_REPLAY_VERSION  # noqa: E402
from tools.audit_query_release_factorial import (  # noqa: E402
    LEGACY_REPLAY_VERSION as LEGACY_FACTORIAL_REPLAY_VERSION,
    audit_bundle,
)
from tools.analyze_query_release_factorial import (  # noqa: E402
    ARM_NAMES,
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_BOOTSTRAP_SEED,
    _read_csv,
    _read_json,
    _sha256_file,
    seed_bootstrap_indices,
    validate_bundle_contract,
)
from tools.analyze_release_state_rollouts import _run_branch  # noqa: E402
from tools.run_query_release_factorial import (  # noqa: E402
    _release_execution_is_distinct,
)


ANALYSIS_SCHEMA = "rgd_factorial_intervention_rollout_v1"
DEFAULT_HORIZON = 20
DEFAULT_GAMMA = 0.99
DEFAULT_EPSILON = 0.02
RELEASE_ACTION_COMPARISON_STAGE = (
    "post_release_guard_and_frame_safety_pre_actuator_bridge"
)
FINAL_ACTUATOR_ACTION_STAGE = "post_shared_actuator_bridge_pre_environment_step"

EVENT_ROW_FIELDS = (
    "arm",
    "seed",
    "request_id",
    "source_frame",
    "release_frame",
    "fast_action",
    "slow_action",
    "release_selected_action",
    "selection_stage_primitive_distinct",
    "candidate_effective_action",
    "final_actuator_action",
    "executed_action",
    "release_guard_rejected",
    "release_action_unavailable",
    "candidate_evaluable",
    "candidate_replay_unavailable",
    "first_step_actuator_distinct",
    "executed_first_step_actuator_distinct",
    "classification",
    "baseline_utility",
    "candidate_utility",
    "utility_delta",
    "baseline_normalized_return",
    "candidate_normalized_return",
    "normalized_return_delta",
    "baseline_collision",
    "candidate_collision",
    "collision_delta",
    "baseline_progress_m",
    "candidate_progress_m",
    "progress_delta_m",
    "baseline_min_ttc_s",
    "candidate_min_ttc_s",
    "min_ttc_delta_s",
    "baseline_mean_abs_jerk_mps3",
    "candidate_mean_abs_jerk_mps3",
    "mean_abs_jerk_delta_mps3",
    "baseline_steps_completed",
    "candidate_steps_completed",
    "baseline_terminal_cause",
    "candidate_terminal_cause",
    "baseline_completed_horizon",
    "candidate_completed_horizon",
    "baseline_branch_trajectory_json",
    "candidate_branch_trajectory_json",
    "baseline_branch_trajectory_sha256",
    "candidate_branch_trajectory_sha256",
    "horizon_steps",
    "gamma",
    "epsilon",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _single(paths: Iterable[Path], *, context: str) -> Path:
    values = list(paths)
    require(len(values) == 1, f"{context}: expected one artifact, found {len(values)}")
    return values[0]


def _fast_continuation_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(config))
    cfg["protocol_name"] = "factorial_intervention_fast_continuation"
    cfg["system_routing"] = {"simple": "fast", "complex": "fast"}
    replay = copy.deepcopy(dict(cfg.get("closed_loop_latency_replay", {}) or {}))
    replay["enable"] = False
    replay["target_systems"] = []
    cfg["closed_loop_latency_replay"] = replay
    cfg["capture_release_snapshots_online"] = False
    cfg["require_release_snapshot_on_release"] = False
    return cfg


def _load_snapshot_bundle(seed_dir: Path) -> Dict[str, ReleaseSnapshot]:
    root = seed_dir / "release_snapshots"
    manifest_path = _single(root.glob("release_snapshots_*.json"), context=str(root))
    manifest = _read_json(manifest_path)
    require(
        manifest.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
        f"{seed_dir}: snapshot bundle schema drift",
    )
    bundle_name = str(manifest.get("bundle_file", "") or "")
    require(bool(bundle_name), f"{seed_dir}: snapshot manifest omits bundle file")
    bundle_path = root / bundle_name
    require(bundle_path.is_file(), f"{seed_dir}: snapshot bundle missing")
    require(
        _sha256_file(bundle_path) == str(manifest.get("bundle_sha256", "") or ""),
        f"{seed_dir}: snapshot bundle SHA256 mismatch",
    )
    payload = pickle.loads(bundle_path.read_bytes())
    require(
        isinstance(payload, Mapping)
        and payload.get("schema") == RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
        f"{seed_dir}: invalid snapshot pickle payload",
    )
    snapshots = dict(payload.get("snapshots", {}) or {})
    require(
        len(snapshots) == int(manifest.get("snapshot_count", -1)),
        f"{seed_dir}: snapshot count mismatch",
    )
    manifest_rows = {
        str(row.get("request_id", "") or ""): dict(row)
        for row in list(manifest.get("snapshots", []) or [])
    }
    require(set(manifest_rows) == set(snapshots), f"{seed_dir}: snapshot key drift")
    for request_id, snapshot in snapshots.items():
        require(isinstance(snapshot, ReleaseSnapshot), f"{seed_dir}: invalid snapshot type")
        validate_release_snapshot_policy_state(
            snapshot,
            context=f"{seed_dir.name} request {request_id}",
        )
        row = manifest_rows[request_id]
        require(
            str(snapshot.snapshot_identity_sha256)
            == str(row.get("snapshot_identity_sha256", "") or ""),
            f"{seed_dir}: snapshot identity manifest mismatch",
        )
    return snapshots


def _load_release_events(seed_dir: Path) -> Dict[str, Dict[str, Any]]:
    event_path = _single(
        (seed_dir / "event_logs").glob("event_log_*.json"),
        context=f"{seed_dir} event log",
    )
    payload = _read_json(event_path)
    events: Dict[str, Dict[str, Any]] = {}
    for raw in list(payload.get("events", []) or []):
        event = dict(raw)
        if not bool(event.get("closed_loop_latency_release_event", False)):
            continue
        request_id = str(event.get("closed_loop_latency_request_id", "") or "")
        require(bool(request_id), f"{seed_dir}: release event has no request ID")
        require(request_id not in events, f"{seed_dir}: duplicate release request ID")
        require(
            bool(event.get("closed_loop_release_snapshot_captured", False)),
            f"{seed_dir}: release event lacks snapshot",
        )
        events[request_id] = event
    return events


def _same_effective_action(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_target = float(left["target_speed_after"])
    right_target = float(right["target_speed_after"])
    require(
        math.isfinite(left_target) and math.isfinite(right_target),
        "first-step actuator equivalence requires finite target speeds",
    )
    if int(left["effective_action"]) != int(right["effective_action"]):
        return False
    return math.isclose(
        left_target, right_target, rel_tol=0.0, abs_tol=1e-6
    )


def _finite_or_blank(value: float) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return number if math.isfinite(number) else ""


def _trajectory_json(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("branch_trajectory_json")
    require(isinstance(value, str) and bool(value), f"{context}: branch trajectory is missing")
    try:
        trajectory = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: branch trajectory is invalid JSON") from exc
    require(isinstance(trajectory, list) and bool(trajectory), f"{context}: branch trajectory is empty")
    return value


def _trajectory_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _action(value: Any, field: str, *, context: str) -> int:
    require(not isinstance(value, bool), f"{context}: {field} must be an action")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: {field} must be an action") from exc
    require(
        math.isfinite(number) and number == int(number) and int(number) in range(5),
        f"{context}: {field} must be an action",
    )
    return int(number)


def _validate_v4_release_action_contract(
    event: Mapping[str, Any],
    *,
    context: str,
) -> Dict[str, Any]:
    """Authenticate the recorded release boundary before matched rollouts."""
    row = dict(event)
    require(
        bool(row.get("closed_loop_latency_release_event", False)),
        f"{context}: event is not a release",
    )
    require(
        str(row.get("release_action_comparison_stage", "") or "")
        == RELEASE_ACTION_COMPARISON_STAGE,
        f"{context}: release action comparison stage drift",
    )
    require(
        str(row.get("final_actuator_action_stage", "") or "")
        == FINAL_ACTUATOR_ACTION_STAGE,
        f"{context}: final actuator action stage drift",
    )
    fast = _action(
        row.get("release_fast_comparator_action"),
        "release_fast_comparator_action",
        context=context,
    )
    selected = _action(
        row.get("release_selected_action"),
        "release_selected_action",
        context=context,
    )
    aligned_fast = _action(
        row.get("closed_loop_release_action_alignment_fast_effective_action"),
        "closed_loop_release_action_alignment_fast_effective_action",
        context=context,
    )
    aligned_slow = _action(
        row.get("closed_loop_release_action_alignment_slow_effective_action"),
        "closed_loop_release_action_alignment_slow_effective_action",
        context=context,
    )
    final_action = _action(
        row.get("final_actuator_action"),
        "final_actuator_action",
        context=context,
    )
    executed = _action(
        row.get("closed_loop_latency_executed_action"),
        "closed_loop_latency_executed_action",
        context=context,
    )
    require(
        aligned_fast == fast,
        f"{context}: release Fast alignment drift",
    )
    require(
        final_action == executed,
        f"{context}: final actuator/executed action drift",
    )
    rejected = bool(row.get("closed_loop_release_opportunity_rejected", False))
    unavailable = bool(row.get("closed_loop_release_action_unavailable", False))
    require(
        not (rejected and unavailable),
        f"{context}: release cannot be both rejected and unavailable",
    )
    selection_distinct = selected != fast
    require(
        isinstance(row.get("release_selection_distinct"), bool)
        and bool(row["release_selection_distinct"]) == selection_distinct,
        f"{context}: release selection distinctness drift",
    )
    outcome = str(row.get("closed_loop_latency_terminal_outcome", "") or "")
    if unavailable:
        require(
            selected == fast and not selection_distinct,
            f"{context}: unavailable release must retain the Fast selection",
        )
        require(outcome == "unavailable", f"{context}: unavailable terminal outcome drift")
    elif rejected:
        require(
            selected == fast and not selection_distinct,
            f"{context}: rejected release must retain the Fast selection",
        )
        require(outcome == "fast_equivalent", f"{context}: rejected terminal outcome drift")
    else:
        require(
            aligned_slow == selected,
            f"{context}: release selected action/slow alignment drift",
        )
        expected_outcome = "distinct_actuation" if selection_distinct else "fast_equivalent"
        require(
            outcome == expected_outcome,
            f"{context}: terminal outcome disagrees with release selection",
        )
    return {
        "fast_action": fast,
        "selected_action": selected,
        "final_action": final_action,
        "executed_action": executed,
        "selection_stage_primitive_distinct": int(selection_distinct),
        "rejected": rejected,
        "unavailable": unavailable,
    }


def _event_effect_row(
    *,
    arm: str,
    seed: int,
    request_id: str,
    source_frame: int,
    frame: int,
    fast_action: int,
    slow_action: int,
    selected_action: Any,
    selection_distinct: Any,
    final_action: int,
    rejected: bool,
    unavailable: bool,
    baseline: Mapping[str, Any],
    horizon: int,
    gamma: float,
    epsilon: float,
) -> Dict[str, Any]:
    baseline_trajectory = _trajectory_json(
        baseline, context=f"{arm}/{seed}/{request_id} matched Fast"
    )
    result = {
        "arm": str(arm),
        "seed": int(seed),
        "request_id": str(request_id),
        "source_frame": int(source_frame),
        "release_frame": int(frame),
        "fast_action": int(fast_action),
        "slow_action": int(slow_action),
        "release_selected_action": selected_action,
        "selection_stage_primitive_distinct": selection_distinct,
        "candidate_effective_action": "",
        "final_actuator_action": int(final_action),
        "executed_action": int(final_action),
        "release_guard_rejected": int(rejected),
        "release_action_unavailable": int(unavailable),
        "candidate_evaluable": 0,
        "candidate_replay_unavailable": 1,
        "first_step_actuator_distinct": 0,
        "executed_first_step_actuator_distinct": 0,
        "classification": "unavailable",
        "baseline_utility": float(baseline["utility"]),
        "candidate_utility": "",
        "utility_delta": "",
        "baseline_normalized_return": float(baseline["normalized_return"]),
        "candidate_normalized_return": "",
        "normalized_return_delta": "",
        "baseline_collision": int(baseline["collision"]),
        "candidate_collision": "",
        "collision_delta": "",
        "baseline_progress_m": float(baseline["progress_m"]),
        "candidate_progress_m": "",
        "progress_delta_m": "",
        "baseline_min_ttc_s": _finite_or_blank(baseline["min_ttc"]),
        "candidate_min_ttc_s": "",
        "min_ttc_delta_s": "",
        "baseline_mean_abs_jerk_mps3": _finite_or_blank(
            baseline.get("mean_abs_jerk_mps3", float("nan"))
        ),
        "candidate_mean_abs_jerk_mps3": "",
        "mean_abs_jerk_delta_mps3": "",
        "baseline_steps_completed": int(baseline["steps_completed"]),
        "candidate_steps_completed": "",
        "baseline_terminal_cause": str(baseline.get("terminal_cause", "unknown")),
        "candidate_terminal_cause": "",
        "baseline_completed_horizon": int(bool(baseline.get("completed_horizon", False))),
        "candidate_completed_horizon": "",
        "baseline_branch_trajectory_json": baseline_trajectory,
        "candidate_branch_trajectory_json": "",
        "baseline_branch_trajectory_sha256": _trajectory_sha256(baseline_trajectory),
        "candidate_branch_trajectory_sha256": "",
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
    }
    require(
        tuple(result) == EVENT_ROW_FIELDS,
        f"{arm}/{seed}/{request_id}: unavailable event-row schema drift",
    )
    return result


def _event_rollout_row(
    *,
    arm: str,
    seed: int,
    request_id: str,
    event: Mapping[str, Any],
    snapshot: ReleaseSnapshot,
    cfg: Mapping[str, Any],
    horizon: int,
    gamma: float,
    epsilon: float,
    legacy_v2: bool = False,
) -> Dict[str, Any]:
    context = f"{arm}/{seed}/{request_id}"
    frame = int(event.get("frame", -1))
    source_frame = int(event.get("closed_loop_latency_source_frame", -1))
    require(frame == int(snapshot.frame), f"{context}: release frame drift")
    require(
        source_frame == int(snapshot.source_frame),
        f"{context}: source frame drift",
    )
    require(
        request_id == str(snapshot.request_id),
        f"{context}: snapshot request mismatch",
    )
    require(
        str(event.get("closed_loop_release_snapshot_identity_sha256", "") or "")
        == str(snapshot.snapshot_identity_sha256),
        f"{context}: event/snapshot identity mismatch",
    )

    if legacy_v2:
        fast_action = _action(
            event.get("closed_loop_execution_state_fast_action"),
            "closed_loop_execution_state_fast_action",
            context=context,
        )
        final_action = _action(
            event.get("closed_loop_latency_executed_action"),
            "closed_loop_latency_executed_action",
            context=context,
        )
        selected_action: Any = ""
        selection_distinct: Any = ""
        rejected = bool(event.get("closed_loop_release_opportunity_rejected", False))
        unavailable = bool(event.get("closed_loop_release_action_unavailable", False))
    else:
        release = _validate_v4_release_action_contract(event, context=context)
        fast_action = int(release["fast_action"])
        final_action = int(release["final_action"])
        selected_action = int(release["selected_action"])
        selection_distinct = int(release["selection_stage_primitive_distinct"])
        rejected = bool(release["rejected"])
        unavailable = bool(release["unavailable"])

    baseline = _run_branch(snapshot, dict(cfg), seed, None, horizon, gamma)
    slow_action = int(event.get("closed_loop_released_slow_action", -1))
    require(slow_action in range(5), f"{context}: invalid slow action")
    require(
        int(baseline["fast_action"]) == fast_action,
        f"{context}: matched Fast proposal drift",
    )
    require(
        int(baseline["effective_action"]) == fast_action,
        f"{context}: matched Fast effective action drift",
    )

    try:
        candidate = _run_branch(snapshot, dict(cfg), seed, slow_action, horizon, gamma)
    except (RuntimeError, ValueError):
        if not unavailable:
            raise
        return _event_effect_row(
            arm=arm,
            seed=seed,
            request_id=request_id,
            source_frame=source_frame,
            frame=frame,
            fast_action=fast_action,
            slow_action=slow_action,
            selected_action=selected_action,
            selection_distinct=selection_distinct,
            final_action=final_action,
            rejected=rejected,
            unavailable=unavailable,
            baseline=baseline,
            horizon=horizon,
            gamma=gamma,
            epsilon=epsilon,
        )
    if unavailable:
        raise ValueError(f"{context}: unavailable release has a matched candidate is executable")

    first_step_distinct = not _same_effective_action(baseline, candidate)
    candidate_reproduces_execution = int(candidate["effective_action"]) == final_action
    if not rejected and not candidate_reproduces_execution:
        classification = "execution_reproduction_mismatch"
        candidate_evaluable = 0
        executed_first_step_distinct = 0
    else:
        candidate_evaluable = 1
        executed_first_step_distinct = int(
            first_step_distinct and not rejected and candidate_reproduces_execution
        )
        if not first_step_distinct:
            classification = "first_step_actuator_equivalent"
        else:
            utility_delta = float(candidate["utility"]) - float(baseline["utility"])
            if utility_delta >= float(epsilon):
                classification = "beneficial"
            elif utility_delta <= -float(epsilon):
                classification = "harmful"
            else:
                classification = "neutral"

    utility_delta = float(candidate["utility"]) - float(baseline["utility"])
    return_delta = float(candidate["normalized_return"]) - float(
        baseline["normalized_return"]
    )
    progress_delta = float(candidate["progress_m"]) - float(baseline["progress_m"])
    collision_delta = int(candidate["collision"]) - int(baseline["collision"])
    baseline_ttc = float(baseline["min_ttc"])
    candidate_ttc = float(candidate["min_ttc"])
    ttc_delta = (
        candidate_ttc - baseline_ttc
        if math.isfinite(candidate_ttc) and math.isfinite(baseline_ttc)
        else float("nan")
    )
    baseline_jerk = float(baseline.get("mean_abs_jerk_mps3", float("nan")))
    candidate_jerk = float(candidate.get("mean_abs_jerk_mps3", float("nan")))
    jerk_delta = (
        candidate_jerk - baseline_jerk
        if math.isfinite(candidate_jerk) and math.isfinite(baseline_jerk)
        else float("nan")
    )
    baseline_trajectory = _trajectory_json(
        baseline, context=f"{context}: matched Fast"
    )
    candidate_trajectory = _trajectory_json(
        candidate, context=f"{context}: matched slow"
    )
    result = {
        "arm": str(arm),
        "seed": int(seed),
        "request_id": str(request_id),
        "source_frame": source_frame,
        "release_frame": frame,
        "fast_action": fast_action,
        "slow_action": slow_action,
        "release_selected_action": selected_action,
        "selection_stage_primitive_distinct": selection_distinct,
        "candidate_effective_action": int(candidate["effective_action"]),
        "final_actuator_action": final_action,
        "executed_action": final_action,
        "release_guard_rejected": int(rejected),
        "release_action_unavailable": int(unavailable),
        "candidate_evaluable": candidate_evaluable,
        "candidate_replay_unavailable": 0,
        "first_step_actuator_distinct": int(first_step_distinct),
        "executed_first_step_actuator_distinct": executed_first_step_distinct,
        "classification": classification,
        "baseline_utility": float(baseline["utility"]),
        "candidate_utility": float(candidate["utility"]),
        "utility_delta": utility_delta if candidate_evaluable else "",
        "baseline_normalized_return": float(baseline["normalized_return"]),
        "candidate_normalized_return": float(candidate["normalized_return"]),
        "normalized_return_delta": return_delta if candidate_evaluable else "",
        "baseline_collision": int(baseline["collision"]),
        "candidate_collision": int(candidate["collision"]),
        "collision_delta": collision_delta if candidate_evaluable else "",
        "baseline_progress_m": float(baseline["progress_m"]),
        "candidate_progress_m": float(candidate["progress_m"]),
        "progress_delta_m": progress_delta if candidate_evaluable else "",
        "baseline_min_ttc_s": _finite_or_blank(baseline_ttc),
        "candidate_min_ttc_s": _finite_or_blank(candidate_ttc) if candidate_evaluable else "",
        "min_ttc_delta_s": _finite_or_blank(ttc_delta) if candidate_evaluable else "",
        "baseline_mean_abs_jerk_mps3": _finite_or_blank(baseline_jerk),
        "candidate_mean_abs_jerk_mps3": _finite_or_blank(candidate_jerk),
        "mean_abs_jerk_delta_mps3": _finite_or_blank(jerk_delta) if candidate_evaluable else "",
        "baseline_steps_completed": int(baseline["steps_completed"]),
        "candidate_steps_completed": int(candidate["steps_completed"]),
        "baseline_terminal_cause": str(baseline.get("terminal_cause", "unknown")),
        "candidate_terminal_cause": str(candidate.get("terminal_cause", "unknown")),
        "baseline_completed_horizon": int(bool(baseline.get("completed_horizon", False))),
        "candidate_completed_horizon": int(bool(candidate.get("completed_horizon", False))),
        "baseline_branch_trajectory_json": baseline_trajectory,
        "candidate_branch_trajectory_json": candidate_trajectory,
        "baseline_branch_trajectory_sha256": _trajectory_sha256(baseline_trajectory),
        "candidate_branch_trajectory_sha256": _trajectory_sha256(candidate_trajectory),
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
    }
    require(tuple(result) == EVENT_ROW_FIELDS, f"{context}: event-row schema drift")
    return result


def _process_cell(task: Mapping[str, Any]) -> List[Dict[str, Any]]:
    arm = str(task["arm"])
    seed = int(task["seed"])
    seed_dir = Path(str(task["seed_dir"]))
    snapshot_payload = _read_json(seed_dir / "experiment_snapshot.json")
    cfg = _fast_continuation_config(dict(snapshot_payload.get("config", {}) or {}))
    events = _load_release_events(seed_dir)
    snapshots = _load_snapshot_bundle(seed_dir)
    require(set(events) == set(snapshots), f"{arm}/{seed}: release/snapshot key mismatch")
    require(
        len(events) == int(task["expected_releases"]),
        f"{arm}/{seed}: release count disagrees with factorial row",
    )
    previous_disable = int(logging.root.manager.disable)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            logging.disable(logging.CRITICAL)
            try:
                return [
                    _event_rollout_row(
                        arm=arm,
                        seed=seed,
                        request_id=request_id,
                        event=events[request_id],
                        snapshot=snapshots[request_id],
                        cfg=cfg,
                        horizon=int(task["horizon"]),
                        gamma=float(task["gamma"]),
                        epsilon=float(task["epsilon"]),
                        legacy_v2=bool(task.get("legacy_v2", False)),
                    )
                    for request_id in sorted(
                        events,
                        key=lambda key: (
                            int(events[key].get("frame", -1)),
                            str(key),
                        ),
                    )
                ]
            finally:
                logging.disable(previous_disable)


def _bootstrap_pooled(
    values_by_seed: Mapping[int, Sequence[float]],
    seeds: Sequence[int],
    indices: np.ndarray,
) -> Tuple[float, float, float, int]:
    pooled = [float(value) for seed in seeds for value in values_by_seed[int(seed)]]
    if not pooled:
        return float("nan"), float("nan"), float("nan"), 0
    samples: List[float] = []
    for draw in indices:
        values = [
            float(value)
            for position in draw
            for value in values_by_seed[int(seeds[int(position)])]
        ]
        if values:
            samples.append(float(np.mean(values)))
    require(bool(samples), "pooled seed bootstrap produced no valid draws")
    low, high = np.quantile(np.asarray(samples), [0.025, 0.975])
    return float(np.mean(pooled)), float(low), float(high), len(samples)


def _row_int(
    row: Mapping[str, Any],
    field: str,
    *,
    legacy_field: Optional[str] = None,
) -> int:
    value = row.get(field)
    if value is None and legacy_field is not None:
        value = row.get(legacy_field)
    if value in (None, ""):
        return 0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid intervention field {field}: {value!r}") from exc
    require(math.isfinite(number) and number == int(number), f"invalid intervention field {field}: {value!r}")
    return int(number)


def _row_finite(row: Mapping[str, Any], field: str) -> Optional[float]:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid intervention field {field}: {value!r}") from exc
    return number if math.isfinite(number) else None


def _summary_row(
    *,
    arm: str,
    metric: str,
    values: Sequence[float],
    indices: np.ndarray,
    draws: int,
    denominator: int,
) -> Dict[str, Any]:
    samples = np.mean(np.asarray(values, dtype=float)[indices], axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "arm": str(arm),
        "metric": str(metric),
        "estimand": "mean_per_simulator_seed",
        "estimate": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "numerator": "",
        "denominator": int(denominator),
        "n_seed_blocks": int(denominator),
        "bootstrap_draws": int(draws),
        "valid_bootstrap_draws": int(draws),
    }


def summarize_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    draws: int,
    bootstrap_seed: int,
    arms: Optional[Sequence[str]] = None,
    allow_custom_arms: bool = False,
) -> List[Dict[str, Any]]:
    ordered_seeds = tuple(sorted(int(seed) for seed in seeds))
    require(bool(ordered_seeds), "intervention summary requires at least one seed")
    ordered_arms = tuple(
        dict.fromkeys(str(arm) for arm in (ARM_NAMES if arms is None else arms))
    )
    require(bool(ordered_arms), "intervention summary requires at least one arm")
    if not allow_custom_arms:
        require(
            all(arm in ARM_NAMES for arm in ordered_arms),
            "intervention summary contains an unknown arm",
        )
    indices = seed_bootstrap_indices(
        len(ordered_seeds), draws=int(draws), bootstrap_seed=int(bootstrap_seed)
    )
    summaries: List[Dict[str, Any]] = []
    for arm in ordered_arms:
        arm_rows = [dict(row) for row in rows if str(row.get("arm", "")) == arm]
        by_seed = {
            seed: [row for row in arm_rows if int(row.get("seed", -1)) == seed]
            for seed in ordered_seeds
        }

        def evaluable(row: Mapping[str, Any]) -> bool:
            value = _row_finite(row, "utility_delta")
            if value is None:
                return False
            if "candidate_evaluable" not in row:
                return True
            return bool(_row_int(row, "candidate_evaluable"))

        def first_step_distinct(row: Mapping[str, Any]) -> bool:
            return bool(
                _row_int(
                    row,
                    "first_step_actuator_distinct",
                    legacy_field="candidate_effect_distinct",
                )
            )

        def executed_distinct(row: Mapping[str, Any]) -> bool:
            return bool(
                _row_int(
                    row,
                    "executed_first_step_actuator_distinct",
                    legacy_field="executed_distinct",
                )
            )

        def beneficial(row: Mapping[str, Any]) -> bool:
            return (
                evaluable(row)
                and first_step_distinct(row)
                and str(row.get("classification", "")) == "beneficial"
            )

        per_seed_metrics = {
            "release_events_per_seed": [float(len(by_seed[seed])) for seed in ordered_seeds],
            "evaluable_release_events_per_seed": [
                float(sum(evaluable(row) for row in by_seed[seed]))
                for seed in ordered_seeds
            ],
            "first_step_actuator_distinct_per_seed": [
                float(
                    sum(
                        first_step_distinct(row)
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
            "executed_first_step_actuator_distinct_per_seed": [
                float(
                    sum(
                        executed_distinct(row)
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
            "selection_stage_primitive_distinct_per_seed": [
                float(
                    sum(
                        _row_int(row, "selection_stage_primitive_distinct")
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
            "selected_utility_gain_per_seed": [
                float(
                    sum(
                        value
                        for row in by_seed[seed]
                        if evaluable(row)
                        and executed_distinct(row)
                        and (value := _row_finite(row, "utility_delta")) is not None
                    )
                )
                for seed in ordered_seeds
            ],
            "rejected_beneficial_per_seed": [
                float(
                    sum(
                        beneficial(row)
                        and bool(_row_int(row, "release_guard_rejected"))
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
            "false_accepts_per_seed": [
                float(
                    sum(
                        evaluable(row)
                        and executed_distinct(row)
                        and str(row.get("classification", "")) == "harmful"
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
            "false_rejects_per_seed": [
                float(
                    sum(
                        beneficial(row)
                        and bool(_row_int(row, "release_guard_rejected"))
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
            "missed_beneficial_opportunities_per_seed": [
                float(
                    sum(
                        beneficial(row) and not executed_distinct(row)
                        for row in by_seed[seed]
                    )
                )
                for seed in ordered_seeds
            ],
        }
        for metric, values in per_seed_metrics.items():
            summaries.append(
                _summary_row(
                    arm=arm,
                    metric=metric,
                    values=values,
                    indices=indices,
                    draws=int(draws),
                    denominator=len(ordered_seeds),
                )
            )

        selected = [
            row
            for row in arm_rows
            if evaluable(row) and executed_distinct(row)
        ]
        for field, metric in (
            ("utility_delta", "utility_delta_per_executed_first_step_intervention"),
            (
                "normalized_return_delta",
                "normalized_return_delta_per_executed_first_step_intervention",
            ),
            ("collision_delta", "collision_delta_per_executed_first_step_intervention"),
            ("progress_delta_m", "progress_delta_m_per_executed_first_step_intervention"),
            ("min_ttc_delta_s", "min_ttc_delta_s_per_executed_first_step_intervention"),
            (
                "mean_abs_jerk_delta_mps3",
                "mean_abs_jerk_delta_mps3_per_executed_first_step_intervention",
            ),
        ):
            values_by_seed = {
                seed: [
                    value
                    for row in by_seed[seed]
                    if evaluable(row)
                    and executed_distinct(row)
                    and (value := _row_finite(row, field)) is not None
                ]
                for seed in ordered_seeds
            }
            point, low, high, valid = _bootstrap_pooled(
                values_by_seed, ordered_seeds, indices
            )
            summaries.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "estimand": "event_conditional_cluster_bootstrap",
                    "estimate": _finite_or_blank(point),
                    "ci_low": _finite_or_blank(low),
                    "ci_high": _finite_or_blank(high),
                    "numerator": "",
                    "denominator": sum(len(values) for values in values_by_seed.values()),
                    "n_seed_blocks": len(ordered_seeds),
                    "bootstrap_draws": int(draws),
                    "valid_bootstrap_draws": int(valid),
                }
            )

        def append_rate(
            metric: str,
            *,
            denominator_predicate,
            numerator_predicate,
        ) -> None:
            values_by_seed = {
                seed: [
                    float(bool(numerator_predicate(row)))
                    for row in by_seed[seed]
                    if denominator_predicate(row)
                ]
                for seed in ordered_seeds
            }
            point, low, high, valid = _bootstrap_pooled(
                values_by_seed, ordered_seeds, indices
            )
            denominator_rows = [
                row for row in arm_rows if denominator_predicate(row)
            ]
            summaries.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "estimand": "event_conditional_cluster_bootstrap",
                    "estimate": _finite_or_blank(point),
                    "ci_low": _finite_or_blank(low),
                    "ci_high": _finite_or_blank(high),
                    "numerator": sum(
                        bool(numerator_predicate(row)) for row in denominator_rows
                    ),
                    "denominator": len(denominator_rows),
                    "n_seed_blocks": len(ordered_seeds),
                    "bootstrap_draws": int(draws),
                    "valid_bootstrap_draws": int(valid),
                }
            )

        append_rate(
            "beneficial_fraction_of_executed_first_step_interventions",
            denominator_predicate=lambda row: evaluable(row) and executed_distinct(row),
            numerator_predicate=beneficial,
        )
        append_rate(
            "executed_distinct_coverage_of_evaluable_releases",
            denominator_predicate=evaluable,
            numerator_predicate=executed_distinct,
        )
        append_rate(
            "harmful_fraction_of_executed_first_step_interventions",
            denominator_predicate=lambda row: evaluable(row) and executed_distinct(row),
            numerator_predicate=lambda row: str(row.get("classification", "")) == "harmful",
        )
        append_rate(
            "beneficial_fraction_of_release_guard_rejections",
            denominator_predicate=lambda row: (
                evaluable(row)
                and first_step_distinct(row)
                and bool(_row_int(row, "release_guard_rejected"))
            ),
            numerator_predicate=beneficial,
        )
        append_rate(
            "missed_beneficial_fraction_of_evaluable_distinct_candidates",
            denominator_predicate=lambda row: evaluable(row) and first_step_distinct(row),
            numerator_predicate=lambda row: beneficial(row) and not executed_distinct(row),
        )
    return summaries


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
) -> None:
    require(bool(rows) or bool(fieldnames), f"refusing to write empty table: {path}")
    columns = list(fieldnames or rows[0])
    require(
        all(tuple(row) == tuple(columns) for row in rows),
        f"inconsistent CSV row schema: {path}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _audit_factorial_bundle(
    bundle: Path,
    *,
    replay_version: str,
    lifecycle_mode: str,
    label: str,
) -> Dict[str, Any]:
    report = dict(audit_bundle(Path(bundle)))
    require(report.get("accepted") is True, f"{label} request audit was not accepted")
    require(
        str(report.get("factorial_replay_version", "") or "") == replay_version,
        f"{label} replay version drift",
    )
    audit_contract = report.get("audit_contract")
    require(
        isinstance(audit_contract, Mapping)
        and audit_contract.get("all_request_ids_lifecycle_closed") is True,
        f"{label} request lifecycle audit is incomplete",
    )
    cells = report.get("cells")
    require(isinstance(cells, list) and bool(cells), f"{label} request audit has no cells")
    for index, cell in enumerate(cells):
        require(
            isinstance(cell, Mapping) and cell.get("accepted") is True,
            f"{label} request audit cell {index} was not accepted",
        )
        observed = str(cell.get("lifecycle_mode", "") or "")
        require(
            observed == lifecycle_mode,
            f"{label} ambiguous lifecycle mode: {observed or '<missing>'}",
        )
    return report


def _audit_request_bundle(bundle: Path) -> Dict[str, Any]:
    return _audit_factorial_bundle(
        bundle,
        replay_version=FACTORIAL_REPLAY_VERSION,
        lifecycle_mode="explicit_dual_event_ids",
        label="request",
    )


def _audit_legacy_v2_bundle(bundle: Path) -> Dict[str, Any]:
    return _audit_factorial_bundle(
        bundle,
        replay_version=LEGACY_FACTORIAL_REPLAY_VERSION,
        lifecycle_mode="legacy_v2_single_request_projection",
        label="legacy-v2",
    )


def _legacy_bundle_contract(
    rows: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    proposal_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Read the minimal v2 aggregate contract after its request audit passes."""
    require(isinstance(run_manifest, Mapping), "legacy run manifest is not an object")
    require(
        isinstance(proposal_manifest, Mapping),
        "legacy proposal manifest is not an object",
    )
    require(
        str(run_manifest.get("factorial_replay_version", "") or "")
        == LEGACY_FACTORIAL_REPLAY_VERSION,
        "legacy run manifest replay version drift",
    )
    require(
        str(proposal_manifest.get("factorial_replay_version", "") or "")
        == LEGACY_FACTORIAL_REPLAY_VERSION,
        "legacy proposal manifest replay version drift",
    )
    try:
        seed_start = int(run_manifest["seed_start"])
        seed_count = int(run_manifest["seed_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("legacy run manifest has an invalid seed cohort") from exc
    require(seed_count > 0, "legacy run manifest has an empty seed cohort")
    run_hash = str(run_manifest.get("proposal_bank_sha256", "") or "")
    proposal_hash = str(proposal_manifest.get("bank_sha256", "") or "")
    require(
        len(run_hash) == 64 and run_hash == proposal_hash,
        "legacy run/proposal bank drift",
    )
    require(bool(rows), "legacy factorial result table is empty")
    return {
        "seeds": tuple(range(seed_start, seed_start + seed_count)),
        "proposal_bank_sha256": run_hash,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--draws", type=int, default=DEFAULT_BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--legacy-v2", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    require(args.horizon > 0, "horizon must be positive")
    require(0.0 < args.gamma <= 1.0, "gamma must lie in (0, 1]")
    require(args.epsilon >= 0.0, "epsilon must be nonnegative")
    require(args.draws > 0 and args.workers > 0, "draws and workers must be positive")
    selected_arms = tuple(dict.fromkeys(str(arm) for arm in args.arms))
    require(bool(selected_arms), "at least one arm must be selected")

    audit_report = (
        _audit_legacy_v2_bundle(args.bundle)
        if bool(args.legacy_v2)
        else _audit_request_bundle(args.bundle)
    )

    result_rows = _read_csv(args.bundle / "factorial_episode_results.csv")
    run_manifest = _read_json(args.bundle / "factorial_run_manifest.json")
    proposal_manifest = _read_json(args.bundle / "proposal_bank_manifest.json")
    contract = (
        _legacy_bundle_contract(result_rows, run_manifest, proposal_manifest)
        if bool(args.legacy_v2)
        else validate_bundle_contract(
            result_rows,
            run_manifest,
            proposal_manifest,
            audit_report=audit_report,
        )
    )
    seeds = tuple(int(seed) for seed in contract["seeds"])
    factorial_rows = {
        (int(row["seed"]), str(row["arm"])): row for row in result_rows
    }
    tasks = [
        {
            "arm": arm,
            "seed": seed,
            "seed_dir": str((args.bundle / arm / f"seed_{seed}").resolve()),
            "expected_releases": int(float(factorial_rows[(seed, arm)]["release_events"])),
            "horizon": int(args.horizon),
            "gamma": float(args.gamma),
            "epsilon": float(args.epsilon),
            "legacy_v2": bool(args.legacy_v2),
        }
        for seed in seeds
        for arm in selected_arms
        if int(float(factorial_rows[(seed, arm)]["release_events"])) > 0
    ]
    event_rows: List[Dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(_process_cell, task): task for task in tasks}
            for future in as_completed(futures):
                event_rows.extend(future.result())
    event_rows.sort(
        key=lambda row: (
            ARM_NAMES.index(str(row["arm"])),
            int(row["seed"]),
            int(row["release_frame"]),
            str(row["request_id"]),
        )
    )
    summaries = summarize_events(
        event_rows,
        seeds=seeds,
        draws=int(args.draws),
        bootstrap_seed=int(args.bootstrap_seed),
        arms=selected_arms,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "factorial_intervention_events.csv"
    summary_path = args.output_dir / "factorial_intervention_summary.csv"
    _write_csv(events_path, event_rows, fieldnames=EVENT_ROW_FIELDS)
    _write_csv(summary_path, summaries)
    manifest = {
        "schema": ANALYSIS_SCHEMA,
        "accepted": True,
        "source_bundle": str(args.bundle.resolve()),
        "proposal_bank_sha256": str(contract["proposal_bank_sha256"]),
        "factorial_replay_version": (
            LEGACY_FACTORIAL_REPLAY_VERSION
            if bool(args.legacy_v2)
            else FACTORIAL_REPLAY_VERSION
        ),
        "request_audit_schema": str(audit_report.get("schema", "") or ""),
        "arms": list(selected_arms),
        "seeds": list(seeds),
        "independent_unit": "simulator_seed",
        "release_snapshot_stage": "pre_release_frame_policy_decision",
        "branch_design": "matched_release_state_first_action_then_shared_fast_continuation",
        "traffic_model": "interactive_deterministic_policy_from_cloned_release_state",
        "fixed_traffic_replay_claimed": False,
        "horizon_steps": int(args.horizon),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "event_count": len(event_rows),
        "bootstrap": {
            "unit": "simulator_seed",
            "draws": int(args.draws),
            "seed": int(args.bootstrap_seed),
        },
        "artifacts": {
            events_path.name: _sha256_file(events_path),
            summary_path.name: _sha256_file(summary_path),
        },
        "summary": summaries,
    }
    manifest_path = args.output_dir / "factorial_intervention_manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(manifest_path))
    print(json.dumps({"accepted": True, "events": len(event_rows), "output": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
