"""Matched-rollout audit of the slow actions that actually changed control.

This analysis is deliberately separate from R-VoD. R-VoD asks whether any
legal alternative could improve the contemporaneous fast action. Here the
estimand is the paired effect of the action returned by the slow executor after
the runtime safety mapping, evaluated from the same release snapshot.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

from tools.analyze_release_state_rollouts import (  # noqa: E402
    RAW_ACTIONS,
    FastBranchAgent,
    _build_fast_config,
    _effective_identity,
    _run_branch,
    capture_release_snapshot,
)
from tools.analyze_v13_main_results import canonical_distinct_actuation  # noqa: E402
from dilu.runtime_episode_setup import create_episode_env  # noqa: E402
from dilu.runtime_frame_state import record_executed_history_frame  # noqa: E402
from dilu.runtime_frame_trace import create_episode_runtime_state  # noqa: E402
from dilu.runtime_support import advance_episode_frame, run_frame_protocol  # noqa: E402
from dilu.safety import UnifiedSafetySystem  # noqa: E402
from dilu.scenario import create_scenario  # noqa: E402


def _trajectory_hash(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        str(row["branch_trajectory_json"]).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def summarize_release_branches(
    event: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
) -> Dict[str, Any]:
    """Separate oracle opportunity from realized slow-proposal value."""
    raw_slow_action = int(event["closed_loop_released_slow_action"])
    actual_matches = [
        row for row in candidates if int(row["raw_action"]) == raw_slow_action
    ]
    if len(actual_matches) != 1:
        raise ValueError(
            f"expected one actual-slow branch for action {raw_slow_action}, "
            f"found {len(actual_matches)}"
        )
    actual = actual_matches[0]
    baseline_identity = _effective_identity(dict(baseline))

    unique_by_effect: Dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in candidates:
        identity = _effective_identity(dict(row))
        current = unique_by_effect.get(identity)
        if current is None or float(row["utility"]) > float(current["utility"]):
            unique_by_effect[identity] = row
    alternatives = [
        row for identity, row in unique_by_effect.items() if identity != baseline_identity
    ]
    oracle_best = max(alternatives, key=lambda row: float(row["utility"]), default=None)
    oracle_advantage = (
        float(oracle_best["utility"]) - float(baseline["utility"])
        if oracle_best is not None
        else float("-inf")
    )

    actual_distinct = _effective_identity(dict(actual)) != baseline_identity
    actual_advantage = (
        float(actual["utility"]) - float(baseline["utility"])
        if actual_distinct
        else 0.0
    )
    return {
        "seed": int(event["seed"]),
        "release_frame": int(event["frame"]),
        "released_raw_slow_action": raw_slow_action,
        "recorded_fast_action": int(event["closed_loop_execution_state_fast_action"]),
        "recorded_final_action": int(
            event.get("closed_loop_latency_executed_action", event.get("final_action", -1))
        ),
        "rollout_fast_effective_action": int(baseline["effective_action"]),
        "rollout_actual_effective_action": int(actual["effective_action"]),
        "rollout_action_matches_recorded_final": bool(
            int(actual["effective_action"])
            == int(
                event.get(
                    "closed_loop_latency_executed_action",
                    event.get("final_action", -1),
                )
            )
        ),
        "fast_utility": float(baseline["utility"]),
        "actual_slow_utility": float(actual["utility"]),
        "actual_slow_advantage": float(actual_advantage),
        "actual_slow_effect_distinct": bool(actual_distinct),
        "actual_slow_corrective": bool(
            actual_distinct and actual_advantage >= float(epsilon)
        ),
        "oracle_best_advantage": (
            float(oracle_advantage) if math.isfinite(oracle_advantage) else ""
        ),
        "oracle_corrective_opportunity": bool(
            oracle_best is not None and oracle_advantage >= float(epsilon)
        ),
        "oracle_best_effective_action": (
            "" if oracle_best is None else int(oracle_best["effective_action"])
        ),
        "fast_collision": int(baseline["collision"]),
        "actual_slow_collision": int(actual["collision"]),
        "collision_difference_actual_minus_fast": int(actual["collision"])
        - int(baseline["collision"]),
        "progress_difference_m_actual_minus_fast": float(actual["progress_m"])
        - float(baseline["progress_m"]),
        "min_ttc_difference_s_actual_minus_fast": float(actual["min_ttc"])
        - float(baseline["min_ttc"]),
        "fast_trajectory_sha256": _trajectory_hash(baseline),
        "actual_slow_trajectory_sha256": _trajectory_hash(actual),
        "horizon_steps": int(baseline["horizon_steps"]),
        "gamma": float(baseline["gamma"]),
        "epsilon": float(epsilon),
    }


def _load_release_events(bundle: Path) -> Dict[int, list[Dict[str, Any]]]:
    root = bundle / "rgd_fixed_policy" / "highway"
    selected: Dict[int, list[Dict[str, Any]]] = {}
    for event_path in sorted(root.glob("seed_*/event_logs/event_log_*.json")):
        payload = json.loads(event_path.read_text(encoding="utf-8"))
        seed = int(payload["episode_id"])
        events = []
        for source in payload.get("events", []):
            event = dict(source)
            event["seed"] = seed
            if canonical_distinct_actuation(event):
                events.append(event)
        if events:
            selected[seed] = events
    if not selected:
        raise ValueError("no canonical distinct actuations found")
    return selected


def _physical_frames(bundle: Path, seed: int) -> list[Dict[str, Any]]:
    path = (
        bundle
        / "rgd_fixed_policy"
        / "highway"
        / f"seed_{seed}"
        / f"ep_{seed}"
        / f"highway_{seed}_physical_frames.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("frames", []) or [])


def _capture_recorded_prefix_snapshots(
    cfg: Dict[str, Any],
    seed: int,
    target_frames: Sequence[int],
    physical_frames: Sequence[Mapping[str, Any]],
    scratch: Path,
):
    """Replay recorded executed actions while rebuilding fast-policy state.

    This path intentionally bypasses the current post-decision sanitizer. The
    historical trace is the authority for the prefix action, so a later safety
    implementation change cannot silently move the reconstructed release state.
    """
    targets = sorted({int(frame) for frame in target_frames})
    env, obs, _, _, _, close_after = create_episode_env(seed, cfg, str(scratch), [seed])
    scenario = create_scenario(env, "highway-v0", seed, None)
    scenario.scenario_type = "highway"
    forced_actions = {
        int(row.get("frame_id", index)): int(row["action_id"])
        for index, row in enumerate(physical_frames[: max(targets) + 1])
    }
    agent = FastBranchAgent(cfg, force_actions=forced_actions)
    safety = UnifiedSafetySystem(cfg)
    history = collections.deque(maxlen=int(cfg.get("history_window", 6) or 6))
    episode_state = create_episode_runtime_state()
    snapshots = {}
    max_position_error = 0.0
    try:
        for frame in range(max(targets) + 1):
            expected = physical_frames[frame]
            vehicle = getattr(getattr(env, "unwrapped", env), "vehicle", None)
            position_error = math.hypot(
                float(vehicle.position[0]) - float(expected["position_x"]),
                float(vehicle.position[1]) - float(expected["position_y"]),
            )
            max_position_error = max(max_position_error, position_error)
            if position_error > 1.0:
                raise RuntimeError(
                    f"seed {seed}: recorded-prefix position error exceeds 1 m "
                    f"({position_error:.3g} m) "
                    f"at frame {frame}"
                )
            if frame in targets:
                snapshots[frame] = capture_release_snapshot(
                    agent.inner,
                    frame=frame,
                    env=env,
                    obs=obs,
                    history=history,
                    previous_action=int(episode_state["action"]),
                )
            vehicle_state = {
                key: copy.deepcopy(getattr(vehicle, key))
                for key in (
                    "target_speed",
                    "speed_index",
                    "target_lane_index",
                )
                if hasattr(vehicle, key)
            }
            (
                _proposed_action,
                q,
                response,
                frame_state,
                t0,
                t_inf,
                _meta,
                frame_image,
                _driving_state,
                _description,
                decision_meta,
            ) = run_frame_protocol(
                frame=frame,
                env=env,
                sce=scenario,
                agent=agent,
                obs=obs,
                prev_action=int(episode_state["action"]),
                cfg=cfg,
                safety_system=safety,
                phys_rec=None,
                reas_rec=None,
                history_buffer=history,
                prev_frame_image=episode_state["prev_image"],
            )
            for key, value in vehicle_state.items():
                setattr(vehicle, key, value)
            recorded_action = int(expected["action_id"])
            agent.record_executed_action(recorded_action)
            record_executed_history_frame(history, frame_state, recorded_action)
            episode_state["action"] = recorded_action
            episode_state["prev_image"] = frame_image
            obs, collision_frame, done, terminal, reward = advance_episode_frame(
                frame=frame,
                env=env,
                sce=scenario,
                q=q,
                resp=response,
                action=recorded_action,
                t0=t0,
                t_inf=t_inf,
                safety_system=safety,
                phys_rec=None,
                collision_frame=int(episode_state["collision_frame"]),
                decision_meta=decision_meta,
                cfg=cfg,
            )
            episode_state["collision_frame"] = collision_frame
            episode_state["episode_reward"] = float(
                episode_state.get("episode_reward", 0.0) or 0.0
            ) + float(reward)
            if done and frame < max(targets):
                raise RuntimeError(
                    f"seed {seed}: recorded prefix terminated before frame {max(targets)} "
                    f"({terminal.get('terminal_cause', 'unknown')})"
                )
        return snapshots, max_position_error
    finally:
        if close_after:
            env.close()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty intervention table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / "formal_protocol.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.horizon <= 0 or not 0.0 < args.gamma <= 1.0 or args.epsilon < 0.0:
        raise ValueError("invalid matched-rollout parameters")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)

    events_by_seed = _load_release_events(args.bundle)
    rows: list[Dict[str, Any]] = []
    branch_rows: list[Dict[str, Any]] = []
    for seed, events in sorted(events_by_seed.items()):
        frames = _physical_frames(args.bundle, seed)
        cfg = _build_fast_config(args.protocol, seed, args.scratch_root / f"seed_{seed}")
        targets = [int(event["frame"]) for event in events]
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
            snapshots, max_position_error = _capture_recorded_prefix_snapshots(
                cfg,
                seed,
                targets,
                frames,
                args.scratch_root / f"seed_{seed}",
            )
            for event in events:
                release_frame = int(event["frame"])
                baseline = _run_branch(
                    snapshots[release_frame], cfg, seed, None, args.horizon, args.gamma
                )
                legal_actions = {
                    int(value)
                    for value in str(baseline["legal_actions"]).split(";")
                    if value != ""
                }
                candidates = [
                    _run_branch(
                        snapshots[release_frame],
                        cfg,
                        seed,
                        action,
                        args.horizon,
                        args.gamma,
                    )
                    for action in RAW_ACTIONS
                    if action in legal_actions
                ]
                row = summarize_release_branches(
                    event, baseline, candidates, epsilon=args.epsilon
                )
                row["prefix_replay_max_position_error_m"] = float(max_position_error)
                rows.append(row)
                branch_rows.append({**baseline, "branch_role": "matched_fast"})
                for candidate in candidates:
                    role = (
                        "actual_slow"
                        if int(candidate["raw_action"])
                        == int(event["closed_loop_released_slow_action"])
                        else "oracle_candidate"
                    )
                    branch_rows.append({**candidate, "branch_role": role})

    rows.sort(key=lambda row: (int(row["seed"]), int(row["release_frame"])))
    branch_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row["release_frame"]),
            str(row["raw_action"]),
        )
    )
    _write_csv(args.output_dir / "actual_intervention_effects.csv", rows)
    _write_csv(args.output_dir / "actual_intervention_branches.csv", branch_rows)
    summary = {
        "analysis_version": "actual_slow_matched_release_rollout_v1",
        "estimand": "actual slow proposal utility minus matched-fast utility at the same release snapshot",
        # Keep the event-level count separate from an action change that
        # survives the common safety/action mapping.  A canonical release can
        # be logged as different while becoming execution-equivalent to Fast.
        "n_canonical_action_changes": len(rows),
        "n_effect_distinct_interventions": sum(
            bool(row["actual_slow_effect_distinct"]) for row in rows
        ),
        "n_distinct_actuations": sum(
            bool(row["actual_slow_effect_distinct"]) for row in rows
        ),
        "actual_corrective_count": sum(bool(row["actual_slow_corrective"]) for row in rows),
        "oracle_opportunity_count": sum(
            bool(row["oracle_corrective_opportunity"]) for row in rows
        ),
        "mean_actual_slow_advantage": sum(
            float(row["actual_slow_advantage"]) for row in rows
        )
        / len(rows),
        "all_rollout_actions_match_recorded_final": all(
            bool(row["rollout_action_matches_recorded_final"]) for row in rows
        ),
        "rows": rows,
    }
    summary_path = args.output_dir / "actual_intervention_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    input_paths = [args.protocol, args.bundle / "result_bundle_manifest.json"]
    for seed in sorted(events_by_seed):
        input_paths.extend(
            [
                args.bundle
                / "rgd_fixed_policy"
                / "highway"
                / f"seed_{seed}"
                / "event_logs"
                / f"event_log_highway_{seed}_{seed}.json",
                args.bundle
                / "rgd_fixed_policy"
                / "highway"
                / f"seed_{seed}"
                / f"ep_{seed}"
                / f"highway_{seed}_physical_frames.json",
            ]
        )
    missing_inputs = [path for path in input_paths if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            "matched-rollout provenance input missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )

    output_paths = [
        args.output_dir / "actual_intervention_effects.csv",
        args.output_dir / "actual_intervention_branches.csv",
        summary_path,
    ]
    manifest = {
        "schema": "actual_slow_matched_release_rollout_manifest_v1",
        "accepted": bool(
            summary["all_rollout_actions_match_recorded_final"]
            and all(
                float(row["prefix_replay_max_position_error_m"]) <= 1e-6
                for row in rows
            )
        ),
        "analysis_version": summary["analysis_version"],
        "analysis_source": _portable_path(Path(__file__)),
        "analysis_source_sha256": _file_sha256(Path(__file__)),
        "bundle": _portable_path(args.bundle),
        "protocol": _portable_path(args.protocol),
        "parameters": {
            "horizon_steps": int(args.horizon),
            "gamma": float(args.gamma),
            "epsilon": float(args.epsilon),
            "prefix_position_tolerance_m": 1e-6,
        },
        "release_events": [
            {"seed": int(row["seed"]), "frame": int(row["release_frame"])}
            for row in rows
        ],
        "input_sha256": {
            _portable_path(path): _file_sha256(path) for path in input_paths
        },
        "outputs": [path.name for path in output_paths],
        "output_sha256": {
            path.name: _file_sha256(path) for path in output_paths
        },
        "summary": {
            key: summary[key]
            for key in (
                "n_canonical_action_changes",
                "n_effect_distinct_interventions",
                "actual_corrective_count",
                "oracle_opportunity_count",
                "mean_actual_slow_advantage",
                "all_rollout_actions_match_recorded_final",
            )
        },
    }
    if not manifest["accepted"]:
        raise RuntimeError("matched-rollout provenance checks did not pass")
    (args.output_dir / "actual_intervention_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
