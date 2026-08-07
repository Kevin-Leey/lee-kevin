"""Build outcome-grounded labels for frozen v12 release states.

This runner deliberately has no gate-selection logic.  Release frames are an
external, precommitted input.  For every unique ``(seed, release_frame)`` it
replays the matched Fast branch and every raw action in the exact per-frame
gate action universe through the same safety/bridge stack and Fast
continuation.  Source, snapshot, and first-step identities are checked before
the resulting corrective-set label is admitted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from importlib import metadata as importlib_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_release_state_rollouts import (  # noqa: E402
    RAW_ACTIONS,
    ReleaseSnapshot,
    _build_fast_config,
    _effective_identity,
    _load_trace,
    _run_branch,
    _trace_paths,
)
from dilu.evaluation.reporter import (  # noqa: E402
    _build_runtime_experiment_config,
    build_runtime_source_hash,
)
from dilu.driver_agent.policy_state import (  # noqa: E402
    DRIVER_POLICY_STATE_SCHEMA,
    policy_state_sha256,
    validate_driver_policy_state,
)
from tools.run_main_table_runtime import load_formal_protocol  # noqa: E402
from tools.v12_floor_overlay import (  # noqa: E402
    DEFAULT_LOCK_PATH,
    VerifiedFloorOverlay,
    assert_floor_overlay_applied,
    enforce_v12_floor_overlay_contract,
    load_optional_verified_floor_overlay,
)


METHOD_VERSION = "identifiable_gate_v12"
LABEL_SOURCE = "matched_release_state_exact_action_rollout_v1"
BRANCH_MANIFEST_SCHEMA = "v12_branch_runner_manifest_v1"
CONTINUATION_CONTRACT_VERSION = "executed_action_feedback_v1"
ACTION_UNIVERSE_SOURCE = "driving_state.effective_action_universe"
BRANCH_ENGINE_PATH = REPO_ROOT / "tools" / "analyze_release_state_rollouts.py"
SNAPSHOT_PRODUCER_PATH = REPO_ROOT / "tools" / "run_mechanism_inprocess.py"
BASE_CONFIG_PATH = REPO_ROOT / "config.yaml"
CHECKPOINT_SCHEMA_VERSION = 1
OUTPUT_SCHEMA_VERSION = 1
POSITION_TOLERANCE_M = 1e-6
SPEED_TOLERANCE_MPS = 1e-6
REQUIRED_SOURCE_ENVIRONMENT_FIELDS = frozenset(
    {
        "python_version",
        "platform",
        "pkg_numpy",
        "pkg_gymnasium",
        "pkg_highway-env",
    }
)

BRANCH_ROWS_FILE = "v12_branch_rows.csv"
LABELS_FILE = "v12_release_labels.csv"
ACCOUNTING_FILE = "v12_branch_accounting.csv"
MANIFEST_FILE = "v12_branch_manifest.json"
RUN_CONTRACT_FILE = "v12_branch_run_contract.json"
CHECKPOINT_DIR = "v12_branch_checkpoints"
TARGET_EVENT_FIELDS = (
    "seed",
    "delay_s",
    "delay_steps",
    "query_frame",
    "release_frame",
    "candidate_state_id",
    "release_state_id",
)


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_payload_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("checkpoint_payload_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _current_runtime_environment() -> Dict[str, str]:
    environment = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for distribution in (
        "numpy",
        "gymnasium",
        "PyYAML",
        "scipy",
        "requests",
        "highway-env",
    ):
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            version = "missing"
        environment[f"pkg_{distribution}"] = str(version)
    return environment


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not bool")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"non-integral {field}: {value!r}")
    if isinstance(value, str) and value.strip() != str(converted):
        raise ValueError(f"non-canonical {field}: {value!r}")
    return converted


def _gate_payload(record: Mapping[str, Any]) -> Dict[str, Any]:
    diagnostics = record.get("rgd_subordinate_diagnostics")
    _require(isinstance(diagnostics, Mapping), "trace omits rgd_subordinate_diagnostics")
    recoverability = diagnostics.get("recoverability_signal")
    _require(isinstance(recoverability, Mapping), "trace omits recoverability_signal")
    gate = recoverability.get("recoverability_gate")
    _require(isinstance(gate, Mapping), "trace omits recoverability_gate")
    return dict(gate)


def _exact_action_contract(record: Mapping[str, Any], *, seed: int, frame: int) -> Dict[str, Any]:
    prefix = f"seed {seed} frame {frame}"
    _require(record.get("schema_version") == "rgd_record_v3", f"{prefix}: trace schema drift")
    _require(str(record.get("system_used", "") or "") == "fast", f"{prefix}: source is not Fast-only")
    gate = _gate_payload(record)
    _require(str(gate.get("method_version", "") or "") == METHOD_VERSION, f"{prefix}: method version drift")

    raw_gate = gate.get("gate_action_universe")
    raw_fast = gate.get("fast_executor_action_universe")
    _require(isinstance(raw_gate, (list, tuple)), f"{prefix}: gate action universe is not explicit")
    _require(isinstance(raw_fast, (list, tuple)), f"{prefix}: fast action universe is not explicit")
    gate_actions = tuple(
        _strict_int(value, field=f"{prefix} gate_action_universe") for value in raw_gate
    )
    fast_actions = tuple(
        _strict_int(value, field=f"{prefix} fast_executor_action_universe")
        for value in raw_fast
    )
    _require(gate_actions, f"{prefix}: empty gate action universe")
    _require(len(set(gate_actions)) == len(gate_actions), f"{prefix}: duplicate gate actions")
    _require(len(set(fast_actions)) == len(fast_actions), f"{prefix}: duplicate fast actions")
    _require(gate_actions == tuple(sorted(gate_actions)), f"{prefix}: non-canonical gate action order")
    _require(fast_actions == tuple(sorted(fast_actions)), f"{prefix}: non-canonical fast action order")
    _require(gate_actions == fast_actions, f"{prefix}: gate/Fast action universe mismatch")
    _require(set(gate_actions).issubset(set(RAW_ACTIONS)), f"{prefix}: unknown raw action in universe")
    _require(
        str(gate.get("gate_action_universe_source", "") or "")
        == ACTION_UNIVERSE_SOURCE,
        f"{prefix}: gate action universe source drift",
    )
    _require(
        str(gate.get("fast_executor_action_universe_source", "") or "")
        == ACTION_UNIVERSE_SOURCE,
        f"{prefix}: Fast action universe source drift",
    )
    _require(gate.get("gate_domain_valid") is True, f"{prefix}: gate domain is not valid")
    _require(gate.get("gate_fail_closed") is False, f"{prefix}: gate trace is fail-closed")
    hold_action = _strict_int(gate.get("hold_action"), field=f"{prefix} hold_action")
    _require(hold_action in gate_actions, f"{prefix}: hold action lies outside exact universe")
    predicted_action = _strict_int(
        record.get("predicted_action_id"), field=f"{prefix} predicted_action_id"
    )
    _require(predicted_action in gate_actions, f"{prefix}: Fast proposal lies outside exact universe")
    _require(hold_action == predicted_action, f"{prefix}: hold/Fast proposal identity drift")
    return {
        "method_version": METHOD_VERSION,
        "gate_action_universe": gate_actions,
        "fast_executor_action_universe": fast_actions,
        "gate_action_universe_source": ACTION_UNIVERSE_SOURCE,
        "fast_executor_action_universe_source": ACTION_UNIVERSE_SOURCE,
        "hold_action": hold_action,
        "predicted_action": predicted_action,
    }


def _vehicle_state(snapshot: ReleaseSnapshot) -> Dict[str, Any]:
    env = getattr(snapshot, "env", None)
    unwrapped = getattr(env, "unwrapped", env)
    vehicle = getattr(unwrapped, "vehicle", None)
    _require(vehicle is not None, "snapshot has no ego vehicle")
    position = getattr(vehicle, "position", None)
    _require(position is not None and len(position) >= 2, "snapshot has no ego position")
    lane_index = getattr(vehicle, "lane_index", None)
    _require(lane_index is not None and len(lane_index) >= 3, "snapshot has no lane identity")
    return {
        "position_x": float(position[0]),
        "position_y": float(position[1]),
        "speed": float(getattr(vehicle, "speed")),
        "lane_id": int(lane_index[2]),
    }


def _pickle_sha256(value: Any) -> str:
    return hashlib.sha256(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _validate_v12_snapshot_policy_state(
    snapshot: ReleaseSnapshot,
    *,
    prefix: str,
) -> Tuple[Dict[str, Any], str]:
    schema = getattr(snapshot, "policy_state_schema", None)
    _require(
        schema == DRIVER_POLICY_STATE_SCHEMA,
        f"{prefix}: policy-state schema drift",
    )
    raw_state = getattr(snapshot, "policy_state", None)
    try:
        state = validate_driver_policy_state(raw_state)
    except ValueError as exc:
        raise ValueError(f"{prefix}: invalid policy state: {exc}") from exc
    recorded_sha256 = str(
        getattr(snapshot, "policy_state_sha256", "") or ""
    )
    _require(
        len(recorded_sha256) == 64
        and all(character in "0123456789abcdef" for character in recorded_sha256),
        f"{prefix}: invalid policy-state SHA256",
    )
    observed_sha256 = policy_state_sha256(state)
    _require(
        recorded_sha256 == observed_sha256,
        f"{prefix}: policy-state SHA256 mismatch",
    )
    return state, observed_sha256


def _validate_snapshot_trace_identity(
    snapshot: ReleaseSnapshot,
    record: Mapping[str, Any],
    physical: Mapping[str, Any],
    previous_physical: Optional[Mapping[str, Any]],
    *,
    seed: int,
    frame: int,
    action_contract: Mapping[str, Any],
    physical_prefix: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    prefix = f"seed {seed} frame {frame}"
    _require(_strict_int(getattr(snapshot, "frame", None), field=f"{prefix} snapshot.frame") == frame, f"{prefix}: snapshot frame drift")
    _require(_strict_int(record.get("frame_id"), field=f"{prefix} reasoning frame") == frame, f"{prefix}: reasoning frame drift")
    _require(_strict_int(physical.get("frame_id"), field=f"{prefix} physical frame") == frame, f"{prefix}: physical frame drift")
    observed = _vehicle_state(snapshot)
    expected = {
        "position_x": float(physical["position_x"]),
        "position_y": float(physical["position_y"]),
        "speed": float(physical["speed"]),
        "lane_id": _strict_int(physical.get("lane_id"), field=f"{prefix} lane_id"),
    }
    position_error = math.hypot(
        observed["position_x"] - expected["position_x"],
        observed["position_y"] - expected["position_y"],
    )
    speed_error = abs(observed["speed"] - expected["speed"])
    _require(position_error <= POSITION_TOLERANCE_M, f"{prefix}: snapshot/trace position drift {position_error:.9g} m")
    _require(speed_error <= SPEED_TOLERANCE_MPS, f"{prefix}: snapshot/trace speed drift {speed_error:.9g} m/s")
    _require(observed["lane_id"] == expected["lane_id"], f"{prefix}: snapshot/trace lane drift")
    expected_previous_action = (
        1
        if previous_physical is None
        else _strict_int(previous_physical.get("action_id"), field=f"{prefix} previous action")
    )
    snapshot_previous_action = _strict_int(
        getattr(snapshot, "previous_action", None), field=f"{prefix} snapshot.previous_action"
    )
    _require(snapshot_previous_action == expected_previous_action, f"{prefix}: previous-action drift")

    snapshot_history = list(getattr(snapshot, "history", []) or [])
    fast_state = getattr(snapshot, "fast_state", None)
    _require(isinstance(fast_state, Mapping), f"{prefix}: invalid Fast runtime state")
    _, policy_state_digest = _validate_v12_snapshot_policy_state(
        snapshot,
        prefix=prefix,
    )
    fast_action_history_raw = fast_state.get("action_history", ())
    fast_action_history = [int(value) for value in list(fast_action_history_raw or [])]
    policy_fast_history = [
        int(value)
        for value in list(
            ((getattr(snapshot, "policy_state", None) or {}).get("fast", {}) or {}).get(
                "action_history", []
            )
        )
    ]
    _require(
        policy_fast_history == fast_action_history,
        f"{prefix}: versioned/legacy Fast action-history drift",
    )
    if physical_prefix is not None:
        history_maxlen = getattr(getattr(snapshot, "history", None), "maxlen", None)
        expected_history_length = min(
            len(physical_prefix),
            len(physical_prefix) if history_maxlen is None else int(history_maxlen),
        )
        _require(
            len(snapshot_history) == expected_history_length,
            f"{prefix}: runtime history length drift",
        )
        for observed_history, source_frame in zip(
            snapshot_history,
            physical_prefix[-expected_history_length:] if expected_history_length else (),
        ):
            _require(
                int(observed_history["action"]) == int(source_frame["action_id"]),
                f"{prefix}: runtime history action drift",
            )
            for history_field, physical_field in (("speed", "speed"),):
                _require(
                    _metric_matches(
                        observed_history[history_field], source_frame[physical_field]
                    ),
                    f"{prefix}: runtime history {history_field} drift",
                )
        fast_maxlen = getattr(fast_action_history_raw, "maxlen", None)
        expected_fast_length = min(
            len(physical_prefix),
            len(physical_prefix) if fast_maxlen is None else int(fast_maxlen),
        )
        expected_fast_actions = [
            int(source_frame["action_id"])
            for source_frame in (
                physical_prefix[-expected_fast_length:] if expected_fast_length else ()
            )
        ]
        _require(
            fast_action_history == expected_fast_actions,
            f"{prefix}: Fast action history drift",
        )

    identity_payload = {
        "seed": int(seed),
        "release_frame": int(frame),
        "position_x": expected["position_x"],
        "position_y": expected["position_y"],
        "speed": expected["speed"],
        "lane_id": expected["lane_id"],
        "previous_action": expected_previous_action,
        "trace_fast_proposal": int(action_contract["predicted_action"]),
        "trace_effective_action": _strict_int(
            physical.get("action_id"), field=f"{prefix} action_id"
        ),
        "gate_action_universe": list(action_contract["gate_action_universe"]),
        "method_version": METHOD_VERSION,
        "snapshot_obs_sha256": _pickle_sha256(getattr(snapshot, "obs", None)),
        "snapshot_history_sha256": _pickle_sha256(getattr(snapshot, "history", None)),
        "snapshot_fast_state_sha256": _pickle_sha256(fast_state),
        "snapshot_policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
        "snapshot_policy_state_sha256": policy_state_digest,
    }
    return _canonical_sha256(identity_payload), {
        **identity_payload,
        "position_error_m": float(position_error),
        "speed_error_mps": float(speed_error),
    }


def _finite_effective_identity(row: Mapping[str, Any], *, seed: int, frame: int) -> Tuple[int, float]:
    identity = _effective_identity(dict(row))
    _require(math.isfinite(float(identity[1])), f"seed {seed} frame {frame}: nonfinite post-bridge target speed")
    return int(identity[0]), float(identity[1])


def _identity_text(identity: Tuple[int, float]) -> str:
    return f"{int(identity[0])}@{float(identity[1]):.6f}"


def _validate_branch_outcome(row: Mapping[str, Any], *, seed: int, frame: int) -> None:
    for field in ("normalized_return", "utility", "progress_m", "target_speed_after"):
        try:
            value = float(row[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"seed {seed} frame {frame}: invalid branch {field}"
            ) from exc
        _require(
            math.isfinite(value),
            f"seed {seed} frame {frame}: nonfinite branch {field}",
        )
    _require(
        int(row.get("collision", -1)) in (0, 1),
        f"seed {seed} frame {frame}: invalid collision label",
    )
    _require(
        _metric_matches(
            row["utility"],
            float(row["normalized_return"]) - float(int(row["collision"])),
        ),
        f"seed {seed} frame {frame}: branch utility derivation drift",
    )


def _validate_runtime_action_contract(
    row: Mapping[str, Any],
    action_contract: Mapping[str, Any],
    *,
    seed: int,
    frame: int,
) -> None:
    expected = ";".join(
        str(action) for action in action_contract["gate_action_universe"]
    )
    _require(
        str(row.get("runtime_effective_action_universe", "")) == expected,
        f"seed {seed} frame {frame}: runtime effective action universe drift",
    )
    _require(
        str(row.get("runtime_gate_action_universe", "")) == expected,
        f"seed {seed} frame {frame}: runtime gate action universe drift",
    )
    _require(
        str(row.get("runtime_gate_action_universe_source", ""))
        == ACTION_UNIVERSE_SOURCE
        and str(row.get("runtime_fast_action_universe_source", ""))
        == ACTION_UNIVERSE_SOURCE,
        f"seed {seed} frame {frame}: runtime action universe source drift",
    )


def _metric_matches(left: Any, right: Any) -> bool:
    left_value = float(left)
    right_value = float(right)
    if math.isinf(left_value) or math.isinf(right_value):
        return left_value == right_value
    return math.isclose(
        left_value, right_value, rel_tol=0.0, abs_tol=1e-12
    )


def _require_equivalent_outcomes(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    seed: int,
    frame: int,
    context: str,
) -> None:
    _require(
        _finite_effective_identity(left, seed=seed, frame=frame)
        == _finite_effective_identity(right, seed=seed, frame=frame),
        f"seed {seed} frame {frame}: {context} effective identity drift",
    )
    for field in (
        "normalized_return",
        "utility",
        "progress_m",
        "target_speed_after",
        "min_ttc",
    ):
        _require(
            _metric_matches(left[field], right[field]),
            f"seed {seed} frame {frame}: {context} {field} drift",
        )
    for field in ("collision", "steps_completed", "effective_action", "fast_action"):
        _require(
            int(left[field]) == int(right[field]),
            f"seed {seed} frame {frame}: {context} {field} drift",
        )
    _require(
        str(left.get("legal_actions", "")) == str(right.get("legal_actions", "")),
        f"seed {seed} frame {frame}: {context} legal action drift",
    )
    try:
        left_trajectory = json.loads(str(left["branch_trajectory_json"]))
        right_trajectory = json.loads(str(right["branch_trajectory_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"seed {seed} frame {frame}: {context} trajectory is invalid"
        ) from exc
    _require(
        isinstance(left_trajectory, list)
        and isinstance(right_trajectory, list)
        and len(left_trajectory) == len(right_trajectory),
        f"seed {seed} frame {frame}: {context} trajectory length drift",
    )
    for offset, (left_step, right_step) in enumerate(
        zip(left_trajectory, right_trajectory)
    ):
        for field in ("frame", "lane_id", "effective_action"):
            _require(
                int(left_step[field]) == int(right_step[field]),
                f"seed {seed} frame {frame}: {context} trajectory {field} drift "
                f"at offset {offset}",
            )
        for field in ("position_x", "position_y", "speed"):
            _require(
                _metric_matches(left_step[field], right_step[field]),
                f"seed {seed} frame {frame}: {context} trajectory {field} drift "
                f"at offset {offset}",
            )


def _validate_matched_fast_trajectory(
    baseline: Mapping[str, Any],
    source_frames: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    frame: int,
) -> None:
    try:
        trajectory = json.loads(str(baseline["branch_trajectory_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"seed {seed} frame {frame}: matched branch trajectory is invalid"
        ) from exc
    _require(
        isinstance(trajectory, list)
        and len(trajectory) == len(source_frames)
        and int(baseline["steps_completed"]) == len(source_frames),
        f"seed {seed} frame {frame}: matched branch horizon drift",
    )
    for offset, (observed, expected) in enumerate(zip(trajectory, source_frames)):
        expected_frame = frame + offset
        _require(
            int(observed.get("frame", -1)) == expected_frame
            and int(expected.get("frame_id", -1)) == expected_frame,
            f"seed {seed} frame {expected_frame}: matched trajectory frame drift",
        )
        position_error = math.hypot(
            float(observed["position_x"]) - float(expected["position_x"]),
            float(observed["position_y"]) - float(expected["position_y"]),
        )
        _require(
            position_error <= POSITION_TOLERANCE_M,
            f"seed {seed} frame {expected_frame}: matched trajectory position drift",
        )
        _require(
            abs(float(observed["speed"]) - float(expected["speed"]))
            <= SPEED_TOLERANCE_MPS,
            f"seed {seed} frame {expected_frame}: matched trajectory speed drift",
        )
        _require(
            int(observed["lane_id"]) == int(expected["lane_id"]),
            f"seed {seed} frame {expected_frame}: matched trajectory lane drift",
        )
        _require(
            int(observed["effective_action"]) == int(expected["action_id"]),
            f"seed {seed} frame {expected_frame}: matched trajectory action drift",
        )


def _derive_release_outcome(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    frame: int,
    predicted_action: int,
    epsilon: float,
) -> Dict[str, Any]:
    _validate_branch_outcome(baseline, seed=seed, frame=frame)
    for candidate in candidates:
        _validate_branch_outcome(candidate, seed=seed, frame=frame)
    predicted = [
        row for row in candidates if int(row["raw_action"]) == int(predicted_action)
    ]
    _require(
        len(predicted) == 1,
        f"seed {seed} frame {frame}: expected one predicted-action candidate",
    )
    _require_equivalent_outcomes(
        baseline,
        predicted[0],
        seed=seed,
        frame=frame,
        context="matched-Fast forced-action replay",
    )
    baseline_identity = _finite_effective_identity(
        baseline, seed=seed, frame=frame
    )
    distinct_by_effect: Dict[Tuple[int, float], Mapping[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda row: int(row["raw_action"])):
        identity = _finite_effective_identity(candidate, seed=seed, frame=frame)
        current = distinct_by_effect.get(identity)
        if current is None:
            distinct_by_effect[identity] = candidate
        else:
            _require_equivalent_outcomes(
                current,
                candidate,
                seed=seed,
                frame=frame,
                context=f"raw-action aliases for {_identity_text(identity)}",
            )
    alternatives = [
        row
        for identity, row in distinct_by_effect.items()
        if identity != baseline_identity
    ]
    corrective_rows = [
        row
        for row in alternatives
        if float(row["utility"]) - float(baseline["utility"]) >= float(epsilon)
    ]
    best_row = max(
        alternatives, key=lambda row: float(row["utility"]), default=None
    )
    best_advantage = (
        float(best_row["utility"]) - float(baseline["utility"])
        if best_row is not None
        else float("-inf")
    )
    representative_by_identity = {
        identity: int(row["raw_action"])
        for identity, row in distinct_by_effect.items()
    }
    corrective_identities = {
        _finite_effective_identity(row, seed=seed, frame=frame)
        for row in corrective_rows
    }
    return {
        "baseline_identity": baseline_identity,
        "distinct_by_effect": distinct_by_effect,
        "alternatives": alternatives,
        "corrective_rows": corrective_rows,
        "best_row": best_row,
        "best_advantage": best_advantage,
        "representative_by_identity": representative_by_identity,
        "corrective_identities": corrective_identities,
    }


def _evaluate_release_state(
    snapshot: ReleaseSnapshot,
    cfg: Dict[str, Any],
    *,
    seed: int,
    frame: int,
    record: Mapping[str, Any],
    physical: Mapping[str, Any],
    state_identity_sha256: str,
    action_contract: Mapping[str, Any],
    horizon: int,
    gamma: float,
    epsilon: float,
    source_continuation: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    baseline = _run_branch(snapshot, cfg, seed, None, horizon, gamma)
    _require(int(baseline["release_frame"]) == frame, f"seed {seed} frame {frame}: baseline release frame drift")
    _require(int(baseline["fast_action"]) == int(action_contract["predicted_action"]), f"seed {seed} frame {frame}: matched Fast proposal drift")
    trace_effective_action = _strict_int(
        physical.get("action_id"), field=f"seed {seed} frame {frame} physical action"
    )
    _require(int(baseline["effective_action"]) == trace_effective_action, f"seed {seed} frame {frame}: matched effective action drift")
    _require(int(baseline["steps_completed"]) > 0, f"seed {seed} frame {frame}: empty baseline branch")
    _validate_branch_outcome(baseline, seed=seed, frame=frame)
    _validate_runtime_action_contract(
        baseline, action_contract, seed=seed, frame=frame
    )
    if source_continuation is not None:
        _validate_matched_fast_trajectory(
            baseline, source_continuation, seed=seed, frame=frame
        )

    candidates = [
        _run_branch(snapshot, cfg, seed, int(action), horizon, gamma)
        for action in action_contract["gate_action_universe"]
    ]
    for action, candidate in zip(action_contract["gate_action_universe"], candidates):
        _require(int(candidate["release_frame"]) == frame, f"seed {seed} frame {frame}: candidate release frame drift")
        _require(int(candidate["raw_action"]) == int(action), f"seed {seed} frame {frame}: candidate raw action drift")
        _require(int(candidate["fast_action"]) == int(action_contract["predicted_action"]), f"seed {seed} frame {frame}: candidate Fast proposal drift")
        _require(str(candidate.get("legal_actions", "")) == str(baseline.get("legal_actions", "")), f"seed {seed} frame {frame}: candidate legal action drift")
        _require(int(candidate["steps_completed"]) > 0, f"seed {seed} frame {frame}: empty candidate branch")
        _validate_branch_outcome(candidate, seed=seed, frame=frame)
        _validate_runtime_action_contract(
            candidate, action_contract, seed=seed, frame=frame
        )

    derived = _derive_release_outcome(
        baseline,
        candidates,
        seed=seed,
        frame=frame,
        predicted_action=int(action_contract["predicted_action"]),
        epsilon=epsilon,
    )
    baseline_identity = derived["baseline_identity"]
    distinct_by_effect = derived["distinct_by_effect"]
    alternatives = derived["alternatives"]
    corrective_rows = derived["corrective_rows"]
    best_row = derived["best_row"]
    best_advantage = derived["best_advantage"]
    representative_by_identity = derived["representative_by_identity"]
    corrective_identities = derived["corrective_identities"]
    universe_text = ";".join(str(value) for value in action_contract["gate_action_universe"])
    common = {
        "method_version": METHOD_VERSION,
        "continuation_contract_version": CONTINUATION_CONTRACT_VERSION,
        "release_state_id": f"{seed}:{frame}",
        "release_state_identity_sha256": state_identity_sha256,
        "gate_action_universe": universe_text,
        "gate_action_universe_source": action_contract["gate_action_universe_source"],
        "fast_executor_action_universe": universe_text,
        "fast_executor_action_universe_source": action_contract[
            "fast_executor_action_universe_source"
        ],
        "exact_action_provenance": 1,
    }
    branch_rows: List[Dict[str, Any]] = []
    branch_rows.append(
        {
            **baseline,
            **common,
            "branch_role": "matched_fast",
            "raw_action_provenance": "complete_deterministic_fast_controller",
            "effective_identity": _identity_text(baseline_identity),
            "matches_matched_fast_identity": 1,
            "effective_identity_representative": 1,
            "effective_identity_representative_raw_action": "fast",
            "advantage_over_matched_fast": 0.0,
            "in_corrective_set": 0,
        }
    )
    for candidate in candidates:
        identity = _finite_effective_identity(candidate, seed=seed, frame=frame)
        representative = representative_by_identity[identity]
        advantage = float(candidate["utility"]) - float(baseline["utility"])
        branch_rows.append(
            {
                **candidate,
                **common,
                "branch_role": "candidate",
                "raw_action_provenance": "exact_trace_gate_action_universe",
                "effective_identity": _identity_text(identity),
                "matches_matched_fast_identity": int(identity == baseline_identity),
                "effective_identity_representative": int(
                    int(candidate["raw_action"]) == representative
                ),
                "effective_identity_representative_raw_action": int(representative),
                "advantage_over_matched_fast": float(advantage),
                "in_corrective_set": int(
                    identity in corrective_identities
                    and int(candidate["raw_action"]) == representative
                ),
            }
        )

    same_as_baseline_count = sum(
        int(_finite_effective_identity(row, seed=seed, frame=frame) == baseline_identity)
        for row in candidates
    )
    label = {
        "seed": int(seed),
        "release_frame": int(frame),
        **common,
        "snapshot_trace_identity_matched": 1,
        "matched_fast_trace_identity_matched": 1,
        "gate_action_count": len(action_contract["gate_action_universe"]),
        "candidate_branch_count": len(candidates),
        "distinct_effective_candidate_count": len(distinct_by_effect),
        "distinct_effective_alternative_count": len(alternatives),
        "candidate_actions_matching_fast_identity": same_as_baseline_count,
        "candidate_effective_aliases_collapsed": len(candidates) - len(distinct_by_effect),
        "corrective_set_action_count": len(corrective_rows),
        "corrective_set_nonempty": int(bool(corrective_rows)),
        "best_advantage": best_advantage if math.isfinite(best_advantage) else "",
        "baseline_utility": float(baseline["utility"]),
        "baseline_collision": int(baseline["collision"]),
        "baseline_effective_action": int(baseline["effective_action"]),
        "baseline_target_speed_after": float(baseline["target_speed_after"]),
        "best_raw_action": "" if best_row is None else int(best_row["raw_action"]),
        "best_effective_action": "" if best_row is None else int(best_row["effective_action"]),
        "best_target_speed_after": "" if best_row is None else float(best_row["target_speed_after"]),
        "best_collision": "" if best_row is None else int(best_row["collision"]),
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
        "utility_definition": "discounted normalized simulator return minus collision indicator",
        "corrective_set_definition": "distinct post-safety/post-bridge action identities with utility advantage >= epsilon",
    }
    return {"branches": branch_rows, "label": label}


def _source_paths(trace_root: Path, seed: int) -> Dict[str, Path]:
    reasoning_path, physical_path = _trace_paths(trace_root, seed)
    seed_dir = reasoning_path.parent.parent
    paths = {
        "reasoning": reasoning_path,
        "physical": physical_path,
        "snapshot_bundle": seed_dir / "snapshots.pkl",
        "experiment_snapshot": seed_dir / "experiment_snapshot.json",
        "runtime_manifest": seed_dir / "runtime_manifest.json",
    }
    for label, path in paths.items():
        _require(path.is_file(), f"seed {seed}: missing {label}: {path}")
    return paths


def _source_hashes(paths: Mapping[str, Path]) -> Dict[str, Dict[str, str]]:
    return {
        label: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for label, path in sorted(paths.items())
    }


def _validate_source_provenance(
    paths: Mapping[str, Path],
    seed: int,
    expected_runtime_source_sha256: str,
    expected_protocol_sha256: str,
    expected_protocol_path: Path,
) -> Dict[str, Any]:
    experiment = json.loads(paths["experiment_snapshot"].read_text(encoding="utf-8-sig"))
    manifest = json.loads(paths["runtime_manifest"].read_text(encoding="utf-8-sig"))
    _require(isinstance(experiment, Mapping) and isinstance(manifest, Mapping), f"seed {seed}: invalid provenance JSON")
    _require(_strict_int(experiment.get("fixed_seed_override"), field=f"seed {seed} fixed_seed_override") == seed, f"seed {seed}: experiment seed drift")
    _require([int(value) for value in experiment.get("seeds_used", [])] == [seed], f"seed {seed}: experiment seed block drift")
    for field in ("protocol_id", "protocol_hash", "config_hash", "source_hash"):
        _require(str(experiment.get(field, "") or "") == str(manifest.get(field, "") or ""), f"seed {seed}: experiment/manifest {field} drift")
        digest = str(experiment.get(field, "") or "")
        if field != "protocol_id":
            _require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"seed {seed}: invalid source {field}",
            )
    _require(
        str(experiment.get("source_hash", "") or "")
        == str(expected_runtime_source_sha256),
        f"seed {seed}: snapshot/current runtime source drift",
    )
    experiment_acquisition = dict(experiment.get("snapshot_acquisition", {}) or {})
    manifest_acquisition = dict(manifest.get("snapshot_acquisition", {}) or {})
    _require(
        experiment_acquisition == manifest_acquisition,
        f"seed {seed}: experiment/manifest acquisition provenance drift",
    )
    expected_acquisition = {
        "schema_version": 2,
        "policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
        "policy_state_integrity": "canonical_json_sha256",
        "producer_path": str(SNAPSHOT_PRODUCER_PATH.resolve()),
        "producer_sha256": _sha256(SNAPSHOT_PRODUCER_PATH),
        "base_config_path": str(BASE_CONFIG_PATH.resolve()),
        "base_config_sha256": _sha256(BASE_CONFIG_PATH),
        "protocol_path": str(expected_protocol_path.resolve()),
        "protocol_sha256": str(expected_protocol_sha256),
    }
    for field in (
        "schema_version",
        "policy_state_schema",
        "policy_state_integrity",
        "producer_path",
        "producer_sha256",
        "base_config_path",
        "base_config_sha256",
        "protocol_path",
        "protocol_sha256",
    ):
        _require(
            experiment_acquisition.get(field) == expected_acquisition[field],
            f"seed {seed}: acquisition provenance drift at {field}",
        )
    expected_artifact_hashes = {
        label: {
            "path": str(Path(paths[label]).resolve()),
            "sha256": _sha256(Path(paths[label])),
        }
        for label in ("reasoning", "physical", "snapshot_bundle")
    }
    _require(
        dict(experiment_acquisition.get("artifact_hashes", {}) or {})
        == expected_artifact_hashes,
        f"seed {seed}: acquisition artifact hash drift",
    )
    experiment_cfg = dict(experiment.get("config", {}) or {})
    manifest_cfg = dict(manifest.get("config", {}) or {})
    _require(experiment_cfg == manifest_cfg, f"seed {seed}: experiment/manifest config drift")
    computed_config_hash = hashlib.sha256(
        json.dumps(
            experiment_cfg,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    _require(
        computed_config_hash == str(experiment.get("config_hash", "") or ""),
        f"seed {seed}: source config hash is not derivable",
    )
    _require(
        str(experiment.get("protocol_id", "") or "").endswith(
            str(experiment.get("protocol_hash", "") or "")[:16]
        ),
        f"seed {seed}: source protocol id/hash drift",
    )
    experiment_environment = dict(experiment.get("runtime_environment", {}) or {})
    manifest_environment = dict(manifest.get("runtime_environment", {}) or {})
    _require(
        experiment_environment == manifest_environment,
        f"seed {seed}: experiment/manifest runtime environment drift",
    )
    _require(
        REQUIRED_SOURCE_ENVIRONMENT_FIELDS.issubset(experiment_environment),
        f"seed {seed}: source runtime environment is incomplete",
    )
    current_environment = _current_runtime_environment()
    for field, source_value in experiment_environment.items():
        _require(
            str(current_environment.get(field, "missing")) == str(source_value),
            f"seed {seed}: runtime environment drift at {field}",
        )
    _require(str(experiment_cfg.get("env_type", "") or "") == "highway-v0", f"seed {seed}: source environment drift")
    _require(str(experiment_cfg.get("scenario_type", "") or "") == "highway", f"seed {seed}: source scenario drift")
    _require(str(experiment_cfg.get("protocol_name", "") or "") == "always_fast", f"seed {seed}: source protocol is not always_fast")
    routing = dict(experiment_cfg.get("system_routing", {}) or {})
    _require(
        str(routing.get("simple", "") or "") == "fast"
        and str(routing.get("complex", "") or "") == "fast",
        f"seed {seed}: source routing is not Fast-only",
    )
    replay = dict(experiment_cfg.get("closed_loop_latency_replay", {}) or {})
    if replay.get("enable") is True:
        targets = sorted(str(value) for value in replay.get("target_systems", []) or [])
        _require(
            targets == ["slow"],
            f"seed {seed}: enabled source replay is not restricted to the inactive slow route",
        )
    else:
        _require(replay.get("enable") is False, f"seed {seed}: invalid replay enable flag")
    return experiment_cfg


def _validate_branch_config(cfg: Mapping[str, Any], source_cfg: Mapping[str, Any], seed: int) -> None:
    embedded_runtime_config = _build_runtime_experiment_config(dict(cfg))
    _require(
        embedded_runtime_config == dict(source_cfg),
        f"seed {seed}: reconstructed protocol runtime config differs from source snapshot",
    )
    consumed_fields = (
        "env_type",
        "scenario_type",
        "simulation_duration",
        "policy_frequency",
        "simulation_frequency",
        "vehicle_count",
        "vehicles_density",
        "ego_spacing",
        "highway_v0_env",
        "initial_speed",
        "lanes_count",
        "target_speed_min",
        "target_speed_max",
        "target_speed_count",
        "hidden_slower_bridge",
        "enable_safety_guard",
        "safety_thresholds",
        "rss_params",
        "lane_change_cooldown",
        "system_routing",
        "rgd_min_observation_frames",
    )
    for field in consumed_fields:
        branch_value = cfg.get(field)
        source_value = source_cfg.get(field)
        if field == "highway_v0_env":
            branch_value = dict(branch_value or {})
            source_value = dict(source_value or {})
        _require(
            branch_value == source_value,
            f"seed {seed}: branch-consumed config drift at {field}",
        )
    _require(
        int(cfg.get("fixed_seed_override", -1)) == int(seed),
        f"seed {seed}: branch seed override drift",
    )
    branch_replay = dict(cfg.get("closed_loop_latency_replay", {}) or {})
    _require(
        branch_replay.get("enable") is False,
        f"seed {seed}: branch continuation did not disable latency replay",
    )


def _load_snapshots(path: Path, targets: Sequence[int], seed: int) -> Dict[int, ReleaseSnapshot]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    _require(isinstance(payload, Mapping), f"seed {seed}: snapshot bundle is not a mapping")
    normalized: Dict[int, ReleaseSnapshot] = {}
    for key, value in payload.items():
        frame = _strict_int(key, field=f"seed {seed} snapshot key")
        _require(frame not in normalized, f"seed {seed}: duplicate normalized snapshot key {frame}")
        _validate_v12_snapshot_policy_state(
            value,
            prefix=f"seed {seed} frame {frame}",
        )
        normalized[frame] = value
    missing = sorted(set(targets) - set(normalized))
    _require(not missing, f"seed {seed}: snapshot bundle misses target frames {missing}")
    return {frame: normalized[frame] for frame in targets}


def _process_seed(
    seed: int,
    requested_targets: Sequence[int],
    trace_root: str,
    protocol_path: str,
    scratch_root: str,
    horizon: int,
    gamma: float,
    epsilon: float,
    expected_runtime_source_sha256: str,
    expected_protocol_sha256: str,
    floor_overlay_path: Optional[str] = None,
    calibration_manifest_path: Optional[str] = None,
    calibration_lock_path: Optional[str] = None,
    v12_partition: Optional[str] = None,
) -> Dict[str, Any]:
    verified_floor_overlay = load_optional_verified_floor_overlay(
        None if floor_overlay_path is None else Path(floor_overlay_path),
        calibration_manifest_path=(
            None
            if calibration_manifest_path is None
            else Path(calibration_manifest_path)
        ),
        protocol_path=Path(protocol_path),
        lock_path=(
            DEFAULT_LOCK_PATH
            if calibration_lock_path is None
            else Path(calibration_lock_path)
        ),
    )
    if v12_partition is not None:
        protocol = load_formal_protocol(Path(protocol_path))
        observed_partition = enforce_v12_floor_overlay_contract(
            str(protocol.get("protocol_name", "") or ""),
            [int(seed)],
            verified_floor_overlay,
        )
        _require(
            observed_partition == v12_partition,
            f"seed {seed}: worker v12 partition differs from the run contract",
        )
    unique_targets = sorted(set(int(frame) for frame in requested_targets))
    root = Path(trace_root)
    paths = _source_paths(root, seed)
    hashes_before = _source_hashes(paths)
    source_cfg = _validate_source_provenance(
        paths,
        seed,
        expected_runtime_source_sha256,
        expected_protocol_sha256,
        Path(protocol_path),
    )
    records, physical_frames = _load_trace(root, seed)
    _require(len(records) == len(physical_frames), f"seed {seed}: trace length drift")
    _require(records, f"seed {seed}: source trace is empty")
    _require(
        all(str(record.get("system_used", "") or "") == "fast" for record in records),
        f"seed {seed}: source trace contains a non-Fast frame",
    )
    for frame in unique_targets:
        _require(frame >= 0, f"seed {seed}: negative release frame {frame}")
        _require(frame < len(records), f"seed {seed}: release frame {frame} exceeds trace")
        _require(frame + int(horizon) <= len(records), f"seed {seed} frame {frame}: insufficient full source horizon")

    snapshots = _load_snapshots(paths["snapshot_bundle"], unique_targets, seed)
    scratch = Path(scratch_root) / f"seed_{seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg = _build_fast_config(
        Path(protocol_path),
        seed,
        scratch,
        verified_floor_overlay=verified_floor_overlay,
    )
    if verified_floor_overlay is not None:
        assert_floor_overlay_applied(cfg, verified_floor_overlay)
    _validate_branch_config(cfg, source_cfg, seed)

    branches: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    max_position_error = 0.0
    max_speed_error = 0.0
    with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink):
        for frame in unique_targets:
            record = records[frame]
            physical = physical_frames[frame]
            action_contract = _exact_action_contract(record, seed=seed, frame=frame)
            state_identity, identity_payload = _validate_snapshot_trace_identity(
                snapshots[frame],
                record,
                physical,
                None if frame == 0 else physical_frames[frame - 1],
                seed=seed,
                frame=frame,
                action_contract=action_contract,
                physical_prefix=physical_frames[:frame],
            )
            max_position_error = max(max_position_error, float(identity_payload["position_error_m"]))
            max_speed_error = max(max_speed_error, float(identity_payload["speed_error_mps"]))
            evaluated = _evaluate_release_state(
                snapshots[frame],
                cfg,
                seed=seed,
                frame=frame,
                record=record,
                physical=physical,
                state_identity_sha256=state_identity,
                action_contract=action_contract,
                horizon=horizon,
                gamma=gamma,
                epsilon=epsilon,
                source_continuation=physical_frames[frame : frame + horizon],
            )
            branches.extend(evaluated["branches"])
            labels.append(evaluated["label"])

    hashes_after = _source_hashes(paths)
    _require(hashes_after == hashes_before, f"seed {seed}: source artifacts changed during branch simulation")
    accounting = {
        "seed": int(seed),
        "status": "complete",
        "requested_target_entries": len(requested_targets),
        "unique_release_states": len(unique_targets),
        "duplicate_targets_excluded": len(requested_targets) - len(unique_targets),
        "release_states_evaluated": len(labels),
        "release_states_excluded": 0,
        "release_state_errors": 0,
        "branch_rows": len(branches),
        "gate_candidate_branches": sum(int(row["gate_action_count"]) for row in labels),
        "candidate_actions_matching_fast_identity": sum(int(row["candidate_actions_matching_fast_identity"]) for row in labels),
        "candidate_effective_aliases_collapsed": sum(int(row["candidate_effective_aliases_collapsed"]) for row in labels),
        "corrective_set_actions": sum(int(row["corrective_set_action_count"]) for row in labels),
        "corrective_release_states": sum(int(row["corrective_set_nonempty"]) for row in labels),
        "max_snapshot_position_error_m": float(max_position_error),
        "max_snapshot_speed_error_mps": float(max_speed_error),
        "error_type": "",
        "error_message": "",
    }
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "continuation_contract_version": CONTINUATION_CONTRACT_VERSION,
        "seed": int(seed),
        "target_frames": unique_targets,
        "source_artifacts": hashes_before,
        "source_execution_contract": {
            "all_trace_frames_fast": True,
            "source_latency_replay_enabled": bool(
                (source_cfg.get("closed_loop_latency_replay", {}) or {}).get(
                    "enable", False
                )
            ),
            "source_latency_replay_target_systems": sorted(
                str(value)
                for value in (
                    (source_cfg.get("closed_loop_latency_replay", {}) or {}).get(
                        "target_systems", []
                    )
                    or []
                )
            ),
            "branch_latency_replay_enabled": False,
            "equivalence": (
                "source replay is inert because it targets only slow while every "
                "source and branch frame uses Fast"
            ),
        },
        "branches": branches,
        "labels": labels,
        "accounting": accounting,
    }


def _exception_result(seed: int, exc: BaseException) -> Dict[str, Any]:
    return {
        "status": "error",
        "seed": int(seed),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _worker(*args: Any) -> Dict[str, Any]:
    seed = int(args[0])
    try:
        return {"status": "ok", "payload": _process_seed(*args)}
    except Exception as exc:  # Return errors so the parent can persist fail-closed accounting.
        return _exception_result(seed, exc)


def _load_target_map(path: Path, seeds: Sequence[int]) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    raw_payload = path.read_bytes()
    payload = json.loads(raw_payload.decode("utf-8-sig"))
    _require(isinstance(payload, Mapping), "snapshot-targets must be a seed-to-frame mapping")
    normalized: Dict[int, List[int]] = {}
    for raw_seed, raw_frames in payload.items():
        seed = _strict_int(raw_seed, field="snapshot-target seed")
        _require(seed not in normalized, f"duplicate normalized target seed {seed}")
        _require(isinstance(raw_frames, list), f"seed {seed}: target frames must be a JSON list")
        frames = [
            _strict_int(value, field=f"seed {seed} release frame") for value in raw_frames
        ]
        _require(all(frame >= 0 for frame in frames), f"seed {seed}: negative release frame")
        normalized[seed] = frames
    _require(set(normalized) == set(seeds), "snapshot-target seed keys differ from the requested seed block")
    _require(any(normalized.values()), "snapshot-target map contains no release states")
    metadata = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "requested_entries": sum(len(values) for values in normalized.values()),
        "unique_release_states": sum(len(set(values)) for values in normalized.values()),
        "duplicate_entries": sum(len(values) - len(set(values)) for values in normalized.values()),
    }
    return normalized, metadata


def _load_target_events(
    path: Path,
    *,
    seeds: Sequence[int],
    targets: Mapping[int, Sequence[int]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_payload = path.read_bytes()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(tuple(reader.fieldnames or ()) == TARGET_EVENT_FIELDS, "target-event columns drift")
        rows = list(reader)
    _require(rows, "target-event table is empty")
    allowed_seeds = set(int(seed) for seed in seeds)
    normalized: List[Dict[str, Any]] = []
    keys: set[Tuple[int, int, int, int]] = set()
    for index, row in enumerate(rows):
        seed = _strict_int(row["seed"], field=f"target event {index} seed")
        delay_steps = _strict_int(
            row["delay_steps"], field=f"target event {index} delay_steps"
        )
        query_frame = _strict_int(
            row["query_frame"], field=f"target event {index} query_frame"
        )
        release_frame = _strict_int(
            row["release_frame"], field=f"target event {index} release_frame"
        )
        delay_s = float(row["delay_s"])
        _require(seed in allowed_seeds, f"target event {index}: seed is outside requested block")
        _require(
            delay_steps >= 0
            and query_frame >= 0
            and release_frame >= 0
            and release_frame - query_frame == delay_steps,
            f"target event {index}: release/query/delay mismatch",
        )
        _require(
            math.isfinite(delay_s) and delay_s >= 0.0,
            f"target event {index}: invalid delay_s",
        )
        _require(
            row["candidate_state_id"] == f"{seed}:{query_frame}:{delay_steps}",
            f"target event {index}: candidate identity drift",
        )
        _require(
            row["release_state_id"] == f"{seed}:{release_frame}",
            f"target event {index}: release identity drift",
        )
        key = (seed, delay_steps, query_frame, release_frame)
        _require(key not in keys, f"duplicate target event {key}")
        keys.add(key)
        normalized.append(
            {
                "seed": seed,
                "delay_s": delay_s,
                "delay_steps": delay_steps,
                "query_frame": query_frame,
                "release_frame": release_frame,
                "candidate_state_id": row["candidate_state_id"],
                "release_state_id": row["release_state_id"],
            }
        )
    _require(
        normalized
        == sorted(
            normalized,
            key=lambda row: (
                row["seed"],
                row["delay_steps"],
                row["query_frame"],
                row["release_frame"],
            ),
        ),
        "target events are not canonically ordered",
    )
    releases = {
        seed: sorted(
            {
                int(row["release_frame"])
                for row in normalized
                if int(row["seed"]) == seed
            }
        )
        for seed in seeds
    }
    _require(
        releases == {seed: sorted(set(int(value) for value in targets[seed])) for seed in seeds},
        "target map and target events describe different release states",
    )
    return normalized, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "semantic_sha256": _canonical_sha256(normalized),
        "row_count": len(normalized),
    }


def _load_target_bundle(
    target_map_path: Path,
    target_events_path: Path,
    target_manifest_path: Path,
    seeds: Sequence[int],
) -> Tuple[Dict[int, List[int]], List[Dict[str, Any]], Dict[str, Any]]:
    targets, map_metadata = _load_target_map(target_map_path, seeds)
    _require(
        all(frames == sorted(set(frames)) for frames in targets.values()),
        "formal snapshot-target frames must be unique and ordered",
    )
    events, event_metadata = _load_target_events(
        target_events_path,
        seeds=seeds,
        targets=targets,
    )
    manifest = json.loads(target_manifest_path.read_text(encoding="utf-8-sig"))
    _require(isinstance(manifest, Mapping), "target manifest must be a JSON object")
    schema = str(manifest.get("schema", "") or "")
    _require(
        schema
        in {
            "identifiable_gate_v12_snapshot_targets_v2",
            "identifiable_gate_v12_locked_snapshot_targets_v2",
        },
        "target manifest schema drift",
    )
    if schema == "identifiable_gate_v12_snapshot_targets_v2":
        map_contract = dict(manifest.get("target_map", {}) or {})
        event_contract = dict(manifest.get("target_events", {}) or {})
        seed_block = dict(manifest.get("seed_block", {}) or {})
        _require(tuple(seed_block.get("seeds", []) or []) == tuple(seeds), "target manifest seed block drift")
        _require(map_contract.get("filename") == target_map_path.name, "target manifest map filename drift")
        _require(event_contract.get("filename") == target_events_path.name, "target manifest event filename drift")
        _require(map_contract.get("sha256") == map_metadata["sha256"], "target manifest map hash drift")
        _require(event_contract.get("sha256") == event_metadata["sha256"], "target manifest event hash drift")
        _require(
            map_contract.get("semantic_sha256")
            == _canonical_sha256({str(seed): targets[seed] for seed in seeds}),
            "target manifest map semantic hash drift",
        )
        _require(event_contract.get("semantic_sha256") == event_metadata["semantic_sha256"], "target manifest event semantic hash drift")
        _require(int(event_contract.get("row_count", -1)) == len(events), "target manifest event count drift")
        manifest_payload = dict(manifest)
        observed_payload_hash = manifest_payload.pop("manifest_payload_sha256", None)
        _require(observed_payload_hash == _canonical_sha256(manifest_payload), "target manifest payload hash drift")
    else:
        _require(tuple(manifest.get("seed_block", []) or []) == tuple(seeds), "locked target manifest seed block drift")
        _require(
            manifest.get("target_map_sha256") == map_metadata["sha256"],
            "locked target manifest map hash drift",
        )
        _require(
            manifest.get("target_events_sha256") == event_metadata["sha256"],
            "locked target manifest event hash drift",
        )
        _require(
            manifest.get("target_map_semantic_hash")
            == _canonical_sha256({str(seed): targets[seed] for seed in seeds}),
            "locked target manifest map semantic hash drift",
        )
        _require(
            manifest.get("target_event_semantic_hash") == event_metadata["semantic_sha256"],
            "locked target manifest event semantic hash drift",
        )
        _require(int(manifest.get("event_count", -1)) == len(events), "locked target manifest event count drift")
        manifest_payload = dict(manifest)
        observed_payload_hash = manifest_payload.pop("manifest_payload_hash", None)
        _require(observed_payload_hash == _canonical_sha256(manifest_payload), "locked target manifest payload hash drift")
    metadata = {
        "target_map": {
            **map_metadata,
            "semantic_sha256": _canonical_sha256(
                {str(seed): targets[seed] for seed in seeds}
            ),
        },
        "target_events": event_metadata,
        "target_manifest": {
            "path": str(target_manifest_path.resolve()),
            "sha256": _sha256(target_manifest_path),
            "semantic_sha256": _canonical_sha256(manifest),
            "schema": schema,
        },
        "requested_target_events": len(events),
        "unique_release_states": sum(len(frames) for frames in targets.values()),
    }
    return targets, events, metadata


def _expand_labels_to_target_events(
    release_labels: Sequence[Mapping[str, Any]],
    target_events: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_release: Dict[Tuple[int, int], Mapping[str, Any]] = {}
    for row in release_labels:
        key = (int(row["seed"]), int(row["release_frame"]))
        _require(key not in by_release, f"duplicate simulated release label {key}")
        by_release[key] = row
    expanded: List[Dict[str, Any]] = []
    for event in target_events:
        key = (int(event["seed"]), int(event["release_frame"]))
        _require(key in by_release, f"target event has no simulated release label {key}")
        label = by_release[key]
        _require(label.get("release_state_id") == event["release_state_id"], "release label/event identity drift")
        expanded.append({**dict(label), **dict(event), "label_source": LABEL_SOURCE})
    _require(set(by_release) == {(int(row["seed"]), int(row["release_frame"])) for row in target_events}, "simulated labels contain a non-target release state")
    return expanded


def _checkpoint_path(output_dir: Path, seed: int) -> Path:
    return output_dir / CHECKPOINT_DIR / f"seed_{seed}.json"


def _validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    seed: int,
    targets: Sequence[int],
    contract_fingerprint: str,
    trace_root: Path,
    horizon: int,
    gamma: float,
    epsilon: float,
) -> Dict[str, Any]:
    _require(
        str(payload.get("checkpoint_payload_sha256", "") or "")
        == _checkpoint_payload_sha256(payload),
        f"seed {seed}: checkpoint payload hash drift",
    )
    _require(int(payload.get("schema_version", -1)) == CHECKPOINT_SCHEMA_VERSION, f"seed {seed}: checkpoint schema drift")
    _require(str(payload.get("method_version", "") or "") == METHOD_VERSION, f"seed {seed}: checkpoint method drift")
    _require(
        str(payload.get("continuation_contract_version", "") or "")
        == CONTINUATION_CONTRACT_VERSION,
        f"seed {seed}: checkpoint continuation contract drift",
    )
    _require(int(payload.get("seed", -1)) == seed, f"seed {seed}: checkpoint seed drift")
    _require(str(payload.get("contract_fingerprint", "") or "") == contract_fingerprint, f"seed {seed}: checkpoint contract drift")
    unique_targets = sorted(set(int(value) for value in targets))
    _require([int(value) for value in payload.get("target_frames", [])] == unique_targets, f"seed {seed}: checkpoint target drift")
    current_hashes = _source_hashes(_source_paths(trace_root, seed))
    _require(dict(payload.get("source_artifacts", {}) or {}) == current_hashes, f"seed {seed}: checkpoint source drift")
    execution = dict(payload.get("source_execution_contract", {}) or {})
    _require(execution.get("all_trace_frames_fast") is True, f"seed {seed}: checkpoint source route drift")
    _require(execution.get("branch_latency_replay_enabled") is False, f"seed {seed}: checkpoint branch replay drift")
    if execution.get("source_latency_replay_enabled") is True:
        _require(
            list(execution.get("source_latency_replay_target_systems", []) or [])
            == ["slow"],
            f"seed {seed}: checkpoint source replay target drift",
        )
    else:
        _require(
            execution.get("source_latency_replay_enabled") is False,
            f"seed {seed}: checkpoint source replay flag drift",
        )
    labels = list(payload.get("labels", []) or [])
    branches = list(payload.get("branches", []) or [])
    _require([int(row["release_frame"]) for row in labels] == unique_targets, f"seed {seed}: checkpoint labels drift")
    branch_groups: Dict[int, List[Mapping[str, Any]]] = {}
    for row in branches:
        branch_groups.setdefault(int(row["release_frame"]), []).append(row)
    _require(set(branch_groups) == set(unique_targets), f"seed {seed}: checkpoint branch frames drift")
    for label in labels:
        frame = int(label["release_frame"])
        identity_sha = str(label.get("release_state_identity_sha256", "") or "")
        _require(
            len(identity_sha) == 64
            and all(character in "0123456789abcdef" for character in identity_sha),
            f"seed {seed} frame {frame}: checkpoint state identity drift",
        )
        _require(int(label.get("seed", -1)) == seed, f"seed {seed} frame {frame}: checkpoint label seed drift")
        _require(str(label.get("method_version", "") or "") == METHOD_VERSION, f"seed {seed} frame {frame}: checkpoint label method drift")
        _require(
            str(label.get("continuation_contract_version", "") or "")
            == CONTINUATION_CONTRACT_VERSION,
            f"seed {seed} frame {frame}: checkpoint label continuation drift",
        )
        _require(str(label.get("release_state_id", "")) == f"{seed}:{frame}", f"seed {seed} frame {frame}: checkpoint label state id drift")
        _require(int(label.get("exact_action_provenance", 0)) == 1, f"seed {seed} frame {frame}: checkpoint exact provenance drift")
        universe = tuple(
            int(value) for value in str(label["gate_action_universe"]).split(";") if value != ""
        )
        _require(
            universe
            and universe == tuple(sorted(set(universe)))
            and set(universe).issubset(set(RAW_ACTIONS)),
            f"seed {seed} frame {frame}: checkpoint label universe drift",
        )
        _require(
            str(label.get("fast_executor_action_universe", ""))
            == str(label.get("gate_action_universe", "")),
            f"seed {seed} frame {frame}: checkpoint Fast universe drift",
        )
        _require(
            str(label.get("gate_action_universe_source", ""))
            == ACTION_UNIVERSE_SOURCE
            and str(label.get("fast_executor_action_universe_source", ""))
            == ACTION_UNIVERSE_SOURCE,
            f"seed {seed} frame {frame}: checkpoint action source drift",
        )
        rows = branch_groups[frame]
        _require(
            len(rows) == 1 + len(universe),
            f"seed {seed} frame {frame}: checkpoint branch count drift",
        )
        _require(
            all(str(row.get("branch_role", "")) in {"matched_fast", "candidate"} for row in rows),
            f"seed {seed} frame {frame}: checkpoint unknown branch role",
        )
        baselines = [row for row in rows if str(row["branch_role"]) == "matched_fast"]
        _require(len(baselines) == 1, f"seed {seed} frame {frame}: checkpoint baseline drift")
        baseline = baselines[0]
        _require(str(baseline.get("raw_action", "")) == "fast", f"seed {seed} frame {frame}: checkpoint baseline raw action drift")
        candidate_actions = sorted(
            int(row["raw_action"]) for row in rows if str(row["branch_role"]) == "candidate"
        )
        _require(candidate_actions == list(universe), f"seed {seed} frame {frame}: checkpoint action universe drift")
        for row in rows:
            _require(int(row.get("seed", -1)) == seed, f"seed {seed} frame {frame}: checkpoint branch seed drift")
            _require(str(row.get("method_version", "") or "") == METHOD_VERSION, f"seed {seed} frame {frame}: checkpoint branch method drift")
            _require(
                str(row.get("continuation_contract_version", "") or "")
                == CONTINUATION_CONTRACT_VERSION,
                f"seed {seed} frame {frame}: checkpoint branch continuation drift",
            )
            _require(str(row.get("release_state_id", "")) == f"{seed}:{frame}", f"seed {seed} frame {frame}: checkpoint branch state id drift")
            _require(str(row.get("release_state_identity_sha256", "")) == identity_sha, f"seed {seed} frame {frame}: checkpoint branch identity drift")
            _require(str(row.get("gate_action_universe", "")) == str(label["gate_action_universe"]), f"seed {seed} frame {frame}: checkpoint branch universe drift")
            _require(int(row.get("exact_action_provenance", 0)) == 1, f"seed {seed} frame {frame}: checkpoint branch provenance drift")
            _require(
                str(row.get("runtime_effective_action_universe", ""))
                == str(label["gate_action_universe"])
                and str(row.get("runtime_gate_action_universe", ""))
                == str(label["gate_action_universe"]),
                f"seed {seed} frame {frame}: checkpoint runtime universe drift",
            )
            _require(
                str(row.get("runtime_gate_action_universe_source", ""))
                == ACTION_UNIVERSE_SOURCE
                and str(row.get("runtime_fast_action_universe_source", ""))
                == ACTION_UNIVERSE_SOURCE,
                f"seed {seed} frame {frame}: checkpoint runtime universe source drift",
            )
            _validate_branch_outcome(row, seed=seed, frame=frame)
            _require(
                int(row.get("horizon_steps", -1)) == int(horizon),
                f"seed {seed} frame {frame}: checkpoint branch horizon drift",
            )
            _require(
                _metric_matches(row.get("gamma"), gamma),
                f"seed {seed} frame {frame}: checkpoint branch gamma drift",
            )
            computed_identity = _identity_text(
                _finite_effective_identity(row, seed=seed, frame=frame)
            )
            _require(str(row.get("effective_identity", "")) == computed_identity, f"seed {seed} frame {frame}: checkpoint effective identity drift")

        candidates = [row for row in rows if str(row["branch_role"]) == "candidate"]
        derived = _derive_release_outcome(
            baseline,
            candidates,
            seed=seed,
            frame=frame,
            predicted_action=int(baseline["fast_action"]),
            epsilon=float(epsilon),
        )
        baseline_identity_tuple = derived["baseline_identity"]
        baseline_identity = _identity_text(baseline_identity_tuple)
        _require(
            str(baseline["effective_identity"]) == baseline_identity,
            f"seed {seed} frame {frame}: checkpoint baseline identity derivation drift",
        )
        _require(
            int(baseline.get("matches_matched_fast_identity", -1)) == 1
            and int(baseline.get("effective_identity_representative", -1)) == 1
            and str(baseline.get("effective_identity_representative_raw_action", ""))
            == "fast"
            and _metric_matches(
                baseline.get("advantage_over_matched_fast"), 0.0
            )
            and int(baseline.get("in_corrective_set", -1)) == 0,
            f"seed {seed} frame {frame}: checkpoint baseline annotations drift",
        )
        identity_groups: Dict[str, List[Mapping[str, Any]]] = {}
        for candidate in candidates:
            identity_groups.setdefault(str(candidate["effective_identity"]), []).append(candidate)
        for identity, aliases in identity_groups.items():
            representatives = [
                row for row in aliases if int(row.get("effective_identity_representative", 0)) == 1
            ]
            _require(len(representatives) == 1, f"seed {seed} frame {frame}: checkpoint representative drift for {identity}")
            expected_representative = min(int(row["raw_action"]) for row in aliases)
            _require(int(representatives[0]["raw_action"]) == expected_representative, f"seed {seed} frame {frame}: checkpoint noncanonical representative")
            for alias in aliases[1:]:
                _require_equivalent_outcomes(
                    aliases[0], alias, seed=seed, frame=frame,
                    context=f"checkpoint aliases for {identity}",
                )
        representative_by_identity = derived["representative_by_identity"]
        corrective_identities = derived["corrective_identities"]
        for candidate in candidates:
            identity = _finite_effective_identity(candidate, seed=seed, frame=frame)
            raw_action = int(candidate["raw_action"])
            representative = int(representative_by_identity[identity])
            expected_advantage = float(candidate["utility"]) - float(
                baseline["utility"]
            )
            expected_in_corrective_set = int(
                identity in corrective_identities and raw_action == representative
            )
            _require(
                int(candidate.get("matches_matched_fast_identity", -1))
                == int(identity == baseline_identity_tuple),
                f"seed {seed} frame {frame}: checkpoint Fast-identity annotation drift",
            )
            _require(
                int(candidate.get("effective_identity_representative", -1))
                == int(raw_action == representative)
                and int(
                    candidate.get(
                        "effective_identity_representative_raw_action", -1
                    )
                )
                == representative,
                f"seed {seed} frame {frame}: checkpoint representative annotation drift",
            )
            _require(
                _metric_matches(
                    candidate.get("advantage_over_matched_fast"),
                    expected_advantage,
                ),
                f"seed {seed} frame {frame}: checkpoint advantage derivation drift",
            )
            _require(
                int(candidate.get("in_corrective_set", -1))
                == expected_in_corrective_set,
                f"seed {seed} frame {frame}: checkpoint corrective annotation drift",
            )

        matching = sum(
            _finite_effective_identity(row, seed=seed, frame=frame)
            == baseline_identity_tuple
            for row in candidates
        )
        alternatives = derived["alternatives"]
        corrective = len(derived["corrective_rows"])
        _require(int(label.get("gate_action_count", -1)) == len(universe), f"seed {seed} frame {frame}: checkpoint gate count drift")
        _require(int(label.get("candidate_branch_count", -1)) == len(candidates), f"seed {seed} frame {frame}: checkpoint candidate count drift")
        _require(int(label.get("distinct_effective_candidate_count", -1)) == len(identity_groups), f"seed {seed} frame {frame}: checkpoint distinct identity count drift")
        _require(int(label.get("distinct_effective_alternative_count", -1)) == len(alternatives), f"seed {seed} frame {frame}: checkpoint alternative count drift")
        _require(int(label.get("candidate_actions_matching_fast_identity", -1)) == matching, f"seed {seed} frame {frame}: checkpoint Fast identity count drift")
        _require(int(label.get("candidate_effective_aliases_collapsed", -1)) == len(candidates) - len(identity_groups), f"seed {seed} frame {frame}: checkpoint alias count drift")
        _require(int(label.get("corrective_set_action_count", -1)) == corrective, f"seed {seed} frame {frame}: checkpoint CSet count drift")
        _require(int(label.get("corrective_set_nonempty", -1)) == int(corrective > 0), f"seed {seed} frame {frame}: checkpoint CSet label drift")
        _require(
            int(label.get("horizon_steps", -1)) == int(horizon)
            and _metric_matches(label.get("gamma"), gamma)
            and _metric_matches(label.get("epsilon"), epsilon),
            f"seed {seed} frame {frame}: checkpoint label rollout contract drift",
        )
        _require(
            _metric_matches(label.get("baseline_utility"), baseline["utility"])
            and int(label.get("baseline_collision", -1))
            == int(baseline["collision"])
            and int(label.get("baseline_effective_action", -1))
            == int(baseline["effective_action"])
            and _metric_matches(
                label.get("baseline_target_speed_after"),
                baseline["target_speed_after"],
            ),
            f"seed {seed} frame {frame}: checkpoint baseline label derivation drift",
        )
        best_row = derived["best_row"]
        if best_row is None:
            _require(
                label.get("best_advantage") in (None, "")
                and label.get("best_raw_action") in (None, "")
                and label.get("best_effective_action") in (None, "")
                and label.get("best_target_speed_after") in (None, "")
                and label.get("best_collision") in (None, ""),
                f"seed {seed} frame {frame}: checkpoint empty-best derivation drift",
            )
        else:
            _require(
                _metric_matches(
                    label.get("best_advantage"), derived["best_advantage"]
                )
                and int(label.get("best_raw_action", -1))
                == int(best_row["raw_action"])
                and int(label.get("best_effective_action", -1))
                == int(best_row["effective_action"])
                and _metric_matches(
                    label.get("best_target_speed_after"),
                    best_row["target_speed_after"],
                )
                and int(label.get("best_collision", -1))
                == int(best_row["collision"]),
                f"seed {seed} frame {frame}: checkpoint best-action derivation drift",
            )

    accounting = dict(payload.get("accounting", {}) or {})
    expected_accounting = {
        "seed": seed,
        "status": "complete",
        "requested_target_entries": len(targets),
        "unique_release_states": len(unique_targets),
        "duplicate_targets_excluded": len(targets) - len(unique_targets),
        "release_states_evaluated": len(labels),
        "release_states_excluded": 0,
        "release_state_errors": 0,
        "branch_rows": len(branches),
        "gate_candidate_branches": sum(int(row["gate_action_count"]) for row in labels),
        "candidate_actions_matching_fast_identity": sum(int(row["candidate_actions_matching_fast_identity"]) for row in labels),
        "candidate_effective_aliases_collapsed": sum(int(row["candidate_effective_aliases_collapsed"]) for row in labels),
        "corrective_set_actions": sum(int(row["corrective_set_action_count"]) for row in labels),
        "corrective_release_states": sum(int(row["corrective_set_nonempty"]) for row in labels),
    }
    for field, expected in expected_accounting.items():
        _require(accounting.get(field) == expected, f"seed {seed}: checkpoint accounting drift at {field}")
    return dict(payload)


def _validate_v12_protocol(
    protocol_path: Path,
    *,
    seeds: Sequence[int],
    horizon: int,
    gamma: float,
    epsilon: float,
) -> str:
    protocol = load_formal_protocol(protocol_path)
    submission = dict(protocol.get("tvt_submission_contract", {}) or {})
    _require(
        str(submission.get("query_gate_method_version", "") or "") == METHOD_VERSION,
        "branch query-gate method version drift",
    )
    _require(
        METHOD_VERSION
        in str(submission.get("mechanism_evaluation_compatibility", "") or ""),
        "branch mechanism compatibility drift",
    )
    rollout = dict(submission.get("release_rollout", {}) or {})
    _require(int(rollout.get("horizon_steps", -1)) == int(horizon), "branch horizon differs from v12 protocol")
    _require(math.isclose(float(rollout.get("gamma", float("nan"))), float(gamma), rel_tol=0.0, abs_tol=1e-12), "branch gamma differs from v12 protocol")
    _require(math.isclose(float(rollout.get("corrective_margin", float("nan"))), float(epsilon), rel_tol=0.0, abs_tol=1e-12), "branch epsilon differs from v12 protocol")
    calibration = dict(submission.get("v12_calibration", {}) or {})
    cohort_specs = {
        "parameter_selection": calibration.get("parameter_selection_seed_range"),
        "fixed_parameter_go_no_go": calibration.get("fixed_parameter_go_no_go_seed_range"),
        "confirmatory_holdout": calibration.get("confirmatory_holdout_seed_range"),
    }
    requested = [int(seed) for seed in seeds]
    for label, raw_spec in cohort_specs.items():
        spec = dict(raw_spec or {})
        if not spec:
            continue
        expected = list(range(int(spec["start"]), int(spec["end"]) + 1))
        _require(int(spec.get("count", -1)) == len(expected), f"v12 {label} seed count drift")
        if requested == expected:
            return label
    raise ValueError("requested seed block is not a preregistered v12 cohort")


def _floor_overlay_run_contract(
    verified: Optional[VerifiedFloorOverlay],
) -> Optional[Dict[str, Any]]:
    if verified is None:
        return None
    return {
        "runtime_binding": dict(verified.runtime_binding),
        "floor_overlay": {
            "path": str(verified.path),
            "sha256": verified.raw_sha256,
            "payload_sha256": verified.payload_sha256,
        },
        "calibration_manifest": {
            "path": str(verified.calibration_manifest_path),
            "sha256": verified.calibration_manifest_sha256,
        },
        "calibration_lock": {
            "path": str(verified.lock_path),
            "sha256": _sha256(verified.lock_path),
        },
    }


def _run_contract(
    *,
    trace_root: Path,
    target_metadata: Mapping[str, Any],
    protocol: Path,
    seeds: Sequence[int],
    horizon: int,
    gamma: float,
    epsilon: float,
    verified_floor_overlay: Optional[VerifiedFloorOverlay] = None,
) -> Dict[str, Any]:
    runner_path = Path(__file__).resolve()
    runtime_source_sha256 = build_runtime_source_hash(REPO_ROOT)
    cohort = _validate_v12_protocol(
        protocol,
        seeds=seeds,
        horizon=horizon,
        gamma=gamma,
        epsilon=epsilon,
    )
    protocol_payload = load_formal_protocol(protocol)
    partition = enforce_v12_floor_overlay_contract(
        str(protocol_payload.get("protocol_name", "") or ""),
        seeds,
        verified_floor_overlay,
    )
    expected_partition = {
        "parameter_selection": "calibration",
        "fixed_parameter_go_no_go": "go_no_go",
        "confirmatory_holdout": "confirmatory_holdout",
    }[cohort]
    _require(
        partition == expected_partition,
        "branch-label cohort and floor-overlay partition disagree",
    )
    contract = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "method_version": METHOD_VERSION,
        "continuation_contract_version": CONTINUATION_CONTRACT_VERSION,
        "trace_root": str(trace_root.resolve()),
        "snapshot_targets": dict(target_metadata),
        "protocol": {"path": str(protocol.resolve()), "sha256": _sha256(protocol)},
        "runner": {"path": str(runner_path), "sha256": _sha256(runner_path)},
        "branch_engine": {
            "path": str(BRANCH_ENGINE_PATH.resolve()),
            "sha256": _sha256(BRANCH_ENGINE_PATH),
        },
        "snapshot_producer": {
            "path": str(SNAPSHOT_PRODUCER_PATH.resolve()),
            "sha256": _sha256(SNAPSHOT_PRODUCER_PATH),
        },
        "base_config": {
            "path": str(BASE_CONFIG_PATH.resolve()),
            "sha256": _sha256(BASE_CONFIG_PATH),
        },
        "runtime_source_sha256": runtime_source_sha256,
        "runtime_environment": _current_runtime_environment(),
        "seeds": [int(seed) for seed in seeds],
        "v12_cohort": cohort,
        "v12_partition": partition,
        "v12_floor_overlay": _floor_overlay_run_contract(
            verified_floor_overlay
        ),
        "horizon_steps": int(horizon),
        "gamma": float(gamma),
        "epsilon": float(epsilon),
        "gate_selection_performed": False,
    }
    contract["contract_fingerprint"] = _canonical_sha256(contract)
    return contract


def _error_accounting(seed: int, requested: Sequence[int], result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "seed": int(seed),
        "status": "error_fail_closed",
        "requested_target_entries": len(requested),
        "unique_release_states": len(set(requested)),
        "duplicate_targets_excluded": len(requested) - len(set(requested)),
        "release_states_evaluated": 0,
        "release_states_excluded": 0,
        "release_state_errors": 1,
        "branch_rows": 0,
        "gate_candidate_branches": 0,
        "candidate_actions_matching_fast_identity": 0,
        "candidate_effective_aliases_collapsed": 0,
        "corrective_set_actions": 0,
        "corrective_release_states": 0,
        "max_snapshot_position_error_m": "",
        "max_snapshot_speed_error_mps": "",
        "error_type": str(result.get("error_type", "RuntimeError")),
        "error_message": str(result.get("error_message", "unknown worker error")),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--snapshot-targets", type=Path, required=True)
    parser.add_argument("--target-events", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=Path("formal_protocol.yaml")
    )
    parser.add_argument(
        "--floor-overlay",
        type=Path,
        default=None,
        help="Immutable v12 runtime floor overlay required after calibration.",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        default=None,
        help="Locked v12 calibration selection authenticating --floor-overlay.",
    )
    parser.add_argument(
        "--calibration-lock",
        type=Path,
        default=DEFAULT_LOCK_PATH,
        help="Immutable v12 calibration lock.",
    )
    parser.add_argument(
        "--holdout-authorization",
        type=Path,
        default=None,
        help="One-shot authorization required for the exact 3000-3029 holdout.",
    )
    parser.add_argument(
        "--go-no-go-manifest",
        type=Path,
        default=None,
        help="Passing fixed-parameter go/no-go evidence bound to holdout authorization.",
    )
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path(os.environ.get("TEMP", ".")) / "v12_branch_label_scratch",
    )
    return parser.parse_args(argv)


def _run_main(args: argparse.Namespace, *, holdout_claim: Any = None) -> int:
    _require(args.seeds > 0, "--seeds must be positive")
    _require(args.horizon > 0, "--horizon must be positive")
    _require(math.isfinite(args.gamma) and 0.0 < args.gamma <= 1.0, "--gamma must lie in (0, 1]")
    _require(math.isfinite(args.epsilon) and args.epsilon >= 0.0, "--epsilon must be nonnegative")
    _require(args.workers > 0, "--workers must be positive")
    _require(args.trace_root.is_dir(), f"trace root not found: {args.trace_root}")
    _require(args.snapshot_targets.is_file(), f"snapshot targets not found: {args.snapshot_targets}")
    _require(args.target_events.is_file(), f"target events not found: {args.target_events}")
    _require(args.target_manifest.is_file(), f"target manifest not found: {args.target_manifest}")
    _require(args.protocol.is_file(), f"protocol not found: {args.protocol}")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    verified_floor_overlay = load_optional_verified_floor_overlay(
        args.floor_overlay,
        calibration_manifest_path=args.calibration_manifest,
        protocol_path=args.protocol,
        lock_path=args.calibration_lock,
    )
    targets, target_events, target_metadata = _load_target_bundle(
        args.snapshot_targets,
        args.target_events,
        args.target_manifest,
        seeds,
    )
    contract = _run_contract(
        trace_root=args.trace_root,
        target_metadata=target_metadata,
        protocol=args.protocol,
        seeds=seeds,
        horizon=args.horizon,
        gamma=args.gamma,
        epsilon=args.epsilon,
        verified_floor_overlay=verified_floor_overlay,
    )
    fingerprint = str(contract["contract_fingerprint"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / RUN_CONTRACT_FILE
    if contract_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        _require(existing_contract == contract, "existing v12 branch run contract drifted")
    else:
        _atomic_write_json(contract_path, contract)
    manifest_path = args.output_dir / MANIFEST_FILE
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "status": "in_progress",
            "method_version": METHOD_VERSION,
            "continuation_contract_version": CONTINUATION_CONTRACT_VERSION,
            "contract_fingerprint": fingerprint,
            "run_contract_sha256": _sha256(contract_path),
        },
    )

    checkpoints: Dict[int, Dict[str, Any]] = {}
    accounting_by_seed: Dict[int, Dict[str, Any]] = {}
    pending: List[int] = []
    failures: List[Dict[str, Any]] = []
    for seed in seeds:
        checkpoint_path = _checkpoint_path(args.output_dir, seed)
        if checkpoint_path.is_file():
            try:
                payload = json.loads(
                    checkpoint_path.read_text(encoding="utf-8-sig")
                )
                checkpoint = _validate_checkpoint(
                    payload,
                    seed=seed,
                    targets=targets[seed],
                    contract_fingerprint=fingerprint,
                    trace_root=args.trace_root,
                    horizon=args.horizon,
                    gamma=args.gamma,
                    epsilon=args.epsilon,
                )
            except Exception as exc:
                result = _exception_result(seed, exc)
                failures.append(result)
                accounting_by_seed[seed] = _error_accounting(
                    seed, targets[seed], result
                )
                continue
            checkpoints[seed] = checkpoint
            accounting_by_seed[seed] = dict(checkpoint["accounting"])
        else:
            pending.append(seed)

    def record_failure(seed: int, result: Mapping[str, Any]) -> None:
        failures.append(dict(result))
        accounting_by_seed[seed] = _error_accounting(seed, targets[seed], result)
        print(
            f"seed={seed} fail-closed: {result.get('error_message', 'unknown error')}",
            flush=True,
        )

    if pending:
        try:
            with ProcessPoolExecutor(
                max_workers=min(int(args.workers), len(pending))
            ) as pool:
                futures = {}
                for seed in pending:
                    try:
                        future = pool.submit(
                            _worker,
                            seed,
                            tuple(targets[seed]),
                            str(args.trace_root.resolve()),
                            str(args.protocol.resolve()),
                            str(args.scratch_root.resolve()),
                            int(args.horizon),
                            float(args.gamma),
                            float(args.epsilon),
                            str(contract["runtime_source_sha256"]),
                            str(contract["protocol"]["sha256"]),
                            (
                                None
                                if verified_floor_overlay is None
                                else str(verified_floor_overlay.path)
                            ),
                            (
                                None
                                if verified_floor_overlay is None
                                else str(
                                    verified_floor_overlay.calibration_manifest_path
                                )
                            ),
                            (
                                None
                                if verified_floor_overlay is None
                                else str(verified_floor_overlay.lock_path)
                            ),
                            str(contract["v12_partition"]),
                        )
                    except Exception as exc:
                        record_failure(seed, _exception_result(seed, exc))
                        continue
                    futures[future] = seed
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = _exception_result(seed, exc)
                    if result.get("status") != "ok":
                        record_failure(seed, result)
                        continue
                    checkpoint = dict(result["payload"])
                    checkpoint["contract_fingerprint"] = fingerprint
                    checkpoint["checkpoint_payload_sha256"] = (
                        _checkpoint_payload_sha256(checkpoint)
                    )
                    _atomic_write_json(
                        _checkpoint_path(args.output_dir, seed), checkpoint
                    )
                    checkpoints[seed] = checkpoint
                    accounting_by_seed[seed] = dict(checkpoint["accounting"])
                    print(
                        f"seed={seed} releases={len(checkpoint['labels'])} "
                        f"branches={len(checkpoint['branches'])}",
                        flush=True,
                    )
        except Exception as exc:
            for seed in pending:
                if seed in checkpoints or seed in accounting_by_seed:
                    continue
                record_failure(seed, _exception_result(seed, exc))

    accounting_rows = [accounting_by_seed[seed] for seed in seeds if seed in accounting_by_seed]
    if failures:
        failure_path = args.output_dir / "v12_branch_failures.json"
        _atomic_write_csv(args.output_dir / ACCOUNTING_FILE, accounting_rows)
        _atomic_write_json(failure_path, failures)
        _atomic_write_json(
            manifest_path,
            {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "status": "failed_closed",
                "method_version": METHOD_VERSION,
                "continuation_contract_version": CONTINUATION_CONTRACT_VERSION,
                "contract_fingerprint": fingerprint,
                "failure_count": len(failures),
                "failures_sha256": _sha256(failure_path),
                "accounting_sha256": _sha256(args.output_dir / ACCOUNTING_FILE),
            },
        )
        raise RuntimeError(
            f"v12 branch labeling failed closed for {len(failures)} seed(s); see {failure_path}"
        )

    _require(set(checkpoints) == set(seeds), "incomplete checkpoint set after branch run")
    for seed in seeds:
        checkpoints[seed] = _validate_checkpoint(
            checkpoints[seed],
            seed=seed,
            targets=targets[seed],
            contract_fingerprint=fingerprint,
            trace_root=args.trace_root,
            horizon=args.horizon,
            gamma=args.gamma,
            epsilon=args.epsilon,
        )
    branch_rows = [
        row for seed in seeds for row in checkpoints[seed]["branches"]
    ]
    release_label_rows = [row for seed in seeds for row in checkpoints[seed]["labels"]]
    branch_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            int(row["release_frame"]),
            0 if str(row["branch_role"]) == "matched_fast" else 1,
            -1 if str(row["raw_action"]) == "fast" else int(row["raw_action"]),
        )
    )
    release_label_rows.sort(key=lambda row: (int(row["seed"]), int(row["release_frame"])))
    label_rows = _expand_labels_to_target_events(release_label_rows, target_events)
    source_artifacts = {
        str(seed): checkpoints[seed]["source_artifacts"] for seed in seeds
    }
    source_execution_contracts = {
        str(seed): checkpoints[seed]["source_execution_contract"] for seed in seeds
    }
    input_hashes_after = {
        "snapshot_targets_sha256": _sha256(args.snapshot_targets),
        "target_events_sha256": _sha256(args.target_events),
        "target_manifest_sha256": _sha256(args.target_manifest),
        "protocol_sha256": _sha256(args.protocol),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "branch_engine_sha256": _sha256(BRANCH_ENGINE_PATH),
        "snapshot_producer_sha256": _sha256(SNAPSHOT_PRODUCER_PATH),
        "base_config_sha256": _sha256(BASE_CONFIG_PATH),
        "runtime_source_sha256": build_runtime_source_hash(REPO_ROOT),
        "runtime_environment_sha256": _canonical_sha256(
            _current_runtime_environment()
        ),
    }
    expected_hashes = {
        "snapshot_targets_sha256": target_metadata["target_map"]["sha256"],
        "target_events_sha256": target_metadata["target_events"]["sha256"],
        "target_manifest_sha256": target_metadata["target_manifest"]["sha256"],
        "protocol_sha256": contract["protocol"]["sha256"],
        "runner_sha256": contract["runner"]["sha256"],
        "branch_engine_sha256": contract["branch_engine"]["sha256"],
        "snapshot_producer_sha256": contract["snapshot_producer"]["sha256"],
        "base_config_sha256": contract["base_config"]["sha256"],
        "runtime_source_sha256": contract["runtime_source_sha256"],
        "runtime_environment_sha256": _canonical_sha256(
            contract["runtime_environment"]
        ),
    }
    overlay_contract = contract.get("v12_floor_overlay")
    if isinstance(overlay_contract, Mapping):
        floor_artifact = dict(overlay_contract["floor_overlay"])
        calibration_artifact = dict(overlay_contract["calibration_manifest"])
        lock_artifact = dict(overlay_contract["calibration_lock"])
        input_hashes_after.update(
            {
                "floor_overlay_sha256": _sha256(Path(floor_artifact["path"])),
                "calibration_manifest_sha256": _sha256(
                    Path(calibration_artifact["path"])
                ),
                "calibration_lock_sha256": _sha256(Path(lock_artifact["path"])),
            }
        )
        expected_hashes.update(
            {
                "floor_overlay_sha256": floor_artifact["sha256"],
                "calibration_manifest_sha256": calibration_artifact["sha256"],
                "calibration_lock_sha256": lock_artifact["sha256"],
            }
        )
    _require(input_hashes_after == expected_hashes, "runner inputs changed during branch labeling")
    _atomic_write_csv(args.output_dir / ACCOUNTING_FILE, accounting_rows)
    _atomic_write_csv(args.output_dir / BRANCH_ROWS_FILE, branch_rows)
    _atomic_write_csv(args.output_dir / LABELS_FILE, label_rows)
    current_source_artifacts = {
        str(seed): _source_hashes(_source_paths(args.trace_root, seed))
        for seed in seeds
    }
    _require(
        current_source_artifacts == source_artifacts,
        "source artifacts changed during final output publication",
    )
    cohort = str(contract["v12_cohort"])
    partition = {
        "parameter_selection": "calibration",
        "fixed_parameter_go_no_go": "go_no_go",
        "confirmatory_holdout": "confirmatory_holdout",
    }[cohort]
    artifact_role = (
        "confirmatory_holdout_branch_labels"
        if cohort == "confirmatory_holdout"
        else f"{partition}_branch_labels"
    )
    holdout_binding: Dict[str, Any] = {}
    if holdout_claim is not None:
        _require(cohort == "confirmatory_holdout", "holdout claim used outside confirmatory cohort")
        holdout_binding = {
            "authorization_id": holdout_claim.authorization.authorization_id,
            "authorization_sha256": holdout_claim.authorization.raw_sha256,
            "target_artifacts": dict(holdout_claim.run_binding["target_artifacts"]),
            "snapshot_producer_manifest_sha256": holdout_claim.run_binding[
                "snapshot_manifest_sha256"
            ],
        }
    manifest = {
        "schema": BRANCH_MANIFEST_SCHEMA,
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "artifact_role": artifact_role,
        "partition": partition,
        **holdout_binding,
        "analysis": "independent outcome-grounded matched-action branch labels",
        "status": "complete",
        "method_version": METHOD_VERSION,
        "label_source": LABEL_SOURCE,
        "continuation_contract_version": CONTINUATION_CONTRACT_VERSION,
        "gate_selection_performed": False,
        "trace_root": str(args.trace_root.resolve()),
        "snapshot_targets": target_metadata,
        "protocol": contract["protocol"],
        "runner": contract["runner"],
        "branch_engine": contract["branch_engine"],
        "snapshot_producer": contract["snapshot_producer"],
        "base_config": contract["base_config"],
        "runtime_source_sha256": contract["runtime_source_sha256"],
        "runtime_environment": contract["runtime_environment"],
        "contract_fingerprint": fingerprint,
        "seeds": seeds,
        "v12_cohort": contract["v12_cohort"],
        "v12_partition": contract["v12_partition"],
        "v12_floor_overlay": contract["v12_floor_overlay"],
        "seed_is_experimental_unit": True,
        "horizon_steps": int(args.horizon),
        "gamma": float(args.gamma),
        "epsilon": float(args.epsilon),
        "utility": "discounted normalized simulator return minus collision indicator",
        "action_identity": "post-safety discrete command plus post-bridge target speed",
        "continuation": "same complete deterministic Fast controller after the release action",
        "corrective_set": "distinct non-Fast effective identities with utility advantage >= epsilon",
        "exact_action_provenance": "exact",
        "exact_action_contract": {
            "source": "per-release rgd_record_v3 recoverability_gate.gate_action_universe",
            "required_method_version": METHOD_VERSION,
            "gate_action_universe_source": ACTION_UNIVERSE_SOURCE,
            "fast_executor_action_universe_source": ACTION_UNIVERSE_SOURCE,
            "gate_and_fast_universes_must_match": True,
            "all_exact_raw_actions_simulated_once": True,
            "action_zero_preserved": True,
        },
        "release_state_identity": {
            "snapshot_frame_matches_trace_frame": True,
            "position_tolerance_m": POSITION_TOLERANCE_M,
            "speed_tolerance_mps": SPEED_TOLERANCE_MPS,
            "lane_and_previous_action_exact": True,
            "history_and_fast_action_prefix_exact": True,
            "snapshot_obs_history_fast_and_policy_state_hashed": True,
            "required_policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
            "matched_fast_proposal_and_effective_action_exact": True,
            "matched_fast_full_horizon_trace_exact": True,
        },
        "resume": {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_dir": str((args.output_dir / CHECKPOINT_DIR).resolve()),
            "source_hashes_revalidated_before_reuse": True,
            "contract_drift_policy": "fail_closed",
        },
        "counts": {
            "requested_target_entries": sum(int(row["requested_target_entries"]) for row in accounting_rows),
            "target_event_rows": len(label_rows),
            "unique_release_states": len(release_label_rows),
            "duplicate_targets_excluded": sum(int(row["duplicate_targets_excluded"]) for row in accounting_rows),
            "release_states_excluded": sum(int(row["release_states_excluded"]) for row in accounting_rows),
            "release_state_errors": 0,
            "branch_rows": len(branch_rows),
            "gate_candidate_branches": sum(int(row["gate_candidate_branches"]) for row in accounting_rows),
            "candidate_actions_matching_fast_identity": sum(int(row["candidate_actions_matching_fast_identity"]) for row in accounting_rows),
            "candidate_effective_aliases_collapsed": sum(int(row["candidate_effective_aliases_collapsed"]) for row in accounting_rows),
            "corrective_set_actions": sum(int(row["corrective_set_actions"]) for row in accounting_rows),
            "corrective_release_states": sum(int(row["corrective_release_states"]) for row in accounting_rows),
        },
        "source_artifacts": source_artifacts,
        "source_execution_contracts": source_execution_contracts,
        "input_hashes": input_hashes_after,
        "output_hashes": {
            BRANCH_ROWS_FILE: _sha256(args.output_dir / BRANCH_ROWS_FILE),
            LABELS_FILE: _sha256(args.output_dir / LABELS_FILE),
            ACCOUNTING_FILE: _sha256(args.output_dir / ACCOUNTING_FILE),
            RUN_CONTRACT_FILE: _sha256(contract_path),
        },
    }
    stale_failures = args.output_dir / "v12_branch_failures.json"
    if stale_failures.is_file():
        stale_failures.unlink()
    manifest["manifest_payload_hash"] = _canonical_sha256(manifest)
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=False), flush=True)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _require(args.seeds > 0, "--seeds must be positive")
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
    cohort = _validate_v12_protocol(
        args.protocol,
        seeds=seeds,
        horizon=args.horizon,
        gamma=args.gamma,
        epsilon=args.epsilon,
    )
    holdout_claim = None
    holdout_args = {
        "--holdout-authorization": args.holdout_authorization,
        "--calibration-manifest": args.calibration_manifest,
        "--go-no-go-manifest": args.go_no_go_manifest,
    }
    if cohort == "confirmatory_holdout":
        missing = [name for name, value in holdout_args.items() if value is None]
        _require(not missing, "confirmatory branch run requires " + ", ".join(missing))
        expected_events = args.snapshot_targets.with_suffix(".events.csv").resolve()
        expected_manifest = args.snapshot_targets.with_suffix(".manifest.json").resolve()
        _require(args.target_events.resolve() == expected_events, "holdout target events are not the authorized sibling artifact")
        _require(args.target_manifest.resolve() == expected_manifest, "holdout target manifest is not the authorized sibling artifact")
        from tools.v12_holdout_guard import begin_branch_consumption

        holdout_claim = begin_branch_consumption(
            authorization_path=args.holdout_authorization,
            protocol_path=args.protocol,
            lock_path=args.calibration_lock,
            calibration_manifest_path=args.calibration_manifest,
            go_no_go_manifest_path=args.go_no_go_manifest,
            target_map_path=args.snapshot_targets,
            branch_output_dir=args.output_dir,
            branch_runner_path=Path(__file__),
        )
    else:
        _require(
            args.holdout_authorization is None and args.go_no_go_manifest is None,
            "holdout authorization inputs are valid only for seeds 3000-3029",
        )
    try:
        result = _run_main(args, holdout_claim=holdout_claim)
        if holdout_claim is not None:
            from tools.v12_holdout_guard import complete_branch_consumption

            complete_branch_consumption(
                holdout_claim,
                args.output_dir / MANIFEST_FILE,
            )
        return result
    except BaseException as exc:
        if holdout_claim is not None:
            from tools.v12_holdout_guard import fail_phase

            try:
                fail_phase(holdout_claim, exc)
            except Exception as guard_exc:
                exc.add_note(
                    "holdout branch failed and fail-closed state recording also failed: "
                    + str(guard_exc)
                )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
