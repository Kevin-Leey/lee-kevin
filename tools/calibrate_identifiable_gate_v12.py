"""Preregistered calibration-only selector for the identifiable v12 gate.

Partition-specific commands prevent the parameter selector from reading a
validation or paper-result partition:

``targets`` enumerates every release state that any preregistered full or
leave-one-out candidate can schedule, allowing one branch-simulation pass.
``select`` joins the resulting calibration labels and chooses the frozen gate
floors. ``go-no-go-locked`` and ``validate-locked`` accept only those frozen
floors and disjoint preregistered seed blocks. A selected calibration candidate
is not held-out validation and the emitted acceptance object states that
distinction explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.driver_agent.reasoning.rgd_support import compute_temporal_survival  # noqa: E402


LOCK_PATH = Path(__file__).with_name("identifiable_gate_v12_calibration_lock.json")
GATE_SUPPORT_PATH = REPO_ROOT / "dilu" / "driver_agent" / "reasoning" / "rgd_support.py"
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "formal_protocol_v12.yaml"
TARGET_MANIFEST_SCHEMA = "identifiable_gate_v12_snapshot_targets_v2"
LABEL_SOURCE = "matched_release_state_exact_action_rollout_v1"
CANONICAL_ACTION_FAMILIES = {
    0: "lateral-left",
    1: "lane-hold",
    2: "lateral-right",
    3: "longitudinal-accelerate",
    4: "longitudinal-decelerate",
}
METHOD_VERSION = "identifiable_gate_v12"
ARM_FULL = "full"
ARM_NO_L = "w/o L"
ARM_NO_A = "w/o A"
ARM_NO_H = "w/o H"
ARMS = (ARM_FULL, ARM_NO_L, ARM_NO_A, ARM_NO_H)
COMPONENT_ARMS = {"L": ARM_NO_L, "A": ARM_NO_A, "H": ARM_NO_H}


@dataclass(frozen=True)
class FactorialArm:
    label: str
    use_l: bool
    use_a: bool
    use_h: bool


VALIDATION_ARMS: Tuple[FactorialArm, ...] = (
    FactorialArm(ARM_FULL, True, True, True),
    FactorialArm(ARM_NO_L, False, True, True),
    FactorialArm(ARM_NO_A, True, False, True),
    FactorialArm(ARM_NO_H, True, True, False),
    FactorialArm("L only", True, False, False),
    FactorialArm("A only", False, True, False),
    FactorialArm("H only", False, False, True),
    FactorialArm("F+I only", False, False, False),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field, None)
    return _semantic_hash(canonical)


def _verify_payload_hash(payload: Mapping[str, Any], field: str, artifact: str) -> None:
    _require(
        str(payload.get(field, "") or "") == _payload_hash(payload, field),
        f"{artifact} payload hash drift",
    )


def _canonical_float(value: float) -> str:
    return format(float(value), ".12g")


def _as_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"0", "false"}:
            return False
        if normalized in {"1", "true"}:
            return True
    raise ValueError(f"{field} must be an exact boolean/0/1 value, got {value!r}")


def _unit(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1], got {value!r}")
    return result


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for item in path:
        if not isinstance(current, Mapping) or item not in current:
            return None
        current = current[item]
    return current


def _same_alias(
    candidates: Sequence[Tuple[str, Any]],
    field: str,
    *,
    required: bool = True,
) -> Any:
    present = [(name, value) for name, value in candidates if value is not None]
    if not present:
        if required:
            raise ValueError(f"v12 trace misses {field}")
        return None
    first_name, first_value = present[0]
    for name, value in present[1:]:
        if isinstance(first_value, (int, float)) and isinstance(value, (int, float)):
            same = math.isclose(
                float(first_value), float(value), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            same = value == first_value
        if not same:
            raise ValueError(
                f"conflicting aliases for {field}: {first_name}={first_value!r}, "
                f"{name}={value!r}"
            )
    return first_value


def _canonical_action_id(value: Any, field: str, *, allow_json_key: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be boolean")
    if isinstance(value, int):
        action = value
    elif allow_json_key and isinstance(value, str) and re.fullmatch(r"0|[1-4]", value):
        action = int(value)
    else:
        raise ValueError(f"{field} is not a canonical action id: {value!r}")
    if action not in CANONICAL_ACTION_FAMILIES:
        raise ValueError(f"{field} action id is outside 0..4: {action}")
    return action


def _action_sequence(value: Any, field: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a nonempty exact action sequence")
    actions = tuple(_canonical_action_id(item, field) for item in value)
    if len(actions) != len(set(actions)):
        raise ValueError(f"{field} contains duplicate action ids")
    return tuple(sorted(actions))


def _cost_map(value: Any, field: str, *, fail_closed_record: bool = False) -> Dict[int, float]:
    if not isinstance(value, Mapping) or (not value and not fail_closed_record):
        raise ValueError(f"{field} must be a nonempty action-cost map")
    result: Dict[int, float] = {}
    for raw_action, raw_cost in value.items():
        try:
            action = _canonical_action_id(raw_action, field, allow_json_key=True)
            cost = _unit(raw_cost, f"{field}[{action}]")
        except (TypeError, ValueError):
            if fail_closed_record:
                continue
            raise
        if action in result:
            raise ValueError(f"{field} has a duplicate normalized action id {action}")
        result[action] = cost
    return result


def derive_relative_support_maneuver_breadth(
    *,
    gate_actions: Sequence[int],
    hold_action: int,
    recovery_costs: Mapping[int, float],
    support_costs: Mapping[int, float],
    maneuver_families: Mapping[int, str],
    viable_cost_threshold: float,
    temperature: float,
) -> Dict[str, Any]:
    """Independently derive v12 A without reading the incumbent support cost."""
    _require(math.isfinite(temperature) and temperature > 0.0, "support temperature must be positive")
    canonical_gate_actions = tuple(
        _canonical_action_id(action, "gate_actions") for action in gate_actions
    )
    canonical_hold = _canonical_action_id(hold_action, "hold_action")
    alternatives = tuple(
        action for action in canonical_gate_actions if action != canonical_hold
    )
    _require(bool(alternatives), "v12 A requires at least one alternative action")
    _require(set(alternatives) <= set(maneuver_families), "maneuver-family mapping is incomplete")
    all_families = tuple(sorted({str(maneuver_families[action]) for action in alternatives}))
    _require(all(value.strip() for value in all_families), "maneuver-family mapping has an empty family")
    raw_feasible = tuple(
        action
        for action in alternatives
        if action in recovery_costs and recovery_costs[action] <= viable_cost_threshold
    )
    family_minima: Dict[str, float] = {}
    for action in raw_feasible:
        _require(action in support_costs, f"raw-feasible action {action} lacks support cost")
        family = str(maneuver_families[action])
        current = family_minima.get(family)
        cost = float(support_costs[action])
        family_minima[family] = cost if current is None else min(current, cost)
    if family_minima:
        support_best = min(family_minima.values())
        effective_mass = sum(
            math.exp(-(cost - support_best) / temperature)
            for cost in family_minima.values()
        )
    else:
        support_best = 1.0
        effective_mass = 0.0
    breadth = effective_mass / len(all_families)
    return {
        "value": float(breadth),
        "all_alternative_families": all_families,
        "raw_feasible_alternative_actions": raw_feasible,
        "raw_feasible_family_min_support_costs": dict(sorted(family_minima.items())),
        "relative_support_best_cost": support_best,
        "relative_support_effective_mass": float(effective_mass),
        "temperature": float(temperature),
        "formula_id": "family_min_support_cost_relative_to_best_exponential_mass_over_all_alternative_families",
        "trace_formula": "sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families",
    }


def _grid_units(spec: Mapping[str, Any], component: str) -> Tuple[int, ...]:
    row = dict((spec.get("lambda_grid", {}) or {}).get(component, {}) or {})
    start = int(round(100 * float(row["start"])))
    stop = int(round(100 * float(row["stop"])))
    step = int(round(100 * float(row["step"])))
    _require(step > 0 and start >= 0 and stop <= 100 and start <= stop, f"invalid {component} grid")
    values = tuple(range(start, stop + 1, step))
    _require(values and values[-1] == stop, f"{component} grid does not land on its stop")
    return values


@dataclass(frozen=True)
class CalibrationSpec:
    lock_id: str
    method_version: str
    seeds: Tuple[int, ...]
    forbidden_seed_ranges: Tuple[Tuple[int, int], ...]
    delay_steps: Tuple[int, ...]
    delay_seconds: Tuple[float, ...]
    policy_frequency_hz: float
    horizon_steps: int
    gamma: float
    epsilon: float
    budget: int
    cooldown_complete_frames: int
    minimum_query_frame_gap: int
    floor_units: Tuple[int, ...]
    i_floor_units: int
    exposure_min: int
    exposure_max: int
    min_changed_opportunities: int
    min_changed_seeds: int
    min_changed_union_fraction: Fraction
    max_changed_jaccard: Fraction
    min_component_levels: int
    min_component_spread: float
    raw_lock: Mapping[str, Any]

    @classmethod
    def from_lock(cls, payload: Mapping[str, Any]) -> "CalibrationSpec":
        method = str(payload.get("method_version", "") or "")
        _require(method == METHOD_VERSION, "calibration lock method_version drift")
        _require(payload.get("artifact_role") == "calibration_lock", "lock artifact role drift")
        seed_block = dict(payload.get("seed_block", {}) or {})
        start = int(seed_block.get("start", -1))
        end = int(seed_block.get("end", -2))
        seeds = tuple(range(start, end + 1))
        _require(len(seeds) == int(seed_block.get("count", -1)), "lock seed count drift")
        _require(seeds == tuple(range(2000, 2040)), "parameter calibration must use seeds 2000-2039")
        forbidden = tuple(
            (int(item[0]), int(item[1]))
            for item in list(payload.get("forbidden_seed_ranges", []) or [])
        )
        for seed in seeds:
            _require(
                not any(low <= seed <= high for low, high in forbidden),
                f"calibration seed {seed} overlaps a forbidden partition",
            )
        frequency = float(payload["policy_frequency_hz"])
        delays = tuple(float(item) for item in payload["delay_strata_s"])
        delay_steps = tuple(int(math.ceil(item * frequency)) for item in delays)
        _require(len(set(delay_steps)) == len(delay_steps), "delay strata collapse to duplicate steps")
        grid_l = _grid_units(payload, "L")
        _require(grid_l == _grid_units(payload, "A") == _grid_units(payload, "H"), "L/A/H grids drift")
        i_row = dict((payload.get("lambda_grid", {}) or {}).get("I", {}) or {})
        _require(i_row.get("outcome_tuned") is False, "I must not be outcome-tuned")
        exposure = dict(payload.get("full_scheduled_exposure_per_delay", {}) or {})
        _require(exposure.get("inclusive") is True, "exposure interval must be inclusive")
        identity = dict(payload.get("identifiability_constraints", {}) or {})
        gap = int(payload["minimum_query_frame_gap"])
        cooldown = int(payload["cooldown_complete_frames"])
        _require(gap == cooldown + 1, "online cooldown semantics require gap=cooldown+1")
        return cls(
            lock_id=str(payload["lock_id"]),
            method_version=method,
            seeds=seeds,
            forbidden_seed_ranges=forbidden,
            delay_steps=delay_steps,
            delay_seconds=delays,
            policy_frequency_hz=frequency,
            horizon_steps=int(payload["horizon_steps"]),
            gamma=float(payload["gamma"]),
            epsilon=float(payload["corrective_margin"]),
            budget=int(payload["budget_per_seed_delay"]),
            cooldown_complete_frames=cooldown,
            minimum_query_frame_gap=gap,
            floor_units=grid_l,
            i_floor_units=int(round(100 * float(i_row["fixed"]))),
            exposure_min=int(exposure["minimum"]),
            exposure_max=int(exposure["maximum"]),
            min_changed_opportunities=int(identity["minimum_leave_one_out_changed_opportunities"]),
            min_changed_seeds=int(identity["minimum_leave_one_out_changed_seeds"]),
            min_changed_union_fraction=Fraction(str(identity["minimum_changed_fraction_of_union"])),
            max_changed_jaccard=Fraction(str(identity["maximum_pairwise_changed_set_jaccard"])),
            min_component_levels=int(identity["minimum_observed_levels_per_component"]),
            min_component_spread=float(identity["minimum_q90_minus_q10"]),
            raw_lock=dict(payload),
        )


def load_spec(path: Path = LOCK_PATH) -> CalibrationSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibrationSpec.from_lock(payload)


def validate_protocol_contract(path: Path, spec: CalibrationSpec) -> Dict[str, Any]:
    _require(path.is_file(), f"missing independent v12 protocol: {path}")
    _require(path.name == str(spec.raw_lock["protocol_file"]), "locked protocol filename drift")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), "v12 protocol is not a mapping")
    submission = dict(payload.get("tvt_submission_contract", {}) or {})
    calibration = dict(submission.get("v12_calibration", {}) or {})
    component = dict(submission.get("component_ablation", {}) or {})
    _require(submission.get("rgd_method_version") == spec.method_version, "protocol method_version drift")

    def expected_range(field: str, seeds: Sequence[int]) -> None:
        row = dict(calibration.get(field, {}) or {})
        _require(
            (int(row.get("start", -1)), int(row.get("end", -2)), int(row.get("count", -1)))
            == (seeds[0], seeds[-1], len(seeds)),
            f"protocol {field} drift",
        )

    expected_range("parameter_selection_seed_range", spec.seeds)
    expected_range("fixed_parameter_go_no_go_seed_range", tuple(range(2040, 2060)))
    expected_range("confirmatory_holdout_seed_range", tuple(range(3000, 3030)))
    _require(tuple(map(float, calibration.get("delay_strata_s", []) or [])) == spec.delay_seconds, "protocol delay strata drift")
    floor_grid = dict(calibration.get("floor_grid", {}) or {})
    for field in ("latency_survival", "maneuver_family_breadth", "corrective_recovery_headroom"):
        row = dict(floor_grid.get(field, {}) or {})
        observed = (
            int(round(100 * float(row.get("start", -1)))),
            int(round(100 * float(row.get("end", -1)))),
            int(round(100 * float(row.get("step", -1)))),
        )
        expected = (spec.floor_units[0], spec.floor_units[-1], spec.floor_units[1] - spec.floor_units[0])
        _require(observed == expected, f"protocol floor grid drift for {field}")
    state_need = dict(floor_grid.get("state_need", {}) or {})
    _require(int(round(100 * float(state_need.get("fixed", -1)))) == spec.i_floor_units, "protocol fixed I drift")
    exposure = dict(calibration.get("scheduled_exposure_per_delay", {}) or {})
    expected_exposure = {
        "calibration_40_seeds": (spec.exposure_min, spec.exposure_max),
        "go_no_go_20_seeds": (60, 100),
        "holdout_30_seeds": (90, 150),
    }
    for field, expected in expected_exposure.items():
        row = dict(exposure.get(field, {}) or {})
        _require((int(row.get("min", -1)), int(row.get("max", -1))) == expected, f"protocol exposure drift for {field}")
    identity = dict(calibration.get("identifiability_constraints", {}) or {})
    _require(int(identity.get("minimum_component_levels", -1)) == spec.min_component_levels, "protocol component-level constraint drift")
    _require(math.isclose(float(identity.get("minimum_component_q90_minus_q10", -1)), spec.min_component_spread, abs_tol=1e-12), "protocol component-spread constraint drift")
    _require(int(identity.get("minimum_leave_one_out_changed_opportunities", -1)) == spec.min_changed_opportunities, "protocol changed-opportunity constraint drift")
    _require(int(identity.get("minimum_leave_one_out_seed_coverage", -1)) == spec.min_changed_seeds, "protocol seed-coverage constraint drift")
    _require(Fraction(str(identity.get("minimum_changed_union_fraction"))) == spec.min_changed_union_fraction, "protocol changed-fraction constraint drift")
    _require(Fraction(str(identity.get("maximum_leave_one_out_change_set_jaccard"))) == spec.max_changed_jaccard, "protocol Jaccard constraint drift")
    _require(int(calibration.get("holdout_open_count", -1)) == 0, "protocol holdout was already opened")
    _require(calibration.get("holdout_recalibration_policy") == "forbidden", "protocol holdout recalibration policy drift")
    _require(int(component.get("budget", -1)) == spec.budget, "protocol budget drift")
    _require(int(component.get("cooldown_frames", -1)) == spec.cooldown_complete_frames, "protocol cooldown drift")
    _require(int(component.get("cooldown_minimum_query_frame_gap", -1)) == spec.minimum_query_frame_gap, "protocol cooldown gap drift")
    _require(component.get("alternative_metric_source") == spec.raw_lock["component_sources"]["A"], "protocol A source drift")
    _require(
        math.isclose(
            float(component.get("support_breadth_temperature", float("nan"))),
            float(spec.raw_lock["support_breadth_temperature"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "protocol support-breadth temperature drift",
    )
    _require(
        component.get("support_breadth_formula") == spec.raw_lock["support_breadth_formula"],
        "protocol support-breadth formula drift",
    )
    _require(component.get("headroom_metric_source") == spec.raw_lock["component_sources"]["H"], "protocol H source drift")
    _require(component.get("need_metric_source") == spec.raw_lock["component_sources"]["I"], "protocol I source drift")
    _require(
        math.isclose(
            float(component.get("viable_cost_threshold", float("nan"))),
            float(spec.raw_lock["viable_cost_threshold"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "protocol viable-cost threshold drift",
    )
    _require(
        math.isclose(
            float(component.get("corrective_headroom_kappa", float("nan"))),
            float(spec.raw_lock["corrective_headroom_kappa"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "protocol headroom kappa drift",
    )
    _require(
        component.get("corrective_headroom_kappa_source")
        == spec.raw_lock["corrective_headroom_kappa_source"],
        "protocol headroom kappa source drift",
    )
    _require(component.get("need_formula") == spec.raw_lock["need_formula"], "protocol need formula drift")
    latency_contract = dict(spec.raw_lock["latency_contract"])
    _require(
        math.isclose(
            float(component.get("latency_policy_frequency_hz", float("nan"))),
            float(latency_contract["policy_frequency_hz"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "protocol latency frequency drift",
    )
    _require(
        math.isclose(
            float(component.get("latency_safety_reserve_s", float("nan"))),
            float(latency_contract["safety_reserve_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "protocol latency reserve drift",
    )
    _require(
        component.get("latency_execution_available_required")
        is bool(latency_contract["llm_backed_execution_available"]),
        "protocol latency execution-availability drift",
    )
    acceptance = dict(component.get("paper_acceptance", {}) or {})
    _require(math.isclose(float(acceptance.get("corrective_yield_noninferiority_margin_per_seed", 0.0)), -0.10, abs_tol=1e-12), "protocol yield noninferiority margin drift")
    return dict(payload)


def locked_partition_spec(spec: CalibrationSpec, partition: str) -> CalibrationSpec:
    contracts = {
        "go_no_go": ("go_no_go_seed_block", tuple(range(2040, 2060))),
        "validation": ("validation_seed_block", tuple(range(3000, 3030))),
    }
    _require(partition in contracts, f"unknown locked partition {partition!r}")
    field, expected = contracts[partition]
    block = dict(spec.raw_lock.get(field, {}) or {})
    start = int(block.get("start", -1))
    end = int(block.get("end", -2))
    seeds = tuple(range(start, end + 1))
    _require(len(seeds) == int(block.get("count", -1)), f"{partition} seed count drift")
    _require(seeds == expected, f"v12 {partition} seed block drift")
    for seed in seeds:
        _require(
            not any(low <= seed <= high for low, high in spec.forbidden_seed_ranges),
            f"{partition} seed {seed} overlaps a forbidden partition",
        )
    return replace(spec, seeds=seeds)


@dataclass(frozen=True, order=True)
class Thresholds:
    l: int  # noqa: E741 - conventional component label L
    a: int
    h: int
    i: int

    @property
    def candidate_id(self) -> str:
        return f"L{self.l:02d}_A{self.a:02d}_H{self.h:02d}_I{self.i:02d}"

    def as_floats(self) -> Dict[str, float]:
        return {
            "lambda_L": self.l / 100.0,
            "lambda_A": self.a / 100.0,
            "lambda_H": self.h / 100.0,
            "lambda_I": self.i / 100.0,
        }


def candidates(spec: CalibrationSpec) -> Tuple[Thresholds, ...]:
    values = tuple(
        Thresholds(lambda_l, lambda_a, lambda_h, spec.i_floor_units)
        for lambda_l, lambda_a, lambda_h in product(spec.floor_units, repeat=3)
    )
    _require(len(values) == len(set(values)), "candidate registry contains duplicate tuples")
    return values


@dataclass(frozen=True)
class FrameFeature:
    seed: int
    frame: int
    episode_frames: int
    permanent_f: bool
    a: float
    h: float
    i: float
    latency_snapshot: Mapping[str, Any]
    schema_version: str
    source_signature: Tuple[str, ...]


def _gate_and_profile(record: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diag = dict(record.get("rgd_subordinate_diagnostics", {}) or {})
    gate = dict(
        ((diag.get("recoverability_signal", {}) or {}).get("recoverability_gate", {}) or {})
    )
    profile = dict(
        ((diag.get("ambiguity_and_conflict", {}) or {}).get("route_ambiguity_profile", {}) or {})
    )
    if not gate:
        raise ValueError("v12 reasoning record misses recoverability_gate")
    return gate, profile


def parse_v12_record(
    record: Mapping[str, Any],
    *,
    seed: int,
    episode_frames: int,
) -> FrameFeature:
    gate, profile = _gate_and_profile(record)
    method = str(
        _same_alias(
            (
                ("record.rgd_method_version", record.get("rgd_method_version")),
                ("gate.method_version", gate.get("method_version")),
                ("gate.rgd_method_version", gate.get("rgd_method_version")),
            ),
            "method_version",
        )
    )
    _require(method == METHOD_VERSION, f"trace method_version is {method!r}, not v12")
    frame = int(record["frame_id"])

    gate_actions = _action_sequence(
        _same_alias(
            (
                ("gate.gate_action_universe", gate.get("gate_action_universe")),
                ("gate.gate_legal_actions", gate.get("gate_legal_actions")),
            ),
            "exact gate action universe",
        ),
        "gate_action_universe",
    )
    fast_actions = _action_sequence(
        _same_alias(
            (
                ("gate.fast_executor_action_universe", gate.get("fast_executor_action_universe")),
                ("gate.fast_legal_actions", gate.get("fast_legal_actions")),
            ),
            "exact fast action universe",
        ),
        "fast_executor_action_universe",
    )
    hold_action = _canonical_action_id(gate["hold_action"], "hold_action")
    predicted_action = _canonical_action_id(
        record.get("predicted_action_id"), "predicted_action_id"
    )
    _require(
        hold_action == predicted_action,
        "v12 hold action differs from the complete Fast proposal",
    )
    gate_domain_valid = _as_bool(gate.get("gate_domain_valid"), "gate_domain_valid")
    gate_fail_closed = _as_bool(gate.get("gate_fail_closed"), "gate_fail_closed")
    _require(gate_domain_valid != gate_fail_closed, "v12 domain/fail-closed flags are incoherent")
    raw_cost_complete_claim = _as_bool(gate.get("raw_cost_complete"), "raw_cost_complete")
    support_cost_complete_claim = _as_bool(
        gate.get("support_cost_complete"), "support_cost_complete"
    )

    recovery_costs = _cost_map(
        _same_alias(
            (
                ("gate.action_recovery_costs", gate.get("action_recovery_costs")),
                ("profile.action_recovery_costs", profile.get("action_recovery_costs")),
            ),
            "action_recovery_costs",
        ),
        "action_recovery_costs",
        fail_closed_record=bool(gate_fail_closed or not gate_domain_valid),
    )
    support_costs = _cost_map(
        _same_alias(
            (
                ("gate.action_support_ranking_costs", gate.get("action_support_ranking_costs")),
                ("profile.action_support_ranking_costs", profile.get("action_support_ranking_costs")),
            ),
            "action_support_ranking_costs",
        ),
        "action_support_ranking_costs",
        fail_closed_record=bool(gate_fail_closed or not gate_domain_valid),
    )
    raw_contract_valid = bool(
        gate_actions == fast_actions
        and hold_action in gate_actions
        and set(gate_actions) == set(recovery_costs)
        and raw_cost_complete_claim
    )
    if gate_domain_valid:
        _require(gate_actions == fast_actions, "valid v12 gate/fast action universes differ")
        _require(hold_action in gate_actions, "valid v12 hold action is outside exact universe")
        _require(raw_contract_valid, "valid v12 gate lacks an exact raw contract")

    _require("viable_cost_threshold" in gate, "v12 trace misses viable_cost_threshold")
    viable_threshold = _unit(gate["viable_cost_threshold"], "viable_cost_threshold")
    _require(
        math.isclose(viable_threshold, 0.55, rel_tol=0.0, abs_tol=1e-12),
        "v12 viable-cost threshold drift",
    )
    absolute_count = int(gate["absolute_alternative_count"])
    absolute_feasible = _as_bool(
        gate.get("absolute_alternative_feasible"), "absolute_alternative_feasible"
    )
    permanent_f = bool(
        raw_contract_valid
        and absolute_feasible
        and absolute_count >= 1
    )
    if raw_contract_valid:
        derived_absolute = sum(
            action != hold_action and recovery_costs[action] <= viable_threshold
            for action in gate_actions
        )
        _require(absolute_count == derived_absolute, "v12 absolute feasibility count is not derivable")
        _require(absolute_feasible == bool(derived_absolute), "v12 absolute feasibility flag is inconsistent")

    raw_mapping = gate.get("action_maneuver_family_mapping")
    _require(isinstance(raw_mapping, Mapping), "v12 trace misses action_maneuver_family_mapping")
    maneuver_families: Dict[int, str] = {}
    for raw_action, raw_family in raw_mapping.items():
        action = _canonical_action_id(
            raw_action, "action_maneuver_family_mapping", allow_json_key=True
        )
        _require(action not in maneuver_families, f"duplicate maneuver-family action {action}")
        maneuver_families[action] = str(raw_family).strip()
    _require(all(maneuver_families.values()), "v12 maneuver-family mapping has empty values")
    for action, family in maneuver_families.items():
        _require(
            family == CANONICAL_ACTION_FAMILIES[action],
            f"v12 maneuver-family mapping drift for action {action}",
        )
    temperature = float(gate.get("support_breadth_temperature", float("nan")))
    _require(
        math.isfinite(temperature) and math.isclose(temperature, 0.10, rel_tol=0.0, abs_tol=1e-12),
        "v12 support-breadth temperature must equal 0.10",
    )
    _require(
        str(gate.get("support_breadth_temperature_source", "") or "")
        == "identifiable_gate_v12.fixed_T_A",
        "v12 support-breadth temperature source drift",
    )
    _require(
        str(gate.get("support_breadth_formula", "") or "")
        == "sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families",
        "v12 support-breadth trace formula drift",
    )
    support_contract_valid = bool(
        raw_contract_valid
        and set(maneuver_families) == set(gate_actions)
        and set(support_costs) == set(gate_actions)
        and support_cost_complete_claim
    )
    if gate_domain_valid:
        _require(gate_actions == fast_actions, "valid v12 gate/fast action universes differ")
        _require(hold_action in gate_actions, "valid v12 hold action is outside exact universe")
        _require(raw_contract_valid, "valid v12 gate lacks an exact raw contract")

    exported_a = _unit(
        _same_alias(
            (
                (
                    "gate.relative_support_weighted_maneuver_family_breadth",
                    gate.get("relative_support_weighted_maneuver_family_breadth"),
                ),
                ("gate.support_effective_breadth", gate.get("support_effective_breadth")),
                ("gate.alternative_viable_ratio", gate.get("alternative_viable_ratio")),
                ("gate.alternative_support_ratio", gate.get("alternative_support_ratio")),
            ),
            "A support-effective breadth",
        ),
        "A",
    )
    a = exported_a if support_contract_valid else 0.0
    if support_contract_valid:
        derived_a = derive_relative_support_maneuver_breadth(
            gate_actions=gate_actions,
            hold_action=hold_action,
            recovery_costs=recovery_costs,
            support_costs=support_costs,
            maneuver_families=maneuver_families,
            viable_cost_threshold=viable_threshold,
            temperature=temperature,
        )
        _require(
            math.isclose(exported_a, float(derived_a["value"]), rel_tol=0.0, abs_tol=1e-12),
            "v12 A is not independently derivable from family-min support costs",
        )
        observed_family_minima = gate.get("support_family_min_costs")
        _require(isinstance(observed_family_minima, Mapping), "v12 support-family minima are missing")
        normalized_family_minima = {
            str(family): float(cost) for family, cost in observed_family_minima.items()
        }
        _require(
            set(normalized_family_minima)
            == set(derived_a["raw_feasible_family_min_support_costs"]),
            "v12 support-family minimum domain drift",
        )
        for family, expected_cost in derived_a["raw_feasible_family_min_support_costs"].items():
            _require(
                math.isclose(
                    normalized_family_minima[family], expected_cost, rel_tol=0.0, abs_tol=1e-12
                ),
                f"v12 support-family minimum drift for {family}",
            )
        _require(
            math.isclose(
                float(gate.get("support_best_family_cost", float("nan"))),
                float(derived_a["relative_support_best_cost"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "v12 relative support best-family cost drift",
        )
        _require(
            math.isclose(
                float(gate.get("support_weighted_family_mass", float("nan"))),
                float(derived_a["relative_support_effective_mass"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "v12 support-weighted family mass drift",
        )
        if gate.get("raw_feasible_alternative_actions") is not None:
            observed_actions = tuple(
                sorted(
                    _canonical_action_id(action, "raw_feasible_alternative_actions")
                    for action in gate["raw_feasible_alternative_actions"]
                )
            )
            _require(
                observed_actions == derived_a["raw_feasible_alternative_actions"],
                "v12 raw-feasible alternative action provenance drift",
            )
        if gate.get("alternative_maneuver_family_total") is not None:
            _require(
                int(gate["alternative_maneuver_family_total"])
                == len(derived_a["all_alternative_families"]),
                "v12 alternative-family denominator drift",
            )
        if gate.get("alternative_maneuver_family_count") is not None:
            _require(
                int(gate["alternative_maneuver_family_count"])
                == len(derived_a["raw_feasible_family_min_support_costs"]),
                "v12 raw-feasible family count drift",
            )
    exported_h = _unit(
        _same_alias(
            (
                ("gate.corrective_recovery_headroom", gate.get("corrective_recovery_headroom")),
                ("gate.relative_corrective_headroom", gate.get("relative_corrective_headroom")),
                ("gate.cost_headroom", gate.get("cost_headroom")),
            ),
            "H corrective recovery headroom",
        ),
        "H",
    )
    _require("corrective_headroom_kappa" in gate, "v12 trace misses H normalization")
    headroom_kappa = _unit(gate["corrective_headroom_kappa"], "corrective_headroom_kappa")
    _require(
        math.isclose(headroom_kappa, viable_threshold, rel_tol=0.0, abs_tol=1e-12),
        "v12 H normalization drift",
    )
    _require(
        str(gate.get("corrective_headroom_kappa_source", "") or "")
        == "recoverable_cost_threshold",
        "v12 H normalization source drift",
    )
    if raw_contract_valid:
        feasible_alternatives = tuple(
            action
            for action in gate_actions
            if action != hold_action and recovery_costs[action] <= viable_threshold
        )
        best_alternative_cost = (
            min(recovery_costs[action] for action in feasible_alternatives)
            if feasible_alternatives
            else 1.0
        )
        corrective_advantage = (
            max(0.0, recovery_costs[hold_action] - best_alternative_cost)
            if feasible_alternatives
            else 0.0
        )
        h = min(1.0, corrective_advantage / headroom_kappa)
        _require(
            math.isclose(exported_h, h, rel_tol=0.0, abs_tol=1e-12),
            "v12 H is not independently derivable from the Fast incumbent and raw costs",
        )
        _require(
            math.isclose(
                float(gate.get("corrective_advantage_raw", float("nan"))),
                corrective_advantage,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "v12 H raw advantage drift",
        )
    else:
        h = 0.0

    exported_i = _unit(
        _same_alias(
            (
                ("gate.state_hazard_need", gate.get("state_hazard_need")),
                ("gate.need_score", gate.get("need_score")),
            ),
            "I state-hazard need",
        ),
        "I",
    )
    need_state_hazard = _unit(gate.get("need_state_hazard"), "need_state_hazard")
    need_pre_screen_hazard = _unit(
        gate.get("need_pre_screen_hazard"), "need_pre_screen_hazard"
    )
    i_value = max(need_state_hazard, need_pre_screen_hazard)
    _require(
        math.isclose(exported_i, i_value, rel_tol=0.0, abs_tol=1e-12),
        "v12 I is not independently derivable from state and pre-screen hazards",
    )

    latency = dict(gate.get("latency", {}) or {})
    for key in (
        "critical_latency_seconds",
        "policy_frequency",
        "latency_prediction_available",
        "llm_backed_execution_available",
        "latency_source",
        "safety_reserve_seconds",
        "effective_delay_steps",
        "recovery_window",
        "latency_survival_by_delay_steps",
        "recovery_window_by_delay_steps",
    ):
        if key in gate and key not in latency:
            latency[key] = gate[key]
    if "source" not in latency and "latency_source" in latency:
        latency["source"] = latency["latency_source"]
    required_latency = {
        "critical_latency_seconds",
        "policy_frequency",
        "latency_prediction_available",
        "llm_backed_execution_available",
        "source",
        "safety_reserve_seconds",
        "effective_delay_steps",
    }
    missing_latency = sorted(required_latency - set(latency))
    _require(not missing_latency, f"v12 complete latency snapshot missing {missing_latency}")
    _require(
        isinstance(latency["source"], str) and bool(latency["source"].strip()),
        "v12 latency source is empty",
    )
    _as_bool(latency["latency_prediction_available"], "latency_prediction_available")
    _require(
        math.isclose(float(latency["policy_frequency"]), 10.0, rel_tol=0.0, abs_tol=1e-12),
        "v12 latency policy frequency drift",
    )
    _require(
        math.isclose(float(latency["safety_reserve_seconds"]), 0.0, rel_tol=0.0, abs_tol=1e-12),
        "v12 latency safety reserve drift",
    )
    _require(
        _as_bool(
            latency["llm_backed_execution_available"],
            "llm_backed_execution_available",
        )
        is True,
        "v12 latency executor availability drift",
    )
    effective_delay_steps = latency["effective_delay_steps"]
    _require(
        isinstance(effective_delay_steps, int)
        and not isinstance(effective_delay_steps, bool)
        and effective_delay_steps >= 0,
        "v12 latency effective delay steps must be a nonnegative integer",
    )

    source_signature = (
        str(gate.get("gate_action_universe_source", "missing") or "missing"),
        str(gate.get("fast_executor_action_universe_source", "missing") or "missing"),
        str(gate.get("alternative_metric_source", "missing") or "missing"),
        str(gate.get("headroom_metric_source", "missing") or "missing"),
        str(gate.get("need_metric_source", "missing") or "missing"),
        str(profile.get("probability_cost_source", gate.get("probability_cost_source", "missing")) or "missing"),
    )
    _require(all(value != "missing" for value in source_signature), "v12 source provenance is incomplete")
    _require(
        source_signature[2] == "relative_support_weighted_maneuver_family_breadth",
        "v12 A source is not relative support-weighted maneuver-family breadth",
    )
    _require(
        source_signature[3] == "incumbent_relative_action_recovery_cost_margin",
        "v12 H source drift",
    )
    _require(source_signature[4] == "state_hazard_and_pre_screen_only", "v12 I source drift")
    _require(source_signature[5] == "action_support_ranking_costs", "v12 support provenance drift")
    return FrameFeature(
        seed=int(seed),
        frame=frame,
        episode_frames=int(episode_frames),
        permanent_f=permanent_f,
        a=a,
        h=h,
        i=i_value,
        latency_snapshot=latency,
        schema_version=str(record.get("schema_version", "missing") or "missing"),
        source_signature=source_signature,
    )


def latency_survival(feature: FrameFeature, delay_steps: int, spec: CalibrationSpec) -> float:
    latency = dict(feature.latency_snapshot)
    for map_name in ("latency_survival_by_delay_steps", "recovery_window_by_delay_steps"):
        mapping = latency.get(map_name)
        if isinstance(mapping, Mapping):
            raw = mapping.get(str(delay_steps), mapping.get(delay_steps))
            if raw is not None:
                return _unit(raw, f"{map_name}[{delay_steps}]")
    exported_steps = latency.get("effective_delay_steps")
    exported_value = latency.get("recovery_window", latency.get("latency_survival"))
    if exported_steps is not None and int(exported_steps) == int(delay_steps) and exported_value is not None:
        return _unit(exported_value, "exported latency survival")
    required = ("critical_latency_seconds", "latency_prediction_available", "source")
    missing = [field for field in required if field not in latency]
    if missing:
        raise ValueError(
            f"v12 latency snapshot cannot reconstruct delay_steps={delay_steps}; missing {missing}"
        )
    frequency = float(latency["policy_frequency"])
    if not math.isclose(frequency, spec.policy_frequency_hz, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("trace policy frequency differs from calibration lock")
    return float(
        compute_temporal_survival(
            critical_latency_seconds=float(latency["critical_latency_seconds"]),
            effective_delay_steps=int(delay_steps),
            policy_frequency=frequency,
            latency_prediction_available=_as_bool(
                latency["latency_prediction_available"], "latency_prediction_available"
            ),
            execution_available=_as_bool(
                latency["llm_backed_execution_available"],
                "llm_backed_execution_available",
            ),
            latency_source=str(latency["source"]),
            safety_reserve_seconds=float(latency["safety_reserve_seconds"]),
        )
    )


def _trace_path(root: Path, seed: int) -> Path:
    settings = (
        root / "always_fast" / "highway" / f"seed_{seed}",
        root / "always_fast" / f"always_fast_latency_1p7s_seed_{seed}",
    )
    matches: List[Path] = []
    for setting in settings:
        matches.extend(setting.glob(f"ep_{seed}/highway_{seed}_reasoning_records.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"seed {seed}: expected exactly one v12 reasoning trace under the run root, "
            f"found {len(matches)}"
        )
    return matches[0]


def load_features(
    trace_root: Path,
    spec: CalibrationSpec,
) -> Tuple[Dict[int, Tuple[FrameFeature, ...]], Dict[str, Any]]:
    traces: Dict[int, Tuple[FrameFeature, ...]] = {}
    files: List[Dict[str, Any]] = []
    for seed in spec.seeds:
        path = _trace_path(trace_root, seed)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        records = list(payload.get("analysis_records", []) or [])
        _require(records, f"seed {seed}: empty analysis_records")
        observed_frames = [int(row["frame_id"]) for row in records]
        _require(
            observed_frames == list(range(len(records))),
            f"seed {seed}: frame_id must be contiguous from zero",
        )
        parsed = tuple(
            parse_v12_record(row, seed=seed, episode_frames=len(records)) for row in records
        )
        traces[seed] = parsed
        files.append({"seed": seed, "sha256": _sha256(path), "name": path.name})
    _require(tuple(sorted(traces)) == spec.seeds, "trace seed inventory differs from lock")
    semantic = [
        {
            "seed": row.seed,
            "frame": row.frame,
            "episode_frames": row.episode_frames,
            "F": row.permanent_f,
            "A": _canonical_float(row.a),
            "H": _canonical_float(row.h),
            "I": _canonical_float(row.i),
            "L": {
                str(step): _canonical_float(latency_survival(row, step, spec))
                for step in spec.delay_steps
            },
            "schema_version": row.schema_version,
            "sources": list(row.source_signature),
        }
        for seed in spec.seeds
        for row in traces[seed]
    ]
    return traces, {
        "files": files,
        "raw_file_set_hash": _semantic_hash(files),
        "semantic_trace_hash": _semantic_hash(semantic),
        "schema_versions": sorted({row.schema_version for rows in traces.values() for row in rows}),
        "source_signatures": sorted({row.source_signature for rows in traces.values() for row in rows}),
    }


@dataclass(frozen=True)
class Opportunity:
    index: int
    seed: int
    delay_steps: int
    delay_s: float
    query_frame: int
    release_frame: int
    episode_frames: int
    evaluable: bool

    @property
    def event_key(self) -> Tuple[int, int, int, int]:
        return (self.seed, self.delay_steps, self.query_frame, self.release_frame)


@dataclass
class OpportunityTable:
    rows: Tuple[Opportunity, ...]
    permanent_f: np.ndarray
    l: np.ndarray  # noqa: E741 - conventional component label L
    a: np.ndarray
    h: np.ndarray
    i: np.ndarray
    groups: Mapping[Tuple[int, int], np.ndarray]


def build_opportunity_table(
    traces: Mapping[int, Sequence[FrameFeature]],
    spec: CalibrationSpec,
) -> OpportunityTable:
    rows: List[Opportunity] = []
    f_values: List[bool] = []
    l_values: List[float] = []
    a_values: List[float] = []
    h_values: List[float] = []
    i_values: List[float] = []
    groups: Dict[Tuple[int, int], List[int]] = {}
    seconds_by_step = dict(zip(spec.delay_steps, spec.delay_seconds))
    for seed in spec.seeds:
        for step in spec.delay_steps:
            group: List[int] = []
            for feature in traces[seed]:
                index = len(rows)
                release = feature.frame + int(step)
                rows.append(
                    Opportunity(
                        index=index,
                        seed=seed,
                        delay_steps=step,
                        delay_s=seconds_by_step[step],
                        query_frame=feature.frame,
                        release_frame=release,
                        episode_frames=feature.episode_frames,
                        evaluable=bool(release + spec.horizon_steps <= feature.episode_frames),
                    )
                )
                f_values.append(feature.permanent_f)
                l_values.append(latency_survival(feature, step, spec))
                a_values.append(feature.a)
                h_values.append(feature.h)
                i_values.append(feature.i)
                group.append(index)
            groups[(seed, step)] = group
    return OpportunityTable(
        rows=tuple(rows),
        permanent_f=np.asarray(f_values, dtype=bool),
        l=np.asarray(l_values, dtype=np.float64),
        a=np.asarray(a_values, dtype=np.float64),
        h=np.asarray(h_values, dtype=np.float64),
        i=np.asarray(i_values, dtype=np.float64),
        groups={key: np.asarray(value, dtype=np.int32) for key, value in groups.items()},
    )


def component_variation(table: OpportunityTable, spec: CalibrationSpec) -> Dict[str, Any]:
    base = table.permanent_f & (table.i >= spec.i_floor_units / 100.0)
    diagnostics: Dict[str, Any] = {}
    for name, values in (("L", table.l), ("A", table.a), ("H", table.h)):
        observed = values[base]
        _require(len(observed) > 0, f"no permanent-F observations for component {name}")
        q10, q90 = np.quantile(observed, [0.10, 0.90])
        levels = len(set(round(float(value), 12) for value in observed.tolist()))
        spread = float(q90 - q10)
        passed = bool(
            levels >= spec.min_component_levels
            and spread + 1e-12 >= spec.min_component_spread
        )
        diagnostics[name] = {
            "observed_levels": levels,
            "q10": float(q10),
            "q90": float(q90),
            "q90_minus_q10": spread,
            "minimum_levels": spec.min_component_levels,
            "minimum_spread": spec.min_component_spread,
            "passed": passed,
        }
    diagnostics["passed"] = all(diagnostics[name]["passed"] for name in ("L", "A", "H"))
    return diagnostics


def eligibility_mask(
    table: OpportunityTable,
    thresholds: Thresholds,
    arm: str,
) -> np.ndarray:
    _require(arm in ARMS, f"unknown v12 arm {arm}")
    mask = table.permanent_f & (table.i >= thresholds.i / 100.0)
    if arm != ARM_NO_L:
        mask = mask & (table.l >= thresholds.l / 100.0)
    if arm != ARM_NO_A:
        mask = mask & (table.a >= thresholds.a / 100.0)
    if arm != ARM_NO_H:
        mask = mask & (table.h >= thresholds.h / 100.0)
    return mask


def factorial_eligibility_mask(
    table: OpportunityTable,
    thresholds: Thresholds,
    arm: FactorialArm,
) -> np.ndarray:
    mask = table.permanent_f & (table.i >= thresholds.i / 100.0)
    if arm.use_l:
        mask = mask & (table.l >= thresholds.l / 100.0)
    if arm.use_a:
        mask = mask & (table.a >= thresholds.a / 100.0)
    if arm.use_h:
        mask = mask & (table.h >= thresholds.h / 100.0)
    return mask


@dataclass(frozen=True)
class Cohort:
    scheduled: Tuple[int, ...]
    evaluated: Tuple[int, ...]
    excluded: Tuple[int, ...]
    q_by_delay: Tuple[Tuple[int, int], ...]
    scheduled_hash: str
    evaluated_hash: str


def _cohort_hash(table: OpportunityTable, indices: Iterable[int]) -> str:
    keys = [table.rows[int(index)].event_key for index in indices]
    return _semantic_hash(sorted(keys))


def schedule_mask(
    table: OpportunityTable,
    mask: np.ndarray,
    spec: CalibrationSpec,
) -> Cohort:
    scheduled: List[int] = []
    q_by_delay: Dict[int, int] = {step: 0 for step in spec.delay_steps}
    for seed in spec.seeds:
        for step in spec.delay_steps:
            last: Optional[int] = None
            calls = 0
            for raw_index in table.groups[(seed, step)]:
                index = int(raw_index)
                if not bool(mask[index]):
                    continue
                frame = table.rows[index].query_frame
                if last is not None and frame - last < spec.minimum_query_frame_gap:
                    continue
                scheduled.append(index)
                q_by_delay[step] += 1
                calls += 1
                last = frame
                if calls >= spec.budget:
                    break
    evaluated = tuple(index for index in scheduled if table.rows[index].evaluable)
    excluded = tuple(index for index in scheduled if not table.rows[index].evaluable)
    scheduled_tuple = tuple(scheduled)
    return Cohort(
        scheduled=scheduled_tuple,
        evaluated=evaluated,
        excluded=excluded,
        q_by_delay=tuple((step, q_by_delay[step]) for step in spec.delay_steps),
        scheduled_hash=_cohort_hash(table, scheduled_tuple),
        evaluated_hash=_cohort_hash(table, evaluated),
    )


def _loo_cache_key(thresholds: Thresholds, arm: str) -> Tuple[Any, ...]:
    if arm == ARM_NO_L:
        return (arm, thresholds.a, thresholds.h, thresholds.i)
    if arm == ARM_NO_A:
        return (arm, thresholds.l, thresholds.h, thresholds.i)
    if arm == ARM_NO_H:
        return (arm, thresholds.l, thresholds.a, thresholds.i)
    raise ValueError(f"not a leave-one-out arm: {arm}")


def target_event_indices(table: OpportunityTable, spec: CalibrationSpec) -> Tuple[int, ...]:
    union: set[int] = set()
    loo_cache: Dict[Tuple[Any, ...], Cohort] = {}
    for threshold in candidates(spec):
        full = schedule_mask(table, eligibility_mask(table, threshold, ARM_FULL), spec)
        union.update(full.evaluated)
        for arm in (ARM_NO_L, ARM_NO_A, ARM_NO_H):
            key = _loo_cache_key(threshold, arm)
            cohort = loo_cache.get(key)
            if cohort is None:
                cohort = schedule_mask(table, eligibility_mask(table, threshold, arm), spec)
                loo_cache[key] = cohort
            union.update(cohort.evaluated)
    return tuple(sorted(union))


def target_payload(
    table: OpportunityTable,
    target_indices: Sequence[int],
    spec: CalibrationSpec,
) -> Tuple[Dict[str, List[int]], List[Dict[str, Any]]]:
    target_map: Dict[str, set[int]] = {str(seed): set() for seed in spec.seeds}
    events: List[Dict[str, Any]] = []
    for index in target_indices:
        row = table.rows[int(index)]
        target_map[str(row.seed)].add(row.release_frame)
        events.append(
            {
                "seed": row.seed,
                "delay_s": row.delay_s,
                "delay_steps": row.delay_steps,
                "query_frame": row.query_frame,
                "release_frame": row.release_frame,
                "candidate_state_id": f"{row.seed}:{row.query_frame}:{row.delay_steps}",
                "release_state_id": f"{row.seed}:{row.release_frame}",
            }
        )
    normalized = {seed: sorted(frames) for seed, frames in target_map.items()}
    events.sort(key=lambda row: (row["seed"], row["delay_steps"], row["query_frame"]))
    return normalized, events


def publish_target_bundle(
    output: Path,
    *,
    target_map: Mapping[str, Sequence[int]],
    events: Sequence[Mapping[str, Any]],
    partition: str,
    spec: CalibrationSpec,
    trace_meta: Mapping[str, Any],
    protocol_path: Path,
    lock_path: Path,
    candidate_contract: Mapping[str, Any],
    calibration_manifest_sha256: Optional[str] = None,
    go_no_go_manifest_sha256: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    _require(partition in {"calibration", "go_no_go", "validation"}, "target partition drift")
    events_path = output.with_suffix(".events.csv")
    manifest_path = output.with_suffix(".manifest.json")
    _write_json(output, target_map)
    _write_csv(events_path, events)
    manifest: Dict[str, Any] = {
        "schema": TARGET_MANIFEST_SCHEMA,
        "schema_version": 2,
        "artifact_role": "preregistered_snapshot_target_bundle",
        "partition": partition,
        "method_version": spec.method_version,
        "label_source_required": LABEL_SOURCE,
        "seed_block": {
            "start": spec.seeds[0],
            "end": spec.seeds[-1],
            "count": len(spec.seeds),
            "seeds": list(spec.seeds),
        },
        "delay_strata_s": list(spec.delay_seconds),
        "delay_steps": list(spec.delay_steps),
        "horizon_steps": spec.horizon_steps,
        "gamma": spec.gamma,
        "epsilon": spec.epsilon,
        "exact_action_provenance_required": True,
        "candidate_contract": dict(candidate_contract),
        "target_map": {
            "filename": output.name,
            "sha256": _sha256(output),
            "semantic_sha256": _semantic_hash(target_map),
            "seed_count": len(target_map),
            "unique_release_targets": sum(len(value) for value in target_map.values()),
        },
        "target_events": {
            "filename": events_path.name,
            "sha256": _sha256(events_path),
            "semantic_sha256": _semantic_hash(list(events)),
            "row_count": len(events),
            "event_key": ["seed", "delay_steps", "query_frame", "release_frame"],
        },
        "producer": {
            "name": "calibrate_identifiable_gate_v12.target_producer",
            "selector_sha256": _sha256(Path(__file__)),
            "gate_support_sha256": _sha256(GATE_SUPPORT_PATH),
        },
        "source": {
            "protocol_sha256": _sha256(protocol_path),
            "lock_sha256": _sha256(lock_path),
            "trace_semantic_hash": trace_meta["semantic_trace_hash"],
            "trace_raw_file_set_hash": trace_meta["raw_file_set_hash"],
            "trace_schema_versions": trace_meta["schema_versions"],
            "trace_source_signatures": trace_meta["source_signatures"],
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "go_no_go_manifest_sha256": go_no_go_manifest_sha256,
        },
    }
    manifest["manifest_payload_sha256"] = _payload_hash(
        manifest, "manifest_payload_sha256"
    )
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def _normalized_target_event(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "seed": int(row["seed"]),
        "delay_s": float(row["delay_s"]),
        "delay_steps": int(row["delay_steps"]),
        "query_frame": int(row["query_frame"]),
        "release_frame": int(row["release_frame"]),
        "candidate_state_id": str(row["candidate_state_id"]),
        "release_state_id": str(row["release_state_id"]),
    }


def validate_target_bundle(
    manifest_path: Path,
    *,
    expected_partition: str,
    spec: CalibrationSpec,
    trace_meta: Mapping[str, Any],
    expected_target_map: Mapping[str, Sequence[int]],
    expected_events: Sequence[Mapping[str, Any]],
    expected_candidate_contract: Mapping[str, Any],
    protocol_path: Path,
    lock_path: Path,
    calibration_manifest_sha256: Optional[str],
    go_no_go_manifest_sha256: Optional[str],
) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == TARGET_MANIFEST_SCHEMA, "target manifest v2 is required")
    _require(int(manifest.get("schema_version", -1)) == 2, "target manifest schema version drift")
    _require(
        manifest.get("artifact_role") == "preregistered_snapshot_target_bundle",
        "target manifest role drift",
    )
    _verify_payload_hash(manifest, "manifest_payload_sha256", "target manifest")
    _require(manifest.get("partition") == expected_partition, "target partition drift")
    _require(manifest.get("method_version") == spec.method_version, "target method drift")
    _require(manifest.get("label_source_required") == LABEL_SOURCE, "target label-source drift")
    _require(manifest.get("exact_action_provenance_required") is True, "target exact-action contract missing")
    seed_block = dict(manifest.get("seed_block", {}) or {})
    _require(tuple(seed_block.get("seeds", []) or []) == spec.seeds, "target seed block drift")
    _require(tuple(map(float, manifest.get("delay_strata_s", []) or [])) == spec.delay_seconds, "target delay strata drift")
    _require(tuple(map(int, manifest.get("delay_steps", []) or [])) == spec.delay_steps, "target delay steps drift")
    _require(int(manifest.get("horizon_steps", -1)) == spec.horizon_steps, "target horizon drift")
    _require(math.isclose(float(manifest.get("gamma", float("nan"))), spec.gamma, abs_tol=1e-12), "target gamma drift")
    _require(math.isclose(float(manifest.get("epsilon", float("nan"))), spec.epsilon, abs_tol=1e-12), "target epsilon drift")
    _require(manifest.get("candidate_contract") == dict(expected_candidate_contract), "target candidate contract drift")
    map_meta = dict(manifest.get("target_map", {}) or {})
    event_meta = dict(manifest.get("target_events", {}) or {})
    _require(Path(str(map_meta.get("filename", ""))).name == map_meta.get("filename"), "target map must be a sibling file")
    _require(Path(str(event_meta.get("filename", ""))).name == event_meta.get("filename"), "target events must be a sibling file")
    target_map_path = manifest_path.parent / str(map_meta.get("filename", ""))
    events_path = manifest_path.parent / str(event_meta.get("filename", ""))
    _require(target_map_path.is_file() and events_path.is_file(), "target bundle files are missing")
    _require(map_meta.get("sha256") == _sha256(target_map_path), "target map raw hash drift")
    _require(event_meta.get("sha256") == _sha256(events_path), "target events raw hash drift")
    observed_map = json.loads(target_map_path.read_text(encoding="utf-8"))
    observed_events = [_normalized_target_event(row) for row in _read_rows(events_path)]
    _require(observed_map == dict(expected_target_map), "target map differs from recomputed candidate union")
    _require(
        observed_events == [_normalized_target_event(row) for row in expected_events],
        "target events differ from recomputed candidate union",
    )
    _require(map_meta.get("semantic_sha256") == _semantic_hash(expected_target_map), "target map semantic hash drift")
    _require(event_meta.get("semantic_sha256") == _semantic_hash(list(expected_events)), "target events semantic hash drift")
    _require(int(event_meta.get("row_count", -1)) == len(expected_events), "target event count drift")
    producer = dict(manifest.get("producer", {}) or {})
    _require(producer.get("name") == "calibrate_identifiable_gate_v12.target_producer", "target producer drift")
    _require(producer.get("selector_sha256") == _sha256(Path(__file__)), "target producer code drift")
    _require(producer.get("gate_support_sha256") == _sha256(GATE_SUPPORT_PATH), "target gate code drift")
    source = dict(manifest.get("source", {}) or {})
    _require(source.get("protocol_sha256") == _sha256(protocol_path), "target protocol hash drift")
    _require(source.get("lock_sha256") == _sha256(lock_path), "target lock hash drift")
    _require(source.get("trace_semantic_hash") == trace_meta["semantic_trace_hash"], "target trace semantic hash drift")
    _require(source.get("trace_raw_file_set_hash") == trace_meta["raw_file_set_hash"], "target trace raw hash drift")
    _require(source.get("calibration_manifest_sha256") == calibration_manifest_sha256, "target calibration hash drift")
    _require(source.get("go_no_go_manifest_sha256") == go_no_go_manifest_sha256, "target go/no-go hash drift")
    return manifest


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, Mapping):
            rows = payload.get("events", payload.get("labels"))
        else:
            rows = None
        if not isinstance(rows, list):
            raise ValueError("JSON branch labels must be a list or contain events/labels")
        return [dict(row) for row in rows]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_branch_manifest(
    path: Path,
    *,
    labels_path: Path,
    spec: CalibrationSpec,
    expected_cohort: str,
) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(manifest, Mapping), "branch manifest must be a JSON object")
    _require(
        manifest.get("schema") == "v12_branch_runner_manifest_v1",
        "branch manifest schema drift",
    )
    _require(manifest.get("status") == "complete", "branch manifest is incomplete")
    _require(manifest.get("method_version") == spec.method_version, "branch manifest method drift")
    _require(manifest.get("label_source") == LABEL_SOURCE, "branch manifest label source drift")
    _require(
        manifest.get("exact_action_provenance") == "exact",
        "branch manifest exact-action provenance missing",
    )
    _require(manifest.get("v12_cohort") == expected_cohort, "branch manifest cohort drift")
    _require(
        tuple(int(seed) for seed in manifest.get("seeds", []) or []) == spec.seeds,
        "branch manifest seed block drift",
    )
    _require(int(manifest.get("horizon_steps", -1)) == spec.horizon_steps, "branch manifest horizon drift")
    _require(
        math.isclose(float(manifest.get("gamma", float("nan"))), spec.gamma, rel_tol=0.0, abs_tol=1e-12),
        "branch manifest gamma drift",
    )
    _require(
        math.isclose(float(manifest.get("epsilon", float("nan"))), spec.epsilon, rel_tol=0.0, abs_tol=1e-12),
        "branch manifest epsilon drift",
    )
    _verify_payload_hash(manifest, "manifest_payload_hash", "branch manifest")
    output_hashes = dict(manifest.get("output_hashes", {}) or {})
    _require(
        output_hashes.get(labels_path.name) == _sha256(labels_path),
        "branch manifest label hash drift",
    )
    return dict(manifest)


@dataclass(frozen=True)
class BranchLabels:
    labels: Mapping[Tuple[int, int, int, int], bool]
    semantic_hash: str
    raw_sha256: str
    method_versions: Tuple[str, ...]
    source_methods: Tuple[str, ...]


def load_branch_labels(
    path: Path,
    *,
    required_keys: Sequence[Tuple[int, int, int, int]],
    spec: CalibrationSpec,
    branch_manifest: Mapping[str, Any],
) -> BranchLabels:
    _require(branch_manifest.get("schema") == "v12_branch_runner_manifest_v1", "verified branch manifest is required")
    _require(branch_manifest.get("method_version") == spec.method_version, "branch manifest method drift")
    _require(branch_manifest.get("label_source") == LABEL_SOURCE, "branch manifest label source drift")
    _require(branch_manifest.get("exact_action_provenance") == "exact", "branch manifest exact-action provenance missing")
    rows = _read_rows(path)
    output_hashes = dict(branch_manifest.get("output_hashes", {}) or {})
    _require(
        output_hashes.get(path.name) == _sha256(path),
        "branch manifest label hash drift",
    )
    counts = dict(branch_manifest.get("counts", {}) or {})
    _require(
        int(counts.get("target_event_rows", -1)) == len(rows),
        "branch manifest target-event label count drift",
    )
    labels: Dict[Tuple[int, int, int, int], bool] = {}
    methods: set[str] = set()
    sources: set[str] = set()
    allowed_steps = set(spec.delay_steps)
    allowed_seeds = set(spec.seeds)
    required_fields = {
        "seed",
        "delay_s",
        "delay_steps",
        "query_frame",
        "release_frame",
        "candidate_state_id",
        "release_state_id",
        "release_state_identity_sha256",
        "method_version",
        "label_source",
        "exact_action_provenance",
        "horizon_steps",
        "gamma",
        "epsilon",
        "corrective_set_nonempty",
    }
    for index, row in enumerate(rows):
        missing_fields = sorted(required_fields - set(row))
        _require(not missing_fields, f"label row {index}: required provenance missing {missing_fields}")
        seed = int(row["seed"])
        query = int(row["query_frame"])
        release = int(row["release_frame"])
        step = int(row["delay_steps"])
        _require(release - query == step, f"label row {index}: release/query/step mismatch")
        _require(seed in allowed_seeds, f"label row {index}: seed {seed} is outside calibration lock")
        _require(step in allowed_steps, f"label row {index}: delay_steps {step} is not preregistered")
        expected_s = dict(zip(spec.delay_steps, spec.delay_seconds))[step]
        _require(
            math.isclose(float(row["delay_s"]), expected_s, rel_tol=0.0, abs_tol=1e-12),
            f"label row {index}: delay_s drift",
        )
        _require(row["candidate_state_id"] == f"{seed}:{query}:{step}", f"label row {index}: candidate identity drift")
        _require(row["release_state_id"] == f"{seed}:{release}", f"label row {index}: release identity drift")
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(row["release_state_identity_sha256"])) is not None,
            f"label row {index}: release-state identity hash drift",
        )
        _require(row["method_version"] == spec.method_version, f"label row {index}: method drift")
        _require(row["label_source"] == LABEL_SOURCE, f"label row {index}: label source drift")
        _require(str(row["exact_action_provenance"]) == "1", f"label row {index}: exact-action provenance drift")
        _require(int(row["horizon_steps"]) == spec.horizon_steps, f"label row {index}: horizon drift")
        _require(math.isclose(float(row["gamma"]), spec.gamma, abs_tol=1e-12), f"label row {index}: gamma drift")
        _require(math.isclose(float(row["epsilon"]), spec.epsilon, abs_tol=1e-12), f"label row {index}: epsilon drift")
        key = (seed, step, query, release)
        _require(key not in labels, f"duplicate calibration branch label key {key}")
        labels[key] = _as_bool(row["corrective_set_nonempty"], "corrective_set_nonempty")
        if row.get("CSet") not in (None, ""):
            _require(
                _as_bool(row["CSet"], "CSet") == labels[key],
                f"label row {index}: duplicate CSet column drift",
            )
        methods.add(str(row["method_version"]))
        sources.add(str(row["label_source"]))
    _require(methods == {spec.method_version}, "branch label method_version drift")
    _require(sources == {LABEL_SOURCE}, "branch label source drift")
    required = set(required_keys)
    observed = set(labels)
    missing = sorted(required - observed)
    extra = sorted(observed - required)
    _require(not missing, f"branch labels miss {len(missing)} preregistered target events; first={missing[:3]}")
    _require(not extra, f"branch labels contain {len(extra)} non-target events; first={extra[:3]}")
    semantic_rows = [
        {"seed": key[0], "delay_steps": key[1], "query_frame": key[2], "release_frame": key[3], "CSet": int(labels[key])}
        for key in sorted(labels)
    ]
    return BranchLabels(
        labels=labels,
        semantic_hash=_semantic_hash(semantic_rows),
        raw_sha256=_sha256(path),
        method_versions=tuple(sorted(methods)),
        source_methods=tuple(sorted(sources)),
    )


@dataclass(frozen=True)
class ArmMetrics:
    q: int
    r: int
    c: int
    excluded: int
    rate: Fraction
    q_over_c: Optional[Fraction]
    r_over_c: Optional[Fraction]
    seed_macro_rate: Optional[Fraction]
    seed_macro_valid_seeds: int
    scheduled_hash: str
    evaluated_hash: str
    q_by_delay: Tuple[Tuple[int, int], ...]


def arm_metrics(
    table: OpportunityTable,
    cohort: Cohort,
    labels: BranchLabels,
    spec: CalibrationSpec,
) -> ArmMetrics:
    c = sum(int(labels.labels[table.rows[index].event_key]) for index in cohort.evaluated)
    q = len(cohort.scheduled)
    r = len(cohort.evaluated)
    _require(0 <= c <= r <= q, "invalid Q/R/C accounting")
    per_seed: Dict[int, Tuple[int, int]] = {seed: (0, 0) for seed in spec.seeds}
    for index in cohort.evaluated:
        event = table.rows[index]
        seed_c, seed_r = per_seed[event.seed]
        per_seed[event.seed] = (
            seed_c + int(labels.labels[event.event_key]),
            seed_r + 1,
        )
    seed_rates = [Fraction(seed_c, seed_r) for seed_c, seed_r in per_seed.values() if seed_r]
    macro = sum(seed_rates, Fraction(0, 1)) / len(seed_rates) if seed_rates else None
    return ArmMetrics(
        q=q,
        r=r,
        c=c,
        excluded=q - r,
        rate=Fraction(c, r) if r else Fraction(0, 1),
        q_over_c=Fraction(q, c) if c else None,
        r_over_c=Fraction(r, c) if c else None,
        seed_macro_rate=macro,
        seed_macro_valid_seeds=len(seed_rates),
        scheduled_hash=cohort.scheduled_hash,
        evaluated_hash=cohort.evaluated_hash,
        q_by_delay=cohort.q_by_delay,
    )


@dataclass(frozen=True)
class CandidateResult:
    thresholds: Thresholds
    feasible: bool
    failure_reasons: Tuple[str, ...]
    arms: Mapping[str, ArmMetrics]
    margins: Mapping[str, Fraction]
    min_margin: Fraction
    q_over_c: Optional[Fraction]
    changed: Mapping[str, Mapping[str, Any]]
    max_changed_jaccard: Fraction

    def rank_key(self) -> Tuple[Any, ...]:
        if not self.feasible or self.q_over_c is None:
            raise ValueError("infeasible candidate has no selection rank")
        return (
            self.min_margin,
            -self.q_over_c,
            self.thresholds.l,
            self.thresholds.a,
            self.thresholds.h,
            self.thresholds.i,
        )


def _fraction_json(value: Optional[Fraction]) -> Dict[str, Any]:
    if value is None:
        return {"numerator": None, "denominator": None, "value": None}
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def evaluate_candidate(
    table: OpportunityTable,
    threshold: Thresholds,
    labels: BranchLabels,
    spec: CalibrationSpec,
    *,
    loo_cache: Optional[Dict[Tuple[Any, ...], Tuple[np.ndarray, Cohort]]] = None,
) -> CandidateResult:
    cache = loo_cache if loo_cache is not None else {}
    full_mask = eligibility_mask(table, threshold, ARM_FULL)
    cohorts: Dict[str, Cohort] = {ARM_FULL: schedule_mask(table, full_mask, spec)}
    masks: Dict[str, np.ndarray] = {ARM_FULL: full_mask}
    for arm in (ARM_NO_L, ARM_NO_A, ARM_NO_H):
        key = _loo_cache_key(threshold, arm)
        cached = cache.get(key)
        if cached is None:
            mask = eligibility_mask(table, threshold, arm)
            cached = (mask, schedule_mask(table, mask, spec))
            cache[key] = cached
        masks[arm], cohorts[arm] = cached

    metrics = {arm: arm_metrics(table, cohorts[arm], labels, spec) for arm in ARMS}
    reasons: List[str] = []
    full = metrics[ARM_FULL]
    q_by_delay = dict(full.q_by_delay)
    for step in spec.delay_steps:
        q = q_by_delay[step]
        if not spec.exposure_min <= q <= spec.exposure_max:
            reasons.append(f"full_scheduled_exposure_delay_{step}={q}_outside_lock")
    if full.c <= 0:
        reasons.append("full_corrective_count_is_zero")
    if full.r <= 0:
        reasons.append("full_evaluated_release_count_is_zero")

    changed: Dict[str, Dict[str, Any]] = {}
    changed_masks: Dict[str, np.ndarray] = {}
    full_evaluated = set(cohorts[ARM_FULL].evaluated)
    for component, arm in COMPONENT_ARMS.items():
        delta = masks[arm] & ~full_mask
        changed_masks[component] = delta
        count = int(np.count_nonzero(delta))
        changed_seed_count = len({table.rows[int(index)].seed for index in np.flatnonzero(delta)})
        union_count = int(np.count_nonzero(masks[arm] | full_mask))
        fraction = Fraction(count, union_count) if union_count else Fraction(0, 1)
        evaluated_added = set(cohorts[arm].evaluated) - full_evaluated
        changed[component] = {
            "eligible_opportunities": count,
            "seed_coverage": changed_seed_count,
            "union_opportunities": union_count,
            "changed_fraction": fraction,
            "new_evaluated_cohort": len(evaluated_added),
            "new_evaluated_cohort_hash": _cohort_hash(table, sorted(evaluated_added)),
        }
        if count < spec.min_changed_opportunities:
            reasons.append(f"{component}_changed_opportunities_below_lock")
        if changed_seed_count < spec.min_changed_seeds:
            reasons.append(f"{component}_changed_seed_coverage_below_lock")
        if fraction < spec.min_changed_union_fraction:
            reasons.append(f"{component}_changed_union_fraction_below_lock")
        if not evaluated_added:
            reasons.append(f"{component}_leave_one_out_has_no_new_evaluated_cohort")
        if metrics[arm].r <= 0:
            reasons.append(f"{component}_leave_one_out_has_no_evaluated_releases")

    jaccards: List[Fraction] = []
    for left, right in combinations(("L", "A", "H"), 2):
        intersection = int(np.count_nonzero(changed_masks[left] & changed_masks[right]))
        union = int(np.count_nonzero(changed_masks[left] | changed_masks[right]))
        value = Fraction(intersection, union) if union else Fraction(1, 1)
        jaccards.append(value)
        changed[f"jaccard_{left}_{right}"] = {"value": value}
        if value >= spec.max_changed_jaccard:
            reasons.append(f"changed_set_jaccard_{left}_{right}_not_below_lock")
    max_jaccard = max(jaccards, default=Fraction(1, 1))

    margins = {
        component: full.rate - metrics[arm].rate
        for component, arm in COMPONENT_ARMS.items()
    }
    min_margin = min(margins.values())
    return CandidateResult(
        thresholds=threshold,
        feasible=not reasons,
        failure_reasons=tuple(reasons),
        arms=metrics,
        margins=margins,
        min_margin=min_margin,
        q_over_c=full.q_over_c,
        changed=changed,
        max_changed_jaccard=max_jaccard,
    )


def select_candidate(results: Sequence[CandidateResult]) -> CandidateResult:
    feasible = [row for row in results if row.feasible and row.q_over_c is not None]
    if not feasible:
        raise RuntimeError("no preregistered v12 calibration candidate satisfies all constraints")
    return max(feasible, key=lambda row: row.rank_key())


def evaluate_all(
    table: OpportunityTable,
    labels: BranchLabels,
    spec: CalibrationSpec,
) -> Tuple[List[CandidateResult], Optional[CandidateResult]]:
    cache: Dict[Tuple[Any, ...], Tuple[np.ndarray, Cohort]] = {}
    results = [
        evaluate_candidate(table, threshold, labels, spec, loo_cache=cache)
        for threshold in candidates(spec)
    ]
    feasible = [row for row in results if row.feasible and row.q_over_c is not None]
    selected = max(feasible, key=lambda row: row.rank_key()) if feasible else None
    return results, selected


def _metric_columns(prefix: str, metric: ArmMetrics) -> Dict[str, Any]:
    return {
        f"{prefix}_Q": metric.q,
        f"{prefix}_R": metric.r,
        f"{prefix}_C": metric.c,
        f"{prefix}_excluded": metric.excluded,
        f"{prefix}_cset_fraction": float(metric.rate),
        f"{prefix}_cset_numerator": metric.rate.numerator,
        f"{prefix}_cset_denominator": metric.rate.denominator,
        f"{prefix}_Q_per_C": "" if metric.q_over_c is None else float(metric.q_over_c),
        f"{prefix}_R_per_C": "" if metric.r_over_c is None else float(metric.r_over_c),
        f"{prefix}_seed_macro_cset": "" if metric.seed_macro_rate is None else float(metric.seed_macro_rate),
        f"{prefix}_seed_macro_valid_seeds": metric.seed_macro_valid_seeds,
        f"{prefix}_scheduled_cohort_hash": metric.scheduled_hash,
        f"{prefix}_evaluated_cohort_hash": metric.evaluated_hash,
    }


def result_row(
    result: CandidateResult,
    selected: Optional[CandidateResult],
    spec: Optional[CalibrationSpec] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "candidate_id": result.thresholds.candidate_id,
        "method_version": METHOD_VERSION,
        "scope": "calibration_only",
        "seed_block": "" if spec is None else f"{spec.seeds[0]}-{spec.seeds[-1]}",
        "component_source_L": "canonical_temporal_survival_from_complete_latency_snapshot",
        "component_source_A": "relative_support_weighted_maneuver_family_breadth",
        "component_source_H": "incumbent_relative_action_recovery_cost_margin",
        "component_source_I": "state_hazard_and_pre_screen_only",
        **result.thresholds.as_floats(),
        "feasible": int(result.feasible),
        "failure_reasons": "|".join(result.failure_reasons),
        "selected": int(selected is not None and result.thresholds == selected.thresholds),
        "min_cset_margin": float(result.min_margin),
        "min_cset_margin_numerator": result.min_margin.numerator,
        "min_cset_margin_denominator": result.min_margin.denominator,
        "full_Q_per_C": "" if result.q_over_c is None else float(result.q_over_c),
        "max_changed_set_jaccard": float(result.max_changed_jaccard),
    }
    for component in ("L", "A", "H"):
        margin = result.margins[component]
        detail = result.changed[component]
        row.update(
            {
                f"margin_vs_wo_{component}": float(margin),
                f"changed_{component}_eligible": detail["eligible_opportunities"],
                f"changed_{component}_seeds": detail["seed_coverage"],
                f"changed_{component}_fraction": float(detail["changed_fraction"]),
                f"changed_{component}_new_evaluated": detail["new_evaluated_cohort"],
                f"changed_{component}_cohort_hash": detail["new_evaluated_cohort_hash"],
            }
        )
    for arm, prefix in (
        (ARM_FULL, "full"),
        (ARM_NO_L, "wo_L"),
        (ARM_NO_A, "wo_A"),
        (ARM_NO_H, "wo_H"),
    ):
        row.update(_metric_columns(prefix, result.arms[arm]))
    for step, q in result.arms[ARM_FULL].q_by_delay:
        row[f"full_Q_delay_{step}"] = q
    return row


def _arm_manifest(metric: ArmMetrics) -> Dict[str, Any]:
    return {
        "Q_scheduled": metric.q,
        "R_evaluated": metric.r,
        "C_corrective": metric.c,
        "excluded_boundary": metric.excluded,
        "pooled_cset": _fraction_json(metric.rate),
        "Q_over_C": _fraction_json(metric.q_over_c),
        "R_over_C": _fraction_json(metric.r_over_c),
        "seed_macro_sensitivity": _fraction_json(metric.seed_macro_rate),
        "seed_macro_valid_seeds": metric.seed_macro_valid_seeds,
        "Q_by_delay_steps": {str(step): q for step, q in metric.q_by_delay},
        "scheduled_cohort_hash": metric.scheduled_hash,
        "evaluated_cohort_hash": metric.evaluated_hash,
    }


def selection_manifest(
    selected: CandidateResult,
    *,
    spec: CalibrationSpec,
    lock_sha256: str,
    trace_meta: Mapping[str, Any],
    label_meta: BranchLabels,
    variation: Mapping[str, Any],
    candidate_count: int,
    protocol_sha256: str,
    gate_support_sha256: str,
) -> Dict[str, Any]:
    selected_payload = {
        "candidate_id": selected.thresholds.candidate_id,
        **selected.thresholds.as_floats(),
        "minimum_full_minus_leave_one_out_cset_margin": _fraction_json(selected.min_margin),
        "full_Q_over_C": _fraction_json(selected.q_over_c),
        "margins": {key: _fraction_json(value) for key, value in selected.margins.items()},
        "arms": {arm: _arm_manifest(selected.arms[arm]) for arm in ARMS},
        "changed_cohorts": {
            key: {
                item: (_fraction_json(value) if isinstance(value, Fraction) else value)
                for item, value in detail.items()
            }
            for key, detail in selected.changed.items()
            if key in {"L", "A", "H"}
        },
        "maximum_pairwise_changed_set_jaccard": _fraction_json(selected.max_changed_jaccard),
    }
    digest_payload = {
        "lock_sha256": lock_sha256,
        "protocol_sha256": protocol_sha256,
        "gate_support_sha256": gate_support_sha256,
        "trace_semantic_hash": trace_meta["semantic_trace_hash"],
        "label_semantic_hash": label_meta.semantic_hash,
        "selected": selected_payload,
    }
    return {
        "schema": "identifiable_gate_v12_calibration_selection_v1",
        "artifact_role": "calibration_lock",
        "method_version": spec.method_version,
        "lock_id": spec.lock_id,
        "calibration_seed_block": {
            "start": spec.seeds[0],
            "end": spec.seeds[-1],
            "count": len(spec.seeds),
            "seeds": list(spec.seeds),
        },
        "delay_strata_s": list(spec.delay_seconds),
        "delay_steps": list(spec.delay_steps),
        "budget_per_seed_delay": spec.budget,
        "cooldown_complete_frames": spec.cooldown_complete_frames,
        "minimum_query_frame_gap": spec.minimum_query_frame_gap,
        "candidate_count": candidate_count,
        "candidate_space_hash": _semantic_hash([row.candidate_id for row in candidates(spec)]),
        "candidate_registry": spec.raw_lock,
        "selection_estimand": "three-delay pooled C/R; simulator seed is the inference cluster",
        "ranking": list(spec.raw_lock["selection_objective"]),
        "component_variation": variation,
        "source": {
            "trace_method": spec.method_version,
            "trace_semantic_hash": trace_meta["semantic_trace_hash"],
            "trace_raw_file_set_hash": trace_meta["raw_file_set_hash"],
            "trace_schema_versions": trace_meta["schema_versions"],
            "trace_source_signatures": trace_meta["source_signatures"],
            "branch_label_method_versions": list(label_meta.method_versions),
            "branch_label_source_methods": list(label_meta.source_methods),
            "branch_label_semantic_hash": label_meta.semantic_hash,
            "branch_label_raw_sha256": label_meta.raw_sha256,
            "lock_sha256": lock_sha256,
            "protocol_file": str(spec.raw_lock["protocol_file"]),
            "protocol_sha256": protocol_sha256,
            "selector_sha256": _sha256(Path(__file__)),
            "gate_support_sha256": gate_support_sha256,
        },
        "selected": selected_payload,
        "calibration_constraints_satisfied": True,
        "paper_acceptance": {
            "scope": "calibration_only",
            "validation_evaluated": False,
            "validation_passed": None,
            "paper_facing_passed": False,
        },
        "selection_digest": _semantic_hash(digest_payload),
        "claim_boundary": (
            "The selected floors use calibration outcomes only. This artifact neither "
            "evaluates nor passes held-out validation and cannot be used as paper acceptance."
        ),
    }


def verify_calibration_manifest(
    path: Path,
    *,
    spec: CalibrationSpec,
    protocol_path: Path,
    lock_path: Path = LOCK_PATH,
) -> Tuple[Dict[str, Any], Thresholds]:
    validate_protocol_contract(protocol_path, spec)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "identifiable_gate_v12_calibration_selection_v1", "not a v12 calibration manifest")
    _require(manifest.get("artifact_role") == "calibration_lock", "calibration artifact role drift")
    _require(manifest.get("method_version") == spec.method_version, "calibration method drift")
    _require(manifest.get("lock_id") == spec.lock_id, "calibration lock id drift")
    seed_block = dict(manifest.get("calibration_seed_block", {}) or {})
    _require(tuple(seed_block.get("seeds", []) or []) == spec.seeds, "calibration seed block drift")
    acceptance = dict(manifest.get("paper_acceptance", {}) or {})
    _require(
        acceptance
        == {
            "scope": "calibration_only",
            "validation_evaluated": False,
            "validation_passed": None,
            "paper_facing_passed": False,
        },
        "calibration-only acceptance boundary was altered",
    )
    _require(manifest.get("calibration_constraints_satisfied") is True, "calibration constraints did not pass")
    source = dict(manifest.get("source", {}) or {})
    _require(source.get("lock_sha256") == _sha256(lock_path), "calibration lock hash drift")
    _require(source.get("selector_sha256") == _sha256(Path(__file__)), "selector code differs from calibration lock")
    _require(source.get("gate_support_sha256") == _sha256(GATE_SUPPORT_PATH), "gate code differs from calibration lock")
    _require(protocol_path.is_file(), f"missing independent v12 protocol: {protocol_path}")
    _require(protocol_path.name == str(spec.raw_lock["protocol_file"]), "locked protocol filename drift")
    _require(source.get("protocol_sha256") == _sha256(protocol_path), "v12 protocol differs from calibration lock")
    expected_space_hash = _semantic_hash([row.candidate_id for row in candidates(spec)])
    _require(manifest.get("candidate_space_hash") == expected_space_hash, "candidate space hash drift")
    selected = dict(manifest.get("selected", {}) or {})
    threshold = Thresholds(
        int(round(100 * float(selected["lambda_L"]))),
        int(round(100 * float(selected["lambda_A"]))),
        int(round(100 * float(selected["lambda_H"]))),
        int(round(100 * float(selected["lambda_I"]))),
    )
    _require(threshold.candidate_id == selected.get("candidate_id"), "selected candidate id/floors mismatch")
    _require(
        threshold.l in spec.floor_units
        and threshold.a in spec.floor_units
        and threshold.h in spec.floor_units
        and threshold.i == spec.i_floor_units,
        "selected floors are outside the preregistered candidate space",
    )
    digest_payload = {
        "lock_sha256": source["lock_sha256"],
        "protocol_sha256": source["protocol_sha256"],
        "gate_support_sha256": source["gate_support_sha256"],
        "trace_semantic_hash": source["trace_semantic_hash"],
        "label_semantic_hash": source["branch_label_semantic_hash"],
        "selected": selected,
    }
    _require(manifest.get("selection_digest") == _semantic_hash(digest_payload), "calibration selection digest drift")
    return manifest, threshold


def load_locked_floor_overlay(
    floor_overlay_path: Path,
    *,
    calibration_manifest_path: Path,
    protocol_path: Path,
    lock_path: Path,
    threshold: Thresholds,
) -> Any:
    from tools.v12_floor_overlay import load_verified_floor_overlay

    overlay = load_verified_floor_overlay(
        floor_overlay_path,
        calibration_manifest_path=calibration_manifest_path,
        protocol_path=protocol_path,
        lock_path=lock_path,
    )
    expected = {
        "rgd_latency_survival_floor": threshold.l / 100.0,
        "rgd_maneuver_breadth_floor": threshold.a / 100.0,
        "rgd_corrective_headroom_floor": threshold.h / 100.0,
        "rgd_state_need_floor": threshold.i / 100.0,
    }
    _require(dict(overlay.floors) == expected, "runtime floor overlay differs from locked floors")
    return overlay


def locked_factorial_cohorts(
    table: OpportunityTable,
    threshold: Thresholds,
    spec: CalibrationSpec,
) -> Dict[str, Cohort]:
    return {
        arm.label: schedule_mask(table, factorial_eligibility_mask(table, threshold, arm), spec)
        for arm in VALIDATION_ARMS
    }


def locked_target_event_indices(
    table: OpportunityTable,
    threshold: Thresholds,
    spec: CalibrationSpec,
) -> Tuple[int, ...]:
    cohorts = locked_factorial_cohorts(table, threshold, spec)
    return tuple(sorted({index for cohort in cohorts.values() for index in cohort.evaluated}))


def _subset_cohort(
    table: OpportunityTable,
    cohort: Cohort,
    predicate: Any,
    spec: CalibrationSpec,
) -> Cohort:
    scheduled = tuple(index for index in cohort.scheduled if predicate(table.rows[index]))
    evaluated = tuple(index for index in scheduled if table.rows[index].evaluable)
    excluded = tuple(index for index in scheduled if not table.rows[index].evaluable)
    by_delay = tuple(
        (step, sum(table.rows[index].delay_steps == step for index in scheduled))
        for step in spec.delay_steps
    )
    return Cohort(
        scheduled=scheduled,
        evaluated=evaluated,
        excluded=excluded,
        q_by_delay=by_delay,
        scheduled_hash=_cohort_hash(table, scheduled),
        evaluated_hash=_cohort_hash(table, evaluated),
    )


def _locked_counts_by_seed(
    table: OpportunityTable,
    cohorts: Mapping[str, Cohort],
    labels: BranchLabels,
    spec: CalibrationSpec,
) -> Dict[str, Dict[int, Tuple[int, int, int]]]:
    output: Dict[str, Dict[int, Tuple[int, int, int]]] = {
        arm.label: {seed: (0, 0, 0) for seed in spec.seeds}
        for arm in VALIDATION_ARMS
    }
    for arm, cohort in cohorts.items():
        scheduled_by_seed = {seed: 0 for seed in spec.seeds}
        evaluated_by_seed = {seed: 0 for seed in spec.seeds}
        corrective_by_seed = {seed: 0 for seed in spec.seeds}
        for index in cohort.scheduled:
            scheduled_by_seed[table.rows[index].seed] += 1
        for index in cohort.evaluated:
            event = table.rows[index]
            evaluated_by_seed[event.seed] += 1
            corrective_by_seed[event.seed] += int(labels.labels[event.event_key])
        output[arm] = {
            seed: (scheduled_by_seed[seed], evaluated_by_seed[seed], corrective_by_seed[seed])
            for seed in spec.seeds
        }
    return output


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def locked_bootstrap(
    counts: Mapping[str, Mapping[int, Tuple[int, int, int]]],
    spec: CalibrationSpec,
) -> Dict[str, Any]:
    validation = dict(spec.raw_lock.get("validation", {}) or {})
    draws = int(validation["bootstrap_draws"])
    bootstrap_seed = int(validation["bootstrap_seed"])
    confidence = float(validation["one_sided_confidence"])
    alpha = 1.0 - confidence
    arms = [arm.label for arm in VALIDATION_ARMS]
    r = np.asarray([[counts[arm][seed][1] for seed in spec.seeds] for arm in arms], dtype=np.int64)
    c = np.asarray([[counts[arm][seed][2] for seed in spec.seeds] for arm in arms], dtype=np.int64)
    rng = np.random.default_rng(bootstrap_seed)
    sampled = rng.integers(0, len(spec.seeds), size=(draws, len(spec.seeds)))
    sampled_r = np.stack([r[index][sampled].sum(axis=1) for index in range(len(arms))])
    sampled_c = np.stack([c[index][sampled].sum(axis=1) for index in range(len(arms))])
    sampled_rates = np.divide(
        sampled_c,
        sampled_r,
        out=np.full(sampled_c.shape, np.nan, dtype=np.float64),
        where=sampled_r > 0,
    )
    loo_labels = (ARM_NO_L, ARM_NO_A, ARM_NO_H)
    loo_indices = [arms.index(label) for label in loo_labels]
    cset_margins = {
        label: sampled_rates[0] - sampled_rates[index]
        for label, index in zip(loo_labels, loo_indices)
    }
    margin_matrix = np.stack([cset_margins[label] for label in loo_labels])
    simultaneous_minimum = np.nanmin(margin_matrix, axis=0)
    cset_lower = {label: _quantile(values[np.isfinite(values)], alpha) for label, values in cset_margins.items()}
    simultaneous_lower = _quantile(
        simultaneous_minimum[np.isfinite(simultaneous_minimum)], alpha
    )
    yield_lower: Dict[str, float] = {}
    for label, index in zip(loo_labels, loo_indices):
        per_seed_difference = c[0] - c[index]
        sampled_difference = per_seed_difference[sampled].mean(axis=1)
        yield_lower[label] = _quantile(sampled_difference, alpha)
    return {
        "draws": draws,
        "bootstrap_seed": bootstrap_seed,
        "cluster_unit": "simulator_seed",
        "one_sided_confidence": confidence,
        "cset_margin_marginal_lower": cset_lower,
        "cset_margin_simultaneous_minimum_lower": simultaneous_lower,
        "corrective_yield_per_seed_lower": yield_lower,
    }


def locked_geometry(
    table: OpportunityTable,
    threshold: Thresholds,
    cohorts: Mapping[str, Cohort],
    spec: CalibrationSpec,
) -> Dict[str, Any]:
    full_mask = factorial_eligibility_mask(table, threshold, VALIDATION_ARMS[0])
    full_evaluated = set(cohorts[ARM_FULL].evaluated)
    details: Dict[str, Any] = {}
    change_masks: Dict[str, np.ndarray] = {}
    passed = True
    by_label = {arm.label: arm for arm in VALIDATION_ARMS}
    for component, arm_label in COMPONENT_ARMS.items():
        mask = factorial_eligibility_mask(table, threshold, by_label[arm_label])
        delta = mask & ~full_mask
        change_masks[component] = delta
        count = int(np.count_nonzero(delta))
        seed_count = len({table.rows[int(index)].seed for index in np.flatnonzero(delta)})
        union_count = int(np.count_nonzero(mask | full_mask))
        fraction = Fraction(count, union_count) if union_count else Fraction(0, 1)
        new_evaluated = set(cohorts[arm_label].evaluated) - full_evaluated
        current_pass = bool(
            count >= spec.min_changed_opportunities
            and seed_count >= spec.min_changed_seeds
            and fraction >= spec.min_changed_union_fraction
            and new_evaluated
        )
        passed = passed and current_pass
        details[component] = {
            "changed_eligible_opportunities": count,
            "changed_seed_coverage": seed_count,
            "changed_fraction_of_union": _fraction_json(fraction),
            "new_evaluated_discordance": len(new_evaluated),
            "new_evaluated_discordance_hash": _cohort_hash(table, sorted(new_evaluated)),
            "passed": current_pass,
        }
    jaccards: Dict[str, float] = {}
    for left, right in combinations(("L", "A", "H"), 2):
        intersection = int(np.count_nonzero(change_masks[left] & change_masks[right]))
        union = int(np.count_nonzero(change_masks[left] | change_masks[right]))
        value = Fraction(intersection, union) if union else Fraction(1, 1)
        jaccards[f"{left}_{right}"] = float(value)
        passed = passed and value < spec.max_changed_jaccard
    return {"components": details, "pairwise_changed_set_jaccard": jaccards, "passed": bool(passed)}


def locked_summary_rows(
    table: OpportunityTable,
    cohorts: Mapping[str, Cohort],
    labels: BranchLabels,
    spec: CalibrationSpec,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm in [item.label for item in VALIDATION_ARMS]:
        strata: List[Tuple[str, Cohort]] = [("pooled", cohorts[arm])]
        strata.extend(
            (
                f"delay_{step}",
                _subset_cohort(
                    table,
                    cohorts[arm],
                    lambda event, step=step: event.delay_steps == step,
                    spec,
                ),
            )
            for step in spec.delay_steps
        )
        for stratum, cohort in strata:
            metric = arm_metrics(table, cohort, labels, spec)
            rows.append(
                {
                    "arm": arm,
                    "stratum": stratum,
                    "Q": metric.q,
                    "R": metric.r,
                    "C": metric.c,
                    "excluded": metric.excluded,
                    "CSet": float(metric.rate),
                    "Q_per_C": "" if metric.q_over_c is None else float(metric.q_over_c),
                    "R_per_C": "" if metric.r_over_c is None else float(metric.r_over_c),
                    "scheduled_cohort_hash": metric.scheduled_hash,
                    "evaluated_cohort_hash": metric.evaluated_hash,
                }
            )
    return rows


def locked_by_seed_delay_rows(
    table: OpportunityTable,
    cohorts: Mapping[str, Cohort],
    labels: BranchLabels,
    spec: CalibrationSpec,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for arm in [item.label for item in VALIDATION_ARMS]:
        for seed in spec.seeds:
            for step in spec.delay_steps:
                cohort = _subset_cohort(
                    table,
                    cohorts[arm],
                    lambda event, seed=seed, step=step: event.seed == seed and event.delay_steps == step,
                    spec,
                )
                metric = arm_metrics(table, cohort, labels, spec)
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "delay_steps": step,
                        "delay_s": dict(zip(spec.delay_steps, spec.delay_seconds))[step],
                        "Q": metric.q,
                        "R": metric.r,
                        "C": metric.c,
                        "excluded": metric.excluded,
                        "CSet": "" if metric.r == 0 else float(metric.rate),
                        "Q_per_C": "" if metric.q_over_c is None else float(metric.q_over_c),
                    }
                )
    return rows


def _locked_exposure_compliance(
    full: ArmMetrics,
    spec: CalibrationSpec,
    partition: str,
) -> Dict[str, Any]:
    contract = dict(
        (spec.raw_lock.get("locked_partition_exposure_per_delay", {}) or {}).get(partition, {}) or {}
    )
    low = int(contract["minimum"])
    high = int(contract["maximum"])
    details = {
        str(step): {"Q": q, "minimum": low, "maximum": high, "passed": low <= q <= high}
        for step, q in full.q_by_delay
    }
    return {"by_delay_steps": details, "passed": all(row["passed"] for row in details.values())}


def locked_acceptance(
    metrics: Mapping[str, ArmMetrics],
    bootstrap: Mapping[str, Any],
    geometry: Mapping[str, Any],
    exposure: Mapping[str, Any],
    *,
    partition: str,
    spec: CalibrationSpec,
) -> Dict[str, Any]:
    full = metrics[ARM_FULL]
    comparator_labels = [arm.label for arm in VALIDATION_ARMS[1:]]
    point_cset = {
        arm: {"margin": float(full.rate - metrics[arm].rate), "passed": full.rate > metrics[arm].rate}
        for arm in comparator_labels
    }
    point_qc: Dict[str, Dict[str, Any]] = {}
    for arm in comparator_labels:
        comparator = metrics[arm].q_over_c
        passed = bool(full.q_over_c is not None and (comparator is None or full.q_over_c < comparator))
        point_qc[arm] = {
            "full_Q_over_C": None if full.q_over_c is None else float(full.q_over_c),
            "comparator_Q_over_C": None if comparator is None else float(comparator),
            "passed": passed,
        }
    loo_labels = (ARM_NO_L, ARM_NO_A, ARM_NO_H)
    loo_discordance = all(bool(geometry["components"][component]["passed"]) for component in ("L", "A", "H"))
    directional = all(point_cset[arm]["passed"] and point_qc[arm]["passed"] for arm in loo_labels)
    if partition == "go_no_go":
        passed = bool(geometry["passed"] and exposure["passed"] and loo_discordance and directional)
        return {
            "scope": "fixed_parameter_go_no_go",
            "validation_evaluated": False,
            "confirmatory_holdout_evaluated": False,
            "passed": passed,
            "paper_facing_passed": False,
            "geometry_passed": bool(geometry["passed"]),
            "exposure_protocol_compliance": exposure,
            "point_cset_direction": point_cset,
            "point_Q_over_C_direction": point_qc,
            "leave_one_out_discordance_passed": loo_discordance,
            "claim_boundary": "This fixed-parameter go/no-go screen is not confirmatory validation.",
        }
    validation = dict(spec.raw_lock.get("validation", {}) or {})
    noninferiority = float(validation["corrective_yield_noninferiority_margin_per_seed"])
    simultaneous_cset_pass = float(bootstrap["cset_margin_simultaneous_minimum_lower"]) > 0.0
    yield_pass = all(
        float(bootstrap["corrective_yield_per_seed_lower"][arm]) > noninferiority
        for arm in loo_labels
    )
    all_point_cset = all(row["passed"] for row in point_cset.values())
    all_point_qc = all(row["passed"] for row in point_qc.values())
    passed = bool(
        geometry["passed"]
        and exposure["passed"]
        and all_point_cset
        and all_point_qc
        and simultaneous_cset_pass
        and yield_pass
    )
    return {
        "scope": "confirmatory_holdout",
        "validation_evaluated": True,
        "validation_passed": passed,
        "paper_facing_passed": passed,
        "geometry_passed": bool(geometry["passed"]),
        "exposure_protocol_compliance": exposure,
        "full_cset_strictly_greater_than_all_seven_arms": all_point_cset,
        "full_Q_over_C_strictly_less_than_all_seven_arms": all_point_qc,
        "point_cset_comparisons": point_cset,
        "point_Q_over_C_comparisons": point_qc,
        "simultaneous_leave_one_out_cset_lower_bound": bootstrap["cset_margin_simultaneous_minimum_lower"],
        "simultaneous_leave_one_out_cset_passed": simultaneous_cset_pass,
        "corrective_yield_noninferiority_margin_per_seed": noninferiority,
        "corrective_yield_noninferiority_passed": yield_pass,
        "passed": passed,
    }
def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_targets(args: argparse.Namespace) -> int:
    spec = load_spec(args.lock)
    validate_protocol_contract(args.protocol, spec)
    traces, trace_meta = load_features(args.trace_root.resolve(), spec)
    table = build_opportunity_table(traces, spec)
    variation = component_variation(table, spec)
    _require(bool(variation["passed"]), "v12 trace fails preregistered component variation")
    indices = target_event_indices(table, spec)
    target_map, events = target_payload(table, indices, spec)
    publish_target_bundle(
        args.output,
        target_map=target_map,
        events=events,
        partition="calibration",
        spec=spec,
        trace_meta=trace_meta,
        protocol_path=args.protocol,
        lock_path=args.lock,
        candidate_contract={
            "mode": "preregistered_candidate_union",
            "candidate_count": len(candidates(spec)),
            "candidate_space_hash": _semantic_hash(
                [row.candidate_id for row in candidates(spec)]
            ),
            "arms": list(ARMS),
            "component_variation": variation,
        },
    )
    print(json.dumps({"events": len(events), "output": str(args.output)}, ensure_ascii=False))
    return 0


def run_select(args: argparse.Namespace) -> int:
    spec = load_spec(args.lock)
    validate_protocol_contract(args.protocol, spec)
    lock_hash_before = _sha256(args.lock)
    label_hash_before = _sha256(args.branch_labels)
    branch_manifest_hash_before = _sha256(args.branch_manifest)
    protocol_hash_before = _sha256(args.protocol)
    gate_hash_before = _sha256(GATE_SUPPORT_PATH)
    traces, trace_meta = load_features(args.trace_root.resolve(), spec)
    trace_raw_before = {row["seed"]: row["sha256"] for row in trace_meta["files"]}
    table = build_opportunity_table(traces, spec)
    variation = component_variation(table, spec)
    _require(bool(variation["passed"]), "v12 trace fails preregistered component variation")
    target_indices = target_event_indices(table, spec)
    required_keys = [table.rows[index].event_key for index in target_indices]
    branch_manifest = load_branch_manifest(
        args.branch_manifest.resolve(),
        labels_path=args.branch_labels.resolve(),
        spec=spec,
        expected_cohort="parameter_selection",
    )
    labels = load_branch_labels(
        args.branch_labels.resolve(),
        required_keys=required_keys,
        spec=spec,
        branch_manifest=branch_manifest,
    )
    results, selected = evaluate_all(table, labels, spec)
    rows = [result_row(row, selected, spec) for row in results]
    manifest = (
        None
        if selected is None
        else selection_manifest(
            selected,
            spec=spec,
            lock_sha256=lock_hash_before,
            trace_meta=trace_meta,
            label_meta=labels,
            variation=variation,
            candidate_count=len(results),
            protocol_sha256=protocol_hash_before,
            gate_support_sha256=gate_hash_before,
        )
    )

    _require(_sha256(args.lock) == lock_hash_before, "calibration lock changed during analysis")
    _require(_sha256(args.branch_labels) == label_hash_before, "branch labels changed during analysis")
    _require(
        _sha256(args.branch_manifest) == branch_manifest_hash_before,
        "branch manifest changed during analysis",
    )
    _require(_sha256(args.protocol) == protocol_hash_before, "v12 protocol changed during analysis")
    _require(_sha256(GATE_SUPPORT_PATH) == gate_hash_before, "v12 gate code changed during analysis")
    for seed in spec.seeds:
        _require(
            _sha256(_trace_path(args.trace_root.resolve(), seed)) == trace_raw_before[seed],
            f"seed {seed} trace changed during analysis",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "v12_calibration_candidates.csv", rows)
    if selected is None or manifest is None:
        failure_counts: Dict[str, int] = {}
        for result in results:
            for reason in result.failure_reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        _write_json(
            args.output_dir / "v12_calibration_failure_manifest.json",
            {
                "schema": "identifiable_gate_v12_calibration_failure_v1",
                "artifact_role": "calibration_lock_failure",
                "method_version": spec.method_version,
                "seed_block": list(spec.seeds),
                "candidate_count": len(results),
                "selected": None,
                "calibration_constraints_satisfied": False,
                "failure_reason_counts": dict(sorted(failure_counts.items())),
                "candidate_table_semantic_hash": _semantic_hash(rows),
                "source": {
                    "lock_sha256": lock_hash_before,
                    "protocol_sha256": protocol_hash_before,
                    "selector_sha256": _sha256(Path(__file__)),
                    "gate_support_sha256": gate_hash_before,
                    "trace_semantic_hash": trace_meta["semantic_trace_hash"],
                    "branch_label_semantic_hash": labels.semantic_hash,
                },
                "paper_acceptance": {
                    "scope": "calibration_only",
                    "validation_evaluated": False,
                    "validation_passed": None,
                    "paper_facing_passed": False,
                },
            },
        )
        raise RuntimeError("no preregistered v12 calibration candidate satisfies all constraints")
    _write_json(args.output_dir / "v12_calibration_selection.json", manifest["selected"])
    _write_json(args.output_dir / "v12_calibration_manifest.json", manifest)
    print(
        json.dumps(
            {
                "selected": selected.thresholds.as_floats(),
                "minimum_margin": float(selected.min_margin),
                "Q_over_C": float(selected.q_over_c) if selected.q_over_c is not None else None,
                "scope": "calibration_only",
            },
            ensure_ascii=False,
        )
    )
    return 0


def verify_go_no_go_manifest(
    path: Path,
    *,
    calibration_manifest_sha256: str,
    protocol_sha256: str,
) -> Dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "identifiable_gate_v12_locked_analysis_v1", "invalid go/no-go manifest schema")
    _require(manifest.get("partition") == "go_no_go", "manifest is not the locked go/no-go partition")
    acceptance = dict(manifest.get("acceptance", {}) or {})
    _require(acceptance.get("scope") == "fixed_parameter_go_no_go", "go/no-go scope drift")
    _require(acceptance.get("passed") is True, "go/no-go did not pass; holdout access is blocked")
    _require(acceptance.get("paper_facing_passed") is False, "go/no-go cannot be paper-facing acceptance")
    source = dict(manifest.get("source", {}) or {})
    _require(source.get("calibration_manifest_sha256") == calibration_manifest_sha256, "go/no-go used a different calibration lock")
    _require(source.get("protocol_sha256") == protocol_sha256, "go/no-go protocol hash drift")
    _require(source.get("selector_sha256") == _sha256(Path(__file__)), "go/no-go selector code hash drift")
    _require(source.get("gate_support_sha256") == _sha256(GATE_SUPPORT_PATH), "go/no-go gate code hash drift")
    digest_payload = {
        "partition": manifest["partition"],
        "seed_block": manifest["seed_block"],
        "locked_thresholds": manifest["locked_thresholds"],
        "source": source,
        "metrics": manifest["metrics"],
        "geometry": manifest["geometry"],
        "acceptance": acceptance,
    }
    _require(manifest.get("analysis_digest") == _semantic_hash(digest_payload), "go/no-go analysis digest drift")
    return manifest


def run_locked_targets(args: argparse.Namespace) -> int:
    base_spec = load_spec(args.lock)
    calibration_manifest, threshold = verify_calibration_manifest(
        args.calibration_manifest.resolve(),
        spec=base_spec,
        protocol_path=args.protocol.resolve(),
        lock_path=args.lock.resolve(),
    )
    spec = locked_partition_spec(base_spec, args.partition)
    holdout_claim = None
    if args.partition == "validation":
        _require(args.go_no_go_manifest is not None, "validation targets require a passing go/no-go manifest")
        _require(args.holdout_authorization is not None, "validation targets require one-shot holdout authorization")
        from tools.v12_holdout_guard import begin_target_generation

        holdout_claim = begin_target_generation(
            authorization_path=args.holdout_authorization,
            protocol_path=args.protocol,
            lock_path=args.lock,
            calibration_manifest_path=args.calibration_manifest,
            go_no_go_manifest_path=args.go_no_go_manifest,
            trace_root=args.trace_root,
            target_map_path=args.output,
        )
    else:
        _require(
            args.holdout_authorization is None,
            "holdout authorization is valid only for validation targets",
        )

    try:
        traces, trace_meta = load_features(args.trace_root.resolve(), spec)
        table = build_opportunity_table(traces, spec)
        variation = component_variation(table, spec)
        _require(bool(variation["passed"]), f"{args.partition} trace fails component variation")
        indices = locked_target_event_indices(table, threshold, spec)
        target_map, events = target_payload(table, indices, spec)
        events_path = args.output.with_suffix(".events.csv")
        manifest_path = args.output.with_suffix(".manifest.json")
        _write_json(args.output, target_map)
        _write_csv(events_path, events)
        manifest: Dict[str, Any] = {
            "schema": "identifiable_gate_v12_locked_snapshot_targets_v2",
            "artifact_role": f"{args.partition}_snapshot_target_lock",
            "partition": args.partition,
            "method_version": spec.method_version,
            "seed_block": list(spec.seeds),
            "locked_thresholds": {"candidate_id": threshold.candidate_id, **threshold.as_floats()},
            "arms": [arm.label for arm in VALIDATION_ARMS],
            "delay_steps": list(spec.delay_steps),
            "event_count": len(events),
            "unique_release_targets": sum(len(value) for value in target_map.values()),
            "target_map_sha256": _sha256(args.output),
            "target_events_sha256": _sha256(events_path),
            "target_map_semantic_hash": _semantic_hash(target_map),
            "target_event_semantic_hash": _semantic_hash(events),
            "trace_semantic_hash": trace_meta["semantic_trace_hash"],
            "trace_raw_file_set_hash": trace_meta["raw_file_set_hash"],
            "component_variation": variation,
            "calibration_manifest_sha256": _sha256(args.calibration_manifest),
            "go_no_go_manifest_sha256": (
                None if args.go_no_go_manifest is None else _sha256(args.go_no_go_manifest)
            ),
            "protocol_sha256": _sha256(args.protocol),
            "selector_sha256": _sha256(Path(__file__)),
            "gate_support_sha256": _sha256(GATE_SUPPORT_PATH),
            "calibration_selection_digest": calibration_manifest["selection_digest"],
        }
        if holdout_claim is not None:
            manifest.update(
                {
                    "authorization_id": holdout_claim.authorization.authorization_id,
                    "authorization_sha256": holdout_claim.authorization.raw_sha256,
                    "trace_producer_manifest_sha256": holdout_claim.run_binding[
                        "trace_producer_manifest_sha256"
                    ],
                }
            )
        manifest["manifest_payload_hash"] = _payload_hash(
            manifest, "manifest_payload_hash"
        )
        _write_json(manifest_path, manifest)
        if holdout_claim is not None:
            from tools.v12_holdout_guard import complete_target_generation

            complete_target_generation(holdout_claim, args.output)
        print(json.dumps({"partition": args.partition, "events": len(events), "output": str(args.output)}))
        return 0
    except BaseException as exc:
        if holdout_claim is not None:
            from tools.v12_holdout_guard import fail_phase

            try:
                fail_phase(holdout_claim, exc)
            except Exception as guard_exc:
                exc.add_note(
                    "holdout target generation failed and state recording also failed: "
                    + str(guard_exc)
                )
        raise


def _locked_metrics_payload(metrics: Mapping[str, ArmMetrics]) -> Dict[str, Any]:
    return {arm.label: _arm_manifest(metrics[arm.label]) for arm in VALIDATION_ARMS}


def run_locked_analysis(args: argparse.Namespace, partition: str) -> int:
    holdout_authorization_path = getattr(args, "holdout_authorization", None)
    if partition == "validation":
        _require(
            holdout_authorization_path is not None,
            "confirmatory holdout analysis requires one-shot holdout authorization",
        )
    else:
        _require(
            holdout_authorization_path is None,
            "holdout authorization is valid only for confirmatory validation",
        )

    base_spec = load_spec(args.lock)
    calibration_path = args.calibration_manifest.resolve()
    calibration_hash = _sha256(calibration_path)
    calibration_manifest, threshold = verify_calibration_manifest(
        calibration_path,
        spec=base_spec,
        protocol_path=args.protocol.resolve(),
        lock_path=args.lock.resolve(),
    )
    go_manifest_hash: Optional[str] = None
    verified_holdout = None
    if partition == "validation":
        _require(args.go_no_go_manifest is not None, "confirmatory holdout requires a passing go/no-go manifest")
        go_path = args.go_no_go_manifest.resolve()
        go_manifest_hash = _sha256(go_path)
        verify_go_no_go_manifest(
            go_path,
            calibration_manifest_sha256=calibration_hash,
            protocol_sha256=_sha256(args.protocol),
        )
        from tools.v12_holdout_guard import verify_holdout_authorization

        verified_holdout = verify_holdout_authorization(
            authorization_path=holdout_authorization_path,
            protocol_path=args.protocol,
            lock_path=args.lock,
            calibration_manifest_path=args.calibration_manifest,
            go_no_go_manifest_path=args.go_no_go_manifest,
        )
        _require(
            verified_holdout.state.get("stage") == "consumed",
            "confirmatory holdout branch workflow is not fully consumed",
        )

    # The partition gate above runs before any holdout trace or label is opened.
    spec = locked_partition_spec(base_spec, partition)
    label_hash_before = _sha256(args.branch_labels)
    branch_manifest_hash_before = _sha256(args.branch_manifest)
    traces, trace_meta = load_features(args.trace_root.resolve(), spec)
    trace_raw_before = {row["seed"]: row["sha256"] for row in trace_meta["files"]}
    table = build_opportunity_table(traces, spec)
    variation = component_variation(table, spec)
    _require(bool(variation["passed"]), f"{partition} trace fails preregistered component variation")
    target_indices = locked_target_event_indices(table, threshold, spec)
    expected_cohort = (
        "fixed_parameter_go_no_go" if partition == "go_no_go" else "confirmatory_holdout"
    )
    branch_manifest = load_branch_manifest(
        args.branch_manifest.resolve(),
        labels_path=args.branch_labels.resolve(),
        spec=spec,
        expected_cohort=expected_cohort,
    )
    if verified_holdout is not None:
        _require(
            branch_manifest.get("authorization_id")
            == verified_holdout.authorization_id,
            "validation branch manifest authorization id drift",
        )
        _require(
            branch_manifest.get("authorization_sha256")
            == verified_holdout.raw_sha256,
            "validation branch manifest authorization hash drift",
        )
    labels = load_branch_labels(
        args.branch_labels.resolve(),
        required_keys=[table.rows[index].event_key for index in target_indices],
        spec=spec,
        branch_manifest=branch_manifest,
    )
    cohorts = locked_factorial_cohorts(table, threshold, spec)
    metrics = {arm: arm_metrics(table, cohort, labels, spec) for arm, cohort in cohorts.items()}
    _require(all(metric.r > 0 for metric in metrics.values()), "a locked factorial arm has no evaluated releases")
    _require(metrics[ARM_FULL].c > 0, "locked full arm has no corrective releases")
    counts = _locked_counts_by_seed(table, cohorts, labels, spec)
    bootstrap = locked_bootstrap(counts, spec)
    geometry = locked_geometry(table, threshold, cohorts, spec)
    exposure = _locked_exposure_compliance(metrics[ARM_FULL], spec, partition)
    acceptance = locked_acceptance(
        metrics,
        bootstrap,
        geometry,
        exposure,
        partition=partition,
        spec=spec,
    )
    source = {
        "calibration_manifest_sha256": calibration_hash,
        "calibration_selection_digest": calibration_manifest["selection_digest"],
        "go_no_go_manifest_sha256": go_manifest_hash,
        "trace_semantic_hash": trace_meta["semantic_trace_hash"],
        "trace_raw_file_set_hash": trace_meta["raw_file_set_hash"],
        "branch_label_semantic_hash": labels.semantic_hash,
        "branch_label_raw_sha256": labels.raw_sha256,
        "lock_sha256": _sha256(args.lock),
        "protocol_sha256": _sha256(args.protocol),
        "selector_sha256": _sha256(Path(__file__)),
        "gate_support_sha256": _sha256(GATE_SUPPORT_PATH),
    }
    if verified_holdout is not None:
        source.update(
            {
                "holdout_authorization_id": verified_holdout.authorization_id,
                "holdout_authorization_sha256": verified_holdout.raw_sha256,
                "holdout_final_stage": str(verified_holdout.state["stage"]),
            }
        )
    metrics_payload = _locked_metrics_payload(metrics)
    digest_payload = {
        "partition": partition,
        "seed_block": list(spec.seeds),
        "locked_thresholds": {"candidate_id": threshold.candidate_id, **threshold.as_floats()},
        "source": source,
        "metrics": metrics_payload,
        "geometry": geometry,
        "acceptance": acceptance,
    }
    manifest = {
        "schema": "identifiable_gate_v12_locked_analysis_v1",
        "artifact_role": f"{partition}_locked_analysis",
        "partition": partition,
        "method_version": spec.method_version,
        "seed_block": list(spec.seeds),
        "locked_thresholds": digest_payload["locked_thresholds"],
        "parameter_search_performed": False,
        "arms": [arm.__dict__ for arm in VALIDATION_ARMS],
        "delay_strata_s": list(spec.delay_seconds),
        "delay_steps": list(spec.delay_steps),
        "Q_C_R_definition": {
            "Q": "scheduled queries including boundary exclusions",
            "R": "complete release+horizon labels",
            "C": "evaluated corrective-set-positive releases",
        },
        "component_variation": variation,
        "geometry": geometry,
        "metrics": metrics_payload,
        "bootstrap": bootstrap,
        "source": source,
        "acceptance": acceptance,
        "paper_acceptance": acceptance,
        "analysis_digest": _semantic_hash(digest_payload),
    }

    _require(_sha256(args.branch_labels) == label_hash_before, "branch labels changed during locked analysis")
    _require(
        _sha256(args.branch_manifest) == branch_manifest_hash_before,
        "branch manifest changed during locked analysis",
    )
    for seed in spec.seeds:
        _require(
            _sha256(_trace_path(args.trace_root.resolve(), seed)) == trace_raw_before[seed],
            f"seed {seed} trace changed during locked analysis",
        )
    _require(calibration_hash == _sha256(calibration_path), "calibration manifest changed during locked analysis")
    if partition == "validation":
        _require(go_manifest_hash == _sha256(args.go_no_go_manifest), "go/no-go manifest changed during validation")

    prefix = "v12_go_no_go" if partition == "go_no_go" else "v12_validation"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / f"{prefix}_summary.csv", locked_summary_rows(table, cohorts, labels, spec))
    _write_csv(
        args.output_dir / f"{prefix}_by_seed_delay.csv",
        locked_by_seed_delay_rows(table, cohorts, labels, spec),
    )
    _write_json(args.output_dir / f"{prefix}_manifest.json", manifest)
    print(
        json.dumps(
            {
                "partition": partition,
                "passed": bool(acceptance["passed"]),
                "validation_evaluated": bool(acceptance.get("validation_evaluated", False)),
            }
        )
    )
    return 0


def run_go_no_go_locked(args: argparse.Namespace) -> int:
    return run_locked_analysis(args, "go_no_go")


def run_validate_locked(args: argparse.Namespace) -> int:
    return run_locked_analysis(args, "validation")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    target_parser = subparsers.add_parser("targets")
    target_parser.add_argument("--trace-root", type=Path, required=True)
    target_parser.add_argument("--output", type=Path, required=True)
    target_parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    target_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    target_parser.set_defaults(handler=run_targets)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--trace-root", type=Path, required=True)
    select_parser.add_argument("--branch-labels", type=Path, required=True)
    select_parser.add_argument("--branch-manifest", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    select_parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    select_parser.set_defaults(handler=run_select)
    locked_targets = subparsers.add_parser("locked-targets")
    locked_targets.add_argument("--partition", choices=("go_no_go", "validation"), required=True)
    locked_targets.add_argument("--trace-root", type=Path, required=True)
    locked_targets.add_argument("--calibration-manifest", type=Path, required=True)
    locked_targets.add_argument("--go-no-go-manifest", type=Path)
    locked_targets.add_argument("--holdout-authorization", type=Path)
    locked_targets.add_argument("--output", type=Path, required=True)
    locked_targets.add_argument("--lock", type=Path, default=LOCK_PATH)
    locked_targets.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    locked_targets.set_defaults(handler=run_locked_targets)
    go_no_go = subparsers.add_parser("go-no-go-locked")
    go_no_go.add_argument("--trace-root", type=Path, required=True)
    go_no_go.add_argument("--branch-labels", type=Path, required=True)
    go_no_go.add_argument("--branch-manifest", type=Path, required=True)
    go_no_go.add_argument("--calibration-manifest", type=Path, required=True)
    go_no_go.add_argument("--output-dir", type=Path, required=True)
    go_no_go.add_argument("--lock", type=Path, default=LOCK_PATH)
    go_no_go.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    go_no_go.set_defaults(handler=run_go_no_go_locked, go_no_go_manifest=None)
    validate = subparsers.add_parser("validate-locked")
    validate.add_argument("--trace-root", type=Path, required=True)
    validate.add_argument("--branch-labels", type=Path, required=True)
    validate.add_argument("--branch-manifest", type=Path, required=True)
    validate.add_argument("--calibration-manifest", type=Path, required=True)
    validate.add_argument("--go-no-go-manifest", type=Path, required=True)
    validate.add_argument("--holdout-authorization", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--lock", type=Path, default=LOCK_PATH)
    validate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    validate.set_defaults(handler=run_validate_locked)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
