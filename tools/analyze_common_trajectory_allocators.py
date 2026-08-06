"""Shared, provenance-aware query allocators for trajectory analyses.

The release-state and component-ablation analyses operate on recorded runtime
diagnostics.  This module keeps their selection logic in one place so that a
missing action-domain or cost record is not silently converted into a positive
query opportunity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dilu.driver_agent.reasoning.rgd_support import compute_temporal_survival


# These values are the frozen allocation contract used by the legacy
# release-state analyses.  Newer analyses may load a stricter protocol, but
# still use the helpers below to interpret one trace record consistently.
RGD_FLOOR = 0.20
RGD_THRESHOLD = 0.15
BUDGET = 6
COOLDOWN = 20
TTC_CUTOFF = 0.208

_SUPPORT_SOURCE = "action_support_ranking_costs"
_HEADROOM_SOURCE = "action_recovery_costs"
_WEIGHTED_SUPPORT_SOURCE = "relative_support_weighted_maneuver_family_breadth"
_RELATIVE_HEADROOM_SOURCE = "incumbent_relative_action_recovery_cost_margin"
_DEFAULT_VIABLE_COST = 0.55
_ACTION_ID_PATTERN = re.compile(r"(?:action[_\s-]*id|id)\s*:\s*(-?\d+)", re.IGNORECASE)
_INTEGER_PATTERN = re.compile(r"-?\d+")


def _nested_gate(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the recorded recoverability gate or reject the trace record."""

    try:
        diagnostics = record["rgd_subordinate_diagnostics"]
        signal = diagnostics["recoverability_signal"]
        gate = signal["recoverability_gate"]
    except (KeyError, TypeError) as exc:
        raise ValueError("record omits recoverability-gate diagnostics") from exc
    if not isinstance(gate, Mapping):
        raise ValueError("recoverability-gate diagnostics must be a mapping")
    return gate


def _profile(record: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = record.get("rgd_subordinate_diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        return {}
    ambiguity = diagnostics.get("ambiguity_and_conflict", {})
    if not isinstance(ambiguity, Mapping):
        return {}
    profile = ambiguity.get("route_ambiguity_profile", {})
    return profile if isinstance(profile, Mapping) else {}


def _as_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _as_unit(value: Any, field: str) -> float:
    number = _as_float(value, field)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return number


def _as_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _action_id(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must contain integer action identifiers")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain integer action identifiers") from exc


def _action_ids(value: Any, field: str) -> tuple[int, ...]:
    """Normalize an action sequence or the recorder's human-readable string."""

    if isinstance(value, str):
        matches = _ACTION_ID_PATTERN.findall(value)
        # Some early artifacts kept a compact ``1;2;3`` action string.
        if not matches and value.strip() and not any(character.isalpha() for character in value):
            matches = _INTEGER_PATTERN.findall(value)
        if not matches:
            raise ValueError(f"{field} does not contain action identifiers")
        return tuple(sorted({_action_id(item, field) for item in matches}))
    if isinstance(value, Mapping) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be an action sequence")
    actions = tuple(sorted({_action_id(item, field) for item in value}))
    if not actions:
        raise ValueError(f"{field} must be nonempty")
    return actions


def _action_costs(value: Any, field: str) -> dict[int, float]:
    if not isinstance(value, Mapping):
        return {}
    costs: dict[int, float] = {}
    for raw_action, raw_cost in value.items():
        action = _action_id(raw_action, field)
        cost = _as_float(raw_cost, f"{field}[{action}]")
        previous = costs.get(action)
        if previous is not None and not math.isclose(previous, cost, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"{field} contains conflicting values for action {action}")
        costs[action] = cost
    return costs


def _resolve_action_domain(
    record: Mapping[str, Any], gate: Mapping[str, Any], profile: Mapping[str, Any]
) -> tuple[tuple[int, ...], str]:
    """Resolve the action universe without promoting an inferred domain to exact."""

    for field in ("gate_legal_actions", "gate_action_universe", "fast_executor_action_universe"):
        if field in gate and gate[field] is not None:
            return _action_ids(gate[field], field), field
    available = record.get("available_actions")
    if available not in (None, ""):
        return _action_ids(available, "available_actions"), "available_actions"

    # Cost keys are sufficient to evaluate a historical aggregate record, but
    # the caller must retain the aggregate-only provenance distinction.
    action_keys: set[int] = set()
    for field in (_HEADROOM_SOURCE, _SUPPORT_SOURCE):
        action_keys.update(_action_costs(profile.get(field), field))
    if action_keys:
        return tuple(sorted(action_keys)), "cost_keys"
    return (), "missing"


def _hold_action(gate: Mapping[str, Any]) -> int:
    value = gate.get("hold_action", 1)
    return _action_id(value, "hold_action")


def _viable_cost_threshold(gate: Mapping[str, Any]) -> float:
    value = gate.get(
        "viable_cost_threshold",
        gate.get("corrective_headroom_kappa", _DEFAULT_VIABLE_COST),
    )
    threshold = _as_float(value, "viable_cost_threshold")
    if threshold <= 0.0:
        raise ValueError("viable_cost_threshold must be positive")
    return threshold


def _costs_cover_domain(costs: Mapping[int, float], domain: Sequence[int]) -> bool:
    return bool(domain) and all(action in costs and math.isfinite(costs[action]) for action in domain)


def _gate_count(gate: Mapping[str, Any], field: str, default: int) -> int:
    value = gate.get(field, default)
    return _as_nonnegative_int(value, field)


def _uses_recorded_weighted_breadth(gate: Mapping[str, Any]) -> bool:
    return str(gate.get("alternative_metric_source", "") or "") == _WEIGHTED_SUPPORT_SOURCE


def _uses_recorded_v11_breadth(gate: Mapping[str, Any]) -> bool:
    return (
        str(gate.get("alternative_metric_source", "") or "") == _SUPPORT_SOURCE
        and "alternative_viable_count" in gate
        and "alternative_viable_ratio" in gate
    )


def _recorded_action_evidence(
    gate: Mapping[str, Any],
    *,
    domain: tuple[int, ...],
    domain_source: str,
    raw_complete: bool,
    support_complete: bool,
    metric_source: str,
    has_cost_diagnostics: bool,
) -> dict[str, Any]:
    count = _gate_count(gate, "alternative_viable_count", 0)
    absolute_count = _gate_count(gate, "absolute_alternative_count", count)
    default_raw_feasible = raw_complete if has_cost_diagnostics else bool(absolute_count)
    default_support_complete = support_complete if has_cost_diagnostics else True
    return {
        "alternative_count": count,
        "absolute_alternative_count": absolute_count,
        "admissible_alternative_fraction": _as_unit(
            gate.get("alternative_viable_ratio", 0.0), "alternative_viable_ratio"
        ),
        "raw_feasibility_valid": bool(
            gate.get("absolute_alternative_feasible", default_raw_feasible)
        ),
        "support_cost_complete": bool(
            gate.get("support_cost_complete", default_support_complete)
        ),
        "action_domain": domain,
        "action_domain_source": domain_source,
        "metric_source": metric_source,
    }


def _action_evidence(record: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve alternative-action evidence for current and historical traces."""

    profile = _profile(record)
    domain, domain_source = _resolve_action_domain(record, gate, profile)
    hold = _hold_action(gate)
    alternatives = tuple(action for action in domain if action != hold)
    raw_costs = _action_costs(profile.get(_HEADROOM_SOURCE), _HEADROOM_SOURCE)
    support_costs = _action_costs(profile.get(_SUPPORT_SOURCE), _SUPPORT_SOURCE)
    raw_complete = _costs_cover_domain(raw_costs, domain)
    support_complete = _costs_cover_domain(support_costs, domain)
    hold_in_domain = bool(domain and hold in domain)
    has_cost_diagnostics = bool(raw_costs or support_costs)

    # V12+ records export a support-weighted maneuver-family quantity.  It is
    # not interchangeable with a simple count ratio, so use the recorded
    # runtime value rather than recomputing a different metric offline.
    if _uses_recorded_weighted_breadth(gate):
        return _recorded_action_evidence(
            gate,
            domain=domain,
            domain_source=domain_source,
            raw_complete=raw_complete,
            support_complete=support_complete,
            metric_source=_WEIGHTED_SUPPORT_SOURCE,
            has_cost_diagnostics=has_cost_diagnostics,
        )

    # The v11 count-ratio was already computed at runtime.  Replaying a raw
    # action map can change its denominator on historical traces, so preserve
    # the recorded scalar when it carries its explicit v11 provenance.
    if _uses_recorded_v11_breadth(gate):
        return _recorded_action_evidence(
            gate,
            domain=domain,
            domain_source=domain_source,
            raw_complete=raw_complete,
            support_complete=support_complete,
            metric_source=_SUPPORT_SOURCE,
            has_cost_diagnostics=has_cost_diagnostics,
        )

    # When action-specific diagnostics are available, reconstruct the legacy
    # support-breadth ratio from their recorded cost landscape.  Support costs
    # take precedence; raw recovery costs remain a labelled historical
    # fallback so downstream analyses can enforce their own fail-closed rule.
    primary_costs: Mapping[int, float] | None
    metric_source: str
    if support_complete:
        primary_costs = support_costs
        metric_source = _SUPPORT_SOURCE
    elif raw_complete:
        primary_costs = raw_costs
        metric_source = "action_recovery_costs_fallback"
    else:
        primary_costs = None
        metric_source = "recorded_aggregate"

    if primary_costs is not None and hold_in_domain:
        threshold = _viable_cost_threshold(gate)
        count = sum(
            1 for action in alternatives if primary_costs[action] <= threshold
        )
        absolute_count = _gate_count(gate, "absolute_alternative_count", len(alternatives))
        fraction = float(count / absolute_count) if absolute_count else 0.0
        return {
            "alternative_count": int(count),
            "absolute_alternative_count": int(absolute_count),
            "admissible_alternative_fraction": fraction,
            "raw_feasibility_valid": bool(raw_complete and hold_in_domain),
            "support_cost_complete": bool(support_complete),
            "action_domain": domain,
            "action_domain_source": domain_source,
            "metric_source": metric_source,
        }

    count = _gate_count(gate, "alternative_viable_count", 0)
    absolute_count = _gate_count(
        gate, "absolute_alternative_count", len(alternatives) or count
    )
    raw_feasible = bool(
        gate.get(
            "absolute_alternative_feasible",
            raw_complete if has_cost_diagnostics else bool(absolute_count),
        )
    )
    # Aggregate runtime fields remain usable when a recorder version did not
    # preserve per-action costs.  An explicit support-completeness flag still
    # takes precedence when the runtime supplied one.
    aggregate_support_complete = bool(
        gate.get(
            "support_cost_complete",
            support_complete if has_cost_diagnostics else True,
        )
    )
    return {
        "alternative_count": int(count),
        "absolute_alternative_count": int(absolute_count),
        "admissible_alternative_fraction": _as_unit(
            gate.get("alternative_viable_ratio", 0.0), "alternative_viable_ratio"
        ),
        "raw_feasibility_valid": raw_feasible,
        "support_cost_complete": aggregate_support_complete,
        "action_domain": domain,
        "action_domain_source": domain_source,
        "metric_source": metric_source,
    }


def _legal_domains(gate: Mapping[str, Any]) -> tuple[str, tuple[tuple[int, ...], ...]]:
    """Describe whether the record preserves an exact legal-action domain."""

    explicit = gate.get("gate_legal_actions")
    if explicit is not None:
        return "exact", (_action_ids(explicit, "gate_legal_actions"),)
    # Older support-breadth-v11 artifacts only exposed aggregate counts.  The
    # candidate domains document that ambiguity without presenting either as a
    # recovered per-frame action set.
    return "aggregate_only", ((1, 2, 3), (1, 3, 4))


def validate_support_breadth_v11_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate v11 support-breadth provenance before offline use.

    This validator deliberately accepts only the historical v11 metric names.
    V12+ support-weighted records use a different quantity and must not be
    relabelled as a v11 count-ratio analysis.
    """

    gate = _nested_gate(record)
    profile = _profile(record)
    if gate.get("alternative_metric_source") != _SUPPORT_SOURCE:
        raise ValueError("unsupported alternative metric provenance")
    if gate.get("headroom_metric_source") != _HEADROOM_SOURCE:
        raise ValueError("unsupported headroom metric provenance")
    if profile.get("probability_cost_source") != _SUPPORT_SOURCE:
        raise ValueError("support probability provenance disagrees with the gate")

    support_costs = _action_costs(profile.get(_SUPPORT_SOURCE), _SUPPORT_SOURCE)
    recovery_costs = _action_costs(profile.get(_HEADROOM_SOURCE), _HEADROOM_SOURCE)
    if not support_costs:
        raise ValueError("support-ranking costs are required")
    if not recovery_costs:
        raise ValueError("recovery costs are required")

    for field in (
        "absolute_alternative_count",
        "alternative_viable_count",
        "alternative_support_count",
    ):
        if field not in gate:
            raise ValueError(f"gate omits {field}")
        _as_nonnegative_int(gate[field], field)
    viable_count = _as_nonnegative_int(
        gate["alternative_viable_count"], "alternative_viable_count"
    )
    support_count = _as_nonnegative_int(
        gate["alternative_support_count"], "alternative_support_count"
    )
    absolute_count = _as_nonnegative_int(
        gate["absolute_alternative_count"], "absolute_alternative_count"
    )
    if viable_count != support_count:
        raise ValueError("support and viability counts disagree")
    if viable_count > absolute_count:
        raise ValueError("viability count exceeds absolute feasibility")
    _as_unit(gate.get("alternative_viable_ratio", 0.0), "alternative_viable_ratio")
    _as_unit(gate.get("cost_headroom", 0.0), "cost_headroom")
    _viable_cost_threshold(gate)

    legal_provenance, verified_domains = _legal_domains(gate)
    if legal_provenance == "exact":
        exact_domain = verified_domains[0]
        if not set(exact_domain).issubset(support_costs):
            raise ValueError("support-ranking costs do not cover the exact legal action domain")
        if not set(exact_domain).issubset(recovery_costs):
            raise ValueError("recovery costs do not cover the exact legal action domain")
    return {
        "gate": gate,
        "profile": profile,
        "legal_action_provenance": legal_provenance,
        "verified_action_domains": verified_domains,
    }


def _latency_survival(gate: Mapping[str, Any], delay_s: float) -> float:
    latency = gate.get("latency", {})
    if not isinstance(latency, Mapping):
        raise ValueError("gate latency diagnostics must be a mapping")
    delay = _as_float(delay_s, "delay_s")
    if delay < 0.0:
        raise ValueError("delay_s must be nonnegative")
    frequency = _as_float(
        latency.get("policy_frequency", gate.get("policy_frequency", 10.0)),
        "policy_frequency",
    )
    if frequency <= 0.0:
        raise ValueError("policy_frequency must be positive")
    critical = _as_float(
        latency.get("critical_latency_seconds", gate.get("critical_latency_seconds")),
        "critical_latency_seconds",
    )
    reserve = _as_float(
        latency.get(
            "safety_reserve_seconds",
            gate.get("latency_safety_reserve_seconds", 0.0),
        ),
        "latency_safety_reserve_seconds",
    )
    if critical < 0.0 or reserve < 0.0:
        raise ValueError("latency window and reserve must be nonnegative")
    delay_steps = int(math.ceil(delay * frequency))
    execution_available = bool(
        gate.get(
            "llm_backed_execution_available",
            gate.get("execution_available", False),
        )
    )
    return float(
        compute_temporal_survival(
            critical_latency_seconds=critical,
            effective_delay_steps=delay_steps,
            policy_frequency=frequency,
            latency_prediction_available=bool(
                gate.get("latency_prediction_available", False)
            ),
            execution_available=execution_available,
            latency_source=str(
                latency.get("source", gate.get("latency_source", "")) or ""
            ),
            safety_reserve_seconds=reserve,
        )
    )


def gate_component_values(
    record: Mapping[str, Any], delay_s: float, *, require_support_breadth_v11: bool = False
) -> dict[str, Any]:
    """Resolve the offline RGD components from one recorded query frame.

    The returned component values intentionally retain raw-feasibility and
    support-completeness flags.  Factorial analyses can then remove only the
    intended support predicate while preserving the non-ablatable feasibility
    condition.
    """

    if not isinstance(record, Mapping):
        raise ValueError("record must be a mapping")
    if require_support_breadth_v11:
        checked = validate_support_breadth_v11_record(record)
        gate = checked["gate"]
    else:
        gate = _nested_gate(record)

    survival = _latency_survival(gate, delay_s)
    evidence = _action_evidence(record, gate)
    headroom = _as_unit(gate.get("cost_headroom", 0.0), "cost_headroom")
    need = _as_unit(gate.get("need_score", 0.0), "need_score")
    alternatives = float(evidence["admissible_alternative_fraction"])
    opportunity = float(survival * math.sqrt(alternatives * headroom))
    priority = float(opportunity * need)
    return {
        "latency_survival": float(survival),
        "admissible_alternative_fraction": alternatives,
        "recovery_headroom": headroom,
        "need_score": need,
        "alternative_count": int(evidence["alternative_count"]),
        "absolute_alternative_count": int(evidence["absolute_alternative_count"]),
        "raw_feasibility_valid": bool(evidence["raw_feasibility_valid"]),
        "support_cost_complete": bool(evidence["support_cost_complete"]),
        "action_domain": tuple(evidence["action_domain"]),
        "action_domain_source": str(evidence["action_domain_source"]),
        "alternative_metric_source": str(evidence["metric_source"]),
        "opportunity": opportunity,
        "priority": priority,
    }


def gate_values(record: Mapping[str, Any], delay_s: float) -> tuple[float, float, int]:
    """Return full-gate opportunity, priority, and viable-action count.

    The public tuple is retained for legacy callers.  Incomplete raw action
    feasibility or explicitly incomplete support evidence fail closed here;
    component-ablation code should use :func:`gate_component_values` directly.
    """

    values = gate_component_values(record, delay_s)
    if not values["raw_feasibility_valid"] or not values["support_cost_complete"]:
        return 0.0, 0.0, 0
    return (
        float(values["opportunity"]),
        float(values["priority"]),
        int(values["alternative_count"]),
    )


def scheduled_frames(
    records: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> list[int]:
    """Apply the frozen per-episode query budget and cooldown in trace order."""

    selected: list[int] = []
    last_frame: int | None = None
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError("trajectory record must be a mapping")
        frame = _as_nonnegative_int(record.get("frame_id", record.get("frame", index)), "frame_id")
        if last_frame is not None and frame <= last_frame:
            raise ValueError("trajectory frame identifiers must be strictly increasing")
        if len(selected) >= BUDGET:
            break
        if not predicate(record):
            continue
        if last_frame is not None and frame - last_frame <= COOLDOWN:
            continue
        selected.append(frame)
        last_frame = frame
    return selected


def _first_finite_score(record: Mapping[str, Any]) -> float | None:
    candidates: list[Any] = [record.get("ttc_route_score"), record.get("ttc_score")]
    diagnostics = record.get("rgd_subordinate_diagnostics", {})
    if isinstance(diagnostics, Mapping):
        baseline = diagnostics.get("baseline_trigger_scores", {})
        if isinstance(baseline, Mapping):
            candidates.insert(0, baseline.get("ttc_route_score"))
    baseline = record.get("baseline_trigger_scores", {})
    if isinstance(baseline, Mapping):
        candidates.insert(0, baseline.get("ttc_route_score"))
    for value in candidates:
        if value in (None, "") or isinstance(value, bool):
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return min(1.0, max(0.0, score))
    return None


def ttc_score(record: Mapping[str, Any]) -> float:
    """Read the frozen TTC-risk score, with a legacy TTC fallback.

    Modern reasoning traces store the normalized score used by the baseline in
    ``baseline_trigger_scores``.  The inverse-TTC fallback is only for older
    records that predate that field.
    """

    if not isinstance(record, Mapping):
        return 0.0
    score = _first_finite_score(record)
    if score is not None:
        return score
    state = record.get("state", record)
    if not isinstance(state, Mapping):
        return 0.0
    try:
        ttc = float(state.get("ttc", math.inf))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(ttc) or ttc <= 0.0:
        return 0.0
    return float(min(1.0, 1.0 / ttc))


__all__ = [
    "BUDGET",
    "COOLDOWN",
    "RGD_FLOOR",
    "RGD_THRESHOLD",
    "TTC_CUTOFF",
    "gate_component_values",
    "gate_values",
    "scheduled_frames",
    "ttc_score",
    "validate_support_breadth_v11_record",
]
