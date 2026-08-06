"""Closed recoverability-gate contracts for RGD routing.

The module keeps route admission separate from the action controller.  It
accepts a frozen fast incumbent, validates the exact action universe, and
admits a slow query only when latency survival, maneuver breadth, corrective
headroom, and state need pass their declared serial conditions.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from dilu.driver_agent.reasoning.decision import RGDDecision, RouteAmbiguityProfile


METHOD_VERSION = "identifiable_gate_v12"
_SUPPORT_TEMPERATURE = 0.10
_DEFAULT_VIABLE_COST_THRESHOLD = 0.55


def _finite(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return default
    return resolved if math.isfinite(resolved) else default


def _unit(value: Any, default: float = 0.0) -> float:
    resolved = _finite(value, default)
    return float(min(1.0, max(0.0, default if resolved is None else resolved)))


def _actions(values: Iterable[Any]) -> Tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _action_map(values: Any) -> Dict[int, Any]:
    if not isinstance(values, Mapping):
        return {}
    result: Dict[int, Any] = {}
    for key, value in values.items():
        try:
            result[int(key)] = value
        except (TypeError, ValueError):
            continue
    return result


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _family(action: int) -> str:
    mapping = {
        0: "lateral-left",
        1: "lane-hold",
        2: "lateral-right",
        3: "longitudinal-accelerate",
        4: "longitudinal-decelerate",
    }
    return mapping.get(int(action), f"action-{int(action)}")


def _core_story(config: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = config.get("core_story") if isinstance(config, Mapping) else None
    if isinstance(direct, Mapping):
        return direct
    risk = config.get("risk_coupling") if isinstance(config, Mapping) else None
    if isinstance(risk, Mapping) and isinstance(risk.get("core_story"), Mapping):
        return risk["core_story"]
    slow = config.get("slow_thinking") if isinstance(config, Mapping) else None
    if isinstance(slow, Mapping):
        risk = slow.get("risk_coupling")
        if isinstance(risk, Mapping) and isinstance(risk.get("core_story"), Mapping):
            return risk["core_story"]
    return {}


def _floor(
    config: Mapping[str, Any], key: str, default: float
) -> Tuple[float, str]:
    if key in config:
        return _unit(config.get(key), default), f"core_story.{key}"
    return _unit(default), f"identifiable_gate_v12.default.{key}"


@dataclass(frozen=True)
class RecoverabilityGateDefinition:
    """Versioned serial-gate configuration resolved from the core story."""

    latency_survival_floor: float
    maneuver_breadth_floor: float
    corrective_headroom_floor: float
    state_need_floor: float
    latency_survival_floor_source: str
    maneuver_breadth_floor_source: str
    corrective_headroom_floor_source: str
    state_need_floor_source: str
    route_component_mode: str = "full"
    viable_cost_threshold: float = _DEFAULT_VIABLE_COST_THRESHOLD
    enable_corridor_gate: bool = False
    enable_budget_gate: bool = False
    enable_margin_gate: bool = False
    enable_heuristic_gate: bool = False
    method_version: str = METHOD_VERSION

    @classmethod
    def from_core_story_config(
        cls, core_story: Optional[Mapping[str, Any]] = None
    ) -> "RecoverabilityGateDefinition":
        config = dict(core_story or {})
        latency, latency_source = _floor(
            config, "rgd_latency_survival_floor", 0.0
        )
        breadth, breadth_source = _floor(
            config, "rgd_maneuver_breadth_floor", 0.0
        )
        headroom, headroom_source = _floor(
            config, "rgd_corrective_headroom_floor", 0.0
        )
        need, need_source = _floor(config, "rgd_state_need_floor", 0.0)
        mode = str(config.get("rgd_route_component_mode", "full") or "full").strip().lower()
        if mode not in {"full", "need_only"}:
            raise ValueError("unsupported RGD route component mode")
        viable = _unit(
            config.get("rgd_viable_cost_threshold", _DEFAULT_VIABLE_COST_THRESHOLD),
            _DEFAULT_VIABLE_COST_THRESHOLD,
        )
        return cls(
            latency_survival_floor=latency,
            maneuver_breadth_floor=breadth,
            corrective_headroom_floor=headroom,
            state_need_floor=need,
            latency_survival_floor_source=latency_source,
            maneuver_breadth_floor_source=breadth_source,
            corrective_headroom_floor_source=headroom_source,
            state_need_floor_source=need_source,
            route_component_mode=mode,
            viable_cost_threshold=viable,
            enable_corridor_gate=bool(
                config.get("rgd_enable_corridor_gate", False)
            ),
            enable_budget_gate=bool(config.get("rgd_enable_budget_gate", False)),
            enable_margin_gate=bool(config.get("rgd_enable_margin_gate", False)),
            enable_heuristic_gate=bool(
                config.get("rgd_enable_heuristic_gate", False)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_version": self.method_version,
            "route_component_mode": self.route_component_mode,
            "latency_survival_floor": self.latency_survival_floor,
            "maneuver_breadth_floor": self.maneuver_breadth_floor,
            "corrective_headroom_floor": self.corrective_headroom_floor,
            "state_need_floor": self.state_need_floor,
            "latency_survival_floor_source": self.latency_survival_floor_source,
            "maneuver_breadth_floor_source": self.maneuver_breadth_floor_source,
            "corrective_headroom_floor_source": self.corrective_headroom_floor_source,
            "state_need_floor_source": self.state_need_floor_source,
            "viable_cost_threshold": self.viable_cost_threshold,
            "enable_corridor_gate": self.enable_corridor_gate,
            "enable_budget_gate": self.enable_budget_gate,
            "enable_margin_gate": self.enable_margin_gate,
            "enable_heuristic_gate": self.enable_heuristic_gate,
        }


@dataclass(frozen=True)
class PaperBaselineDefinition:
    """Frozen baseline route trigger used only for explicitly named controls."""

    trigger_mode: str = "none"
    random_slow_probability: float = 0.0
    uncertainty_cutoff: float = 1.0
    exposure_probability: float = 1.0
    ttc_cutoff: float = 1.0

    @classmethod
    def from_core_story_config(
        cls, core_story: Optional[Mapping[str, Any]] = None
    ) -> "PaperBaselineDefinition":
        config = dict(core_story or {})
        mode = str(config.get("paper_baseline_trigger_mode", "none") or "none").strip().lower()
        if mode not in {"none", "random_budget", "uncertainty", "ttc"}:
            raise ValueError("unsupported paper baseline trigger mode")
        return cls(
            trigger_mode=mode,
            random_slow_probability=_unit(
                config.get("paper_baseline_random_slow_probability", 0.0)
            ),
            uncertainty_cutoff=_unit(
                config.get("paper_baseline_uncertainty_cutoff", 1.0), 1.0
            ),
            exposure_probability=_unit(
                config.get("paper_baseline_exposure_probability", 1.0), 1.0
            ),
            ttc_cutoff=_unit(config.get("paper_baseline_ttc_cutoff", 1.0), 1.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_mode": self.trigger_mode,
            "random_slow_probability": self.random_slow_probability,
            "uncertainty_cutoff": self.uncertainty_cutoff,
            "exposure_probability": self.exposure_probability,
            "ttc_cutoff": self.ttc_cutoff,
        }


@dataclass(frozen=True)
class RGDExecutionContract:
    """One serial gate and one optional paper baseline definition."""

    gate_definition: RecoverabilityGateDefinition
    paper_baseline: PaperBaselineDefinition
    method_version: str = METHOD_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_version": self.method_version,
            "gate_definition": self.gate_definition.to_dict(),
            "paper_baseline": self.paper_baseline.to_dict(),
        }


@dataclass(frozen=True)
class RecoverabilityAssessment:
    """Auditable result of evaluating the serial admission contract."""

    method_version: str
    hold_action: int
    gate_action_universe: Tuple[int, ...]
    fast_executor_action_universe: Tuple[int, ...]
    gate_action_universe_source: str
    fast_executor_action_universe_source: str
    gate_domain_valid: bool
    gate_fail_closed: bool
    gate_fail_closed_reason: str
    gate_fail_closed_reasons: Tuple[str, ...]
    raw_cost_complete: bool
    missing_raw_cost_actions: Tuple[int, ...]
    nonfinite_raw_cost_actions: Tuple[int, ...]
    support_cost_complete: bool
    missing_support_cost_actions: Tuple[int, ...]
    nonfinite_support_cost_actions: Tuple[int, ...]
    support_diagnostic_complete: bool
    support_diagnostic_count: int
    support_diagnostic_effective_mass: float
    support_family_min_costs: Dict[str, float]
    support_best_family_cost: float
    support_weighted_family_mass: float
    support_breadth_formula: str
    support_breadth_temperature: float
    support_breadth_temperature_source: str
    action_maneuver_family_mapping: Dict[int, str]
    raw_feasible_alternative_actions: Tuple[int, ...]
    raw_feasible_alternative_families: Tuple[str, ...]
    alternative_maneuver_family_count: int
    alternative_maneuver_family_total: int
    alternative_viable_count: int
    alternative_viable_ratio: float
    absolute_alternative_count: int
    absolute_alternative_ratio: float
    absolute_alternative_feasible: bool
    alternative_metric_source: str
    headroom_metric_source: str
    viable_cost_threshold: float
    cost_headroom: float
    relative_corrective_headroom: float
    corrective_headroom_kappa: float
    corrective_headroom_kappa_source: str
    corrective_advantage_raw: float
    absolute_recovery_depth: float
    recovery_window: float
    post_latency_opportunity: float
    need_score: float
    need_state_hazard: float
    need_pre_screen_hazard: float
    need_metric_source: str
    recoverability_score: float
    latency_survival_floor: float
    maneuver_breadth_floor: float
    corrective_headroom_floor: float
    state_need_floor: float
    latency_survival_floor_source: str
    maneuver_breadth_floor_source: str
    corrective_headroom_floor_source: str
    state_need_floor_source: str
    latency_survival_pass: bool
    maneuver_breadth_pass: bool
    corrective_headroom_pass: bool
    state_need_pass: bool
    domain_contract_pass: bool
    executor_available_pass: bool
    latency_prediction_pass: bool
    absolute_feasibility_pass: bool
    serial_gate_pass: bool
    serial_gate_failed_components: Tuple[str, ...]
    opportunity_eligible: bool
    gate_active: bool
    component_pressures: Dict[str, Any]
    latency_context: Dict[str, Any]
    collapse_risk: float
    value_of_computation: float
    route_component_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return _json_value({
            key: value
            for key, value in self.__dict__.items()
        })

    def to_paper_dict(self) -> Dict[str, float]:
        return {
            "recovery_window": self.recovery_window,
            "action_space_affordance": self.alternative_viable_ratio,
            "commitment_reversibility": self.cost_headroom,
            "soft_recoverability": self.post_latency_opportunity,
            "recoverable_deliberation_priority": self.recoverability_score,
        }


@dataclass(frozen=True)
class RecoverabilityRoutingDecision:
    """The single closed-object route verdict exposed to the orchestrator."""

    selected_system: str
    route_score: float
    decision_threshold: float
    score_gap: float
    route_reason: str
    gate_active: bool
    serial_gate_pass: bool
    decision_threshold_override_ignored: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_system": self.selected_system,
            "route_score": self.route_score,
            "decision_threshold": self.decision_threshold,
            "score_gap": self.score_gap,
            "route_reason": self.route_reason,
            "gate_active": self.gate_active,
            "serial_gate_pass": self.serial_gate_pass,
            "decision_threshold_override_ignored": self.decision_threshold_override_ignored,
        }


def build_rgd_execution_contract(
    core_story: Optional[Mapping[str, Any]] = None,
) -> RGDExecutionContract:
    config = dict(core_story or {})
    return RGDExecutionContract(
        gate_definition=RecoverabilityGateDefinition.from_core_story_config(config),
        paper_baseline=PaperBaselineDefinition.from_core_story_config(config),
    )


def compute_temporal_survival(
    *,
    critical_latency_seconds: float,
    effective_delay_steps: int,
    policy_frequency: float,
    latency_prediction_available: bool,
    execution_available: bool,
    latency_source: Optional[str],
    safety_reserve_seconds: float = 0.0,
) -> float:
    """Return the remaining fraction of the critical decision window.

    A zero-step request is only valid with an explicit prediction source and a
    usable slow executor.  This prevents a missing latency estimate from being
    interpreted as an instantaneous request.
    """

    source = str(latency_source or "").strip().lower()
    if (
        not latency_prediction_available
        or not execution_available
        or not source
        or source in {"unknown", "none", "unavailable"}
    ):
        return 0.0
    critical = _finite(critical_latency_seconds)
    frequency = _finite(policy_frequency)
    reserve = _finite(safety_reserve_seconds)
    if (
        critical is None
        or frequency is None
        or reserve is None
        or critical < 0.0
        or frequency <= 0.0
        or reserve < 0.0
        or isinstance(effective_delay_steps, bool)
    ):
        return 0.0
    try:
        steps = int(effective_delay_steps)
    except (TypeError, ValueError):
        return 0.0
    if steps < 0:
        return 0.0
    delay_seconds = float(steps) / frequency
    if critical == 0.0:
        return 1.0 if delay_seconds == 0.0 and reserve == 0.0 else 0.0
    return _unit((critical - delay_seconds - reserve) / critical)


def _critical_window_seconds(state: Optional[Any], short_horizon_seconds: float) -> float:
    horizon = max(0.0, _finite(short_horizon_seconds, 0.0) or 0.0)
    if state is None:
        return horizon
    gap = getattr(state, "junction_gap", {})
    if isinstance(gap, Mapping) and bool(gap.get("occupied_corridor", False)):
        return 0.0
    ttc = _finite(getattr(state, "ttc", None))
    if ttc is not None and ttc >= 0.0:
        # The observed collision horizon is the actual release deadline.  The
        # RAD short horizon remains a rollout diagnostic, not a cap that turns
        # a distant, observable lead into an artificial immediate timeout.
        return ttc
    return horizon


def build_slow_path_latency_context(
    *,
    llm_available: bool,
    llm_invoke_timeout_s: float,
    short_horizon_seconds: float,
    state: Optional[Any] = None,
    predicted_slow_latency_s: Optional[float] = None,
    latency_source: Optional[str] = None,
    safety_reserve_seconds: float = 0.0,
    policy_frequency: float = 10.0,
    resolved_delay_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Bind a latency estimate to the simulator's discrete control period."""

    frequency = _finite(policy_frequency)
    if frequency is None or frequency <= 0.0:
        raise ValueError("policy frequency must be positive and finite")
    requested = _finite(predicted_slow_latency_s)
    source = str(latency_source or "unknown")
    available = requested is not None and requested >= 0.0 and source.strip().lower() not in {"", "unknown", "none"}
    delay_steps = 0
    if available:
        if resolved_delay_steps is not None:
            if isinstance(resolved_delay_steps, bool):
                available = False
            else:
                try:
                    delay_steps = int(resolved_delay_steps)
                except (TypeError, ValueError):
                    available = False
                if delay_steps < 0:
                    available = False
        else:
            delay_steps = int(math.ceil(float(requested) * frequency - 1e-12))
    if not available:
        delay_steps = 0
    scheduled_seconds = float(delay_steps) / frequency
    critical = _critical_window_seconds(state, short_horizon_seconds)
    reserve = max(0.0, _finite(safety_reserve_seconds, 0.0) or 0.0)
    recovery_window = compute_temporal_survival(
        critical_latency_seconds=critical,
        effective_delay_steps=delay_steps,
        policy_frequency=frequency,
        latency_prediction_available=bool(available),
        execution_available=bool(llm_available),
        latency_source=source,
        safety_reserve_seconds=reserve,
    )
    return {
        "requested_slow_latency_seconds": requested,
        "predicted_slow_latency_seconds": scheduled_seconds if available else 0.0,
        "latency_budget_seconds": critical,
        "safe_waiting_margin_seconds": critical,
        "critical_latency_seconds": critical,
        "reasoning_latency_pressure": _unit(1.0 - recovery_window),
        "recovery_window": recovery_window,
        "effective_delay_steps": int(delay_steps),
        "policy_frequency": float(frequency),
        "safety_reserve_seconds": reserve,
        "latency_prediction_available": bool(available),
        "llm_backed_execution_available": bool(llm_available),
        "execution_available": bool(llm_available),
        "latency_source": source,
        "llm_invoke_timeout_s": max(0.0, _finite(llm_invoke_timeout_s, 0.0) or 0.0),
        "runtime_reaction_time_seconds": 0.0,
        "runtime_reaction_time_source": "runtime_latency_contract",
    }


def compute_recoverability_collapse_risk(
    principle: Optional[Mapping[str, Any]], rad_meta: Optional[Mapping[str, Any]]
) -> float:
    principle = dict(principle or {})
    rad_meta = dict(rad_meta or {})
    irreversible = _unit(
        rad_meta.get(
            "short_horizon_irreversible_risk",
            principle.get("short_horizon_irreversible_risk", 0.0),
        )
    )
    hazard = _unit(rad_meta.get("state_hazard_score", irreversible))
    remaining = _unit(principle.get("recovery_budget_remaining", 1.0), 1.0)
    return _unit(max(irreversible, hazard, 1.0 - remaining))


def _cost_contract(
    values: Mapping[int, Any], universe: Tuple[int, ...]
) -> Tuple[Dict[int, float], Tuple[int, ...], Tuple[int, ...]]:
    resolved: Dict[int, float] = {}
    missing = []
    nonfinite = []
    for action in universe:
        if action not in values:
            missing.append(action)
            continue
        value = _finite(values[action])
        if value is None:
            nonfinite.append(action)
            continue
        resolved[action] = float(value)
    return resolved, tuple(missing), tuple(nonfinite)


def compute_recoverability_assessment(
    *,
    gate_definition: RecoverabilityGateDefinition,
    principle: Optional[Mapping[str, Any]],
    rad_meta: Optional[Mapping[str, Any]],
    latency_context: Optional[Mapping[str, Any]],
    pre_screen_context: Optional[Mapping[str, Any]],
    legal_actions: Sequence[Any],
    hold_action: int,
    route_ambiguity_profile: Optional[RouteAmbiguityProfile] = None,
) -> RecoverabilityAssessment:
    """Evaluate the closed RGD admission contract on one frozen action domain."""

    del route_ambiguity_profile
    principle = dict(principle or {})
    meta = dict(rad_meta or {})
    latency = dict(latency_context or {})
    pre_screen = dict(pre_screen_context or {})
    legal_universe = _actions(legal_actions)
    gate_declared = meta.get("gate_action_universe")
    gate_universe = _actions(gate_declared) if gate_declared is not None else legal_universe
    fast_declared = meta.get("fast_executor_action_universe")
    fast_universe = _actions(fast_declared) if fast_declared is not None else gate_universe
    gate_source = str(meta.get("gate_action_universe_source", "legal_actions") or "legal_actions")
    fast_source = str(
        meta.get("fast_executor_action_universe_source", gate_source) or gate_source
    )
    hold = int(hold_action)

    failures = []
    if not gate_universe:
        failures.append("empty_gate_action_universe")
    if gate_universe != legal_universe:
        failures.append("gate_legal_action_universe_mismatch")
    if gate_universe != fast_universe:
        failures.append("gate_fast_action_universe_mismatch")
    if hold not in gate_universe:
        failures.append("hold_action_outside_gate_action_universe")

    raw_values = _action_map(meta.get("action_recovery_costs", {}))
    raw_costs, missing_raw, nonfinite_raw = _cost_contract(raw_values, gate_universe)
    if missing_raw:
        failures.append("missing_raw_action_cost")
    if nonfinite_raw:
        failures.append("nonfinite_raw_action_cost")
    raw_complete = not missing_raw and not nonfinite_raw and set(raw_costs) == set(gate_universe)
    gate_domain_valid = not any(
        reason in failures
        for reason in (
            "empty_gate_action_universe",
            "gate_legal_action_universe_mismatch",
            "gate_fast_action_universe_mismatch",
            "hold_action_outside_gate_action_universe",
        )
    )
    gate_fail_closed = bool(failures)
    gate_fail_closed_reason = failures[0] if failures else "none"

    support_values = _action_map(meta.get("action_support_ranking_costs", {}))
    support_costs, missing_support, nonfinite_support = _cost_contract(
        support_values, gate_universe
    )
    support_complete = (
        not missing_support
        and not nonfinite_support
        and set(support_costs) == set(gate_universe)
    )

    action_family_mapping = {action: _family(action) for action in gate_universe}
    alternatives = tuple(action for action in gate_universe if action != hold)
    all_families = tuple(sorted({_family(action) for action in alternatives}))
    raw_feasible_actions = tuple(
        action
        for action in alternatives
        if raw_complete and raw_costs.get(action, math.inf) <= gate_definition.viable_cost_threshold
    )
    raw_feasible_families = tuple(
        sorted({_family(action) for action in raw_feasible_actions})
    )
    absolute_count = len(raw_feasible_actions)
    absolute_ratio = (
        float(absolute_count) / float(len(alternatives)) if alternatives else 0.0
    )
    absolute_feasible = bool(absolute_count > 0)

    family_support_costs: Dict[str, float] = {}
    if support_complete and raw_complete:
        for action in raw_feasible_actions:
            family = _family(action)
            value = support_costs[action]
            previous = family_support_costs.get(family)
            if previous is None or value < previous:
                family_support_costs[family] = value
    support_best = min(family_support_costs.values()) if family_support_costs else 0.0
    support_mass = 0.0
    if family_support_costs:
        support_mass = sum(
            math.exp(-(cost - support_best) / _SUPPORT_TEMPERATURE)
            for cost in family_support_costs.values()
        )
    breadth = (
        support_mass / float(len(all_families))
        if support_complete and all_families
        else 0.0
    )

    best_alternative_cost = (
        min(raw_costs[action] for action in raw_feasible_actions)
        if raw_feasible_actions and raw_complete
        else None
    )
    hold_cost = raw_costs.get(hold)
    if hold_cost is None or best_alternative_cost is None:
        corrective_advantage = 0.0
        relative_headroom = 0.0
        absolute_depth = 0.0
    else:
        corrective_advantage = max(0.0, hold_cost - best_alternative_cost)
        relative_headroom = _unit(
            corrective_advantage / gate_definition.viable_cost_threshold
        )
        absolute_depth = _unit(
            (gate_definition.viable_cost_threshold - best_alternative_cost)
            / gate_definition.viable_cost_threshold
        )

    recovery_window = _unit(latency.get("recovery_window", 0.0))
    latency_prediction = bool(latency.get("latency_prediction_available", False))
    executor_available = bool(
        latency.get(
            "llm_backed_execution_available",
            latency.get("execution_available", False),
        )
    )
    # A zero post-delay window is not a recoverable escalation even when a
    # configuration uses the neutral floor of zero.
    latency_survival_pass = (
        recovery_window > 0.0
        and recovery_window >= gate_definition.latency_survival_floor
    )
    maneuver_breadth_pass = (
        support_complete and breadth >= gate_definition.maneuver_breadth_floor
    )
    corrective_headroom_pass = (
        relative_headroom >= gate_definition.corrective_headroom_floor
    )
    pre_score = _unit(pre_screen.get("pre_screen_score", 0.0))
    state_hazard = _unit(
        meta.get(
            "state_hazard_score",
            meta.get("short_horizon_irreversible_risk", principle.get("short_horizon_irreversible_risk", 0.0)),
        )
    )
    need_score = max(state_hazard, pre_score)
    state_need_pass = need_score >= gate_definition.state_need_floor
    domain_contract_pass = bool(gate_domain_valid and raw_complete)
    absolute_feasibility_pass = bool(absolute_feasible)

    post_latency = (
        min(recovery_window, breadth, relative_headroom)
        if domain_contract_pass and executor_available and latency_prediction and absolute_feasible
        else 0.0
    )
    opportunity_eligible = bool(
        domain_contract_pass
        and executor_available
        and latency_prediction
        and absolute_feasible
        and latency_survival_pass
        and maneuver_breadth_pass
        and corrective_headroom_pass
        and post_latency > 0.0
    )
    full_components = {
        "latency_survival": latency_survival_pass,
        "maneuver_breadth": maneuver_breadth_pass,
        "corrective_headroom": corrective_headroom_pass,
        "state_need": state_need_pass,
    }
    if gate_definition.route_component_mode == "need_only":
        selected_components = {"state_need": state_need_pass}
    else:
        selected_components = full_components
    serial_pass = bool(
        domain_contract_pass
        and executor_available
        and latency_prediction
        and absolute_feasible
        and all(selected_components.values())
    )
    failed_components = []
    if not domain_contract_pass:
        failed_components.append("domain_contract")
    if not executor_available:
        failed_components.append("executor_available")
    if not latency_prediction:
        failed_components.append("latency_prediction")
    if not absolute_feasible:
        failed_components.append("absolute_feasibility")
    for name, passed in selected_components.items():
        if not passed:
            failed_components.append(name)
    if gate_definition.route_component_mode == "full" and not support_complete and "maneuver_breadth" not in failed_components:
        failed_components.append("maneuver_breadth")

    if not support_complete:
        full_gate_fail_closed_reason = "maneuver_breadth_support_incomplete"
    elif gate_fail_closed:
        full_gate_fail_closed_reason = gate_fail_closed_reason
    else:
        full_gate_fail_closed_reason = "none"
    collapse = compute_recoverability_collapse_risk(principle, meta)
    component_pressures = {
        "gate_composition": "explicit_serial_floors",
        "serial_bottleneck_value": min(
            recovery_window, breadth, relative_headroom, need_score
        ),
        "support_evidence_fail_closed": not support_complete,
        "full_gate_fail_closed_reason": full_gate_fail_closed_reason,
        "route_component_mode": gate_definition.route_component_mode,
    }
    return RecoverabilityAssessment(
        method_version=str(meta.get("method_version", METHOD_VERSION) or METHOD_VERSION),
        hold_action=hold,
        gate_action_universe=gate_universe,
        fast_executor_action_universe=fast_universe,
        gate_action_universe_source=gate_source,
        fast_executor_action_universe_source=fast_source,
        gate_domain_valid=gate_domain_valid,
        gate_fail_closed=gate_fail_closed,
        gate_fail_closed_reason=gate_fail_closed_reason,
        gate_fail_closed_reasons=tuple(failures),
        raw_cost_complete=raw_complete,
        missing_raw_cost_actions=missing_raw,
        nonfinite_raw_cost_actions=nonfinite_raw,
        support_cost_complete=support_complete,
        missing_support_cost_actions=missing_support,
        nonfinite_support_cost_actions=nonfinite_support,
        support_diagnostic_complete=support_complete,
        support_diagnostic_count=len(family_support_costs),
        support_diagnostic_effective_mass=float(support_mass),
        support_family_min_costs=dict(family_support_costs),
        support_best_family_cost=float(support_best),
        support_weighted_family_mass=float(support_mass),
        support_breadth_formula="sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families",
        support_breadth_temperature=_SUPPORT_TEMPERATURE,
        support_breadth_temperature_source="identifiable_gate_v12.fixed_T_A",
        action_maneuver_family_mapping=action_family_mapping,
        raw_feasible_alternative_actions=raw_feasible_actions,
        raw_feasible_alternative_families=raw_feasible_families,
        alternative_maneuver_family_count=len(raw_feasible_families),
        alternative_maneuver_family_total=len(all_families),
        alternative_viable_count=absolute_count,
        alternative_viable_ratio=float(breadth),
        absolute_alternative_count=absolute_count,
        absolute_alternative_ratio=absolute_ratio,
        absolute_alternative_feasible=absolute_feasible,
        alternative_metric_source="relative_support_weighted_maneuver_family_breadth",
        headroom_metric_source="incumbent_relative_action_recovery_cost_margin",
        viable_cost_threshold=gate_definition.viable_cost_threshold,
        cost_headroom=float(relative_headroom),
        relative_corrective_headroom=float(relative_headroom),
        corrective_headroom_kappa=gate_definition.viable_cost_threshold,
        corrective_headroom_kappa_source="identifiable_gate_v12.fixed_kappa",
        corrective_advantage_raw=float(corrective_advantage),
        absolute_recovery_depth=float(absolute_depth),
        recovery_window=float(recovery_window),
        post_latency_opportunity=float(post_latency),
        need_score=float(need_score),
        need_state_hazard=float(state_hazard),
        need_pre_screen_hazard=float(pre_score),
        need_metric_source="state_hazard_and_pre_screen_only",
        recoverability_score=float(need_score),
        latency_survival_floor=gate_definition.latency_survival_floor,
        maneuver_breadth_floor=gate_definition.maneuver_breadth_floor,
        corrective_headroom_floor=gate_definition.corrective_headroom_floor,
        state_need_floor=gate_definition.state_need_floor,
        latency_survival_floor_source=gate_definition.latency_survival_floor_source,
        maneuver_breadth_floor_source=gate_definition.maneuver_breadth_floor_source,
        corrective_headroom_floor_source=gate_definition.corrective_headroom_floor_source,
        state_need_floor_source=gate_definition.state_need_floor_source,
        latency_survival_pass=latency_survival_pass,
        maneuver_breadth_pass=maneuver_breadth_pass,
        corrective_headroom_pass=corrective_headroom_pass,
        state_need_pass=state_need_pass,
        domain_contract_pass=domain_contract_pass,
        executor_available_pass=executor_available,
        latency_prediction_pass=latency_prediction,
        absolute_feasibility_pass=absolute_feasibility_pass,
        serial_gate_pass=serial_pass,
        serial_gate_failed_components=tuple(failed_components),
        opportunity_eligible=opportunity_eligible,
        gate_active=serial_pass,
        component_pressures=component_pressures,
        latency_context=latency,
        collapse_risk=collapse,
        value_of_computation=_unit(need_score * post_latency),
        route_component_mode=gate_definition.route_component_mode,
    )


def compute_recoverability_gate_diagnostics(
    gate_definition: RecoverabilityGateDefinition,
    principle: Optional[Mapping[str, Any]],
    rad_meta: Optional[Mapping[str, Any]],
    *,
    recoverability_assessment: RecoverabilityAssessment,
) -> Dict[str, Any]:
    """Export one diagnostic object from the canonical assessment."""

    del principle, rad_meta
    assessment = recoverability_assessment
    diagnostics = assessment.to_dict()
    diagnostics.update(
        {
            "method_version": assessment.method_version,
            "rgd_gate_active": assessment.gate_active,
            "active_gate_policy": "recoverability_closed_object",
            "score_boundary": gate_definition.state_need_floor,
            "score_gap": assessment.recoverability_score - gate_definition.state_need_floor,
            "opportunity_floor": min(
                gate_definition.latency_survival_floor,
                gate_definition.maneuver_breadth_floor,
                gate_definition.corrective_headroom_floor,
            ),
            "legacy_route_threshold_ignored": False,
            "relative_support_weighted_maneuver_family_breadth": assessment.alternative_viable_ratio,
            "recoverability_score": assessment.recoverability_score,
            "recoverability_recovery_window": assessment.recovery_window,
            "recoverability_alternative_viable_ratio": assessment.alternative_viable_ratio,
            "recoverability_cost_headroom": assessment.cost_headroom,
            "recoverability_post_latency_opportunity": assessment.post_latency_opportunity,
            "predicted_slow_latency_seconds": _finite(
                assessment.latency_context.get("predicted_slow_latency_seconds"), 0.0
            )
            or 0.0,
            "latency_budget_seconds": _finite(
                assessment.latency_context.get("latency_budget_seconds"), 0.0
            )
            or 0.0,
            "reasoning_latency_pressure": _finite(
                assessment.latency_context.get("reasoning_latency_pressure"), 0.0
            )
            or 0.0,
            "critical_latency_seconds": _finite(
                assessment.latency_context.get("critical_latency_seconds"), 0.0
            )
            or 0.0,
            "effective_delay_steps": int(
                assessment.latency_context.get("effective_delay_steps", 0) or 0
            ),
            "policy_frequency": _finite(
                assessment.latency_context.get("policy_frequency"), 0.0
            )
            or 0.0,
            "safety_reserve_seconds": _finite(
                assessment.latency_context.get("safety_reserve_seconds"), 0.0
            )
            or 0.0,
            "latency_prediction_available": bool(
                assessment.latency_context.get("latency_prediction_available", False)
            ),
            "llm_backed_execution_available": bool(
                assessment.latency_context.get("llm_backed_execution_available", False)
            ),
            "latency_source": str(
                assessment.latency_context.get("latency_source", "unknown") or "unknown"
            ),
        }
    )
    return diagnostics


def resolve_closed_recoverability_route(
    stats: Dict[str, Any],
    recoverability_assessment: RecoverabilityAssessment,
    gate_diagnostics: Dict[str, Any],
    *,
    decision_threshold_override: Optional[float] = None,
) -> RecoverabilityRoutingDecision:
    """Resolve the route exclusively from the serial admission verdict."""

    assessment = recoverability_assessment
    threshold = assessment.state_need_floor
    override_ignored = decision_threshold_override is not None
    if override_ignored:
        gate_diagnostics["legacy_route_threshold_ignored"] = True
    selected = "slow" if assessment.gate_active else "fast"
    if assessment.gate_active:
        reason = "recoverability_serial_gate_pass"
    elif assessment.gate_fail_closed:
        reason = "recoverability_domain_fail_closed"
    elif not assessment.absolute_feasibility_pass:
        reason = "recoverability_no_absolute_alternative"
    else:
        reason = "recoverability_serial_gate_blocked"
    route_score = assessment.recoverability_score
    decision = RecoverabilityRoutingDecision(
        selected_system=selected,
        route_score=route_score,
        decision_threshold=threshold,
        score_gap=route_score - threshold,
        route_reason=reason,
        gate_active=assessment.gate_active,
        serial_gate_pass=assessment.serial_gate_pass,
        decision_threshold_override_ignored=override_ignored,
    )
    gate_diagnostics.update(
        {
            "rgd_gate_active": assessment.gate_active,
            "score_boundary": threshold,
            "score_gap": decision.score_gap,
            "selected_system": selected,
            "route_reason": reason,
        }
    )
    stats.update(
        {
            "recoverability_score": route_score,
            "recoverability_route_boundary": threshold,
            "recoverability_route_margin": decision.score_gap,
            "recoverability_route_policy": "recoverability_closed_object",
            "recoverability_gate_active": assessment.gate_active,
            "recoverability_collapse_risk": assessment.collapse_risk,
            "recoverability_value_of_computation": assessment.value_of_computation,
        }
    )
    return decision


def build_failure_pre_screen_config(
    risk_config: Optional[Mapping[str, Any]], *, env_type: str = ""
) -> Dict[str, Any]:
    config = dict(risk_config or {})
    screen = config.get("failure_pre_screen", {})
    if not isinstance(screen, Mapping):
        raise ValueError("failure_pre_screen configuration must be a mapping")
    return {
        "enable": bool(screen.get("enable", True)),
        "env_type": str(env_type or ""),
    }


def compute_failure_pre_screen(
    *,
    state: Any,
    principle: Optional[Mapping[str, Any]],
    rad_meta: Optional[Mapping[str, Any]],
    config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Publish immediate state pressure without creating a second route gate."""

    settings = dict(config or {})
    if not bool(settings.get("enable", True)):
        return {
            "pre_screen_score": 0.0,
            "pre_screen_trigger": False,
            "pre_screen_reason": "disabled",
            "soft_recoverability_floor": 0.0,
            "components": {},
        }
    principle = dict(principle or {})
    meta = dict(rad_meta or {})
    ttc_pressure = _unit(meta.get("ttc_pressure", 0.0))
    proximity = _unit(meta.get("proximity_complexity", 0.0))
    cross = _unit(meta.get("cross_traffic_pressure", 0.0))
    irreversible = _unit(
        meta.get(
            "short_horizon_irreversible_risk",
            principle.get("short_horizon_irreversible_risk", 0.0),
        )
    )
    score = max(ttc_pressure, proximity, cross, irreversible)
    if ttc_pressure == score and score > 0.0:
        reason = "ttc"
    elif cross == score and score > 0.0:
        reason = "cross_traffic"
    elif proximity == score and score > 0.0:
        reason = "proximity"
    elif score > 0.0:
        reason = "irreversibility"
    else:
        reason = "none"
    return {
        "pre_screen_score": score,
        "pre_screen_trigger": bool(score > 0.0),
        "pre_screen_reason": reason,
        "soft_recoverability_floor": 0.0,
        "components": {
            "ttc_pressure": ttc_pressure,
            "proximity_complexity": proximity,
            "cross_traffic_pressure": cross,
            "irreversible_risk": irreversible,
        },
        "source": "state_pressure_pre_screen",
    }


def resolve_release_dominance_guard(
    *,
    slow_action: int,
    matched_fast_action: int,
    risk_scores: Optional[Mapping[Any, Any]],
    risk_margin: float = 0.0,
    require_strict_improvement: bool = True,
    progress_guard: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Keep a slow action only when it dominates the matched fast action."""

    slow = int(slow_action)
    fast = int(matched_fast_action)
    scores = _action_map(risk_scores or {})
    slow_risk = _finite(scores.get(slow))
    fast_risk = _finite(scores.get(fast))
    margin = max(0.0, _finite(risk_margin, 0.0) or 0.0)
    meta: Dict[str, Any] = {
        "release_dominance_guard_applied": True,
        "release_dominance_guard_slow_action": slow,
        "release_dominance_guard_fast_action": fast,
        "release_dominance_guard_risk_margin": margin,
        "release_dominance_guard_require_strict_improvement": bool(require_strict_improvement),
        "release_dominance_guard_slow_risk": slow_risk,
        "release_dominance_guard_fast_risk": fast_risk,
        "release_dominance_guard_risk_gain": 0.0,
        "release_dominance_guard_retained_slow": False,
        "release_dominance_guard_fallback_to_fast": False,
        "release_dominance_guard_progress_fallback": False,
        "release_dominance_guard_progress_reason": "no_clear_progress_regression",
    }
    if slow_risk is None or fast_risk is None:
        meta.update(
            {
                "release_dominance_guard_fallback_to_fast": True,
                "release_dominance_guard_reason": "missing_risk_score_fast_fallback",
            }
        )
        return fast, meta
    gain = fast_risk - slow_risk
    meta["release_dominance_guard_risk_gain"] = gain
    improved = gain > margin if require_strict_improvement else gain >= margin
    if not improved:
        meta.update(
            {
                "release_dominance_guard_fallback_to_fast": True,
                "release_dominance_guard_reason": "slow_not_risk_dominant_fast_fallback",
            }
        )
        return fast, meta

    progress = dict(progress_guard or {})
    speed = _finite(progress.get("speed"))
    front = _finite(progress.get("front_distance"))
    ttc = _finite(progress.get("ttc"))
    thw = _finite(progress.get("thw"))
    clear = (
        speed is not None
        and front is not None
        and ttc is not None
        and thw is not None
        and front >= 24.0
        and ttc >= 4.0
        and thw >= 1.2
    )
    progress_regression = (
        (fast == 3 and slow in {1, 4})
        or (fast == 1 and slow == 4)
    )
    if clear and progress_regression:
        progress_reason = (
            "slow_brake_without_lead_pressure"
            if fast == 1 and slow == 4
            else "slow_longitudinal_progress_regression"
        )
        meta.update(
            {
                "release_dominance_guard_fallback_to_fast": True,
                "release_dominance_guard_progress_fallback": True,
                "release_dominance_guard_progress_reason": progress_reason,
                "release_dominance_guard_reason": "slow_progress_regression_fast_fallback",
            }
        )
        return fast, meta
    meta.update(
        {
            "release_dominance_guard_retained_slow": True,
            "release_dominance_guard_reason": "slow_risk_dominates",
        }
    )
    return slow, meta


def _stable_unit_interval(*parts: Any) -> float:
    encoded = json.dumps(
        list(parts), ensure_ascii=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return integer / float(1 << 64)


def resolve_paper_baseline_trigger(
    config: Mapping[str, Any],
    stats: Dict[str, Any],
    baseline: PaperBaselineDefinition,
    route_ambiguity_profile: RouteAmbiguityProfile,
    rad_meta: Mapping[str, Any],
    fused_route_score: float,
) -> Tuple[str, float, str]:
    """Resolve a declared control route without changing the RGD evidence."""

    current = PaperBaselineDefinition.from_core_story_config(_core_story(config))
    definition = current if current.trigger_mode == baseline.trigger_mode else baseline
    mode = definition.trigger_mode
    seed = config.get("fixed_seed_override")
    frame = int(stats.get("decision_count", 0) or 0)
    stats["paper_baseline_trigger_mode"] = mode
    if mode == "random_budget":
        draw = _stable_unit_interval("random_budget_v2", seed, frame)
        stats["paper_baseline_random_draw"] = draw
        stats["paper_baseline_random_slow_probability"] = definition.random_slow_probability
        slow = draw < definition.random_slow_probability
        return ("slow" if slow else "fast", definition.random_slow_probability, "random_budget")
    if mode == "uncertainty":
        entropy = _unit(route_ambiguity_profile.ambiguity_entropy)
        exposure = _stable_unit_interval("baseline_exposure_v1", "uncertainty", seed, frame)
        stats["paper_baseline_uncertainty"] = entropy
        stats["paper_baseline_uncertainty_cutoff"] = definition.uncertainty_cutoff
        stats["paper_baseline_exposure_draw"] = exposure
        stats["paper_baseline_exposure_probability"] = definition.exposure_probability
        slow = entropy >= definition.uncertainty_cutoff and exposure < definition.exposure_probability
        return ("slow" if slow else "fast", entropy, "uncertainty")
    if mode == "ttc":
        ttc_pressure = _unit(rad_meta.get("ttc_pressure", fused_route_score))
        stats["paper_baseline_ttc_pressure"] = ttc_pressure
        stats["paper_baseline_ttc_cutoff"] = definition.ttc_cutoff
        return (
            "slow" if ttc_pressure >= definition.ttc_cutoff else "fast",
            ttc_pressure,
            "ttc",
        )
    return "fast", 0.0, "none"


def export_route_ambiguity_to_decision(decision: RGDDecision) -> None:
    profile = decision.ambiguity_profile
    if profile is None:
        return
    decision.stats.update(
        {
            "route_ambiguity_profile": profile.to_dict(),
            "route_ambiguity_entropy": profile.ambiguity_entropy,
            "route_ambiguity_gap": profile.ambiguity_gap,
            "route_ambiguity_disagreement": profile.evidence_disagreement,
        }
    )


def bridge_orchestrator_stats_into_decision(
    stats: Mapping[str, Any], decision: RGDDecision
) -> None:
    """Copy route evidence into the resolved decision without losing action data."""

    merged = dict(stats or {})
    merged.update(dict(decision.stats or {}))
    merged.setdefault("proposed_action", int(decision.action))
    merged.setdefault("final_action", int(decision.action))
    decision.stats = merged


def build_recoverability_public_signal(decision_meta: Mapping[str, Any]) -> Dict[str, float]:
    meta = dict(decision_meta or {})
    gate = meta.get("recoverability_gate")
    gate_public = gate.get("public_signal", {}) if isinstance(gate, Mapping) else {}

    def pick(*keys: str) -> float:
        for key in keys:
            if key in meta:
                return _unit(meta[key])
            if key in gate_public:
                return _unit(gate_public[key])
            if isinstance(gate, Mapping) and key in gate:
                return _unit(gate[key])
        return 0.0

    return {
        "recovery_window": pick("recoverability_recovery_window", "recovery_window"),
        "action_space_affordance": pick(
            "recoverability_alternative_viable_ratio", "action_space_affordance", "alternative_viable_ratio"
        ),
        "commitment_reversibility": pick(
            "recoverability_cost_headroom", "commitment_reversibility", "cost_headroom"
        ),
        "soft_recoverability": pick(
            "recoverability_post_latency_opportunity", "soft_recoverability", "post_latency_opportunity"
        ),
        "recoverable_deliberation_priority": pick(
            "recoverability_score", "recoverable_deliberation_priority"
        ),
    }


def build_recoverability_minimal_explanation(
    decision_meta: Mapping[str, Any]
) -> Dict[str, Any]:
    gate = dict(decision_meta.get("recoverability_gate", {}) or {})
    return {
        "selected_system": str(decision_meta.get("system_used", gate.get("selected_system", "fast"))),
        "route_reason": str(decision_meta.get("route_reason", gate.get("route_reason", "unknown"))),
        "gate_active": bool(gate.get("gate_active", False)),
        "serial_gate_pass": bool(gate.get("serial_gate_pass", False)),
    }


def build_route_authority_audit_fields(
    proposed_action: int,
    final_action: int,
    *,
    safety_override: bool,
    shield_override: bool,
) -> Dict[str, Any]:
    return {
        "proposed_action": int(proposed_action),
        "final_action": int(final_action),
        "route_action_changed": bool(int(proposed_action) != int(final_action)),
        "safety_override": bool(safety_override),
        "shield_override": bool(shield_override),
    }


def build_claim_downgrade_signal_fields(
    decision_meta: Mapping[str, Any],
    proposed_action: int,
    final_action: int,
    *,
    safety_override: bool,
    shield_override: bool,
) -> Dict[str, Any]:
    return {
        "route_action_changed": bool(int(proposed_action) != int(final_action)),
        "safety_override": bool(safety_override),
        "shield_override": bool(shield_override),
        "slow_path_failed": bool(
            decision_meta.get("slow_reasoning_success") is False
            and str(decision_meta.get("system_used", "")) == "fast_after_slow_failure"
        ),
    }
