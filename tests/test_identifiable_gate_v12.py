import math
from types import SimpleNamespace

import pytest

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.rad import RADSignalController
from dilu.driver_agent.reasoning.rgd_core import RGDOrchestrator
from dilu.driver_agent.reasoning.decision import RGDDecision
from dilu.driver_agent.reasoning.rgd_support import (
    RecoverabilityGateDefinition,
    compute_recoverability_assessment,
    compute_recoverability_gate_diagnostics,
    resolve_closed_recoverability_route,
)
from dilu.evaluation.decision_trace import build_decision_meta
from dilu.runtime_frame_trace import build_episode_event


METHOD_VERSION = "identifiable_gate_v12"


def _latency_context():
    return {
        "predicted_slow_latency_seconds": 0.0,
        "latency_budget_seconds": 1.0,
        "safe_waiting_margin_seconds": 1.0,
        "reasoning_latency_pressure": 0.0,
        "recovery_window": 1.0,
        "critical_latency_seconds": 1.0,
        "latency_source": "unit_test",
        "policy_frequency": 10.0,
        "safety_reserve_seconds": 0.0,
        "effective_delay_steps": 0,
        "latency_prediction_available": True,
        "runtime_reaction_time_seconds": 0.0,
        "runtime_reaction_time_source": "unit_test",
        "llm_backed_execution_available": True,
    }


def _principle():
    return {
        "corridor_width": 2,
        "recovery_budget_remaining": 0.7,
        "dominance_margin": 0.4,
        "principle_satisfied": True,
        "reachable_safe_set_ratio": 0.6,
        "viable_headroom": 0.7,
        "short_horizon_irreversible_risk": 0.8,
    }


def _rad_meta(**overrides):
    payload = {
        "method_version": METHOD_VERSION,
        "gate_action_universe": [0, 1, 3, 4],
        "fast_executor_action_universe": [0, 1, 3, 4],
        "gate_action_universe_source": "fast_executor_contract",
        "fast_executor_action_universe_source": "fast_stack_union",
        "action_recovery_costs": {0: 0.50, 1: 0.20, 3: 0.30, 4: 0.80},
        "action_support_ranking_costs": {0: 0.50, 1: 0.20, 3: 0.30, 4: 0.80},
        "support_breadth_temperature": 0.10,
        "support_breadth_temperature_source": "rad_config_default",
        "state_hazard_score": 0.70,
        "ttc_pressure": 0.60,
        "proximity_complexity": 0.70,
        "prediction_horizon_s": 0.8,
    }
    payload.update(overrides)
    return payload


def _assess(meta=None, *, hold=0, universe=(0, 1, 3, 4), pre_score=0.4):
    return compute_recoverability_assessment(
        gate_definition=RecoverabilityGateDefinition.from_core_story_config({}),
        principle=_principle(),
        rad_meta=meta or _rad_meta(),
        latency_context=_latency_context(),
        pre_screen_context={
            "pre_screen_score": pre_score,
            "pre_screen_trigger": pre_score > 0.0,
            "pre_screen_reason": "unit_test",
            "soft_recoverability_floor": 0.0,
        },
        legal_actions=tuple(universe),
        hold_action=hold,
    )


def test_action_zero_and_exact_action_domains_are_preserved():
    assessment = _assess(hold=0)

    assert assessment.method_version == METHOD_VERSION
    assert assessment.hold_action == 0
    assert assessment.gate_action_universe == (0, 1, 3, 4)
    assert assessment.fast_executor_action_universe == (0, 1, 3, 4)
    assert assessment.gate_domain_valid is True
    assert assessment.gate_fail_closed is False


@pytest.mark.parametrize(
    ("meta", "hold", "reason"),
    [
        (_rad_meta(), 2, "hold_action_outside_gate_action_universe"),
        (
            _rad_meta(fast_executor_action_universe=[0, 1, 3]),
            0,
            "gate_fast_action_universe_mismatch",
        ),
        (
            _rad_meta(action_recovery_costs={0: 0.5, 1: 0.2, 3: 0.3}),
            0,
            "missing_raw_action_cost",
        ),
        (
            _rad_meta(action_recovery_costs={0: 0.5, 1: math.nan, 3: 0.3, 4: 0.8}),
            0,
            "nonfinite_raw_action_cost",
        ),
    ],
)
def test_domain_and_raw_cost_contracts_fail_closed(meta, hold, reason):
    assessment = _assess(meta, hold=hold)

    assert assessment.gate_fail_closed is True
    assert reason in assessment.gate_fail_closed_reasons
    assert assessment.opportunity_eligible is False
    assert assessment.gate_active is False
    assert assessment.post_latency_opportunity == 0.0


@pytest.mark.parametrize(
    ("support_costs", "missing", "nonfinite"),
    [
        ({0: 0.5, 1: 0.2, 3: 0.3}, (4,), ()),
        ({0: 0.5, 1: math.inf, 3: 0.3, 4: 0.8}, (), (1,)),
    ],
)
def test_support_evidence_failure_closes_a_without_polluting_raw_f(
    support_costs, missing, nonfinite
):
    assessment = _assess(
        _rad_meta(action_support_ranking_costs=support_costs),
        hold=0,
    )

    assert assessment.raw_cost_complete is True
    assert assessment.gate_domain_valid is True
    assert assessment.domain_contract_pass is True
    assert assessment.gate_fail_closed is False
    assert assessment.absolute_alternative_feasible is True
    assert assessment.absolute_feasibility_pass is True
    assert assessment.support_cost_complete is False
    assert assessment.missing_support_cost_actions == missing
    assert assessment.nonfinite_support_cost_actions == nonfinite
    assert assessment.alternative_viable_ratio == 0.0
    assert assessment.maneuver_breadth_pass is False
    assert assessment.serial_gate_failed_components == ("maneuver_breadth",)
    assert assessment.serial_gate_pass is False
    assert assessment.opportunity_eligible is False
    assert assessment.gate_active is False
    assert assessment.component_pressures["support_evidence_fail_closed"] is True
    assert assessment.component_pressures["full_gate_fail_closed_reason"] == (
        "maneuver_breadth_support_incomplete"
    )


def test_support_evidence_is_removable_only_with_a_predicate():
    gate = RecoverabilityGateDefinition.from_core_story_config(
        {"rgd_route_component_mode": "need_only"}
    )
    assessment = compute_recoverability_assessment(
        gate_definition=gate,
        principle=_principle(),
        rad_meta=_rad_meta(
            action_support_ranking_costs={0: 0.5, 1: 0.2, 3: 0.3}
        ),
        latency_context=_latency_context(),
        pre_screen_context={"pre_screen_score": 0.4, "pre_screen_trigger": True},
        legal_actions=(0, 1, 3, 4),
        hold_action=0,
    )

    assert assessment.domain_contract_pass is True
    assert assessment.absolute_feasibility_pass is True
    assert assessment.support_cost_complete is False
    assert assessment.maneuver_breadth_pass is False
    assert assessment.gate_active is True


def _fast_override_context(collapse=0.4, pre_screen=0.3, ambiguity=0.2):
    return {
        "recoverability_assessment": {"collapse_risk": collapse},
        "gate_diagnostics": {
            "collapse_risk": collapse,
            "pre_screen_score": pre_screen,
        },
        "failure_pre_screen": {"pre_screen_score": pre_screen},
        "route_ambiguity_profile": {"intervention_risk": ambiguity},
    }


def _fast_decision(action):
    return RGDDecision(
        action=action,
        reasoning="unit test",
        confidence=1.0,
        system_used="fast",
        route_label="rgd_core",
        route_score=0.0,
        stats={"rule_name": "unit_test_rule"},
    )


def test_fast_incumbent_identity_binds_action_universe_and_override_context():
    orchestrator = object.__new__(RGDOrchestrator)
    context = _fast_override_context()

    identity = orchestrator._build_fast_incumbent_identity(
        action=0,
        action_universe=(0, 1, 3),
        override_context=context,
    )

    assert identity["action_id"] == 0
    assert identity["action_universe"] == [0, 1, 3]
    assert identity["stage"] == "query_state_complete_fast_stack_pre_route_pre_safety"
    assert identity["source"] == "recoverability_provisional_fast_action"
    assert identity["contract_version"] == "fast_incumbent_v1"
    assert len(identity["override_context_sha256"]) == 64
    assert len(identity["identity_sha256"]) == 64

    h_only_change = _fast_override_context()
    h_only_change["recoverability_assessment"]["relative_corrective_headroom"] = 0.9
    same_identity = orchestrator._build_fast_incumbent_identity(
        action=0,
        action_universe=(0, 1, 3),
        override_context=h_only_change,
    )
    assert same_identity["identity_sha256"] == identity["identity_sha256"]

    current_context_change = _fast_override_context(collapse=0.6)
    changed_identity = orchestrator._build_fast_incumbent_identity(
        action=0,
        action_universe=(0, 1, 3),
        override_context=current_context_change,
    )
    assert changed_identity["identity_sha256"] != identity["identity_sha256"]


def test_frozen_fast_override_context_excludes_h_and_final_route_fields():
    orchestrator = object.__new__(RGDOrchestrator)
    profile = SimpleNamespace(intervention_risk=0.2)

    context = orchestrator._build_h_independent_fast_override_context(
        principle=_principle(),
        rad_meta=_rad_meta(),
        route_ambiguity_profile=profile,
        failure_pre_screen={"pre_screen_score": 0.3, "components": {"x": 1.0}},
    )

    assert context["context_stage"] == (
        "current_frame_frozen_h_independent_fast_override"
    )
    assert set(context) == {
        "recoverability_assessment",
        "gate_diagnostics",
        "route_ambiguity_profile",
        "failure_pre_screen",
        "context_stage",
    }
    assert context["route_ambiguity_profile"] == {"intervention_risk": 0.2}
    assert context["failure_pre_screen"] == {"pre_screen_score": 0.3}
    assert "relative_corrective_headroom" not in str(context)
    assert "routing_decision" not in context


def test_fast_incumbent_match_fails_closed_on_action_universe_or_context_drift():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.stats = {}
    context = _fast_override_context()
    identity = orchestrator._build_fast_incumbent_identity(
        action=0,
        action_universe=(0, 1, 3),
        override_context=context,
    )

    assert orchestrator._assert_fast_incumbent_match(
        expected_identity=identity,
        observed_decision=_fast_decision(0),
        action_universe=(0, 1, 3),
        override_context=context,
        observed_source="actual_fast_execution",
    ) is True
    assert orchestrator.stats["rgd_actual_fast_identity_match"] is True

    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator._assert_fast_incumbent_match(
            expected_identity=identity,
            observed_decision=_fast_decision(1),
            action_universe=(0, 1, 3),
            override_context=context,
            observed_source="matched_fast_query",
        )
    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator._assert_fast_incumbent_match(
            expected_identity=identity,
            observed_decision=_fast_decision(0),
            action_universe=(0, 1),
            override_context=context,
            observed_source="matched_fast_query",
        )
    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator._assert_fast_incumbent_match(
            expected_identity=identity,
            observed_decision=_fast_decision(0),
            action_universe=(0, 1, 3),
            override_context=_fast_override_context(pre_screen=0.9),
            observed_source="matched_fast_query",
        )

    tampered_identity = dict(identity, action_id=1)
    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator._assert_fast_incumbent_match(
            expected_identity=tampered_identity,
            observed_decision=_fast_decision(0),
            action_universe=(0, 1, 3),
            override_context=context,
            observed_source="actual_fast_execution",
        )
    assert orchestrator.stats["rgd_fast_incumbent_expected_identity_valid"] is False


def test_current_fast_incumbent_binding_attaches_hash_and_fails_closed_on_drift():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.stats = {}
    orchestrator._current_fast_override_context = _fast_override_context()
    orchestrator._current_fast_incumbent_identity = (
        orchestrator._build_fast_incumbent_identity(
            action=0,
            action_universe=(0, 1, 3),
            override_context=orchestrator._current_fast_override_context,
        )
    )
    state = DrivingState(legal_actions=[0, 1, 3])
    state.set_effective_action_universe([0, 1, 3], source="unit_test")
    observed = _fast_decision(0)

    assert orchestrator._assert_current_fast_incumbent_match(
        state=state,
        observed_decision=observed,
        observed_source="matched_fast_query",
    ) is True
    assert observed.stats["fast_incumbent_identity_sha256"] == (
        orchestrator._current_fast_incumbent_identity["identity_sha256"]
    )
    assert observed.stats["fast_incumbent_identity_match"] is True

    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator._assert_current_fast_incumbent_match(
            state=state,
            observed_decision=_fast_decision(1),
            observed_source="actual_fast_execution",
        )


def test_decide_actual_fast_branch_fails_closed_against_frozen_incumbent():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.stats = {
        "fast_decisions": 0,
        "total_latency_ms": 0.0,
        "decision_count": 0,
    }
    orchestrator._forced_route_system = None
    orchestrator._rgd_signal_provider = True
    orchestrator._slow_call_cooldown_remaining = 0
    orchestrator._slow_call_budget = None
    orchestrator._slow_call_attempts = 0
    orchestrator._last_rad_meta = {}
    context = _fast_override_context()
    state = DrivingState(legal_actions=[0, 1, 3])
    state.set_effective_action_universe([0, 1, 3], source="unit_test")

    def route(_state, _force_system):
        orchestrator._current_fast_override_context = context
        orchestrator._current_fast_incumbent_identity = (
            orchestrator._build_fast_incumbent_identity(
                action=0,
                action_universe=(0, 1, 3),
                override_context=context,
            )
        )
        return "fast", 0.0, None

    orchestrator._route_system = route
    orchestrator._execute_fast = (
        lambda _state, _route_score, fast_override_context=None: _fast_decision(1)
    )

    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator.decide(state)
    assert orchestrator.stats["rgd_actual_fast_identity_match"] is False


def test_slow_path_matched_fast_fails_closed_before_slow_invocation():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.stats = {}
    orchestrator._current_fast_override_context = _fast_override_context()
    orchestrator._current_fast_incumbent_identity = (
        orchestrator._build_fast_incumbent_identity(
            action=0,
            action_universe=(0, 1, 3),
            override_context=orchestrator._current_fast_override_context,
        )
    )
    state = DrivingState(legal_actions=[0, 1, 3])
    state.set_effective_action_universe([0, 1, 3], source="unit_test")
    orchestrator._peek_full_fast_decision = (
        lambda _state, fast_override_context=None: _fast_decision(1)
    )
    orchestrator.slow = SimpleNamespace(
        think=lambda **_kwargs: pytest.fail("slow path must not run after identity drift")
    )

    with pytest.raises(RuntimeError, match="incumbent identity drift"):
        orchestrator._execute_slow_path(
            state=state,
            route_score=1.0,
            route_ambiguity_profile=None,
            recoverability_context={},
        )
    assert orchestrator.stats["rgd_matched_fast_identity_match"] is False


def test_affordance_is_relative_support_weighted_maneuver_family_breadth():
    assessment = _assess()

    assert assessment.alternative_viable_count == 2
    assert assessment.absolute_alternative_count == 2
    assert assessment.alternative_maneuver_family_count == 2
    assert assessment.alternative_maneuver_family_total == 3
    expected_mass = 1.0 + math.exp(-1.0)
    assert assessment.alternative_viable_ratio == pytest.approx(expected_mass / 3.0)
    assert assessment.alternative_metric_source == (
        "relative_support_weighted_maneuver_family_breadth"
    )
    assert assessment.action_maneuver_family_mapping == {
        0: "lateral-left",
        1: "lane-hold",
        3: "longitudinal-accelerate",
        4: "longitudinal-decelerate",
    }
    assert assessment.raw_feasible_alternative_actions == (1, 3)
    assert assessment.raw_feasible_alternative_families == (
        "lane-hold",
        "longitudinal-accelerate",
    )
    assert assessment.support_breadth_temperature == pytest.approx(0.10)
    assert assessment.support_breadth_temperature_source == "identifiable_gate_v12.fixed_T_A"
    assert assessment.support_breadth_formula == (
        "sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families"
    )
    assert assessment.support_family_min_costs == {
        "lane-hold": pytest.approx(0.20),
        "longitudinal-accelerate": pytest.approx(0.30),
    }
    assert assessment.support_best_family_cost == pytest.approx(0.20)
    assert assessment.support_weighted_family_mass == pytest.approx(expected_mass)


def test_relative_headroom_and_absolute_depth_are_separate_and_derivable():
    assessment = _assess()

    assert assessment.relative_corrective_headroom == pytest.approx((0.50 - 0.20) / 0.55)
    assert assessment.cost_headroom == assessment.relative_corrective_headroom
    assert assessment.corrective_headroom_kappa == pytest.approx(0.55)
    assert assessment.absolute_recovery_depth == pytest.approx((0.55 - 0.20) / 0.55)


def test_support_breadth_and_corrective_headroom_are_independent():
    baseline = _assess()
    support_changed = _assess(
        _rad_meta(action_support_ranking_costs={0: 0.9, 1: 0.2, 3: 0.7, 4: 0.8})
    )
    hold_changed = _assess(
        _rad_meta(action_recovery_costs={0: 0.40, 1: 0.20, 3: 0.30, 4: 0.80})
    )

    assert support_changed.relative_corrective_headroom == baseline.relative_corrective_headroom
    assert support_changed.alternative_viable_ratio < baseline.alternative_viable_ratio
    assert support_changed.support_diagnostic_effective_mass < baseline.support_diagnostic_effective_mass
    assert hold_changed.relative_corrective_headroom < baseline.relative_corrective_headroom
    assert hold_changed.alternative_viable_ratio == baseline.alternative_viable_ratio


def test_common_support_offset_does_not_change_relative_affordance():
    baseline = _assess(
        _rad_meta(action_support_ranking_costs={0: 0.40, 1: 0.20, 3: 0.30, 4: 0.70})
    )
    shifted = _assess(
        _rad_meta(action_support_ranking_costs={0: 0.60, 1: 0.40, 3: 0.50, 4: 0.90})
    )

    assert shifted.alternative_viable_ratio == pytest.approx(
        baseline.alternative_viable_ratio
    )


def test_equally_supported_raw_feasible_family_strictly_increases_affordance():
    one_family = _assess(
        _rad_meta(
            action_recovery_costs={0: 0.50, 1: 0.20, 3: 0.80, 4: 0.80},
            action_support_ranking_costs={0: 0.50, 1: 0.20, 3: 0.20, 4: 0.20},
        )
    )
    two_families = _assess(
        _rad_meta(
            action_recovery_costs={0: 0.50, 1: 0.20, 3: 0.20, 4: 0.80},
            action_support_ranking_costs={0: 0.50, 1: 0.20, 3: 0.20, 4: 0.20},
        )
    )

    assert one_family.alternative_viable_ratio == pytest.approx(1.0 / 3.0)
    assert two_families.alternative_viable_ratio == pytest.approx(2.0 / 3.0)
    assert two_families.alternative_viable_ratio > one_family.alternative_viable_ratio


def test_need_reads_only_state_hazard_and_pre_screen():
    baseline = _assess(pre_score=0.4)
    changed_costs = _assess(
        _rad_meta(
            action_recovery_costs={0: 0.54, 1: 0.01, 3: 0.54, 4: 0.54},
            action_support_ranking_costs={0: 0.99, 1: 0.01, 3: 0.99, 4: 0.99},
            corrective_gap=1.0,
            action_cost_entropy=1.0,
        ),
        pre_score=0.4,
    )

    assert baseline.need_score == pytest.approx(0.70)
    assert changed_costs.need_score == baseline.need_score
    assert baseline.need_metric_source == "state_hazard_and_pre_screen_only"
    assert baseline.need_state_hazard == pytest.approx(0.70)
    assert baseline.need_pre_screen_hazard == pytest.approx(0.40)


def test_serial_gate_uses_the_bottleneck_component_not_a_compensatory_product():
    assessment = _assess(pre_score=0.4)

    expected_bottleneck = min(
        assessment.recovery_window,
        assessment.alternative_viable_ratio,
        assessment.relative_corrective_headroom,
        assessment.need_score,
    )
    assert assessment.post_latency_opportunity == pytest.approx(
        min(
            assessment.recovery_window,
            assessment.alternative_viable_ratio,
            assessment.relative_corrective_headroom,
        )
    )
    assert assessment.recoverability_score == pytest.approx(assessment.need_score)
    assert assessment.component_pressures["gate_composition"] == "explicit_serial_floors"
    assert assessment.component_pressures["serial_bottleneck_value"] == pytest.approx(
        expected_bottleneck
    )


def test_four_explicit_floors_and_sources_drive_serial_passes():
    gate = RecoverabilityGateDefinition.from_core_story_config(
        {
            "rgd_latency_survival_floor": 0.8,
            "rgd_maneuver_breadth_floor": 0.8,
            "rgd_corrective_headroom_floor": 0.5,
            "rgd_state_need_floor": 0.6,
        }
    )
    assessment = compute_recoverability_assessment(
        gate_definition=gate,
        principle=_principle(),
        rad_meta=_rad_meta(),
        latency_context=_latency_context(),
        pre_screen_context={"pre_screen_score": 0.4, "pre_screen_trigger": True},
        legal_actions=(0, 1, 3, 4),
        hold_action=0,
    )

    assert assessment.latency_survival_pass is True
    assert assessment.maneuver_breadth_pass is False
    assert assessment.corrective_headroom_pass is True
    assert assessment.state_need_pass is True
    assert assessment.serial_gate_pass is False
    assert assessment.maneuver_breadth_floor == pytest.approx(0.8)
    assert assessment.maneuver_breadth_floor_source.endswith(
        "rgd_maneuver_breadth_floor"
    )


def test_legacy_external_threshold_cannot_form_a_second_gate():
    assessment = _assess(pre_score=0.4)
    gate = RecoverabilityGateDefinition.from_core_story_config({})
    diagnostics = compute_recoverability_gate_diagnostics(
        gate,
        _principle(),
        _rad_meta(),
        recoverability_assessment=assessment,
    )

    decision = resolve_closed_recoverability_route(
        {}, assessment, diagnostics, decision_threshold_override=0.99
    )

    assert assessment.serial_gate_pass is True
    assert decision.selected_system == "slow"
    assert decision.decision_threshold == pytest.approx(assessment.state_need_floor)
    assert diagnostics["legacy_route_threshold_ignored"] is True


def test_gate_snapshot_exports_the_complete_latency_contract():
    assessment = _assess()
    gate = RecoverabilityGateDefinition.from_core_story_config({})
    diagnostics = compute_recoverability_gate_diagnostics(
        gate,
        _principle(),
        _rad_meta(),
        recoverability_assessment=assessment,
    )
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.stats = {}
    orchestrator._rgd_threshold_audit_band = 0.02
    orchestrator._rgd_threshold_provenance = {}
    orchestrator._paper_baseline = SimpleNamespace(trigger_mode="none")

    orchestrator._export_recoverability_gate_snapshot(
        {
            "gate_diagnostics": diagnostics,
            "recoverability_assessment": assessment,
            "routing_decision": None,
            "execution_route_score": assessment.recoverability_score,
        },
        "slow",
        "unit_test",
    )

    latency = orchestrator.stats["recoverability_gate"]["latency"]
    assert latency["policy_frequency"] == pytest.approx(10.0)
    assert latency["safety_reserve_seconds"] == pytest.approx(0.0)
    assert latency["llm_backed_execution_available"] is True
    assert latency["latency_prediction_available"] is True
    assert latency["effective_delay_steps"] == 0
    assert latency["source"] == "unit_test"


def test_injected_safety_decomposition_makes_rad_raw_costs_action_discriminative():
    state = DrivingState(
        ego_speed=20.0,
        front_distance=35.0,
        ttc=5.0,
        thw=2.0,
        scenario_type="highway",
        legal_actions=[0, 1, 3],
        can_change_left=True,
        left_front_distance=40.0,
        left_rear_distance=20.0,
    )
    state.__dict__["_safety_cost_decomposition"] = {
        0: {"total": 0.10, "safety": 0.05, "comfort": 0.05, "efficiency": 0.0},
        1: {"total": 0.30, "safety": 0.25, "comfort": 0.0, "efficiency": 0.05},
        3: {"total": 0.70, "safety": 0.65, "comfort": 0.0, "efficiency": 0.05},
    }

    _, meta = RADSignalController().estimate_signal(
        state,
        conflict_score=0.0,
        action_universe=(0, 1, 3),
    )

    assert meta["raw_cost_complete"] is True
    assert meta["raw_cost_source"] == "safety_cost_decomposition"
    assert len(set(meta["action_recovery_costs"].values())) == 3
    assert meta["action_recovery_costs"][0] < meta["action_recovery_costs"][1]
    assert meta["action_recovery_costs"][1] < meta["action_recovery_costs"][3]
    for action in (0, 1, 3):
        parts = meta["action_recovery_cost_parts"][action]
        assert parts["raw_cost_formula"] == "common_risk_plus_residual_action_penalty"
        assert parts["residual_action_penalty_source"] == (
            "safety_cost_decomposition.total_minus_domain_min"
        )


def test_missing_injected_legal_action_cost_is_exported_for_fail_closed_gate():
    state = DrivingState(legal_actions=[0, 1], can_change_left=True)
    state.__dict__["_safety_cost_decomposition"] = {
        0: {"total": 0.1, "safety": 0.1, "comfort": 0.0, "efficiency": 0.0},
    }

    _, meta = RADSignalController().estimate_signal(
        state,
        conflict_score=0.0,
        action_universe=(0, 1),
    )

    assert meta["raw_cost_complete"] is False
    assert meta["missing_raw_cost_actions"] == [1]
    assessment = _assess(
        dict(
            meta,
            method_version=METHOD_VERSION,
            gate_action_universe=[0, 1],
            fast_executor_action_universe=[0, 1],
            state_hazard_score=0.5,
        ),
        hold=0,
        universe=(0, 1),
    )
    assert assessment.gate_fail_closed is True
    assert "missing_raw_action_cost" in assessment.gate_fail_closed_reasons


def test_effective_universe_includes_only_conditioned_highway_hold_action():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.fast = SimpleNamespace(action_history=[int(ActionType.LANE_LEFT)])
    state = DrivingState(
        ego_speed=18.0,
        front_distance=20.0,
        front_speed=14.0,
        ttc=5.0,
        thw=1.0,
        scenario_type="highway",
        legal_actions=[0, 1, 3, 4],
        can_change_left=False,
        left_front_distance=40.0,
        left_rear_distance=8.0,
        left_rear_speed=15.0,
    )

    universe, provenance = orchestrator._resolve_fast_executor_action_universe(state)

    assert universe == (0, 1, 3, 4)
    assert tuple(state.get_available_actions()) == universe
    assert provenance["includes_highway_pass_override"] is False
    assert provenance["includes_highway_hold_override"] is True


def test_effective_universe_includes_conditioned_pass_without_unconditional_readd():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.fast = SimpleNamespace(action_history=[])
    state = DrivingState(
        ego_speed=18.0,
        front_distance=20.0,
        front_speed=14.0,
        ttc=5.0,
        thw=1.0,
        scenario_type="highway",
        legal_actions=[0, 1, 3, 4],
        can_change_left=False,
        left_front_distance=40.0,
        left_rear_distance=8.0,
        left_rear_speed=15.0,
    )

    universe, provenance = orchestrator._resolve_fast_executor_action_universe(state)

    assert universe == (0, 1, 3, 4)
    assert tuple(state.get_available_actions()) == universe
    assert provenance["includes_highway_pass_override"] is True
    assert provenance["includes_highway_hold_override"] is False


def test_untriggered_controlled_gap_does_not_expand_effective_universe():
    orchestrator = object.__new__(RGDOrchestrator)
    orchestrator.fast = SimpleNamespace(action_history=[])
    state = DrivingState(
        ego_speed=18.0,
        front_distance=80.0,
        front_speed=18.0,
        ttc=8.0,
        thw=4.0,
        scenario_type="highway",
        legal_actions=[0, 1, 3, 4],
        can_change_left=False,
        left_front_distance=40.0,
        left_rear_distance=8.0,
        left_rear_speed=15.0,
    )

    universe, provenance = orchestrator._resolve_fast_executor_action_universe(state)

    assert universe == (1, 3, 4)
    assert tuple(state.get_available_actions()) == universe
    assert provenance["controlled_gap_candidates"] == [0]
    assert provenance["includes_highway_pass_override"] is False


def test_v12_gate_provenance_survives_decision_and_episode_trace():
    assessment = _assess()
    diagnostics = compute_recoverability_gate_diagnostics(
        RecoverabilityGateDefinition.from_core_story_config({}),
        _principle(),
        _rad_meta(),
        recoverability_assessment=assessment,
    )
    source = {
        "system_used": "fast",
        "proposed_action": 0,
        "final_action": 0,
        "rgd_method_version": METHOD_VERSION,
        "recoverability_gate": diagnostics,
        **{
            f"recoverability_{key}": value
            for key, value in {
                "gate_action_universe": list(assessment.gate_action_universe),
                "fast_executor_action_universe": list(assessment.fast_executor_action_universe),
                "gate_domain_valid": assessment.gate_domain_valid,
                "gate_fail_closed": assessment.gate_fail_closed,
                "gate_fail_closed_reason": assessment.gate_fail_closed_reason,
                "gate_fail_closed_reasons": list(assessment.gate_fail_closed_reasons),
                "alternative_maneuver_family_count": assessment.alternative_maneuver_family_count,
                "alternative_maneuver_family_total": assessment.alternative_maneuver_family_total,
                "action_maneuver_family_mapping": assessment.action_maneuver_family_mapping,
                "raw_feasible_alternative_actions": list(assessment.raw_feasible_alternative_actions),
                "raw_feasible_alternative_families": list(assessment.raw_feasible_alternative_families),
                "support_diagnostic_effective_mass": assessment.support_diagnostic_effective_mass,
                "support_cost_complete": assessment.support_cost_complete,
                "missing_support_cost_actions": list(assessment.missing_support_cost_actions),
                "nonfinite_support_cost_actions": list(assessment.nonfinite_support_cost_actions),
                "support_family_min_costs": assessment.support_family_min_costs,
                "support_best_family_cost": assessment.support_best_family_cost,
                "support_weighted_family_mass": assessment.support_weighted_family_mass,
                "support_breadth_formula": assessment.support_breadth_formula,
                "support_breadth_temperature": assessment.support_breadth_temperature,
                "relative_corrective_headroom": assessment.relative_corrective_headroom,
                "corrective_headroom_kappa": assessment.corrective_headroom_kappa,
                "absolute_recovery_depth": assessment.absolute_recovery_depth,
                "need_state_hazard": assessment.need_state_hazard,
                "need_pre_screen_hazard": assessment.need_pre_screen_hazard,
                "need_metric_source": assessment.need_metric_source,
                "latency_survival_floor": assessment.latency_survival_floor,
                "maneuver_breadth_floor": assessment.maneuver_breadth_floor,
                "corrective_headroom_floor": assessment.corrective_headroom_floor,
                "state_need_floor": assessment.state_need_floor,
                "latency_survival_pass": assessment.latency_survival_pass,
                "maneuver_breadth_pass": assessment.maneuver_breadth_pass,
                "corrective_headroom_pass": assessment.corrective_headroom_pass,
                "state_need_pass": assessment.state_need_pass,
                "serial_gate_pass": assessment.serial_gate_pass,
            }.items()
        },
    }

    decision = build_decision_meta(source, proposed_action=0, final_action=0)
    event = build_episode_event(0, {}, decision, {})

    for payload in (decision, event):
        assert payload["rgd_method_version"] == METHOD_VERSION
        assert payload["recoverability_gate_action_universe"] == [0, 1, 3, 4]
        assert payload["recoverability_fast_executor_action_universe"] == [0, 1, 3, 4]
        assert payload["recoverability_relative_corrective_headroom"] == pytest.approx(
            assessment.relative_corrective_headroom
        )
        assert payload["recoverability_alternative_maneuver_family_count"] == 2
        assert payload["recoverability_action_maneuver_family_mapping"] == {
            "0": "lateral-left",
            "1": "lane-hold",
            "3": "longitudinal-accelerate",
            "4": "longitudinal-decelerate",
        }
        assert payload["recoverability_support_diagnostic_effective_mass"] == pytest.approx(
            assessment.support_diagnostic_effective_mass
        )
        assert payload["recoverability_support_cost_complete"] is True
        assert payload["recoverability_support_family_min_costs"] == {
            "lane-hold": pytest.approx(0.20),
            "longitudinal-accelerate": pytest.approx(0.30),
        }
        assert payload["recoverability_support_breadth_formula"] == (
            "sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families"
        )
        assert payload["recoverability_latency_survival_floor"] == pytest.approx(
            assessment.latency_survival_floor
        )
        assert payload["recoverability_maneuver_breadth_pass"] is True
        assert payload["recoverability_serial_gate_pass"] is True
        assert payload["proposed_action"] == 0
        assert payload["final_action"] == 0
