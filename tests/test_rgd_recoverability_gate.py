import math
import unittest

from dilu.driver_agent.base.state import DrivingState
from dilu.driver_agent.reasoning.rgd_support import (
    PaperBaselineDefinition,
    RecoverabilityGateDefinition,
    build_recoverability_public_signal,
    build_slow_path_latency_context,
    compute_recoverability_assessment,
    compute_recoverability_gate_diagnostics,
    compute_temporal_survival,
    resolve_closed_recoverability_route,
    resolve_release_dominance_guard,
    resolve_paper_baseline_trigger,
)
from dilu.driver_agent.reasoning.decision import RouteAmbiguityProfile
from tools.analyze_common_trajectory_allocators import gate_values


class CorrectedRecoverabilityGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = RecoverabilityGateDefinition.from_core_story_config({})
        self.state = DrivingState(
            ego_speed=20.0,
            front_distance=80.0,
            front_relative_speed=0.0,
            ttc=6.0,
            thw=4.0,
            scenario_type="highway",
        )
        self.principle = {
            "corridor_width": 2,
            "recovery_budget_remaining": 0.7,
            "dominance_margin": 0.4,
            "principle_satisfied": True,
            "reachable_safe_set_ratio": 0.6,
            "viable_headroom": 0.7,
            "short_horizon_irreversible_risk": 0.8,
        }
        self.rad_meta = {
            "best_action": 3,
            "action_recovery_costs": {1: 0.70, 3: 0.30, 4: 0.72},
            "action_support_ranking_costs": {1: 0.70, 3: 0.30, 4: 0.72},
            "state_hazard_score": 0.8,
            "short_horizon_irreversible_risk": 0.8,
            "prediction_horizon_s": 0.8,
        }
        self.pre_screen = {
            "pre_screen_score": 0.0,
            "pre_screen_trigger": False,
            "pre_screen_reason": "none",
            "soft_recoverability_floor": 0.20,
        }

    def latency(self, seconds, state=None):
        return build_slow_path_latency_context(
            llm_available=True,
            llm_invoke_timeout_s=30.0,
            short_horizon_seconds=0.8,
            state=state or self.state,
            predicted_slow_latency_s=seconds,
            latency_source="unit_test",
            policy_frequency=10.0,
        )

    def assess(self, latency, *, rad_meta=None, pre_screen=None, gate=None, legal=(1, 3, 4), hold=1):
        return compute_recoverability_assessment(
            gate_definition=gate or self.gate,
            principle=self.principle,
            rad_meta=rad_meta or self.rad_meta,
            latency_context=latency,
            pre_screen_context=pre_screen or self.pre_screen,
            legal_actions=tuple(legal),
            hold_action=hold,
        )

    def route(self, assessment, threshold=0.0):
        diagnostics = compute_recoverability_gate_diagnostics(
            self.gate,
            self.principle,
            self.rad_meta,
            recoverability_assessment=assessment,
        )
        return resolve_closed_recoverability_route(
            {}, assessment, diagnostics, decision_threshold_override=threshold
        )

    def test_configured_latency_is_discretized_consistently(self):
        context = self.latency(1.7)
        self.assertEqual(context["effective_delay_steps"], 17)
        self.assertAlmostEqual(context["predicted_slow_latency_seconds"], 1.7)
        self.assertEqual(context["latency_source"], "unit_test")

    def test_valid_zero_step_survives_a_zero_deadline(self):
        occupied = DrivingState(
            ego_speed=10.0,
            scenario_type="intersection",
            junction_gap={"occupied_corridor": True},
        )
        context = self.latency(0.0, occupied)
        self.assertEqual(context["critical_latency_seconds"], 0.0)
        self.assertEqual(context["effective_delay_steps"], 0)
        self.assertEqual(context["recovery_window"], 1.0)

    def test_release_dominance_guard_retains_only_strictly_lower_risk_slow_action(self):
        resolved, meta = resolve_release_dominance_guard(
            slow_action=2,
            matched_fast_action=0,
            risk_scores={0: 0.40, 2: 0.20},
        )
        self.assertEqual(resolved, 2)
        self.assertTrue(meta["release_dominance_guard_retained_slow"])
        self.assertAlmostEqual(meta["release_dominance_guard_risk_gain"], 0.20)

        resolved, meta = resolve_release_dominance_guard(
            slow_action=1,
            matched_fast_action=0,
            risk_scores={0: 0.15, 1: 0.50},
        )
        self.assertEqual(resolved, 0)
        self.assertTrue(meta["release_dominance_guard_fallback_to_fast"])

    def test_release_dominance_guard_ties_and_missing_scores_fail_closed(self):
        tied, tied_meta = resolve_release_dominance_guard(
            slow_action=2,
            matched_fast_action=0,
            risk_scores={0: 0.25, 2: 0.25},
        )
        self.assertEqual(tied, 0)
        self.assertEqual(
            tied_meta["release_dominance_guard_reason"],
            "slow_not_risk_dominant_fast_fallback",
        )

        missing, missing_meta = resolve_release_dominance_guard(
            slow_action=2,
            matched_fast_action=0,
            risk_scores={2: 0.10},
        )
        self.assertEqual(missing, 0)
        self.assertEqual(
            missing_meta["release_dominance_guard_reason"],
            "missing_risk_score_fast_fallback",
        )

    def test_progress_guard_vetoes_idle_or_braking_on_clear_fast_path(self):
        """A lower instantaneous slow risk must not discard unneeded progress.

        This is the counterfactual that motivated the guard: both actions are
        evaluated on the same risk surface, but a clear lead and non-critical
        TTC/THW make an ``IDLE``/``SLOWER`` replacement of ``FASTER`` a
        longitudinal-progress regression.  The guard should return the exact
        matched-fast action and expose a dedicated reason for auditability.
        """
        progress_state = {
            "speed": 20.0,
            "front_distance": 50.0,
            "ttc": 5.0,
            "thw": 2.5,
        }
        for slow_action in (1, 4):
            resolved, meta = resolve_release_dominance_guard(
                slow_action=slow_action,
                matched_fast_action=3,
                risk_scores={slow_action: 0.10, 3: 0.45},
                progress_guard=progress_state,
            )
            self.assertEqual(resolved, 3)
            self.assertFalse(meta["release_dominance_guard_retained_slow"])
            self.assertTrue(meta["release_dominance_guard_progress_fallback"])
            self.assertEqual(
                meta["release_dominance_guard_progress_reason"],
                "slow_longitudinal_progress_regression",
            )
            self.assertEqual(
                meta["release_dominance_guard_reason"],
                "slow_progress_regression_fast_fallback",
            )

    def test_progress_guard_vetoes_braking_when_matched_fast_is_idle_on_open_lead(self):
        """An unnecessary brake must not replace an otherwise valid IDLE path.

        Phase two of the progress guard covers the common case in which the
        contemporaneous fast controller is already cruising (``IDLE``).  A
        delayed ``SLOWER`` proposal can have a lower instantaneous risk score,
        yet still throw away useful longitudinal progress when the lead is
        open and both TTC and THW are comfortably non-critical.  In that case
        the guard returns the exact matched-fast action.
        """
        open_lead = {
            "speed": 20.0,
            "front_distance": 40.0,
            "ttc": 7.0,
            "thw": 1.5,
        }
        resolved, meta = resolve_release_dominance_guard(
            slow_action=4,  # SLOWER
            matched_fast_action=1,  # IDLE
            risk_scores={4: 0.10, 1: 0.45},
            progress_guard=open_lead,
        )
        self.assertEqual(resolved, 1)
        self.assertFalse(meta["release_dominance_guard_retained_slow"])
        self.assertTrue(meta["release_dominance_guard_fallback_to_fast"])
        self.assertTrue(meta["release_dominance_guard_progress_fallback"])
        self.assertEqual(
            meta["release_dominance_guard_progress_reason"],
            "slow_brake_without_lead_pressure",
        )
        self.assertEqual(
            meta["release_dominance_guard_reason"],
            "slow_progress_regression_fast_fallback",
        )

    def test_progress_guard_keeps_slower_when_following_is_urgent(self):
        """Urgent following pressure must preserve a safer SLOWER proposal.

        A small gap and critical TTC/THW make the progress-preservation
        precondition false.  The guard therefore defers to the shared risk
        comparison and retains SLOWER when it is strictly safer than IDLE.
        """
        urgent_following = {
            "speed": 20.0,
            "front_distance": 12.0,
            "ttc": 1.8,
            "thw": 0.55,
        }
        resolved, meta = resolve_release_dominance_guard(
            slow_action=4,  # SLOWER
            matched_fast_action=1,  # IDLE
            risk_scores={4: 0.10, 1: 0.45},
            progress_guard=urgent_following,
        )
        self.assertEqual(resolved, 4)
        self.assertTrue(meta["release_dominance_guard_retained_slow"])
        self.assertFalse(meta["release_dominance_guard_fallback_to_fast"])
        self.assertFalse(meta["release_dominance_guard_progress_fallback"])
        self.assertEqual(
            meta["release_dominance_guard_progress_reason"],
            "no_clear_progress_regression",
        )
        self.assertEqual(meta["release_dominance_guard_reason"], "slow_risk_dominates")

    def test_progress_guard_does_not_veto_lane_change_or_nonclear_path(self):
        """Lateral escape remains eligible, and a constrained lead is not clear."""
        clear = {
            "speed": 20.0,
            "front_distance": 50.0,
            "ttc": 5.0,
            "thw": 2.5,
        }
        lane_change, lane_meta = resolve_release_dominance_guard(
            slow_action=0,
            matched_fast_action=3,
            risk_scores={0: 0.10, 3: 0.45},
            progress_guard=clear,
        )
        self.assertEqual(lane_change, 0)
        self.assertTrue(lane_meta["release_dominance_guard_retained_slow"])
        self.assertFalse(lane_meta["release_dominance_guard_progress_fallback"])

        constrained = dict(clear, front_distance=23.9)
        retained, retained_meta = resolve_release_dominance_guard(
            slow_action=1,
            matched_fast_action=3,
            risk_scores={1: 0.10, 3: 0.45},
            progress_guard=constrained,
        )
        self.assertEqual(retained, 1)
        self.assertTrue(retained_meta["release_dominance_guard_retained_slow"])
        self.assertEqual(
            retained_meta["release_dominance_guard_progress_reason"],
            "no_clear_progress_regression",
        )

    def test_progress_guard_missing_metrics_preserves_risk_only_decision(self):
        """Incomplete kinematic provenance must not manufacture a veto."""
        resolved, meta = resolve_release_dominance_guard(
            slow_action=1,
            matched_fast_action=3,
            risk_scores={1: 0.10, 3: 0.45},
            progress_guard={"speed": 20.0},
        )
        self.assertEqual(resolved, 1)
        self.assertTrue(meta["release_dominance_guard_retained_slow"])
        self.assertFalse(meta["release_dominance_guard_progress_fallback"])
        self.assertEqual(
            meta["release_dominance_guard_progress_reason"],
            "no_clear_progress_regression",
        )

    def test_positive_delay_retains_deadline_margin_formula(self):
        survival = compute_temporal_survival(
            critical_latency_seconds=2.0,
            effective_delay_steps=5,
            policy_frequency=10.0,
            latency_prediction_available=True,
            execution_available=True,
            latency_source="unit_test",
            safety_reserve_seconds=0.2,
        )
        self.assertAlmostEqual(survival, (2.0 - 0.5 - 0.2) / 2.0)

    def test_zero_step_cannot_bypass_missing_provenance_or_executor(self):
        base = {
            "critical_latency_seconds": 0.0,
            "effective_delay_steps": 0,
            "policy_frequency": 10.0,
            "latency_prediction_available": True,
            "execution_available": True,
            "latency_source": "unit_test",
        }
        self.assertEqual(compute_temporal_survival(**dict(base, latency_source="unknown")), 0.0)
        self.assertEqual(compute_temporal_survival(**dict(base, execution_available=False)), 0.0)
        self.assertEqual(
            compute_temporal_survival(**dict(base, latency_prediction_available=False)),
            0.0,
        )

    def test_offline_allocator_uses_runtime_temporal_survival(self):
        context = self.latency(0.0, DrivingState(
            ego_speed=10.0,
            scenario_type="intersection",
            junction_gap={"occupied_corridor": True},
        ))
        gate = {
            "latency_prediction_available": context["latency_prediction_available"],
            "llm_backed_execution_available": context["llm_backed_execution_available"],
            "alternative_viable_ratio": 0.64,
            "cost_headroom": 0.81,
            "need_score": 0.50,
            "alternative_viable_count": 1,
            "latency": {
                "critical_latency_seconds": context["critical_latency_seconds"],
                "policy_frequency": context["policy_frequency"],
                "source": context["latency_source"],
            },
        }
        record = {
            "rgd_subordinate_diagnostics": {
                "recoverability_signal": {"recoverability_gate": gate}
            }
        }
        opportunity, priority, alternatives = gate_values(record, 0.0)
        self.assertAlmostEqual(opportunity, 0.72)
        self.assertAlmostEqual(priority, 0.36)
        self.assertEqual(alternatives, 1)

        gate["latency"]["source"] = "unknown"
        self.assertEqual(gate_values(record, 0.0)[:2], (0.0, 0.0))

    def test_latency_survival_and_scores_are_monotone(self):
        assessments = [self.assess(self.latency(value)) for value in (0.0, 0.8, 1.7, 2.5)]
        for field in ("recovery_window", "post_latency_opportunity", "recoverability_score"):
            values = [getattr(item, field) for item in assessments]
            self.assertTrue(all(left >= right for left, right in zip(values, values[1:])), (field, values))

    def test_latency_at_deadline_fails_closed_despite_risk(self):
        urgent = DrivingState(ego_speed=20.0, front_distance=15.0, front_relative_speed=-5.0, ttc=1.0, scenario_type="highway")
        high_pre = dict(self.pre_screen, pre_screen_score=1.0, pre_screen_trigger=True, pre_screen_reason="ttc")
        assessment = self.assess(self.latency(1.7, urgent), pre_screen=high_pre)
        self.assertEqual(assessment.recovery_window, 0.0)
        self.assertEqual(assessment.post_latency_opportunity, 0.0)
        self.assertFalse(assessment.opportunity_eligible)
        self.assertEqual(self.route(assessment).selected_system, "fast")

    def test_no_viable_non_hold_alternative_fails_closed(self):
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.30, 3: 0.80, 4: 0.90},
            action_support_ranking_costs={1: 0.30, 3: 0.80, 4: 0.90},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, hold=1)
        self.assertEqual(assessment.alternative_viable_count, 0)
        self.assertEqual(assessment.post_latency_opportunity, 0.0)
        self.assertEqual(self.route(assessment).selected_system, "fast")

    def test_relative_support_weights_maneuver_family_breadth(self):
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.30, 3: 0.30, 4: 0.30},
            action_support_ranking_costs={1: 0.30, 3: 0.80, 4: 0.30},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, hold=1)
        self.assertEqual(assessment.absolute_alternative_count, 2)
        self.assertEqual(assessment.alternative_viable_count, 2)
        self.assertAlmostEqual(
            assessment.alternative_viable_ratio,
            (1.0 + math.exp(-5.0)) / 2.0,
        )
        self.assertTrue(assessment.support_diagnostic_complete)
        self.assertLess(assessment.support_diagnostic_effective_mass, 2.0)

    def test_absolute_viability_cannot_be_created_by_support_ranking(self):
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.30, 3: 0.80, 4: 0.90},
            action_support_ranking_costs={1: 0.30, 3: 0.10, 4: 0.10},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, hold=1)
        self.assertEqual(assessment.absolute_alternative_count, 0)
        self.assertEqual(assessment.alternative_viable_count, 0)
        self.assertFalse(assessment.opportunity_eligible)
        self.assertEqual(self.route(assessment).selected_system, "fast")

    def test_nonfinite_raw_cost_cannot_create_headroom_or_support(self):
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.30, 3: float("nan"), 4: 0.90},
            action_support_ranking_costs={1: 0.30, 3: 0.10, 4: 0.90},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, hold=1)
        self.assertEqual(assessment.absolute_alternative_count, 0)
        self.assertEqual(assessment.alternative_viable_count, 0)
        self.assertFalse(assessment.opportunity_eligible)

    def test_illegal_low_cost_action_does_not_create_eligibility(self):
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.90, 3: 0.10, 4: 0.90},
            action_support_ranking_costs={1: 0.90, 3: 0.10, 4: 0.90},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, legal=(1, 4), hold=1)
        self.assertEqual(assessment.alternative_viable_count, 0)
        self.assertFalse(assessment.opportunity_eligible)

    def test_pre_screen_cannot_bypass_floor(self):
        high_pre = dict(self.pre_screen, pre_screen_score=1.0, pre_screen_trigger=True, pre_screen_reason="cross_traffic")
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.90, 3: 0.90, 4: 0.90},
            action_support_ranking_costs={1: 0.90, 3: 0.90, 4: 0.90},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, pre_screen=high_pre)
        self.assertEqual(self.route(assessment).selected_system, "fast")

    def test_explicit_latency_floor_controls_route(self):
        assessment = self.assess(self.latency(1.7))
        self.assertGreaterEqual(assessment.post_latency_opportunity, 0.20)
        self.assertEqual(self.route(assessment, threshold=0.20).selected_system, "slow")
        legacy_floor = dict(self.pre_screen, soft_recoverability_floor=0.95)
        legacy_ignored = self.assess(self.latency(1.7), pre_screen=legacy_floor)
        self.assertEqual(self.route(legacy_ignored, threshold=0.0).selected_system, "slow")
        raised_gate = RecoverabilityGateDefinition.from_core_story_config(
            {"rgd_latency_survival_floor": 0.95}
        )
        blocked = self.assess(self.latency(1.7), gate=raised_gate)
        self.assertEqual(self.route(blocked, threshold=0.0).selected_system, "fast")

    def test_missing_latency_prediction_fails_closed(self):
        context = build_slow_path_latency_context(
            llm_available=True,
            llm_invoke_timeout_s=30.0,
            short_horizon_seconds=0.8,
            state=self.state,
        )
        assessment = self.assess(context)
        self.assertFalse(context["latency_prediction_available"])
        self.assertEqual(context["recovery_window"], 0.0)
        self.assertFalse(assessment.opportunity_eligible)

    def test_need_only_still_respects_absolute_action_feasibility(self):
        need_gate = RecoverabilityGateDefinition.from_core_story_config({"rgd_route_component_mode": "need_only"})
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.90, 3: 0.90, 4: 0.90},
            action_support_ranking_costs={1: 0.90, 3: 0.90, 4: 0.90},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, gate=need_gate)
        diagnostics = compute_recoverability_gate_diagnostics(
            need_gate, self.principle, costs, recoverability_assessment=assessment
        )
        decision = resolve_closed_recoverability_route(
            {}, assessment, diagnostics, decision_threshold_override=0.50
        )
        self.assertEqual(assessment.post_latency_opportunity, 0.0)
        self.assertFalse(assessment.absolute_alternative_feasible)
        self.assertEqual(decision.selected_system, "fast")

    def test_missing_action_support_costs_close_a_without_changing_raw_f(self):
        costs = {
            "best_action": 3,
            "action_recovery_costs": {1: 0.30, 3: 0.30, 4: 0.30},
            "short_horizon_irreversible_risk": 0.8,
            "prediction_horizon_s": 0.8,
        }
        assessment = self.assess(self.latency(0.0), rad_meta=costs, hold=1)
        self.assertEqual(
            assessment.alternative_metric_source,
            "relative_support_weighted_maneuver_family_breadth",
        )
        self.assertEqual(assessment.alternative_viable_count, 2)
        self.assertFalse(assessment.support_diagnostic_complete)
        self.assertFalse(assessment.support_cost_complete)
        self.assertTrue(assessment.absolute_alternative_feasible)
        self.assertTrue(assessment.domain_contract_pass)
        self.assertFalse(assessment.gate_fail_closed)
        self.assertFalse(assessment.maneuver_breadth_pass)
        self.assertIn("maneuver_breadth", assessment.serial_gate_failed_components)
        self.assertFalse(assessment.opportunity_eligible)

    def test_need_only_can_remove_soft_support_without_removing_feasibility(self):
        need_gate = RecoverabilityGateDefinition.from_core_story_config({"rgd_route_component_mode": "need_only"})
        costs = dict(
            self.rad_meta,
            action_recovery_costs={1: 0.30, 3: 0.30, 4: 0.30},
            action_support_ranking_costs={1: 0.30, 3: 0.90, 4: 0.90},
        )
        assessment = self.assess(self.latency(0.0), rad_meta=costs, gate=need_gate)
        diagnostics = compute_recoverability_gate_diagnostics(
            need_gate, self.principle, costs, recoverability_assessment=assessment
        )
        decision = resolve_closed_recoverability_route(
            {}, assessment, diagnostics, decision_threshold_override=0.50
        )
        self.assertTrue(assessment.absolute_alternative_feasible)
        self.assertFalse(assessment.opportunity_eligible)
        self.assertEqual(decision.selected_system, "slow")

    def test_public_coordinates_equal_internal_gate_terms(self):
        assessment = self.assess(self.latency(1.7))
        public = assessment.to_paper_dict()
        self.assertAlmostEqual(public["recovery_window"], assessment.recovery_window)
        self.assertAlmostEqual(public["action_space_affordance"], assessment.alternative_viable_ratio)
        self.assertAlmostEqual(public["commitment_reversibility"], assessment.cost_headroom)
        self.assertAlmostEqual(public["soft_recoverability"], assessment.post_latency_opportunity)
        self.assertAlmostEqual(public["recoverable_deliberation_priority"], assessment.recoverability_score)

    def test_public_signal_prefers_canonical_fields_over_legacy_aliases(self):
        public = build_recoverability_public_signal(
            {
                "recovery_window": 0.01,
                "recoverability_recovery_window": 0.70,
                "action_space_affordance": 0.02,
                "recoverability_alternative_viable_ratio": 0.80,
                "commitment_reversibility": 0.03,
                "recoverability_cost_headroom": 0.90,
                "soft_recoverability": 0.04,
                "recoverability_post_latency_opportunity": 0.60,
                "recoverable_deliberation_priority": 0.95,
                "recoverability_score": 0.20,
            }
        )
        self.assertAlmostEqual(public["recovery_window"], 0.70)
        self.assertAlmostEqual(public["action_space_affordance"], 0.80)
        self.assertAlmostEqual(public["commitment_reversibility"], 0.90)
        self.assertAlmostEqual(public["soft_recoverability"], 0.60)
        self.assertAlmostEqual(public["recoverable_deliberation_priority"], 0.20)

    def test_baseline_exposure_reads_orchestrator_risk_config(self):
        core = {
            "paper_baseline_trigger_mode": "uncertainty",
            "paper_baseline_uncertainty_cutoff": 1.0,
            "paper_baseline_exposure_probability": 0.0,
        }
        baseline = PaperBaselineDefinition.from_core_story_config(core)
        profile = RouteAmbiguityProfile(
            action_probabilities={1: 1.0},
            action_recovery_costs={1: 0.2},
            ambiguity_best_action=1,
            selected_probability=1.0,
            ambiguity_entropy=1.0,
            ambiguity_gap=0.0,
            evidence_disagreement=0.0,
            intervention_risk=0.0,
        )
        stats = {"decision_count": 7}
        result = resolve_paper_baseline_trigger(
            {"risk_coupling": {"core_story": core}, "fixed_seed_override": 31},
            stats,
            baseline,
            profile,
            {"ttc_pressure": 0.0, "proximity_complexity": 0.0},
            0.2,
        )
        self.assertEqual(result[0], "fast")
        self.assertEqual(stats["paper_baseline_exposure_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
