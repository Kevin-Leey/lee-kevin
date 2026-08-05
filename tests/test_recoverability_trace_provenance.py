from dilu.driver_agent.reasoning.decision import RouteAmbiguityProfile
from dilu.evaluation.decision_trace import build_decision_meta
from dilu.runtime_frame_trace import build_episode_event


def test_route_profile_serializes_raw_and_support_costs_separately():
    profile = RouteAmbiguityProfile(
        action_probabilities={0: 0.7, 1: 0.3},
        action_recovery_costs={0: 0.2, 1: 0.3},
        action_support_ranking_costs={0: 0.8, 1: 0.3},
        probability_cost_source="action_support_ranking_costs",
        ambiguity_best_action=1,
        selected_probability=0.7,
        ambiguity_entropy=0.5,
        ambiguity_gap=0.4,
        evidence_disagreement=0.0,
        intervention_risk=0.2,
        executed_action=0,
    )

    payload = profile.to_dict()

    assert payload["action_recovery_costs"] == {"0": 0.2, "1": 0.3}
    assert payload["action_support_ranking_costs"] == {"0": 0.8, "1": 0.3}
    assert payload["probability_cost_source"] == "action_support_ranking_costs"
    assert payload["executed_action"] == 0


def test_trace_builders_preserve_gate_provenance_and_action_zero():
    source = {
        "system_used": "fast",
        "proposed_action": 0,
        "final_action": 0,
        "recoverability_hold_action": 0,
        "recoverability_alternative_support_count": 2,
        "recoverability_alternative_support_ratio": 0.5,
        "recoverability_absolute_alternative_count": 3,
        "recoverability_absolute_alternative_ratio": 0.75,
        "recoverability_absolute_alternative_feasible": True,
        "recoverability_alternative_metric_source": "action_support_ranking_costs",
        "recoverability_headroom_metric_source": "action_recovery_costs",
        "recoverability_viable_cost_threshold": 0.55,
        "recoverability_gate": {
            "alternative_support_count": 2,
            "alternative_support_ratio": 0.5,
            "absolute_alternative_count": 3,
            "absolute_alternative_ratio": 0.75,
            "absolute_alternative_feasible": True,
            "alternative_metric_source": "action_support_ranking_costs",
            "headroom_metric_source": "action_recovery_costs",
            "viable_cost_threshold": 0.55,
        },
    }

    decision = build_decision_meta(source, proposed_action=0, final_action=0)
    event = build_episode_event(0, {}, decision, {})

    assert decision["recoverability_hold_action"] == 0
    assert event["proposed_action"] == 0
    assert event["final_action"] == 0
    assert event["recoverability_hold_action"] == 0
    assert event["recoverability_alternative_support_count"] == 2
    assert event["recoverability_absolute_alternative_count"] == 3
    assert event["recoverability_absolute_alternative_feasible"] is True
    assert event["recoverability_alternative_metric_source"] == "action_support_ranking_costs"
    assert event["recoverability_headroom_metric_source"] == "action_recovery_costs"
    assert event["recoverability_viable_cost_threshold"] == 0.55
