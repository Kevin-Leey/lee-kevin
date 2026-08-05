import copy
import math
from pathlib import Path

import pytest
import yaml

from tools.analyze_common_trajectory_allocators import (
    gate_component_values,
    validate_support_breadth_v11_record,
)
from tools.analyze_rgd_component_ablation import (
    ARM_SPECS,
    _paper_acceptance,
    _validate_protocol_contract,
    _validate_source_manifest,
    component_scores,
)
from tools.verify_rgd_component_ablation import close_optional, verify_paper_acceptance


COMPONENTS = {
    "latency_survival": 0.5,
    "admissible_alternative_fraction": 0.25,
    "recovery_headroom": 0.64,
    "need_score": 0.8,
    "alternative_count": 2,
    "absolute_alternative_count": 2,
    "raw_feasibility_valid": True,
    "support_cost_complete": True,
}


def arm(label):
    return next(spec for spec in ARM_SPECS if spec.label == label)


def test_full_gate_reproduces_frozen_formula():
    result = component_scores(COMPONENTS, arm("RGD"))
    assert math.isclose(result["ablated_opportunity"], 0.2)
    assert math.isclose(result["ablated_priority"], 0.16)
    assert result["eligible"] is True


def test_leave_one_out_uses_multiplicative_identity():
    no_l = component_scores(COMPONENTS, arm("RGD w/o L"))
    no_a = component_scores(COMPONENTS, arm("RGD w/o A"))
    no_h = component_scores(COMPONENTS, arm("RGD w/o H"))
    assert math.isclose(no_l["ablated_opportunity"], 0.4)
    assert math.isclose(no_a["ablated_opportunity"], 0.4)
    assert math.isclose(no_h["ablated_opportunity"], 0.25)


def test_removing_a_also_removes_its_hard_count_check():
    components = dict(COMPONENTS, alternative_count=0)
    assert component_scores(components, arm("RGD"))["eligible"] is False
    assert component_scores(components, arm("RGD w/o A"))["hard_alternative_ok"] is True


def test_absolute_action_feasibility_remains_fail_closed_in_every_arm():
    components = dict(COMPONENTS, absolute_alternative_count=0)
    for spec in ARM_SPECS:
        result = component_scores(components, spec)
        assert result["absolute_alternative_ok"] is False
        assert result["eligible"] is False


def test_offline_support_failure_matches_runtime_remove_only_a_semantics():
    components = dict(
        COMPONENTS,
        admissible_alternative_fraction=0.0,
        alternative_count=0,
        support_cost_complete=False,
    )

    full = component_scores(components, arm("RGD"))
    no_a = component_scores(components, arm("RGD w/o A"))

    assert full["raw_feasibility_ok"] is True
    assert full["support_evidence_ok"] is False
    assert full["eligible"] is False
    assert no_a["raw_feasibility_ok"] is True
    assert no_a["support_evidence_check_active"] is False
    assert no_a["support_evidence_ok"] is True
    assert no_a["eligible"] is True


def test_raw_feasibility_failure_remains_non_ablatable_offline():
    components = dict(COMPONENTS, raw_feasibility_valid=False)

    for spec in ARM_SPECS:
        result = component_scores(components, spec)
        assert result["raw_feasibility_ok"] is False
        assert result["eligible"] is False


def test_factorial_design_contains_each_binary_mask_once():
    masks = {(spec.use_l, spec.use_a, spec.use_h) for spec in ARM_SPECS}
    assert len(ARM_SPECS) == 8
    assert len(masks) == 8


def v11_record(*, exact_domain=True):
    threshold = 0.55
    recovery = {0: 0.9, 1: 0.3, 2: 0.4, 3: 0.7}
    support = {0: 0.9, 1: 0.3, 2: 0.5, 3: 0.4}
    gate = {
        "threshold": 0.15,
        "opportunity_floor": 0.20,
        "need_score": 0.8,
        "latency_prediction_available": True,
        "latency": {
            "critical_latency_seconds": 6.0,
            "source": "closed_loop_latency_replay.extra_latency_s",
        },
        "alternative_viable_count": 1,
        "alternative_viable_ratio": 0.5,
        "alternative_support_count": 1,
        "alternative_support_ratio": 0.5,
        "absolute_alternative_count": 1,
        "absolute_alternative_ratio": 0.5,
        "absolute_alternative_feasible": True,
        "alternative_metric_source": "action_support_ranking_costs",
        "headroom_metric_source": "action_recovery_costs",
        "viable_cost_threshold": threshold,
        "cost_headroom": (threshold - 0.3) / threshold,
        "hold_action": 1,
    }
    if exact_domain:
        gate["gate_legal_actions"] = [1, 2, 3]
    return {
        "schema_version": "rgd_record_v3",
        "available_actions": "Action_id: 0 Action_id: 1 Action_id: 2 Action_id: 3",
        "rgd_subordinate_diagnostics": {
            "recoverability_signal": {"recoverability_gate": gate},
            "ambiguity_and_conflict": {
                "route_ambiguity_profile": {
                    "action_recovery_costs": recovery,
                    "action_support_ranking_costs": support,
                    "probability_cost_source": "action_support_ranking_costs",
                }
            },
        },
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(alternative_metric_source="action_recovery_costs"),
        lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(headroom_metric_source="unknown"),
        lambda record: record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"]["route_ambiguity_profile"].update(probability_cost_source="action_recovery_costs"),
        lambda record: record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"]["route_ambiguity_profile"].update(action_support_ranking_costs={}),
        lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].pop("absolute_alternative_count"),
        lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(alternative_viable_count=2, alternative_support_count=2),
    ),
)
def test_strict_v11_gate_rejects_legacy_or_inconsistent_provenance(mutation):
    record = v11_record()
    mutation(record)
    with pytest.raises(ValueError):
        gate_component_values(record, 1.7, require_support_breadth_v11=True)


def test_v11_gate_distinguishes_exact_and_aggregate_action_provenance():
    exact = validate_support_breadth_v11_record(v11_record())
    aggregate = validate_support_breadth_v11_record(v11_record(exact_domain=False))
    assert exact["legal_action_provenance"] == "exact"
    assert exact["verified_action_domains"] == ((1, 2, 3),)
    assert aggregate["legal_action_provenance"] == "aggregate_only"
    assert len(aggregate["verified_action_domains"]) > 1


def test_component_scores_requires_explicit_absolute_feasibility():
    components = dict(COMPONENTS)
    del components["absolute_alternative_count"]
    with pytest.raises(KeyError):
        component_scores(components, arm("RGD w/o A"))


def protocol_payload():
    return {
        "tvt_submission_contract": {
            "rgd_method_version": "support_breadth_v11",
            "component_ablation": {
                "components": [
                    "latency_survival",
                    "action_support_breadth",
                    "recovery_headroom",
                ],
                "seed_range": {"start": 1000, "end": 1029, "count": 30},
                "delay_s": 1.7,
                "horizon_steps": 20,
                "gamma": 0.99,
                "corrective_margin": 0.02,
                "opportunity_floor": 0.20,
                "priority_threshold": 0.15,
                "budget": 6,
                "cooldown_frames": 20,
                "removed_component_value": 1.0,
                "remove_support_hard_gate_when_A_removed": True,
                "absolute_alternative_feasibility_non_ablatable": True,
                "alternative_metric_source": "action_support_ranking_costs",
                "headroom_metric_source": "action_recovery_costs",
                "viable_cost_threshold": 0.55,
                "threshold_policy": "fixed_across_cells_without_retuning",
                "bootstrap_draws": 20000,
            },
        },
        "runtime_config": {
            "rgd_decision_threshold": 0.15,
            "slow_call_budget": 6,
            "slow_call_cooldown_frames": 20,
        },
    }


def write_protocol(path: Path, payload=None):
    path.write_text(yaml.safe_dump(payload or protocol_payload()), encoding="utf-8")


def validation_kwargs():
    return {
        "seeds": list(range(1000, 1030)),
        "delay_s": 1.7,
        "horizon": 20,
        "gamma": 0.99,
        "epsilon": 0.02,
        "bootstrap_draws": 20000,
    }


def source_manifest(trace_root: Path, protocol_path: Path):
    return {
        "trace_root": str(trace_root.resolve()),
        "protocol": str(protocol_path.resolve()),
        "seeds": list(range(1000, 1030)),
        "seed_is_experimental_unit": True,
        "method_version": "support_breadth_v11",
        "alternative_metric_source": "action_support_ranking_costs",
        "headroom_metric_source": "action_recovery_costs",
        "absolute_alternative_feasibility_non_ablatable": True,
        "viable_cost_threshold": 0.55,
        "delays_s": [0.7, 1.7, 2.7],
        "horizon_steps": 20,
        "gamma": 0.99,
        "epsilon": 0.02,
        "policy_frequency_hz": 10,
        "bootstrap_draws": 20000,
        "bootstrap_seed": 20260717,
    }


def test_protocol_and_source_contracts_fail_closed(tmp_path):
    protocol_path = tmp_path / "formal.yaml"
    trace_root = tmp_path / "traces"
    write_protocol(protocol_path)
    _validate_protocol_contract(protocol_path, **validation_kwargs())
    source = source_manifest(trace_root, protocol_path)
    _validate_source_manifest(
        source,
        trace_root=trace_root,
        protocol_path=protocol_path,
        **validation_kwargs(),
    )

    for field, bad_value in (
        ("method_version", "raw_affordance_v10"),
        ("alternative_metric_source", "action_recovery_costs"),
        ("headroom_metric_source", "unknown"),
        ("absolute_alternative_feasibility_non_ablatable", False),
        ("viable_cost_threshold", 0.50),
        ("delays_s", [0.7, 2.7]),
    ):
        tampered = copy.deepcopy(source)
        tampered[field] = bad_value
        with pytest.raises(ValueError):
            _validate_source_manifest(
                tampered,
                trace_root=trace_root,
                protocol_path=protocol_path,
                **validation_kwargs(),
            )

    tampered_protocol = protocol_payload()
    tampered_protocol["tvt_submission_contract"]["component_ablation"]["priority_threshold"] = 0.16
    write_protocol(protocol_path, tampered_protocol)
    with pytest.raises(ValueError):
        _validate_protocol_contract(protocol_path, **validation_kwargs())


def ablation_summary(full=0.60, no_l=0.59, no_a=0.50, no_h=0.40):
    return [
        {"arm": "RGD", "corrective_set_fraction": full},
        {"arm": "RGD w/o L", "corrective_set_fraction": no_l},
        {"arm": "RGD w/o A", "corrective_set_fraction": no_a},
        {"arm": "RGD w/o H", "corrective_set_fraction": no_h},
    ]


def test_paper_acceptance_requires_strict_superiority_and_exact_provenance():
    passing = _paper_acceptance(ablation_summary(), legal_action_provenance="exact")
    assert passing["metric_passed"] is True
    assert passing["passed"] is True

    aggregate = _paper_acceptance(
        ablation_summary(), legal_action_provenance="aggregate_only"
    )
    assert aggregate["metric_passed"] is True
    assert aggregate["passed"] is False

    tied = _paper_acceptance(
        ablation_summary(no_l=0.60), legal_action_provenance="exact"
    )
    assert tied["metric_passed"] is False
    assert tied["comparators"][0]["margin_fraction"] == 0.0
    assert tied["passed"] is False

    forged = dict(tied, passed=True)
    with pytest.raises(AssertionError):
        verify_paper_acceptance(forged, ablation_summary(no_l=0.60), "exact")


def test_optional_fraction_accepts_only_symmetric_missing_values():
    close_optional("", "", "zero-release fraction")
    close_optional(None, "", "zero-release fraction")
    close_optional("0.25", 0.25, "observed fraction")
    with pytest.raises(AssertionError, match="optional value"):
        close_optional("", 0.0, "one-sided missing fraction")
