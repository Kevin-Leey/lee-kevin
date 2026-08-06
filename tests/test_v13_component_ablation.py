import json
import math

import pytest

from tools.analyze_v13_component_ablation import (
    ARMS,
    bootstrap_rate,
    branch_outcome,
    normalize_fixed_horizon_branch,
    summarize,
)
from dilu.evaluation.factorial_replay import (
    COMPONENT_ABLATION_ARMS,
    ComponentAblationQueryPolicy,
    QueryAdmissionContext,
)


def _branch(
    raw_action,
    *,
    fast_action=1,
    effective_action=1,
    target_speed=20.0,
    utility=0.2,
    gate_domain="1;2;3",
):
    return {
        "raw_action": raw_action,
        "fast_action": fast_action,
        "effective_action": effective_action,
        "target_speed_after": target_speed,
        "normalized_return": utility,
        "utility": utility,
        "progress_m": 10.0,
        "min_ttc": 3.0,
        "collision": 0,
        "steps_completed": 20,
        "branch_trajectory_json": json.dumps(
            [
                {
                    "frame": offset,
                    "position_x": float(offset),
                    "position_y": 0.0,
                    "speed": 20.0,
                    "lane_id": 0,
                    "effective_action": effective_action,
                }
                for offset in range(20)
            ],
            separators=(",", ":"),
        ),
        "runtime_gate_action_universe": gate_domain,
    }


def test_fixed_horizon_normalization_uses_common_denominator_after_termination():
    row = _branch("fast", utility=0.5)
    row["steps_completed"] = 10
    row["branch_trajectory_json"] = json.dumps(json.loads(row["branch_trajectory_json"])[:10])
    normalized = normalize_fixed_horizon_branch(row, horizon=20, gamma=0.99)
    expected = 0.5 * sum(0.99**i for i in range(10)) / sum(0.99**i for i in range(20))
    assert normalized["normalized_return"] == pytest.approx(expected)
    assert normalized["utility"] == pytest.approx(expected)
    assert normalized["realized_normalized_return"] == 0.5
    assert normalized["return_denominator_steps"] == 20
    assert normalized["post_terminal_reward_convention"] == "zero_increment_absorbing_state"


def test_branch_outcome_requires_exact_gate_action_coverage():
    baseline = _branch("fast")
    rows = [baseline, _branch(1), _branch(2, effective_action=2)]
    with pytest.raises(ValueError, match="exactly cover gate domain"):
        branch_outcome(rows, 0.02)


def test_branch_outcome_rejects_divergent_raw_action_aliases():
    baseline = _branch("fast")
    rows = [
        baseline,
        _branch(1),
        _branch(2, effective_action=2, utility=0.25),
        _branch(3, effective_action=2, utility=0.30),
    ]
    with pytest.raises(ValueError, match="raw-action alias"):
        branch_outcome(rows, 0.02)


def test_bootstrap_rate_rejects_zero_exposure_instead_of_quantile_crash():
    counts = {"Full RGD": {seed: (0, 0) for seed in range(20)}}
    with pytest.raises(ValueError, match="no evaluated releases"):
        bootstrap_rate(
            counts,
            "Full RGD",
            list(range(20)),
            draws=100,
            bootstrap_seed=7,
        )


def test_summarize_reconciles_event_and_selection_accounting():
    seeds = [6000]
    events = [
        {"arm": spec.label, "seed": 6000, "corrective_set_nonempty": 0}
        for spec in ARMS
    ]
    accounting = [
        {
            "arm": spec.label,
            "seed": 6000,
            "scheduled_count": 1,
            "excluded_count": 0,
            "evaluated_count": 1 if spec.label != "w/o H" else 2,
        }
        for spec in ARMS
    ]
    with pytest.raises(ValueError, match="event/accounting denominator drift"):
        summarize(events, accounting, seeds, draws=20, bootstrap_seed=3)


def test_fixed_horizon_collision_penalty_remains_one():
    row = _branch("fast", utility=-0.5)
    row["collision"] = 1
    row["normalized_return"] = 0.5
    row["steps_completed"] = 10
    row["branch_trajectory_json"] = json.dumps(json.loads(row["branch_trajectory_json"])[:10])
    normalized = normalize_fixed_horizon_branch(row, horizon=20, gamma=0.99)
    assert math.isclose(normalized["utility"], normalized["normalized_return"] - 1.0)


def _admission_context(**overrides):
    gate = {
        "domain_contract_pass": True,
        "executor_available_pass": True,
        "latency_prediction_pass": True,
        "absolute_feasibility_pass": True,
        "latency_survival_pass": True,
        "maneuver_breadth_pass": True,
        "corrective_headroom_pass": True,
        "state_need_pass": True,
        "serial_gate_pass": True,
    }
    gate.update(overrides)
    return QueryAdmissionContext(
        frame=12,
        fast_action=1,
        query_metadata={"recoverability_gate": gate},
    )


def _component_arm(name):
    return next(arm for arm in COMPONENT_ABLATION_ARMS if arm.name == name)


def test_component_ablation_removes_only_the_named_serial_predicate():
    full = ComponentAblationQueryPolicy(_component_arm("full"))
    without_n = ComponentAblationQueryPolicy(_component_arm("without_n"))
    context = _admission_context(state_need_pass=False, serial_gate_pass=False)

    assert full.decide(context).admit is False
    decision = without_n.decide(context)
    assert decision.admit is True
    assert decision.audit["component_ablation_state_need_retained"] is False
    assert decision.audit["component_ablation_non_ablatable_pass"] is True


def test_component_ablation_keeps_base_feasibility_and_requires_complete_metadata():
    without_n = ComponentAblationQueryPolicy(_component_arm("without_n"))
    assert without_n.decide(
        _admission_context(
            state_need_pass=False,
            serial_gate_pass=False,
            absolute_feasibility_pass=False,
        )
    ).admit is False

    incomplete = _admission_context().query_metadata["recoverability_gate"]
    incomplete.pop("maneuver_breadth_pass")
    with pytest.raises(ValueError, match="boolean gate fields"):
        without_n.decide(
            QueryAdmissionContext(
                frame=12,
                fast_action=1,
                query_metadata={"recoverability_gate": incomplete},
            )
        )
