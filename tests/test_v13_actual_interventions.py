import json

from tools.analyze_v13_actual_interventions import (
    _prefix_execution_requested_action,
    summarize_release_branches,
)


def _branch(raw_action, effective_action, utility, *, target_speed=20.0, collision=0):
    return {
        "raw_action": raw_action,
        "effective_action": effective_action,
        "target_speed_after": target_speed,
        "utility": utility,
        "collision": collision,
        "progress_m": 10.0 + utility,
        "min_ttc": 2.0 + utility,
        "branch_trajectory_json": json.dumps([{"action": effective_action}]),
        "horizon_steps": 20,
        "gamma": 0.99,
    }


def _event(slow_action=4, final_action=4):
    return {
        "seed": 5000,
        "frame": 10,
        "closed_loop_released_slow_action": slow_action,
        "closed_loop_execution_state_fast_action": 1,
        "closed_loop_latency_executed_action": final_action,
    }


def test_actual_slow_value_is_not_replaced_by_oracle_opportunity():
    baseline = _branch("fast", 1, 0.5)
    actual = _branch(4, 4, 0.3)
    oracle = _branch(2, 2, 0.8)

    result = summarize_release_branches(
        _event(), baseline, [actual, oracle], epsilon=0.02
    )

    assert result["oracle_corrective_opportunity"] is True
    assert result["actual_slow_corrective"] is False
    assert result["actual_slow_advantage"] == -0.2


def test_safety_equivalent_actual_action_has_zero_intervention_value():
    baseline = _branch("fast", 1, 0.5)
    mapped_equivalent = _branch(4, 1, 0.1)

    result = summarize_release_branches(
        _event(final_action=1), baseline, [mapped_equivalent], epsilon=0.02
    )

    assert result["actual_slow_effect_distinct"] is False
    assert result["actual_slow_advantage"] == 0.0
    assert result["actual_slow_corrective"] is False


def test_prefix_action_reconstructs_pre_bridge_execution_stage():
    base = {
        "closed_loop_latency_original_final_action": 2,
        "final_action": 1,
    }
    assert _prefix_execution_requested_action(base) == 2

    issuance = {
        **base,
        "closed_loop_latency_issuance_event": True,
        "slow_request_valid_return": True,
        "closed_loop_latency_hold_action": 3,
    }
    assert _prefix_execution_requested_action(issuance) == 3

    accepted_release = {
        **base,
        "closed_loop_latency_release_event": True,
        "closed_loop_release_action_unavailable": False,
        "closed_loop_release_opportunity_rejected": False,
        "closed_loop_release_action_alignment_evaluated": True,
        "closed_loop_release_action_alignment_pass": True,
        "closed_loop_release_action_alignment_slow_effective_action": 4,
        "closed_loop_execution_state_fast_action": 1,
    }
    assert _prefix_execution_requested_action(accepted_release) == 4

    rejected_release = {
        **accepted_release,
        "closed_loop_release_opportunity_rejected": True,
    }
    assert _prefix_execution_requested_action(rejected_release) == 1
