import json

import pytest

from dilu.evaluation.export_builders import build_public_comprehensive_metrics
from dilu.evaluation.metrics_aggregator import MetricsAggregator


def _reasoning_record(*, ttc: float, queried: bool, latency: float = 0.0):
    system_used = "slow" if queried else "fast"
    return {
        "frame_id": 0,
        "system_used": system_used,
        "slow_reasoning_mode": "online_llm" if queried else "fast_path",
        "slow_reasoning_success": queried,
        "slow_request_attempted": queried,
        "slow_request_valid_return": queried,
        "slow_request_failed": False,
        "inference_latency": latency,
        "ttc": ttc,
        "rgd_subordinate_diagnostics": {
            "recoverability_signal": {
                "recoverability_gate": {
                    "gate_active": False,
                    "near_threshold": False,
                }
            },
            "recoverability_public_signal": {},
            "slow_path_objective": {},
            "safety_envelope": {
                "risk_event": False,
                "safety_override": False,
                "shield_override": False,
                "emergency_level": 0,
                "route_action_changed": False,
            },
        },
    }


def _event(*, ttc: float, queried: bool, released: bool = False, distinct: bool = False):
    return {
        "system_used": "fast",
        "slow_request_attempted": queried,
        "slow_request_valid_return": queried,
        "slow_request_failed": False,
        "ttc": ttc,
        "risk_event": False,
        "safety_override": False,
        "shield_override": False,
        "emergency_level": 0,
        "closed_loop_latency_release_event": released,
        "closed_loop_release_actuation_distinct": distinct,
    }


def test_query_metrics_use_attempts_and_preserve_negative_allocation_difference(tmp_path):
    aggregator = MetricsAggregator("test", str(tmp_path))
    events = [
        _event(ttc=1.0, queried=True),
        _event(ttc=1.0, queried=False),
        _event(ttc=10.0, queried=True),
        _event(ttc=10.0, queried=True),
    ]
    events[0].update(
        {
            "system_used": "fast_after_slow_failure",
            "slow_request_valid_return": False,
            "slow_request_failed": True,
        }
    )
    aggregator.all_event_records = events

    story = aggregator._selective_allocation_story([])

    assert story["risk_frame_count"] == 2
    assert story["queried_frame_count"] == 3
    assert story["queried_risk_frame_count"] == 1
    assert story["risk_conditional_query_recall"] == pytest.approx(0.5)
    assert story["queried_frame_risk_precision"] == pytest.approx(1.0 / 3.0)
    assert story["high_vs_low_query_rate_difference"] == pytest.approx(-0.5)
    assert story["slow_compute_seconds_available"] is False
    assert story["slow_compute_seconds"] is None


def test_release_rvod_and_compute_yields_keep_their_denominators_explicit(tmp_path):
    event_dir = tmp_path / "event_logs"
    event_dir.mkdir()
    events = [
        _event(ttc=1.0, queried=True, released=True, distinct=True),
        _event(ttc=1.0, queried=True, released=True, distinct=False),
        _event(ttc=10.0, queried=True),
        _event(ttc=10.0, queried=False),
    ]
    (event_dir / "event_log_test.json").write_text(
        json.dumps({"events": events}), encoding="utf-8"
    )
    aggregator = MetricsAggregator("test", str(tmp_path))
    aggregator.all_reasoning_records = [
        _reasoning_record(ttc=1.0, queried=True, latency=0.2),
        _reasoning_record(ttc=1.0, queried=True, latency=0.2),
        _reasoning_record(ttc=10.0, queried=True, latency=0.2),
        _reasoning_record(ttc=10.0, queried=False),
    ]
    aggregator.evaluation_metrics_list = [
        {"corrective_set_nonempty": "1"},
        {"corrective_set_nonempty": 0},
        {"corrective_set_nonempty": True},
    ]

    story = aggregator._selective_allocation_story(aggregator.all_reasoning_records)

    assert story["released_response_count"] == 2
    assert story["distinct_corrective_actuation_count"] == 1
    assert story["actuation_yield"] == pytest.approx(0.5)
    assert story["slow_compute_call_count"] == 3
    assert story["compute_per_corrective_release"] == pytest.approx(3.0)
    assert story["slow_compute_seconds_available"] is True
    assert story["slow_compute_seconds"] == pytest.approx(0.6)
    assert story["compute_seconds_per_corrective_release"] == pytest.approx(0.6)
    assert story["rvod_evaluated_release_count"] == 3
    assert story["rvod_positive_release_count"] == 2
    assert story["rvod_positive_yield"] == pytest.approx(2.0 / 3.0)


def test_undefined_yields_are_null_instead_of_fabricated_zeroes(tmp_path):
    aggregator = MetricsAggregator("test", str(tmp_path))

    story = aggregator._selective_allocation_story([])

    assert story["selective_allocation_metrics_available"] is False
    assert story["risk_conditional_query_recall"] is None
    assert story["queried_frame_risk_precision"] is None
    assert story["high_vs_low_query_rate_difference"] is None
    assert story["actuation_yield"] is None
    assert story["compute_per_corrective_release"] is None
    assert story["slow_compute_seconds_available"] is False
    assert story["slow_compute_seconds"] is None
    assert story["rvod_positive_yield"] is None


def test_actuation_yield_recomputes_stale_derived_flags_from_final_actions(tmp_path):
    returned_to_fast = _event(
        ttc=1.0, queried=False, released=True, distinct=True
    )
    changed_from_fast = _event(
        ttc=1.0, queried=False, released=True, distinct=False
    )
    for record, final_action in ((returned_to_fast, 1), (changed_from_fast, 4)):
        record.update(
            {
                "closed_loop_execution_state_fast_action": 1,
                "closed_loop_released_slow_action": 4,
                "closed_loop_release_action_alignment_evaluated": True,
                "closed_loop_release_action_alignment_pass": True,
                "final_action": final_action,
            }
        )
    aggregator = MetricsAggregator("test", str(tmp_path))
    aggregator.all_event_records = [returned_to_fast, changed_from_fast]

    story = aggregator._selective_allocation_story([])

    assert story["released_response_count"] == 2
    assert story["distinct_corrective_actuation_count"] == 1
    assert story["actuation_yield"] == pytest.approx(0.5)


def test_v4_action_yield_uses_same_stage_selection_not_final_actuator(tmp_path):
    bridge_only_change = _event(
        ttc=1.0, queried=False, released=True, distinct=True
    )
    selected_difference_returning_to_fast_id = _event(
        ttc=1.0, queried=False, released=True, distinct=False
    )
    for record, selected_action, final_action in (
        (bridge_only_change, 1, 4),
        (selected_difference_returning_to_fast_id, 4, 1),
    ):
        record.update(
            {
                "release_fast_comparator_action": 1,
                "release_selected_action": selected_action,
                "release_action_comparison_stage": (
                    "post_release_guard_and_frame_safety_pre_actuator_bridge"
                ),
                "final_actuator_action": final_action,
                "closed_loop_released_slow_action": 4,
                "closed_loop_release_action_alignment_evaluated": True,
                "closed_loop_release_action_alignment_pass": True,
            }
        )
    aggregator = MetricsAggregator("test", str(tmp_path))
    aggregator.all_event_records = [
        bridge_only_change,
        selected_difference_returning_to_fast_id,
    ]

    story = aggregator._selective_allocation_story([])

    assert story["distinct_corrective_actuation_count"] == 1
    assert story["actuation_yield"] == pytest.approx(0.5)
    assert story["effect_distinctness_available"] is False


def test_request_scoped_aggregation_keeps_overlapping_issue_and_timeout(tmp_path):
    old_issue = _event(ttc=2.0, queried=True)
    old_issue.update(
        {
            "closed_loop_latency_issuance_event": True,
            "closed_loop_latency_issued_request_id": "old-timeout",
            "closed_loop_latency_terminal_event": False,
            "closed_loop_latency_terminal_request_id": "",
        }
    )
    overlap = _event(ttc=2.0, queried=True)
    overlap.update(
        {
            "closed_loop_latency_issuance_event": True,
            "closed_loop_latency_issued_request_id": "new-valid",
            "closed_loop_latency_terminal_event": True,
            "closed_loop_latency_terminal_request_id": "old-timeout",
            "closed_loop_latency_terminal_response_outcome": "timeout",
        }
    )
    new_terminal = _event(ttc=2.0, queried=False, released=True)
    new_terminal.update(
        {
            "closed_loop_latency_issuance_event": False,
            "closed_loop_latency_issued_request_id": "",
            "closed_loop_latency_terminal_event": True,
            "closed_loop_latency_terminal_request_id": "new-valid",
            "closed_loop_latency_terminal_response_outcome": "valid",
        }
    )
    aggregator = MetricsAggregator("test", str(tmp_path))
    aggregator.all_event_records = [old_issue, overlap, new_terminal]
    aggregator.all_reasoning_records = [
        _reasoning_record(ttc=2.0, queried=True),
        _reasoning_record(ttc=2.0, queried=True),
    ]

    story = aggregator._reasoning_story()

    assert story["slow_request_lifecycle_request_scoped"] is True
    assert story["slow_attempts"] == 2
    assert story["slow_attempt_successes"] == 1
    assert story["slow_attempt_failures"] == 1
    assert story["slow_attempt_terminal_outcomes"] == 2
    assert story["slow_attempt_pending"] == 0


def test_legacy_hru_increases_with_risk_instead_of_rewarding_lower_risk(tmp_path):
    lower_risk = MetricsAggregator("lower", str(tmp_path / "lower"))
    lower_risk.all_reasoning_records = [
        _reasoning_record(ttc=1.6, queried=True),
        _reasoning_record(ttc=10.0, queried=False),
    ]
    higher_risk = MetricsAggregator("higher", str(tmp_path / "higher"))
    higher_risk.all_reasoning_records = [
        _reasoning_record(ttc=0.4, queried=True),
        _reasoning_record(ttc=10.0, queried=False),
    ]

    lower_value = lower_risk._reasoning_story()[
        "budget_normalized_independent_high_risk_utility"
    ]
    higher_value = higher_risk._reasoning_story()[
        "budget_normalized_independent_high_risk_utility"
    ]

    assert lower_value == pytest.approx(1.2)
    assert higher_value == pytest.approx(1.8)
    assert higher_value > lower_value


def test_new_metrics_are_present_on_the_public_export_surface(tmp_path):
    aggregator = MetricsAggregator("test", str(tmp_path))
    aggregator.physical_metrics_list = [
        {
            "total_frames": 2,
            "collision": False,
            "success_completion": True,
            "avg_speed": 5.0,
        }
    ]
    aggregator.all_reasoning_records = [
        _reasoning_record(ttc=1.0, queried=True),
        _reasoning_record(ttc=10.0, queried=False),
    ]

    payload = build_public_comprehensive_metrics(
        aggregator.calculate_comprehensive_metrics()
    )

    assert payload["risk_conditional_query_recall"] == pytest.approx(1.0)
    assert payload["queried_frame_risk_precision"] == pytest.approx(1.0)
    assert payload["high_vs_low_query_rate_difference"] == pytest.approx(1.0)
    assert payload["rvod_positive_yield"] is None
    assert payload["actuation_yield"] is None
    assert payload["compute_per_corrective_release"] is None


def test_invalid_rvod_label_fails_closed(tmp_path):
    aggregator = MetricsAggregator("test", str(tmp_path))
    aggregator.evaluation_metrics_list = [{"corrective_set_nonempty": "unknown"}]

    with pytest.raises(ValueError, match="must be a binary label"):
        aggregator._selective_allocation_story([])
