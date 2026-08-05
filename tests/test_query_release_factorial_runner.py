import json
import logging
import sys
from io import StringIO

import pytest

from dilu.evaluation.factorial_replay import FactorialArm
from tools.run_query_release_factorial import (
    DEFAULT_PROPOSAL_SOURCE_POLICY,
    DISTINCT_ACTION_METRIC_STAGE,
    _build_empirical_latency_profile,
    _empirical_assignment,
    _factorial_group_config,
    _latency_seconds_to_policy_steps,
    _primitive_selection_metric_fields,
    _query_event,
    _release_execution_is_distinct,
    _stress_assignment,
    _validate_event_lifecycle_contract,
    _validate_outcome_metrics,
    _validate_proposal_source,
    _validate_query_gate_accounting,
    _validate_request_outcome_accounting,
    _worker_output_context,
)


def _lifecycle_event(**overrides):
    event = {
        "frame": 0,
        "factorial_candidate_query": False,
        "factorial_query_issued": False,
        "factorial_query_rejection_reason": "",
        "factorial_candidate_request_id": "",
        "factorial_shared_response_outcome": "",
        "closed_loop_latency_issuance_event": False,
        "closed_loop_latency_issued_request_id": "",
        "closed_loop_latency_issued_response_outcome": "",
        "closed_loop_latency_terminal_event": False,
        "closed_loop_latency_terminal_request_id": "",
        "closed_loop_latency_terminal_response_outcome": "",
        "closed_loop_latency_terminal_outcome": "",
        "closed_loop_latency_release_event": False,
        "closed_loop_latency_timeout_event": False,
        "closed_loop_latency_failure_event": False,
    }
    event.update(overrides)
    return event


def _valid_metric_payload():
    return {
        "total_episodes": 1,
        "collision_rate": 0.0,
        "success_rate": 1.0,
        "success_number": 1,
        "avg_route_completion": 0.75,
        "avg_episode_reward": -2.0,
        "avg_driving_distance": 12.0,
        "avg_speed_all_frames": 3.0,
        "avg_runtime_per_frame": 0.01,
    }


_REQUIRED_OUTCOME_METRICS = (
    "success_rate",
    "avg_route_completion",
    "avg_episode_reward",
    "avg_driving_distance",
    "avg_speed_all_frames",
    "avg_runtime_per_frame",
)


def _issuance_event(request_id, outcome, *, frame):
    return _lifecycle_event(
        frame=frame,
        factorial_candidate_query=True,
        factorial_query_issued=True,
        factorial_candidate_request_id=request_id,
        factorial_shared_response_outcome=outcome,
        closed_loop_latency_issuance_event=True,
        closed_loop_latency_issued_request_id=request_id,
        closed_loop_latency_issued_response_outcome=outcome,
        closed_loop_latency_terminal_outcome="pending",
    )


def _release_event(request_id, *, frame, fast_action=1, selected_action=1):
    return _lifecycle_event(
        frame=frame,
        closed_loop_latency_terminal_event=True,
        closed_loop_latency_terminal_request_id=request_id,
        closed_loop_latency_terminal_response_outcome="valid",
        closed_loop_latency_terminal_outcome=(
            "distinct_actuation"
            if selected_action != fast_action
            else "fast_equivalent"
        ),
        closed_loop_latency_release_event=True,
        closed_loop_release_action_unavailable=False,
        closed_loop_release_opportunity_rejected=False,
        release_fast_comparator_action=fast_action,
        release_selected_action=selected_action,
        release_action_comparison_stage=DISTINCT_ACTION_METRIC_STAGE,
        release_selection_distinct=selected_action != fast_action,
    )


def test_stress_assignment_depends_only_on_request_identity():
    request_id = "factorial:5000:84:04"

    first = _stress_assignment(request_id)
    second = _stress_assignment(request_id)

    assert first == second == (22, "timeout")
    assert _stress_assignment("factorial:5000:0:00") == (22, "valid")


def test_empirical_assignment_is_request_deterministic_and_order_invariant():
    samples = [
        (5001, 20, 2.01),
        (5000, 10, 0.61),
        (5000, 30, 1.70),
    ]
    profile = _build_empirical_latency_profile(samples)
    reordered_profile = _build_empirical_latency_profile(list(reversed(samples)))

    request_id = "factorial:5000:84:04"
    first = _empirical_assignment(request_id, profile["_sample_steps"])
    second = _empirical_assignment(request_id, reordered_profile["_sample_steps"])

    assert profile["profile_sha256"] == reordered_profile["profile_sha256"]
    assert profile["_sample_steps"] == (7, 17, 21)
    assert first == second
    assert first in {7, 17, 21}
    assert profile["sample_count"] == 3
    assert profile["policy_frequency_hz"] == 10.0


def test_factorial_prediction_is_independent_of_request_latency_profile():
    protocol = {
        "groups": {
            "always_fast": {
                "id": "always_fast",
                "runtime_overrides": {},
            }
        }
    }
    arm = FactorialArm("full", True, True)

    group_cfg = _factorial_group_config(
        protocol,
        arm,
        predicted_latency_s=0.71,
    )
    overrides = group_cfg["runtime_overrides"]
    replay = overrides["closed_loop_latency_replay"]

    assert replay["extra_latency_s"] == pytest.approx(0.71)
    assert replay["delay_steps"] == 8
    assert overrides["factorial_predicted_latency_s"] == pytest.approx(0.71)
    assert overrides["factorial_predicted_latency_steps"] == 8
    assert _latency_seconds_to_policy_steps(2.7) == 27


def test_gate_independent_source_requires_forced_slow_snapshot(tmp_path):
    snapshot = tmp_path / "experiment_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "fixed_seed_override": 5000,
                "config": {
                    "protocol_name": "always_slow",
                    "system_routing": {"simple": "slow", "complex": "slow"},
                },
            }
        ),
        encoding="utf-8",
    )

    _validate_proposal_source(
        snapshot,
        seed=5000,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["config"]["protocol_name"] = "rgd_fixed_policy"
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be always_slow"):
        _validate_proposal_source(
            snapshot,
            seed=5000,
            source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
        )


def test_gate_independent_source_fails_on_seed_provenance_drift(tmp_path):
    snapshot = tmp_path / "experiment_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "fixed_seed_override": 5001,
                "config": {
                    "protocol_name": "always_slow",
                    "system_routing": {"simple": "slow", "complex": "slow"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="seed provenance mismatch"):
        _validate_proposal_source(
            snapshot,
            seed=5000,
            source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
        )


def test_gate_independent_source_uses_scheduled_slow_attempts_without_latency_replay():
    event = {
        "frame": 21,
        "closed_loop_latency_source_frame": 21,
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "closed_loop_latency_eligible": False,
        "closed_loop_latency_release_event": False,
    }

    assert _query_event(
        event,
        source_policy=DEFAULT_PROPOSAL_SOURCE_POLICY,
    )
    assert not _query_event(
        event,
        source_policy="legacy_gate_positive_diagnostic",
    )


def test_default_worker_context_suppresses_prebound_logs_and_streams(capsys):
    log_stream = StringIO()
    logger = logging.getLogger("test.factorial_worker_quiet")
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    previous_logging_disable = logging.root.manager.disable
    logger.handlers = [logging.StreamHandler(log_stream)]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        with _worker_output_context(verbose=False):
            print("hidden stdout")
            print("hidden stderr", file=sys.stderr)
            logger.info("hidden prebound log")
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
    assert log_stream.getvalue() == ""
    assert logging.root.manager.disable == previous_logging_disable


def test_verbose_worker_context_preserves_logs_and_streams(capsys):
    log_stream = StringIO()
    logger = logging.getLogger("test.factorial_worker_verbose")
    previous_handlers = list(logger.handlers)
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers = [logging.StreamHandler(log_stream)]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    try:
        with _worker_output_context(verbose=True):
            print("visible stdout")
            print("visible stderr", file=sys.stderr)
            logger.info("visible prebound log")
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    captured = capsys.readouterr()
    assert captured.out == "visible stdout\n"
    assert captured.err == "visible stderr\n"
    assert log_stream.getvalue() == "visible prebound log\n"


def test_lifecycle_contract_supports_same_frame_issuance_and_old_terminal():
    event = _lifecycle_event(
        frame=1,
        factorial_candidate_query=True,
        factorial_query_issued=True,
        factorial_candidate_request_id="new-valid",
        factorial_shared_response_outcome="valid",
        closed_loop_latency_issuance_event=True,
        closed_loop_latency_issued_request_id="new-valid",
        closed_loop_latency_issued_response_outcome="valid",
        closed_loop_latency_terminal_event=True,
        closed_loop_latency_terminal_request_id="old-timeout",
        closed_loop_latency_terminal_response_outcome="timeout",
        closed_loop_latency_terminal_outcome="timeout",
        closed_loop_latency_timeout_event=True,
    )

    _validate_event_lifecycle_contract(
        [_issuance_event("old-timeout", "timeout", frame=0), event],
        context="test arm",
    )


def test_lifecycle_contract_accepts_valid_release_terminal_labels():
    _validate_event_lifecycle_contract(
        [
            _issuance_event("released", "valid", frame=0),
            _release_event("released", frame=1),
        ],
        context="test arm",
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"closed_loop_latency_issuance_event": False},
            "issuance event disagrees",
        ),
        (
            {"closed_loop_latency_issued_response_outcome": "failure"},
            "issuance outcome disagrees",
        ),
        (
            {"closed_loop_latency_terminal_event": False},
            "terminal marker disagrees",
        ),
        (
            {"closed_loop_latency_terminal_response_outcome": "failure"},
            "terminal response outcome disagrees",
        ),
        (
            {"closed_loop_latency_terminal_outcome": "pending"},
            "asynchronous terminal outcome disagrees",
        ),
        (
            {"closed_loop_latency_release_event": True},
            "not mutually exclusive",
        ),
    ],
)
def test_lifecycle_contract_rejects_inconsistent_markers_and_outcomes(
    mutation,
    message,
):
    event = _lifecycle_event(
        frame=1,
        factorial_candidate_query=True,
        factorial_query_issued=True,
        factorial_candidate_request_id="new-valid",
        factorial_shared_response_outcome="valid",
        closed_loop_latency_issuance_event=True,
        closed_loop_latency_issued_request_id="new-valid",
        closed_loop_latency_issued_response_outcome="valid",
        closed_loop_latency_terminal_event=True,
        closed_loop_latency_terminal_request_id="old-timeout",
        closed_loop_latency_terminal_response_outcome="timeout",
        closed_loop_latency_terminal_outcome="timeout",
        closed_loop_latency_timeout_event=True,
    )
    event.update(mutation)

    with pytest.raises(RuntimeError, match=message):
        _validate_event_lifecycle_contract(
            [_issuance_event("old-timeout", "timeout", frame=0), event],
            context="test arm",
        )


def test_lifecycle_contract_rejects_terminal_fields_without_terminal_event():
    event = _lifecycle_event(
        closed_loop_latency_terminal_request_id="orphan",
        closed_loop_latency_terminal_outcome="rejected",
    )

    with pytest.raises(RuntimeError, match="non-terminal event carries"):
        _validate_event_lifecycle_contract([event], context="test arm")


def test_lifecycle_contract_rejects_terminal_before_issuance():
    terminal = _lifecycle_event(
        frame=0,
        closed_loop_latency_terminal_event=True,
        closed_loop_latency_terminal_request_id="late-issuance",
        closed_loop_latency_terminal_response_outcome="timeout",
        closed_loop_latency_terminal_outcome="timeout",
        closed_loop_latency_timeout_event=True,
    )

    with pytest.raises(RuntimeError, match="terminal precedes issuance"):
        _validate_event_lifecycle_contract(
            [terminal, _issuance_event("late-issuance", "timeout", frame=1)],
            context="test arm",
        )


def test_lifecycle_contract_rejects_multiple_terminal_flags_without_marker():
    event = _lifecycle_event(
        closed_loop_latency_timeout_event=True,
        closed_loop_latency_failure_event=True,
    )

    with pytest.raises(RuntimeError, match="not mutually exclusive"):
        _validate_event_lifecycle_contract([event], context="test arm")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {"release_selected_action": None},
            "release_selected_action must be an integer action",
        ),
        (
            {"release_selection_distinct": False},
            "release_selection_distinct disagrees",
        ),
        (
            {"closed_loop_latency_terminal_outcome": "fast_equivalent"},
            "terminal outcome disagrees",
        ),
        (
            {
                "closed_loop_release_opportunity_rejected": True,
                "release_selected_action": 3,
            },
            "must select the Fast action",
        ),
    ],
)
def test_release_selection_contract_fails_closed(mutation, message):
    event = _release_event("released", frame=1, selected_action=3)
    event.update(mutation)

    with pytest.raises(RuntimeError, match=message):
        _release_execution_is_distinct(event)


def test_request_outcomes_are_reconciled_by_request_id():
    _validate_request_outcome_accounting(
        {"released": "valid", "pending": "timeout"},
        {"released": "valid"},
        {"pending": "timeout"},
        context="test arm",
    )

    with pytest.raises(RuntimeError, match="issuance/terminal outcome mismatch"):
        _validate_request_outcome_accounting(
            {"request": "valid"},
            {"request": "timeout"},
            {},
            context="test arm",
        )
    with pytest.raises(RuntimeError, match="issuance/pending outcome mismatch"):
        _validate_request_outcome_accounting(
            {"request": "failure"},
            {},
            {"request": "unknown"},
            context="test arm",
        )


def test_query_disabled_arm_cannot_reject_candidates():
    arm = FactorialArm("neither", False, False)
    candidates = [
        _lifecycle_event(
            factorial_candidate_query=True,
            factorial_query_issued=True,
        )
    ]

    _validate_query_gate_accounting(
        arm,
        candidate_events=candidates,
        candidate_count=1,
        issued_count=1,
        gate_rejected_count=0,
        context="test arm",
    )

    with pytest.raises(RuntimeError, match="issuance accounting mismatch"):
        _validate_query_gate_accounting(
            arm,
            candidate_events=candidates,
            candidate_count=1,
            issued_count=0,
            gate_rejected_count=0,
            context="test arm",
        )
    with pytest.raises(RuntimeError, match="query-disabled arm"):
        _validate_query_gate_accounting(
            arm,
            candidate_events=[
                _lifecycle_event(
                    factorial_candidate_query=True,
                    factorial_query_rejection_reason="query_gate_failed",
                )
            ],
            candidate_count=1,
            issued_count=0,
            gate_rejected_count=1,
            context="test arm",
        )


def test_query_enabled_arm_requires_exact_issued_rejected_partition():
    arm = FactorialArm("full", True, True)
    candidates = [
        _lifecycle_event(
            factorial_candidate_query=True,
            factorial_query_issued=True,
        ),
        _lifecycle_event(
            factorial_candidate_query=True,
            factorial_query_issued=False,
            factorial_query_rejection_reason="query_gate_failed",
        ),
    ]

    _validate_query_gate_accounting(
        arm,
        candidate_events=candidates,
        candidate_count=2,
        issued_count=1,
        gate_rejected_count=1,
        context="test arm",
    )
    with pytest.raises(RuntimeError, match="rejection accounting mismatch"):
        _validate_query_gate_accounting(
            arm,
            candidate_events=candidates,
            candidate_count=2,
            issued_count=1,
            gate_rejected_count=0,
            context="test arm",
        )
def test_outcome_metrics_are_strictly_validated():
    assert _validate_outcome_metrics(
        {"collision": False},
        _valid_metric_payload(),
        context="test arm",
    ) == {
        "collision": 0,
        "success_rate": 1.0,
        "route_completion": 0.75,
        "episode_reward": -2.0,
        "driving_distance": 12.0,
        "avg_speed": 3.0,
        "runtime_per_frame": 0.01,
    }


@pytest.mark.parametrize("field", _REQUIRED_OUTCOME_METRICS)
def test_outcome_metrics_reject_missing_values(field):
    metrics = _valid_metric_payload()
    metrics.pop(field)

    with pytest.raises(RuntimeError, match="required metric"):
        _validate_outcome_metrics(
            {"collision": False},
            metrics,
            context="test arm",
        )


@pytest.mark.parametrize("field", _REQUIRED_OUTCOME_METRICS)
@pytest.mark.parametrize("bad_value", [None, True, float("nan"), float("inf")])
def test_outcome_metrics_reject_non_numeric_or_nonfinite_values(field, bad_value):
    metrics = _valid_metric_payload()
    metrics[field] = bad_value

    with pytest.raises(RuntimeError, match="must be (numeric|finite)"):
        _validate_outcome_metrics(
            {"collision": False},
            metrics,
            context="test arm",
        )


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("success_rate", -0.01),
        ("success_rate", 1.01),
        ("avg_route_completion", -0.01),
        ("avg_route_completion", 1.01),
        ("avg_driving_distance", -0.01),
        ("avg_speed_all_frames", -0.01),
        ("avg_runtime_per_frame", -0.01),
    ],
)
def test_outcome_metrics_reject_values_outside_semantic_ranges(field, bad_value):
    metrics = _valid_metric_payload()
    metrics[field] = bad_value

    with pytest.raises(RuntimeError, match="must be at (least|most)"):
        _validate_outcome_metrics(
            {"collision": False},
            metrics,
            context="test arm",
        )


@pytest.mark.parametrize(
    "summary",
    [{}, {"collision": None}, {"collision": 0}, {"collision": float("nan")}],
)
def test_outcome_metrics_require_explicit_collision_boolean(summary):
    with pytest.raises(RuntimeError, match="collision outcome"):
        _validate_outcome_metrics(
            summary,
            _valid_metric_payload(),
            context="test arm",
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("total_episodes", 2, "exactly one episode"),
        ("total_episodes", True, "exactly one episode"),
        ("success_number", 2, "success_number must be binary"),
        ("success_number", False, "success_number must be binary"),
        ("success_rate", 0.5, "success rate disagrees"),
        ("collision_rate", 1.0, "collision rate disagrees"),
    ],
)
def test_outcome_metrics_reconcile_single_episode_report(field, bad_value, message):
    metrics = _valid_metric_payload()
    metrics[field] = bad_value

    with pytest.raises(RuntimeError, match=message):
        _validate_outcome_metrics(
            {"collision": False},
            metrics,
            context="test arm",
        )


def test_aligned_distinct_actuations_is_only_a_primitive_selection_alias():
    fields = _primitive_selection_metric_fields(3, aligned_count=2)

    assert fields["distinct_actuations"] == 3
    assert fields["primitive_distinct_selections"] == 3
    assert fields["aligned_distinct_actuations"] == 2
    assert fields["distinct_action_metric_stage"] == DISTINCT_ACTION_METRIC_STAGE
    assert (
        fields["aligned_distinct_actuations_stage"]
        == DISTINCT_ACTION_METRIC_STAGE
    )
    assert fields["effect_distinctness_available"] is False


@pytest.mark.parametrize("primitive,aligned", [(1, 2), (-1, 0), (1, -1)])
def test_primitive_selection_alias_rejects_invalid_aligned_subset(
    primitive,
    aligned,
):
    with pytest.raises(ValueError, match="aligned <= primitive"):
        _primitive_selection_metric_fields(
            primitive,
            aligned_count=aligned,
        )
