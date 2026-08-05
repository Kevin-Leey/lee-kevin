from tools.analyze_v13_main_results import canonical_distinct_actuation


def _event(**updates):
    event = {
        "closed_loop_latency_release_event": True,
        "closed_loop_release_action_alignment_evaluated": True,
        "closed_loop_release_action_alignment_pass": True,
        "closed_loop_release_opportunity_rejected": False,
        "closed_loop_release_action_unavailable": False,
        "closed_loop_execution_state_fast_action": 1,
        "closed_loop_latency_executed_action": 4,
    }
    event.update(updates)
    return event


def test_canonical_distinct_actuation_uses_final_executed_action():
    assert canonical_distinct_actuation(_event()) is True
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_latency_executed_action=1)
        )
        is False
    )


def test_canonical_distinct_actuation_requires_evaluated_passing_alignment():
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_action_alignment_evaluated=False)
        )
        is False
    )
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_action_alignment_pass=False)
        )
        is False
    )


def test_canonical_distinct_actuation_rejects_unavailable_or_rejected_release():
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_action_unavailable=True)
        )
        is False
    )
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_opportunity_rejected=True)
        )
        is False
    )
