import math
from types import SimpleNamespace

import pytest

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.runtime_frame_trace import build_episode_event, classify_release_lifecycle
from dilu.runtime_support import (
    _apply_closed_loop_latency_replay,
    _request_latency_contract,
    _resolve_latency_replay_delay,
)


def _cfg(*, seconds=0.2, revalidate=False):
    return {
        "policy_frequency": 10,
        "closed_loop_latency_replay": {
            "enable": True,
            "extra_latency_s": seconds,
            "delay_steps": 2,
            "target_systems": ["slow"],
            "release_opportunity_revalidation": {"enable": revalidate},
        },
    }


def _query_meta(request_id="request-0", *, slow_action=4, fast_action=1, steps=2):
    return {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "slow_request_failed": False,
        "closed_loop_latency_request_id": request_id,
        "closed_loop_latency_response_outcome": "valid",
        "closed_loop_scripted_latency_steps": steps,
        "query_state_fast_proposal_action": fast_action,
        "query_state_slow_pre_guard_action": slow_action,
        "query_state_slow_released_action": slow_action,
    }


def _state(actions=(0, 1, 2, 3, 4)):
    state = DrivingState(
        ego_speed=20.0,
        ego_lane=1,
        total_lanes=3,
        front_distance=20.0,
        ttc=3.0,
        thw=1.0,
        legal_actions=list(actions),
    )
    state.set_effective_action_universe(actions, source="latency-replay-test")
    return state


def test_seconds_are_authoritative_at_the_bound_policy_frequency():
    cfg = _cfg(seconds=0.26)
    env = SimpleNamespace(unwrapped=SimpleNamespace(config={"policy_frequency": 4}))

    contract = _resolve_latency_replay_delay(cfg, env)

    assert contract["predicted_seconds"] == pytest.approx(0.26)
    assert contract["predicted_steps"] == 2
    assert contract["scheduled_steps"] == 2
    assert contract["policy_frequency_hz"] == pytest.approx(4.0)
    assert contract["configured_steps_consistent"] is True


@pytest.mark.parametrize("invalid", [-1, 2.5, float("nan"), float("inf"), "2", True])
def test_scripted_request_delay_requires_a_nonnegative_integer(invalid):
    contract = _resolve_latency_replay_delay(_cfg())
    with pytest.raises(ValueError, match="nonnegative integer"):
        _request_latency_contract(
            contract, {"closed_loop_scripted_latency_steps": invalid}
        )


def test_request_level_delay_changes_scheduling_without_changing_prediction():
    contract = _resolve_latency_replay_delay(_cfg())

    request = _request_latency_contract(
        contract, {"closed_loop_scripted_latency_steps": 5}
    )

    assert request["predicted_steps"] == 2
    assert request["scheduled_steps"] == 5
    assert request["scheduled_seconds"] == pytest.approx(0.5)
    assert request["source"] == "decision_meta.closed_loop_scripted_latency_steps"


def test_query_pending_and_release_use_request_scoped_timing():
    cfg = _cfg()
    episode = {"action": 4, "latency_replay_queue": []}
    query = _query_meta()

    query_action = _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=query,
        episode_state=episode,
        cfg=cfg,
    )
    pending = {"system_used": "fast"}
    pending_action = _apply_closed_loop_latency_replay(
        frame=1,
        action=3,
        decision_meta=pending,
        episode_state=episode,
        cfg=cfg,
    )
    terminal = {"system_used": "fast"}
    terminal_action = _apply_closed_loop_latency_replay(
        frame=2,
        action=1,
        decision_meta=terminal,
        episode_state=episode,
        cfg=cfg,
        driving_state=_state(),
    )

    assert query_action == int(ActionType.IDLE)
    assert pending_action == int(ActionType.FASTER)
    assert terminal_action == int(ActionType.SLOWER)
    assert query["closed_loop_latency_issuance_event"] is True
    assert query["closed_loop_latency_realized_available"] is False
    assert math.isnan(query["closed_loop_latency_realized_seconds"])
    assert pending["closed_loop_latency_pending_age_steps"] == 1
    assert terminal["closed_loop_latency_terminal_request_id"] == "request-0"
    assert terminal["closed_loop_latency_realized_steps"] == 2
    assert terminal["closed_loop_latency_realized_seconds"] == pytest.approx(0.2)
    assert terminal["closed_loop_latency_release_event"] is True
    assert episode["latency_replay_queue"] == []


def test_only_one_scripted_request_can_be_pending():
    cfg = _cfg()
    episode = {"action": 1, "latency_replay_queue": []}
    first = _query_meta("first", fast_action=1, steps=5)
    second = _query_meta("second", fast_action=3, steps=1)

    _apply_closed_loop_latency_replay(
        frame=0, action=4, decision_meta=first, episode_state=episode, cfg=cfg
    )
    executed = _apply_closed_loop_latency_replay(
        frame=1, action=4, decision_meta=second, episode_state=episode, cfg=cfg
    )

    assert executed == int(ActionType.FASTER)
    assert second["closed_loop_latency_issuance_event"] is False
    assert [row["request_id"] for row in episode["latency_replay_queue"]] == [
        "first"
    ]


def test_release_frame_does_not_issue_a_second_request():
    cfg = _cfg()
    episode = {"action": 1, "latency_replay_queue": []}
    _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=_query_meta("first", steps=1),
        episode_state=episode,
        cfg=cfg,
    )
    release_and_query = _query_meta("second", slow_action=3, fast_action=1, steps=1)

    executed = _apply_closed_loop_latency_replay(
        frame=1,
        action=3,
        decision_meta=release_and_query,
        episode_state=episode,
        cfg=cfg,
        driving_state=_state(),
    )

    assert executed == int(ActionType.SLOWER)
    assert release_and_query["closed_loop_latency_terminal_request_id"] == "first"
    assert release_and_query["closed_loop_latency_issuance_event"] is False
    assert episode["latency_replay_queue"] == []


def test_unavailable_delayed_action_falls_back_to_current_fast():
    cfg = _cfg()
    episode = {"action": 1, "latency_replay_queue": []}
    _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=_query_meta(steps=1),
        episode_state=episode,
        cfg=cfg,
    )
    terminal = {"system_used": "fast"}

    executed = _apply_closed_loop_latency_replay(
        frame=1,
        action=3,
        decision_meta=terminal,
        episode_state=episode,
        cfg=cfg,
        driving_state=_state((1, 2, 3)),
    )

    assert executed == int(ActionType.FASTER)
    assert terminal["closed_loop_release_action_unavailable"] is True
    assert terminal["closed_loop_release_opportunity_rejected"] is True


def test_invalid_latency_prediction_cannot_grant_slow_authority():
    cfg = _cfg(seconds=float("nan"))
    episode = {"action": 1, "latency_replay_queue": []}
    meta = _query_meta()
    meta.pop("closed_loop_latency_response_outcome")
    meta.pop("closed_loop_scripted_latency_steps")

    executed = _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=meta,
        episode_state=episode,
        cfg=cfg,
    )

    assert executed == int(ActionType.IDLE)
    assert meta["closed_loop_latency_invalid_prediction_fallback"] is True
    assert episode["latency_replay_queue"] == []


def test_release_lifecycle_requires_alignment_for_distinct_actuation():
    common = {
        "release_event": True,
        "release_fast_action": 1,
        "released_slow_action": 4,
        "executed_action": 4,
    }

    not_evaluated = classify_release_lifecycle(
        **common,
        release_alignment_evaluated=False,
        release_alignment_passed=True,
    )
    aligned = classify_release_lifecycle(
        **common,
        release_alignment_evaluated=True,
        release_alignment_passed=True,
    )

    assert not_evaluated["release_selection_distinct"] is True
    assert not_evaluated["closed_loop_release_actuation_distinct"] is False
    assert aligned["closed_loop_release_actuation_distinct"] is True


def test_modern_release_event_validates_action_stages_and_recomputes_labels():
    metadata = {
        "factorial_replay_version": "factorial_v4",
        "closed_loop_latency_release_event": True,
        "closed_loop_latency_terminal_response_outcome": "valid",
        "closed_loop_released_slow_action": 4,
        "closed_loop_release_action_alignment_evaluated": True,
        "closed_loop_release_action_alignment_pass": True,
        "release_fast_comparator_action": 1,
        "release_selected_action": 4,
        "release_action_comparison_stage": (
            "post_release_guard_pre_final_safety_projection"
        ),
        "final_actuator_action": 1,
        "final_actuator_action_stage": (
            "post_shared_actuator_bridge_pre_environment_step"
        ),
    }

    event = build_episode_event(4, {}, metadata, {})

    assert event["release_selection_distinct"] is True
    assert event["closed_loop_release_actuation_distinct"] is False
    assert event["closed_loop_latency_terminal_outcome"] == "distinct_actuation"
    assert event["closed_loop_release_actuation_comparison_available"] is False

    invalid = dict(metadata)
    invalid["release_action_comparison_stage"] = "query_state"
    with pytest.raises(RuntimeError, match="invalid release action comparison stage"):
        build_episode_event(4, {}, invalid, {})

    native_missing_stage = dict(metadata)
    native_missing_stage.pop("factorial_replay_version")
    native_missing_stage.pop("release_action_comparison_stage")
    native_missing_stage["native_async_slow_path"] = True
    with pytest.raises(RuntimeError, match="explicit action-stage contract"):
        build_episode_event(4, {}, native_missing_stage, {})


def test_scripted_terminal_carries_latched_response_hash():
    cfg = _cfg()
    episode = {}
    issued = _query_meta("hashed-response")
    issued["factorial_shared_response_sha256"] = "a" * 64
    _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=issued,
        episode_state=episode,
        cfg=cfg,
    )

    terminal = {"system_used": "fast"}
    _apply_closed_loop_latency_replay(
        frame=2,
        action=1,
        decision_meta=terminal,
        episode_state=episode,
        cfg=cfg,
    )

    assert terminal["closed_loop_latency_terminal_request_id"] == "hashed-response"
    assert terminal["closed_loop_latency_response_sha256"] == "a" * 64


def test_invalid_response_hash_does_not_reserve_request_id():
    cfg = _cfg()
    episode = {}
    issued = _query_meta("invalid-hash")
    issued["factorial_shared_response_sha256"] = "not-a-digest"

    with pytest.raises(RuntimeError, match="invalid SHA256"):
        _apply_closed_loop_latency_replay(
            frame=0,
            action=4,
            decision_meta=issued,
            episode_state=episode,
            cfg=cfg,
        )

    assert episode["latency_replay_queue"] == []
    assert episode["_latency_request_ids"] == set()


def test_incomplete_release_audit_does_not_consume_request():
    cfg = _cfg(revalidate=True)
    episode = {}
    issued = _query_meta("incomplete-audit")
    _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=issued,
        episode_state=episode,
        cfg=cfg,
    )
    queue_before = [dict(item) for item in episode["latency_replay_queue"]]
    agent = SimpleNamespace(
        orchestrator=SimpleNamespace(
            evaluate_release_proposal=lambda **kwargs: {"release_pass": True}
        )
    )

    with pytest.raises(RuntimeError, match="incomplete audit"):
        _apply_closed_loop_latency_replay(
            frame=2,
            action=1,
            decision_meta={"system_used": "fast"},
            episode_state=episode,
            cfg=cfg,
            agent=agent,
            driving_state=_state(),
        )

    assert episode["latency_replay_queue"] == queue_before
    assert episode["_latency_terminal_request_ids"] == set()
