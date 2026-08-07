import concurrent.futures
import copy
import threading
import time
from collections import deque
from types import MethodType, SimpleNamespace

import pytest

import dilu.runtime_support as runtime_support
from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.driverAgentV2 import DriverAgentV2
from dilu.driver_agent.reasoning.decision import RGDDecision
from dilu.driver_agent.reasoning.rgd_core import RGDOrchestrator
from dilu.driver_agent.reasoning.slow_thinker import (
    SlowPathUnavailableError,
    SlowRequest,
    SlowThinker,
)
from dilu.evaluation.factorial_replay import FACTORIAL_ARMS, ProposalReplayAgent
from dilu.evaluation.release_snapshot import (
    capture_release_snapshot,
    validate_release_snapshot_policy_state,
)
from dilu.runtime_frame_trace import build_episode_event, create_episode_runtime_state


def _state(frame: int, fast_action: int = int(ActionType.IDLE)) -> DrivingState:
    state = DrivingState(
        ego_speed=18.0 + frame,
        ego_lane=0,
        total_lanes=2,
        scenario_type="highway",
        env_type="highway-v0",
        front_distance=30.0,
        front_speed=18.0,
        ttc=6.0,
        thw=2.0,
        legal_actions=[0, 1, 2, 3, 4],
        can_change_right=True,
    )
    state.set_effective_action_universe(
        [0, 1, 2, 3, 4], source="async-test"
    )
    state.__dict__["_runtime_frame_id"] = int(frame)
    state.__dict__["_test_fast_action"] = int(fast_action)
    return state


def _decision(action: int) -> RGDDecision:
    return RGDDecision(
        action=int(action),
        reasoning=f"decision:{int(action)}",
        confidence=1.0,
        system_used="slow",
        route_label="controlled",
        route_score=1.0,
        stats={"slow_reasoning_success": True},
    )


class _ControlledSlow:
    """Deterministic Future source for orchestrator lifecycle tests."""

    def __init__(self, timeout_s: float = 30.0) -> None:
        self.timeout_s = float(timeout_s)
        self.requests = []

    def submit(self, **kwargs) -> SlowRequest:
        now = time.perf_counter()
        request = SlowRequest(
            request_id=str(kwargs["request_id"]),
            episode_token=str(kwargs["episode_token"]),
            source_frame=int(kwargs["source_frame"]),
            release_frame=int(kwargs["release_frame"]),
            submitted_at_monotonic=now,
            deadline_at_monotonic=now + self.timeout_s,
            future=concurrent.futures.Future(),
        )
        self.requests.append(request)
        return request

    @staticmethod
    def is_ready(request: SlowRequest) -> bool:
        return SlowThinker.is_ready(request)

    @staticmethod
    def poll(request: SlowRequest):
        return SlowThinker.poll(request)

    @staticmethod
    def cancel(request: SlowRequest) -> bool:
        return SlowThinker.cancel(request)

    def get_runtime_budget_state(self):
        return {
            "llm_available": True,
            "provider_available": True,
            "executor_capacity_available": True,
            "llm_invoke_timeout_s": self.timeout_s,
        }


def _async_config(*, budget: int = 1, revalidation: bool = False):
    return {
        "env_type": "highway-v0",
        "policy_frequency": 10,
        "slow_call_budget": int(budget),
        "slow_call_cooldown_frames": 0,
        "rgd_signal_provider": False,
        "asynchronous_slow_path": {"enable": True, "min_release_frames": 1},
        "closed_loop_latency_replay": {
            "release_opportunity_revalidation": {"enable": bool(revalidation)}
        },
    }


def _install_controlled_route(orchestrator: RGDOrchestrator):
    fast_calls = []

    def resolve_route(self, state, force_system):
        del state, force_system
        self.stats["route_reason"] = "controlled_slow_route"
        return "slow", 1.0, None

    def execute_fast(self, state, route_score, fast_override_context=None):
        del fast_override_context
        action = int(state.__dict__["_test_fast_action"])
        fast_calls.append((int(state.__dict__["_runtime_frame_id"]), action))
        return RGDDecision(
            action=action,
            reasoning=f"fresh-fast:{action}",
            confidence=1.0,
            system_used="fast",
            route_label="controlled_fast",
            route_score=float(route_score),
            stats={"rule_name": f"fast-{action}"},
        )

    orchestrator._resolve_requested_route = MethodType(resolve_route, orchestrator)
    orchestrator._execute_fast = MethodType(execute_fast, orchestrator)
    return fast_calls


def _controlled_orchestrator(*, budget: int = 1):
    slow = _ControlledSlow()
    orchestrator = RGDOrchestrator(
        fast_thinker=object(),
        slow_thinker=slow,
        config=_async_config(budget=budget, revalidation=False),
    )
    calls = _install_controlled_route(orchestrator)
    return orchestrator, slow, calls


def test_native_async_and_scripted_latency_engines_are_mutually_exclusive():
    cfg = _async_config()
    cfg["closed_loop_latency_replay"]["enable"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        RGDOrchestrator(object(), object(), cfg)


def test_external_replay_request_uses_the_runtime_budget_and_cooldown():
    orchestrator = RGDOrchestrator(
        object(),
        object(),
        {
            "slow_call_budget": 2,
            "slow_call_cooldown_frames": 3,
            "asynchronous_slow_path": {"enable": False},
        },
    )

    orchestrator.record_external_slow_request()
    assert orchestrator.stats["slow_call_attempts"] == 1
    assert orchestrator.stats["slow_call_budget_remaining"] == 1
    assert orchestrator._slow_call_cooldown_remaining == 4
    with pytest.raises(RuntimeError, match="cooldown"):
        orchestrator.record_external_slow_request()

    orchestrator._slow_call_cooldown_remaining = 0
    orchestrator.record_external_slow_request()
    assert orchestrator.stats["slow_call_attempts"] == 2
    assert orchestrator.stats["slow_call_budget_remaining"] == 0
    with pytest.raises(RuntimeError, match="budget"):
        orchestrator.record_external_slow_request()


def test_forced_fast_control_does_not_enable_wall_clock_pacing():
    cfg = _async_config()
    cfg["system_routing"] = {"simple": "fast", "complex": "fast"}
    agent = DriverAgentV2(config=cfg, slow_action_provider=lambda *_: 1)
    try:
        assert agent.uses_native_async_policy_pacing() is False
    finally:
        agent.close()


def test_async_request_is_not_issued_after_the_response_window_closes():
    orchestrator, slow, _ = _controlled_orchestrator(budget=2)
    orchestrator._async_latest_source_frame = 0

    decision = orchestrator.decide(_state(1, int(ActionType.IDLE)))

    assert slow.requests == []
    assert decision.stats["closed_loop_latency_issuance_event"] is False
    assert "episode_response_window_closed" in decision.stats["route_reason"]


def test_slow_thinker_submit_and_poll_are_nonblocking_and_freeze_inputs():
    provider_started = threading.Event()
    provider_release = threading.Event()
    observed = {}

    def provider(state, context):
        provider_started.set()
        if not provider_release.wait(timeout=2.0):
            raise RuntimeError("test provider was not released")
        observed["speed"] = state.ego_speed
        observed["nested"] = context["nested"]["value"]
        return int(ActionType.SLOWER)

    thinker = SlowThinker(
        {"request_timeout_s": 2.0, "max_workers": 1},
        action_provider=provider,
    )
    request = None
    try:
        state = _state(0)
        context = {"nested": {"value": 7}}
        started_at = time.perf_counter()
        request = thinker.submit(
            request_id="async-copy",
            episode_token="episode-copy",
            source_frame=0,
            release_frame=1,
            state=state,
            recoverability_context=context,
        )
        assert time.perf_counter() - started_at < 0.25
        assert provider_started.wait(timeout=1.0)

        poll_started = time.perf_counter()
        assert thinker.poll(request) is None
        assert time.perf_counter() - poll_started < 0.25

        state.ego_speed = 99.0
        context["nested"]["value"] = 99
        provider_release.set()
        request.future.result(timeout=1.0)
        result = thinker.poll(request)

        assert result is not None
        assert result.action == int(ActionType.SLOWER)
        assert observed == {"speed": 18.0, "nested": 7}
    finally:
        provider_release.set()
        if request is not None:
            request.future.result(timeout=2.0)
        thinker.shutdown(wait=True)


def test_slow_thinker_concurrent_submit_admits_exactly_one(monkeypatch):
    provider_release = threading.Event()

    def provider(state, context):
        del state, context
        if not provider_release.wait(timeout=2.0):
            raise RuntimeError("test provider was not released")
        return int(ActionType.IDLE)

    thinker = SlowThinker(
        {"request_timeout_s": 2.0, "max_workers": 1},
        action_provider=provider,
    )
    # This barrier makes the pre-fix check-then-submit race deterministic.  The
    # fixed implementation performs admission directly under the executor lock
    # and therefore never consults this monkeypatched observational API.
    original_budget = thinker.get_runtime_budget_state
    budget_barrier = threading.Barrier(2)

    def synchronized_budget():
        budget = original_budget()
        budget_barrier.wait(timeout=2.0)
        return budget

    monkeypatch.setattr(thinker, "get_runtime_budget_state", synchronized_budget)
    start = threading.Barrier(3)
    outcomes = []

    def submit(index):
        start.wait(timeout=2.0)
        try:
            request = thinker.submit(
                request_id=f"concurrent-{index}",
                episode_token="episode",
                source_frame=0,
                release_frame=1,
                state=_state(0),
            )
            outcomes.append(("accepted", request))
        except SlowPathUnavailableError as exc:
            outcomes.append((str(exc.failure_reason), None))

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=2.0)
    for thread in threads:
        thread.join(timeout=2.0)
        assert not thread.is_alive()

    provider_release.set()
    thinker.shutdown(wait=True)
    assert [label for label, _ in outcomes].count("accepted") == 1
    assert [label for label, _ in outcomes].count(
        "slow_executor_capacity_exhausted"
    ) == 1


def test_slow_thinker_poll_uses_worker_completion_time_before_deadline():
    future = concurrent.futures.Future()
    future.set_result(_decision(int(ActionType.IDLE)))
    now = time.perf_counter()
    request = SlowRequest(
        request_id="deadline-race",
        episode_token="episode",
        source_frame=0,
        release_frame=1,
        submitted_at_monotonic=now - 2.0,
        deadline_at_monotonic=now - 0.5,
        future=future,
        completion_clock={"completed_at_monotonic": now - 1.0},
    )

    assert SlowThinker.poll(request).action == int(ActionType.IDLE)


def test_slow_thinker_rejects_fractional_action_identity():
    thinker = SlowThinker(action_provider=lambda state, context: 1.5)
    try:
        with pytest.raises(
            SlowPathUnavailableError,
            match="offline_proposal_action_identity_invalid",
        ):
            thinker.think(state=_state(0))
    finally:
        thinker.shutdown(wait=True)


def test_orchestrator_keeps_one_request_and_recomputes_fast_while_pending():
    orchestrator, slow, fast_calls = _controlled_orchestrator(budget=3)

    issued = orchestrator.decide(_state(0, int(ActionType.IDLE)))
    request = slow.requests[0]
    assert issued.action == int(ActionType.IDLE)
    assert issued.stats["closed_loop_latency_issuance_event"] is True
    assert orchestrator.prepare_frame(0) is None

    pending = orchestrator.decide(_state(1, int(ActionType.FASTER)))

    assert pending.action == int(ActionType.FASTER)
    assert pending.stats["slow_request_pending"] is True
    assert len(slow.requests) == 1
    assert fast_calls == [
        (0, int(ActionType.IDLE)),
        (1, int(ActionType.FASTER)),
    ]
    assert request.release_frame == 1

    dropped = orchestrator.end_episode("test_cleanup")
    assert [row["request_id"] for row in dropped] == [request.request_id]


@pytest.mark.parametrize(
    ("outcome", "expected_action"),
    [
        ("valid", int(ActionType.SLOWER)),
        ("failure", int(ActionType.IDLE)),
        ("timeout", int(ActionType.IDLE)),
    ],
)
def test_orchestrator_emits_valid_failure_and_timeout_once(outcome, expected_action):
    orchestrator, slow, _ = _controlled_orchestrator(budget=1)
    query = orchestrator.decide(_state(0, int(ActionType.IDLE)))
    request = slow.requests[0]

    assert query.stats["closed_loop_latency_terminal_event"] is False
    if outcome == "valid":
        request.future.set_result(_decision(int(ActionType.SLOWER)))
    elif outcome == "failure":
        request.future.set_exception(SlowPathUnavailableError("provider_failure"))
    else:
        request.deadline_at_monotonic = time.perf_counter() - 1.0

    descriptor = orchestrator.prepare_frame(1)
    assert descriptor is not None
    assert descriptor["response_outcome"] == outcome
    terminal = orchestrator.decide(_state(1, int(ActionType.IDLE)))

    assert terminal.action == expected_action
    assert terminal.stats["closed_loop_latency_terminal_event"] is True
    assert terminal.stats["closed_loop_latency_terminal_request_id"] == request.request_id
    assert terminal.stats["closed_loop_latency_terminal_response_outcome"] == outcome
    assert orchestrator.pending_slow_requests() == []

    assert orchestrator.prepare_frame(2) is None
    following = orchestrator.decide(_state(2, int(ActionType.FASTER)))
    assert following.stats["closed_loop_latency_terminal_event"] is False
    assert len(slow.requests) == 1


def test_orchestrator_drop_is_idempotent_and_terminal():
    orchestrator, slow, _ = _controlled_orchestrator(budget=1)
    orchestrator.decide(_state(0))
    request = slow.requests[0]

    first = orchestrator.end_episode("truncated")
    second = orchestrator.end_episode("duplicate")

    assert len(first) == 1
    assert first[0]["request_id"] == request.request_id
    assert first[0]["terminal_outcome"] == "dropped_at_episode_end"
    assert first[0]["drop_reason"] == "truncated"
    assert second == []
    assert orchestrator.pending_slow_requests() == []
    assert request.future.cancelled()
    with pytest.raises(RuntimeError, match="episode ledger is closed"):
        orchestrator.decide(_state(1))


def test_async_failure_keeps_latched_request_when_fast_fallback_raises():
    orchestrator, slow, _ = _controlled_orchestrator()
    orchestrator.decide(_state(0, int(ActionType.IDLE)))
    request = slow.requests[0]
    request.future.set_exception(SlowPathUnavailableError("provider_failure"))
    assert orchestrator.prepare_frame(1)["request_id"] == request.request_id

    original_fast = orchestrator._execute_fast

    def fail_fast(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("transient fast failure")

    orchestrator._execute_fast = fail_fast
    with pytest.raises(RuntimeError, match="transient fast failure"):
        orchestrator.decide(_state(1, int(ActionType.FASTER)))

    assert orchestrator.pending_slow_requests()[0]["request_id"] == request.request_id
    orchestrator._execute_fast = original_fast
    terminal = orchestrator.decide(_state(1, int(ActionType.FASTER)))
    assert terminal.stats["closed_loop_latency_terminal_request_id"] == request.request_id
    assert terminal.stats["closed_loop_latency_failure_event"] is True


def test_execute_episode_step_rejects_unmatched_native_terminal(monkeypatch):
    descriptor = {
        "request_id": "latched-request",
        "source_frame": 0,
        "release_frame": 1,
        "available_frame": 1,
        "scheduled_steps": 1,
        "response_outcome": "valid",
        "native_async": True,
    }
    agent = SimpleNamespace(prepare_frame=lambda frame: dict(descriptor))
    monkeypatch.setattr(
        runtime_support,
        "_pace_native_async_policy_frame",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        runtime_support,
        "_capture_online_release_snapshot_if_due",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        runtime_support,
        "run_frame_protocol",
        lambda **kwargs: (
            int(ActionType.IDLE),
            "",
            "",
            {},
            0.0,
            0.0,
            {},
            None,
            _state(1),
            "",
            {
                "native_async_slow_path": True,
                "closed_loop_latency_terminal_event": False,
                "closed_loop_latency_terminal_request_id": "",
            },
        ),
    )

    with pytest.raises(RuntimeError, match="does not match the frame decision"):
        runtime_support.execute_episode_step(
            frame=1,
            env=SimpleNamespace(),
            sce=SimpleNamespace(),
            agent=agent,
            obs=None,
            cfg={"closed_loop_latency_replay": {"enable": False}},
            safety_system=SimpleNamespace(),
            phys_rec=None,
            reas_rec=None,
            history_buffer=deque(maxlen=2),
            episode_state=create_episode_runtime_state(),
        )


def test_episode_event_latches_frame_and_orders_sets_deterministically():
    event = build_episode_event(
        7,
        {"frame": 99, "action_set": {4, 1, 3}},
        {"frame": 100, "final_action": 1},
        {"frame": 101},
    )

    assert event["frame"] == 7
    assert event["action_set"] == [1, 3, 4]


def _release_evaluator():
    config = {
        "closed_loop_latency_replay": {
            "release_opportunity_revalidation": {
                "enable": True,
                "action_cost_alignment": {
                    "enable": True,
                    "cost_margin": 0.20,
                    "required_method_version": "identifiable_gate_v12",
                    "required_raw_cost_source": "safety_cost_decomposition",
                },
            }
        }
    }
    orchestrator = RGDOrchestrator(object(), object(), config)
    assessment = {
        "domain_contract_pass": True,
        "absolute_feasibility_pass": True,
        "maneuver_breadth_pass": True,
        "raw_feasible_alternative_actions": [int(ActionType.SLOWER)],
        "corrective_headroom_pass": True,
        "state_need_pass": True,
    }
    rad = {
        "action_recovery_costs": {
            int(ActionType.IDLE): 0.80,
            int(ActionType.SLOWER): 0.60,
        },
        "method_version": "identifiable_gate_v12",
        "raw_cost_source": "safety_cost_decomposition",
        "raw_cost_complete": True,
    }
    return orchestrator, assessment, rad


def _evaluate(orchestrator, state, assessment, rad, *, fast=1, slow=4):
    orchestrator._last_recoverability_context = {
        "recoverability_assessment": copy.deepcopy(assessment)
    }
    orchestrator._last_rad_meta = copy.deepcopy(rad)
    return orchestrator.evaluate_release_proposal(
        state=state,
        fast_action=int(fast),
        slow_action=int(slow),
    )


def test_release_evaluator_enforces_all_gates_margin_and_cost_provenance():
    orchestrator, assessment, rad = _release_evaluator()
    state = _state(3)

    accepted = _evaluate(orchestrator, state, assessment, rad)
    assert accepted["release_pass"] is True
    assert accepted["a_pass"] is True
    assert accepted["h_pass"] is True
    assert accepted["n_pass"] is True
    assert accepted["distinct"] is True
    assert accepted["alignment_evaluated"] is True
    assert accepted["alignment_pass"] is True
    assert accepted["alignment_margin"] == pytest.approx(0.20)
    assert accepted["method_version_pass"] is True
    assert accepted["raw_cost_source_pass"] is True
    assert accepted["cost_provenance_pass"] is True
    assert accepted["raw_cost_complete"] is True

    no_a = copy.deepcopy(assessment)
    no_a["raw_feasible_alternative_actions"] = []
    assert _evaluate(orchestrator, state, no_a, rad)["a_pass"] is False

    no_h = copy.deepcopy(assessment)
    no_h["corrective_headroom_pass"] = False
    assert _evaluate(orchestrator, state, no_h, rad)["h_pass"] is False

    no_n = copy.deepcopy(assessment)
    no_n["state_need_pass"] = False
    assert _evaluate(orchestrator, state, no_n, rad)["n_pass"] is False

    same_action = copy.deepcopy(assessment)
    same_action["raw_feasible_alternative_actions"] = [int(ActionType.IDLE)]
    equivalent = _evaluate(
        orchestrator,
        state,
        same_action,
        rad,
        fast=int(ActionType.IDLE),
        slow=int(ActionType.IDLE),
    )
    assert equivalent["distinct"] is False
    assert equivalent["release_pass"] is False

    weak_margin = copy.deepcopy(rad)
    weak_margin["action_recovery_costs"][int(ActionType.SLOWER)] = 0.61
    margin_rejected = _evaluate(orchestrator, state, assessment, weak_margin)
    assert margin_rejected["alignment_evaluated"] is True
    assert margin_rejected["alignment_pass"] is False
    assert margin_rejected["release_pass"] is False

    wrong_method = copy.deepcopy(rad)
    wrong_method["method_version"] = "other_method"
    method_rejected = _evaluate(orchestrator, state, assessment, wrong_method)
    assert method_rejected["method_version_pass"] is False
    assert method_rejected["cost_provenance_pass"] is False
    assert method_rejected["alignment_evaluated"] is False
    assert method_rejected["release_pass"] is False

    wrong_source = copy.deepcopy(rad)
    wrong_source["raw_cost_source"] = "heuristic_cost"
    source_rejected = _evaluate(orchestrator, state, assessment, wrong_source)
    assert source_rejected["raw_cost_source_pass"] is False
    assert source_rejected["cost_provenance_pass"] is False
    assert source_rejected["alignment_evaluated"] is False
    assert source_rejected["release_pass"] is False


def test_driver_pending_snapshot_rules_and_restore_are_atomic():
    agent = DriverAgentV2(
        config=_async_config(budget=1, revalidation=False),
        slow_action_provider=lambda state, context: int(ActionType.SLOWER),
    )
    controlled = _ControlledSlow()
    agent.orchestrator.slow = controlled
    _install_controlled_route(agent.orchestrator)

    idle = DriverAgentV2(
        config={
            "env_type": "highway-v0",
            "policy_frequency": 10,
            "system_routing": {"simple": "fast", "complex": "fast"},
            "slow_call_budget": 0,
        }
    )
    try:
        assert agent.uses_native_async_policy_pacing() is True
        assert idle.uses_native_async_policy_pacing() is False
        agent.decide(_state(0))
        pending = agent.pending_slow_requests()
        assert len(pending) == 1

        release_state = agent.snapshot_release_policy_state()
        with pytest.raises(RuntimeError, match="standalone policy snapshot"):
            agent.snapshot_policy_state()
        with pytest.raises(RuntimeError, match="restore policy state"):
            agent.restore_policy_state(copy.deepcopy(release_state))
        assert agent.snapshot_release_policy_state() == release_state

        snapshot = capture_release_snapshot(
            agent,
            frame=1,
            env={"frame": 1},
            obs=[1.0],
            history=deque([{"frame": 0, "action": 1}], maxlen=4),
            previous_action=int(ActionType.IDLE),
            pending_request=pending[0],
        )
        assert snapshot.request_id == pending[0]["request_id"]
        assert validate_release_snapshot_policy_state(
            snapshot, context="async driver release"
        ) == release_state

        baseline = idle.snapshot_policy_state()
        invalid = copy.deepcopy(baseline)
        invalid["orchestrator"]["decision_count"] += 7
        invalid["fast"]["action_history_capacity"] += 1
        with pytest.raises(ValueError, match="capacity differs"):
            idle.restore_policy_state(invalid)
        assert idle.snapshot_policy_state() == baseline
    finally:
        agent.end_episode("test_cleanup")
        agent.close()
        idle.close()


class _FakePolicyClock:
    def __init__(self, *, now=0.0, order=None):
        self.now = float(now)
        self.order = order
        self.monotonic_calls = 0
        self.sleep_calls = []

    def monotonic(self):
        self.monotonic_calls += 1
        return self.now

    def sleep(self, duration):
        duration = float(duration)
        self.sleep_calls.append(duration)
        if self.order is not None:
            self.order.append("pace")
        self.now += duration

    def advance(self, duration):
        self.now += float(duration)


def _install_policy_clock(monkeypatch, clock, *, perf_counter=None):
    monkeypatch.setattr(
        runtime_support,
        "time",
        SimpleNamespace(
            monotonic=clock.monotonic,
            sleep=clock.sleep,
            perf_counter=perf_counter or time.perf_counter,
        ),
    )


def test_native_async_policy_pacing_uses_absolute_10_hz_deadlines(monkeypatch):
    clock = _FakePolicyClock(now=12.0)
    _install_policy_clock(monkeypatch, clock)
    agent = SimpleNamespace(uses_native_async_policy_pacing=lambda: True)
    episode = {}

    frame_zero = runtime_support._pace_native_async_policy_frame(
        frame=0, agent=agent, episode_state=episode
    )
    frame_one = runtime_support._pace_native_async_policy_frame(
        frame=1, agent=agent, episode_state=episode
    )
    clock.advance(0.15)
    overrun = runtime_support._pace_native_async_policy_frame(
        frame=2, agent=agent, episode_state=episode
    )
    recovered = runtime_support._pace_native_async_policy_frame(
        frame=3, agent=agent, episode_state=episode
    )

    assert frame_zero["policy_pacing_sleep_s"] == 0.0
    assert frame_one["policy_pacing_sleep_s"] == pytest.approx(0.1)
    assert overrun["policy_pacing_sleep_s"] == 0.0
    assert recovered["policy_pacing_sleep_s"] == pytest.approx(0.05)
    assert clock.sleep_calls == pytest.approx([0.1, 0.05])
    assert episode["_total_policy_pacing_sleep_s"] == pytest.approx(0.15)


def test_scripted_and_factorial_agents_never_consult_the_wall_clock(monkeypatch):
    clock = _FakePolicyClock()
    _install_policy_clock(monkeypatch, clock)
    inner = SimpleNamespace(
        orchestrator=SimpleNamespace(_async_slow_path_enabled=True)
    )
    factorial = ProposalReplayAgent(
        inner,
        {},
        arm=FACTORIAL_ARMS[0],
        bank_sha256="factorial-bank",
    )

    scripted_meta = runtime_support._pace_native_async_policy_frame(
        frame=1,
        agent=SimpleNamespace(orchestrator=inner.orchestrator),
        episode_state={},
    )
    factorial_meta = runtime_support._pace_native_async_policy_frame(
        frame=1,
        agent=factorial,
        episode_state={},
    )

    assert scripted_meta["policy_pacing_enabled"] is False
    assert factorial_meta["policy_pacing_enabled"] is False
    assert clock.monotonic_calls == 0
    assert clock.sleep_calls == []


def test_algorithm_runtime_excludes_measured_policy_pacing_sleep():
    episode = {
        "_last_policy_pacing_sleep_s": 0.1,
        "_total_policy_pacing_sleep_s": 0.4,
    }

    assert runtime_support.exclude_policy_pacing_sleep(0.137, episode) == pytest.approx(
        0.037
    )
    assert runtime_support.exclude_policy_pacing_sleep(0.05, episode) == 0.0
    assert runtime_support.exclude_policy_pacing_sleep(0.137, {}) == pytest.approx(
        0.137
    )


class _RuntimeAgent:
    def __init__(self, order):
        self.order = order
        self.frame = -1
        self.fast_thinker = SimpleNamespace(snapshot_runtime_state=lambda: {})
        self.orchestrator = SimpleNamespace(
            evaluate_release_proposal=self._evaluate_release
        )

    def uses_native_async_policy_pacing(self):
        return True

    def prepare_frame(self, frame):
        self.frame = int(frame)
        self.order.append("prepare")
        return {
            "request_id": f"native-{self.frame}",
            "source_frame": max(0, self.frame - 1),
            "release_frame": self.frame,
            "available_frame": self.frame,
            "scheduled_steps": 1,
            "response_outcome": "valid",
            "native_async": True,
        }

    def _evaluate_release(self, **kwargs):
        del kwargs
        self.order.append("revalidation")
        return {"release_pass": True}

    def decide(self, state):
        self.order.append("decide")
        self.orchestrator.evaluate_release_proposal(state=state)
        request_id = f"native-{self.frame}"
        return (
            int(ActionType.SLOWER),
            "native async release",
            {
                "native_async_slow_path": True,
                "latency_ms": 7.5,
                "system_used": "slow_release",
                "closed_loop_latency_terminal_event": True,
                "closed_loop_latency_terminal_request_id": request_id,
                "closed_loop_latency_terminal_response_outcome": "valid",
                "closed_loop_latency_response_outcome": "valid",
                "closed_loop_latency_release_event": True,
                "closed_loop_execution_state_fast_action": int(ActionType.IDLE),
                "closed_loop_released_slow_action": int(ActionType.SLOWER),
                "release_fast_comparator_action": int(ActionType.IDLE),
                "release_selected_action": int(ActionType.SLOWER),
                "release_action_comparison_stage": (
                    "post_release_guard_pre_final_safety_projection"
                ),
            },
        )

    def record_executed_action(self, action):
        del action


class _RuntimeSafety:
    def __init__(self, order):
        self.order = order
        self.calls = 0

    def apply_action_safety_stack(self, action, actions, state, *, frame):
        del actions, state, frame
        self.calls += 1
        self.order.append("safety")
        return SimpleNamespace(
            final_action=int(action),
            safety_override=False,
            shield_override=False,
            emergency_level=0,
            safety_reason="",
            shield_reason="",
        )


class _RuntimeEnv:
    def __init__(self, order):
        self.order = order
        self.unwrapped = self
        self.vehicle = None

    def step(self, action):
        self.order.append("env.step")
        return [float(action)], 0.0, False, False, {"crashed": False}


class _RuntimeReasoning:
    def __init__(self):
        self.records = []

    def record_reasoning(self, **kwargs):
        self.records.append(dict(kwargs))


def test_execute_episode_step_applies_safety_once_in_the_release_order(monkeypatch):
    order = []
    clock = _FakePolicyClock(order=order)
    _install_policy_clock(monkeypatch, clock, perf_counter=clock.monotonic)
    agent = _RuntimeAgent(order)
    safety = _RuntimeSafety(order)
    env = _RuntimeEnv(order)
    reasoning = _RuntimeReasoning()
    state = _state(0)

    monkeypatch.setattr(
        runtime_support,
        "extract_runtime_state",
        lambda env, sce, cfg: {
            "scenario_type": "highway",
            "speed": 18.0,
            "front_distance": 30.0,
            "ttc": 6.0,
            "thw": 2.0,
        },
    )
    monkeypatch.setattr(
        runtime_support, "_raw_available_actions", lambda env: [1, 4]
    )
    monkeypatch.setattr(
        runtime_support,
        "build_frame_driving_state",
        lambda *args, **kwargs: copy.deepcopy(state),
    )
    monkeypatch.setattr(
        runtime_support, "inject_safety_cost_bridge", lambda *args, **kwargs: None
    )

    def capture_probe(**kwargs):
        assert kwargs["pending_request"]["response_outcome"] == "valid"
        order.append("snapshot")

    monkeypatch.setattr(
        runtime_support, "_capture_online_release_snapshot_if_due", capture_probe
    )
    monkeypatch.setattr(
        runtime_support,
        "_apply_unavailable_slower_brake_assist",
        lambda env, action, cfg: int(action),
    )
    monkeypatch.setattr(
        runtime_support,
        "_release_highway_hidden_brake_target",
        lambda env, action, cfg: int(action),
    )

    episode = create_episode_runtime_state()
    history = deque(maxlen=4)
    cfg = {
        "render_mode": "none",
        "capture_release_snapshots_online": True,
        "require_release_snapshot_on_release": False,
        "closed_loop_latency_replay": {"enable": False},
    }
    for frame in range(2):
        runtime_support.execute_episode_step(
            frame=frame,
            env=env,
            sce=SimpleNamespace(describe=lambda value: f"frame:{value}"),
            agent=agent,
            obs=[float(frame)],
            cfg=cfg,
            safety_system=safety,
            phys_rec=None,
            reas_rec=reasoning,
            history_buffer=history,
            episode_state=episode,
        )

    expected_frame_order = [
        "prepare",
        "snapshot",
        "decide",
        "revalidation",
        "safety",
        "env.step",
    ]
    assert order == expected_frame_order + ["pace"] + expected_frame_order
    assert safety.calls == 2
    assert len(episode["event_log"]) == 2
    assert episode["event_log"][0]["policy_pacing_sleep_s"] == 0.0
    assert episode["event_log"][1]["policy_pacing_sleep_s"] == pytest.approx(0.1)
    assert [event["latency_ms"] for event in episode["event_log"]] == [7.5, 7.5]
    assert [
        record["inference_end_time"] - record["inference_start_time"]
        for record in reasoning.records
    ] == [0.0, 0.0]
