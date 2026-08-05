from types import SimpleNamespace

import pytest

from dilu.driver_agent.reasoning.decision import RGDDecision
from dilu.driver_agent.reasoning.rgd_core import RGDOrchestrator
from dilu.driver_agent.reasoning.rgd_support import build_slow_path_latency_context
from dilu.driver_agent.driverAgentV2 import DriverAgentV2
from dilu.latency_contract import bind_latency_contract, resolve_latency_contract


def _environment(frequency: float):
    return SimpleNamespace(
        unwrapped=SimpleNamespace(config={"policy_frequency": frequency})
    )


def test_seconds_are_bound_to_the_final_environment_frequency():
    cfg = {
        "policy_frequency": 10,
        "closed_loop_latency_replay": {
            "enable": True,
            "extra_latency_s": 0.26,
            "delay_steps": 99,
        },
    }

    bound = bind_latency_contract(cfg, _environment(4))
    contract = resolve_latency_contract(bound)

    assert contract["predicted_seconds"] == pytest.approx(0.26)
    assert contract["predicted_steps"] == 2
    assert contract["scheduled_steps"] == 2
    assert contract["scheduled_seconds"] == pytest.approx(0.5)
    assert contract["policy_frequency_hz"] == pytest.approx(4.0)
    assert contract["configured_steps"] == 99
    assert contract["configured_steps_consistent"] is False


def test_driver_preparation_binds_the_scenario_environment_contract():
    driver = object.__new__(DriverAgentV2)
    driver.sce = SimpleNamespace(env=_environment(4).unwrapped)

    prepared = driver._prepare_runtime_config(
        {
            "policy_frequency": 10,
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.26,
            },
        }
    )

    assert prepared["_resolved_policy_frequency_hz"] == pytest.approx(4.0)
    assert prepared["_resolved_latency_contract"]["scheduled_steps"] == 2


def test_legacy_steps_remain_supported_when_seconds_are_absent():
    cfg = {
        "closed_loop_latency_replay": {
            "enable": True,
            "delay_steps": 3,
        }
    }

    contract = resolve_latency_contract(cfg, _environment(4))

    assert contract["source"] == "closed_loop_latency_replay.delay_steps_legacy"
    assert contract["predicted_steps"] == 3
    assert contract["scheduled_steps"] == 3
    assert contract["predicted_seconds"] == pytest.approx(0.75)
    assert contract["scheduled_seconds"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    "raw_value,reason",
    [("not-a-number", "invalid_numeric_latency"), (float("nan"), "nonfinite_latency"), (float("inf"), "nonfinite_latency"), (-0.1, "negative_latency")],
)
def test_invalid_seconds_fail_closed_instead_of_becoming_zero_delay(raw_value, reason):
    contract = resolve_latency_contract(
        {
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": raw_value,
                "delay_steps": 0,
            }
        },
        _environment(10),
    )

    assert contract["prediction_available"] is False
    assert contract["prediction_invalid_reason"] == reason
    assert contract["scheduled_steps"] == 0
    assert contract["configured_steps_consistent"] is False


def test_invalid_legacy_steps_fail_closed():
    contract = resolve_latency_contract(
        {
            "closed_loop_latency_replay": {
                "enable": True,
                "delay_steps": -1,
            }
        },
        _environment(10),
    )

    assert contract["prediction_available"] is False
    assert contract["prediction_invalid_reason"] == "invalid_delay_steps"
    assert contract["scheduled_steps"] == 0


def test_gate_context_uses_the_contract_step_count_without_requantizing():
    context = build_slow_path_latency_context(
        llm_available=True,
        llm_invoke_timeout_s=30.0,
        short_horizon_seconds=2.0,
        predicted_slow_latency_s=0.26,
        latency_source="unit_test",
        policy_frequency=4.0,
        resolved_delay_steps=2,
    )

    assert context["effective_delay_steps"] == 2
    assert context["predicted_slow_latency_seconds"] == pytest.approx(0.5)
    assert context["requested_slow_latency_seconds"] == pytest.approx(0.26)


def test_release_guard_uses_scheduled_contract_not_legacy_audit_steps():
    bound = bind_latency_contract(
        {
            "release_dominance_guard": {"enable": True},
            "closed_loop_latency_replay": {
                "enable": True,
                "extra_latency_s": 0.26,
                "delay_steps": 0,
            },
        },
        _environment(4),
    )
    slow = SimpleNamespace(
        think=lambda **_kwargs: SimpleNamespace(
            action=4,
            reasoning="slow",
            confidence=1.0,
            thinking_steps=[],
            agent_opinions={"risk_scores_by_action": {1: 0.5, 4: 0.1}},
        )
    )
    orchestrator = RGDOrchestrator(
        fast_thinker=SimpleNamespace(),
        slow_thinker=slow,
        config=bound,
    )
    orchestrator._peek_full_fast_decision = lambda *_args, **_kwargs: RGDDecision(
        action=1,
        reasoning="fast",
        confidence=1.0,
        system_used="fast",
        route_label="rgd_core",
        route_score=0.0,
        stats={"rule_name": "unit_test"},
    )
    orchestrator._highway_fast_pass_override = lambda _state, action: {
        "rgd_highway_pass_resolved_action": int(action)
    }

    decision = orchestrator._execute_slow_path(
        state=SimpleNamespace(),
        route_score=1.0,
        route_ambiguity_profile=None,
        recoverability_context={},
    )

    assert decision.stats["release_dominance_guard_scope"] == (
        "deferred_positive_delay_release"
    )
    assert decision.stats["release_dominance_guard_scheduled_delay_steps"] == 2
    assert decision.stats["release_dominance_guard_applied"] is False
