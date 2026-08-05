from types import SimpleNamespace

import pytest

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.rad import RADSignalController
from dilu.runtime_frame_state import inject_safety_cost_bridge
from dilu.runtime_support import (
    _apply_unavailable_slower_brake_assist,
    _available_actions_description,
    _inject_hidden_slower_action,
    _release_highway_hidden_brake_target,
)


def _risk_state(**overrides):
    state = {
        "speed": 20.0,
        "front_dist": 6.0,
        "front_speed": 14.0,
        "ttc": 1.0,
        "thw": 0.3,
    }
    state.update(overrides)
    return state


def test_hidden_slower_is_injected_only_for_enabled_risky_following():
    raw = [int(ActionType.IDLE), int(ActionType.LANE_LEFT), int(ActionType.FASTER)]
    expected_without_brake = sorted(raw)

    assert _inject_hidden_slower_action(raw, _risk_state(), {"hidden_slower_bridge": {"enable": False}}) == raw
    assert _inject_hidden_slower_action(
        raw,
        _risk_state(front_dist=80.0, ttc=20.0, thw=4.0),
        {"hidden_slower_bridge": {"enable": True}},
    ) == expected_without_brake

    effective = _inject_hidden_slower_action(raw, _risk_state(), {"hidden_slower_bridge": {"enable": True}})
    assert int(ActionType.SLOWER) in effective


def test_safety_cost_bridge_uses_authoritative_hidden_slower_universe():
    captured = {}

    class SafetyProbe:
        def get_action_cost_decomposition(self, state, available_actions):
            captured["actions"] = list(available_actions)
            return {
                int(action): {
                    "total": 0.1 + 0.05 * index,
                    "safety": 0.1 + 0.05 * index,
                    "comfort": 0.0,
                    "efficiency": 0.0,
                }
                for index, action in enumerate(available_actions)
            }

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            get_available_actions=lambda: [int(ActionType.IDLE)]
        )
    )
    driving_state = DrivingState(
        legal_actions=[int(ActionType.IDLE), int(ActionType.SLOWER)],
        front_distance=10.0,
        ttc=1.5,
        thw=0.5,
    )

    inject_safety_cost_bridge(driving_state, env, _risk_state(), SafetyProbe())

    assert captured["actions"] == [int(ActionType.IDLE), int(ActionType.SLOWER)]
    decomposition = driving_state.__dict__["_safety_cost_decomposition"]
    assert sorted(decomposition) == [int(ActionType.IDLE), int(ActionType.SLOWER)]
    _, rad_meta = RADSignalController().estimate_signal(
        driving_state,
        conflict_score=0.0,
        action_universe=(int(ActionType.IDLE), int(ActionType.SLOWER)),
    )
    assert rad_meta["raw_cost_complete"] is True
    assert rad_meta["missing_raw_cost_actions"] == []
    assert rad_meta["nonfinite_raw_cost_actions"] == []


def test_hidden_slower_brake_assist_pins_target_below_simulator_floor():
    vehicle = SimpleNamespace(
        speed=20.0,
        speed_index=0,
        target_speed=20.0,
        target_speeds=[20.0, 25.0, 30.0],
    )
    unwrapped = SimpleNamespace(
        get_available_actions=lambda: [int(ActionType.IDLE), int(ActionType.FASTER)],
        vehicle=vehicle,
        config={"env_type": "highway-v0"},
    )
    env = SimpleNamespace(unwrapped=unwrapped, config={}, spec=SimpleNamespace(id="highway-v0"))
    cfg = {
        "env_type": "highway-v0",
        "scenario_type": "highway",
        "hidden_slower_bridge": {"enable": True},
        "_current_runtime_state": _risk_state(),
    }

    executed = _apply_unavailable_slower_brake_assist(env, int(ActionType.SLOWER), cfg)

    assert executed == int(ActionType.IDLE)
    assert vehicle.speed_index == 0
    assert vehicle.target_speed < 20.0


def test_effective_action_description_records_injected_brake():
    text = _available_actions_description([int(ActionType.IDLE), int(ActionType.SLOWER)])

    assert "Action_id: 1" in text
    assert "Action_id: 4" in text


def test_brake_assist_is_noop_when_bridge_is_disabled():
    vehicle = SimpleNamespace(speed=20.0, speed_index=0, target_speed=20.0, target_speeds=[20.0, 25.0, 30.0])
    unwrapped = SimpleNamespace(
        get_available_actions=lambda: [int(ActionType.IDLE)],
        vehicle=vehicle,
        config={"env_type": "highway-v0"},
    )
    env = SimpleNamespace(unwrapped=unwrapped, config={}, spec=SimpleNamespace(id="highway-v0"))

    executed = _apply_unavailable_slower_brake_assist(
        env,
        int(ActionType.SLOWER),
        {"hidden_slower_bridge": {"enable": False}},
    )

    assert executed == int(ActionType.SLOWER)
    assert vehicle.target_speed == pytest.approx(20.0)


def test_hidden_brake_latch_blocks_cruise_recovery_until_release_is_safe():
    vehicle = SimpleNamespace(
        speed=9.0,
        speed_index=0,
        target_speed=20.0,
        target_speeds=[20.0, 25.0, 30.0],
    )
    unwrapped = SimpleNamespace(
        get_available_actions=lambda: [int(ActionType.IDLE), int(ActionType.FASTER)],
        vehicle=vehicle,
        config={"env_type": "highway-v0"},
    )
    env = SimpleNamespace(unwrapped=unwrapped, config={}, spec=SimpleNamespace(id="highway-v0"))
    cfg = {
        "env_type": "highway-v0",
        "scenario_type": "highway",
        "hidden_slower_bridge": {"enable": True},
        "_current_runtime_state": _risk_state(speed=9.0, front_dist=5.5, front_speed=5.0, ttc=1.4, thw=0.61),
    }

    assert _apply_unavailable_slower_brake_assist(env, int(ActionType.SLOWER), cfg) == int(ActionType.IDLE)
    brake_target = vehicle.target_speed
    assert brake_target < 20.0

    # Emulate a simulator/controller overwriting the continuous target between frames.
    vehicle.target_speed = 20.0
    held_action = _release_highway_hidden_brake_target(env, int(ActionType.FASTER), cfg)

    assert held_action == int(ActionType.IDLE)
    assert vehicle.speed_index == 0
    assert vehicle.target_speed == pytest.approx(brake_target)

    vehicle.target_speed = 20.0
    held_idle = _release_highway_hidden_brake_target(env, int(ActionType.IDLE), cfg)

    assert held_idle == int(ActionType.IDLE)
    assert vehicle.target_speed == pytest.approx(brake_target)

    cfg["_current_runtime_state"] = _risk_state(
        speed=9.0,
        front_dist=30.0,
        front_speed=10.0,
        ttc=float("inf"),
        thw=3.3,
    )
    released_action = _release_highway_hidden_brake_target(env, int(ActionType.FASTER), cfg)

    assert released_action == int(ActionType.FASTER)
    assert vehicle.speed_index == 0
    assert vehicle.target_speed == pytest.approx(20.0)


def test_hidden_brake_latch_keeps_lateral_escape_but_blocks_adjacent_release():
    vehicle = SimpleNamespace(
        speed=5.2,
        speed_index=0,
        target_speed=8.0,
        target_speeds=[20.0, 25.0, 30.0],
    )
    setattr(vehicle, "_dilu_highway_hidden_brake_latched", True)
    setattr(vehicle, "_dilu_highway_hidden_brake_target", 8.0)
    env = SimpleNamespace(unwrapped=SimpleNamespace(vehicle=vehicle))
    cfg = {
        "env_type": "highway-v0",
        "scenario_type": "highway",
        "hidden_slower_bridge": {"enable": True},
        "_current_runtime_state": {
            "scenario_type": "highway",
            "speed": 5.2,
            "front_dist": 30.0,
            "front_speed": 8.0,
            "ttc": float("inf"),
            "thw": 5.7,
            "closest_vehicle_distance": 6.44,
            "closest_vehicle_longitudinal": 3.29,
            "closest_vehicle_lateral": -5.53,
            "closest_vehicle_closing_speed": -0.56,
        },
    }

    lateral_action = _release_highway_hidden_brake_target(env, int(ActionType.LANE_LEFT), cfg)
    faster_action = _release_highway_hidden_brake_target(env, int(ActionType.FASTER), cfg)

    assert lateral_action == int(ActionType.LANE_LEFT)
    assert faster_action == int(ActionType.IDLE)
    assert vehicle.target_speed == pytest.approx(8.0)
