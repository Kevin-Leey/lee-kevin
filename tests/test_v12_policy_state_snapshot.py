import collections
import copy
import pickle

import pytest

from dilu.driver_agent.base.state import DrivingState
from dilu.driver_agent.policy_state import (
    DRIVER_POLICY_STATE_SCHEMA,
    FAST_POLICY_STATE_SCHEMA,
    RAD_POLICY_STATE_SCHEMA,
    RGD_POLICY_STATE_SCHEMA,
    policy_state_sha256,
)
from tools.analyze_release_state_rollouts import (
    FastBranchAgent,
    OfflineDriverAgent,
    ReleaseSnapshot,
    capture_release_snapshot,
    validate_release_snapshot_policy_state,
)
from tools.run_v12_branch_labels import _load_snapshots


def _config():
    return {
        "env_type": "highway-v0",
        "system_routing": {"simple": "fast", "complex": "fast"},
        "slow_call_budget": 6,
        "slow_call_cooldown_frames": 20,
        "fast_thinking": {"lane_change_cooldown": 4},
        "policy_frequency": 10,
    }


def _state(seed: int, frame: int) -> DrivingState:
    state = DrivingState(
        ego_speed=18.0 + float((seed + frame) % 7),
        front_distance=22.0 + float((seed * 3 + frame) % 45),
        front_speed=17.0,
        front_relative_speed=-2.0,
        ttc=3.5 + float((seed + frame) % 8) * 0.4,
        thw=1.2 + float((seed + frame) % 6) * 0.3,
        scenario_type="highway",
        env_type="highway-v0",
        legal_actions=[0, 1, 3, 4],
        can_change_left=True,
        left_front_distance=45.0,
        left_rear_distance=20.0,
        left_front_speed=20.0,
        left_rear_speed=18.0,
        nearby_vehicle_count=3,
        history_frames=[
            {
                "speed": 18.0,
                "ttc": 4.0,
                "thw": 2.0,
                "action": int((seed + frame) % 5),
            }
        ],
    )
    state.__dict__["_safety_cost_decomposition"] = {
        0: {"total": 0.20, "safety": 0.10, "comfort": 0.10, "efficiency": 0.00},
        1: {"total": 0.30, "safety": 0.20, "comfort": 0.05, "efficiency": 0.05},
        3: {"total": 0.50, "safety": 0.30, "comfort": 0.10, "efficiency": 0.10},
        4: {"total": 0.15, "safety": 0.10, "comfort": 0.05, "efficiency": 0.00},
    }
    return state


def _prime(agent: OfflineDriverAgent, seed: int, frames: int = 8) -> None:
    for frame in range(frames):
        action, _, _ = agent.decide(_state(seed, frame))
        agent.record_executed_action(action)


def _release_snapshot(policy_state):
    return ReleaseSnapshot(
        frame=7,
        env=None,
        obs=None,
        fast_state={
            "action_history": collections.deque(maxlen=12),
            "stats": {},
            "last_rule_match": None,
        },
        history=collections.deque(maxlen=6),
        previous_action=1,
        policy_state_schema=DRIVER_POLICY_STATE_SCHEMA,
        policy_state=policy_state,
        policy_state_sha256=policy_state_sha256(policy_state),
    )


def test_policy_state_round_trip_preserves_every_allowlisted_field():
    source = OfflineDriverAgent(config=_config())
    for action in (3, 1, 0, 4):
        source.record_executed_action(action)
    orchestrator = source.orchestrator
    orchestrator.stats["decision_count"] = 17
    orchestrator._support_progress_cooldown = 3
    orchestrator._rgd_cruise_progress_cooldown = 2
    orchestrator._rgd_cruise_recovery_frames = 5
    orchestrator._slow_call_attempts = 4
    orchestrator._slow_call_cooldown_remaining = 9
    orchestrator._rad_controller._corridor_boundary_ema = 0.41
    orchestrator._rad_controller._corridor_width_ema = 2.25
    orchestrator._rad_controller._last_corridor_stage = "critical"

    snapshot = source.snapshot_policy_state()
    serialized = pickle.loads(pickle.dumps(snapshot, protocol=pickle.HIGHEST_PROTOCOL))
    restored = OfflineDriverAgent(config=_config())
    restored.restore_policy_state(serialized)

    assert restored.snapshot_policy_state() == snapshot
    assert set(snapshot) == {"schema", "fast", "orchestrator"}
    assert set(snapshot["fast"]) == {
        "schema",
        "action_history",
        "action_history_capacity",
    }
    assert set(snapshot["orchestrator"]) == {
        "schema",
        "decision_count",
        "support_progress_cooldown",
        "rgd_cruise_progress_cooldown",
        "rgd_cruise_recovery_frames",
        "slow_call_attempts",
        "slow_call_cooldown_remaining",
        "rad",
    }
    assert set(snapshot["orchestrator"]["rad"]) == {
        "schema",
        "corridor_boundary_ema",
        "corridor_width_ema",
        "last_corridor_stage",
    }


def test_snapshot_producer_capture_authenticates_every_target_policy_state():
    agent = OfflineDriverAgent(config=_config())
    _prime(agent, 2004)
    history = collections.deque(({"action": 3, "speed": 21.0},), maxlen=6)

    snapshot = capture_release_snapshot(
        agent,
        frame=8,
        env={"frame": 8},
        obs=[1.0, 2.0],
        history=history,
        previous_action=3,
    )

    assert snapshot.policy_state_schema == DRIVER_POLICY_STATE_SCHEMA
    assert snapshot.policy_state == agent.snapshot_policy_state()
    assert snapshot.policy_state_sha256 == policy_state_sha256(
        snapshot.policy_state
    )
    assert snapshot.policy_state is not agent.snapshot_policy_state()
    assert snapshot.history is not history


@pytest.mark.parametrize("seed", [2004, 2011, 2026])
def test_known_proposal_regressions_restore_the_first_fast_proposal(seed):
    source = OfflineDriverAgent(config=_config())
    _prime(source, seed)
    replay = OfflineDriverAgent(config=_config())
    replay.restore_policy_state(source.snapshot_policy_state())

    source_action, _, source_meta = source.decide(_state(seed, 8))
    replay_action, _, replay_meta = replay.decide(_state(seed, 8))

    assert replay_action == source_action
    assert replay_meta["recoverability_gate"]["hold_action"] == (
        source_meta["recoverability_gate"]["hold_action"]
    )
    assert replay_meta["recoverability_gate"]["gate_action_universe"] == (
        source_meta["recoverability_gate"]["gate_action_universe"]
    )


@pytest.mark.parametrize("seed", [2001, 2010, 2023])
def test_known_continuation_regressions_replay_the_full_horizon(seed):
    source = OfflineDriverAgent(config=_config())
    _prime(source, seed)
    replay = OfflineDriverAgent(config=_config())
    replay.restore_policy_state(source.snapshot_policy_state())

    source_actions = []
    replay_actions = []
    for frame in range(8, 28):
        source_action, _, _ = source.decide(_state(seed, frame))
        replay_action, _, _ = replay.decide(_state(seed, frame))
        source_actions.append(source_action)
        replay_actions.append(replay_action)
        source.record_executed_action(source_action)
        replay.record_executed_action(replay_action)

    assert replay_actions == source_actions
    assert replay.snapshot_policy_state() == source.snapshot_policy_state()


def test_candidate_order_cannot_leak_policy_state_between_branches():
    source = OfflineDriverAgent(config=_config())
    _prime(source, 2004)
    policy_state = source.snapshot_policy_state()

    def evaluate(order):
        outcomes = {}
        for candidate in order:
            agent = FastBranchAgent(
                _config(),
                policy_state=policy_state,
                force_frame=8,
                force_action=candidate,
            )
            agent.frame = 8
            effective_action, _, _ = agent.decide(_state(2004, 8))
            proposal = agent.fast_actions[8]
            agent.record_executed_action(effective_action)
            continuation_action, _, _ = agent.decide(_state(2004, 9))
            outcomes[candidate] = (
                effective_action,
                proposal,
                continuation_action,
                agent.inner.snapshot_policy_state(),
            )
        return outcomes

    forward = evaluate((0, 1, 3))
    reverse = evaluate((3, 1, 0))

    assert forward == reverse
    assert source.snapshot_policy_state() == policy_state


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda snapshot: setattr(snapshot, "policy_state_schema", "legacy"),
            "policy-state schema drift",
        ),
        (
            lambda snapshot: snapshot.policy_state["orchestrator"].pop(
                "support_progress_cooldown"
            ),
            "field drift",
        ),
        (
            lambda snapshot: snapshot.policy_state["orchestrator"].update(
                unexpected_counter=1
            ),
            "field drift",
        ),
        (
            lambda snapshot: snapshot.policy_state["orchestrator"]["rad"].update(
                schema="rad_policy_state_v0"
            ),
            "schema drift",
        ),
        (
            lambda snapshot: snapshot.policy_state["orchestrator"].update(
                rgd_cruise_recovery_frames=4
            ),
            "SHA256 mismatch",
        ),
        (
            lambda snapshot: setattr(snapshot, "policy_state_sha256", None),
            "invalid policy-state SHA256",
        ),
    ],
)
def test_v12_snapshot_loader_fails_closed_on_schema_field_and_hash_tampering(
    tmp_path, mutation, message
):
    source = OfflineDriverAgent(config=_config())
    _prime(source, 2004)
    snapshot = _release_snapshot(source.snapshot_policy_state())
    mutation(snapshot)
    path = tmp_path / "snapshots.pkl"
    with path.open("wb") as handle:
        pickle.dump({7: snapshot}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match=message):
        _load_snapshots(path, [7], 2004)


def test_v12_rejects_legacy_bundle_while_legacy_fast_restore_remains_available(
    tmp_path,
):
    fast_state = {
        "action_history": collections.deque((3, 1), maxlen=12),
        "stats": {},
        "last_rule_match": None,
    }
    legacy = ReleaseSnapshot(
        frame=7,
        env=None,
        obs=None,
        fast_state=copy.deepcopy(fast_state),
        history=collections.deque(maxlen=6),
        previous_action=1,
    )
    path = tmp_path / "snapshots.pkl"
    with path.open("wb") as handle:
        pickle.dump({7: legacy}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with pytest.raises(ValueError, match="policy-state schema drift"):
        _load_snapshots(path, [7], 2004)

    with pytest.raises(ValueError, match="explicit diagnostic opt-in"):
        FastBranchAgent(_config(), fast_state=fast_state)

    with pytest.raises(ValueError, match="policy-state schema drift"):
        validate_release_snapshot_policy_state(
            legacy,
            context="legacy direct analyzer",
        )

    legacy_agent = FastBranchAgent(
        _config(),
        fast_state=fast_state,
        allow_legacy_fast_state=True,
    )
    assert list(legacy_agent.inner.fast_thinker.action_history) == [3, 1]


def test_policy_state_schema_constants_are_versioned_and_explicit():
    assert DRIVER_POLICY_STATE_SCHEMA == "driver_agent_v2_policy_state_v1"
    assert FAST_POLICY_STATE_SCHEMA == "fast_thinker_policy_state_v1"
    assert RGD_POLICY_STATE_SCHEMA == "rgd_orchestrator_policy_state_v1"
    assert RAD_POLICY_STATE_SCHEMA == "rad_signal_controller_policy_state_v1"
