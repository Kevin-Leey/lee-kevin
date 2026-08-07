import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dilu.evaluation.factorial_replay import (
    FACTORIAL_ARMS,
    FACTORIAL_REPLAY_VERSION,
    FAST_ONLY_ARM,
    FORMAL_FACTORIAL_ARMS,
    FactorialArm,
    ProposalRecord,
    ProposalReplayAgent,
    configure_factorial_arm,
    proposal_bank_sha256,
)
from dilu.evaluation.decision_trace import build_decision_meta
from dilu.runtime_frame_trace import build_episode_event
from dilu.runtime_support import (
    _apply_closed_loop_latency_replay,
    _capture_online_release_snapshot_if_due,
)
from tools.run_query_release_factorial import _release_execution_is_distinct


class _FakeState:
    def __init__(self, label: str, available=range(5)) -> None:
        self.label = label
        self.legal_actions = tuple(int(action) for action in available)

    def __str__(self) -> str:
        return self.label


class _FakeInnerAgent:
    def __init__(self, *, gate_pass: bool, action: int = 1) -> None:
        self.gate_pass = bool(gate_pass)
        self.action = int(action)
        self.fast_thinker = SimpleNamespace(name="fast")
        self.external_request_count = 0
        self.orchestrator = SimpleNamespace(
            name="rgd",
            record_external_slow_request=self._record_external_slow_request,
        )
        self.recorded_actions = []
        self.restored_snapshot = None

    def _record_external_slow_request(self):
        self.external_request_count += 1

    def decide(self, state):
        return (
            self.action,
            f"fast:{state}",
            {
                "recoverability_gate": {
                    "serial_gate_pass": self.gate_pass,
                },
                "inner_metadata": "preserved",
            },
        )

    def snapshot_policy_state(self):
        return {"policy": "snapshot"}

    def restore_policy_state(self, snapshot):
        self.restored_snapshot = snapshot

    def record_executed_action(self, action):
        self.recorded_actions.append(int(action))


def _proposal(
    *,
    seed: int = 4000,
    frame: int = 0,
    request_id: str = "factorial:4000:0:00",
    action: int = 4,
    latency_steps: int = 17,
    outcome: str = "valid",
) -> ProposalRecord:
    return ProposalRecord(
        seed=seed,
        source_frame=frame,
        request_id=request_id,
        raw_slow_action=action,
        latency_steps=latency_steps,
        outcome=outcome,
        response_text=f"response:{request_id}",
        response_sha256="a" * 64,
        source_artifact="frozen/reasoning.jsonl",
    )


def _arm(name: str) -> FactorialArm:
    return next(arm for arm in FACTORIAL_ARMS if arm.name == name)


def test_factorial_arms_cover_the_query_release_truth_table_once():
    observed = {
        arm.name: (arm.query_gate_enabled, arm.release_guard_enabled)
        for arm in FACTORIAL_ARMS
    }

    assert observed == {
        "full": (True, True),
        "query_only": (True, False),
        "release_only": (False, True),
        "neither": (False, False),
    }
    assert len(FACTORIAL_ARMS) == len(set(observed.values())) == 4


def test_formal_design_adds_a_genuine_fast_only_control():
    assert FORMAL_FACTORIAL_ARMS[:-1] == FACTORIAL_ARMS
    assert FORMAL_FACTORIAL_ARMS[-1] == FAST_ONLY_ARM

    proposal = _proposal()
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=True),
        {proposal.source_frame: proposal},
        arm=FAST_ONLY_ARM,
        bank_sha256="bank-digest",
    )

    action, response, metadata = agent.decide(_FakeState("fast-only"))

    assert action == 1
    assert response == "fast:fast-only"
    assert metadata["factorial_candidate_query"] is True
    assert metadata["factorial_query_issued"] is False
    assert metadata["factorial_query_rejection_reason"] == "fast_only_control"
    assert agent.inner.external_request_count == 0
    assert agent.candidate_count == agent.gate_rejected_count == 1
    assert agent.issued_count == 0


@pytest.mark.parametrize("arm_name", ("full", "query_only"))
def test_enabled_query_gate_rejects_a_frozen_candidate(arm_name):
    proposal = _proposal()
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=False),
        {proposal.source_frame: proposal},
        arm=_arm(arm_name),
        bank_sha256="bank-digest",
    )

    action, response, metadata = agent.decide(_FakeState("state-0"))

    assert action == 1
    assert response == "fast:state-0"
    assert metadata["factorial_candidate_query"] is True
    assert metadata["factorial_candidate_request_id"] == proposal.request_id
    assert metadata["factorial_query_gate_pass"] is False
    assert metadata["factorial_query_issued"] is False
    assert metadata["factorial_query_rejection_reason"] == "query_gate_failed"
    assert "closed_loop_latency_request_id" not in metadata
    assert agent.candidate_count == 1
    assert agent.issued_count == 0
    assert agent.gate_rejected_count == 1
    assert agent.inner.external_request_count == 0
    assert agent.last_system_used == "fast"


def test_all_arms_replay_the_same_frozen_proposal_identity_when_eligible():
    proposal = _proposal()
    observed = []

    for arm in FACTORIAL_ARMS:
        agent = ProposalReplayAgent(
            _FakeInnerAgent(gate_pass=True),
            {proposal.source_frame: proposal},
            arm=arm,
            bank_sha256="bank-digest",
        )
        action, response, metadata = agent.decide(_FakeState("shared-state"))
        observed.append(
            (
                action,
                response,
                metadata["factorial_candidate_request_id"],
                metadata["closed_loop_latency_request_id"],
                metadata["slow_request_id"],
                metadata["factorial_shared_raw_slow_action"],
                metadata["factorial_shared_latency_steps"],
                metadata["factorial_shared_response_sha256"],
                metadata["factorial_shared_response_outcome"],
            )
        )
        assert metadata["factorial_arm"] == arm.name
        assert metadata["factorial_query_issued"] is True
        assert metadata["factorial_policy_state_synchronized"] is True
        assert metadata["slow_request_attempted"] is True
        assert metadata["slow_request_valid_return"] is True
        assert agent.inner.external_request_count == 1
        assert agent.candidate_count == agent.issued_count == 1

    assert len(set(observed)) == 1
    assert observed[0] == (
        proposal.raw_slow_action,
        proposal.response_text,
        proposal.request_id,
        proposal.request_id,
        proposal.request_id,
        proposal.raw_slow_action,
        proposal.latency_steps,
        proposal.response_sha256,
        proposal.outcome,
    )


def test_query_gate_disabled_arm_issues_candidate_even_when_gate_fails():
    proposal = _proposal()
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=False),
        {proposal.source_frame: proposal},
        arm=_arm("release_only"),
        bank_sha256="bank-digest",
    )

    action, _, metadata = agent.decide(_FakeState("state-0"))

    assert action == proposal.raw_slow_action
    assert metadata["factorial_query_gate_pass"] is False
    assert metadata["factorial_query_issued"] is True
    assert metadata["closed_loop_latency_request_id"] == proposal.request_id
    assert agent.issued_count == 1
    assert agent.gate_rejected_count == 0


def test_logged_proposal_falls_back_to_fast_when_illegal_on_the_arm_trajectory():
    proposal = _proposal(action=4)
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=True, action=1),
        {proposal.source_frame: proposal},
        arm=_arm("neither"),
        bank_sha256="bank-digest",
    )

    action, _, metadata = agent.decide(
        _FakeState("counterfactual-state", available=(0, 1, 2, 3))
    )

    assert action == 1
    assert metadata["factorial_shared_raw_slow_action"] == 4
    assert metadata["factorial_query_action_available"] is False
    assert metadata["factorial_query_mapped_action"] == 1
    assert metadata["query_state_slow_pre_guard_action"] == 4
    assert metadata["query_state_slow_released_action"] == 1
    assert metadata["factorial_query_mapping_reason"] == (
        "raw_action_unavailable_fast_fallback"
    )


def test_timeout_is_issued_as_a_pending_request_and_executes_fast():
    proposal = _proposal(outcome="timeout", latency_steps=2)
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=True, action=2),
        {proposal.source_frame: proposal},
        arm=_arm("full"),
        bank_sha256="bank-digest",
    )

    action, response, metadata = agent.decide(_FakeState("state-0"))

    assert action == 2
    assert response == "[proposal timeout pending]"
    assert metadata["factorial_query_issued"] is True
    assert metadata["factorial_policy_state_synchronized"] is True
    assert metadata["closed_loop_latency_request_id"] == proposal.request_id
    assert metadata["closed_loop_latency_response_outcome"] == "timeout"
    assert metadata["closed_loop_latency_terminal_outcome"] == "pending"
    assert metadata["closed_loop_latency_timeout_event"] is False
    assert metadata["slow_request_attempted"] is True
    assert metadata["slow_request_valid_return"] is False
    assert metadata["slow_request_failed"] is False
    assert metadata["slow_reasoning_success"] is False
    assert metadata["slow_reasoning_failure_reason"] == ""
    assert metadata["system_used"] == "slow"
    assert metadata["route_label"] == "factorial_shared_proposal_pending"
    assert agent.candidate_count == agent.issued_count == 1
    assert agent.timeout_count == 0
    assert agent.inner.external_request_count == 1
    assert agent.last_system_used == "slow"


def test_decision_trace_preserves_the_request_scoped_factorial_contract():
    proposal = _proposal(latency_steps=35)
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=True),
        {proposal.source_frame: proposal},
        arm=_arm("full"),
        bank_sha256="b" * 64,
    )
    action, _, metadata = agent.decide(_FakeState("state-0"))

    traced = build_decision_meta(
        metadata,
        proposed_action=action,
        final_action=action,
    )

    assert traced["closed_loop_latency_request_id"] == proposal.request_id
    assert traced["slow_request_id"] == proposal.request_id
    assert traced["closed_loop_scripted_latency_steps"] == 35
    assert traced["factorial_candidate_request_id"] == proposal.request_id
    assert traced["factorial_proposal_bank_sha256"] == "b" * 64
    assert traced["factorial_query_issued"] is True
    assert traced["factorial_policy_state_synchronized"] is True
    assert traced["slow_request_attempted"] is True
    assert traced["slow_request_valid_return"] is True

    agent.decide(_FakeState("pending-state"))
    assert agent.timeout_count == 0
    agent.decide(_FakeState("terminal-state"))
    assert agent.timeout_count == 1


def test_issued_query_fails_closed_without_policy_state_synchronization_hook():
    proposal = _proposal()
    inner = _FakeInnerAgent(gate_pass=True)
    inner.orchestrator = SimpleNamespace(name="rgd")
    agent = ProposalReplayAgent(
        inner,
        {proposal.source_frame: proposal},
        arm=_arm("full"),
        bank_sha256="bank-digest",
    )

    with pytest.raises(RuntimeError, match="record_external_slow_request"):
        agent.decide(_FakeState("state-0"))

    assert agent.candidate_count == 1
    assert agent.issued_count == 0


def test_valid_factorial_proposal_keeps_the_existing_delayed_release_contract():
    proposal = _proposal(outcome="valid", latency_steps=2)
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=True, action=2),
        {proposal.source_frame: proposal},
        arm=_arm("full"),
        bank_sha256="bank-digest",
    )
    proposed_action, _, raw_meta = agent.decide(_FakeState("source-state"))
    source_meta = build_decision_meta(
        raw_meta,
        proposed_action=proposed_action,
        final_action=proposed_action,
    )
    cfg = {
        "policy_frequency": 10,
        "closed_loop_latency_replay": {
            "enable": True,
            "delay_steps": 2,
            "extra_latency_s": 0.2,
            "target_systems": ["slow"],
        },
    }
    episode = {"action": 1, "latency_replay_queue": []}

    source_action = _apply_closed_loop_latency_replay(
        frame=0,
        action=proposed_action,
        decision_meta=source_meta,
        episode_state=episode,
        cfg=cfg,
    )
    episode["action"] = source_action
    assert source_action == 2
    assert source_meta["closed_loop_latency_terminal_outcome"] == "pending"
    assert source_meta["slow_request_valid_return"] is True

    for frame, fast_action in ((1, 3), (2, 1)):
        release_meta = {"system_used": "fast"}
        released = _apply_closed_loop_latency_replay(
            frame=frame,
            action=fast_action,
            decision_meta=release_meta,
            episode_state=episode,
            cfg=cfg,
        )
        episode["action"] = released

    assert released == proposal.raw_slow_action
    assert release_meta["closed_loop_latency_release_event"] is True
    assert release_meta["closed_loop_latency_timeout_event"] is False
    assert release_meta["closed_loop_latency_failure_event"] is False
    assert release_meta["closed_loop_latency_response_outcome"] == "valid"
    assert release_meta["closed_loop_latency_request_id"] == proposal.request_id
    assert release_meta["closed_loop_latency_realized_steps"] == 2
    assert episode["latency_replay_queue"] == []


@pytest.mark.parametrize(
    ("outcome", "timeout_event", "failure_event"),
    (("timeout", True, False), ("failure", False, True)),
)
def test_failed_proposal_terminates_asynchronously_without_a_release_snapshot(
    outcome, timeout_event, failure_event
):
    proposal = _proposal(outcome=outcome, latency_steps=2)
    agent = ProposalReplayAgent(
        _FakeInnerAgent(gate_pass=True, action=2),
        {proposal.source_frame: proposal},
        arm=_arm("full"),
        bank_sha256="bank-digest",
    )
    proposed_action, _, raw_meta = agent.decide(_FakeState("source-state"))
    source_meta = build_decision_meta(
        raw_meta,
        proposed_action=proposed_action,
        final_action=proposed_action,
    )
    cfg = {
        "policy_frequency": 10,
        "capture_release_snapshots_online": True,
        "require_release_snapshot_on_release": True,
        "closed_loop_latency_replay": {
            "enable": True,
            "delay_steps": 2,
            "extra_latency_s": 0.2,
            "target_systems": ["slow"],
        },
    }
    episode = {
        "action": 1,
        "latency_replay_queue": [],
        "release_snapshots": {},
    }

    source_action = _apply_closed_loop_latency_replay(
        frame=0,
        action=proposed_action,
        decision_meta=source_meta,
        episode_state=episode,
        cfg=cfg,
    )
    episode["action"] = source_action

    assert source_action == 2
    assert source_meta["closed_loop_latency_terminal_outcome"] == "pending"
    assert source_meta["closed_loop_latency_release_event"] is False
    assert source_meta["closed_loop_latency_timeout_event"] is False
    assert source_meta["closed_loop_latency_failure_event"] is False
    assert source_meta["slow_request_failed"] is False
    assert len(episode["latency_replay_queue"]) == 1
    assert episode["latency_replay_queue"][0]["request_id"] == proposal.request_id
    assert episode["latency_replay_queue"][0]["response_outcome"] == outcome

    pending_meta = {"system_used": "fast"}
    pending_action = _apply_closed_loop_latency_replay(
        frame=1,
        action=3,
        decision_meta=pending_meta,
        episode_state=episode,
        cfg=cfg,
    )
    episode["action"] = pending_action
    assert pending_action == 3
    assert pending_meta["closed_loop_latency_terminal_outcome"] == "pending"
    assert pending_meta["closed_loop_latency_request_id"] == proposal.request_id
    assert pending_meta["slow_request_attempted"] is False

    _capture_online_release_snapshot_if_due(
        frame=2,
        env=object(),
        obs=object(),
        agent=object(),
        history_buffer=[],
        episode_state=episode,
        cfg=cfg,
    )
    assert episode["release_snapshots"] == {}

    terminal_meta = {"system_used": "fast"}
    terminal_action = _apply_closed_loop_latency_replay(
        frame=2,
        action=1,
        decision_meta=terminal_meta,
        episode_state=episode,
        cfg=cfg,
    )

    assert terminal_action == 1
    assert terminal_meta["closed_loop_latency_request_id"] == proposal.request_id
    assert terminal_meta["closed_loop_latency_response_outcome"] == outcome
    assert terminal_meta["closed_loop_latency_terminal_outcome"] == outcome
    assert terminal_meta["closed_loop_latency_timeout_event"] is timeout_event
    assert terminal_meta["closed_loop_latency_failure_event"] is failure_event
    assert terminal_meta["closed_loop_latency_release_event"] is False
    assert terminal_meta["closed_loop_release_snapshot_captured"] is False
    assert terminal_meta["closed_loop_latency_scheduled_steps"] == 2
    assert terminal_meta["closed_loop_latency_realized_steps"] == 2
    assert terminal_meta["slow_request_failed"] is False
    assert terminal_meta["slow_reasoning_failure_reason"] == outcome
    assert episode["latency_replay_queue"] == []
    assert episode["release_snapshots"] == {}

    event = build_episode_event(2, {}, terminal_meta, {})
    assert event["closed_loop_latency_response_outcome"] == outcome
    assert event["closed_loop_latency_terminal_outcome"] == outcome
    assert event["closed_loop_latency_timeout_event"] is timeout_event
    assert event["closed_loop_latency_failure_event"] is failure_event
    assert event["closed_loop_latency_release_event"] is False
    assert event["slow_request_attempted"] is False
    assert event["slow_request_failed"] is False


def test_second_failed_response_is_not_issued_while_one_request_is_pending():
    cfg = {
        "policy_frequency": 10,
        "closed_loop_latency_replay": {
            "enable": True,
            "delay_steps": 2,
            "extra_latency_s": 0.2,
            "target_systems": ["slow"],
        },
    }
    episode = {"action": 1, "latency_replay_queue": []}

    def issue(frame, request_id, outcome, latency_steps, fast_action):
        meta = {
            "system_used": "slow",
            "slow_request_attempted": True,
            "slow_request_valid_return": False,
            "slow_request_failed": False,
            "closed_loop_latency_request_id": request_id,
            "slow_request_id": request_id,
            "closed_loop_latency_response_outcome": outcome,
            "closed_loop_latency_terminal_outcome": "pending",
            "closed_loop_scripted_latency_steps": latency_steps,
            "query_state_fast_proposal_action": fast_action,
        }
        executed = _apply_closed_loop_latency_replay(
            frame=frame,
            action=fast_action,
            decision_meta=meta,
            episode_state=episode,
            cfg=cfg,
        )
        episode["action"] = executed
        return meta

    issue(0, "request-z", "timeout", 2, 1)
    suppressed = issue(1, "request-a", "failure", 1, 2)
    assert suppressed["closed_loop_latency_issuance_event"] is False
    assert [item["request_id"] for item in episode["latency_replay_queue"]] == [
        "request-z"
    ]

    first_meta = {"system_used": "fast"}
    first_action = _apply_closed_loop_latency_replay(
        frame=2,
        action=3,
        decision_meta=first_meta,
        episode_state=episode,
        cfg=cfg,
    )
    episode["action"] = first_action
    assert first_action == 3
    assert first_meta["closed_loop_latency_request_id"] == "request-z"
    assert first_meta["closed_loop_latency_timeout_event"] is True
    assert episode["latency_replay_queue"] == []


def test_zero_step_failed_response_still_has_a_distinct_pending_source_event():
    cfg = {
        "policy_frequency": 10,
        "closed_loop_latency_replay": {
            "enable": True,
            "delay_steps": 0,
            "extra_latency_s": 0.0,
            "target_systems": ["slow"],
        },
    }
    episode = {"action": 1, "latency_replay_queue": []}
    source_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": False,
        "slow_request_failed": False,
        "closed_loop_latency_request_id": "zero-timeout",
        "closed_loop_latency_response_outcome": "timeout",
        "closed_loop_latency_terminal_outcome": "pending",
        "closed_loop_scripted_latency_steps": 0,
        "query_state_fast_proposal_action": 2,
    }

    source_action = _apply_closed_loop_latency_replay(
        frame=0,
        action=2,
        decision_meta=source_meta,
        episode_state=episode,
        cfg=cfg,
    )
    assert source_action == 2
    assert source_meta["closed_loop_latency_terminal_outcome"] == "pending"
    assert source_meta["closed_loop_latency_timeout_event"] is False
    assert len(episode["latency_replay_queue"]) == 1

    terminal_meta = {"system_used": "fast"}
    terminal_action = _apply_closed_loop_latency_replay(
        frame=1,
        action=3,
        decision_meta=terminal_meta,
        episode_state=episode,
        cfg=cfg,
    )
    assert terminal_action == 3
    assert terminal_meta["closed_loop_latency_terminal_outcome"] == "timeout"
    assert terminal_meta["closed_loop_latency_scheduled_steps"] == 0
    assert terminal_meta["closed_loop_latency_realized_steps"] == 1


def test_unguarded_release_counts_the_observed_execution_change():
    event = {
        "closed_loop_latency_release_event": True,
        "closed_loop_execution_state_fast_action": 1,
        "closed_loop_latency_executed_action": 4,
        "closed_loop_release_action_alignment_evaluated": False,
        "closed_loop_release_action_alignment_pass": False,
    }

    assert _release_execution_is_distinct(event) is True
    assert _release_execution_is_distinct(
        {**event, "closed_loop_release_opportunity_rejected": True}
    ) is False


@pytest.mark.parametrize(
    ("arm_name", "guard_enabled"),
    (
        ("full", True),
        ("query_only", False),
        ("release_only", True),
        ("neither", False),
    ),
)
def test_arm_configuration_controls_both_release_guard_layers_without_mutation(
    arm_name, guard_enabled
):
    base = {
        "closed_loop_latency_replay": {
            "enable": False,
            "target_systems": ["fast"],
            "unrelated": "preserved",
            "release_opportunity_revalidation": {
                "enable": not guard_enabled,
                "unrelated": "preserved",
                "action_cost_alignment": {
                    "enable": not guard_enabled,
                    "margin": 0.02,
                },
            },
        },
        "capture_release_snapshots_online": False,
        "require_release_snapshot_on_release": False,
    }
    original = copy.deepcopy(base)

    configured = configure_factorial_arm(base, _arm(arm_name))
    replay = configured["closed_loop_latency_replay"]
    revalidation = replay["release_opportunity_revalidation"]

    assert base == original
    assert replay["enable"] is True
    assert replay["target_systems"] == ["slow"]
    assert replay["unrelated"] == "preserved"
    assert revalidation["enable"] is guard_enabled
    assert revalidation["unrelated"] == "preserved"
    assert revalidation["action_cost_alignment"]["enable"] is guard_enabled
    assert revalidation["action_cost_alignment"]["margin"] == 0.02
    assert configured["capture_release_snapshots_online"] is True
    assert configured["require_release_snapshot_on_release"] is True
    assert configured["factorial_replay_version"] == FACTORIAL_REPLAY_VERSION
    assert configured["factorial_arm"] == {
        "name": arm_name,
        "query_gate_enabled": _arm(arm_name).query_gate_enabled,
        "release_guard_enabled": guard_enabled,
    }


def test_proposal_bank_hash_is_order_independent_but_content_sensitive():
    early = _proposal(seed=4000, frame=3, request_id="request-early", action=1)
    late = _proposal(seed=4001, frame=9, request_id="request-late", action=3)
    canonical = {4000: {3: early}, 4001: {9: late}}
    reordered = {4001: {9: late}, 4000: {3: early}}

    digest = proposal_bank_sha256(canonical)

    assert proposal_bank_sha256(reordered) == digest
    assert len(digest) == 64
    assert proposal_bank_sha256({**canonical, 4002: {}}) != digest
    assert proposal_bank_sha256(
        {
            4000: {3: replace(early, source_artifact="relocated/reasoning.json")},
            4001: {9: replace(late, source_artifact="another/root/reasoning.json")},
        }
    ) == digest
    assert proposal_bank_sha256(
        {
            4000: {3: early},
            4001: {
                9: _proposal(
                    seed=4001,
                    frame=9,
                    request_id="request-late",
                    action=4,
                )
            },
        }
    ) != digest
