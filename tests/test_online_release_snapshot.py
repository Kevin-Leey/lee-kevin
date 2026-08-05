import collections
import copy
import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

from dilu.driver_agent.policy_state import (
    DRIVER_POLICY_STATE_SCHEMA,
    FAST_POLICY_STATE_SCHEMA,
    RAD_POLICY_STATE_SCHEMA,
    RGD_POLICY_STATE_SCHEMA,
)
from dilu.evaluation.release_snapshot import (
    RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
    capture_release_snapshot,
    save_release_snapshot_bundle,
    validate_release_snapshot_policy_state,
)
from dilu.runtime_frame_trace import build_episode_event, create_episode_runtime_state
from dilu.runtime_support import (
    _apply_closed_loop_latency_replay,
    _capture_online_release_snapshot_if_due,
    _selected_ready_latency_request,
)


def _policy_state():
    return {
        "schema": DRIVER_POLICY_STATE_SCHEMA,
        "fast": {
            "schema": FAST_POLICY_STATE_SCHEMA,
            "action_history": [1, 3],
            "action_history_capacity": 12,
        },
        "orchestrator": {
            "schema": RGD_POLICY_STATE_SCHEMA,
            "decision_count": 4,
            "support_progress_cooldown": 0,
            "rgd_cruise_progress_cooldown": 0,
            "rgd_cruise_recovery_frames": 0,
            "slow_call_attempts": 1,
            "slow_call_cooldown_remaining": 0,
            "rad": {
                "schema": RAD_POLICY_STATE_SCHEMA,
                "corridor_boundary_ema": None,
                "corridor_width_ema": None,
                "last_corridor_stage": None,
            },
        },
    }


class _FakeFastThinker:
    def snapshot_runtime_state(self):
        return {
            "action_history": collections.deque([1, 3], maxlen=12),
            "stats": {},
            "last_rule_match": None,
        }


class _FakeAgent:
    fast_thinker = _FakeFastThinker()

    def snapshot_policy_state(self):
        return copy.deepcopy(_policy_state())


def _cfg():
    return {
        "policy_frequency": 10,
        "capture_release_snapshots_online": True,
        "require_release_snapshot_on_release": True,
        "closed_loop_latency_replay": {
            "enable": True,
            "extra_latency_s": 0.2,
            "delay_steps": 2,
            "target_systems": ["slow"],
        },
    }


def test_online_snapshot_is_bound_to_the_released_request(tmp_path):
    cfg = _cfg()
    episode = create_episode_runtime_state()
    query_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "query_state_fast_proposal_action": 1,
    }
    episode["action"] = _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=query_meta,
        episode_state=episode,
        cfg=cfg,
    )
    request_id = query_meta["closed_loop_latency_request_id"]

    _capture_online_release_snapshot_if_due(
        frame=2,
        env=SimpleNamespace(name="env-at-release"),
        obs=[2.0, 3.0],
        agent=_FakeAgent(),
        history_buffer=collections.deque([{"frame": 1}], maxlen=6),
        episode_state=episode,
        cfg=cfg,
    )
    release_meta = {"system_used": "fast"}
    released = _apply_closed_loop_latency_replay(
        frame=2,
        action=1,
        decision_meta=release_meta,
        episode_state=episode,
        cfg=cfg,
    )

    assert released == 4
    assert release_meta["closed_loop_latency_request_id"] == request_id
    assert release_meta["closed_loop_release_snapshot_captured"] is True
    snapshot = episode["release_snapshots"][request_id]
    assert snapshot.frame == 2
    assert snapshot.source_frame == 0
    assert snapshot.scheduled_release_frame == 2
    assert snapshot.request_id == request_id
    assert validate_release_snapshot_policy_state(snapshot, context="test") == _policy_state()

    event = build_episode_event(
        frame=2,
        state={},
        decision_meta=release_meta,
        terminal_outcome={},
    )
    assert event["closed_loop_release_snapshot_captured"] is True
    assert event["closed_loop_release_snapshot_identity_sha256"]

    bundle_path, manifest_path, digest = save_release_snapshot_bundle(
        episode["release_snapshots"],
        tmp_path,
        prefix="highway_0",
        episode_id=0,
    )
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == digest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == RELEASE_SNAPSHOT_BUNDLE_SCHEMA
    assert manifest["snapshot_count"] == 1
    payload = pickle.loads(bundle_path.read_bytes())
    assert set(payload["snapshots"]) == {request_id}


def test_explicit_zero_step_response_waits_for_a_snapshot_backed_release():
    cfg = _cfg()
    cfg["closed_loop_latency_replay"].update(
        {"extra_latency_s": 0.0, "delay_steps": 0}
    )
    episode = create_episode_runtime_state()
    query_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "slow_request_failed": False,
        "closed_loop_latency_request_id": "zero-valid",
        "closed_loop_latency_response_outcome": "valid",
        "closed_loop_latency_terminal_outcome": "pending",
        "closed_loop_scripted_latency_steps": 0,
        "query_state_fast_proposal_action": 1,
    }

    source_action = _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=query_meta,
        episode_state=episode,
        cfg=cfg,
    )
    episode["action"] = source_action

    assert source_action == 1
    assert query_meta["closed_loop_latency_terminal_outcome"] == "pending"
    assert query_meta["closed_loop_latency_release_event"] is False
    assert len(episode["latency_replay_queue"]) == 1

    _capture_online_release_snapshot_if_due(
        frame=1,
        env=SimpleNamespace(name="env-at-release"),
        obs=[1.0],
        agent=_FakeAgent(),
        history_buffer=collections.deque([], maxlen=6),
        episode_state=episode,
        cfg=cfg,
    )
    release_meta = {"system_used": "fast"}
    released = _apply_closed_loop_latency_replay(
        frame=1,
        action=2,
        decision_meta=release_meta,
        episode_state=episode,
        cfg=cfg,
    )

    assert released == 4
    assert release_meta["closed_loop_latency_release_event"] is True
    assert release_meta["closed_loop_release_snapshot_captured"] is True
    assert release_meta["closed_loop_latency_scheduled_steps"] == 0
    assert release_meta["closed_loop_latency_realized_steps"] == 1
    snapshot = episode["release_snapshots"]["zero-valid"]
    assert snapshot.frame == 1
    assert snapshot.source_frame == 0
    assert snapshot.scheduled_release_frame == 0


def test_required_capture_fails_closed_when_snapshot_is_missing():
    cfg = _cfg()
    episode = create_episode_runtime_state()
    query_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "query_state_fast_proposal_action": 1,
    }
    _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=query_meta,
        episode_state=episode,
        cfg=cfg,
    )
    with pytest.raises(RuntimeError, match="has no online snapshot"):
        _apply_closed_loop_latency_replay(
            frame=2,
            action=1,
            decision_meta={"system_used": "fast"},
            episode_state=episode,
            cfg=cfg,
        )
    assert len(episode["latency_replay_queue"]) == 1
    assert (
        episode["latency_replay_queue"][0]["request_id"]
        == query_meta["closed_loop_latency_request_id"]
    )


def test_invalid_release_snapshot_preserves_the_selected_queue_entry():
    cfg = _cfg()
    episode = create_episode_runtime_state()
    query_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "query_state_fast_proposal_action": 1,
    }
    _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=query_meta,
        episode_state=episode,
        cfg=cfg,
    )
    request_id = query_meta["closed_loop_latency_request_id"]
    _capture_online_release_snapshot_if_due(
        frame=2,
        env=SimpleNamespace(name="env-at-release"),
        obs=[2.0],
        agent=_FakeAgent(),
        history_buffer=collections.deque([], maxlen=6),
        episode_state=episode,
        cfg=cfg,
    )
    episode["release_snapshots"][request_id].request_id = "tampered"

    with pytest.raises(ValueError, match="identity mismatch"):
        _apply_closed_loop_latency_replay(
            frame=2,
            action=1,
            decision_meta={"system_used": "fast"},
            episode_state=episode,
            cfg=cfg,
        )

    assert [
        item["request_id"] for item in episode["latency_replay_queue"]
    ] == [request_id]


def test_same_frame_issuance_is_rolled_back_when_old_release_validation_fails():
    cfg = _cfg()
    episode = create_episode_runtime_state()
    old_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "closed_loop_latency_request_id": "old-valid",
        "closed_loop_latency_response_outcome": "valid",
        "query_state_fast_proposal_action": 1,
    }
    episode["action"] = _apply_closed_loop_latency_replay(
        frame=0,
        action=4,
        decision_meta=old_meta,
        episode_state=episode,
        cfg=cfg,
    )
    queue_before = copy.deepcopy(episode["latency_replay_queue"])

    new_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "closed_loop_latency_request_id": "new-valid",
        "closed_loop_latency_response_outcome": "valid",
        "query_state_fast_proposal_action": 2,
    }
    with pytest.raises(RuntimeError, match="old-valid.*has no online snapshot"):
        _apply_closed_loop_latency_replay(
            frame=2,
            action=3,
            decision_meta=new_meta,
            episode_state=episode,
            cfg=cfg,
        )

    assert episode["latency_replay_queue"] == queue_before
    assert episode["_latency_request_ids"] == {"old-valid"}


def test_terminal_request_id_cannot_be_reused_later_in_the_episode():
    cfg = _cfg()
    cfg["capture_release_snapshots_online"] = False
    cfg["require_release_snapshot_on_release"] = False
    cfg["closed_loop_latency_replay"].update(
        {"extra_latency_s": 0.0, "delay_steps": 0}
    )
    episode = create_episode_runtime_state()
    source_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": False,
        "closed_loop_latency_request_id": "reused-timeout",
        "closed_loop_latency_response_outcome": "timeout",
        "closed_loop_scripted_latency_steps": 0,
        "query_state_fast_proposal_action": 1,
    }
    _apply_closed_loop_latency_replay(
        frame=0,
        action=1,
        decision_meta=source_meta,
        episode_state=episode,
        cfg=cfg,
    )
    terminal_meta = {"system_used": "fast"}
    _apply_closed_loop_latency_replay(
        frame=1,
        action=2,
        decision_meta=terminal_meta,
        episode_state=episode,
        cfg=cfg,
    )
    assert terminal_meta["closed_loop_latency_terminal_request_id"] == "reused-timeout"
    assert episode["latency_replay_queue"] == []

    duplicate_meta = dict(source_meta)
    with pytest.raises(RuntimeError, match="duplicate episode latency request ID"):
        _apply_closed_loop_latency_replay(
            frame=2,
            action=1,
            decision_meta=duplicate_meta,
            episode_state=episode,
            cfg=cfg,
        )


def test_selected_request_missing_id_never_borrows_same_frame_issuance_id():
    cfg = _cfg()
    episode = create_episode_runtime_state()
    episode["latency_replay_queue"].append(
        {
            "request_id": "",
            "release_frame": 1,
            "available_frame": 1,
            "source_frame": 0,
            "response_outcome": "timeout",
        }
    )
    queue_before = copy.deepcopy(episode["latency_replay_queue"])
    new_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "closed_loop_latency_request_id": "new-valid",
        "closed_loop_latency_response_outcome": "valid",
        "query_state_fast_proposal_action": 1,
    }

    with pytest.raises(
        RuntimeError,
        match="selected queued latency request is missing its request ID",
    ):
        _apply_closed_loop_latency_replay(
            frame=1,
            action=4,
            decision_meta=new_meta,
            episode_state=episode,
            cfg=cfg,
        )

    assert episode["latency_replay_queue"] == queue_before
    assert "new-valid" not in episode["_latency_request_ids"]


def test_zero_delay_request_lifecycle_is_independent_of_unrelated_pending_queue():
    cfg = _cfg()
    cfg["capture_release_snapshots_online"] = False
    cfg["require_release_snapshot_on_release"] = False
    episode = create_episode_runtime_state()
    future_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": False,
        "closed_loop_latency_request_id": "future-timeout",
        "closed_loop_latency_response_outcome": "timeout",
        "closed_loop_scripted_latency_steps": 5,
        "query_state_fast_proposal_action": 1,
    }
    _apply_closed_loop_latency_replay(
        frame=0,
        action=1,
        decision_meta=future_meta,
        episode_state=episode,
        cfg=cfg,
    )
    queue_before = copy.deepcopy(episode["latency_replay_queue"])

    implicit_meta = {
        "system_used": "slow",
        "slow_request_attempted": True,
        "slow_request_valid_return": True,
        "closed_loop_latency_request_id": "zero-implicit",
        "closed_loop_scripted_latency_steps": 0,
        "query_state_fast_proposal_action": 1,
    }
    implicit_action = _apply_closed_loop_latency_replay(
        frame=1,
        action=4,
        decision_meta=implicit_meta,
        episode_state=episode,
        cfg=cfg,
    )
    assert implicit_action == 4
    assert implicit_meta["closed_loop_latency_issuance_event"] is False
    assert implicit_meta["closed_loop_latency_terminal_event"] is False
    assert episode["latency_replay_queue"] == queue_before

    explicit_meta = {
        **implicit_meta,
        "closed_loop_latency_request_id": "zero-explicit",
        "closed_loop_latency_response_outcome": "valid",
    }
    explicit_action = _apply_closed_loop_latency_replay(
        frame=2,
        action=4,
        decision_meta=explicit_meta,
        episode_state=episode,
        cfg=cfg,
    )
    assert explicit_action == 1
    assert explicit_meta["closed_loop_latency_issuance_event"] is True
    assert explicit_meta["closed_loop_latency_terminal_event"] is False
    assert explicit_meta["closed_loop_latency_terminal_outcome"] == "pending"
    assert [
        item["request_id"] for item in episode["latency_replay_queue"]
    ] == ["future-timeout", "zero-explicit"]

    release_meta = {"system_used": "fast"}
    released = _apply_closed_loop_latency_replay(
        frame=3,
        action=2,
        decision_meta=release_meta,
        episode_state=episode,
        cfg=cfg,
    )
    assert released == 4
    assert release_meta["closed_loop_latency_terminal_request_id"] == "zero-explicit"
    assert release_meta["closed_loop_latency_release_event"] is True
    assert [
        item["request_id"] for item in episode["latency_replay_queue"]
    ] == ["future-timeout"]


def test_out_of_order_queue_selects_earliest_due_request_by_id():
    episode = {
        "latency_replay_queue": [
            {"request_id": "late", "release_frame": 5, "source_frame": 0},
            {"request_id": "first", "release_frame": 3, "source_frame": 2},
            {"request_id": "second", "release_frame": 3, "source_frame": 4},
        ]
    }
    selected = _selected_ready_latency_request(episode, frame=5)
    assert selected["request_id"] == "first"


def test_snapshot_identity_detects_request_metadata_tampering():
    snapshot = capture_release_snapshot(
        _FakeAgent(),
        frame=4,
        env=SimpleNamespace(),
        obs=[0.0],
        history=collections.deque(maxlen=6),
        previous_action=1,
        pending_request={
            "request_id": "request-a",
            "source_frame": 1,
            "release_frame": 4,
        },
    )
    snapshot.request_id = "request-b"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_release_snapshot_policy_state(snapshot, context="tampered")
