import csv
import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from dilu.driver_agent.policy_state import (
    DRIVER_POLICY_STATE_SCHEMA,
    FAST_POLICY_STATE_SCHEMA,
    RAD_POLICY_STATE_SCHEMA,
    RGD_POLICY_STATE_SCHEMA,
)
from dilu.evaluation.factorial_replay import (
    FACTORIAL_ARMS,
    FACTORIAL_EVENT_SCHEMA,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
    FACTORIAL_RUN_SCHEMA,
)
from dilu.evaluation.release_snapshot import (
    RELEASE_SNAPSHOT_CAPTURE_STAGE,
    RELEASE_SNAPSHOT_SCHEMA,
    capture_release_snapshot,
    save_release_snapshot_bundle,
)
from tools.audit_query_release_factorial import (
    AuditError,
    _audit_candidate_coverage,
    _audit_shared_candidate_identities,
    audit_bundle,
    main,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _policy_state():
    return {
        "schema": DRIVER_POLICY_STATE_SCHEMA,
        "fast": {
            "schema": FAST_POLICY_STATE_SCHEMA,
            "action_history": [1],
            "action_history_capacity": 12,
        },
        "orchestrator": {
            "schema": RGD_POLICY_STATE_SCHEMA,
            "decision_count": 1,
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


class _Fast:
    def snapshot_runtime_state(self):
        return {"action_history": [1]}


class _Agent:
    fast_thinker = _Fast()

    def snapshot_policy_state(self):
        return _policy_state()


def _request_event(
    *,
    frame,
    arm,
    bank_hash,
    request_id,
    latency_steps,
    outcome,
    candidate=False,
    issued=False,
    terminal=None,
    snapshot=None,
):
    terminal_kind = terminal or ""
    is_terminal = bool(terminal_kind)
    realized_steps = frame if is_terminal else -1
    event = {
        "frame": frame,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "factorial_arm": arm.name,
        "factorial_query_gate_enabled": arm.query_gate_enabled,
        "factorial_release_guard_enabled": arm.release_guard_enabled,
        "factorial_proposal_bank_sha256": bank_hash,
        "factorial_candidate_query": candidate,
        "factorial_query_issued": issued,
        "factorial_policy_state_synchronized": issued,
        "factorial_candidate_request_id": request_id if candidate else "",
        "factorial_query_gate_pass": True if candidate else False,
        "factorial_shared_raw_slow_action": 4 if candidate else -1,
        "factorial_shared_latency_steps": latency_steps if candidate else -1,
        "factorial_shared_response_sha256": (
            hashlib.sha256(b"slow-response").hexdigest() if candidate else ""
        ),
        "factorial_shared_response_outcome": outcome if candidate else "",
        "closed_loop_latency_issuance_event": issued,
        "closed_loop_latency_issued_request_id": request_id if issued else "",
        "closed_loop_latency_issued_response_outcome": outcome if issued else "",
        "closed_loop_latency_terminal_event": is_terminal,
        "closed_loop_latency_terminal_request_id": request_id if is_terminal else "",
        "closed_loop_latency_terminal_response_outcome": outcome if is_terminal else "",
        "closed_loop_latency_request_id": request_id,
        "closed_loop_latency_response_outcome": outcome,
        "closed_loop_latency_terminal_outcome": (
            terminal_kind if terminal_kind in {"timeout", "failure"} else
            "fast_equivalent" if terminal_kind == "release" else "pending"
        ),
        "closed_loop_latency_release_event": terminal_kind == "release",
        "closed_loop_latency_timeout_event": terminal_kind == "timeout",
        "closed_loop_latency_failure_event": terminal_kind == "failure",
        "closed_loop_latency_delay_steps": latency_steps,
        "closed_loop_latency_scheduled_seconds": latency_steps / 10.0,
        "closed_loop_latency_scheduled_steps": latency_steps,
        "closed_loop_latency_realized_seconds": realized_steps / 10.0 if is_terminal else float("nan"),
        "closed_loop_latency_realized_steps": realized_steps,
        "closed_loop_latency_realized_available": is_terminal,
        "closed_loop_latency_realized_source": "simulator_frame_delta" if is_terminal else "not_released",
        "closed_loop_latency_policy_frequency_hz": 10.0,
        "closed_loop_latency_scheduled_release_frame": latency_steps,
        "closed_loop_latency_source_frame": 0,
        "closed_loop_latency_source_system": "slow",
        "closed_loop_release_snapshot_captured": terminal_kind == "release",
        "closed_loop_release_snapshot_schema": RELEASE_SNAPSHOT_SCHEMA if snapshot else "",
        "closed_loop_release_snapshot_capture_stage": RELEASE_SNAPSHOT_CAPTURE_STAGE if snapshot else "",
        "closed_loop_release_snapshot_identity_sha256": (
            snapshot.snapshot_identity_sha256 if snapshot else ""
        ),
    }
    return event


def _build_bundle(tmp_path, *, outcome="valid", pending=False):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source_root = tmp_path / "source"
    source_seed = source_root / "seed_10"
    (source_seed / "event_logs").mkdir(parents=True)
    (source_seed / "ep_10").mkdir()
    source_event = source_seed / "event_logs" / "event_log_highway_10_10.json"
    source_reasoning = source_seed / "ep_10" / "highway_10_reasoning_records.json"
    source_snapshot = source_seed / "experiment_snapshot.json"
    source_event.write_text('{"events": []}\n', encoding="utf-8")
    source_reasoning.write_text('{"analysis_records": []}\n', encoding="utf-8")
    source_snapshot.write_text(
        json.dumps(
            {
                "fixed_seed_override": 10,
                "config": {
                    "protocol_name": "always_slow",
                    "system_routing": {"simple": "slow", "complex": "slow"},
                },
            }
        ),
        encoding="utf-8",
    )

    request_id = "factorial:10:0:00"
    latency_steps = 5 if pending else 2
    response_sha = hashlib.sha256(b"slow-response").hexdigest()
    bank_payload = [
        {
            "seed": 10,
            "records": [
                {
                    "seed": 10,
                    "source_frame": 0,
                    "request_id": request_id,
                    "raw_slow_action": 4,
                    "latency_steps": latency_steps,
                    "outcome": outcome,
                    "response_text": "slow-response",
                    "response_sha256": response_sha,
                }
            ],
        }
    ]
    bank_hash = _json_sha256(bank_payload)
    proposal_manifest = {
        "schema": FACTORIAL_PROPOSAL_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "source_root": str(source_root.resolve()),
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "latency_profile": "fixture",
        "bank_sha256": bank_hash,
        "seed_count": 1,
        "proposal_count": 1,
        "source_artifacts": [
            {
                "seed": 10,
                "event_log": {
                    "path": str(source_event.relative_to(source_root)),
                    "sha256": _sha256(source_event),
                },
                "reasoning_trace": {
                    "path": str(source_reasoning.relative_to(source_root)),
                    "sha256": _sha256(source_reasoning),
                },
                "experiment_snapshot": {
                    "path": str(source_snapshot.relative_to(source_root)),
                    "sha256": _sha256(source_snapshot),
                },
            }
        ],
        "bank_payload": bank_payload,
    }
    (bundle / "proposal_bank_manifest.json").write_text(
        json.dumps(proposal_manifest), encoding="utf-8"
    )
    run_manifest = {
        "schema": FACTORIAL_RUN_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "latency_profile": "fixture",
        "proposal_bank_sha256": bank_hash,
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "seed_start": 10,
        "seed_count": 1,
        "result_rows": 4,
        "arms": [asdict(arm) for arm in FACTORIAL_ARMS],
        "randomized_block_run_order": [
            {"seed": 10, "order": order, "arm": arm.name}
            for order, arm in enumerate(reversed(FACTORIAL_ARMS))
        ],
    }
    (bundle / "factorial_run_manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )

    result_rows = []
    for arm in FACTORIAL_ARMS:
        seed_dir = bundle / arm.name / "seed_10"
        event_dir = seed_dir / "event_logs"
        event_dir.mkdir(parents=True)
        release_count = int(outcome == "valid" and not pending)
        timeout_count = int(outcome == "timeout" and not pending)
        failure_count = int(outcome == "failure" and not pending)
        pending_count = int(pending)
        snapshot = None
        bundle_rel = manifest_rel = bundle_digest = None
        if release_count:
            snapshot = capture_release_snapshot(
                _Agent(),
                frame=latency_steps,
                env=SimpleNamespace(name="fixture-env"),
                obs=[1.0],
                history=[],
                previous_action=1,
                pending_request={
                    "request_id": request_id,
                    "source_frame": 0,
                    "release_frame": latency_steps,
                },
            )
            bundle_path, manifest_path, bundle_digest = save_release_snapshot_bundle(
                {request_id: snapshot},
                seed_dir / "release_snapshots",
                prefix="highway_10",
                episode_id=10,
            )
            bundle_rel = str(bundle_path.relative_to(seed_dir))
            manifest_rel = str(manifest_path.relative_to(seed_dir))

        frame_count = 2 if pending else latency_steps + 1
        events = []
        for frame in range(frame_count):
            terminal = None
            if not pending and frame == latency_steps:
                terminal = "release" if outcome == "valid" else outcome
            events.append(
                _request_event(
                    frame=frame,
                    arm=arm,
                    bank_hash=bank_hash,
                    request_id=request_id,
                    latency_steps=latency_steps,
                    outcome=outcome,
                    candidate=frame == 0,
                    issued=frame == 0,
                    terminal=terminal,
                    snapshot=snapshot if terminal == "release" else None,
                )
            )
        pending_rows = []
        if pending:
            pending_rows.append(
                {
                    "request_id": request_id,
                    "source_frame": 0,
                    "release_frame": latency_steps,
                    "response_outcome": outcome,
                    "terminal_outcome": "dropped_at_episode_end",
                }
            )
        event_payload = {
            "schema_version": FACTORIAL_EVENT_SCHEMA,
            "episode_id": 10,
            "event_count": len(events),
            "pending_release_count": pending_count,
            "pending_releases_dropped_at_episode_end": pending_rows,
            "release_snapshot_count": release_count,
            "release_snapshot_bundle": bundle_rel,
            "release_snapshot_manifest": manifest_rel,
            "release_snapshot_bundle_sha256": bundle_digest,
            "terminal_cause": "truncated",
            "events": events,
        }
        (event_dir / "event_log_highway_10_10.json").write_text(
            json.dumps(event_payload, allow_nan=True), encoding="utf-8"
        )
        result_rows.append(
            {
                "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
                "arm": arm.name,
                "query_gate_enabled": arm.query_gate_enabled,
                "release_guard_enabled": arm.release_guard_enabled,
                "seed": 10,
                "candidate_queries": 1,
                "issued_queries": 1,
                "query_gate_rejections": 0,
                "timeouts": timeout_count,
                "scheduled_timeouts": int(outcome == "timeout"),
                "failure_events": failure_count,
                "release_events": release_count,
                "pending_at_episode_end": pending_count,
                "pending_timeouts_at_episode_end": int(pending and outcome == "timeout"),
                "snapshot_count": release_count,
                "proposal_bank_sha256": bank_hash,
                "candidate_source_policy": "scheduled_always_slow",
                "candidate_source_gate_independent": True,
            }
        )
    with (bundle / "factorial_episode_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    return bundle


def _event_path(bundle, arm="full"):
    return bundle / arm / "seed_10" / "event_logs" / "event_log_highway_10_10.json"


def test_full_request_and_snapshot_audit_accepts_authenticated_bundle(tmp_path):
    bundle = _build_bundle(tmp_path)

    report = audit_bundle(bundle)

    assert report["accepted"] is True
    assert report["aggregate"]["arm_seed_cells"] == 4
    assert report["aggregate"]["candidate_queries"] == 4
    assert report["aggregate"]["issued_queries"] == 4
    assert report["aggregate"]["release_events"] == 4
    assert report["aggregate"]["snapshot_count"] == 4
    assert report["aggregate"]["cross_arm_candidate_identity_comparisons"] == 3
    assert {cell["lifecycle_mode"] for cell in report["cells"]} == {
        "explicit_dual_event_ids"
    }


def test_post_terminal_proposals_are_right_censored_but_reached_ones_are_required():
    proposal_records = {
        "factorial:10:0:00": {"source_frame": 0},
        "factorial:10:3:01": {"source_frame": 3},
    }
    candidates = {"factorial:10:0:00": {"source_frame": 0}}
    events = [
        {"frame": 0, "episode_done": False},
        {"frame": 1, "episode_done": False},
        {"frame": 2, "episode_done": True},
    ]

    final_frame, reachable, censored = _audit_candidate_coverage(
        candidates=candidates,
        proposal_records=proposal_records,
        event_payload={"terminal_cause": "collision"},
        events=events,
        cell="fixture",
    )

    assert final_frame == 2
    assert reachable == {"factorial:10:0:00"}
    assert censored == {"factorial:10:3:01"}

    with pytest.raises(AuditError, match="candidate/reachable-proposal coverage mismatch"):
        _audit_candidate_coverage(
            candidates={},
            proposal_records=proposal_records,
            event_payload={"terminal_cause": "collision"},
            events=events,
            cell="fixture",
        )


def test_cross_arm_identity_check_uses_common_reachable_proposals():
    shared = {"request_id": "factorial:10:0:00", "source_frame": 0}
    later = {"request_id": "factorial:10:3:01", "source_frame": 3}
    cells = [
        {"seed": 10, "arm": "full", "candidate_identities": {shared["request_id"]: shared, later["request_id"]: later}},
        {"seed": 10, "arm": "release_only", "candidate_identities": {shared["request_id"]: shared}},
        {"seed": 10, "arm": "neither", "candidate_identities": {shared["request_id"]: shared, later["request_id"]: later}},
    ]

    assert _audit_shared_candidate_identities(cells) == 2

    cells[2]["candidate_identities"][shared["request_id"]] = {**shared, "source_frame": 1}
    with pytest.raises(AuditError, match="cross-arm shared proposal identity drift"):
        _audit_shared_candidate_identities(cells)


def test_candidate_proposal_identity_tampering_fails_closed(tmp_path):
    bundle = _build_bundle(tmp_path)
    path = _event_path(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["factorial_shared_response_sha256"] = "0" * 64
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(AuditError, match="shared proposal response hash drift"):
        audit_bundle(bundle)


def test_empty_seed_proposal_block_fails_closed(tmp_path):
    bundle = _build_bundle(tmp_path)
    proposal_path = bundle / "proposal_bank_manifest.json"
    run_path = bundle / "factorial_run_manifest.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    proposal["bank_payload"][0]["records"] = []
    proposal["proposal_count"] = 0
    bank_hash = _json_sha256(proposal["bank_payload"])
    proposal["bank_sha256"] = bank_hash
    run["proposal_bank_sha256"] = bank_hash
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(AuditError, match="proposal bank has no candidates"):
        audit_bundle(bundle)


def test_explicit_terminal_marker_drift_is_rejected(tmp_path):
    bundle = _build_bundle(tmp_path)
    path = _event_path(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["closed_loop_latency_release_event"] = True
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(AuditError, match="explicit terminal flag drift"):
        audit_bundle(bundle)


def test_snapshot_pickle_file_hash_tampering_fails_closed(tmp_path):
    bundle = _build_bundle(tmp_path)
    event_payload = json.loads(_event_path(bundle).read_text(encoding="utf-8"))
    bundle_path = bundle / "full" / "seed_10" / event_payload["release_snapshot_bundle"]
    bundle_path.write_bytes(bundle_path.read_bytes() + b"tampered")

    with pytest.raises(AuditError, match="bundle file SHA256 mismatch"):
        audit_bundle(bundle)


def test_timeout_lifecycle_is_accepted_but_snapshot_claim_is_rejected(tmp_path):
    bundle = _build_bundle(tmp_path, outcome="timeout")
    report = audit_bundle(bundle)
    assert report["aggregate"]["timeouts"] == 4
    assert report["aggregate"]["snapshot_count"] == 0

    path = _event_path(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][-1]["closed_loop_release_snapshot_captured"] = True
    payload["events"][-1]["closed_loop_release_snapshot_identity_sha256"] = "a" * 64
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(AuditError, match="release iff snapshot-captured"):
        audit_bundle(bundle)


def test_pending_request_closes_accounting_and_orphan_pending_fails(tmp_path):
    bundle = _build_bundle(tmp_path, pending=True)
    report = audit_bundle(bundle)
    assert report["aggregate"]["pending_at_episode_end"] == 4
    assert report["aggregate"]["release_events"] == 0

    path = _event_path(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pending_releases_dropped_at_episode_end"][0]["request_id"] = "orphan"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    with pytest.raises(AuditError, match="orphan pending request ID"):
        audit_bundle(bundle)


def test_legacy_v3_pending_rows_use_the_enclosing_episode_end_marker(tmp_path):
    bundle = _build_bundle(tmp_path, pending=True)
    path = _event_path(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["pending_releases_dropped_at_episode_end"][0].pop(
        "terminal_outcome"
    )
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    report = audit_bundle(bundle)

    assert report["accepted"] is True


def test_cli_writes_machine_readable_failure_report(tmp_path):
    bundle = _build_bundle(tmp_path)
    path = _event_path(bundle)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "rgd_event_log_v1"
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    output = tmp_path / "audit.json"

    status = main(["--bundle", str(bundle), "--output", str(output)])

    assert status == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["accepted"] is False
    assert "unsupported/legacy event schema" in report["errors"][0]
