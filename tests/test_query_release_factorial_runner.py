import csv
import json
import hashlib
import logging
import sys
from io import StringIO
from pathlib import Path

import pytest

from dilu.evaluation.factorial_replay import FactorialArm, ProposalRecord
from tools.run_query_release_factorial import (
    DEFAULT_PROPOSAL_SOURCE_POLICY,
    DISTINCT_ACTION_METRIC_STAGE,
    _build_empirical_latency_profile,
    _empirical_assignment,
    _factorial_group_config,
    _latency_seconds_to_policy_steps,
    load_proposal_bank,
    _named_latency_assignment,
    _primitive_selection_metric_fields,
    _proposal_manifest,
    _query_event,
    _release_execution_is_distinct,
    _stress_assignment,
    _save_factorial_episode,
    _validate_event_lifecycle_contract,
    _validate_formal_factorial_preflight,
    _validate_formal_proposal_source_bundle,
    _validate_outcome_metrics,
    _validate_proposal_source,
    _validate_query_gate_accounting,
    _validate_request_outcome_accounting,
    _worker_output_context,
    parse_args,
)
from tools.run_main_table_runtime import load_formal_base_config, load_formal_protocol


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


def test_formal_five_arm_defaults_and_seed_contract(tmp_path):
    args = parse_args(["--result-root", str(tmp_path)])
    assert args.design == "five_arm"
    assert args.seed_start == 5000
    assert args.seeds == 30
    assert args.latency_profile == "frozen"

    protocol = load_formal_protocol(Path("formal_protocol.yaml"))
    base_cfg = load_formal_base_config(protocol)
    horizon = _validate_formal_factorial_preflight(
        protocol=protocol,
        base_cfg=base_cfg,
        design=args.design,
        seeds=list(range(5000, 5030)),
        latency_profile=args.latency_profile,
        fixed_latency_steps=args.fixed_delay_steps,
        result_root=tmp_path,
    )
    assert horizon.expected_policy_steps == 300
    with pytest.raises(ValueError, match="seed cohort drift"):
        _validate_formal_factorial_preflight(
            protocol=protocol,
            base_cfg=base_cfg,
            design=args.design,
            seeds=list(range(5001, 5031)),
            latency_profile=args.latency_profile,
            fixed_latency_steps=args.fixed_delay_steps,
            result_root=tmp_path,
        )


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


def test_named_latency_profiles_are_deterministic_and_semantically_distinct():
    first = {
        "seed": 5000,
        "request_id": "factorial:5000:0:00",
        "outcome": "valid",
    }
    second = {
        "seed": 5000,
        "request_id": "factorial:5000:21:01",
        "outcome": "valid",
    }

    for profile in ("jitter", "burst", "drop", "out_of_order"):
        assert _named_latency_assignment(
            profile, first, median_steps=17
        ) == _named_latency_assignment(profile, first, median_steps=17)

    early = _named_latency_assignment("out_of_order", first, median_steps=17)
    late = _named_latency_assignment("out_of_order", second, median_steps=17)
    assert early == (27, "valid")
    assert late == (7, "valid")
    assert _named_latency_assignment("burst", first, median_steps=17)[0] in {17, 32}
    drop_steps, drop_outcome = _named_latency_assignment("drop", first, median_steps=17)
    assert drop_steps in {17, 27}
    assert drop_outcome in {"valid", "timeout"}


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
    assert replay["proposal_backed_execution_available"] is True
    assert overrides["factorial_predicted_latency_s"] == pytest.approx(0.71)
    assert overrides["factorial_predicted_latency_steps"] == 8
    assert overrides["asynchronous_slow_path"]["enable"] is False
    assert _latency_seconds_to_policy_steps(2.7) == 27


def _write_native_proposal_source(tmp_path, *, tamper_hash=False, dropped=False):
    seed_dir = tmp_path / "seed_5000"
    event_dir = seed_dir / "event_logs"
    reasoning_dir = seed_dir / "ep_5000"
    event_dir.mkdir(parents=True)
    reasoning_dir.mkdir(parents=True)
    request_id = "online:episode:0:0000"
    response = '{"action":4,"confidence":0.9,"reason":"decelerate"}'
    identity = {
        "action": 4,
        "confidence": 0.9,
        "reasoning": "decelerate",
        "response": response,
    }
    response_hash = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if tamper_hash:
        response_hash = "0" * 64
    issuance = {
        "frame": 0,
        "native_async_slow_path": True,
        "slow_request_attempted": True,
        "closed_loop_latency_source_frame": 0,
        "closed_loop_latency_issuance_event": True,
        "closed_loop_latency_issued_request_id": request_id,
        "closed_loop_latency_issued_response_outcome": "pending",
        "query_state_fast_proposal_action": 1,
    }
    terminal = {
        "frame": 16,
        "native_async_slow_path": True,
        "closed_loop_latency_terminal_event": True,
        "closed_loop_latency_terminal_request_id": request_id,
        "closed_loop_latency_terminal_response_outcome": "valid",
        "closed_loop_latency_response_outcome": "valid",
        "slow_response_action": 4,
        "slow_response_confidence": 0.9,
        "slow_response_reasoning": "decelerate",
        "slow_response_text": response,
        "closed_loop_latency_response_sha256": response_hash,
        "slow_response_wall_latency_s": 1.572,
        "query_state_slow_pre_guard_action": 4,
    }
    payload = {
        "events": [issuance] if dropped else [issuance, terminal],
        "pending_releases_dropped_at_episode_end": (
            [{"request_id": request_id}] if dropped else []
        ),
    }
    (event_dir / "event_log_highway_5000_5000.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    (reasoning_dir / "highway_5000_reasoning_records.json").write_text(
        json.dumps(
            {
                "analysis_records": [
                    {"frame_id": 0, "predicted_action_id": 1},
                    {"frame_id": 16, "predicted_action_id": 4},
                ]
            }
        ),
        encoding="utf-8",
    )
    (seed_dir / "experiment_snapshot.json").write_text(
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
    return tmp_path, response_hash


def _write_formal_proposal_source_bundle(tmp_path, *, partition):
    seeds = (
        tuple(range(5000, 5030))
        if partition == "main"
        else tuple(range(6000, 6020))
    )
    bundle_root = tmp_path / f"{partition}_bundle"
    source_root = bundle_root / "always_slow" / "highway"
    backend = {
        "provider": "siliconflow",
        "requested_model": "Qwen/Qwen3-8B",
        "resolved_chat_model": "Qwen/Qwen3-8B",
    }
    rows = []
    for seed in seeds:
        seed_dir = source_root / f"seed_{seed}"
        event_dir = seed_dir / "event_logs"
        reasoning_dir = seed_dir / "ep_0"
        event_dir.mkdir(parents=True)
        reasoning_dir.mkdir(parents=True)
        (event_dir / f"event_log_highway_{seed}.json").write_text(
            json.dumps({"events": []}), encoding="utf-8"
        )
        reasoning_path = reasoning_dir / f"highway_{seed}_reasoning_records.json"
        reasoning_path.write_text(
            json.dumps({"analysis_records": []}), encoding="utf-8"
        )
        identities = {
            "protocol_id": f"always_slow::{seed}",
            "protocol_hash": hashlib.sha256(
                f"protocol:{seed}".encode("ascii")
            ).hexdigest(),
            "config_hash": hashlib.sha256(
                f"config:{seed}".encode("ascii")
            ).hexdigest(),
            "source_hash": hashlib.sha256(b"formal-source").hexdigest(),
        }
        config = {
            "protocol_name": "always_slow",
            "protocol_version": 13,
            "env_type": "highway-v0",
            "episodes_num": 1,
            "simulation_duration": 30,
            "policy_frequency": 10,
            "simulation_frequency": 10,
            "fixed_seed_override": seed,
            "system_routing": {"simple": "slow", "complex": "slow"},
        }
        common = {
            **identities,
            "fixed_seed_override": seed,
            "seed_start": seed,
            "resolved_seeds": [seed],
            "protocol_name": "always_slow",
            "env_type": "highway-v0",
            "config": config,
            "protocol_manifest": {
                "protocol_name": "always_slow",
                "protocol_version": 13,
                "selected_group": "always_slow",
                "selected_environment": "highway-v0",
            },
            "llm_backend": backend,
        }
        (seed_dir / "runtime_manifest.json").write_text(
            json.dumps(common), encoding="utf-8"
        )
        (seed_dir / "experiment_snapshot.json").write_text(
            json.dumps({**common, "seeds_used": [seed]}), encoding="utf-8"
        )
        rows.append(
            {
                "group": "always_slow",
                "env": "highway-v0",
                "seed_idx": seed,
                "fixed_seed_override": seed,
                "seed_start": seed,
                "requested_seed_start": seeds[0],
                "episodes_run": 1,
                "result_dir": str(seed_dir.resolve()),
                **identities,
            }
        )

    run_rows_path = bundle_root / "always_slow" / "always_slow_run_rows.csv"
    with run_rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    horizon = {
        "episode_duration_s": 30,
        "policy_frequency_hz": 10,
        "simulation_frequency_hz": 10,
        "expected_policy_steps": 300,
    }
    manifest_path = bundle_root / "result_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_kind": "formal_run",
                "partition": partition,
                "groups": ["always_slow"],
                "envs": ["highway-v0"],
                "group_env_matrix": {"always_slow": ["highway-v0"]},
                "seeds": len(seeds),
                "episodes": 1,
                "seed_start": seeds[0],
                "seed_labels": list(seeds),
                "seed_value": None,
                "simulation_duration": 30,
                **horizon,
                "execution_horizon_by_group_env": {
                    "always_slow": {"highway-v0": horizon}
                },
                "method_version": "action_aligned_release_gate_v13",
                "query_gate_method_version": "identifiable_gate_v12",
                "release_contract_version": "action_cost_alignment_v2",
            }
        ),
        encoding="utf-8",
    )
    return {
        "source_root": source_root,
        "bundle_root": bundle_root,
        "manifest_path": manifest_path,
        "run_rows_path": run_rows_path,
        "seeds": seeds,
    }


def _formal_proposal_bank(source_root, seeds):
    response_hash = hashlib.sha256(b"formal-response").hexdigest()
    return {
        seed: {
            0: ProposalRecord(
                seed=seed,
                source_frame=0,
                request_id=f"factorial:{seed}:0:00",
                raw_slow_action=1,
                latency_steps=17,
                outcome="valid",
                response_text="formal-response",
                response_sha256=response_hash,
                source_artifact=str(
                    source_root
                    / f"seed_{seed}"
                    / "ep_0"
                    / f"highway_{seed}_reasoning_records.json"
                ),
            )
        }
        for seed in seeds
    }


def _rewrite_json(path, mutate):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(
    ("partition", "latency_profile", "fixed_steps"),
    [("main", "frozen", None), ("mechanism", "fixed", 17)],
)
def test_proposal_manifest_closes_over_formal_source_bundle(
    tmp_path, partition, latency_profile, fixed_steps
):
    source = _write_formal_proposal_source_bundle(tmp_path, partition=partition)
    bank = _formal_proposal_bank(source["source_root"], source["seeds"])

    manifest = _proposal_manifest(
        bank,
        source_root=source["source_root"],
        latency_profile=latency_profile,
        fixed_latency_steps=fixed_steps,
    )

    formal = manifest["formal_source_bundle"]
    assert formal["partition"] == partition
    assert formal["method_version"] == "action_aligned_release_gate_v13"
    assert formal["expected_policy_steps"] == 300
    assert formal["result_bundle_manifest"]["sha256"]
    assert formal["run_rows"]["sha256"]
    assert all("runtime_manifest" in row for row in manifest["source_artifacts"])


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_manifest", "no enclosing result bundle manifest"),
        ("wrong_version", "method_version mismatch"),
        ("wrong_partition", "partition/cohort mismatch"),
        ("wrong_horizon", "expected_policy_steps mismatch"),
        ("missing_group", "missing always_slow"),
        ("wrong_backend", "LLM provider mismatch"),
        ("identity_drift", "config_hash identity mismatch"),
    ],
)
def test_formal_proposal_source_rejects_provenance_drift(
    tmp_path, corruption, message
):
    source = _write_formal_proposal_source_bundle(tmp_path, partition="main")
    if corruption == "missing_manifest":
        source["manifest_path"].unlink()
    elif corruption == "wrong_version":
        _rewrite_json(
            source["manifest_path"],
            lambda payload: payload.__setitem__("method_version", "wrong"),
        )
    elif corruption == "wrong_partition":
        _rewrite_json(
            source["manifest_path"],
            lambda payload: payload.__setitem__("partition", "mechanism"),
        )
    elif corruption == "wrong_horizon":
        _rewrite_json(
            source["manifest_path"],
            lambda payload: payload.__setitem__("expected_policy_steps", 299),
        )
    elif corruption == "missing_group":
        _rewrite_json(
            source["manifest_path"],
            lambda payload: payload.__setitem__("groups", []),
        )
    elif corruption == "wrong_backend":
        for name in ("runtime_manifest.json", "experiment_snapshot.json"):
            _rewrite_json(
                source["source_root"] / "seed_5000" / name,
                lambda payload: payload["llm_backend"].__setitem__(
                    "provider", "openai"
                ),
            )
    elif corruption == "identity_drift":
        _rewrite_json(
            source["source_root"] / "seed_5000" / "runtime_manifest.json",
            lambda payload: payload.__setitem__("config_hash", "f" * 64),
        )

    with pytest.raises(RuntimeError, match=message):
        _validate_formal_proposal_source_bundle(
            source["source_root"], source["seeds"]
        )


def test_formal_proposal_source_rejects_partial_seed_cohort(tmp_path):
    source = _write_formal_proposal_source_bundle(tmp_path, partition="main")

    with pytest.raises(RuntimeError, match="seed cohort must be exactly"):
        _validate_formal_proposal_source_bundle(
            source["source_root"], source["seeds"][:-1]
        )


def _write_scheduled_proposal_source(tmp_path, frames):
    seed_dir = tmp_path / "seed_5000"
    event_dir = seed_dir / "event_logs"
    reasoning_dir = seed_dir / "ep_0"
    event_dir.mkdir(parents=True)
    reasoning_dir.mkdir(parents=True)
    events = [
        {
            "frame": frame,
            "closed_loop_latency_source_frame": frame,
            "slow_request_attempted": True,
            "slow_response_action": 1,
            "slow_response_wall_latency_s": 1.0,
            "slow_response_text": "response",
            "closed_loop_latency_response_outcome": "valid",
        }
        for frame in frames
    ]
    (event_dir / "event_log_highway_5000.json").write_text(
        json.dumps({"events": events}), encoding="utf-8"
    )
    (reasoning_dir / "highway_5000_reasoning_records.json").write_text(
        json.dumps({"analysis_records": []}), encoding="utf-8"
    )
    (seed_dir / "experiment_snapshot.json").write_text(
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
    return tmp_path


def test_proposal_bank_rejects_more_than_six_source_requests(tmp_path):
    source = _write_scheduled_proposal_source(tmp_path, range(0, 147, 21))

    with pytest.raises(RuntimeError, match="formal budget of 6"):
        load_proposal_bank(source, [5000], latency_profile="frozen")


def test_proposal_bank_rejects_source_frames_less_than_21_apart(tmp_path):
    source = _write_scheduled_proposal_source(tmp_path, [0, 20])

    with pytest.raises(RuntimeError, match="source-frame gap is below 21"):
        load_proposal_bank(source, [5000], latency_profile="frozen")


def test_factorial_event_json_replaces_nonfinite_values_with_null(tmp_path):
    _save_factorial_episode(
        root=tmp_path,
        seed=7,
        prefix="highway_7",
        events=[
            {
                "frame": 0,
                "terminal_cause": "truncated",
                "latency": float("nan"),
                "nested": [float("inf"), {"value": float("-inf")}],
            }
        ],
        pending=[],
        snapshots={},
        physical_recorder=None,
        reasoning_recorder=None,
    )
    event_path = tmp_path / "event_logs" / "event_log_highway_7_7.json"
    raw = event_path.read_text(encoding="utf-8")

    def reject_constant(value):
        raise AssertionError(f"non-standard JSON constant persisted: {value}")

    payload = json.loads(raw, parse_constant=reject_constant)
    assert payload["events"][0]["latency"] is None
    assert payload["events"][0]["nested"] == [None, {"value": None}]


def test_native_proposal_bank_joins_issuance_to_terminal_by_request_id(tmp_path):
    source, response_hash = _write_native_proposal_source(tmp_path)

    bank = load_proposal_bank(source, [5000], latency_profile="frozen")

    proposal = bank[5000][0]
    assert proposal.raw_slow_action == 4
    assert proposal.latency_steps == 16
    assert proposal.outcome == "valid"
    assert proposal.response_sha256 == response_hash


def test_native_proposal_bank_rejects_tampered_response_identity(tmp_path):
    source, _ = _write_native_proposal_source(tmp_path, tamper_hash=True)
    with pytest.raises(ValueError, match="does not match"):
        load_proposal_bank(source, [5000], latency_profile="frozen")


def test_native_proposal_bank_rejects_dropped_request(tmp_path):
    source, _ = _write_native_proposal_source(tmp_path, dropped=True)
    with pytest.raises(RuntimeError, match="dropped requests"):
        load_proposal_bank(source, [5000], latency_profile="frozen")


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


def test_fast_only_arm_suppresses_every_frozen_candidate():
    arm = FactorialArm("fast_only", False, False)
    candidates = [
        _lifecycle_event(
            factorial_candidate_query=True,
            factorial_query_issued=False,
            factorial_query_rejection_reason="fast_only_control",
        )
    ]

    _validate_query_gate_accounting(
        arm,
        candidate_events=candidates,
        candidate_count=1,
        issued_count=0,
        gate_rejected_count=1,
        context="fast-only",
    )
    _validate_event_lifecycle_contract(candidates, context="fast-only")
    with pytest.raises(RuntimeError, match="provenance drift"):
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
            context="fast-only",
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
