import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

import tools.run_v12_branch_labels as branch_labels
import tools.analyze_release_state_rollouts as release_rollouts
import tools.run_mechanism_inprocess as mechanism_runner
from dilu.driver_agent.policy_state import (
    DRIVER_POLICY_STATE_SCHEMA,
    FAST_POLICY_STATE_SCHEMA,
    RAD_POLICY_STATE_SCHEMA,
    RGD_POLICY_STATE_SCHEMA,
    policy_state_sha256,
)


def make_policy_state(*, history=(), decision_count=5):
    return {
        "schema": DRIVER_POLICY_STATE_SCHEMA,
        "fast": {
            "schema": FAST_POLICY_STATE_SCHEMA,
            "action_history": list(history),
            "action_history_capacity": 12,
        },
        "orchestrator": {
            "schema": RGD_POLICY_STATE_SCHEMA,
            "decision_count": decision_count,
            "support_progress_cooldown": 0,
            "rgd_cruise_progress_cooldown": 0,
            "rgd_cruise_recovery_frames": 0,
            "slow_call_attempts": 0,
            "slow_call_cooldown_remaining": 0,
            "rad": {
                "schema": RAD_POLICY_STATE_SCHEMA,
                "corridor_boundary_ema": None,
                "corridor_width_ema": None,
                "last_corridor_stage": None,
            },
        },
    }


def make_record(*, universe=(0, 1, 3), method=branch_labels.METHOD_VERSION):
    return {
        "schema_version": "rgd_record_v3",
        "frame_id": 5,
        "system_used": "fast",
        "predicted_action_id": 1,
        "rgd_subordinate_diagnostics": {
            "recoverability_signal": {
                "recoverability_gate": {
                    "method_version": method,
                    "gate_action_universe": list(universe),
                    "fast_executor_action_universe": list(universe),
                    "gate_action_universe_source": branch_labels.ACTION_UNIVERSE_SOURCE,
                    "fast_executor_action_universe_source": branch_labels.ACTION_UNIVERSE_SOURCE,
                    "gate_domain_valid": True,
                    "gate_fail_closed": False,
                    "hold_action": 1,
                }
            }
        },
    }


def make_snapshot(*, frame=5, previous_action=3):
    vehicle = SimpleNamespace(
        position=(12.5, 4.0),
        speed=21.0,
        lane_index=("a", "b", 2),
    )
    env = SimpleNamespace(unwrapped=SimpleNamespace(vehicle=vehicle))
    policy_state = make_policy_state(decision_count=frame)
    return SimpleNamespace(
        frame=frame,
        env=env,
        obs=None,
        fast_state={},
        history=[],
        previous_action=previous_action,
        policy_state_schema=DRIVER_POLICY_STATE_SCHEMA,
        policy_state=policy_state,
        policy_state_sha256=policy_state_sha256(policy_state),
    )


def make_physical(*, frame=5, action=1):
    return {
        "frame_id": frame,
        "position_x": 12.5,
        "position_y": 4.0,
        "speed": 21.0,
        "lane_id": 2,
        "action_id": action,
    }


def test_exact_action_contract_preserves_action_zero_and_sources():
    contract = branch_labels._exact_action_contract(make_record(), seed=7, frame=5)

    assert contract["method_version"] == branch_labels.METHOD_VERSION
    assert contract["gate_action_universe"] == (0, 1, 3)
    assert contract["fast_executor_action_universe"] == (0, 1, 3)
    assert contract["hold_action"] == 1


def test_branch_agent_forwards_post_safety_action_to_inner_controller():
    recorded = []
    agent = object.__new__(release_rollouts.FastBranchAgent)
    agent.inner = SimpleNamespace(
        record_executed_action=lambda action: recorded.append(action)
    )

    agent.record_executed_action(4)

    assert recorded == [4]


def test_trace_paths_never_cross_pair_complementary_partial_layouts(tmp_path):
    seed = 7
    legacy = (
        tmp_path
        / "always_fast"
        / f"always_fast_latency_1p7s_seed_{seed}"
        / f"ep_{seed}"
    )
    current = tmp_path / "always_fast" / "highway" / f"seed_{seed}" / f"ep_{seed}"
    legacy.mkdir(parents=True)
    current.mkdir(parents=True)
    (legacy / f"highway_{seed}_reasoning_records.json").write_text("{}", encoding="utf-8")
    (current / f"highway_{seed}_physical_frames.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="incomplete trace pair"):
        release_rollouts._trace_paths(tmp_path, seed)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(
                method_version="support_breadth_v11"
            ),
            "method version drift",
        ),
        (
            lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(
                fast_executor_action_universe=[0, 1]
            ),
            "universe mismatch",
        ),
        (
            lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(
                gate_action_universe=[0, 1, 1]
            ),
            "duplicate gate actions",
        ),
        (
            lambda record: record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"].update(
                gate_fail_closed=True
            ),
            "gate trace is fail-closed",
        ),
    ],
)
def test_exact_action_contract_rejects_legacy_or_ambiguous_domains(mutation, message):
    record = make_record()
    mutation(record)

    with pytest.raises(ValueError, match=message):
        branch_labels._exact_action_contract(record, seed=7, frame=5)


def test_snapshot_identity_matches_trace_and_previous_effective_action():
    record = make_record()
    contract = branch_labels._exact_action_contract(record, seed=7, frame=5)
    identity, payload = branch_labels._validate_snapshot_trace_identity(
        make_snapshot(),
        record,
        make_physical(),
        make_physical(frame=4, action=3),
        seed=7,
        frame=5,
        action_contract=contract,
    )

    assert len(identity) == 64
    assert payload["previous_action"] == 3
    assert payload["trace_fast_proposal"] == 1
    assert payload["trace_effective_action"] == 1
    assert payload["position_error_m"] == 0.0


def test_snapshot_identity_fails_closed_on_state_drift():
    record = make_record()
    contract = branch_labels._exact_action_contract(record, seed=7, frame=5)
    physical = make_physical()
    physical["position_x"] += 0.01

    with pytest.raises(ValueError, match="position drift"):
        branch_labels._validate_snapshot_trace_identity(
            make_snapshot(),
            record,
            physical,
            make_physical(frame=4, action=3),
            seed=7,
            frame=5,
            action_contract=contract,
        )


def branch_row(raw_action, *, effective_action, target_speed, utility):
    return {
        "seed": 7,
        "release_frame": 5,
        "release_state_id": "7:5",
        "raw_action": raw_action,
        "fast_action": 1,
        "legal_actions": "0;1;3",
        "runtime_effective_action_universe": "0;1;3",
        "runtime_gate_action_universe": "0;1;3",
        "runtime_gate_action_universe_source": branch_labels.ACTION_UNIVERSE_SOURCE,
        "runtime_fast_action_universe_source": branch_labels.ACTION_UNIVERSE_SOURCE,
        "effective_action": effective_action,
        "target_speed_after": target_speed,
        "acceleration_after": 0.0,
        "horizon_steps": 20,
        "steps_completed": 20,
        "gamma": 0.99,
        "normalized_return": utility,
        "collision": 0,
        "min_ttc": 5.0,
        "progress_m": 10.0,
        "utility": utility,
        "branch_trajectory_json": json.dumps(
            [
                {
                    "frame": 5,
                    "position_x": 12.5,
                    "position_y": 4.0,
                    "speed": 21.0,
                    "lane_id": 2,
                    "effective_action": effective_action,
                }
            ]
        ),
    }


def test_release_evaluation_runs_exact_universe_and_labels_distinct_effects():
    rows = {
        None: branch_row("fast", effective_action=1, target_speed=20.0, utility=0.0),
        0: branch_row(0, effective_action=0, target_speed=18.0, utility=0.05),
        1: branch_row(1, effective_action=1, target_speed=20.0, utility=0.0),
        3: branch_row(3, effective_action=0, target_speed=18.0, utility=0.05),
    }
    contract = branch_labels._exact_action_contract(make_record(), seed=7, frame=5)

    with patch.object(
        branch_labels,
        "_run_branch",
        side_effect=lambda snapshot, cfg, seed, action, horizon, gamma: dict(rows[action]),
    ) as run_branch:
        result = branch_labels._evaluate_release_state(
            make_snapshot(),
            {},
            seed=7,
            frame=5,
            record=make_record(),
            physical=make_physical(),
            state_identity_sha256="a" * 64,
            action_contract=contract,
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )

    assert [call.args[3] for call in run_branch.call_args_list] == [None, 0, 1, 3]
    label = result["label"]
    assert label["gate_action_universe"] == "0;1;3"
    assert label["candidate_branch_count"] == 3
    assert label["distinct_effective_candidate_count"] == 2
    assert label["distinct_effective_alternative_count"] == 1
    assert label["candidate_actions_matching_fast_identity"] == 1
    assert label["candidate_effective_aliases_collapsed"] == 1
    assert label["corrective_set_action_count"] == 1
    assert label["corrective_set_nonempty"] == 1
    candidate_rows = {
        int(row["raw_action"]): row
        for row in result["branches"]
        if row["branch_role"] == "candidate"
    }
    assert set(candidate_rows) == {0, 1, 3}
    assert candidate_rows[0]["in_corrective_set"] == 1
    assert candidate_rows[3]["in_corrective_set"] == 0
    assert all(
        row["raw_action_provenance"] == "exact_trace_gate_action_universe"
        for row in candidate_rows.values()
    )


def test_release_evaluation_rejects_outcome_drift_between_effective_aliases():
    rows = {
        None: branch_row("fast", effective_action=1, target_speed=20.0, utility=0.0),
        0: branch_row(0, effective_action=0, target_speed=18.0, utility=0.03),
        1: branch_row(1, effective_action=1, target_speed=20.0, utility=0.0),
        3: branch_row(3, effective_action=0, target_speed=18.0, utility=0.05),
    }
    contract = branch_labels._exact_action_contract(make_record(), seed=7, frame=5)
    with (
        patch.object(
            branch_labels,
            "_run_branch",
            side_effect=lambda snapshot, cfg, seed, action, horizon, gamma: dict(rows[action]),
        ),
        pytest.raises(
            ValueError,
            match="raw-action aliases.*(normalized_return|utility) drift",
        ),
    ):
        branch_labels._evaluate_release_state(
            make_snapshot(),
            {},
            seed=7,
            frame=5,
            record=make_record(),
            physical=make_physical(),
            state_identity_sha256="a" * 64,
            action_contract=contract,
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )


def test_release_evaluation_rejects_nonfinite_label_inputs():
    rows = {
        None: branch_row("fast", effective_action=1, target_speed=20.0, utility=float("nan")),
        0: branch_row(0, effective_action=0, target_speed=18.0, utility=0.05),
        1: branch_row(1, effective_action=1, target_speed=20.0, utility=0.0),
        3: branch_row(3, effective_action=0, target_speed=18.0, utility=0.05),
    }
    contract = branch_labels._exact_action_contract(make_record(), seed=7, frame=5)
    with (
        patch.object(
            branch_labels,
            "_run_branch",
            side_effect=lambda snapshot, cfg, seed, action, horizon, gamma: dict(rows[action]),
        ),
        pytest.raises(ValueError, match="nonfinite branch (normalized_return|utility)"),
    ):
        branch_labels._evaluate_release_state(
            make_snapshot(),
            {},
            seed=7,
            frame=5,
            record=make_record(),
            physical=make_physical(),
            state_identity_sha256="a" * 64,
            action_contract=contract,
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )


def test_release_evaluation_rejects_hidden_trajectory_drift_between_aliases():
    rows = {
        None: branch_row("fast", effective_action=1, target_speed=20.0, utility=0.0),
        0: branch_row(0, effective_action=0, target_speed=18.0, utility=0.05),
        1: branch_row(1, effective_action=1, target_speed=20.0, utility=0.0),
        3: branch_row(3, effective_action=0, target_speed=18.0, utility=0.05),
    }
    changed = json.loads(rows[3]["branch_trajectory_json"])
    changed[0]["speed"] += 0.1
    rows[3]["branch_trajectory_json"] = json.dumps(changed)
    contract = branch_labels._exact_action_contract(make_record(), seed=7, frame=5)
    with (
        patch.object(
            branch_labels,
            "_run_branch",
            side_effect=lambda snapshot, cfg, seed, action, horizon, gamma: dict(rows[action]),
        ),
        pytest.raises(ValueError, match="trajectory speed drift"),
    ):
        branch_labels._evaluate_release_state(
            make_snapshot(),
            {},
            seed=7,
            frame=5,
            record=make_record(),
            physical=make_physical(),
            state_identity_sha256="a" * 64,
            action_contract=contract,
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )


def test_process_seed_deduplicates_release_states_before_simulation(tmp_path):
    records = [
        {"frame_id": 0, "system_used": "fast"},
        {"frame_id": 1, "system_used": "fast"},
    ]
    physical = [{"frame_id": 0}, {"frame_id": 1}]
    evaluated = {
        "branches": [{"seed": 7, "release_frame": 1}],
        "label": {
            "seed": 7,
            "release_frame": 1,
            "gate_action_count": 2,
            "candidate_actions_matching_fast_identity": 1,
            "candidate_effective_aliases_collapsed": 0,
            "corrective_set_action_count": 1,
            "corrective_set_nonempty": 1,
        },
    }
    with (
        patch.object(
            branch_labels,
            "_source_paths",
            return_value={"dummy": tmp_path / "x", "snapshot_bundle": tmp_path / "snapshots.pkl"},
        ),
        patch.object(branch_labels, "_source_hashes", return_value={"dummy": {"sha256": "1"}}),
        patch.object(branch_labels, "_validate_source_provenance", return_value={}),
        patch.object(branch_labels, "_load_trace", return_value=(records, physical)),
        patch.object(branch_labels, "_load_snapshots", return_value={1: make_snapshot(frame=1)}),
        patch.object(branch_labels, "_build_fast_config", return_value={}),
        patch.object(branch_labels, "_validate_branch_config"),
        patch.object(branch_labels, "_exact_action_contract", return_value={}),
        patch.object(
            branch_labels,
            "_validate_snapshot_trace_identity",
            return_value=("a" * 64, {"position_error_m": 0.0, "speed_error_mps": 0.0}),
        ),
        patch.object(branch_labels, "_evaluate_release_state", return_value=evaluated) as evaluate,
    ):
        result = branch_labels._process_seed(
            7,
            [1, 1],
            str(tmp_path),
            str(tmp_path / "protocol.yaml"),
            str(tmp_path / "scratch"),
            1,
            0.99,
            0.02,
            "runtime-source-hash",
            "protocol-source-hash",
        )

    evaluate.assert_called_once()
    assert result["target_frames"] == [1]
    assert result["accounting"]["requested_target_entries"] == 2
    assert result["accounting"]["unique_release_states"] == 1
    assert result["accounting"]["duplicate_targets_excluded"] == 1


def test_process_seed_accepts_a_seed_with_no_release_states(tmp_path):
    paths = {"dummy": tmp_path / "x", "snapshot_bundle": tmp_path / "snapshots.pkl"}
    with (
        patch.object(branch_labels, "_source_paths", return_value=paths),
        patch.object(branch_labels, "_source_hashes", return_value={"dummy": {"sha256": "1"}}),
        patch.object(branch_labels, "_validate_source_provenance", return_value={}),
        patch.object(
            branch_labels,
            "_load_trace",
            return_value=([{"frame_id": 0, "system_used": "fast"}], [{"frame_id": 0}]),
        ),
        patch.object(branch_labels, "_load_snapshots", return_value={}),
        patch.object(branch_labels, "_build_fast_config", return_value={}),
        patch.object(branch_labels, "_validate_branch_config"),
        patch.object(branch_labels, "_evaluate_release_state") as evaluate,
    ):
        result = branch_labels._process_seed(
            7,
            [],
            str(tmp_path),
            str(tmp_path / "protocol.yaml"),
            str(tmp_path / "scratch"),
            1,
            0.99,
            0.02,
            "runtime-source-hash",
            "protocol-source-hash",
        )

    evaluate.assert_not_called()
    assert result["target_frames"] == []
    assert result["branches"] == []
    assert result["labels"] == []
    assert result["accounting"]["release_states_evaluated"] == 0


def test_target_map_requires_exact_seed_block_and_accounts_duplicates(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"7": [5, 5], "8": [4]}), encoding="utf-8")

    targets, metadata = branch_labels._load_target_map(path, [7, 8])

    assert targets == {7: [5, 5], 8: [4]}
    assert metadata["requested_entries"] == 3
    assert metadata["unique_release_states"] == 2
    assert metadata["duplicate_entries"] == 1
    with pytest.raises(ValueError, match="seed keys differ"):
        branch_labels._load_target_map(path, [7])


def test_source_provenance_rejects_runtime_code_drift(tmp_path):
    config = {
        "protocol_name": "always_fast",
        "env_type": "highway-v0",
        "scenario_type": "highway",
        "closed_loop_latency_replay": {"enable": False},
        "system_routing": {"simple": "fast", "complex": "fast"},
    }
    config_hash = branch_labels.hashlib.sha256(
        json.dumps(
            config, sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")
    ).hexdigest()
    protocol_hash = "1" * 64
    protocol_path = tmp_path / "formal_protocol.yaml"
    artifact_paths = {
        "reasoning": tmp_path / "reasoning.json",
        "physical": tmp_path / "physical.json",
        "snapshot_bundle": tmp_path / "snapshots.pkl",
    }
    for label, path in artifact_paths.items():
        path.write_bytes(label.encode("ascii"))
    acquisition = {
        "schema_version": 2,
        "policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
        "policy_state_integrity": "canonical_json_sha256",
        "producer_path": str(branch_labels.SNAPSHOT_PRODUCER_PATH.resolve()),
        "producer_sha256": branch_labels._sha256(
            branch_labels.SNAPSHOT_PRODUCER_PATH
        ),
        "base_config_path": str(branch_labels.BASE_CONFIG_PATH.resolve()),
        "base_config_sha256": branch_labels._sha256(
            branch_labels.BASE_CONFIG_PATH
        ),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_hash,
        "artifact_hashes": {
            label: {
                "path": str(path.resolve()),
                "sha256": branch_labels._sha256(path),
            }
            for label, path in sorted(artifact_paths.items())
        },
    }
    common = {
        "protocol_id": f"always_fast::{protocol_hash[:16]}",
        "protocol_hash": protocol_hash,
        "config_hash": config_hash,
        "source_hash": "3" * 64,
        "config": config,
        "runtime_environment": branch_labels._current_runtime_environment(),
        "snapshot_acquisition": acquisition,
    }
    experiment = {
        **common,
        "fixed_seed_override": 7,
        "seeds_used": [7],
    }
    experiment_path = tmp_path / "experiment_snapshot.json"
    manifest_path = tmp_path / "runtime_manifest.json"
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    manifest_path.write_text(json.dumps(common), encoding="utf-8")
    paths = {
        "experiment_snapshot": experiment_path,
        "runtime_manifest": manifest_path,
        **artifact_paths,
    }

    assert branch_labels._validate_source_provenance(
        paths, 7, "3" * 64, protocol_hash, protocol_path
    ) == config
    with pytest.raises(ValueError, match="snapshot/current runtime source drift"):
        branch_labels._validate_source_provenance(
            paths, 7, "4" * 64, protocol_hash, protocol_path
        )
    with pytest.raises(ValueError, match="acquisition provenance drift at protocol_sha256"):
        branch_labels._validate_source_provenance(
            paths, 7, "3" * 64, "4" * 64, protocol_path
        )


def test_checkpoint_resume_revalidates_exact_candidate_universe(tmp_path):
    source_hashes = {"snapshot_bundle": {"path": "snapshots.pkl", "sha256": "1" * 64}}
    rows = {
        None: branch_row("fast", effective_action=1, target_speed=20.0, utility=0.0),
        0: branch_row(0, effective_action=0, target_speed=18.0, utility=0.05),
        1: branch_row(1, effective_action=1, target_speed=20.0, utility=0.0),
        3: branch_row(3, effective_action=0, target_speed=18.0, utility=0.05),
    }
    contract = branch_labels._exact_action_contract(make_record(), seed=7, frame=5)
    with patch.object(
        branch_labels,
        "_run_branch",
        side_effect=lambda snapshot, cfg, seed, action, horizon, gamma: dict(rows[action]),
    ):
        evaluation = branch_labels._evaluate_release_state(
            make_snapshot(),
            {},
            seed=7,
            frame=5,
            record=make_record(),
            physical=make_physical(),
            state_identity_sha256="a" * 64,
            action_contract=contract,
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )
    payload = {
        "schema_version": branch_labels.CHECKPOINT_SCHEMA_VERSION,
        "method_version": branch_labels.METHOD_VERSION,
        "continuation_contract_version": branch_labels.CONTINUATION_CONTRACT_VERSION,
        "seed": 7,
        "contract_fingerprint": "f" * 64,
        "target_frames": [5],
        "source_artifacts": source_hashes,
        "source_execution_contract": {
            "all_trace_frames_fast": True,
            "source_latency_replay_enabled": True,
            "source_latency_replay_target_systems": ["slow"],
            "branch_latency_replay_enabled": False,
            "equivalence": "inert Fast-only replay",
        },
        "labels": evaluation["label"] and [evaluation["label"]],
        "branches": evaluation["branches"],
        "accounting": {
            "seed": 7,
            "status": "complete",
            "requested_target_entries": 1,
            "unique_release_states": 1,
            "duplicate_targets_excluded": 0,
            "release_states_evaluated": 1,
            "release_states_excluded": 0,
            "release_state_errors": 0,
            "branch_rows": 4,
            "gate_candidate_branches": 3,
            "candidate_actions_matching_fast_identity": 1,
            "candidate_effective_aliases_collapsed": 1,
            "corrective_set_actions": 1,
            "corrective_release_states": 1,
        },
    }
    payload["checkpoint_payload_sha256"] = branch_labels._checkpoint_payload_sha256(
        payload
    )
    with (
        patch.object(branch_labels, "_source_paths", return_value={}),
        patch.object(branch_labels, "_source_hashes", return_value=source_hashes),
    ):
        resumed = branch_labels._validate_checkpoint(
            payload,
            seed=7,
            targets=[5],
            contract_fingerprint="f" * 64,
            trace_root=tmp_path,
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )
        assert resumed["target_frames"] == [5]
        tampered = json.loads(json.dumps(payload))
        tampered["branches"].pop()
        tampered["checkpoint_payload_sha256"] = (
            branch_labels._checkpoint_payload_sha256(tampered)
        )
        with pytest.raises(ValueError, match="(branch count|action universe) drift"):
            branch_labels._validate_checkpoint(
                tampered,
                seed=7,
                targets=[5],
                contract_fingerprint="f" * 64,
                trace_root=tmp_path,
                horizon=20,
                gamma=0.99,
                epsilon=0.02,
            )


def test_runner_source_contains_no_allocator_selection_dependency():
    source = branch_labels.Path(branch_labels.__file__).read_text(encoding="utf-8")

    assert "_selected_queries" not in source
    assert "gate_is_eligible" not in source
    assert "scheduled_frames" not in source


def test_v12_query_gate_contract_rejects_incompatible_and_unregistered_inputs(tmp_path):
    protocol = branch_labels.REPO_ROOT / "formal_protocol.yaml"

    assert branch_labels._validate_v12_protocol(
        protocol,
        seeds=list(range(2000, 2040)),
        horizon=20,
        gamma=0.99,
        epsilon=0.02,
    ) == "parameter_selection"

    incompatible = tmp_path / "formal_protocol.yaml"
    payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    payload["tvt_submission_contract"]["query_gate_method_version"] = "support_breadth_v11"
    incompatible.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="query-gate method version"):
        branch_labels._validate_v12_protocol(
            incompatible,
            seeds=list(range(2000, 2040)),
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )
    with pytest.raises(ValueError, match="not a preregistered v12 cohort"):
        branch_labels._validate_v12_protocol(
            protocol,
            seeds=[7],
            horizon=20,
            gamma=0.99,
            epsilon=0.02,
        )


def test_snapshot_producer_writes_identical_historical_artifact_hashes(tmp_path):
    result_dir = tmp_path / "seed_7"
    result_dir.mkdir()
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("protocol_version: 12\n", encoding="utf-8")
    artifacts = {
        "reasoning": tmp_path / "reasoning.json",
        "physical": tmp_path / "physical.json",
        "snapshot_bundle": tmp_path / "snapshots.pkl",
    }
    for label, path in artifacts.items():
        path.write_bytes(label.encode("ascii"))
    for name in ("experiment_snapshot.json", "runtime_manifest.json"):
        (result_dir / name).write_text("{}", encoding="utf-8")

    mechanism_runner._write_snapshot_acquisition_provenance(
        result_dir, protocol, artifacts
    )

    experiment = json.loads(
        (result_dir / "experiment_snapshot.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (result_dir / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    assert experiment["snapshot_acquisition"] == manifest["snapshot_acquisition"]
    provenance = experiment["snapshot_acquisition"]
    assert provenance["producer_sha256"] == mechanism_runner._sha256(
        branch_labels.SNAPSHOT_PRODUCER_PATH
    )
    assert provenance["base_config_sha256"] == mechanism_runner._sha256(
        branch_labels.BASE_CONFIG_PATH
    )
    assert set(provenance["artifact_hashes"]) == {
        "reasoning",
        "physical",
        "snapshot_bundle",
    }


def test_matched_fast_full_horizon_must_reproduce_source_trace():
    source = [
        {
            "frame_id": 5,
            "position_x": 10.0,
            "position_y": 4.0,
            "speed": 20.0,
            "lane_id": 1,
            "action_id": 1,
        },
        {
            "frame_id": 6,
            "position_x": 12.0,
            "position_y": 4.0,
            "speed": 20.1,
            "lane_id": 1,
            "action_id": 3,
        },
    ]
    trajectory = [
        {
            "frame": row["frame_id"],
            "position_x": row["position_x"],
            "position_y": row["position_y"],
            "speed": row["speed"],
            "lane_id": row["lane_id"],
            "effective_action": row["action_id"],
        }
        for row in source
    ]
    baseline = {
        "steps_completed": 2,
        "branch_trajectory_json": json.dumps(trajectory),
    }

    branch_labels._validate_matched_fast_trajectory(
        baseline, source, seed=7, frame=5
    )
    source[1]["action_id"] = 4
    with pytest.raises(ValueError, match="trajectory action drift"):
        branch_labels._validate_matched_fast_trajectory(
            baseline, source, seed=7, frame=5
        )


def test_worker_is_spawn_safe_and_returns_fail_closed_accounting_input(tmp_path):
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        result = pool.submit(
            branch_labels._worker,
            7,
            (),
            str(tmp_path),
            str(tmp_path / "protocol.yaml"),
            str(tmp_path / "scratch"),
            1,
            0.99,
            0.02,
            "runtime-source-hash",
            "protocol-source-hash",
        ).result(timeout=30)

    assert result["status"] == "error"
    assert result["seed"] == 7
    assert result["error_type"] in {"RuntimeError", "FileNotFoundError", "ValueError"}
