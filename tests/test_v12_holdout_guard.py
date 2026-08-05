import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dilu.driver_agent.policy_state import DRIVER_POLICY_STATE_SCHEMA
from tools import v12_holdout_guard as guard


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


@pytest.fixture
def issued_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "HOLDOUT_LEDGER_ROOT", tmp_path / "central_ledger")
    sources = {
        name: tmp_path / f"{name}.json"
        for name in ("protocol", "lock", "calibration", "go_no_go")
    }
    for path in sources.values():
        _write_json(path, {})
    source_payload = {
        "calibration_lock_id": "identifiable-gate-v12-calibration-20260720",
        "calibration_manifest_sha256": "1" * 64,
        "calibration_selection_digest": "2" * 64,
        "go_no_go_manifest_sha256": "3" * 64,
        "go_no_go_analysis_digest": "4" * 64,
        "protocol_sha256": "5" * 64,
        "lock_sha256": "6" * 64,
        "selector_sha256": "7" * 64,
        "gate_support_sha256": "8" * 64,
        "producer_sha256": guard.sha256_file(guard.PRODUCER_PATH),
        "branch_engine_sha256": guard.sha256_file(guard.BRANCH_ENGINE_PATH),
        "branch_runner_sha256": guard.sha256_file(guard.BRANCH_RUNNER_PATH),
        "base_config_sha256": guard.sha256_file(guard.BASE_CONFIG_PATH),
        "runtime_source_sha256": "9" * 64,
        "holdout_guard_sha256": guard.sha256_file(Path(guard.__file__)),
    }
    current_source = {"value": dict(source_payload)}
    monkeypatch.setattr(
        guard,
        "_verified_source_payload",
        lambda paths: dict(current_source["value"]),
    )
    authorization = tmp_path / "authorization" / "v12_holdout_authorization.json"
    guard.issue_holdout_authorization(
        authorization_path=authorization,
        protocol_path=sources["protocol"],
        lock_path=sources["lock"],
        calibration_manifest_path=sources["calibration"],
        go_no_go_manifest_path=sources["go_no_go"],
    )
    kwargs = {
        "authorization_path": authorization,
        "protocol_path": sources["protocol"],
        "lock_path": sources["lock"],
        "calibration_manifest_path": sources["calibration"],
        "go_no_go_manifest_path": sources["go_no_go"],
    }
    return kwargs, current_source, tmp_path


def _begin_trace(issued_holdout, *, run_stamp="trace_once"):
    kwargs, _, tmp_path = issued_holdout
    return guard.begin_producer_phase(
        **kwargs,
        seeds=guard.CONFIRMATORY_SEEDS,
        result_root=tmp_path / "results",
        run_stamp=run_stamp,
        no_snapshots=True,
        snapshot_targets=None,
    )


def _minimal_producer_manifest(claim: guard.HoldoutPhaseClaim) -> Path:
    phase_root = Path(claim.run_binding["phase_root"])
    source = claim.authorization.payload["source"]
    artifacts = []
    trace_files = []
    for seed in guard.CONFIRMATORY_SEEDS:
        result_dir = phase_root / "always_fast" / "highway" / f"seed_{seed}"
        reasoning = (
            result_dir
            / f"ep_{seed}"
            / f"highway_{seed}_reasoning_records.json"
        )
        physical = (
            result_dir
            / f"ep_{seed}"
            / f"highway_{seed}_physical_frames.json"
        )
        _write_json(reasoning, {"analysis_records": [{"frame_id": 0}]})
        _write_json(physical, {"frames": []})
        artifact_hashes = {
            "reasoning": {
                "path": str(reasoning.resolve()),
                "sha256": guard.sha256_file(reasoning),
            },
            "physical": {
                "path": str(physical.resolve()),
                "sha256": guard.sha256_file(physical),
            },
        }
        provenance = {
            "schema_version": 2,
            "policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
            "policy_state_integrity": "canonical_json_sha256",
            "producer_path": str(guard.PRODUCER_PATH.resolve()),
            "producer_sha256": source["producer_sha256"],
            "base_config_path": str(guard.BASE_CONFIG_PATH.resolve()),
            "base_config_sha256": source["base_config_sha256"],
            "branch_engine_path": str(guard.BRANCH_ENGINE_PATH.resolve()),
            "branch_engine_sha256": source["branch_engine_sha256"],
            "runtime_source_sha256": source["runtime_source_sha256"],
            "protocol_path": str(claim.authorization.source_paths["protocol"]),
            "protocol_sha256": source["protocol_sha256"],
            "artifact_hashes": artifact_hashes,
        }
        experiment = result_dir / "experiment_snapshot.json"
        runtime = result_dir / "runtime_manifest.json"
        _write_json(experiment, {"snapshot_acquisition": provenance})
        _write_json(runtime, {"snapshot_acquisition": provenance})
        paths = {
            "experiment_snapshot": experiment,
            "runtime_manifest": runtime,
            "reasoning": reasoning,
            "physical": physical,
        }
        for role, path in paths.items():
            artifacts.append(
                {
                    "seed": seed,
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": guard.sha256_file(path),
                }
            )
        trace_files.append(
            {
                "seed": seed,
                "sha256": guard.sha256_file(reasoning),
                "name": reasoning.name,
            }
        )
    payload = {
        "schema": guard.PRODUCER_MANIFEST_SCHEMA,
        "artifact_role": "confirmatory_holdout_trace_acquisition",
        "method_version": guard.METHOD_VERSION,
        "status": "completed",
        "authorization_id": claim.authorization.authorization_id,
        "authorization_sha256": claim.authorization.raw_sha256,
        "run_id": claim.run_id,
        "phase": "trace",
        "seed_block": guard.seed_block_payload(),
        "run_binding": dict(claim.run_binding),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "trace_raw_file_set_hash": guard.canonical_sha256(trace_files),
    }
    payload["manifest_payload_sha256"] = guard.canonical_sha256(payload)
    path = phase_root / "v12_holdout_producer_manifest.json"
    _write_json(path, payload)
    return path


def test_seed_classifier_rejects_partial_and_overlapping_holdout_ranges():
    assert guard.classify_seed_request(2000, 40) == "ordinary"
    assert guard.classify_seed_request(2040, 20) == "ordinary"
    assert guard.classify_seed_request(3000, 30) == "confirmatory_holdout"
    for start, count in ((3000, 1), (2999, 31), (3001, 29), (2990, 50)):
        with pytest.raises(ValueError, match="exactly seeds 3000-3029"):
            guard.classify_seed_request(start, count)


def test_real_minimal_trace_artifacts_complete_the_one_shot_stage(issued_holdout):
    claim, permit = _begin_trace(issued_holdout)
    assert permit.seeds == guard.CONFIRMATORY_SEEDS
    assert guard.validate_runtime_holdout_marker(
        permit.runtime_marker, seed=3000
    )["run_id"] == claim.run_id
    opened = guard.load_json_strict(claim.authorization.state_path)
    assert opened["stage"] == "trace_open"
    assert opened["open_count"] == 1
    assert opened["capability_consumed"] is True

    manifest = _minimal_producer_manifest(claim)
    state = guard.complete_producer_phase(claim, manifest)
    assert state["stage"] == "traces_generated"
    assert state["open_count"] == 1
    assert state["bindings"]["trace_producer"]["manifest_sha256"] == guard.sha256_file(manifest)
    assert claim.claim_path.is_file()
    assert guard._outcome_path(claim.authorization.authorization_id, "trace").is_file()


def test_concurrent_open_has_exactly_one_winner_and_replay_is_burned(issued_holdout):
    def attempt():
        try:
            return _begin_trace(issued_holdout)[0]
        except Exception as exc:  # noqa: BLE001 - the losing race is the assertion
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: attempt(), range(2)))
    claims = [item for item in results if isinstance(item, guard.HoldoutPhaseClaim)]
    failures = [item for item in results if isinstance(item, Exception)]
    assert len(claims) == 1
    assert len(failures) == 1
    guard.fail_phase(claims[0], RuntimeError("intentional unit-test stop"))
    with pytest.raises(ValueError, match="requires stage authorized_unopened"):
        _begin_trace(issued_holdout)


def test_state_rollback_and_copied_authorization_cannot_replay(issued_holdout):
    kwargs, _, tmp_path = issued_holdout
    state_path = guard.authorization_state_path(kwargs["authorization_path"])
    pristine_state = state_path.read_bytes()
    claim, _ = _begin_trace(issued_holdout)
    guard.fail_phase(claim, "stop")

    state_path.write_bytes(pristine_state)
    with pytest.raises(ValueError, match="already claimed|replay"):
        _begin_trace(issued_holdout)

    copied = tmp_path / "copied" / "v12_holdout_authorization.json"
    copied.parent.mkdir()
    shutil.copy2(kwargs["authorization_path"], copied)
    shutil.copy2(state_path, copied.with_name("v12_holdout_authorization_state.json"))
    with pytest.raises(ValueError, match="copied or relocated"):
        guard.verify_holdout_authorization(
            **{**kwargs, "authorization_path": copied}
        )


def test_stale_source_and_direct_seed_access_fail_closed(issued_holdout):
    kwargs, current_source, _ = issued_holdout
    current_source["value"]["go_no_go_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="stale or forged"):
        guard.verify_holdout_authorization(**kwargs)

    with pytest.raises(ValueError, match="direct confirmatory"):
        guard.assert_producer_permit(None, seed=3000, capture_snapshots=False)
    guard.assert_producer_permit(None, seed=2999, capture_snapshots=False)


def test_same_evidence_cannot_issue_a_second_authorization(issued_holdout):
    kwargs, _, tmp_path = issued_holdout
    with pytest.raises(ValueError, match="second issuance is forbidden"):
        guard.issue_holdout_authorization(
            **{
                **kwargs,
                "authorization_path": (
                    tmp_path
                    / "different_directory"
                    / "v12_holdout_authorization.json"
                ),
            }
        )


def test_snapshot_phase_cannot_capture_all_frames(issued_holdout):
    kwargs, _, tmp_path = issued_holdout
    with pytest.raises(ValueError, match="locked snapshot targets"):
        guard.begin_producer_phase(
            **kwargs,
            seeds=guard.CONFIRMATORY_SEEDS,
            result_root=tmp_path / "snapshot_results",
            run_stamp="snapshot_once",
            no_snapshots=False,
            snapshot_targets=None,
        )
