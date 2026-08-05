import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from dilu.evaluation import reporter
from tools.run_main_table import _bind_setting_seed
from tools.run_main_table_runtime import (
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
)
from tools.run_main_table_support import write_overall_comparison_assets, write_rows_csv
from tools.v12_floor_overlay import (
    APPLIED_STATUS,
    FLOOR_OVERLAY_SCHEMA,
    FLOOR_SELECTION_SOURCE,
    apply_floor_overlay,
)
from tools.verify_v12_main_results import (
    AcceptanceError,
    COMPARISON_HEADLINE_FIELDS,
    EXPECTED_ARMS,
    EXPECTED_SEEDS,
    METHOD_VERSION,
    REPO_ROOT,
    verify,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _protocol_copy(path: Path) -> Path:
    payload = yaml.safe_load((REPO_ROOT / "formal_protocol_v12.yaml").read_text(encoding="utf-8"))
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_repository_protocol_declares_frozen_six_arm_main_contract():
    payload = yaml.safe_load(
        (REPO_ROOT / "formal_protocol_v12.yaml").read_text(encoding="utf-8")
    )
    main = payload["tvt_submission_contract"]["evidence_artifacts"]["artifacts"][
        "main_results"
    ]
    assert tuple(main["required_groups"]) == EXPECTED_ARMS
    assert main["environment"] == "highway-v0"
    assert main["episodes_per_key"] == 1
    assert main["zero_query_allowed_groups"] == ["always_fast"]
    assert (
        main["technical_slow_failure_policy"]
        == "reject_any_scheduled_slow_failure_or_fallback"
    )
    always_slow = payload["groups"]["always_slow"]
    assert always_slow["publication_track"] == "main_text"
    assert always_slow["runtime_overrides"]["slow_call_budget"] == 6
    assert always_slow["runtime_overrides"]["slow_call_cooldown_frames"] == 20
    assert "always_slow" in payload["paper_baselines"]["main_text_groups"]
    assert "always_slow" not in payload["paper_baselines"]["archive_only_groups"]


def _fake_floor_overlay(root: Path):
    overlay_path = root / "inputs" / "floor_overlay.json"
    calibration_path = root / "inputs" / "calibration_manifest.json"
    protocol_path = root / "inputs" / "formal_protocol_v12.yaml"
    lock_path = root / "inputs" / "calibration_lock.json"
    _write_json(overlay_path, {})
    _write_json(calibration_path, {})
    _write_json(protocol_path, {})
    _write_json(lock_path, {})
    floors = {
        "rgd_latency_survival_floor": 0.20,
        "rgd_maneuver_breadth_floor": 0.20,
        "rgd_corrective_headroom_floor": 0.20,
        "rgd_state_need_floor": 0.20,
    }
    runtime_binding = {
        "schema": FLOOR_OVERLAY_SCHEMA,
        "method_version": METHOD_VERSION,
        "floor_overlay_sha256": "1" * 64,
        "floor_overlay_payload_sha256": "2" * 64,
        "calibration_manifest_sha256": "3" * 64,
        "calibration_selection_digest": "4" * 64,
        "calibration_lock_id": "test-calibration-lock",
        "candidate_id": "test-candidate",
        "floor_source": FLOOR_SELECTION_SOURCE,
        "floors": dict(floors),
        "protocol_sha256": "5" * 64,
        "lock_sha256": "6" * 64,
        "selector_sha256": "7" * 64,
        "gate_support_sha256": "8" * 64,
    }
    return SimpleNamespace(
        path=overlay_path.resolve(),
        calibration_manifest_path=calibration_path.resolve(),
        protocol_path=protocol_path.resolve(),
        lock_path=lock_path.resolve(),
        floors=floors,
        runtime_binding=runtime_binding,
    )


def _event(seed: int, *, slow: bool):
    system = "slow" if slow else "fast"
    return {
        "frame": 0,
        "system_used": system,
        "rgd_method_version": METHOD_VERSION,
        "slow_request_attempted": slow,
        "slow_request_valid_return": slow,
        "slow_request_failed": False,
        "slow_reasoning_success": slow,
        "slow_reasoning_failure_reason": "",
        "episode_done": True,
        "terminal_cause": "truncated",
        "crashed": False,
        "arrive_dest": False,
        "route_completion": 1.0,
        "seed": seed,
    }


def _row_metrics(arm: str, *, slow: bool):
    values = {
        "collision_rate": 0.0,
        "success_rate": 1.0,
        "avg_route_completion": 1.0,
        "avg_episode_reward": 1.0,
        "avg_driving_distance": 10.0,
        "avg_speed_safety_qualified": 5.0,
        "avg_runtime_per_frame": 0.01,
        "budget_normalized_independent_high_risk_utility": 0.1,
        "independent_selective_routing_gain": 0.1,
    }
    assert set(values) == set(COMPARISON_HEADLINE_FIELDS)
    values.update(
        {
            "experiment_name": arm,
            "total_episodes": 1,
            "total_frames": 1,
            "single_core_method_name": "Recoverability-Gated Deliberation",
            "primary_evaluation_subject": "fixed-policy RGD",
            "evaluation_protocol_name": arm,
            "evaluation_runtime_stable": True,
            "runtime_integrity_clean": True,
            "runtime_integrity_violation_rate": 0.0,
            "slow_call_rate": float(slow),
            "slow_call_success_rate": float(slow),
        }
    )
    return values


def _build_bundle(tmp_path: Path, monkeypatch, *, zero_query_arm=None):
    protocol_path = _protocol_copy(tmp_path / "formal_protocol_v12_test.yaml")
    protocol = load_formal_protocol(protocol_path)
    fake_overlay = _fake_floor_overlay(tmp_path)
    base_cfg = load_formal_base_config(protocol, REPO_ROOT / "config.yaml")
    base_cfg = apply_floor_overlay(base_cfg, fake_overlay)
    live_source_hash = reporter.build_runtime_source_hash(REPO_ROOT)
    bundle = tmp_path / "formal_run" / "paper_main_v12"
    rows_by_arm = {}

    monkeypatch.setattr(
        "tools.v12_floor_overlay.load_verified_floor_overlay",
        lambda *args, **kwargs: fake_overlay,
    )
    memory_artifact = {
        "path": str((REPO_ROOT / "memories/state_mem/state_memory.db").resolve()),
        "exists": True,
        "size_bytes": 1,
        "sha256": "9" * 64,
        "runtime_enabled": False,
    }
    with patch.object(reporter, "build_runtime_source_hash", return_value=live_source_hash), patch.object(
        reporter, "_build_state_memory_artifact", return_value=memory_artifact
    ), patch.object(reporter.importlib_metadata, "version", return_value="test"), patch.object(
        reporter.subprocess,
        "check_output",
        side_effect=subprocess.CalledProcessError(128, "git"),
    ):
        for arm in EXPECTED_ARMS:
            group_cfg = dict(protocol["groups"][arm])
            arm_rows = []
            for seed in EXPECTED_SEEDS:
                result_dir = bundle / arm / "highway" / f"seed_{seed:02d}"
                result_dir.mkdir(parents=True, exist_ok=True)
                cfg = build_group_config(
                    base_cfg,
                    arm,
                    group_cfg,
                    "highway-v0",
                    1,
                    result_dir,
                    protocol,
                )
                _bind_setting_seed(cfg, seed)
                reporter.save_experiment_snapshot(cfg, str(result_dir), seed)
                runtime_manifest = json.loads(
                    (result_dir / "runtime_manifest.json").read_text(encoding="utf-8")
                )

                slow = arm != "always_fast" and arm != zero_query_arm
                event = _event(seed, slow=slow)
                _write_json(
                    result_dir / "event_logs" / f"event_log_highway_{seed}_{seed}.json",
                    {
                        "schema_version": "rgd_event_log_v2",
                        "episode_id": seed,
                        "event_count": 1,
                        "pending_release_count": 0,
                        "pending_releases_dropped_at_episode_end": [],
                        "terminal_cause": "truncated",
                        "events": [event],
                    },
                )
                trace_dir = result_dir / f"ep_{seed}"
                _write_json(
                    trace_dir / f"highway_{seed}_reasoning_records.json",
                    {
                        "episode_id": seed,
                        "record_count": 1,
                        "analysis_records": [
                            {
                                "frame_id": 0,
                                "system_used": event["system_used"],
                                "slow_reasoning_success": event["slow_reasoning_success"],
                                "slow_reasoning_failure_reason": "",
                            }
                        ],
                    },
                )
                _write_json(
                    trace_dir / f"highway_{seed}_physical_frames.json",
                    {
                        "episode_id": seed,
                        "frame_count": 1,
                        "frames": [{"frame_id": 0, "speed": 5.0}],
                    },
                )
                metrics = _row_metrics(arm, slow=slow)
                _write_json(
                    result_dir / f"{arm}_rgd_metrics.json",
                    {"comprehensive_metrics": metrics},
                )
                row = {
                    "group": arm,
                    "env": "highway-v0",
                    "seed_idx": seed,
                    "episodes_run": 1,
                    "fixed_seed_override": seed,
                    "seed_start": seed,
                    "requested_seed_start": EXPECTED_SEEDS[0],
                    "total_frames": 1,
                    "result_dir": str(result_dir.resolve()),
                    **{field: runtime_manifest[field] for field in ("protocol_id", "protocol_hash", "config_hash", "source_hash")},
                    **{field: metrics[field] for field in COMPARISON_HEADLINE_FIELDS},
                    "slow_call_rate": metrics["slow_call_rate"],
                    "slow_call_success_rate": metrics["slow_call_success_rate"],
                }
                arm_rows.append(row)
            rows_by_arm[arm] = arm_rows
            write_rows_csv(bundle / arm / f"{arm}_run_rows.csv", arm_rows)

    _write_json(
        bundle / "result_bundle_manifest.json",
        {
            "updated": "test",
            "bundle_kind": "formal_run",
            "bundle_root": str(bundle.resolve()),
            "groups": list(EXPECTED_ARMS),
            "group_env_matrix": {arm: ["highway-v0"] for arm in EXPECTED_ARMS},
            "envs": ["highway-v0"],
            "seeds": 30,
            "episodes": 1,
            "seed_policy": "fixed_per_setting",
            "seed_start": EXPECTED_SEEDS[0],
            "seed_labels": list(EXPECTED_SEEDS),
            "seed_value": None,
            "simulation_duration": None,
            "formal_protocol_path": str(protocol_path.resolve()),
            "entry_artifacts": [
                "result_bundle_manifest.json",
                "overall_group_comparison.csv",
                "overall_group_comparison.json",
            ],
        },
    )
    write_overall_comparison_assets(bundle, rows_by_arm, EXPECTED_ARMS, ["highway-v0"])
    return bundle, protocol_path


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_complete_six_arm_matrix_accepts_explicit_always_fast_q0(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    report = verify(bundle, protocol_path=protocol)
    by_arm = {row["arm"]: row for row in report["arms"]}
    assert report["accepted"] is True
    assert report["episodes"] == 180
    assert by_arm["always_fast"]["queries"] == 0
    assert by_arm["always_slow"]["queries"] == by_arm["always_slow"]["frames"] == 30


def test_missing_arm_seed_row_fails_closed(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    path = bundle / "risk_budget" / "risk_budget_run_rows.csv"
    rows = _read_rows(path)[:-1]
    write_rows_csv(path, rows)
    with pytest.raises(AcceptanceError, match="expected 30 run rows"):
        verify(bundle, protocol_path=protocol)


def test_identical_duplicate_arm_seed_row_fails_closed(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    path = bundle / "random_budget" / "random_budget_run_rows.csv"
    rows = _read_rows(path)
    write_rows_csv(path, [*rows, dict(rows[0])])
    with pytest.raises(AcceptanceError):
        verify(bundle, protocol_path=protocol)


def test_synchronized_source_hash_forgery_fails_against_live_source(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    arm = "uncertainty_budget"
    seed = EXPECTED_SEEDS[0]
    forged = "f" * 64
    result_dir = bundle / arm / "highway" / f"seed_{seed}"
    for name in ("runtime_manifest.json", "experiment_snapshot.json"):
        path = result_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source_hash"] = forged
        _write_json(path, payload)
    rows_path = bundle / arm / f"{arm}_run_rows.csv"
    rows = _read_rows(rows_path)
    rows[0]["source_hash"] = forged
    write_rows_csv(rows_path, rows)
    with pytest.raises(AcceptanceError, match="live runtime source"):
        verify(bundle, protocol_path=protocol)


def test_http_401_fallback_is_rejected_despite_process_exit_zero(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    arm = "risk_budget"
    seed = EXPECTED_SEEDS[0]
    result_dir = bundle / arm / "highway" / f"seed_{seed}"
    event_path = next((result_dir / "event_logs").glob("event_log_*.json"))
    event_payload = json.loads(event_path.read_text(encoding="utf-8"))
    event = event_payload["events"][0]
    event.update(
        {
            "system_used": "fast_after_slow_failure",
            "slow_request_attempted": True,
            "slow_request_valid_return": False,
            "slow_request_failed": True,
            "slow_reasoning_success": False,
            "slow_reasoning_failure_reason": "llm_invoke_failed:RuntimeError",
        }
    )
    _write_json(event_path, event_payload)
    reasoning_path = next(result_dir.glob("ep_*/*_reasoning_records.json"))
    reasoning = json.loads(reasoning_path.read_text(encoding="utf-8"))
    reasoning["analysis_records"][0].update(
        {
            "system_used": "fast_after_slow_failure",
            "slow_reasoning_success": False,
            "slow_reasoning_failure_reason": "llm_invoke_failed:RuntimeError",
        }
    )
    _write_json(reasoning_path, reasoning)
    _write_json(
        result_dir / "process_status.json",
        {"producer_exit_code": 0, "http_status": 401, "stderr": "HTTP 401 unauthorized"},
    )
    with pytest.raises(AcceptanceError):
        verify(bundle, protocol_path=protocol)


def test_zero_queries_are_allowed_only_for_always_fast(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch, zero_query_arm="risk_budget")
    with pytest.raises(AcceptanceError, match="Q=0 is permitted only for always_fast"):
        verify(bundle, protocol_path=protocol)


def test_fast_q_zero_must_be_explicit_and_trace_derived(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    path = bundle / "always_fast" / "always_fast_run_rows.csv"
    rows = _read_rows(path)
    rows[0]["slow_call_rate"] = "N/A"
    write_rows_csv(path, rows)
    with pytest.raises(AcceptanceError, match="slow_call_rate must be numeric"):
        verify(bundle, protocol_path=protocol)


def test_reported_query_rate_must_equal_trace_attempts(tmp_path, monkeypatch):
    bundle, protocol = _build_bundle(tmp_path, monkeypatch)
    path = bundle / "rgd_fixed_policy" / "rgd_fixed_policy_run_rows.csv"
    rows = _read_rows(path)
    rows[0]["slow_call_rate"] = "0"
    write_rows_csv(path, rows)
    with pytest.raises(AcceptanceError, match="Q rate differs from traces"):
        verify(bundle, protocol_path=protocol)
