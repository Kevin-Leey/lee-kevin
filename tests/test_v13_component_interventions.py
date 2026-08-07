from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from dilu.evaluation.factorial_replay import COMPONENT_ABLATION_ARMS
from dilu.evaluation.factorial_replay import (
    FACTORIAL_EVENT_SCHEMA,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
)
from dilu.evaluation.release_snapshot import RELEASE_SNAPSHOT_BUNDLE_SCHEMA
from tools import analyze_v13_component_interventions as component_analyzer
from tools import verify_v13_component_ablation as component_verifier
from tools.analyze_factorial_interventions import EVENT_ROW_FIELDS, summarize_events
from tools.analyze_v13_component_interventions import (
    AUDIT_FILE,
    EVENTS_FILE,
    MAIN_EFFECTS_FILE,
    MANIFEST_FILE,
    OUTPUT_FILES,
    _episode_effects,
)
from tools.run_main_table_runtime import (
    load_formal_protocol,
    resolve_policy_execution_horizon,
)
from tools.run_v13_component_ablation import (
    COMPONENT_ABLATION_DESIGN,
    COMPONENT_ABLATION_RUN_SCHEMA,
    COMPONENT_ABLATION_VERSION,
)


def _event(seed, *, executed, classification, rejected=0, utility=0.2):
    return {
        "arm": "full",
        "seed": seed,
        "candidate_evaluable": 1,
        "first_step_actuator_distinct": 1,
        "executed_first_step_actuator_distinct": executed,
        "release_guard_rejected": rejected,
        "classification": classification,
        "utility_delta": utility,
        "normalized_return_delta": utility,
        "collision_delta": -1 if classification == "beneficial" else 1,
        "progress_delta_m": utility * 10.0,
        "min_ttc_delta_s": utility,
        "mean_abs_jerk_delta_mps3": -utility,
    }


def test_release_summary_reports_coverage_errors_and_endpoint_effects():
    summary = summarize_events(
        [
            _event(1, executed=1, classification="harmful", utility=-0.2),
            _event(2, executed=0, classification="beneficial", rejected=1),
        ],
        seeds=(1, 2),
        draws=40,
        bootstrap_seed=9,
        arms=("full",),
    )
    values = {row["metric"]: row for row in summary}

    assert values["executed_distinct_coverage_of_evaluable_releases"]["estimate"] == 0.5
    assert values["harmful_fraction_of_executed_first_step_interventions"]["estimate"] == 1.0
    assert values["beneficial_fraction_of_release_guard_rejections"]["estimate"] == 1.0
    assert values["missed_beneficial_fraction_of_evaluable_distinct_candidates"]["estimate"] == 0.5
    assert values["collision_delta_per_executed_first_step_intervention"]["estimate"] == 1.0
    assert values["progress_delta_m_per_executed_first_step_intervention"]["estimate"] == -2.0


def _episode_rows():
    rows = []
    for seed, offset in ((10, 0.0), (11, 2.0)):
        values = {
            "full": 10.0 + offset,
            "without_l": 9.0 + offset,
            "without_a": 8.0 + offset,
            "without_h": 7.0 + offset,
            "without_n": 8.0 + offset,
            "without_h_and_n": 3.0 + offset,
        }
        for arm, value in values.items():
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "collision": 0.0,
                    "route_completion": 1.0,
                    "episode_reward": value,
                    "driving_distance": value * 10.0,
                    "avg_speed": value,
                }
            )
    return rows


def test_episode_effects_include_the_prespecified_h_by_n_interaction():
    arms = tuple(arm.name for arm in COMPONENT_ABLATION_ARMS)
    effects = _episode_effects(
        _episode_rows(),
        seeds=(10, 11),
        arms=arms,
        draws=40,
        bootstrap_seed=4,
    )
    values = {(row["effect"], row["metric"]): row for row in effects}

    assert values[("full_minus_without_n", "episode_reward")]["estimate"] == 2.0
    assert values[("h_x_n_interaction", "episode_reward")]["estimate"] == -2.0
    assert values[("h_x_n_interaction", "driving_distance")]["estimate"] == -20.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload, *, allow_nan=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=True,
            indent=2,
            allow_nan=allow_nan,
            default=lambda value: value.isoformat(),
        )
        + "\n",
        encoding="utf-8",
    )


def _synthetic_rollout_row(task):
    row = {field: "" for field in EVENT_ROW_FIELDS}
    baseline_trajectory = json.dumps([{"frame": 17, "effective_action": 1}])
    candidate_trajectory = json.dumps([{"frame": 17, "effective_action": 4}])
    row.update(
        {
            "arm": str(task["arm"]),
            "seed": int(task["seed"]),
            "request_id": "seed-6000-request-0",
            "source_frame": 0,
            "release_frame": 17,
            "fast_action": 1,
            "slow_action": 4,
            "release_selected_action": 4,
            "selection_stage_primitive_distinct": 1,
            "candidate_effective_action": 4,
            "final_actuator_action": 4,
            "executed_action": 4,
            "release_guard_rejected": 0,
            "release_action_unavailable": 0,
            "candidate_evaluable": 1,
            "candidate_replay_unavailable": 0,
            "first_step_actuator_distinct": 1,
            "executed_first_step_actuator_distinct": 1,
            "classification": "beneficial",
            "baseline_utility": 0.0,
            "candidate_utility": 0.3,
            "utility_delta": 0.3,
            "baseline_normalized_return": 0.0,
            "candidate_normalized_return": 0.3,
            "normalized_return_delta": 0.3,
            "baseline_collision": 0,
            "candidate_collision": 0,
            "collision_delta": 0,
            "baseline_progress_m": 0.0,
            "candidate_progress_m": 3.0,
            "progress_delta_m": 3.0,
            "baseline_min_ttc_s": 2.0,
            "candidate_min_ttc_s": 3.0,
            "min_ttc_delta_s": 1.0,
            "baseline_mean_abs_jerk_mps3": 1.0,
            "candidate_mean_abs_jerk_mps3": 0.5,
            "mean_abs_jerk_delta_mps3": -0.5,
            "baseline_steps_completed": 20,
            "candidate_steps_completed": 20,
            "baseline_terminal_cause": "horizon",
            "candidate_terminal_cause": "horizon",
            "baseline_completed_horizon": 1,
            "candidate_completed_horizon": 1,
            "baseline_branch_trajectory_json": baseline_trajectory,
            "candidate_branch_trajectory_json": candidate_trajectory,
            "baseline_branch_trajectory_sha256": hashlib.sha256(
                baseline_trajectory.encode("utf-8")
            ).hexdigest(),
            "candidate_branch_trajectory_sha256": hashlib.sha256(
                candidate_trajectory.encode("utf-8")
            ).hexdigest(),
            "horizon_steps": int(task["horizon"]),
            "gamma": float(task["gamma"]),
            "epsilon": float(task["epsilon"]),
        }
    )
    assert tuple(row) == EVENT_ROW_FIELDS
    return [row]


def _synthetic_event(arm, frame, bank_sha256, snapshot_identity):
    spec = next(value for value in COMPONENT_ABLATION_ARMS if value.name == arm)
    event = {
        "frame": frame,
        "_runtime_available_actions": [0, 1, 2, 3, 4],
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "factorial_arm": arm,
        "factorial_query_gate_enabled": True,
        "factorial_release_guard_enabled": True,
        "factorial_proposal_bank_sha256": bank_sha256,
        "factorial_candidate_query": False,
        "factorial_candidate_request_id": "",
        "factorial_query_issued": False,
        "closed_loop_latency_issuance_event": False,
        "closed_loop_latency_issued_request_id": "",
        "closed_loop_latency_issued_response_outcome": "",
        "closed_loop_latency_terminal_event": False,
        "closed_loop_latency_terminal_request_id": "",
        "closed_loop_latency_terminal_response_outcome": "",
        "closed_loop_latency_release_event": False,
        "closed_loop_latency_timeout_event": False,
        "closed_loop_latency_failure_event": False,
        "closed_loop_release_snapshot_captured": False,
    }
    if frame == 0:
        gate = {
            "method_version": "identifiable_gate_v12",
            "gate_composition": "explicit_serial_floors",
            "gate_action_universe_source": (
                "driving_state.effective_action_universe"
            ),
            "fast_executor_action_universe_source": (
                "driving_state.effective_action_universe"
            ),
            "alternative_metric_source": (
                "relative_support_weighted_maneuver_family_breadth"
            ),
            "headroom_metric_source": (
                "incumbent_relative_action_recovery_cost_margin"
            ),
            "need_metric_source": "state_hazard_and_pre_screen_only",
            "support_breadth_formula": (
                "sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families"
            ),
            "corrective_headroom_kappa_source": "identifiable_gate_v12.fixed_kappa",
            "viable_cost_threshold": 0.55,
            "support_breadth_temperature": 0.10,
            "corrective_headroom_kappa": 0.55,
            "latency_survival_floor": 0.05,
            "maneuver_breadth_floor": 0.55,
            "corrective_headroom_floor": 0.10,
            "state_need_floor": 0.20,
            "effective_delay_steps": 17,
            "policy_frequency": 10.0,
            "safety_reserve_seconds": 0.0,
            "absolute_alternative_feasibility_non_ablatable": True,
            "domain_contract_pass": True,
            "executor_available_pass": True,
            "latency_prediction_pass": True,
            "absolute_feasibility_pass": True,
            "latency_survival_pass": True,
            "maneuver_breadth_pass": True,
            "corrective_headroom_pass": True,
            "state_need_pass": True,
            "serial_gate_pass": True,
        }
        event.update(
            {
                "recoverability_gate": gate,
                "factorial_candidate_query": True,
                "factorial_candidate_request_id": "seed-6000-request-0",
                "factorial_query_gate_pass": True,
                "factorial_shared_raw_slow_action": 4,
                "factorial_shared_latency_steps": 17,
                "factorial_shared_response_sha256": "",
                "factorial_shared_response_outcome": "valid",
                "factorial_query_issued": True,
                "closed_loop_latency_issuance_event": True,
                "closed_loop_latency_issued_request_id": "seed-6000-request-0",
                "closed_loop_latency_issued_response_outcome": "valid",
                "closed_loop_latency_request_id": "seed-6000-request-0",
                "component_ablation_policy": "serial_predicate_removal",
                "component_ablation_arm": arm,
                "component_ablation_removed_components": ";".join(
                    spec.removed_components
                ),
                "component_ablation_non_ablatable_pass": True,
                "component_ablation_full_equivalent_to_serial_gate": True,
            }
        )
        for name in (
            "latency_survival",
            "maneuver_breadth",
            "corrective_headroom",
            "state_need",
        ):
            event[f"component_ablation_{name}_pass"] = True
            event[f"component_ablation_{name}_retained"] = (
                name not in spec.removed_components
            )
    if frame == 17:
        event.update(
            {
                "closed_loop_latency_terminal_event": True,
                "closed_loop_latency_terminal_request_id": "seed-6000-request-0",
                "closed_loop_latency_terminal_response_outcome": "valid",
                "closed_loop_latency_terminal_outcome": "distinct_actuation",
                "closed_loop_latency_request_id": "seed-6000-request-0",
                "closed_loop_latency_source_frame": 0,
                "closed_loop_latency_delay_steps": 17,
                "closed_loop_latency_scheduled_release_frame": 17,
                "closed_loop_latency_release_event": True,
                "closed_loop_release_snapshot_captured": True,
                "closed_loop_release_snapshot_identity_sha256": snapshot_identity,
                "closed_loop_released_slow_action": 4,
                "closed_loop_execution_state_fast_action": 1,
                "release_fast_comparator_action": 1,
                "release_selected_action": 4,
                "release_selection_distinct": True,
                "release_action_comparison_stage": (
                    component_analyzer.CURRENT_RELEASE_SELECTION_STAGE
                ),
                "final_actuator_action": 4,
                "final_action": 4,
                "closed_loop_latency_executed_action": 4,
                "final_actuator_action_stage": (
                    component_analyzer.CURRENT_FINAL_ACTUATOR_STAGE
                ),
                "closed_loop_release_opportunity_rejected": False,
                "closed_loop_release_action_unavailable": False,
            }
        )
    return event


def _build_synthetic_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "raw_bundle"
    artifact = tmp_path / "analysis"
    bundle.mkdir()

    protocol = load_formal_protocol(Path("formal_protocol.yaml"))
    submission = protocol["tvt_submission_contract"]
    contract = submission["component_ablation"]
    contract["seed_range"] = {"start": 6000, "end": 6000, "count": 1}
    contract["bootstrap_draws"] = 40
    submission["evidence_artifacts"]["artifacts"]["component_ablation"][
        "required_manifest_values"
    ]["bootstrap_draws"] = 40
    protocol_path = tmp_path / "synthetic_protocol.yaml"
    _write_json(protocol_path, protocol)
    protocol_sha256 = _sha256(protocol_path)
    execution = resolve_policy_execution_horizon(
        contract["execution_contract"], context="synthetic component test"
    ).as_manifest()

    proposal_record = {
        "seed": 6000,
        "source_frame": 0,
        "request_id": "seed-6000-request-0",
        "raw_slow_action": 4,
        "latency_steps": 17,
        "outcome": "valid",
        "response_text": "",
        "response_sha256": "",
    }
    bank_payload = [{"seed": 6000, "records": [proposal_record]}]
    bank_sha256 = component_analyzer._canonical_sha256(bank_payload)
    _write_json(
        bundle / "proposal_bank_manifest.json",
        {
            "schema": FACTORIAL_PROPOSAL_SCHEMA,
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "source_root": str(tmp_path / "source"),
            "candidate_source_policy": contract["candidate_source_policy"],
            "candidate_source_gate_independent": True,
            "latency_profile": "fixed",
            "fixed_latency_steps": 17,
            "fixed_latency_seconds": 1.7,
            "bank_sha256": bank_sha256,
            "seed_count": 1,
            "proposal_count": 1,
            "source_artifacts": [],
            "bank_payload": bank_payload,
        },
    )

    metric_values = {
        "full": 10.0,
        "without_l": 9.0,
        "without_a": 8.0,
        "without_h": 7.0,
        "without_n": 8.0,
        "without_h_and_n": 3.0,
    }
    result_rows = []
    for spec in COMPONENT_ABLATION_ARMS:
        value = metric_values[spec.name]
        result_rows.append(
            {
                "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
                "arm": spec.name,
                "query_gate_enabled": True,
                "release_guard_enabled": True,
                "seed": 6000,
                "proposal_bank_sha256": bank_sha256,
                "candidate_source_policy": contract["candidate_source_policy"],
                "candidate_source_gate_independent": True,
                **execution,
                "frames_executed": 18,
                "collision": 0.0,
                "route_completion": 1.0,
                "episode_reward": value,
                "driving_distance": value * 10.0,
                "avg_speed": value,
                "candidate_queries": 1,
                "issued_queries": 1,
                "query_gate_rejections": 0,
                "scheduled_timeouts": 0,
                "timeouts": 0,
                "failure_events": 0,
                "release_events": 1,
                "pending_at_episode_end": 0,
                "pending_timeouts_at_episode_end": 0,
                "snapshot_count": 1,
                "component_ablation_arm": spec.name,
                "component_ablation_display_name": spec.display_name,
                "component_ablation_removed_components": ";".join(
                    spec.removed_components
                ),
            }
        )
    component_analyzer._write_csv(
        bundle / "component_ablation_episode_results.csv", result_rows
    )

    arms = []
    for spec in COMPONENT_ABLATION_ARMS:
        arm = asdict(spec)
        arm["removed_components"] = list(spec.removed_components)
        arms.append(arm)
    _write_json(
        bundle / "component_ablation_run_manifest.json",
        {
            "schema": COMPONENT_ABLATION_RUN_SCHEMA,
            "component_ablation_version": COMPONENT_ABLATION_VERSION,
            "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
            "method_version": submission["rgd_method_version"],
            "query_gate_method_version": submission["query_gate_method_version"],
            "release_contract_version": submission["release_contract_version"],
            "design": COMPONENT_ABLATION_DESIGN,
            "protocol_path": str(protocol_path.resolve()),
            "protocol_sha256": protocol_sha256,
            "source_root": str(tmp_path / "source"),
            "proposal_bank_sha256": bank_sha256,
            "latency_profile": "fixed",
            "fixed_latency_steps": 17,
            "predicted_latency_s": 1.7,
            "delay_s": [1.7],
            "candidate_source_policy": contract["candidate_source_policy"],
            "candidate_source_gate_independent": True,
            "seed_start": 6000,
            "seed_count": 1,
            "arms": arms,
            "query_gate_enabled": True,
            "release_guard_enabled": True,
            "randomized_block_run_order": [],
            "result_rows": 6,
            **execution,
        },
    )

    for spec in COMPONENT_ABLATION_ARMS:
        seed_dir = bundle / spec.name / "seed_6000"
        _write_json(seed_dir / "experiment_snapshot.json", {"config": {}})
        snapshot_root = seed_dir / "release_snapshots"
        snapshot_root.mkdir(parents=True)
        bundle_path = snapshot_root / "release_snapshots_synthetic_6000.pkl"
        bundle_path.write_bytes(f"synthetic:{spec.name}".encode("ascii"))
        snapshot_identity = hashlib.sha256(
            f"snapshot:{spec.name}:6000".encode("ascii")
        ).hexdigest()
        snapshot_manifest_path = (
            snapshot_root / "release_snapshots_synthetic_6000.json"
        )
        _write_json(
            snapshot_manifest_path,
            {
                "schema": RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
                "episode_id": 6000,
                "snapshot_count": 1,
                "bundle_file": bundle_path.name,
                "bundle_sha256": _sha256(bundle_path),
                "snapshots": [
                    {
                        "request_id": "seed-6000-request-0",
                        "snapshot_identity_sha256": snapshot_identity,
                    }
                ],
            },
        )
        events = [
            _synthetic_event(spec.name, frame, bank_sha256, snapshot_identity)
            for frame in range(18)
        ]
        _write_json(
            seed_dir / "event_logs" / "event_log_synthetic_6000.json",
            {
                "schema_version": FACTORIAL_EVENT_SCHEMA,
                "episode_id": 6000,
                "event_count": len(events),
                "pending_release_count": 0,
                "pending_releases_dropped_at_episode_end": [],
                "release_snapshot_count": 1,
                "release_snapshot_bundle": (
                    f"release_snapshots/{bundle_path.name}"
                ),
                "release_snapshot_manifest": (
                    f"release_snapshots/{snapshot_manifest_path.name}"
                ),
                "release_snapshot_bundle_sha256": _sha256(bundle_path),
                "events": events,
            },
        )

    monkeypatch.setattr(
        component_analyzer, "_process_component_cell", _synthetic_rollout_row
    )
    return bundle, artifact, protocol_path


def _strict_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite constant: {value}")
        ),
    )


def test_six_arm_analysis_and_verifier_close_from_raw_bundle(tmp_path, monkeypatch):
    bundle, artifact, protocol_path = _build_synthetic_bundle(tmp_path, monkeypatch)
    assert component_analyzer.main(
        [
            "--bundle",
            str(bundle),
            "--output-dir",
            str(artifact),
            "--workers",
            "1",
        ]
    ) == 0
    assert component_verifier.main(
        [
            "--artifact",
            str(artifact),
            "--protocol",
            str(protocol_path),
        ]
    ) == 0

    expected_files = {
        MANIFEST_FILE,
        AUDIT_FILE,
        *OUTPUT_FILES,
        "v13_component_ablation_verification.json",
    }
    assert expected_files <= {path.name for path in artifact.iterdir()}
    for name in (MANIFEST_FILE, AUDIT_FILE, "v13_component_ablation_verification.json"):
        assert _strict_json(artifact / name)

    effects = component_analyzer._read_csv(artifact / MAIN_EFFECTS_FILE)
    interaction = next(
        row
        for row in effects
        if row["effect"] == "h_x_n_interaction"
        and row["metric"] == "episode_reward"
    )
    assert float(interaction["estimate"]) == -2.0
    report = _strict_json(artifact / "v13_component_ablation_verification.json")
    assert report["h_x_n_interaction_estimates"]["episode_reward"] == -2.0

    events_path = artifact / EVENTS_FILE
    original_events = events_path.read_bytes()
    events_path.write_bytes(original_events + b"\n")
    with pytest.raises(ValueError, match="output hash drift"):
        component_verifier.verify(artifact, protocol_path)
    events_path.write_bytes(original_events)

    raw_snapshot = bundle / "full" / "seed_6000" / "experiment_snapshot.json"
    raw_snapshot.write_text('{"config":{},"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="input hash drift"):
        component_verifier.verify(artifact, protocol_path)
