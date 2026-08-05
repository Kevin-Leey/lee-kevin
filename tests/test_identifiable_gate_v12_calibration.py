import csv
import hashlib
import json
import math
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from tools.calibrate_identifiable_gate_v12 import (
    ARM_FULL,
    ARM_NO_A,
    ARM_NO_H,
    ARM_NO_L,
    ARMS,
    BranchLabels,
    CandidateResult,
    Opportunity,
    OpportunityTable,
    ArmMetrics,
    Thresholds,
    VALIDATION_ARMS,
    arm_metrics,
    candidates,
    derive_relative_support_maneuver_breadth,
    latency_survival,
    load_branch_labels,
    load_spec,
    locked_acceptance,
    locked_bootstrap,
    locked_partition_spec,
    parse_v12_record,
    parse_args,
    run_locked_analysis,
    schedule_mask,
    select_candidate,
    validate_protocol_contract,
    DEFAULT_PROTOCOL_PATH,
)


def small_spec(**overrides):
    base = load_spec()
    defaults = {
        "seeds": (2000,),
        "delay_steps": (7,),
        "delay_seconds": (0.7,),
        "floor_units": (10, 20),
        "i_floor_units": 20,
        "exposure_min": 1,
        "exposure_max": 6,
        "min_changed_opportunities": 1,
        "min_changed_seeds": 1,
        "min_changed_union_fraction": Fraction(0, 1),
        "max_changed_jaccard": Fraction(1, 1),
        "min_component_levels": 1,
        "min_component_spread": 0.0,
    }
    defaults.update(overrides)
    return replace(base, **defaults)


def v12_record(frame=0):
    costs = {0: 0.5, 1: 0.2, 3: 0.3, 4: 0.8}
    relative_support_breadth = (1.0 + math.exp(-1.0)) / 3.0
    relative_headroom = (0.5 - 0.2) / 0.55
    gate = {
        "method_version": "identifiable_gate_v12",
        "gate_action_universe": [0, 1, 3, 4],
        "fast_executor_action_universe": [0, 1, 3, 4],
        "gate_action_universe_source": "fast_executor_contract",
        "fast_executor_action_universe_source": "fast_stack_union",
        "gate_domain_valid": True,
        "gate_fail_closed": False,
        "raw_cost_complete": True,
        "hold_action": 0,
        "absolute_alternative_count": 2,
        "absolute_alternative_feasible": True,
        "viable_cost_threshold": 0.55,
        "alternative_viable_ratio": relative_support_breadth,
        "relative_support_weighted_maneuver_family_breadth": relative_support_breadth,
        "alternative_maneuver_family_count": 2,
        "alternative_maneuver_family_total": 3,
        "action_maneuver_family_mapping": {
            0: "lateral-left",
            1: "lane-hold",
            3: "longitudinal-accelerate",
            4: "longitudinal-decelerate",
        },
        "raw_feasible_alternative_actions": [1, 3],
        "support_breadth_temperature": 0.10,
        "support_breadth_temperature_source": "identifiable_gate_v12.fixed_T_A",
        "support_breadth_formula": "sum_exp(-(s_m-s_star)/T_A)/num_all_alternative_families",
        "support_cost_complete": True,
        "support_family_min_costs": {
            "lane-hold": 0.2,
            "longitudinal-accelerate": 0.3,
        },
        "support_best_family_cost": 0.2,
        "support_weighted_family_mass": 1.0 + math.exp(-1.0),
        "relative_corrective_headroom": relative_headroom,
        "cost_headroom": relative_headroom,
        "corrective_headroom_kappa": 0.55,
        "corrective_headroom_kappa_source": "recoverable_cost_threshold",
        "corrective_advantage_raw": 0.3,
        "need_state_hazard": 0.7,
        "need_pre_screen_hazard": 0.2,
        "need_score": 0.7,
        "alternative_metric_source": "relative_support_weighted_maneuver_family_breadth",
        "headroom_metric_source": "incumbent_relative_action_recovery_cost_margin",
        "need_metric_source": "state_hazard_and_pre_screen_only",
        "critical_latency_seconds": 3.0,
        "latency_prediction_available": True,
        "llm_backed_execution_available": True,
        "latency_source": "unit_test_kinematics",
        "policy_frequency": 10.0,
        "safety_reserve_seconds": 0.0,
        "effective_delay_steps": 17,
    }
    return {
        "frame_id": frame,
        "predicted_action_id": 0,
        "schema_version": "rgd_record_v4",
        "rgd_method_version": "identifiable_gate_v12",
        "rgd_subordinate_diagnostics": {
            "recoverability_signal": {"recoverability_gate": gate},
            "ambiguity_and_conflict": {
                "route_ambiguity_profile": {
                    "action_recovery_costs": dict(costs),
                    "action_support_ranking_costs": dict(costs),
                    "probability_cost_source": "action_support_ranking_costs",
                }
            },
        },
    }


def opportunity_table(frames, *, episode_frames=100):
    rows = tuple(
        Opportunity(
            index=index,
            seed=2000,
            delay_steps=7,
            delay_s=0.7,
            query_frame=frame,
            release_frame=frame + 7,
            episode_frames=episode_frames,
            evaluable=frame + 7 + 20 <= episode_frames,
        )
        for index, frame in enumerate(frames)
    )
    size = len(rows)
    return OpportunityTable(
        rows=rows,
        permanent_f=np.ones(size, dtype=bool),
        l=np.ones(size),
        a=np.ones(size),
        h=np.ones(size),
        i=np.ones(size),
        groups={(2000, 7): np.arange(size, dtype=np.int32)},
    )


def metric(*, q, r, c):
    return ArmMetrics(
        q=q,
        r=r,
        c=c,
        excluded=q - r,
        rate=Fraction(c, r),
        q_over_c=Fraction(q, c) if c else None,
        r_over_c=Fraction(r, c) if c else None,
        seed_macro_rate=Fraction(c, r),
        seed_macro_valid_seeds=1,
        scheduled_hash="s",
        evaluated_hash="e",
        q_by_delay=((7, q),),
    )


def candidate_result(thresholds, *, margin, q_over_c):
    arms = {arm: metric(q=10, r=10, c=5) for arm in ARMS}
    return CandidateResult(
        thresholds=thresholds,
        feasible=True,
        failure_reasons=(),
        arms=arms,
        margins={"L": margin, "A": margin, "H": margin},
        min_margin=margin,
        q_over_c=q_over_c,
        changed={},
        max_changed_jaccard=Fraction(0, 1),
    )


def branch_manifest_stub(path=None):
    manifest = {
        "schema": "v12_branch_runner_manifest_v1",
        "method_version": "identifiable_gate_v12",
        "label_source": "matched_release_state_exact_action_rollout_v1",
        "exact_action_provenance": "exact",
    }
    if path is not None:
        raw = path.read_bytes()
        with path.open(newline="", encoding="utf-8") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        manifest["output_hashes"] = {
            path.name: hashlib.sha256(raw).hexdigest(),
        }
        manifest["counts"] = {"target_event_rows": row_count}
    return manifest


def valid_label_row():
    return {
        "seed": 2000,
        "delay_s": 0.7,
        "query_frame": 0,
        "release_frame": 7,
        "delay_steps": 7,
        "candidate_state_id": "2000:0:7",
        "release_state_id": "2000:7",
        "release_state_identity_sha256": "a" * 64,
        "method_version": "identifiable_gate_v12",
        "label_source": "matched_release_state_exact_action_rollout_v1",
        "exact_action_provenance": 1,
        "horizon_steps": 20,
        "gamma": 0.99,
        "epsilon": 0.02,
        "corrective_set_nonempty": 1,
    }


def test_static_lock_has_disjoint_calibration_go_no_go_and_holdout_blocks():
    spec = load_spec()
    assert spec.seeds == tuple(range(2000, 2040))
    assert locked_partition_spec(spec, "go_no_go").seeds == tuple(range(2040, 2060))
    assert locked_partition_spec(spec, "validation").seeds == tuple(range(3000, 3030))
    assert spec.minimum_query_frame_gap == spec.cooldown_complete_frames + 1 == 21
    assert len(candidates(spec)) == 18**3
    assert {row.i for row in candidates(spec)} == {20}
    validate_protocol_contract(DEFAULT_PROTOCOL_PATH, spec)


def test_holdout_cli_carries_the_single_authorization_through_targets_and_analysis():
    targets = parse_args(
        [
            "locked-targets",
            "--partition",
            "validation",
            "--trace-root",
            "traces",
            "--calibration-manifest",
            "calibration.json",
            "--go-no-go-manifest",
            "go.json",
            "--holdout-authorization",
            "authorization.json",
            "--output",
            "targets.json",
        ]
    )
    analysis = parse_args(
        [
            "validate-locked",
            "--trace-root",
            "traces",
            "--branch-labels",
            "labels.csv",
            "--branch-manifest",
            "branch.json",
            "--calibration-manifest",
            "calibration.json",
            "--go-no-go-manifest",
            "go.json",
            "--holdout-authorization",
            "authorization.json",
            "--output-dir",
            "analysis",
        ]
    )

    assert targets.holdout_authorization.name == "authorization.json"
    assert analysis.holdout_authorization == targets.holdout_authorization


def test_holdout_analysis_rejects_missing_authorization_before_reading_inputs():
    with pytest.raises(ValueError, match="requires one-shot holdout authorization"):
        run_locked_analysis(
            SimpleNamespace(holdout_authorization=None),
            "validation",
        )


def test_v12_parser_uses_exact_domains_and_reconstructs_each_delay():
    spec = small_spec(delay_steps=(7, 17, 27), delay_seconds=(0.7, 1.7, 2.7))
    feature = parse_v12_record(
        json.loads(json.dumps(v12_record())), seed=2000, episode_frames=100
    )

    assert feature.permanent_f is True
    assert feature.a == pytest.approx((1.0 + math.exp(-1.0)) / 3.0)
    assert feature.h == pytest.approx((0.5 - 0.2) / 0.55)
    assert feature.i == pytest.approx(0.7)
    survivals = [latency_survival(feature, step, spec) for step in spec.delay_steps]
    assert survivals[0] > survivals[1] > survivals[2] >= 0.0


def test_v12_parser_rejects_legacy_method_and_conflicting_domains():
    legacy = v12_record()
    legacy["rgd_method_version"] = "support_breadth_v11"
    with pytest.raises(ValueError, match="conflicting aliases|not v12"):
        parse_v12_record(legacy, seed=2000, episode_frames=100)

    mismatch = v12_record()
    gate = mismatch["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    gate["fast_executor_action_universe"] = [0, 1, 3]
    with pytest.raises(ValueError, match="universes differ"):
        parse_v12_record(mismatch, seed=2000, episode_frames=100)


def test_fail_closed_record_remains_a_permanent_f_negative_not_a_trace_abort():
    record = v12_record()
    gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    profile = record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"]["route_ambiguity_profile"]
    gate["gate_domain_valid"] = False
    gate["gate_fail_closed"] = True
    profile["action_recovery_costs"][1] = float("nan")

    feature = parse_v12_record(record, seed=2000, episode_frames=100)
    assert feature.permanent_f is False


def test_support_failure_closes_a_without_contaminating_raw_only_f():
    record = v12_record()
    gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    profile = record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"]["route_ambiguity_profile"]
    gate["support_cost_complete"] = False
    profile["action_support_ranking_costs"].pop(3)

    feature = parse_v12_record(record, seed=2000, episode_frames=100)
    assert feature.permanent_f is True
    assert feature.a == 0.0


def test_relative_support_breadth_excludes_hold_and_raw_infeasible_costs():
    derived = derive_relative_support_maneuver_breadth(
        gate_actions=(0, 1, 3, 4),
        hold_action=0,
        recovery_costs={0: 0.5, 1: 0.2, 3: 0.3, 4: 0.8},
        support_costs={0: 0.0, 1: 0.2, 3: 0.3, 4: 0.0},
        maneuver_families={
            0: "lateral-left",
            1: "lane-hold",
            3: "longitudinal-accelerate",
            4: "longitudinal-decelerate",
        },
        viable_cost_threshold=0.55,
        temperature=0.10,
    )

    assert derived["relative_support_best_cost"] == pytest.approx(0.2)
    assert derived["raw_feasible_alternative_actions"] == (1, 3)
    assert derived["all_alternative_families"] == (
        "lane-hold",
        "longitudinal-accelerate",
        "longitudinal-decelerate",
    )
    assert derived["value"] == pytest.approx((1.0 + math.exp(-1.0)) / 3.0)


def test_v12_parser_rejects_old_a_source_and_nonderivable_export():
    old_source = v12_record()
    gate = old_source["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    gate["alternative_metric_source"] = "raw_feasible_maneuver_family_breadth"
    with pytest.raises(ValueError, match="A source"):
        parse_v12_record(old_source, seed=2000, episode_frames=100)

    tampered = v12_record()
    gate = tampered["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    gate["relative_support_weighted_maneuver_family_breadth"] += 0.1
    gate["alternative_viable_ratio"] += 0.1
    with pytest.raises(ValueError, match="not independently derivable"):
        parse_v12_record(tampered, seed=2000, episode_frames=100)


def test_complete_raw_domain_with_no_feasible_alternative_is_valid_f_false():
    record = v12_record()
    gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    profile = record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"]["route_ambiguity_profile"]
    raw_costs = {0: 0.2, 1: 0.7, 3: 0.8, 4: 0.9}
    profile["action_recovery_costs"] = raw_costs
    gate.update(
        {
            "absolute_alternative_count": 0,
            "absolute_alternative_feasible": False,
            "alternative_viable_ratio": 0.0,
            "relative_support_weighted_maneuver_family_breadth": 0.0,
            "alternative_maneuver_family_count": 0,
            "raw_feasible_alternative_actions": [],
                "support_family_min_costs": {},
                "support_best_family_cost": 1.0,
                "support_weighted_family_mass": 0.0,
                "relative_corrective_headroom": 0.0,
                "cost_headroom": 0.0,
                "corrective_advantage_raw": 0.0,
            }
        )

    feature = parse_v12_record(record, seed=2000, episode_frames=100)
    assert feature.permanent_f is False
    assert feature.a == 0.0

    gate["support_best_family_cost"] = 0.0
    with pytest.raises(ValueError, match="best-family cost drift"):
        parse_v12_record(record, seed=2000, episode_frames=100)


@pytest.mark.parametrize(
    "bad_action",
    (1.5, True, "01", "1", 5),
)
def test_action_sequence_rejects_noncanonical_or_out_of_range_ids(bad_action):
    record = v12_record()
    gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    gate["gate_action_universe"] = [0, bad_action, 3, 4]
    with pytest.raises(ValueError, match="canonical action id|outside|boolean"):
        parse_v12_record(record, seed=2000, episode_frames=100)


@pytest.mark.parametrize("bad_hold", (1.0, True, "01", 5))
def test_hold_rejects_noncanonical_or_out_of_range_ids(bad_hold):
    record = v12_record()
    gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    gate["hold_action"] = bad_hold
    with pytest.raises(ValueError, match="canonical action id|outside|boolean"):
        parse_v12_record(record, seed=2000, episode_frames=100)


@pytest.mark.parametrize("bad_key", ("01", "5", 1.5, True))
def test_cost_map_rejects_noncanonical_action_keys(bad_key):
    record = v12_record()
    profile = record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"]["route_ambiguity_profile"]
    profile["action_recovery_costs"] = {0: 0.5, bad_key: 0.2, 3: 0.3, 4: 0.8}
    with pytest.raises(ValueError, match="canonical action id|outside|boolean"):
        parse_v12_record(record, seed=2000, episode_frames=100)


def test_maneuver_family_mapping_is_canonical_not_trace_defined():
    record = v12_record()
    gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
    gate["action_maneuver_family_mapping"][0] = "lane-hold"
    with pytest.raises(ValueError, match="mapping drift"):
        parse_v12_record(record, seed=2000, episode_frames=100)


def test_online_scheduler_requires_twenty_complete_intermediate_frames():
    spec = small_spec()
    table = opportunity_table([0, 19, 20, 21, 41, 42])
    cohort = schedule_mask(table, np.ones(len(table.rows), dtype=bool), spec)

    assert [table.rows[index].query_frame for index in cohort.scheduled] == [0, 21, 42]


def test_q_over_c_counts_boundary_exclusions_but_cset_uses_evaluated_r():
    spec = small_spec()
    table = opportunity_table([0, 21, 78], episode_frames=100)
    cohort = schedule_mask(table, np.ones(len(table.rows), dtype=bool), spec)
    keys = {table.rows[index].event_key: True for index in cohort.evaluated}
    labels = BranchLabels(keys, "semantic", "raw", (), ())
    observed = arm_metrics(table, cohort, labels, spec)

    assert (observed.q, observed.r, observed.c, observed.excluded) == (3, 2, 2, 1)
    assert observed.rate == Fraction(1, 1)
    assert observed.q_over_c == Fraction(3, 2)
    assert observed.r_over_c == Fraction(1, 1)


def test_selector_uses_margin_then_qc_then_stricter_floor_tuple():
    loose = Thresholds(10, 10, 10, 20)
    strict = Thresholds(20, 20, 20, 20)
    margin_wins = candidate_result(loose, margin=Fraction(1, 10), q_over_c=Fraction(9, 1))
    qc_loses = candidate_result(strict, margin=Fraction(9, 100), q_over_c=Fraction(1, 1))
    assert select_candidate([qc_loses, margin_wins]) is margin_wins

    better_qc = candidate_result(loose, margin=Fraction(1, 10), q_over_c=Fraction(2, 1))
    worse_qc = candidate_result(strict, margin=Fraction(1, 10), q_over_c=Fraction(3, 1))
    assert select_candidate([worse_qc, better_qc]) is better_qc

    tied_loose = candidate_result(loose, margin=Fraction(-1, 100), q_over_c=Fraction(2, 1))
    tied_strict = candidate_result(strict, margin=Fraction(-1, 100), q_over_c=Fraction(2, 1))
    assert select_candidate([tied_loose, tied_strict]) is tied_strict


def test_branch_labels_require_an_exact_unique_preregistered_event_join(tmp_path):
    spec = small_spec()
    path = tmp_path / "labels.csv"
    fields = [
        "seed", "delay_s", "query_frame", "release_frame", "delay_steps",
        "candidate_state_id", "release_state_id", "release_state_identity_sha256",
        "method_version", "label_source", "exact_action_provenance",
        "horizon_steps", "gamma", "epsilon", "corrective_set_nonempty",
    ]
    rows = [valid_label_row()]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    labels = load_branch_labels(
        path,
        required_keys=[(2000, 7, 0, 7)],
        spec=spec,
        branch_manifest=branch_manifest_stub(path),
    )
    assert labels.labels[(2000, 7, 0, 7)] is True

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(rows[0])
    with pytest.raises(ValueError, match="duplicate"):
        load_branch_labels(
            path,
            required_keys=[(2000, 7, 0, 7)],
            spec=spec,
            branch_manifest=branch_manifest_stub(path),
        )


def test_branch_labels_reject_a_label_file_changed_after_manifest(tmp_path):
    path = tmp_path / "labels.csv"
    row = valid_label_row()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    manifest = branch_manifest_stub(path)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="branch manifest label hash drift"):
        load_branch_labels(
            path,
            required_keys=[(2000, 7, 0, 7)],
            spec=small_spec(),
            branch_manifest=manifest,
        )


@pytest.mark.parametrize(
    "field",
    (
        "method_version",
        "label_source",
        "exact_action_provenance",
        "horizon_steps",
        "gamma",
        "epsilon",
        "release_state_identity_sha256",
    ),
)
def test_branch_labels_reject_missing_required_provenance(tmp_path, field):
    row = valid_label_row()
    row.pop(field)
    path = tmp_path / f"missing_{field}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="required provenance missing"):
        load_branch_labels(
            path,
            required_keys=[(2000, 7, 0, 7)],
            spec=small_spec(),
            branch_manifest=branch_manifest_stub(path),
        )


def test_go_no_go_never_claims_validation_and_holdout_acceptance_is_strict():
    spec = load_spec()
    full = metric(q=16, r=10, c=8)
    comparator = metric(q=21, r=10, c=7)
    metrics = {arm.label: comparator for arm in VALIDATION_ARMS}
    metrics[ARM_FULL] = full
    geometry = {
        "passed": True,
        "components": {name: {"passed": True} for name in ("L", "A", "H")},
    }
    exposure = {"passed": True}
    bootstrap = {
        "cset_margin_simultaneous_minimum_lower": 0.01,
        "corrective_yield_per_seed_lower": {
            ARM_NO_L: -0.05,
            ARM_NO_A: -0.05,
            ARM_NO_H: -0.05,
        },
    }

    go = locked_acceptance(metrics, bootstrap, geometry, exposure, partition="go_no_go", spec=spec)
    assert go["passed"] is True
    assert go["validation_evaluated"] is False
    assert go["paper_facing_passed"] is False

    validation = locked_acceptance(metrics, bootstrap, geometry, exposure, partition="validation", spec=spec)
    assert validation["validation_evaluated"] is True
    assert validation["validation_passed"] is True
    assert validation["full_cset_strictly_greater_than_all_seven_arms"] is True
    assert validation["full_Q_over_C_strictly_less_than_all_seven_arms"] is True

    tied = dict(metrics, **{ARM_NO_L: full})
    rejected = locked_acceptance(tied, bootstrap, geometry, exposure, partition="validation", spec=spec)
    assert rejected["passed"] is False


def test_locked_bootstrap_resamples_seed_clusters_and_preserves_pairing():
    spec = small_spec(seeds=(2000, 2001))
    counts = {}
    for arm in VALIDATION_ARMS:
        corrective = 4 if arm.label == ARM_FULL else 3
        counts[arm.label] = {
            2000: (6, 5, corrective),
            2001: (6, 5, corrective),
        }

    result = locked_bootstrap(counts, spec)
    assert result["cluster_unit"] == "simulator_seed"
    assert result["cset_margin_simultaneous_minimum_lower"] == pytest.approx(0.2)
    for arm in (ARM_NO_L, ARM_NO_A, ARM_NO_H):
        assert result["corrective_yield_per_seed_lower"][arm] == pytest.approx(1.0)


def test_infeasible_candidate_set_fails_closed():
    row = candidate_result(Thresholds(10, 10, 10, 20), margin=Fraction(1, 10), q_over_c=Fraction(2, 1))
    row = replace(row, feasible=False, failure_reasons=("exposure",))
    with pytest.raises(RuntimeError, match="no preregistered"):
        select_candidate([row])
