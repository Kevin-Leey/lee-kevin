import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

import tools.analyze_factorial_latency_error as analyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = (
    PROJECT_ROOT / "results" / "rgd_factorial_confirmatory_20260731"
)
FORMAL_ANALYSIS = FORMAL_ROOT / "latency_error_analysis"


def _proposal(seed, source_frame, latency_steps, outcome, action):
    response_text = f"seed={seed};frame={source_frame};action={action}"
    return {
        "seed": seed,
        "source_frame": source_frame,
        "request_id": f"factorial:{seed}:{source_frame}",
        "raw_slow_action": action,
        "latency_steps": latency_steps,
        "outcome": outcome,
        "response_text": response_text,
        "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
    }


def _bank_payload(records_by_seed):
    return [
        {"seed": seed, "records": records}
        for seed, records in sorted(records_by_seed.items())
    ]


def _write_bank(root, profile, payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    manifest = {
        "latency_profile": profile,
        "bank_sha256": hashlib.sha256(encoded).hexdigest(),
        "seed_count": len(payload),
        "proposal_count": sum(len(block["records"]) for block in payload),
        "bank_payload": payload,
    }
    root.mkdir(parents=True)
    (root / "proposal_bank_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _lifecycle_record(record):
    outcome = record["outcome"]
    lifecycle = {
        "closed_loop_latency_request_id": record["request_id"],
        "closed_loop_latency_issuance_event": True,
        "closed_loop_latency_source_frame": record["source_frame"],
        "closed_loop_latency_issued_response_outcome": outcome,
        "closed_loop_latency_terminal_event": True,
    }
    if outcome == "valid":
        lifecycle.update(
            {
                "closed_loop_latency_release_event": True,
                "closed_loop_latency_realized_steps": record["latency_steps"],
                "release_selected_action": record["raw_slow_action"],
                "release_fast_comparator_action": 1,
                "release_selection_distinct": record["raw_slow_action"] != 1,
                "closed_loop_release_actuation_distinct": (
                    record["raw_slow_action"] != 1
                ),
                "closed_loop_release_action_alignment_pass": True,
                "closed_loop_latency_terminal_outcome": "released",
            }
        )
    elif outcome == "timeout":
        lifecycle.update(
            {
                "closed_loop_latency_timeout_event": True,
                "closed_loop_latency_terminal_outcome": "timeout",
            }
        )
    else:
        lifecycle.update(
            {
                "closed_loop_latency_failure_event": True,
                "closed_loop_latency_terminal_outcome": "failure",
            }
        )
    return {
        "frame_id": record["source_frame"] + record["latency_steps"],
        "rgd_subordinate_diagnostics": {"release_lifecycle": lifecycle},
    }


def _write_stress_arm_inputs(root, payload, predicted_steps=17):
    episode_rows = []
    for arm in analyzer.ARMS:
        for block in payload:
            seed = block["seed"]
            seed_root = root / arm / f"seed_{seed}"
            trace_root = seed_root / f"ep_{seed}"
            trace_root.mkdir(parents=True)
            (seed_root / "runtime_manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "closed_loop_latency_replay": {
                                "delay_steps": predicted_steps
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            trace = {
                "analysis_records": [
                    _lifecycle_record(record) for record in block["records"]
                ]
            }
            (trace_root / "synthetic_reasoning_records.json").write_text(
                json.dumps(trace), encoding="utf-8"
            )
            released = sum(record["outcome"] == "valid" for record in block["records"])
            episode_rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "issued_queries": len(block["records"]),
                    "release_events": released,
                    "primitive_distinct_selections": released,
                    "collision": 0,
                    "success_rate": 1,
                    "route_completion": 1,
                    "episode_reward": 10 + seed,
                    "driving_distance": 100 + seed,
                    "avg_speed": 20,
                }
            )
    with (root / "factorial_episode_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_compare_banks_accepts_only_latency_and_outcome_changes():
    frozen = [_proposal(1, 10, 17, "valid", 4)]
    stress = [dict(frozen[0], latency_steps=35, outcome="timeout")]

    rows = analyzer.compare_banks(frozen, stress)

    assert rows == [
        {
            "request_id": "factorial:1:10",
            "seed": 1,
            "source_frame": 10,
            "same_proposal_payload": 1,
            "frozen_latency_steps": 17,
            "stress_latency_steps": 35,
            "frozen_outcome": "valid",
            "stress_outcome": "timeout",
        }
    ]


@pytest.mark.parametrize("field", ["raw_slow_action", "response_text", "response_sha256"])
def test_compare_banks_rejects_proposal_payload_drift(field):
    frozen = [_proposal(1, 10, 17, "valid", 4)]
    stress = [dict(frozen[0])]
    stress[0][field] = "tampered" if field != "raw_slow_action" else 2

    with pytest.raises(ValueError, match="proposal payload drift"):
        analyzer.compare_banks(frozen, stress)


def test_compare_banks_rejects_request_identity_drift():
    frozen = [_proposal(1, 10, 17, "valid", 4)]
    stress = [dict(frozen[0], request_id="different-request")]

    with pytest.raises(ValueError, match="request ids differ"):
        analyzer.compare_banks(frozen, stress)


@pytest.mark.parametrize(
    ("event", "bank_outcome", "expected"),
    [
        ({}, "valid", "not_issued"),
        ({"issued": True, "released": True, "timeout": True}, "timeout", "released"),
        ({"issued": True, "timeout": True}, "valid", "timeout"),
        ({"issued": True}, "timeout", "timeout"),
        ({"issued": True, "failure": True}, "valid", "failure"),
        ({"issued": True}, "failure", "failure"),
        ({"issued": True}, "valid", "pending"),
    ],
)
def test_terminal_status_classification_is_fail_closed(event, bank_outcome, expected):
    assert analyzer.classify_status(bank_outcome, event) == expected


def test_prediction_error_direction_and_magnitude_boundaries():
    assert analyzer.error_direction(-1) == "earlier_than_predicted"
    assert analyzer.error_direction(0) == "matched_prediction"
    assert analyzer.error_direction(1) == "later_than_predicted"
    assert analyzer.error_magnitude_bin(-0.5) == "abs_error_le_0.5s"
    assert analyzer.error_magnitude_bin(0.5001) == "abs_error_0.5_to_1.0s"
    assert analyzer.error_magnitude_bin(-1.0) == "abs_error_0.5_to_1.0s"
    assert analyzer.error_magnitude_bin(1.0001) == "abs_error_gt_1.0s"


def test_seed_block_bootstrap_is_deterministic_and_uses_seed_aggregates():
    seed_rows = [
        {
            "arm": "full",
            "seed": 1,
            "latency": "early",
            "candidates": 10,
            "issued": 2,
            "released": 1,
            "timeouts": 1,
            "failures": 0,
            "pending": 0,
            "distinct_selections": 1,
            "distinct_actuations": 1,
        },
        {
            "arm": "full",
            "seed": 2,
            "latency": "early",
            "candidates": 1,
            "issued": 1,
            "released": 1,
            "timeouts": 0,
            "failures": 0,
            "pending": 0,
            "distinct_selections": 0,
            "distinct_actuations": 0,
        },
    ]

    first = analyzer.summarize_strata(
        seed_rows, "latency", draws=200, bootstrap_seed=41
    )
    second = analyzer.summarize_strata(
        seed_rows, "latency", draws=200, bootstrap_seed=41
    )
    expected_ci = analyzer.bootstrap_ratio(
        seed_rows,
        "issued",
        "candidates",
        draws=200,
        rng=random.Random(41),
    )

    assert first == second
    assert len(first) == 1
    assert first[0]["n_seeds_with_candidates"] == 2
    assert first[0]["issue_rate"] == pytest.approx(3 / 11)
    assert (
        first[0]["issue_rate_ci_low"],
        first[0]["issue_rate_ci_high"],
    ) == expected_ci


def test_cli_writes_complete_analysis_from_a_synthetic_paired_bundle(
    tmp_path, monkeypatch
):
    stress_root = tmp_path / "stress"
    frozen_root = tmp_path / "frozen"
    output_dir = tmp_path / "analysis"
    stress_records = {
        11: [
            _proposal(11, 0, 12, "valid", 4),
            _proposal(11, 20, 27, "timeout", 1),
            _proposal(11, 40, 17, "failure", 2),
        ],
        12: [
            _proposal(12, 0, 5, "valid", 0),
            _proposal(12, 20, 22, "valid", 1),
            _proposal(12, 40, 35, "timeout", 3),
        ],
    }
    stress_payload = _bank_payload(stress_records)
    frozen_payload = _bank_payload(
        {
            seed: [dict(record, latency_steps=17, outcome="valid") for record in records]
            for seed, records in stress_records.items()
        }
    )
    _write_bank(stress_root, "stress", stress_payload)
    _write_bank(frozen_root, "frozen", frozen_payload)
    _write_stress_arm_inputs(stress_root, stress_payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_factorial_latency_error.py",
            "--stress-root",
            str(stress_root),
            "--frozen-root",
            str(frozen_root),
            "--output-dir",
            str(output_dir),
            "--bootstrap-draws",
            "80",
            "--bootstrap-seed",
            "23",
        ],
    )

    assert analyzer.main() == 0

    expected_outputs = {
        "latency_error_analysis_manifest.json",
        "proposal_bank_comparison.csv",
        "stress_request_lifecycle.csv",
        "stress_latency_seed_strata.csv",
        "stress_latency_stratified_summary.csv",
        "stress_error_direction_seed_strata.csv",
        "stress_error_direction_summary.csv",
        "stress_error_magnitude_seed_strata.csv",
        "stress_error_magnitude_summary.csv",
        "stress_seed_exposure_outcomes.csv",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_outputs
    manifest = json.loads(
        (output_dir / "latency_error_analysis_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["accepted"] is True
    assert manifest["independent_unit"] == "simulator_seed"
    assert manifest["prediction"] == {
        "policy_frequency_hz": 10,
        "seconds": 1.7,
        "steps": 17,
    }
    assert manifest["proposal_count"] == 6
    assert manifest["seed_count"] == 2
    assert manifest["stress_schedule_counts"] == {
        "5": 1,
        "12": 1,
        "17": 1,
        "22": 1,
        "27": 1,
        "35": 1,
    }
    lifecycle = _read_csv(output_dir / "stress_request_lifecycle.csv")
    assert len(lifecycle) == len(analyzer.ARMS) * 6
    matched_failure = next(
        row
        for row in lifecycle
        if row["arm"] == "full" and row["request_id"] == "factorial:11:40"
    )
    assert matched_failure["signed_error_steps"] == "0"
    assert matched_failure["error_direction"] == "matched_prediction"
    assert matched_failure["terminal_status"] == "failure"


def test_formal_analysis_artifacts_are_complete_and_bound_to_their_inputs():
    expected_outputs = {
        "latency_error_analysis_manifest.json",
        "proposal_bank_comparison.csv",
        "stress_request_lifecycle.csv",
        "stress_latency_seed_strata.csv",
        "stress_latency_stratified_summary.csv",
        "stress_error_direction_seed_strata.csv",
        "stress_error_direction_summary.csv",
        "stress_error_magnitude_seed_strata.csv",
        "stress_error_magnitude_summary.csv",
        "stress_seed_exposure_outcomes.csv",
    }
    assert expected_outputs <= {path.name for path in FORMAL_ANALYSIS.iterdir()}

    manifest = json.loads(
        (FORMAL_ANALYSIS / "latency_error_analysis_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["accepted"] is True
    assert manifest["schema"] == "rgd_factorial_latency_error_analysis_v1"
    assert manifest["independent_unit"] == "simulator_seed"
    assert manifest["bootstrap"] == {
        "cluster": "simulator_seed",
        "confidence_level": 0.95,
        "draws": 20000,
        "method": "percentile cluster bootstrap",
        "seed": 20260801,
    }
    assert manifest["prediction"]["steps"] == 17
    assert manifest["proposal_count"] == 178
    assert manifest["seed_count"] == 30
    assert manifest["stress_schedule_counts"] == {
        "5": 34,
        "9": 29,
        "12": 31,
        "22": 29,
        "27": 25,
        "35": 30,
    }
    assert manifest["stress_outcome_counts"] == {"timeout": 35, "valid": 143}

    input_paths = {
        "stress_proposal_bank_manifest.json": FORMAL_ROOT
        / "stress_v5"
        / "proposal_bank_manifest.json",
        "stress_factorial_episode_results.csv": FORMAL_ROOT
        / "stress_v5"
        / "factorial_episode_results.csv",
        "frozen_proposal_bank_manifest.json": FORMAL_ROOT
        / "frozen_v5"
        / "proposal_bank_manifest.json",
    }
    assert manifest["input_sha256"] == {
        name: analyzer.sha256(path) for name, path in input_paths.items()
    }

    comparison = _read_csv(FORMAL_ANALYSIS / "proposal_bank_comparison.csv")
    assert len(comparison) == 178
    assert len({row["request_id"] for row in comparison}) == 178
    assert {row["same_proposal_payload"] for row in comparison} == {"1"}

    lifecycle = _read_csv(FORMAL_ANALYSIS / "stress_request_lifecycle.csv")
    assert len(lifecycle) == len(analyzer.ARMS) * 178
    by_arm = Counter(row["arm"] for row in lifecycle)
    assert by_arm == Counter({arm: 178 for arm in analyzer.ARMS})
    lifecycle_totals = {
        arm: {
            field: sum(int(row[field]) for row in lifecycle if row["arm"] == arm)
            for field in ("issued", "released", "timeout", "failure", "pending")
        }
        for arm in analyzer.ARMS
    }
    assert lifecycle_totals == {
        "full": {
            "issued": 15,
            "released": 13,
            "timeout": 2,
            "failure": 0,
            "pending": 0,
        },
        "query_only": {
            "issued": 16,
            "released": 13,
            "timeout": 3,
            "failure": 0,
            "pending": 0,
        },
        "release_only": {
            "issued": 178,
            "released": 143,
            "timeout": 35,
            "failure": 0,
            "pending": 0,
        },
        "neither": {
            "issued": 178,
            "released": 143,
            "timeout": 35,
            "failure": 0,
            "pending": 0,
        },
    }

    assert len(_read_csv(FORMAL_ANALYSIS / "stress_seed_exposure_outcomes.csv")) == 120
    assert len(_read_csv(FORMAL_ANALYSIS / "stress_latency_stratified_summary.csv")) == 24
    assert len(_read_csv(FORMAL_ANALYSIS / "stress_error_direction_summary.csv")) == 8
    assert len(_read_csv(FORMAL_ANALYSIS / "stress_error_magnitude_summary.csv")) == 12
