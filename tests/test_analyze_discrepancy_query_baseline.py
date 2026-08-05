import csv
import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from dilu.evaluation.factorial_replay import (
    FACTORIAL_ARMS,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
    FACTORIAL_RUN_SCHEMA,
)
from dilu.evaluation.discrepancy_query_gate import (
    DISCREPANCY_ARTIFACT_SCHEMA,
    DISCREPANCY_GATE_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    canonical_json_sha256,
    feature_schema_payload,
)
from tools.analyze_discrepancy_query_baseline import (
    BASELINE_ARM_FLAGS,
    BASELINE_ARM_NAMES,
    DISCREPANCY_BASELINE_VERSION,
    DISCREPANCY_RUN_SCHEMA,
    PAIRED_CONTRASTS,
    analyze_discrepancy_comparison,
    main,
    validate_bundled_model_artifact,
    validate_discrepancy_bundle_contract,
)
from tools.analyze_query_release_factorial import (
    ARM_BY_NAME,
    ARM_NAMES,
    DISTINCT_ACTION_METRIC_STAGE,
    METRICS,
    validate_bundle_contract,
)


def make_model_artifact():
    source_rows = []
    for seed, label_read in ((1, True), (2, True), (3, False), (4, False)):
        digest = hashlib.sha256(str(seed).encode("ascii")).hexdigest()
        source_rows.append(
            {
                "seed": seed,
                "event_log": {"path": f"seed_{seed}/event.json", "sha256": digest},
                "reasoning_trace": {
                    "path": f"seed_{seed}/reasoning.json",
                    "sha256": digest,
                },
                "experiment_snapshot": {
                    "path": f"seed_{seed}/snapshot.json",
                    "sha256": digest,
                },
                "record_count": 2,
                "discrepancy_label_read": label_read,
            }
        )
    split = {
        "unit": "simulator_seed",
        "fit_seeds": [1, 2],
        "calibration_seeds": [3, 4],
        "disjoint": True,
        "fit_labels_use_slow_fast_query_discrepancy": True,
        "calibration_labels_use_slow_fast_query_discrepancy": False,
    }
    calibration = {
        "rule": "exact_rgd_exposure_count_midpoint_threshold_v1",
        "record_count": 4,
        "seed_count": 2,
        "target_policy": "rgd_serial_query_gate",
        "target_invocations": 2,
        "target_exposure_rate": 0.5,
        "threshold": 0.5,
        "comparison_operator": ">=",
        "achieved_invocations": 2,
        "achieved_exposure_rate": 0.5,
        "uses_outcomes": False,
        "uses_discrepancy_labels": False,
    }
    payload = {
        "schema": DISCREPANCY_ARTIFACT_SCHEMA,
        "version": DISCREPANCY_GATE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema": feature_schema_payload(),
        "routing_input_contract": {
            "query_time_only": True,
            "proposal_record_visible": False,
            "slow_response_visible": False,
            "slow_action_visible": False,
            "release_outcome_visible": False,
        },
        "training_source": {
            "policy": "scheduled_always_slow",
            "root": "synthetic",
            "source_content_sha256": canonical_json_sha256(source_rows),
            "artifacts": source_rows,
        },
        "seed_split": split,
        "fit": {
            "record_count": 4,
            "seed_count": 2,
            "class_counts": {"agreement": 2, "discrepancy": 2},
            "label": "query-time discrepancy",
            "seed_block_cross_validation": {
                "role": "fit_cohort_diagnostic_only",
                "used_for_model_selection": False,
                "splitter": "GroupKFold",
                "group_unit": "simulator_seed",
                "n_splits": 2,
                "record_count": 4,
                "seed_count": 2,
                "classification_threshold": 0.5,
                "roc_auc": 0.5,
                "average_precision": 0.5,
                "balanced_accuracy": 0.5,
                "brier_score": 0.25,
                "folds": [
                    {
                        "fold": 0,
                        "validation_seeds": [2],
                        "record_count": 2,
                        "class_counts": {"agreement": 1, "discrepancy": 1},
                        "roc_auc": 0.5,
                    },
                    {
                        "fold": 1,
                        "validation_seeds": [1],
                        "record_count": 2,
                        "class_counts": {"agreement": 1, "discrepancy": 1},
                        "roc_auc": 0.5,
                    },
                ],
            },
        },
        "model": {
            "pipeline": "StandardScaler->LogisticRegression",
            "sklearn_version": "test",
            "standard_scaler": {
                "parameters": {"copy": True, "with_mean": True, "with_std": True},
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "variance": [1.0] * len(FEATURE_NAMES),
                "samples_seen": 4,
            },
            "logistic_regression": {
                "parameters": {"solver": "liblinear"},
                "classes": [0, 1],
                "coefficients": [0.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
                "iterations": 1,
            },
        },
        "calibration": calibration,
    }
    payload["artifact_sha256"] = canonical_json_sha256(payload)
    return payload


MODEL_ARTIFACT = make_model_artifact()
MODEL_SHA256 = MODEL_ARTIFACT["artifact_sha256"]


def empty_bank_payload(seeds):
    payload = []
    for seed in sorted(seeds):
        records = []
        for ordinal, frame in enumerate((0, 21, 42)):
            response_text = f"response:{seed}:{frame}"
            records.append(
                {
                    "seed": int(seed),
                    "source_frame": frame,
                    "request_id": f"factorial:{seed}:{frame}:{ordinal:02d}",
                    "raw_slow_action": ordinal,
                    "latency_steps": 17,
                    "outcome": "valid",
                    "response_text": response_text,
                    "response_sha256": hashlib.sha256(
                        response_text.encode("utf-8")
                    ).hexdigest(),
                }
            )
        payload.append({"seed": int(seed), "records": records})
    return payload


def bank_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metric_values():
    return {
        "candidate_queries": 3,
        "issued_queries": 2,
        "query_gate_rejections": 1,
        "scheduled_timeouts": 0,
        "timeouts": 0,
        "failure_events": 0,
        "release_events": 2,
        "distinct_actuations": 1,
        "primitive_distinct_selections": 1,
        "distinct_action_metric_stage": DISTINCT_ACTION_METRIC_STAGE,
        "effect_distinctness_available": False,
        "pending_at_episode_end": 0,
        "pending_timeouts_at_episode_end": 0,
        "snapshot_count": 2,
        "collision": 0,
        "success_rate": 0.5,
        "route_completion": 0.8,
        "episode_reward": 10.0,
        "driving_distance": 100.0,
        "avg_speed": 20.0,
        "runtime_per_frame": 0.01,
    }


def make_core_row(seed, arm_name, proposal_hash, **overrides):
    arm = ARM_BY_NAME[arm_name]
    row = {
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "arm": arm_name,
        "query_gate_enabled": arm.query_gate_enabled,
        "release_guard_enabled": arm.release_guard_enabled,
        "seed": seed,
        "proposal_bank_sha256": proposal_hash,
        "aligned_distinct_actuations": int(arm.release_guard_enabled),
        **metric_values(),
    }
    row.update(overrides)
    return row


def make_baseline_row(seed, arm_name, proposal_hash, **overrides):
    flags = BASELINE_ARM_FLAGS[arm_name]
    row = {
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "discrepancy_baseline_version": DISCREPANCY_BASELINE_VERSION,
        "model_artifact_sha256": MODEL_SHA256,
        "arm": arm_name,
        "query_gate_enabled": flags["query_gate_enabled"],
        "release_guard_enabled": flags["release_guard_enabled"],
        "seed": seed,
        "proposal_bank_sha256": proposal_hash,
        "aligned_distinct_actuations": int(flags["release_guard_enabled"]),
        **metric_values(),
    }
    row.update(overrides)
    return row


def make_bundle_inputs(seeds=(10, 11)):
    payload = empty_bank_payload(seeds)
    proposal_hash = bank_sha256(payload)
    proposal_manifest = {
        "schema": FACTORIAL_PROPOSAL_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "latency_profile": "frozen",
        "bank_sha256": proposal_hash,
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "seed_count": len(seeds),
        "proposal_count": 3 * len(seeds),
        "bank_payload": payload,
    }
    core_rows = [
        make_core_row(seed, arm, proposal_hash)
        for seed in seeds
        for arm in ARM_NAMES
    ]
    core_run = {
        "schema": FACTORIAL_RUN_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "latency_profile": "frozen",
        "proposal_bank_sha256": proposal_hash,
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "seed_start": seeds[0],
        "seed_count": len(seeds),
        "result_rows": len(core_rows),
        "arms": [asdict(arm) for arm in FACTORIAL_ARMS],
        "randomized_block_run_order": [
            {"seed": seed, "order": order, "arm": arm}
            for seed in seeds
            for order, arm in enumerate(reversed(ARM_NAMES))
        ],
    }
    baseline_rows = [
        make_baseline_row(seed, arm, proposal_hash)
        for seed in seeds
        for arm in BASELINE_ARM_NAMES
    ]
    baseline_run = {
        "schema": DISCREPANCY_RUN_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "discrepancy_baseline_version": DISCREPANCY_BASELINE_VERSION,
        "discrepancy_gate_version": DISCREPANCY_GATE_VERSION,
        "discrepancy_gate_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_artifact_sha256": MODEL_SHA256,
        "latency_profile": "frozen",
        "proposal_bank_sha256": proposal_hash,
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "seed_start": seeds[0],
        "seed_count": len(seeds),
        "evaluation_seeds": list(seeds),
        "evaluation_training_seed_disjoint": True,
        "result_rows": len(baseline_rows),
        "arms": [
            {"name": name, **dict(BASELINE_ARM_FLAGS[name])}
            for name in BASELINE_ARM_NAMES
        ],
        "randomized_block_run_order": [
            {"seed": seed, "order": order, "arm": arm}
            for seed in seeds
            for order, arm in enumerate(reversed(BASELINE_ARM_NAMES))
        ],
        "model_artifact": {
            "path": "discrepancy_query_gate_model.json",
            "file_sha256": "b" * 64,
            "artifact_sha256": MODEL_SHA256,
            "training_source_content_sha256": MODEL_ARTIFACT["training_source"][
                "source_content_sha256"
            ],
            "seed_split": copy.deepcopy(MODEL_ARTIFACT["seed_split"]),
            "fit_class_counts": copy.deepcopy(
                MODEL_ARTIFACT["fit"]["class_counts"]
            ),
            "calibration": copy.deepcopy(MODEL_ARTIFACT["calibration"]),
        },
    }
    return core_rows, core_run, proposal_manifest, baseline_rows, baseline_run


def validate_inputs(inputs):
    core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
    core_contract = validate_bundle_contract(
        core_rows, core_run, proposal_manifest
    )
    baseline_contract = validate_discrepancy_bundle_contract(
        baseline_rows,
        baseline_run,
        proposal_manifest,
        core_contract=core_contract,
    )
    return core_contract, baseline_contract


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DiscrepancyBundleValidationTests(unittest.TestCase):
    def test_accepts_complete_bundle_matched_to_core(self):
        core_contract, baseline_contract = validate_inputs(make_bundle_inputs())

        self.assertEqual(baseline_contract["seeds"], core_contract["seeds"])
        self.assertEqual(
            baseline_contract["proposal_bank_sha256"],
            core_contract["proposal_bank_sha256"],
        )
        self.assertEqual(baseline_contract["model_artifact_sha256"], MODEL_SHA256)

    def test_bank_hash_and_seed_mismatches_fail_closed(self):
        inputs = make_bundle_inputs()
        core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
        core_contract = validate_bundle_contract(core_rows, core_run, proposal_manifest)

        baseline_run["proposal_bank_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "proposal bank does not match core"):
            validate_discrepancy_bundle_contract(
                baseline_rows,
                baseline_run,
                proposal_manifest,
                core_contract=core_contract,
            )

        inputs = make_bundle_inputs()
        core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
        core_contract = validate_bundle_contract(core_rows, core_run, proposal_manifest)
        baseline_run["seed_start"] = 11
        with self.assertRaisesRegex(ValueError, "seed cohort does not match core"):
            validate_discrepancy_bundle_contract(
                baseline_rows,
                baseline_run,
                proposal_manifest,
                core_contract=core_contract,
            )

    def test_arm_and_lifecycle_mismatches_fail_closed(self):
        inputs = make_bundle_inputs()
        core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
        core_contract = validate_bundle_contract(core_rows, core_run, proposal_manifest)
        baseline_run["arms"][0]["query_gate_enabled"] = False
        with self.assertRaisesRegex(ValueError, "manifest query flag mismatch"):
            validate_discrepancy_bundle_contract(
                baseline_rows,
                baseline_run,
                proposal_manifest,
                core_contract=core_contract,
            )

        inputs = make_bundle_inputs()
        core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
        core_contract = validate_bundle_contract(core_rows, core_run, proposal_manifest)
        baseline_rows[0]["candidate_queries"] = 4
        with self.assertRaisesRegex(ValueError, "candidate-query accounting mismatch"):
            validate_discrepancy_bundle_contract(
                baseline_rows,
                baseline_run,
                proposal_manifest,
                core_contract=core_contract,
            )

    def test_training_evaluation_overlap_fails_closed(self):
        inputs = make_bundle_inputs()
        core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
        core_contract = validate_bundle_contract(core_rows, core_run, proposal_manifest)
        baseline_run["model_artifact"]["seed_split"]["fit_seeds"][0] = 10

        with self.assertRaisesRegex(ValueError, "evaluation cohort overlaps"):
            validate_discrepancy_bundle_contract(
                baseline_rows,
                baseline_run,
                proposal_manifest,
                core_contract=core_contract,
            )

    def test_bundled_model_file_hash_drift_fails_closed(self):
        _, _, _, _, baseline_run = make_bundle_inputs()
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            path = bundle / "discrepancy_query_gate_model.json"
            model_text = json.dumps(MODEL_ARTIFACT)
            path.write_text(model_text, encoding="utf-8")
            baseline_run["model_artifact"]["file_sha256"] = hashlib.sha256(
                model_text.encode("utf-8")
            ).hexdigest()

            resolved, artifact = validate_bundled_model_artifact(bundle, baseline_run)
            self.assertEqual(resolved, path.resolve())
            self.assertEqual(artifact["artifact_sha256"], MODEL_SHA256)

            path.write_text(model_text + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file hash mismatch"):
                validate_bundled_model_artifact(bundle, baseline_run)


class DiscrepancyAnalysisTests(unittest.TestCase):
    def test_paired_contrasts_use_seed_blocks(self):
        inputs = make_bundle_inputs()
        core_rows, _, _, baseline_rows, _ = inputs
        core_reward = {
            (10, "query_only"): 10.0,
            (10, "full"): 15.0,
            (11, "query_only"): 20.0,
            (11, "full"): 25.0,
        }
        baseline_reward = {
            (10, "discrepancy_only"): 6.0,
            (10, "discrepancy_release"): 9.0,
            (11, "discrepancy_only"): 12.0,
            (11, "discrepancy_release"): 17.0,
        }
        for row in core_rows:
            if (row["seed"], row["arm"]) in core_reward:
                row["episode_reward"] = core_reward[(row["seed"], row["arm"])]
        for row in baseline_rows:
            row["episode_reward"] = baseline_reward[(row["seed"], row["arm"])]
        core_contract, baseline_contract = validate_inputs(inputs)

        analysis = analyze_discrepancy_comparison(
            core_contract["matrix"],
            baseline_contract["matrix"],
            seeds=core_contract["seeds"],
            draws=200,
            bootstrap_seed=19,
        )
        estimates = {
            row["effect"]: row["estimate"]
            for row in analysis["paired_contrasts"]
            if row["metric"] == "episode_reward"
        }
        self.assertEqual(estimates["query_only_minus_discrepancy_only"], 6.0)
        self.assertEqual(estimates["full_minus_discrepancy_release"], 7.0)
        self.assertEqual(
            estimates["discrepancy_release_minus_discrepancy_only"], 4.0
        )
        self.assertEqual(estimates["full_minus_discrepancy_only"], 11.0)

    def test_cli_writes_atomic_json_and_csv_outputs(self):
        inputs = make_bundle_inputs()
        core_rows, core_run, proposal_manifest, baseline_rows, baseline_run = inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            factorial_bundle = root / "factorial"
            baseline_bundle = root / "baseline"
            output_dir = root / "analysis"
            factorial_bundle.mkdir()
            baseline_bundle.mkdir()
            write_csv(
                factorial_bundle / "factorial_episode_results.csv", core_rows
            )
            write_csv(
                baseline_bundle / "discrepancy_episode_results.csv", baseline_rows
            )
            (factorial_bundle / "factorial_run_manifest.json").write_text(
                json.dumps(core_run), encoding="utf-8"
            )
            (factorial_bundle / "proposal_bank_manifest.json").write_text(
                json.dumps(proposal_manifest), encoding="utf-8"
            )
            (baseline_bundle / "discrepancy_run_manifest.json").write_text(
                json.dumps(baseline_run), encoding="utf-8"
            )
            (baseline_bundle / "proposal_bank_manifest.json").write_text(
                json.dumps(proposal_manifest), encoding="utf-8"
            )
            model_text = json.dumps(MODEL_ARTIFACT)
            (baseline_bundle / "discrepancy_query_gate_model.json").write_text(
                model_text, encoding="utf-8"
            )
            baseline_run["model_artifact"]["file_sha256"] = hashlib.sha256(
                model_text.encode("utf-8")
            ).hexdigest()
            (baseline_bundle / "discrepancy_run_manifest.json").write_text(
                json.dumps(baseline_run), encoding="utf-8"
            )

            status = main(
                [
                    "--factorial-bundle",
                    str(factorial_bundle),
                    "--baseline-bundle",
                    str(baseline_bundle),
                    "--output-dir",
                    str(output_dir),
                    "--draws",
                    "40",
                    "--bootstrap-seed",
                    "23",
                ]
            )

            self.assertEqual(status, 0)
            payload = json.loads(
                (output_dir / "discrepancy_query_baseline_analysis.json").read_text()
            )
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["design"]["independent_unit"], "simulator_seed")
            self.assertEqual(payload["model_artifact_sha256"], MODEL_SHA256)
            with (output_dir / "discrepancy_query_arm_summary.csv").open(
                newline=""
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4 * len(METRICS))
            with (output_dir / "discrepancy_query_paired_contrasts.csv").open(
                newline=""
            ) as handle:
                self.assertEqual(
                    len(list(csv.DictReader(handle))),
                    len(PAIRED_CONTRASTS) * len(METRICS),
                )


if __name__ == "__main__":
    unittest.main()
