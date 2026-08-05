import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from dilu.evaluation.factorial_replay import (
    FACTORIAL_ARMS,
    FACTORIAL_PROPOSAL_SCHEMA,
    FACTORIAL_REPLAY_VERSION,
    FACTORIAL_RUN_SCHEMA,
)
from tools.analyze_query_release_factorial import (
    ARM_BY_NAME,
    ARM_NAMES,
    DISTINCT_ACTION_METRIC_STAGE,
    METRICS,
    PAIRWISE_CONTRASTS,
    RIGHT_CENSORING_AUDIT_POLICY,
    analyze_factorial_rows,
    factorial_effects,
    main,
    seed_bootstrap_indices,
    validate_bundle_contract,
    validate_factorial_rows,
)
from tools.audit_query_release_factorial import AUDIT_SCHEMA


def make_bank_payload(seeds):
    payload = []
    for seed in sorted(seeds):
        records = []
        for ordinal, source_frame in enumerate((3, 9)):
            response_text = f"seed={seed};frame={source_frame}"
            records.append(
                {
                    "seed": int(seed),
                    "source_frame": source_frame,
                    "request_id": (
                        f"factorial:{int(seed)}:{source_frame}:{ordinal:02d}"
                    ),
                    "raw_slow_action": ordinal + 1,
                    "latency_steps": ordinal + 2,
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


def make_row(seed, arm_name, *, proposal_hash, **overrides):
    arm = ARM_BY_NAME[arm_name]
    row = {
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "arm": arm_name,
        "query_gate_enabled": arm.query_gate_enabled,
        "release_guard_enabled": arm.release_guard_enabled,
        "seed": seed,
        "candidate_queries": 2,
        "issued_queries": 2,
        "query_gate_rejections": 0,
        "scheduled_timeouts": 0,
        "timeouts": 0,
        "failure_events": 0,
        "release_events": 2,
        "distinct_actuations": 1,
        "primitive_distinct_selections": 1,
        "aligned_distinct_actuations": int(arm.release_guard_enabled),
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
        "proposal_bank_sha256": proposal_hash,
    }
    row.update(overrides)
    return row


def make_rows(seeds=(10, 11)):
    proposal_hash = bank_sha256(make_bank_payload(seeds))
    return [
        make_row(seed, arm, proposal_hash=proposal_hash)
        for seed in seeds
        for arm in ARM_NAMES
    ]


def make_manifests(seeds=(10, 11)):
    payload = make_bank_payload(seeds)
    proposal_hash = bank_sha256(payload)
    run_order = [
        {"seed": seed, "order": order, "arm": arm}
        for seed in seeds
        for order, arm in enumerate(reversed(ARM_NAMES))
    ]
    run_manifest = {
        "schema": FACTORIAL_RUN_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "latency_profile": "frozen",
        "proposal_bank_sha256": proposal_hash,
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "seed_start": seeds[0],
        "seed_count": len(seeds),
        "result_rows": len(seeds) * len(ARM_NAMES),
        "arms": [asdict(arm) for arm in FACTORIAL_ARMS],
        "randomized_block_run_order": run_order,
    }
    proposal_manifest = {
        "schema": FACTORIAL_PROPOSAL_SCHEMA,
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "latency_profile": "frozen",
        "bank_sha256": proposal_hash,
        "candidate_source_policy": "scheduled_always_slow",
        "candidate_source_gate_independent": True,
        "seed_count": len(seeds),
        "proposal_count": sum(len(block["records"]) for block in payload),
        "bank_payload": payload,
    }
    return run_manifest, proposal_manifest


def rebind_bank(rows, run_manifest, proposal_manifest):
    payload = proposal_manifest["bank_payload"]
    proposal_hash = bank_sha256(payload)
    proposal_manifest["bank_sha256"] = proposal_hash
    proposal_manifest["proposal_count"] = sum(
        len(block["records"]) for block in payload
    )
    run_manifest["proposal_bank_sha256"] = proposal_hash
    for row in rows:
        row["proposal_bank_sha256"] = proposal_hash


def make_verified_audit_report(
    rows,
    run_manifest,
    proposal_manifest,
    *,
    bundle="fixture",
):
    request_ids_by_seed = {
        int(block["seed"]): [record["request_id"] for record in block["records"]]
        for block in proposal_manifest["bank_payload"]
    }
    proposal_counts_by_seed = {
        seed: len(request_ids) for seed, request_ids in request_ids_by_seed.items()
    }
    cells = []
    for row in rows:
        seed = int(row["seed"])
        candidate_queries = int(row["candidate_queries"])
        censored_ids = request_ids_by_seed[seed][candidate_queries:]
        cells.append(
            {
                "accepted": True,
                "seed": seed,
                "arm": row["arm"],
                "candidate_queries": candidate_queries,
                "reachable_proposal_count": candidate_queries,
                "right_censored_proposal_count": len(censored_ids),
                "right_censored_proposal_ids": censored_ids,
            }
        )
    return {
        "schema": AUDIT_SCHEMA,
        "accepted": True,
        "bundle": str(bundle),
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "proposal_bank_sha256": run_manifest["proposal_bank_sha256"],
        "audit_contract": {
            "proposal_bank_hash_authenticated": True,
            "proposal_source_files_authenticated": True,
            "all_candidate_records_bound_to_bank": True,
            "all_request_ids_lifecycle_closed": True,
            "release_iff_one_authenticated_snapshot": True,
            "timeout_failure_pending_snapshot_forbidden": True,
            "cross_arm_candidate_identity_authenticated": True,
            "cross_arm_comparison_scope": "common_reachable_proposals",
            "right_censoring_policy": RIGHT_CENSORING_AUDIT_POLICY,
        },
        "aggregate": {
            "seed_count": len(proposal_counts_by_seed),
            "arm_count": len(ARM_NAMES),
            "arm_seed_cells": len(cells),
            "proposal_count": sum(proposal_counts_by_seed.values()),
            "candidate_queries": sum(cell["candidate_queries"] for cell in cells),
            "reachable_proposal_count": sum(
                cell["reachable_proposal_count"] for cell in cells
            ),
            "right_censored_proposal_count": sum(
                cell["right_censored_proposal_count"] for cell in cells
            ),
        },
        "cells": cells,
        "errors": [],
    }


class FactorialContrastTests(unittest.TestCase):
    def test_standard_main_effects_and_difference_in_differences(self):
        effects = factorial_effects(
            {"neither": 0.0, "query_only": 2.0, "release_only": 4.0, "full": 10.0}
        )

        self.assertEqual(effects["query_main_effect"], 4.0)
        self.assertEqual(effects["release_main_effect"], 6.0)
        self.assertEqual(effects["query_x_release_interaction"], 4.0)

    def test_bootstrap_draws_are_deterministic_seed_blocks(self):
        first = seed_bootstrap_indices(3, draws=12, bootstrap_seed=77)
        second = seed_bootstrap_indices(3, draws=12, bootstrap_seed=77)

        self.assertEqual(first.shape, (12, 3))
        self.assertTrue((first == second).all())


class FactorialValidationTests(unittest.TestCase):
    def test_missing_and_duplicate_arms_fail_closed(self):
        rows = make_rows()
        missing = [row for row in rows if not (row["seed"] == 10 and row["arm"] == "full")]
        with self.assertRaisesRegex(ValueError, "incomplete factorial matrix"):
            validate_factorial_rows(missing)

        with self.assertRaisesRegex(ValueError, "duplicate factorial arm"):
            validate_factorial_rows(rows + [dict(rows[0])])

    def test_mixed_bank_hash_and_nonfinite_metric_fail_closed(self):
        mixed = make_rows()
        mixed[0]["proposal_bank_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "mixed proposal-bank hashes"):
            validate_factorial_rows(mixed)

        nonfinite = make_rows()
        nonfinite[0]["episode_reward"] = "nan"
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_factorial_rows(nonfinite)

    def test_lifecycle_accounting_and_release_snapshot_pairing_fail_closed(self):
        rows = make_rows()
        rows[0]["snapshot_count"] = 1
        with self.assertRaisesRegex(ValueError, "release/snapshot coverage mismatch"):
            validate_factorial_rows(rows)

    def test_v4_primitive_distinctness_contract_fails_closed(self):
        mutations = (
            (
                lambda row: row.pop("primitive_distinct_selections"),
                "primitive_distinct_selections",
            ),
            (
                lambda row: row.__setitem__("distinct_action_metric_stage", "final"),
                "metric stage drift",
            ),
            (
                lambda row: row.__setitem__("effect_distinctness_available", True),
                "must not claim effect distinctness",
            ),
            (
                lambda row: row.__setitem__("distinct_actuations", 0),
                "alias disagrees with primitive selections",
            ),
            (
                lambda row: row.__setitem__("aligned_distinct_actuations", 2),
                "exceed primitive selections",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                rows = make_rows()
                mutate(rows[0])
                with self.assertRaisesRegex(ValueError, message):
                    validate_factorial_rows(rows)

    def test_bundle_contract_verifies_manifest_and_canonical_bank_hash(self):
        rows = make_rows()
        run_manifest, proposal_manifest = make_manifests()

        contract = validate_bundle_contract(rows, run_manifest, proposal_manifest)
        self.assertEqual(contract["seeds"], (10, 11))

        proposal_manifest["bank_payload"][0]["records"] = [
            {"seed": 10, "source_frame": 1}
        ]
        proposal_manifest["proposal_count"] = 1
        with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
            validate_bundle_contract(rows, run_manifest, proposal_manifest)

    def test_proposal_record_schema_and_identity_fail_closed(self):
        mutations = (
            (
                lambda payload: payload[0]["records"][0].pop("response_text"),
                "proposal-record schema mismatch",
            ),
            (
                lambda payload: payload[1].__setitem__("seed", 10),
                "duplicate proposal bank seed",
            ),
            (
                lambda payload: payload[0]["records"][1].__setitem__(
                    "source_frame", 3
                ),
                "duplicate proposal source frame",
            ),
            (
                lambda payload: payload[1]["records"][0].__setitem__(
                    "request_id", payload[0]["records"][0]["request_id"]
                ),
                "duplicate proposal request ID",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                rows = make_rows()
                run_manifest, proposal_manifest = make_manifests()
                mutate(proposal_manifest["bank_payload"])
                rebind_bank(rows, run_manifest, proposal_manifest)
                with self.assertRaisesRegex(ValueError, message):
                    validate_bundle_contract(rows, run_manifest, proposal_manifest)

    def test_proposal_action_latency_outcome_and_response_hash_fail_closed(self):
        mutations = (
            (
                lambda record: record.__setitem__("raw_slow_action", 5),
                "action outside discrete action universe",
            ),
            (
                lambda record: record.__setitem__("latency_steps", -1),
                "negative .*latency_steps",
            ),
            (
                lambda record: record.__setitem__("outcome", "unknown"),
                "invalid response outcome",
            ),
            (
                lambda record: record.__setitem__(
                    "response_text", record["response_text"] + " tampered"
                ),
                "response text/hash mismatch",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                rows = make_rows()
                run_manifest, proposal_manifest = make_manifests()
                mutate(proposal_manifest["bank_payload"][0]["records"][0])
                rebind_bank(rows, run_manifest, proposal_manifest)
                with self.assertRaisesRegex(ValueError, message):
                    validate_bundle_contract(rows, run_manifest, proposal_manifest)

    def test_each_arm_candidate_count_is_bound_to_seed_proposal_count(self):
        rows = make_rows()
        run_manifest, proposal_manifest = make_manifests()
        rows[0]["candidate_queries"] = 3
        rows[0]["query_gate_rejections"] = 1

        with self.assertRaisesRegex(ValueError, "candidate count does not match"):
            validate_bundle_contract(rows, run_manifest, proposal_manifest)

    def test_audited_terminal_censoring_allows_only_the_authenticated_cell(self):
        rows = make_rows()
        run_manifest, proposal_manifest = make_manifests()
        censored = next(
            row
            for row in rows
            if row["seed"] == 10 and row["arm"] == "release_only"
        )
        censored.update(
            {
                "candidate_queries": 1,
                "issued_queries": 1,
                "release_events": 1,
                "snapshot_count": 1,
            }
        )
        audit_report = make_verified_audit_report(
            rows, run_manifest, proposal_manifest
        )

        contract = validate_bundle_contract(
            rows,
            run_manifest,
            proposal_manifest,
            audit_report=audit_report,
        )

        self.assertEqual(
            contract["matrix"][(10, "release_only")]["candidate_queries"], 1
        )
        self.assertTrue(contract["request_audit"]["right_censoring_authorized"])
        self.assertEqual(
            contract["request_audit"]["right_censored_proposal_count"], 1
        )
        self.assertRegex(contract["request_audit"]["sha256"], r"^[0-9a-f]{64}$")

    def test_audited_terminal_censoring_rejects_report_binding_drift(self):
        rows = make_rows()
        run_manifest, proposal_manifest = make_manifests()
        censored = next(
            row
            for row in rows
            if row["seed"] == 10 and row["arm"] == "release_only"
        )
        censored.update(
            {
                "candidate_queries": 1,
                "issued_queries": 1,
                "release_events": 1,
                "snapshot_count": 1,
            }
        )

        mutations = (
            (
                lambda report: report.__setitem__("schema", "unexpected"),
                "audit schema",
            ),
            (
                lambda report: report.__setitem__(
                    "factorial_replay_version", "unexpected"
                ),
                "replay-version drift",
            ),
            (
                lambda report: report.__setitem__("proposal_bank_sha256", "0" * 64),
                "proposal-bank hash mismatch",
            ),
            (
                lambda report: report["cells"][0].__setitem__("seed", 99),
                "outside seed cohort",
            ),
            (
                lambda report: report["aggregate"].__setitem__("proposal_count", 1),
                "proposal-bank count mismatch",
            ),
        )
        for mutate, message in mutations:
            with self.subTest(message=message):
                audit_report = make_verified_audit_report(
                    rows, run_manifest, proposal_manifest
                )
                mutate(audit_report)
                with self.assertRaisesRegex(ValueError, message):
                    validate_bundle_contract(
                        rows,
                        run_manifest,
                        proposal_manifest,
                        audit_report=audit_report,
                    )

    def test_empty_seed_proposal_block_fails_closed(self):
        rows = make_rows()
        run_manifest, proposal_manifest = make_manifests()
        proposal_manifest["bank_payload"][0]["records"] = []
        rebind_bank(rows, run_manifest, proposal_manifest)

        with self.assertRaisesRegex(ValueError, "proposal bank has no candidates"):
            validate_bundle_contract(rows, run_manifest, proposal_manifest)


class FactorialAnalysisTests(unittest.TestCase):
    def test_analysis_is_paired_by_seed_and_deterministic(self):
        rows = make_rows()
        reward_by_seed_arm = {
            (10, "neither"): 0.0,
            (10, "query_only"): 2.0,
            (10, "release_only"): 4.0,
            (10, "full"): 10.0,
            (11, "neither"): 1.0,
            (11, "query_only"): 5.0,
            (11, "release_only"): 3.0,
            (11, "full"): 11.0,
        }
        for row in rows:
            row["episode_reward"] = reward_by_seed_arm[(row["seed"], row["arm"])]

        first = analyze_factorial_rows(rows, draws=200, bootstrap_seed=9)
        second = analyze_factorial_rows(rows, draws=200, bootstrap_seed=9)

        self.assertEqual(first, second)
        effects = {
            row["effect"]: row
            for row in first["paired_effects"]
            if row["metric"] == "episode_reward"
        }
        self.assertEqual(effects["query_main_effect"]["estimate"], 5.0)
        self.assertEqual(effects["release_main_effect"]["estimate"], 5.0)
        self.assertEqual(effects["query_x_release_interaction"]["estimate"], 4.0)
        self.assertEqual(effects["query_main_effect"]["n_seed_blocks"], 2)
        self.assertEqual(effects["full_minus_neither"]["estimate"], 10.0)
        self.assertEqual(effects["full_minus_query_only"]["estimate"], 7.0)
        self.assertEqual(effects["full_minus_release_only"]["estimate"], 7.0)

        full_summary = next(
            row
            for row in first["arm_summaries"]
            if row["arm"] == "full" and row["metric"] == "episode_reward"
        )
        self.assertEqual(full_summary["mean"], 10.5)

    def test_cli_writes_json_and_two_concise_csv_tables(self):
        rows = make_rows()
        run_manifest, proposal_manifest = make_manifests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            output = root / "analysis"
            bundle.mkdir()
            with (bundle / "factorial_episode_results.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (bundle / "factorial_run_manifest.json").write_text(
                json.dumps(run_manifest), encoding="utf-8"
            )
            (bundle / "proposal_bank_manifest.json").write_text(
                json.dumps(proposal_manifest), encoding="utf-8"
            )

            audit_report = make_verified_audit_report(
                rows,
                run_manifest,
                proposal_manifest,
                bundle=bundle.resolve(),
            )
            with patch(
                "tools.analyze_query_release_factorial.audit_bundle",
                return_value=audit_report,
            ) as audit:
                status = main(
                    [
                        "--bundle",
                        str(bundle),
                        "--output-dir",
                        str(output),
                        "--draws",
                        "40",
                        "--bootstrap-seed",
                        "13",
                    ]
                )

            self.assertEqual(status, 0)
            audit.assert_called_once_with(bundle.resolve())
            payload = json.loads((output / "factorial_analysis.json").read_text())
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["design"]["independent_unit"], "simulator_seed")
            self.assertEqual(
                payload["request_audit"]["proposal_bank_sha256"],
                proposal_manifest["bank_sha256"],
            )
            self.assertTrue(
                payload["bootstrap"][
                    "shared_draw_matrix_across_arms_metrics_and_contrasts"
                ]
            )
            with (output / "factorial_arm_summary.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4 * len(METRICS))
            with (output / "factorial_paired_effects.csv").open(newline="") as handle:
                self.assertEqual(
                    len(list(csv.DictReader(handle))),
                    (3 + len(PAIRWISE_CONTRASTS)) * len(METRICS),
                )


if __name__ == "__main__":
    unittest.main()
