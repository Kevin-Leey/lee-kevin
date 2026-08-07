import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import tools.analyze_factorial_interventions as analyzer
from tools.analyze_factorial_interventions import summarize_events


class InterventionSummaryTests(unittest.TestCase):
    def test_empty_seed_blocks_remain_in_per_seed_estimands(self):
        # Old in-memory row names remain readable, but emitted metrics use the
        # precise first-step actuator terminology.
        rows = [
            {
                "arm": "full",
                "seed": 10,
                "candidate_effect_distinct": 1,
                "executed_distinct": 1,
                "release_guard_rejected": 0,
                "classification": "beneficial",
                "utility_delta": 0.4,
            }
        ]

        summary = summarize_events(rows, seeds=(10, 11), draws=100, bootstrap_seed=3)
        by_metric = {
            row["metric"]: row for row in summary if row["arm"] == "full"
        }

        self.assertEqual(
            by_metric["executed_first_step_actuator_distinct_per_seed"][
                "estimate"
            ],
            0.5,
        )
        self.assertEqual(
            by_metric["selected_utility_gain_per_seed"]["estimate"], 0.2
        )
        beneficial = by_metric[
            "beneficial_fraction_of_executed_first_step_interventions"
        ]
        self.assertEqual(beneficial["estimate"], 1.0)
        self.assertEqual(beneficial["denominator"], 1)

    def test_cluster_bootstrap_is_deterministic(self):
        rows = [
            {
                "arm": "query_only",
                "seed": seed,
                "first_step_actuator_distinct": 1,
                "executed_first_step_actuator_distinct": 1,
                "release_guard_rejected": 0,
                "classification": "beneficial" if seed == 1 else "harmful",
                "utility_delta": value,
            }
            for seed, value in ((1, 0.3), (2, -0.2))
        ]

        first = summarize_events(rows, seeds=(1, 2), draws=50, bootstrap_seed=7)
        second = summarize_events(rows, seeds=(1, 2), draws=50, bootstrap_seed=7)

        self.assertEqual(first, second)
        utility = next(
            row
            for row in first
            if row["arm"] == "query_only"
            and row["metric"]
            == "utility_delta_per_executed_first_step_intervention"
        )
        self.assertTrue(np.isclose(utility["estimate"], 0.05))

    def test_selected_zero_release_arm_is_reported_without_unselected_arms(self):
        summary = summarize_events(
            [],
            seeds=(1, 2),
            draws=20,
            bootstrap_seed=5,
            arms=("query_only",),
        )

        self.assertEqual({row["arm"] for row in summary}, {"query_only"})
        release = next(
            row for row in summary if row["metric"] == "release_events_per_seed"
        )
        conditional = next(
            row
            for row in summary
            if row["metric"]
            == "utility_delta_per_executed_first_step_intervention"
        )
        self.assertEqual(release["estimate"], 0.0)
        self.assertEqual(conditional["estimate"], "")
        self.assertEqual(conditional["denominator"], 0)
        self.assertEqual(conditional["valid_bootstrap_draws"], 0)


def _v4_release_event(**overrides):
    event = {
        "frame": 10,
        "closed_loop_latency_source_frame": 4,
        "closed_loop_release_snapshot_identity_sha256": "snapshot-id",
        "closed_loop_latency_release_event": True,
        "closed_loop_released_slow_action": 4,
        "release_fast_comparator_action": 1,
        "release_selected_action": 4,
        "release_action_comparison_stage": analyzer.RELEASE_ACTION_COMPARISON_STAGE,
        "release_selection_distinct": True,
        "closed_loop_release_action_alignment_fast_effective_action": 1,
        "closed_loop_release_action_alignment_slow_effective_action": 4,
        "final_actuator_action": 4,
        "closed_loop_latency_executed_action": 4,
        "final_actuator_action_stage": (
            "post_shared_actuator_bridge_pre_environment_step"
        ),
        "closed_loop_release_opportunity_rejected": False,
        "closed_loop_release_action_unavailable": False,
        "closed_loop_latency_terminal_outcome": "distinct_actuation",
    }
    event.update(overrides)
    return event


def _branch(*, effective_action, target_speed, utility):
    return {
        "fast_action": 1,
        "effective_action": effective_action,
        "target_speed_after": target_speed,
        "utility": utility,
        "normalized_return": utility,
        "progress_m": utility * 10.0,
        "collision": 0,
        "min_ttc": 3.0,
        "mean_abs_jerk_mps3": 0.2,
        "steps_completed": 20,
        "terminal_cause": "horizon",
        "completed_horizon": True,
        "branch_trajectory_json": json.dumps(
            [{"frame": 10, "effective_action": effective_action}],
            separators=(",", ":"),
        ),
    }


class InterventionEventContractTests(unittest.TestCase):
    def test_missing_target_speed_cannot_be_called_first_step_equivalent(self):
        left = _branch(effective_action=1, target_speed=float("nan"), utility=0.1)
        right = _branch(effective_action=1, target_speed=float("nan"), utility=0.1)

        with self.assertRaisesRegex(ValueError, "finite target speeds"):
            analyzer._same_effective_action(left, right)

    def test_v4_stage_action_and_branch_tampering_fails_closed(self):
        invalid_events = (
            _v4_release_event(release_action_comparison_stage="wrong-stage"),
            _v4_release_event(final_actuator_action_stage="wrong-stage"),
            _v4_release_event(release_selection_distinct=False),
            _v4_release_event(
                closed_loop_latency_terminal_outcome="fast_equivalent"
            ),
            _v4_release_event(release_selected_action=2),
            _v4_release_event(closed_loop_release_opportunity_rejected=True),
            _v4_release_event(
                closed_loop_release_opportunity_rejected=True,
                closed_loop_release_action_unavailable=True,
            ),
        )

        for event in invalid_events:
            with self.subTest(event=event):
                with self.assertRaises(ValueError):
                    analyzer._validate_v4_release_action_contract(
                        event,
                        context="full/1/request",
                    )

    def test_first_step_equivalence_does_not_claim_full_effect_equivalence(self):
        snapshot = SimpleNamespace(
            frame=10,
            source_frame=4,
            request_id="request",
            snapshot_identity_sha256="snapshot-id",
        )
        baseline = _branch(effective_action=1, target_speed=20.0, utility=0.1)
        candidate = _branch(effective_action=1, target_speed=20.0, utility=0.8)
        event = _v4_release_event(
            final_actuator_action=1,
            closed_loop_latency_executed_action=1,
        )

        with mock.patch.object(
            analyzer,
            "_run_branch",
            side_effect=(baseline, candidate),
        ):
            row = analyzer._event_rollout_row(
                arm="full",
                seed=1,
                request_id="request",
                event=event,
                snapshot=snapshot,
                cfg={},
                horizon=20,
                gamma=0.99,
                epsilon=0.02,
            )

        self.assertEqual(row["classification"], "first_step_actuator_equivalent")
        self.assertEqual(row["first_step_actuator_distinct"], 0)
        self.assertEqual(row["selection_stage_primitive_distinct"], 1)
        self.assertNotEqual(row["utility_delta"], 0.0)
        self.assertNotIn("candidate_effect_distinct", row)

    def test_final_action_mismatch_is_retained_but_excluded_from_effects(self):
        snapshot = SimpleNamespace(
            frame=10,
            source_frame=4,
            request_id="request",
            snapshot_identity_sha256="snapshot-id",
        )
        baseline = _branch(effective_action=1, target_speed=20.0, utility=0.1)
        candidate = _branch(effective_action=4, target_speed=18.0, utility=0.2)
        event = _v4_release_event(
            final_actuator_action=1,
            closed_loop_latency_executed_action=1,
        )

        with mock.patch.object(
            analyzer,
            "_run_branch",
            side_effect=(baseline, candidate),
        ):
            row = analyzer._event_rollout_row(
                arm="full",
                seed=1,
                request_id="request",
                event=event,
                snapshot=snapshot,
                cfg={},
                horizon=20,
                gamma=0.99,
                epsilon=0.02,
            )

        self.assertEqual(row["classification"], "execution_reproduction_mismatch")
        self.assertEqual(row["candidate_evaluable"], 0)
        self.assertEqual(row["executed_first_step_actuator_distinct"], 0)
        self.assertEqual(row["utility_delta"], "")

    def test_legacy_unavailable_release_remains_auditable_without_v4_fields(self):
        snapshot = SimpleNamespace(
            frame=10,
            source_frame=4,
            request_id="request",
            snapshot_identity_sha256="snapshot-id",
        )
        baseline = _branch(effective_action=1, target_speed=20.0, utility=0.1)
        event = {
            "frame": 10,
            "closed_loop_latency_source_frame": 4,
            "closed_loop_release_snapshot_identity_sha256": "snapshot-id",
            "closed_loop_released_slow_action": 4,
            "closed_loop_execution_state_fast_action": 1,
            "closed_loop_latency_executed_action": 1,
            "closed_loop_release_opportunity_rejected": False,
            "closed_loop_release_action_unavailable": True,
        }

        with mock.patch.object(
            analyzer,
            "_run_branch",
            side_effect=(
                baseline,
                ValueError("action 4 is not available at frame 10"),
            ),
        ):
            row = analyzer._event_rollout_row(
                arm="full",
                seed=1,
                request_id="request",
                event=event,
                snapshot=snapshot,
                cfg={},
                horizon=20,
                gamma=0.99,
                epsilon=0.02,
                legacy_v2=True,
            )

        self.assertEqual(row["classification"], "unavailable")
        self.assertEqual(row["selection_stage_primitive_distinct"], "")
        self.assertEqual(row["candidate_evaluable"], 0)

    def test_legacy_presafety_fast_command_may_project_to_another_action(self):
        snapshot = SimpleNamespace(
            frame=10,
            source_frame=4,
            request_id="request",
            snapshot_identity_sha256="snapshot-id",
        )
        baseline = _branch(effective_action=4, target_speed=18.0, utility=0.1)
        candidate = _branch(effective_action=4, target_speed=18.0, utility=0.2)
        event = {
            "frame": 10,
            "closed_loop_latency_source_frame": 4,
            "closed_loop_release_snapshot_identity_sha256": "snapshot-id",
            "closed_loop_released_slow_action": 4,
            "closed_loop_execution_state_fast_action": 1,
            "closed_loop_latency_executed_action": 4,
            "closed_loop_release_opportunity_rejected": False,
            "closed_loop_release_action_unavailable": False,
        }

        with mock.patch.object(
            analyzer,
            "_run_branch",
            side_effect=(baseline, candidate),
        ):
            row = analyzer._event_rollout_row(
                arm="full",
                seed=1,
                request_id="request",
                event=event,
                snapshot=snapshot,
                cfg={},
                horizon=20,
                gamma=0.99,
                epsilon=0.02,
                legacy_v2=True,
            )

        self.assertEqual(row["fast_action"], 1)
        self.assertEqual(row["candidate_effective_action"], 4)
        self.assertEqual(row["first_step_actuator_distinct"], 0)

    def test_unavailable_flag_requires_matched_replay_non_evaluability(self):
        snapshot = SimpleNamespace(
            frame=10,
            source_frame=4,
            request_id="request",
            snapshot_identity_sha256="snapshot-id",
        )
        baseline = _branch(effective_action=1, target_speed=20.0, utility=0.1)
        candidate = _branch(effective_action=4, target_speed=18.0, utility=0.2)
        event = _v4_release_event(
            release_selected_action=1,
            release_selection_distinct=False,
            final_actuator_action=1,
            closed_loop_latency_executed_action=1,
            closed_loop_release_action_unavailable=True,
            closed_loop_latency_terminal_outcome="unavailable",
        )

        with mock.patch.object(
            analyzer,
            "_run_branch",
            side_effect=(baseline, candidate),
        ):
            with self.assertRaisesRegex(ValueError, "matched candidate is executable"):
                analyzer._event_rollout_row(
                    arm="full",
                    seed=1,
                    request_id="request",
                    event=event,
                    snapshot=snapshot,
                    cfg={},
                    horizon=20,
                    gamma=0.99,
                    epsilon=0.02,
                )

        with mock.patch.object(
            analyzer,
            "_run_branch",
            side_effect=(
                baseline,
                ValueError("action 4 is not available at frame 10"),
            ),
        ):
            row = analyzer._event_rollout_row(
                arm="full",
                seed=1,
                request_id="request",
                event=event,
                snapshot=snapshot,
                cfg={},
                horizon=20,
                gamma=0.99,
                epsilon=0.02,
            )

        self.assertEqual(row["classification"], "unavailable")
        self.assertEqual(row["candidate_replay_unavailable"], 1)


class InterventionCliContractTests(unittest.TestCase):
    def test_legacy_v2_audit_runs_before_aggregate_artifacts_are_read(self):
        with mock.patch.object(
            analyzer,
            "audit_bundle",
            side_effect=ValueError("ambiguous legacy lifecycle"),
        ) as audit, mock.patch.object(analyzer, "_read_csv") as read_csv:
            with self.assertRaisesRegex(ValueError, "ambiguous legacy lifecycle"):
                analyzer.main(
                    [
                        "--bundle",
                        "missing-bundle",
                        "--output-dir",
                        "unused-output",
                        "--legacy-v2",
                    ]
                )

        audit.assert_called_once()
        read_csv.assert_not_called()

    def test_legacy_v2_rejects_unknown_or_ambiguous_lifecycle_mode(self):
        report = {
            "accepted": True,
            "factorial_replay_version": analyzer.LEGACY_FACTORIAL_REPLAY_VERSION,
            "audit_contract": {"all_request_ids_lifecycle_closed": True},
            "cells": [{"accepted": True, "lifecycle_mode": "ambiguous"}],
        }
        with mock.patch.object(analyzer, "audit_bundle", return_value=report):
            with self.assertRaisesRegex(ValueError, "ambiguous lifecycle mode"):
                analyzer._audit_legacy_v2_bundle(Path("bundle"))

    def test_v4_request_audit_runs_before_aggregate_artifacts_are_read(self):
        with mock.patch.object(
            analyzer,
            "audit_bundle",
            side_effect=ValueError("invalid v4 snapshot binding"),
        ) as audit, mock.patch.object(analyzer, "_read_csv") as read_csv:
            with self.assertRaisesRegex(ValueError, "invalid v4 snapshot binding"):
                analyzer.main(
                    [
                        "--bundle",
                        "missing-bundle",
                        "--output-dir",
                        "unused-output",
                    ]
                )

        audit.assert_called_once()
        read_csv.assert_not_called()

    def test_main_accepts_a_selected_arm_with_zero_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "analysis"
            result_rows = [
                {"seed": "10", "arm": "query_only", "release_events": "0"}
            ]
            contract = {
                "seeds": (10,),
                "proposal_bank_sha256": "a" * 64,
            }
            with mock.patch.object(
                analyzer,
                "_audit_request_bundle",
                return_value={
                    "accepted": True,
                    "schema": "request-audit",
                    "factorial_replay_version": analyzer.FACTORIAL_REPLAY_VERSION,
                    "cells": [
                        {
                            "accepted": True,
                            "lifecycle_mode": "explicit_dual_event_ids",
                        }
                    ],
                },
            ), mock.patch.object(
                analyzer,
                "_read_csv",
                return_value=result_rows,
            ), mock.patch.object(
                analyzer,
                "_read_json",
                return_value={},
            ), mock.patch.object(
                analyzer,
                "validate_bundle_contract",
                return_value=contract,
            ), contextlib.redirect_stdout(io.StringIO()):
                status = analyzer.main(
                    [
                        "--bundle",
                        str(root / "bundle"),
                        "--output-dir",
                        str(output),
                        "--arms",
                        "query_only",
                        "--draws",
                        "20",
                        "--workers",
                        "1",
                    ]
                )

            self.assertEqual(status, 0)
            manifest = json.loads(
                (output / "factorial_intervention_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["arms"], ["query_only"])
            self.assertEqual(manifest["event_count"], 0)
            with (
                output / "factorial_intervention_events.csv"
            ).open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(list(reader), [])
                self.assertIn("first_step_actuator_distinct", reader.fieldnames)
            with (
                output / "factorial_intervention_summary.csv"
            ).open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(
                    {row["arm"] for row in csv.DictReader(handle)},
                    {"query_only"},
                )


if __name__ == "__main__":
    unittest.main()
