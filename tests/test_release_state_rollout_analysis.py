import types
import unittest
from unittest.mock import patch

import numpy as np

import tools.analyze_release_state_rollouts as release_rollouts


def make_record(frame_id, *, strict_v11=False):
    record = {
        "frame_id": frame_id,
        "rgd_subordinate_diagnostics": {
            "recoverability_signal": {
                "recoverability_gate": {
                    "latency": {
                        "critical_latency_seconds": 3.4,
                        "policy_frequency": 10.0,
                        "source": "configured_execution_latency",
                    },
                    "latency_prediction_available": True,
                    "llm_backed_execution_available": True,
                    "alternative_viable_ratio": 0.25,
                    "alternative_viable_count": 2,
                    "cost_headroom": 0.64,
                    "need_score": 0.8,
                }
            }
        },
    }
    if strict_v11:
        gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
        gate.update(
            {
                "alternative_metric_source": "action_support_ranking_costs",
                "headroom_metric_source": "action_recovery_costs",
                "viable_cost_threshold": 0.55,
                "absolute_alternative_feasible": True,
            }
        )
    return record


class FixedQuerySelectionContractTests(unittest.TestCase):
    def test_accounting_reports_scheduled_excluded_and_evaluated_for_each_allocator(self):
        selected = {
            "RGD": [0, 20, 64],
            "TTC-delay": [1, 21, 64],
            "TTC-risk": [2, 22, 64],
        }

        _, accounting = release_rollouts._selection_contract(
            selected,
            seed=17,
            record_count=100,
            delays=(0.7, 1.7, 2.7),
            horizon=20,
        )

        main_rows = [row for row in accounting if row["allocator"] != "RGD-fixed"]
        self.assertEqual({row["allocator"] for row in main_rows}, {"RGD", "TTC-delay", "TTC-risk"})
        for row in main_rows:
            self.assertEqual(row["scheduled_count"], 3)
            self.assertEqual(row["excluded_count"], 1)
            self.assertEqual(row["evaluated_count"], 2)
            self.assertEqual(
                row["scheduled_count"],
                row["excluded_count"] + row["evaluated_count"],
            )

    def test_fixed_delay_cohort_prevents_delay_specific_attrition(self):
        selected = {
            "RGD": [0, 20, 64],
            "TTC-delay": [],
            "TTC-risk": [],
        }

        release_specs, accounting = release_rollouts._selection_contract(
            selected,
            seed=18,
            record_count=100,
            delays=(0.7, 1.7, 2.7),
            horizon=20,
        )

        fixed_rows = [row for row in accounting if row["allocator"] == "RGD-fixed"]
        self.assertEqual(len(fixed_rows), 3)
        self.assertEqual({row["scheduled_count"] for row in fixed_rows}, {3})
        self.assertEqual({row["excluded_count"] for row in fixed_rows}, {1})
        self.assertEqual({row["evaluated_count"] for row in fixed_rows}, {2})
        fixed_queries_by_delay = {}
        for allocator, query_frame, delay_s, _ in release_specs:
            if allocator == "RGD-fixed":
                fixed_queries_by_delay.setdefault(delay_s, set()).add(query_frame)
        self.assertEqual(set(fixed_queries_by_delay), {0.7, 1.7, 2.7})
        self.assertEqual(len({frozenset(queries) for queries in fixed_queries_by_delay.values()}), 1)
        self.assertEqual(next(iter(fixed_queries_by_delay.values())), {0, 20})


class CandidateStateSchemaTests(unittest.TestCase):
    def test_candidate_components_keep_need_latency_alternatives_and_headroom_separate(self):
        components = release_rollouts._query_gate_components(make_record(0), 1.7)

        self.assertAlmostEqual(components["need_score"], 0.8)
        self.assertAlmostEqual(components["latency_survival"], 0.5)
        self.assertAlmostEqual(components["admissible_alternative_fraction"], 0.25)
        self.assertAlmostEqual(components["recovery_headroom"], 0.64)
        self.assertAlmostEqual(components["opportunity"], 0.2)
        self.assertAlmostEqual(components["priority"], 0.16)

    def test_legacy_trace_recovers_action_specific_support_breadth(self):
        record = make_record(0)
        record["available_actions"] = (
            "IDLE Action_id: 1\nAcceleration Action_id: 3\nDeceleration Action_id: 4"
        )
        gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
        gate["hold_action"] = 1
        record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"] = {
            "route_ambiguity_profile": {
                "action_recovery_costs": {1: 0.30, 3: 0.80, 4: 0.30}
            }
        }

        components = release_rollouts._query_gate_components(record, 1.7)

        self.assertEqual(components["alternative_count"], 1)
        self.assertEqual(components["absolute_alternative_count"], 2)
        self.assertAlmostEqual(components["admissible_alternative_fraction"], 0.5)

    def test_new_profile_schema_prefers_explicit_support_costs(self):
        record = make_record(0)
        record["available_actions"] = (
            "IDLE Action_id: 1\nAcceleration Action_id: 3\nDeceleration Action_id: 4"
        )
        gate = record["rgd_subordinate_diagnostics"]["recoverability_signal"]["recoverability_gate"]
        gate["hold_action"] = 1
        record["rgd_subordinate_diagnostics"]["ambiguity_and_conflict"] = {
            "route_ambiguity_profile": {
                "action_recovery_costs": {1: 0.10, 3: 0.10, 4: 0.10},
                "action_support_ranking_costs": {1: 0.30, 3: 0.80, 4: 0.30},
                "probability_cost_source": "action_support_ranking_costs",
            }
        }

        components = release_rollouts._query_gate_components(record, 1.7)

        self.assertEqual(components["alternative_count"], 1)
        self.assertAlmostEqual(components["admissible_alternative_fraction"], 0.5)

    def test_event_rows_link_candidate_state_to_outcome_grounded_corrective_label(self):
        records = [make_record(frame_id, strict_v11=True) for frame_id in range(60)]
        selected = {"RGD": [0], "TTC-delay": [0], "TTC-risk": [0]}

        def fake_capture(_, __, target_frames, ___, ____):
            return ({frame: types.SimpleNamespace(frame=frame) for frame in target_frames}, 0.0)

        def fake_branch(snapshot, _, __, raw_action, ___, ____):
            is_fast = raw_action is None
            effective_action = 1 if is_fast or raw_action == 1 else 2
            return {
                "seed": 7,
                "release_frame": snapshot.frame,
                "raw_action": "fast" if is_fast else raw_action,
                "fast_action": 1,
                "legal_actions": "1;2",
                "effective_action": effective_action,
                "target_speed_after": 20.0,
                "horizon_steps": 20,
                "steps_completed": 20,
                "gamma": 0.99,
                "normalized_return": 0.0,
                "collision": 0,
                "min_ttc": 5.0,
                "progress_m": 1.0,
                "utility": 0.05 if raw_action == 2 else 0.0,
            }

        with (
            patch.object(release_rollouts, "_load_trace", return_value=(records, [])),
            patch.object(release_rollouts, "_selected_queries", return_value=selected),
            patch.object(release_rollouts, "_build_fast_config", return_value={}),
            patch.object(release_rollouts, "_capture_release_snapshots", side_effect=fake_capture),
            patch.object(release_rollouts, "_run_branch", side_effect=fake_branch),
        ):
            result = release_rollouts._process_seed(
                7,
                "unused-trace-root",
                "unused-protocol",
                (0.7, 1.7, 2.7),
                20,
                0.99,
                0.02,
                "unused-scratch-root",
            )

        event = next(
            row
            for row in result["events"]
            if row["allocator"] == "RGD" and row["delay_s"] == 1.7
        )
        self.assertEqual(event["candidate_state_id"], "7:0:17")
        self.assertAlmostEqual(event["need_score"], 0.8)
        self.assertAlmostEqual(event["latency_survival"], 0.5)
        self.assertAlmostEqual(event["admissible_alternative_fraction"], 0.25)
        self.assertAlmostEqual(event["recovery_headroom"], 0.64)
        self.assertEqual(event["corrective_set_nonempty"], 1)


class PairedFixedDelayBootstrapTests(unittest.TestCase):
    @staticmethod
    def fixed_row(seed, query_frame, delay_s, corrective):
        return {
            "seed": seed,
            "allocator": "RGD-fixed",
            "query_frame": query_frame,
            "delay_s": delay_s,
            "corrective_set_nonempty": corrective,
        }

    def test_paired_ci_retains_zero_query_seed_in_resampling_cohort(self):
        rows = [
            self.fixed_row(10, 0, 0.7, 1),
            self.fixed_row(10, 0, 2.7, 0),
            self.fixed_row(11, 0, 0.7, 0),
            self.fixed_row(11, 0, 2.7, 1),
        ]
        cohort = (10, 11, 12)  # Seed 12 has zero fixed queries and remains a cluster.

        class RecordingRng:
            def __init__(self):
                self.populations = []

            def choice(self, values, size, replace):
                self.populations.append(tuple(int(value) for value in values))
                self.assertion = (size, replace)
                return np.asarray(values)

        rng = RecordingRng()
        with patch.object(release_rollouts.np.random, "default_rng", return_value=rng):
            point, low, high, valid_draws = release_rollouts._cluster_bootstrap_paired_delay_difference(
                rows,
                cohort,
                draws=7,
            )

        self.assertAlmostEqual(point, 0.0)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.0)
        self.assertEqual(valid_draws, 7)
        self.assertEqual(rng.populations, [cohort] * 7)

    def test_paired_ci_rejects_delay_specific_query_attrition(self):
        rows = [
            self.fixed_row(10, 0, 0.7, 1),
            self.fixed_row(10, 0, 2.7, 0),
            self.fixed_row(11, 0, 0.7, 1),
        ]

        with self.assertRaisesRegex(ValueError, "complete paired block"):
            release_rollouts._cluster_bootstrap_paired_delay_difference(
                rows,
                (10, 11, 12),
                draws=7,
            )


if __name__ == "__main__":
    unittest.main()
