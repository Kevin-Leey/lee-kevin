import unittest

from tools.analyze_padriver_transfer import (
    EXPECTED_TRANSFER_KEYS,
    audit_source_rows,
    build_padriver_table,
    macro_paired_endpoints,
    macro_summary,
    paired_endpoints,
    safety_first_pairwise_ranking,
    summarize,
)


def make_source_row(key, *, source_hash="a" * 64):
    group, lanes, density, seed = key
    return {
        "group": group,
        "seed_idx": str(seed),
        "episodes_run": "1",
        "protocol_id": f"{group}::{'b' * 16}",
        "protocol_hash": "b" * 64,
        "config_hash": "c" * 64,
        "source_hash": source_hash,
        "evaluation_protocol_fixed_policy": "True",
        "result_dir": f"unused/{group}/{lanes}/{density}/{seed}",
        "transfer_lanes_count": str(lanes),
        "transfer_vehicles_density": str(density),
        "transfer_vehicle_count": "30",
        "transfer_scope": "smoke_only_not_formal_transfer_evidence",
    }


class TransferSourceAuditTests(unittest.TestCase):
    def test_complete_multi_shard_union_is_accepted_and_recorded(self):
        rows = [make_source_row(key) for key in sorted(EXPECTED_TRANSFER_KEYS)]

        audit = audit_source_rows(rows[:180] + rows[180:360] + rows[360:])

        self.assertEqual(audit["observed_unique_keys"], 540)
        self.assertEqual(audit["source_hash"], "a" * 64)
        self.assertEqual(audit["source_hash_row_count"], 540)
        self.assertEqual(audit["overlapping_rows"], 0)

    def test_incomplete_factorial_fails_closed(self):
        rows = [make_source_row(key) for key in sorted(EXPECTED_TRANSFER_KEYS)][:-1]

        with self.assertRaisesRegex(RuntimeError, "complete 540-run factorial"):
            audit_source_rows(rows)

    def test_mixed_source_hashes_fail_closed(self):
        rows = [make_source_row(key) for key in sorted(EXPECTED_TRANSFER_KEYS)]
        rows[-1]["source_hash"] = "d" * 64

        with self.assertRaisesRegex(RuntimeError, "mix executable source hashes"):
            audit_source_rows(rows)

    def test_overlapping_shards_are_reported_without_losing_unique_keys(self):
        rows = [make_source_row(key) for key in sorted(EXPECTED_TRANSFER_KEYS)]

        audit = audit_source_rows(rows + rows[:3])

        self.assertEqual(audit["observed_unique_keys"], 540)
        self.assertEqual(audit["source_rows"], 543)
        self.assertEqual(audit["overlapping_rows"], 3)


def make_episode(
    group,
    seed,
    lanes,
    density,
    *,
    success,
    collision,
    distance,
    frames,
    speed_sum,
    safe_frames=None,
    keep_frames=None,
):
    safe_frames = frames if safe_frames is None else safe_frames
    keep_frames = frames if keep_frames is None else keep_frames
    return {
        "group": group,
        "seed": seed,
        "lanes_count": lanes,
        "vehicles_density": density,
        "success": success,
        "collision": collision,
        "distance_m": distance,
        "speed_kmh": speed_sum / frames,
        "speed_sum_kmh": speed_sum,
        "safe_distance_rate": safe_frames / frames,
        "safe_distance_frames": safe_frames,
        "keep_rate": keep_frames / frames,
        "keep_frames": keep_frames,
        "runtime_s_per_frame": 0.1,
        "slow_call_rate": 0.2,
        "frames": frames,
        "collision_events_per_1000_frames": 1000.0 * collision / frames,
        "first_step_collision": 0,
        "observed_max_lane_id": lanes - 1,
        "replay_enabled": False,
        "replay_delay_positive": False,
        "queries": 0,
        "immediate_returns": 0,
        "releases": 0,
        "unavailable": 0,
        "rewritten": 0,
        "divergent": 0,
        "preserved": 0,
        "slow_fallback": 0,
    }


class TransferAnalysisPopulationTests(unittest.TestCase):
    def test_table_vii_uses_all_episodes_and_all_realized_frames(self):
        episodes = [
            make_episode(
                "rgd_fixed_policy", 0, 4, 2.0,
                success=1, collision=0, distance=100.0,
                frames=1, speed_sum=36.0,
            ),
            make_episode(
                "rgd_fixed_policy", 1, 4, 2.0,
                success=0, collision=1, distance=20.0,
                frames=9, speed_sum=0.0, safe_frames=0, keep_frames=0,
            ),
        ]

        row = summarize(episodes)[0]

        self.assertAlmostEqual(row["distance_all_episode_m"], 60.0)
        self.assertAlmostEqual(row["speed_all_realized_frames_kmh"], 3.6)
        self.assertAlmostEqual(row["safe_distance_all_realized_frames_rate"], 0.1)
        self.assertAlmostEqual(row["keep_all_realized_frames_rate"], 0.1)
        self.assertAlmostEqual(row["success_rate"], 0.5)
        self.assertNotIn("distance_success_m", row)
        self.assertNotIn("speed_success_kmh", row)
        self.assertNotIn("safe_distance_success_rate", row)
        self.assertNotIn("keep_success_rate", row)

    def test_success_subset_is_confined_to_padriver_format(self):
        episodes = [
            make_episode(
                "rgd_fixed_policy", 0, 4, 2.0,
                success=1, collision=0, distance=100.0,
                frames=2, speed_sum=72.0,
            ),
            make_episode(
                "rgd_fixed_policy", 1, 4, 2.0,
                success=0, collision=1, distance=20.0,
                frames=8, speed_sum=0.0, safe_frames=0, keep_frames=0,
            ),
        ]

        ours = build_padriver_table(episodes)[-1]

        self.assertEqual(ours["evaluation"], "RGD (ours)")
        self.assertAlmostEqual(ours["distance_m"], 100.0)
        self.assertAlmostEqual(ours["speed_kmh"], 36.0)
        self.assertIn("successful episodes", ours["analysis_population"])
        self.assertAlmostEqual(ours["collision_rate"], 0.5)
        self.assertEqual(ours["success_count"], 1)


class TransferPairedInferenceTests(unittest.TestCase):
    @staticmethod
    def factorial_episodes():
        rows = []
        for lanes in (4, 5, 6):
            for density in (2.0, 3.0):
                offset = lanes + density
                rows.extend(
                    [
                        make_episode(
                            "rgd_fixed_policy", 0, lanes, density,
                            success=1, collision=0, distance=100.0 + offset,
                            frames=1, speed_sum=80.0,
                        ),
                        make_episode(
                            "rgd_fixed_policy", 1, lanes, density,
                            success=0, collision=1, distance=40.0 + offset,
                            frames=3, speed_sum=120.0,
                        ),
                        make_episode(
                            "risk_budget", 0, lanes, density,
                            success=0, collision=1, distance=80.0 + offset,
                            frames=2, speed_sum=80.0,
                        ),
                        make_episode(
                            "risk_budget", 1, lanes, density,
                            success=0, collision=1, distance=20.0 + offset,
                            frames=2, speed_sum=80.0,
                        ),
                        make_episode(
                            "always_fast", 0, lanes, density,
                            success=0, collision=1, distance=75.0 + offset,
                            frames=2, speed_sum=76.0,
                        ),
                        make_episode(
                            "always_fast", 1, lanes, density,
                            success=0, collision=1, distance=15.0 + offset,
                            frames=2, speed_sum=76.0,
                        ),
                    ]
                )
        return rows

    def test_within_cell_contrasts_are_seed_paired(self):
        paired = paired_endpoints(
            self.factorial_episodes(), seed_ids=(0, 1), draws=400
        )
        row = next(
            item for item in paired
            if item["lanes_count"] == 4
            and item["vehicles_density"] == 2.0
            and item["baseline"] == "risk_budget"
        )

        self.assertEqual(row["cluster_unit"], "seed_within_cell")
        self.assertEqual((row["wins"], row["losses"], row["ties"]), (1, 0, 1))
        self.assertAlmostEqual(row["paired_success_difference"], 0.5)
        self.assertAlmostEqual(row["paired_collision_difference"], -0.5)
        self.assertAlmostEqual(row["paired_distance_all_episode_difference_m"], 20.0)
        self.assertAlmostEqual(row["paired_speed_all_realized_frames_difference_kmh"], 10.0)
        self.assertLessEqual(row["paired_success_difference_ci_low"], 0.5)
        self.assertGreaterEqual(row["paired_success_difference_ci_high"], 0.5)

    def test_macro_ci_clusters_each_seed_across_all_six_cells(self):
        episodes = self.factorial_episodes()
        summaries = macro_summary(episodes, seed_ids=(0, 1), draws=400)
        rgd = next(row for row in summaries if row["group"] == "rgd_fixed_policy")
        contrasts = macro_paired_endpoints(
            episodes, seed_ids=(0, 1), draws=400
        )
        distance = next(
            row for row in contrasts
            if row["baseline"] == "risk_budget"
            and row["metric"] == "distance_all_episode_m"
        )

        self.assertEqual(rgd["cells"], 6)
        self.assertEqual(rgd["seed_clusters"], 2)
        self.assertEqual(rgd["episodes"], 12)
        self.assertEqual(rgd["cluster_unit"], "seed_spanning_all_six_cells")
        self.assertAlmostEqual(rgd["speed_all_realized_frames_macro_kmh"], 50.0)
        self.assertAlmostEqual(distance["difference_rgd_minus_baseline"], 20.0)
        self.assertLessEqual(distance["paired_cluster_ci_low"], 20.0)
        self.assertGreaterEqual(distance["paired_cluster_ci_high"], 20.0)


class SafetyFirstRankingTests(unittest.TestCase):
    def test_conflicting_safety_metrics_are_not_resolved_by_lower_tiers(self):
        rows = [
            {
                "group": "left",
                "label": "Left",
                "collision_rate_macro": 0.1,
                "success_rate_macro": 0.8,
                "distance_all_episode_macro_m": 1000.0,
                "speed_all_realized_frames_macro_kmh": 100.0,
                "slow_call_rate_macro": 0.0,
                "runtime_s_per_frame_macro": 0.0,
            },
            {
                "group": "right",
                "label": "Right",
                "collision_rate_macro": 0.2,
                "success_rate_macro": 0.9,
                "distance_all_episode_macro_m": 1.0,
                "speed_all_realized_frames_macro_kmh": 1.0,
                "slow_call_rate_macro": 1.0,
                "runtime_s_per_frame_macro": 1.0,
            },
        ]

        comparison = safety_first_pairwise_ranking(rows)[0]

        self.assertEqual(comparison["relation"], "pareto_incomparable_at_tier_1")
        self.assertEqual(comparison["decisive_tier"], 1)
        self.assertFalse(comparison["weighted_score_used"])
        self.assertNotIn("score", comparison)


if __name__ == "__main__":
    unittest.main()
