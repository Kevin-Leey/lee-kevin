import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


from dilu.driver_agent.reasoning.decision import RouteAmbiguityProfile
from dilu.driver_agent.reasoning.rgd_support import (
    PaperBaselineDefinition,
    _stable_unit_interval,
    resolve_paper_baseline_trigger,
)
from dilu.evaluation import reporter
from dilu.runtime_episode_setup import create_episode_env
from tools.run_main_table import (
    _apply_cli_runtime_overrides,
    _completed_metrics,
    _execute_seed_blocks,
    _filter_active_cohort_rows,
    _merge_group_run_rows,
    _ordered_groups_for_seed,
    _resume_row_is_complete,
    _validate_v12_main_preflight,
    _validate_v13_preflight,
    _validate_v13_resolved_cell,
    _write_cell_completion_manifest,
    main as run_main_table,
    parse_args,
)
from tools.run_main_table_runtime import (
    build_group_config,
    iter_selected_groups,
    load_formal_base_config,
    load_formal_protocol,
    resolve_policy_execution_horizon,
)
from tools.result_bundle_pipeline import write_result_bundle_manifest
from tools.protocol_io import dump_json
from dilu.scenario.env_builder import build_env_config
from dilu.evaluation.metrics_aggregator import MetricsAggregator
from tools import run_main_table as main_table_module


class _DummyHighwayEnv:
    def __init__(self):
        self.unwrapped = self
        self.vehicle = None
        self.reset_seeds = []

    def reset(self, *, seed):
        self.reset_seeds.append(seed)
        return {"seed": seed}, {}


class MainTableFreshSeedProtocolTests(unittest.TestCase):
    def _config(self, result_dir: Path, group_name: str = "random_budget"):
        base_cfg = {
            "simulation_duration": 30,
            "LLM_PROVIDER": "siliconflow",
            "SILICONFLOW_API_KEY": "snapshot-test-key",
            "SILICONFLOW_BASE_URL": "https://api.siliconflow.cn/v1",
            "SILICONFLOW_CHAT_MODEL": "Qwen/Qwen3-8B",
            "slow_thinking": {"risk_coupling": {"core_story": {}}},
            "training": {"protocol": {}},
        }
        protocol = {
            "protocol_name": "seed_protocol_test",
            "protocol_version": 1,
            "claim_guardrails": {},
            "_source_path": "formal_protocol.yaml",
        }
        group_cfg = {"id": "P5", "runtime_overrides": {}}
        cfg = build_group_config(
            base_cfg,
            group_name,
            group_cfg,
            "highway-v0",
            1,
            result_dir,
            protocol,
        )
        args = argparse.Namespace(seed_value=None, simulation_duration=None)
        _apply_cli_runtime_overrides(cfg, args, 100)
        return cfg

    @staticmethod
    def _write_minimal_cell_closure(result_dir, expected_identity):
        episode_id = 100
        prefix = "highway_100"
        ep_dir = result_dir / "ep_100"
        event_path = result_dir / "event_logs" / f"event_log_{prefix}_{episode_id}.json"
        event = {"frame": 0, "done": True, "episode_done": True, "terminal_cause": "truncated"}
        dump_json(
            event_path,
            {
                "schema_version": "rgd_event_log_v3",
                "episode_id": episode_id,
                "prefix": prefix,
                "event_count": 1,
                "pending_release_count": 0,
                "terminal_cause": "truncated",
                "events": [event],
                "pending_releases_dropped_at_episode_end": [],
                "release_snapshot_count": 0,
                "release_snapshot_bundle": None,
                "release_snapshot_manifest": None,
                "release_snapshot_bundle_sha256": None,
            },
        )
        dump_json(
            ep_dir / f"{prefix}_reasoning_records.json",
            {"episode_id": episode_id, "record_count": 1, "analysis_records": [{}]},
        )
        dump_json(
            ep_dir / f"{prefix}_physical_frames.json",
            {
                "episode_id": episode_id,
                "frame_count": 1,
                "frames": [{"frame_id": 0, "done": True}],
                "metrics": {"total_frames": 1},
            },
        )
        dump_json(
            result_dir / f"episode_result_{prefix}_{episode_id}.json",
            {
                "episode_id": episode_id,
                "prefix": prefix,
                "frame_count": 1,
                "event_log": str(event_path.resolve()),
            },
        )
        dump_json(
            result_dir / "random_budget_rgd_metrics.json",
            {"comprehensive_metrics": {"total_frames": 1}},
        )
        _write_cell_completion_manifest(
            result_dir,
            group_name="random_budget",
            env_type="highway-v0",
            seed_label=100,
            episodes=1,
            expected_identity=expected_identity,
            runtime_integrity_checks=[True],
            runtime_integrity_records=[
                {
                    "episode_id": 100,
                    "identity_start": {"runtime_identity_sha256": "test"},
                    "identity_end": {"runtime_identity_sha256": "test"},
                    "clean": True,
                }
            ],
        )

    @staticmethod
    def _profile():
        return RouteAmbiguityProfile(
            action_probabilities={1: 1.0},
            action_recovery_costs={1: 0.2},
            ambiguity_best_action=1,
            selected_probability=1.0,
            ambiguity_entropy=1.0,
            ambiguity_gap=0.0,
            evidence_disagreement=0.0,
            intervention_risk=0.0,
        )

    def test_fresh_seed_reaches_actual_environment_reset_without_modulo_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            env = _DummyHighwayEnv()
            with patch("dilu.runtime_episode_setup.require_highway_env", return_value="highway-v0"), patch(
                "dilu.runtime_episode_setup._acquire_highway_env", return_value=(env, False)
            ):
                _, _, _, _, actual_seed, _ = create_episode_env(
                    ep=100,
                    cfg=cfg,
                    result_dir=tmp,
                    seed_pool=list(range(30)),
                )
        self.assertEqual(actual_seed, 100)
        self.assertEqual(env.reset_seeds, [100])
        self.assertEqual(cfg["fixed_seed_override"], 100)
        self.assertEqual(cfg["_paper_protocol_config"]["runtime_config"]["fixed_seed_override"], 100)

    def test_null_fixed_seed_override_uses_seed_pool_without_int_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._config(Path(tmp))
            cfg["fixed_seed_override"] = None
            env = _DummyHighwayEnv()
            with patch("dilu.runtime_episode_setup.require_highway_env", return_value="highway-v0"), patch(
                "dilu.runtime_episode_setup._acquire_highway_env", return_value=(env, False)
            ):
                _, _, _, _, actual_seed, _ = create_episode_env(
                    ep=0,
                    cfg=cfg,
                    result_dir=tmp,
                    seed_pool=[17],
                )
        self.assertEqual(actual_seed, 17)
        self.assertEqual(env.reset_seeds, [17])

    def test_random_and_uncertainty_hashes_use_the_bound_fresh_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_100 = self._config(Path(tmp) / "seed100")
            cfg_101 = self._config(Path(tmp) / "seed101")
            _apply_cli_runtime_overrides(
                cfg_101,
                argparse.Namespace(seed_value=None, simulation_duration=None),
                101,
            )

        random_core = {
            "paper_baseline_trigger_mode": "random_budget",
            "paper_baseline_random_slow_probability": 0.5,
        }
        random_definition = PaperBaselineDefinition.from_core_story_config(random_core)
        random_draws = []
        for cfg in (cfg_100, cfg_101):
            stats = {"decision_count": 7}
            resolve_paper_baseline_trigger(
                {"risk_coupling": {"core_story": random_core}, "fixed_seed_override": cfg["fixed_seed_override"]},
                stats,
                random_definition,
                self._profile(),
                {"ttc_pressure": 0.0, "proximity_complexity": 0.0},
                0.2,
            )
            random_draws.append(stats["paper_baseline_random_draw"])
        self.assertEqual(random_draws[0], _stable_unit_interval("random_budget_v2", 100, 7))
        self.assertEqual(random_draws[1], _stable_unit_interval("random_budget_v2", 101, 7))
        self.assertNotEqual(random_draws[0], random_draws[1])

        uncertainty_core = {
            "paper_baseline_trigger_mode": "uncertainty",
            "paper_baseline_uncertainty_cutoff": 1.0,
            "paper_baseline_exposure_probability": 0.5,
        }
        uncertainty_definition = PaperBaselineDefinition.from_core_story_config(uncertainty_core)
        exposure_draws = []
        for cfg in (cfg_100, cfg_101):
            stats = {"decision_count": 7}
            resolve_paper_baseline_trigger(
                {"risk_coupling": {"core_story": uncertainty_core}, "fixed_seed_override": cfg["fixed_seed_override"]},
                stats,
                uncertainty_definition,
                self._profile(),
                {"ttc_pressure": 0.0, "proximity_complexity": 0.0},
                0.2,
            )
            exposure_draws.append(stats["paper_baseline_exposure_draw"])
        self.assertEqual(exposure_draws[0], _stable_unit_interval("baseline_exposure_v1", "uncertainty", 100, 7))
        self.assertEqual(exposure_draws[1], _stable_unit_interval("baseline_exposure_v1", "uncertainty", 101, 7))
        self.assertNotEqual(exposure_draws[0], exposure_draws[1])

    def test_snapshot_manifest_config_and_resume_share_one_seed_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp) / "setting"
            result_dir.mkdir(parents=True)
            cfg = self._config(result_dir)
            with patch.object(reporter.importlib_metadata, "version", return_value="test"):
                reporter.save_experiment_snapshot(cfg, str(result_dir), 100)
                expected_identity = reporter.build_experiment_identity(cfg, 100)

            manifest = json.loads((result_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
            snapshot = json.loads((result_dir / "experiment_snapshot.json").read_text(encoding="utf-8"))
            serialized_outputs = json.dumps({"manifest": manifest, "snapshot": snapshot}, sort_keys=True)
            self.assertNotIn("snapshot-test-key", serialized_outputs)
            self.assertEqual(manifest["llm_backend"]["provider"], "siliconflow")
            self.assertEqual(manifest["llm_backend"]["requested_model"], "Qwen/Qwen3-8B")
            self.assertEqual(manifest["llm_backend"]["base_url_origin"], "https://api.siliconflow.cn")
            self.assertEqual(manifest["fixed_seed_override"], 100)
            self.assertEqual(manifest["resolved_seeds"], [100])
            self.assertEqual(manifest["config"]["fixed_seed_override"], 100)
            self.assertEqual(manifest["source_hash"], expected_identity["source_hash"])
            self.assertEqual(snapshot["fixed_seed_override"], 100)
            self.assertEqual(snapshot["seed_start"], 100)
            self.assertEqual(snapshot["seeds_used"], [100])
            self.assertEqual(snapshot["config"]["fixed_seed_override"], 100)
            self.assertEqual(snapshot["config_hash"], expected_identity["config_hash"])
            self.assertEqual(snapshot["source_hash"], expected_identity["source_hash"])

            row = {
                "group": "random_budget",
                "env": "highway-v0",
                "seed_idx": 100,
                "fixed_seed_override": 100,
                "episodes_run": 1,
                "protocol_id": manifest["protocol_id"],
                "protocol_hash": manifest["protocol_hash"],
                "config_hash": manifest["config_hash"],
                "source_hash": manifest["source_hash"],
            }
            # Legacy cells with only the two identity manifests are not a
            # resumable artifact closure.
            self.assertFalse(
                _resume_row_is_complete(
                    row,
                    result_dir,
                    group_name="random_budget",
                    env_type="highway-v0",
                    seed_label=100,
                    episodes=1,
                    expected_identity=expected_identity,
                )
            )
            self._write_minimal_cell_closure(result_dir, expected_identity)
            self.assertTrue(
                _resume_row_is_complete(
                    row,
                    result_dir,
                    group_name="random_budget",
                    env_type="highway-v0",
                    seed_label=100,
                    episodes=1,
                    expected_identity=expected_identity,
                )
            )

            legacy_row = dict(row, fixed_seed_override=None)
            self.assertFalse(
                _resume_row_is_complete(
                    legacy_row,
                    result_dir,
                    group_name="random_budget",
                    env_type="highway-v0",
                    seed_label=100,
                    episodes=1,
                    expected_identity=expected_identity,
                )
            )
            for field in ("source_hash", "config_hash", "protocol_hash"):
                mismatched_identity = dict(expected_identity)
                mismatched_identity[field] = f"different-{field}"
                self.assertFalse(
                    _resume_row_is_complete(
                        row,
                        result_dir,
                        group_name="random_budget",
                        env_type="highway-v0",
                        seed_label=100,
                        episodes=1,
                        expected_identity=mismatched_identity,
                    ),
                    field,
                )
            physical_path = (
                result_dir / "ep_100" / "highway_100_physical_frames.json"
            )
            original_physical = physical_path.read_text(encoding="utf-8")
            physical_path.write_text(original_physical + "\n", encoding="utf-8")
            self.assertFalse(
                _resume_row_is_complete(
                    row,
                    result_dir,
                    group_name="random_budget",
                    env_type="highway-v0",
                    seed_label=100,
                    episodes=1,
                    expected_identity=expected_identity,
                )
            )
            physical_path.write_text(original_physical, encoding="utf-8")
            manifest["config"]["fixed_seed_override"] = 101
            (result_dir / "runtime_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertFalse(
                _resume_row_is_complete(
                    row,
                    result_dir,
                    group_name="random_budget",
                    env_type="highway-v0",
                    seed_label=100,
                    episodes=1,
                    expected_identity=expected_identity,
                )
            )
            merged = _merge_group_run_rows([legacy_row], [row])
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["fixed_seed_override"], 100)

    def test_bundle_manifest_records_the_complete_fresh_seed_schedule(self):
        horizon = resolve_policy_execution_horizon(
            {
                "simulation_duration": 30,
                "policy_frequency": 10,
                "simulation_frequency": 10,
            }
        ).as_manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_result_bundle_manifest(
                Path(tmp),
                "formal_run",
                "2026-07-18/00-00-00",
                ["random_budget", "uncertainty_budget"],
                ["highway-v0"],
                30,
                1,
                None,
                None,
                "formal_protocol.yaml",
                {
                    "random_budget": ["highway-v0"],
                    "uncertainty_budget": ["highway-v0"],
                },
                seed_start=100,
                execution_horizon_by_group_env={
                    "random_budget": {"highway-v0": horizon},
                    "uncertainty_budget": {"highway-v0": horizon},
                },
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed_policy"], "fixed_per_setting")
        self.assertEqual(manifest["seed_start"], 100)
        self.assertEqual(manifest["seed_labels"], list(range(100, 130)))
        self.assertEqual(manifest["simulation_duration"], 30)
        self.assertEqual(manifest["episode_duration_s"], 30)
        self.assertEqual(manifest["policy_frequency_hz"], 10)
        self.assertEqual(manifest["simulation_frequency_hz"], 10)
        self.assertEqual(manifest["expected_policy_steps"], 300)

    def test_highway_duration_seconds_resolve_to_policy_frames_without_unit_drift(self):
        cfg = {
            "env_type": "highway-v0",
            "simulation_duration": 30,
            "policy_frequency": 10,
            "simulation_frequency": 10,
        }
        horizon = resolve_policy_execution_horizon(cfg)

        self.assertEqual(horizon.expected_policy_steps, 300)
        self.assertEqual(build_env_config(cfg)["duration"], 30)

    def test_unknown_group_is_rejected_instead_of_silently_filtered(self):
        with self.assertRaisesRegex(ValueError, "does not define requested group"):
            iter_selected_groups({"always_fast": {}}, ["always-fast"])

    def test_v12_main_preflight_accepts_only_the_frozen_six_arm_contract(self):
        protocol = load_formal_protocol(Path("formal_protocol.yaml"))
        cfg = load_formal_base_config(protocol)
        cfg["SILICONFLOW_API_KEY"] = "test-only-key"
        groups = list(
            protocol["tvt_submission_contract"]["evidence_artifacts"]["artifacts"][
                "main_results"
            ]["required_groups"]
        )

        _validate_v12_main_preflight(
            protocol=protocol,
            base_cfg=cfg,
            partition="main",
            selected_groups=groups,
            envs=["highway-v0"],
            seed_labels=list(range(4000, 4030)),
            episodes=1,
        )
        with self.assertRaisesRegex(ValueError, "complete ordered group contract"):
            _validate_v12_main_preflight(
                protocol=protocol,
                base_cfg=cfg,
                partition="main",
                selected_groups=[group for group in groups if group != "always_slow"],
                envs=["highway-v0"],
                seed_labels=list(range(4000, 4030)),
                episodes=1,
            )

    def test_v13_preflight_locks_main_and_mechanism_cohorts(self):
        protocol = load_formal_protocol(Path("formal_protocol.yaml"))
        cfg = load_formal_base_config(protocol)
        cfg.update(
            {
                "LLM_PROVIDER": "siliconflow",
                "SILICONFLOW_CHAT_MODEL": "Qwen/Qwen3-8B",
            }
        )
        groups = list(
            protocol["tvt_submission_contract"]["evidence_artifacts"]["artifacts"][
                "main_results"
            ]["required_groups"]
        )

        _validate_v13_preflight(
            protocol=protocol,
            base_cfg=cfg,
            partition="main",
            selected_groups=groups,
            envs=["highway-v0"],
            seed_labels=list(range(5000, 5030)),
            episodes=1,
            simulation_duration=30,
        )
        _validate_v13_preflight(
            protocol=protocol,
            base_cfg=cfg,
            partition="mechanism",
            selected_groups=["always_fast", "always_slow"],
            envs=["highway-v0"],
            seed_labels=list(range(6000, 6020)),
            episodes=1,
            simulation_duration=30,
        )
        with self.assertRaisesRegex(ValueError, "seed cohort drift"):
            _validate_v13_preflight(
                protocol=protocol,
                base_cfg=cfg,
                partition="main",
                selected_groups=groups,
                envs=["highway-v0"],
                seed_labels=list(range(5001, 5031)),
                episodes=1,
                simulation_duration=30,
            )

    def test_v13_resolved_cell_rejects_group_frequency_override(self):
        protocol = load_formal_protocol(Path("formal_protocol.yaml"))
        cfg = load_formal_base_config(protocol)
        cfg.update(
            {
                "LLM_PROVIDER": "siliconflow",
                "SILICONFLOW_CHAT_MODEL": "Qwen/Qwen3-8B",
            }
        )
        drifted_spec = copy.deepcopy(protocol["groups"]["always_fast"])
        drifted_spec.setdefault("runtime_overrides", {})["policy_frequency"] = 5
        resolved = build_group_config(
            cfg,
            "always_fast",
            drifted_spec,
            "highway-v0",
            1,
            Path("unused"),
            protocol,
        )

        with self.assertRaisesRegex(ValueError, "differs from formal contract"):
            _validate_v13_resolved_cell(protocol, "main", resolved)

    def test_v13_formal_defaults_select_main_matrix_and_seed_block(self):
        captured = {}

        def stop_after_preflight(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop-before-execution")

        with patch.object(
            main_table_module, "_validate_v13_preflight", side_effect=stop_after_preflight
        ):
            with self.assertRaisesRegex(RuntimeError, "stop-before-execution"):
                run_main_table(
                    [
                        "--protocol",
                        "formal_protocol.yaml",
                        "--config",
                        "config.yaml",
                        "--mode",
                        "formal_run",
                    ]
                )
        self.assertEqual(captured["partition"], "main")
        self.assertEqual(captured["envs"], ["highway-v0"])
        self.assertEqual(captured["seed_labels"], list(range(5000, 5030)))
        self.assertEqual(len(captured["selected_groups"]), 6)

    def test_partition_is_enumerated_and_nonformal_requires_authorization(self):
        with self.assertRaises(SystemExit):
            parse_args(["--partition", "typo"])
        with self.assertRaisesRegex(ValueError, "allow-nonformal"):
            run_main_table(
                [
                    "--protocol",
                    "formal_protocol.yaml",
                    "--config",
                    "config.yaml",
                    "--mode",
                    "quick_check",
                    "--groups",
                    "always_fast",
                    "--envs",
                    "highway-v0",
                    "--seeds",
                    "1",
                ]
            )

    def test_seed_block_order_is_deterministic_independent_of_worker_count(self):
        groups = ["rgd_fixed_policy", "always_fast", "always_slow", "random_budget"]
        first = _ordered_groups_for_seed(groups, 5000, True)
        second = _ordered_groups_for_seed(groups, 5000, True)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted(groups))
        tasks = [{"seed": seed, "cells": []} for seed in (5000, 5001, 5002)]
        self.assertEqual(
            [item["seed"] for item in _execute_seed_blocks(tasks, 1)],
            [5000, 5001, 5002],
        )
        self.assertEqual(
            [item["seed"] for item in _execute_seed_blocks(tasks, 2)],
            [5000, 5001, 5002],
        )

    def test_summary_rows_are_limited_to_the_active_seed_cohort(self):
        rows = [
            {"group": "always_fast", "env": "highway-v0", "seed_idx": seed}
            for seed in (4999, 5000, 5001, 5030)
        ]
        selected = _filter_active_cohort_rows(
            rows,
            group_name="always_fast",
            envs=["highway-v0"],
            seed_labels=[5000, 5001],
        )
        self.assertEqual([row["seed_idx"] for row in selected], [5000, 5001])

    def test_completed_metrics_closes_valid_timeout_failure_and_drop_lifecycle(self):
        aggregate = MetricsAggregator("always_slow", "unused")
        aggregate.physical_metrics_list = [
            {
                "total_frames": 4,
                "collision": False,
                "success_completion": True,
                "avg_speed": 1.0,
                "driving_distance": 2.0,
                "episode_total_reward": 1.0,
            }
        ]
        def issuance(request_id):
            return {
                "closed_loop_latency_issuance_event": True,
                "closed_loop_latency_issued_request_id": request_id,
                "closed_loop_latency_issued_response_outcome": "pending",
            }
        def terminal(request_id, outcome, wall, e2e, *, issue=False):
            return {
                **(issuance(request_id) if issue else {}),
                "closed_loop_latency_terminal_event": True,
                "closed_loop_latency_terminal_request_id": request_id,
                "closed_loop_latency_terminal_response_outcome": outcome,
                "slow_response_wall_latency_s": wall,
                "closed_loop_latency_realized_seconds": e2e,
                "closed_loop_latency_realized_source": "simulator_frame_delta",
            }
        events = [
            issuance("a"),
            terminal("a", "valid", 1.0, 0.1),
            {**issuance("b"), **terminal("b", "timeout", 2.0, 0.2)},
            {**issuance("c"), **terminal("c", "failure", 3.0, 0.3)},
            issuance("d"),
        ]
        aggregate.physical_metrics_list[0]["total_frames"] = len(events)
        metrics = _completed_metrics(
            aggregate,
            group_name="always_slow",
            runtime_seconds=[0.01] * len(events),
            events=events,
            dropped_rows=[{"request_id": "d", "terminal_outcome": "dropped_at_episode_end"}],
            runtime_integrity_checks=[True],
            protocol={"claim_guardrails": {}},
        )
        self.assertEqual(metrics["request_count"], 4)
        self.assertEqual(metrics["valid_response_count"], 1)
        self.assertEqual(metrics["timeout_count"], 1)
        self.assertEqual(metrics["failure_count"], 1)
        self.assertEqual(metrics["dropped_at_episode_end_count"], 1)
        self.assertTrue(metrics["request_lifecycle_closed"])
        self.assertAlmostEqual(metrics["terminal_wall_latency_mean_s"], 2.0)
        self.assertAlmostEqual(metrics["simulator_e2e_latency_mean_s"], 0.2)

    def test_completed_metrics_reports_runtime_identity_violation_without_clean_flag(self):
        aggregate = MetricsAggregator("always_fast", "unused")
        aggregate.physical_metrics_list = [{
            "total_frames": 1,
            "collision": False,
            "success_completion": True,
            "avg_speed": 1.0,
            "driving_distance": 1.0,
            "episode_total_reward": 1.0,
        }]
        metrics = _completed_metrics(
            aggregate,
            group_name="always_fast",
            runtime_seconds=[0.01],
            events=[{"done": True}],
            runtime_integrity_checks=[True, False],
            protocol={"claim_guardrails": {}},
        )
        self.assertFalse(metrics["runtime_integrity_clean"])
        self.assertEqual(metrics["runtime_integrity_violation_rate"], 0.5)
        self.assertIsNone(metrics["terminal_wall_latency_mean_s"])
        self.assertIsNone(metrics["simulator_e2e_latency_mean_s"])

    def test_protocol_json_writer_replaces_nonfinite_values_with_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonfinite.json"
            dump_json(path, {"nan": float("nan"), "inf": float("inf"), "nested": [float("-inf")]})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload, {"nan": None, "inf": None, "nested": [None]})


if __name__ == "__main__":
    unittest.main()
