import argparse
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
    _merge_group_run_rows,
    _resume_row_is_complete,
    _validate_v12_main_preflight,
)
from tools.run_main_table_runtime import (
    build_group_config,
    iter_selected_groups,
    load_formal_base_config,
    load_formal_protocol,
)
from tools.result_bundle_pipeline import write_result_bundle_manifest


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
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed_policy"], "fixed_per_setting")
        self.assertEqual(manifest["seed_start"], 100)
        self.assertEqual(manifest["seed_labels"], list(range(100, 130)))

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


if __name__ == "__main__":
    unittest.main()
