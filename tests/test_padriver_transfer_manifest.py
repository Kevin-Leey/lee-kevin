import json
import tempfile
import unittest
from pathlib import Path

from tools.run_main_table_runtime import build_group_config, load_formal_base_config, load_formal_protocol
from tools.run_padriver_transfer_smoke import (
    _apply_transfer_overrides,
    _effective_config_matches_request,
    _save_transfer_experiment_snapshot,
    _transfer_trace_files_complete,
    _validate_transfer_snapshot_pair,
    parse_args,
)


class PaDriverTransferManifestTest(unittest.TestCase):
    def _write_snapshot_pair(self, root: Path):
        args = parse_args(
            [
                "--episodes",
                "2",
                "--simulation-duration",
                "30",
                "--policy-frequency",
                "10",
                "--simulation-frequency",
                "10",
                "--preserve-executor-action",
            ]
        )
        protocol = load_formal_protocol(Path("formal_protocol.yaml"))
        group_cfg = dict(protocol["groups"]["always_fast"])
        cfg = build_group_config(
            load_formal_base_config(protocol),
            "always_fast",
            group_cfg,
            "highway-v0",
            2,
            root,
            protocol,
        )
        effective = _apply_transfer_overrides(
            cfg, args, lane_count=5, density=3.0, seed=17
        )
        _save_transfer_experiment_snapshot(cfg, root, 17, effective)
        return effective

    def test_snapshot_pair_records_executed_zero_latency_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            effective = self._write_snapshot_pair(root)
            _validate_transfer_snapshot_pair(root, effective)

            manifest = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
            snapshot = json.loads((root / "experiment_snapshot.json").read_text(encoding="utf-8"))
            for payload in (manifest, snapshot):
                contract = payload["transfer_effective_config"]
                self.assertEqual(contract["lanes_count"], 5)
                self.assertEqual(contract["vehicles_density"], 3.0)
                self.assertEqual(contract["vehicle_count"], 30)
                self.assertEqual(contract["policy_frequency_hz"], 10)
                self.assertEqual(contract["simulation_frequency_hz"], 10)
                self.assertEqual(contract["resolved_env_seeds"], [17, 17])
                self.assertEqual(
                    contract["episode_contracts"],
                    [
                        {"episode_index": 0, "episode_id": 34, "env_seed": 17},
                        {"episode_index": 1, "episode_id": 35, "env_seed": 17},
                    ],
                )
                self.assertTrue(contract["zero_additional_latency"])
                self.assertEqual(contract["additional_latency_s"], 0.0)
                self.assertEqual(contract["rgd_predicted_slow_latency_s"], 0.0)
                self.assertEqual(
                    contract["rgd_predicted_slow_latency_source"],
                    "lane_transfer_configured_extra_latency",
                )
                self.assertFalse(contract["risk_calibration"]["enable"])
                self.assertTrue(contract["preserve_executor_action"])
                self.assertFalse(payload["config"]["slow_thinking"]["risk_calibration"]["enable"])
                self.assertTrue(payload["config"]["preserve_executor_action"])
            self.assertEqual(manifest["config"], snapshot["config"])
            self.assertEqual(manifest["protocol_hash"], snapshot["protocol_hash"])
            self.assertEqual(manifest["config_hash"], snapshot["config_hash"])

    def test_integrity_check_rejects_snapshot_config_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            effective = self._write_snapshot_pair(root)
            snapshot_path = root / "experiment_snapshot.json"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            snapshot["config"]["lanes_count"] = 6
            snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lanes_count"):
                _validate_transfer_snapshot_pair(root, effective)

    def test_snapshot_provenance_carries_zero_delay_guard_and_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = parse_args(
                [
                    "--episodes", "1", "--simulation-duration", "30",
                    "--policy-frequency", "10", "--simulation-frequency", "10",
                    "--preserve-executor-action", "--release-dominance-guard",
                    "--release-dominance-margin", "0.05", "--min-observation-frames", "2",
                ]
            )
            protocol = load_formal_protocol(Path("formal_protocol.yaml"))
            group_cfg = dict(protocol["groups"]["rgd_fixed_policy"])
            cfg = build_group_config(
                load_formal_base_config(protocol), "rgd_fixed_policy", group_cfg,
                "highway-v0", 1, root, protocol,
            )
            effective = _apply_transfer_overrides(
                cfg, args, lane_count=4, density=3.0, seed=3
            )
            _save_transfer_experiment_snapshot(cfg, root, 3, effective)
            _validate_transfer_snapshot_pair(root, effective)
            manifest = json.loads((root / "runtime_manifest.json").read_text(encoding="utf-8"))
            contract = manifest["transfer_effective_config"]
            self.assertEqual(contract["rgd_min_observation_frames"], 2)
            self.assertEqual(contract["release_dominance_guard"], {
                "enable": True,
                "risk_margin": 0.05,
                "require_strict_improvement": True,
                "scope": "query_equals_release_zero_delay",
            })

    def test_resume_matching_uses_protocol_warmup_when_cli_is_omitted(self):
        args = parse_args(["--preserve-executor-action"])
        effective = {
            "env_type": "highway-v0",
            "lanes_count": 4,
            "vehicles_density": 3.0,
            "vehicle_count": 30,
            "simulation_duration": 30,
            "policy_frequency_hz": 10,
            "simulation_frequency_hz": 10,
            "fixed_seed_override": 0,
            "resolved_env_seeds": [0],
            "episode_contracts": [{"episode_index": 0, "episode_id": 0, "env_seed": 0}],
            "additional_latency_s": 0.0,
            "zero_additional_latency": True,
            "rgd_predicted_slow_latency_s": 0.0,
            "rgd_predicted_slow_latency_source": "lane_transfer_configured_extra_latency",
            "closed_loop_latency_replay": {
                "enable": False, "extra_latency_s": 0.0,
                "delay_steps": 0, "target_systems": ["slow"],
            },
            "release_dominance_guard": {
                "enable": False, "risk_margin": 0.0,
                "require_strict_improvement": True,
                "scope": "query_equals_release_zero_delay",
            },
            "target_lane_projection_enable": False,
            "preserve_executor_action": True,
            "risk_calibration": {"enable": False},
            "rgd_min_observation_frames": 2,
        }
        self.assertTrue(_effective_config_matches_request(effective, args, 4, 3.0, 0, 2))
        self.assertFalse(_effective_config_matches_request(effective, args, 4, 3.0, 0, 1))

    def test_resume_requires_one_parseable_trace_triplet_per_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            event_dir = root / "event_logs"
            ep0 = root / "ep_0"
            ep1 = root / "ep_1"
            event_dir.mkdir(parents=True)
            ep0.mkdir()
            ep1.mkdir()
            for index, episode_dir in enumerate((ep0, ep1)):
                (episode_dir / f"highway_{index}_physical_frames.json").write_text(
                    json.dumps({"frames": [{"frame_id": 0}]}), encoding="utf-8"
                )
                (episode_dir / f"highway_{index}_reasoning_records.json").write_text(
                    json.dumps({"records": [{"frame_id": 0}]}), encoding="utf-8"
                )
                (event_dir / f"event_log_highway_{index}_{index}.json").write_text(
                    json.dumps({"events": [{"frame": 0}]}), encoding="utf-8"
                )

            self.assertTrue(_transfer_trace_files_complete(root, episodes=2))
            (ep1 / "highway_1_physical_frames.json").unlink()
            self.assertFalse(_transfer_trace_files_complete(root, episodes=2))


if __name__ == "__main__":
    unittest.main()
