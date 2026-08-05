import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from tools.run_closed_loop_latency_sweep import (
    FORMAL_LATENCY_SWEEP_SCOPE,
    FORMAL_MECHANISM_TRACE_SCOPE,
    SMOKE_SCOPE,
    _apply_latency_overrides,
    _is_formal_closed_loop_latency_sweep,
    _is_formal_evidence_acquisition,
    _is_formal_mechanism_trace_acquisition,
    _scope_label,
    _write_report,
    parse_args,
)


class LatencySweepScopeTests(unittest.TestCase):
    def test_latency_override_preserves_release_arbitration_config(self):
        release_cfg = {
            "release_opportunity_revalidation": {
                "enable": True,
                "action_cost_alignment": {
                    "enable": True,
                    "cost_margin": 0.02,
                },
            }
        }
        cfg = {
            "closed_loop_latency_replay": release_cfg,
            "_paper_protocol_config": {
                "runtime_config": {"closed_loop_latency_replay": release_cfg.copy()}
            },
        }
        args = Namespace(
            episodes=1,
            simulation_duration=30,
            policy_frequency=10,
            simulation_frequency=10,
            scope="smoke_only",
        )
        plan = {
            "acquisition_protocol_hash": "acquisition-hash",
            "schedule_hash": "schedule-hash",
        }
        schedule_entry = {
            "setting_contract_hash": "setting-hash",
            "seed_block_index": 0,
            "latency_order_position": 0,
        }

        _apply_latency_overrides(cfg, args, 1.7, 5000, plan, schedule_entry)

        for target in (cfg, cfg["_paper_protocol_config"]["runtime_config"]):
            replay = target["closed_loop_latency_replay"]
            self.assertTrue(replay["release_opportunity_revalidation"]["enable"])
            self.assertTrue(
                replay["release_opportunity_revalidation"]["action_cost_alignment"]["enable"]
            )
            self.assertEqual(replay["delay_steps"], 17)
            self.assertEqual(replay["extra_latency_s"], 1.7)

    def test_auto_recognizes_formal_mechanism_trace_acquisition(self):
        args = parse_args(
            [
                "--groups",
                "always_fast",
                "--latencies",
                "1.7",
                "--seed-start",
                "160",
                "--seeds",
                "30",
            ]
        )
        self.assertEqual(_scope_label(args), FORMAL_MECHANISM_TRACE_SCOPE)
        self.assertFalse(_is_formal_closed_loop_latency_sweep(args))
        self.assertTrue(_is_formal_mechanism_trace_acquisition(args))
        self.assertTrue(_is_formal_evidence_acquisition(args))

    def test_small_always_fast_probe_remains_smoke(self):
        args = parse_args(
            ["--groups", "always_fast", "--latencies", "1.7", "--seed-start", "160", "--seeds", "2"]
        )
        self.assertEqual(_scope_label(args), SMOKE_SCOPE)
        self.assertFalse(_is_formal_mechanism_trace_acquisition(args))
        self.assertFalse(_is_formal_evidence_acquisition(args))

    def test_nonprotocol_frequency_remains_smoke(self):
        args = parse_args(
            [
                "--groups",
                "always_fast",
                "--latencies",
                "1.7",
                "--seeds",
                "30",
                "--policy-frequency",
                "5",
            ]
        )
        self.assertEqual(_scope_label(args), SMOKE_SCOPE)

    def test_auto_preserves_complete_formal_latency_sweep(self):
        args = parse_args(
            [
                "--groups",
                "rgd_fixed_policy",
                "--latencies",
                "0.0",
                "0.7",
                "1.7",
                "2.7",
                "--seeds",
                "30",
            ]
        )
        self.assertEqual(_scope_label(args), FORMAL_LATENCY_SWEEP_SCOPE)
        self.assertTrue(_is_formal_closed_loop_latency_sweep(args))
        self.assertTrue(_is_formal_evidence_acquisition(args))

    def test_explicit_formal_mechanism_scope_rejects_smoke_scale(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    [
                        "--groups",
                        "always_fast",
                        "--latencies",
                        "1.7",
                        "--seeds",
                        "1",
                        "--scope",
                        FORMAL_MECHANISM_TRACE_SCOPE,
                    ]
                )

    def test_explicit_smoke_scope_remains_smoke_at_formal_scale(self):
        args = parse_args(
            [
                "--groups",
                "always_fast",
                "--latencies",
                "1.7",
                "--seeds",
                "30",
                "--scope",
                SMOKE_SCOPE,
            ]
        )
        self.assertEqual(_scope_label(args), SMOKE_SCOPE)
        self.assertFalse(_is_formal_evidence_acquisition(args))

    def test_manifest_records_resolved_formal_mechanism_scope(self):
        args = parse_args(
            [
                "--groups",
                "always_fast",
                "--latencies",
                "1.7",
                "--seed-start",
                "160",
                "--seeds",
                "30",
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_report(root, [], args)
            manifest = json.loads((root / "closed_loop_latency_sweep_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["scope"], FORMAL_MECHANISM_TRACE_SCOPE)
        self.assertEqual(manifest["scope_request"], "auto")
        self.assertEqual(manifest["scope_resolution"], "auto_protocol_match")
        self.assertTrue(manifest["is_formal_mechanism_trace_acquisition"])
        self.assertTrue(manifest["is_formal_evidence_acquisition"])
        self.assertFalse(manifest["is_formal_closed_loop_latency_sweep"])


if __name__ == "__main__":
    unittest.main()
