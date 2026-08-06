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
    _is_formal_closed_loop_latency_sweep,
    _is_formal_evidence_acquisition,
    _is_formal_mechanism_trace_acquisition,
    _scope_label,
    _write_report,
    parse_args,
)


class LatencySweepScopeTests(unittest.TestCase):
    def test_auto_recognizes_formal_mechanism_trace_acquisition(self):
        args = parse_args(
            [
                "--groups",
                "always_fast",
                "--latencies",
                "0.0",
                "--seed-start",
                "6000",
                "--seeds",
                "20",
            ]
        )
        self.assertEqual(_scope_label(args), FORMAL_MECHANISM_TRACE_SCOPE)
        self.assertFalse(_is_formal_closed_loop_latency_sweep(args))
        self.assertTrue(_is_formal_mechanism_trace_acquisition(args))
        self.assertTrue(_is_formal_evidence_acquisition(args))

    def test_small_always_fast_probe_remains_smoke(self):
        args = parse_args(
            ["--groups", "always_fast", "--latencies", "0.0", "--seed-start", "6000", "--seeds", "2"]
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
                "0.0",
                "--seed-start",
                "6000",
                "--seeds",
                "20",
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
                        "0.0",
                        "--seed-start",
                        "6000",
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
                "0.0",
                "--seed-start",
                "6000",
                "--seeds",
                "20",
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
                "0.0",
                "--seed-start",
                "6000",
                "--seeds",
                "20",
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
