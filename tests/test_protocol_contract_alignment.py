"""Static invariants for the paper-facing highway evaluation contract."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
PROTOCOL_PATH = ROOT / "formal_protocol.yaml"


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class ProtocolContractAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.config = load_yaml(CONFIG_PATH)
        self.protocol = load_yaml(PROTOCOL_PATH)
        self.submission = self.protocol["tvt_submission_contract"]
        self.runtime = self.protocol["runtime_config"]

    def test_table_vii_is_a_zero_added_latency_highway_stress_protocol(self):
        table_vii = self.submission["table_vii"]
        transfer = self.config["paper_evidence_protocol"]["transfer"]

        self.assertEqual(table_vii["label"], "Table VII")
        self.assertEqual(table_vii["environment"], "highway-v0")
        self.assertEqual(table_vii["additional_latency_s"], 0.0)
        self.assertEqual(table_vii["latency_safety_reserve_s"], 0.0)
        self.assertEqual(self.submission["transfer_added_delay_s"], 0.0)
        self.assertEqual(transfer["table_label"], "Table VII")
        self.assertEqual(transfer["added_delay_s"], 0.0)
        self.assertEqual(transfer["latency_safety_reserve_s"], 0.0)

    def test_highway_runtime_reserve_and_executor_preservation_are_frozen(self):
        self.assertEqual(self.config["rgd_latency_safety_reserve_s"], 0.0)
        self.assertEqual(self.runtime["rgd_latency_safety_reserve_s"], 0.0)
        self.assertIs(self.config["slow_thinking"]["risk_calibration"]["enable"], False)
        self.assertIs(self.runtime["slow_thinking"]["risk_calibration"]["enable"], False)
        self.assertIs(self.submission["table_vii"]["risk_calibration_enabled"], False)

    def test_retry_contract_matches_runtime_configuration(self):
        retry = self.submission["table_vii"]["llm_retry"]
        self.assertEqual(self.config["LLM_MAX_ATTEMPTS"], 3)
        self.assertEqual(self.runtime["LLM_MAX_ATTEMPTS"], 3)
        self.assertEqual(retry["max_attempts_including_initial_request"], 3)
        self.assertEqual(self.config["LLM_RETRY_BACKOFF_S"], 0.5)
        self.assertEqual(self.runtime["LLM_RETRY_BACKOFF_S"], 0.5)
        self.assertEqual(retry["initial_backoff_s"], 0.5)
        self.assertEqual(retry["schedule"], "exponential")

    def test_only_the_four_declared_executor_models_are_in_scope(self):
        expected = [
            ("siliconflow", "Qwen/Qwen3-8B", "Qwen3-8B"),
            ("siliconflow", "Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
            ("siliconflow", "Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
            ("openai_compatible", "gpt-5.6-sol", "GPT-5.6-sol"),
        ]
        actual_config = [
            (entry["provider"], entry["model"], entry["label"])
            for entry in self.config["MULTI_LLM_MODELS"]
        ]
        actual_protocol = [
            (entry["provider"], entry["model"], entry["label"])
            for entry in self.submission["table_vii"]["slow_executor_models"]
        ]

        self.assertEqual(actual_config, expected)
        self.assertEqual(actual_protocol, expected)
        self.assertNotIn("grok", str(actual_config).lower())
        self.assertNotIn("grok", str(actual_protocol).lower())

    def test_submission_contract_has_no_anonymous_artifact_field(self):
        self.assertNotIn("anonymous_artifact", self.submission)
        self.assertFalse(any("tvt_anonymized_artifact" in str(value) for value in self.submission.values()))


if __name__ == "__main__":
    unittest.main()
