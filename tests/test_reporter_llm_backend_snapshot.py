import json
import unittest

from dilu.evaluation.reporter import _resolve_llm_backend_snapshot


class ReporterLlmBackendSnapshotTests(unittest.TestCase):
    def test_resolves_siliconflow_qwen_without_credentials(self):
        secret = "siliconflow-secret"
        cfg = {
            "LLM_PROVIDER": "siliconflow",
            "SILICONFLOW_CHAT_MODEL": "Qwen/Qwen3-8B",
            "SILICONFLOW_API_KEY": secret,
            "SILICONFLOW_BASE_URL": (
                "https://endpoint-user:endpoint-password@api.siliconflow.cn/v1"
                "?api_key=query-secret"
            ),
            "QWEN_TEMPERATURE": 0.0,
            "QWEN_REQUEST_TIMEOUT_S": 120.0,
        }

        snapshot = _resolve_llm_backend_snapshot(cfg)

        self.assertEqual(snapshot["provider"], "siliconflow")
        self.assertEqual(snapshot["requested_model"], "Qwen/Qwen3-8B")
        self.assertEqual(snapshot["resolved_chat_model"], "Qwen/Qwen3-8B")
        self.assertEqual(snapshot["base_url_origin"], "https://api.siliconflow.cn")
        serialized = json.dumps(snapshot, sort_keys=True)
        for credential in (secret, "endpoint-user", "endpoint-password", "query-secret"):
            self.assertNotIn(credential, serialized)

    def test_resolves_openai_compatible_gpt_without_credentials(self):
        secret = "openai-compatible-secret"
        cfg = {
            "LLM_PROVIDER": "openai-compatible",
            "OPENAI_CHAT_MODEL": "gpt-5.6-sol",
            "OPENAI_API_KEY": secret,
            "OPENAI_BASE_URL": "https://gateway-user:gateway-password@bizdecipher.com/v1?token=url-secret",
            "QWEN_TEMPERATURE": 0.2,
            "QWEN_REQUEST_TIMEOUT_S": 90.0,
            "LLM_REASONING_EFFORT": "none",
        }

        snapshot = _resolve_llm_backend_snapshot(cfg)

        self.assertEqual(snapshot["provider"], "openai_compatible")
        self.assertEqual(snapshot["requested_model"], "gpt-5.6-sol")
        self.assertEqual(snapshot["resolved_chat_model"], "gpt-5.6-sol")
        self.assertEqual(snapshot["base_url_origin"], "https://bizdecipher.com")
        self.assertEqual(snapshot["temperature"], 0.2)
        self.assertEqual(snapshot["request_timeout_s"], 90.0)
        self.assertEqual(snapshot["reasoning_effort"], "none")
        serialized = json.dumps(snapshot, sort_keys=True)
        for credential in (secret, "gateway-user", "gateway-password", "url-secret"):
            self.assertNotIn(credential, serialized)


if __name__ == "__main__":
    unittest.main()
