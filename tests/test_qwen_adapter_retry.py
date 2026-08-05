import os
import unittest
from unittest.mock import MagicMock, patch

from dilu.driver_agent.qwen_adapter import QwenChatModel


class QwenAdapterRetryTests(unittest.TestCase):
    def build_model(self, **overrides):
        env = {
            "QWEN_CHAT_MODEL": "test-model",
            "QWEN_CHAT_MODEL_CONFIG": "test-model",
            "QWEN_REQUEST_TIMEOUT_S": "1",
            "LLM_PROVIDER": "openai_compatible",
            "OPENAI_BASE_URL": "https://example.invalid/v1",
            "OPENAI_API_KEY": "test-key",
            "LLM_MAX_ATTEMPTS": "3",
            "LLM_RETRY_BACKOFF_S": "0",
            "LLM_FORCE_JSON_RESPONSE": "false",
        }
        env.update(overrides)
        with patch.dict(os.environ, env, clear=False):
            return QwenChatModel(max_tokens=16)

    @staticmethod
    def response(status, payload=None, text=""):
        response = MagicMock()
        response.status_code = status
        response.text = text
        response.json.return_value = payload or {}
        return response

    def test_transient_503_is_retried(self):
        model = self.build_model()
        session = MagicMock()
        session.post.side_effect = [
            self.response(503, text="temporarily unavailable"),
            self.response(503, text="temporarily unavailable"),
            self.response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]

        response = model._post_with_retry(session, {})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.post.call_count, 3)

    def test_nonretryable_auth_error_returns_immediately(self):
        model = self.build_model()
        session = MagicMock()
        session.post.return_value = self.response(401, text="unauthorized")

        response = model._post_with_retry(session, {})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(session.post.call_count, 1)

    def test_openai_compatible_payload_filters_qwen_thinking_fields(self):
        model = self.build_model(QWEN_ENABLE_THINKING="true", QWEN_THINKING_BUDGET="128")

        payload = model._build_payload("system", "user")

        self.assertNotIn("enable_thinking", payload)
        self.assertNotIn("thinking_budget", payload)

    def test_qwen_siliconflow_payload_keeps_configured_thinking_fields(self):
        env = {
            "QWEN_CHAT_MODEL": "Qwen/Qwen3-8B",
            "QWEN_CHAT_MODEL_CONFIG": "Qwen/Qwen3-8B",
            "QWEN_REQUEST_TIMEOUT_S": "1",
            "LLM_PROVIDER": "siliconflow",
            "SILICONFLOW_BASE_URL": "https://example.invalid/v1",
            "SILICONFLOW_API_KEY": "test-key",
            "QWEN_ENABLE_THINKING": "true",
            "QWEN_THINKING_BUDGET": "128",
            "LLM_FORCE_JSON_RESPONSE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            model = QwenChatModel(max_tokens=16)

        payload = model._build_payload("system", "user")

        self.assertIs(payload["enable_thinking"], True)
        self.assertEqual(payload["thinking_budget"], 128)


if __name__ == "__main__":
    unittest.main()
