import csv
import json
from pathlib import Path

import pytest
import yaml

from tools.run_multi_llm_executor_probe import (
    _resolved_retry_contract,
    _summarise_model,
    _write_model_protocol,
)


def test_model_protocol_overrides_executor_without_serializing_credentials(tmp_path: Path):
    base_protocol = tmp_path / "base.yaml"
    output_protocol = tmp_path / "model.yaml"
    base_protocol.write_text(
        "runtime_config: {}\ngroups:\n  rgd_fixed_policy:\n    runtime_overrides: {}\n",
        encoding="utf-8",
    )

    _write_model_protocol(
        base_protocol,
        output_protocol,
        {"provider": "openai_compatible", "model": "gpt-5.6-sol", "api_key": "must-not-persist"},
        {"OPENAI_API_KEY": "base-secret", "QWEN_REQUEST_TIMEOUT_S": 120.0},
        min_observation_frames=2,
    )

    rendered = output_protocol.read_text(encoding="utf-8")
    protocol = yaml.safe_load(rendered)
    runtime = protocol["runtime_config"]
    assert "base-secret" not in rendered
    assert "must-not-persist" not in rendered
    assert runtime["LLM_PROVIDER"] == "openai_compatible"
    assert runtime["OPENAI_CHAT_MODEL"] == "gpt-5.6-sol"
    assert runtime["rgd_min_observation_frames"] == 2
    assert runtime["LLM_MAX_ATTEMPTS"] == 3
    assert runtime["LLM_RETRY_BACKOFF_S"] == 0.5
    assert "QWEN_ENABLE_THINKING" not in runtime
    assert "QWEN_THINKING_BUDGET" not in runtime
    assert "OPENAI_API_KEY" not in runtime


def _write_probe_run(
    root: Path,
    *,
    provider: str = "siliconflow",
    model: str = "Qwen/Qwen3-8B",
    records=None,
) -> Path:
    result_dir = root / "highway" / "seed_7"
    trace_dir = result_dir / "ep_7"
    trace_dir.mkdir(parents=True)
    records = list(records or [])
    retry = _resolved_retry_contract({})
    (result_dir / "runtime_manifest.json").write_text(
        json.dumps(
            {
                "llm_backend": {
                    "provider": provider,
                    "requested_model": model,
                    "retry_contract": retry,
                }
            }
        ),
        encoding="utf-8",
    )
    (trace_dir / "highway_7_reasoning_records.json").write_text(
        json.dumps({"episode_id": 7, "record_count": len(records), "analysis_records": records}),
        encoding="utf-8",
    )
    rows_path = root / "rgd_fixed_policy_run_rows.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "env", "seed_idx", "episodes_run", "total_frames", "result_dir", "collision_rate",
                "success_rate", "avg_driving_distance", "avg_speed_all_frames", "avg_speed_safety_qualified",
                "slow_call_rate", "llm_action_preservation_rate", "avg_runtime_per_frame", "latency_p95",
                "route_action_preservation_rate", "safety_override_rate", "independent_selective_routing_gain",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "env": "highway-v0",
                "seed_idx": 7,
                "episodes_run": 1,
                "total_frames": len(records),
                "result_dir": str(result_dir),
            }
        )
    return rows_path


def test_summary_counts_every_slow_attempt_and_failure_type(tmp_path: Path):
    records = [
        {"frame_id": 0, "system_used": "fast", "slow_reasoning_success": False},
        {"frame_id": 1, "system_used": "slow", "slow_reasoning_success": True, "slow_reasoning_failure_reason": ""},
        {
            "frame_id": 2,
            "system_used": "fast_after_slow_failure",
            "slow_reasoning_success": False,
            "slow_reasoning_failure_reason": "structured_parse_failed:ValueError",
        },
        {
            "frame_id": 3,
            "system_used": "fast_after_slow_failure",
            "slow_reasoning_success": False,
            "slow_reasoning_failure_reason": "llm_invoke_timeout:60.00s",
        },
    ]
    rows_path = _write_probe_run(tmp_path, records=records)

    summary = _summarise_model(
        {"provider": "siliconflow", "model": "Qwen/Qwen3-8B", "label": "Qwen3-8B"},
        rows_path,
        expected_envs=["highway-v0"],
        expected_seeds=[7],
        retry_contract=_resolved_retry_contract({}),
    )

    assert summary["slow_attempts"] == 3
    assert summary["slow_successes"] == 1
    assert summary["slow_failures"] == 2
    assert summary["slow_parse_failures"] == 1
    assert summary["slow_fallbacks"] == 2
    assert summary["slow_call_success_rate"] == pytest.approx(1 / 3)
    assert summary["slow_call_rate"] == pytest.approx(3 / 4)


@pytest.mark.parametrize(
    ("provider", "model", "error"),
    [
        ("openai_compatible", "Qwen/Qwen3-8B", "LLM manifest drift"),
        ("siliconflow", "Qwen/Qwen2.5-7B-Instruct", "LLM manifest drift"),
    ],
)
def test_summary_fails_closed_on_manifest_model_drift(tmp_path: Path, provider: str, model: str, error: str):
    rows_path = _write_probe_run(
        tmp_path,
        provider=provider,
        model=model,
        records=[{"frame_id": 0, "system_used": "fast", "slow_reasoning_success": False}],
    )

    with pytest.raises(ValueError, match=error):
        _summarise_model(
            {"provider": "siliconflow", "model": "Qwen/Qwen3-8B", "label": "Qwen3-8B"},
            rows_path,
            expected_envs=["highway-v0"],
            expected_seeds=[7],
            retry_contract=_resolved_retry_contract({}),
        )


def test_summary_fails_closed_when_expected_seed_is_missing(tmp_path: Path):
    rows_path = _write_probe_run(
        tmp_path,
        records=[{"frame_id": 0, "system_used": "fast", "slow_reasoning_success": False}],
    )

    with pytest.raises(ValueError, match="incomplete model matrix"):
        _summarise_model(
            {"provider": "siliconflow", "model": "Qwen/Qwen3-8B", "label": "Qwen3-8B"},
            rows_path,
            expected_envs=["highway-v0"],
            expected_seeds=[7, 8],
            retry_contract=_resolved_retry_contract({}),
        )


def test_summary_fails_closed_when_retry_contract_is_missing(tmp_path: Path):
    rows_path = _write_probe_run(
        tmp_path,
        records=[{"frame_id": 0, "system_used": "fast", "slow_reasoning_success": False}],
    )
    manifest_path = tmp_path / "highway" / "seed_7" / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["llm_backend"]["retry_contract"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="retryable_http_statuses"):
        _summarise_model(
            {"provider": "siliconflow", "model": "Qwen/Qwen3-8B", "label": "Qwen3-8B"},
            rows_path,
            expected_envs=["highway-v0"],
            expected_seeds=[7],
            retry_contract=_resolved_retry_contract({}),
        )
