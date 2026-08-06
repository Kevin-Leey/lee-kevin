import json
import math
from pathlib import Path

import run_dilu

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.kinematic_risk import KinematicRiskActionProvider
from dilu.evaluation.physical_metrics import PhysicalMetricsRecorder
from dilu.evaluation.reasoning_recorder import ReasoningRecord, ReasoningRecorder
from dilu.runtime_episode_setup import _slow_executor_kwargs
from tools.run_main_table_runtime import build_group_config


ROOT = Path(__file__).resolve().parents[1]


def test_direct_entrypoint_runs_the_current_rgd_stack_without_legacy_langchain(tmp_path):
    output = tmp_path / "entry"

    assert run_dilu.main(
        [
            "--config",
            str(ROOT / "config.yaml"),
            "--result-folder",
            str(output),
            "--episodes",
            "1",
            "--seed-start",
            "123",
            "--simulation-duration",
            "2",
            "--force-fast",
        ]
    ) == 0

    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["seeds"] == [123]
    assert summary["total_frames"] == 2
    events = json.loads(
        (output / "event_logs" / "event_log_highway_0_0.json").read_text(
            encoding="utf-8"
        )
    )["events"]
    assert events[-1]["episode_done"] is True
    assert (output / "ep_0" / "reasoning_records_0.json").is_file()


def test_kinematic_executor_uses_the_shared_effective_action_universe():
    state = DrivingState(
        legal_actions=[int(ActionType.IDLE), int(ActionType.FASTER), int(ActionType.SLOWER)]
    )
    state.__dict__["_safety_cost_decomposition"] = {
        int(ActionType.IDLE): {"safety": 0.3, "total": 0.35},
        int(ActionType.FASTER): {"safety": 0.7, "total": 0.7},
        int(ActionType.SLOWER): {"safety": 0.1, "total": 0.2},
    }

    decision = KinematicRiskActionProvider()(state, {})

    assert decision.action == int(ActionType.SLOWER)
    assert decision.stats["slow_reasoning_mode"] == "kinematic_risk"
    assert decision.agent_opinions["risk_scores_by_action"][int(ActionType.SLOWER)] == 0.1


def test_fast_only_route_does_not_require_a_slow_executor_configuration():
    assert _slow_executor_kwargs(
        {"system_routing": {"simple": "fast", "complex": "fast"}}
    ) == {}


def test_recorders_write_strict_json_when_observations_include_infinity(tmp_path):
    physical = PhysicalMetricsRecorder(0, 1, str(tmp_path))
    physical.record_frame(
        0,
        {"speed": 8.0, "ttc": math.inf, "thw": math.inf, "pos": (math.nan, 0.0)},
        action=int(ActionType.IDLE),
        info={"distance": math.inf},
    )
    physical.save()
    physical_payload = json.loads(
        (tmp_path / "physical_frames_0.json").read_text(encoding="utf-8")
    )
    assert physical_payload["frames"][0]["ttc"] is None
    assert physical_payload["frames"][0]["info"]["distance"] is None

    reasoning = ReasoningRecorder(0, str(tmp_path))
    reasoning.records.append(
        ReasoningRecord(
            frame_id=0,
            timestamp=0.0,
            scenario_description="state",
            available_actions="actions",
            driving_intention="drive",
            full_response="fast",
            predicted_action_id=int(ActionType.IDLE),
            predicted_action_name="IDLE",
            inference_latency=0.0,
            system_used="fast",
            route_reason="test",
            rgd_execution_route_score=0.0,
            fast_rule_name="test",
            fast_smoothness_override=False,
            slow_reasoning_mode="none",
            slow_reasoning_success=False,
            slow_reasoning_failure_reason="",
            rgd_subordinate_diagnostics={"distance": math.inf},
        )
    )
    reasoning.save()
    reasoning_payload = json.loads(
        (tmp_path / "reasoning_records_0.json").read_text(encoding="utf-8")
    )
    assert reasoning_payload["analysis_records"][0]["rgd_subordinate_diagnostics"]["distance"] is None


def test_group_config_applies_environment_runtime_settings_before_group_overrides(tmp_path):
    cfg = build_group_config(
        {"simulation_duration": 10, "slow_thinking": {"max_tokens": 16}},
        "group",
        {"id": "G", "runtime_overrides": {"simulation_duration": 30}},
        "merge-v0",
        1,
        tmp_path,
        {
            "protocol_name": "test",
            "runtime_config_by_env": {
                "merge-v0": {
                    "simulation_duration": 20,
                    "policy_frequency": 4,
                    "slow_thinking": {"executor": "kinematic_risk"},
                }
            },
        },
    )

    assert cfg["simulation_duration"] == 30
    assert cfg["policy_frequency"] == 4
    assert cfg["slow_thinking"] == {
        "max_tokens": 16,
        "executor": "kinematic_risk",
    }
