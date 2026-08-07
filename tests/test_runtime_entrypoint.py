import json
import math
import csv
from pathlib import Path
from unittest.mock import patch

from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.kinematic_risk import KinematicRiskActionProvider
from dilu.evaluation.physical_metrics import PhysicalMetricsRecorder
from dilu.evaluation.reasoning_recorder import ReasoningRecord, ReasoningRecorder
from dilu.runtime_episode_setup import _slow_executor_kwargs
from tools import run_main_table
from tools.run_main_table_runtime import build_group_config


ROOT = Path(__file__).resolve().parents[1]


def test_standard_runner_executes_the_current_rgd_stack_without_legacy_langchain(tmp_path):
    result_root = tmp_path / "results"

    assert run_main_table.main(
        [
            "--protocol",
            str(ROOT / "formal_protocol.yaml"),
            "--config",
            str(ROOT / "config.yaml"),
            "--result-root",
            str(result_root),
            "--mode",
            "quick_check",
            "--allow-nonformal",
            "--run-stamp",
            "entry",
            "--groups",
            "always_fast",
            "--envs",
            "highway-v0",
            "--episodes",
            "1",
            "--seed-start",
            "123",
            "--seeds",
            "1",
            "--simulation-duration",
            "1",
            "--no-resume",
        ]
    ) == 0

    output = result_root / "quick_check" / "entry"
    manifest = json.loads(
        (output / "result_bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["seed_start"] == 123
    assert manifest["seeds"] == 1
    result_dir = output / "always_fast" / "highway" / "seed_123"
    events = json.loads(
        next((result_dir / "event_logs").glob("event_log_*.json")).read_text(
            encoding="utf-8"
        )
    )["events"]
    assert events[-1]["episode_done"] is True
    assert len(events) == 10
    assert next((result_dir / "ep_123").glob("reasoning_records_*.json")).is_file()


def test_standard_runner_parallelizes_independent_seed_blocks(tmp_path):
    result_root = tmp_path / "parallel-results"
    argv = [
            "--protocol",
            str(ROOT / "formal_protocol.yaml"),
            "--config",
            str(ROOT / "config.yaml"),
            "--result-root",
            str(result_root),
            "--mode",
            "quick_check",
            "--allow-nonformal",
            "--run-stamp",
            "parallel-entry",
            "--groups",
            "always_fast",
            "--envs",
            "highway-v0",
            "--episodes",
            "1",
            "--seed-start",
            "321",
            "--seeds",
            "2",
            "--simulation-duration",
            "1",
            "--workers",
            "2",
            "--no-resume",
        ]
    assert run_main_table.main(argv) == 0

    bundle = result_root / "quick_check" / "parallel-entry"
    manifest = json.loads(
        (bundle / "result_bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["workers"] == 2
    assert manifest["bundle_completion_state"] == "complete"
    assert manifest["completed_cell_count"] == 2
    for seed in (321, 322):
        marker = bundle / "always_fast" / "highway" / f"seed_{seed}" / "cell_completion_manifest.json"
        assert marker.is_file()
    rows_path = bundle / "always_fast" / "always_fast_run_rows.csv"
    with rows_path.open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["seed_idx"]) for row in rows] == [321, 322]

    # Simulate a main-process failure after workers completed their cell
    # markers but before the shared row CSV survived. Resume must reconstruct
    # rows from authenticated cell artifacts without rerunning simulation.
    rows_path.unlink()
    with patch.object(
        run_main_table, "_run_setting", side_effect=AssertionError("rerun")
    ) as run_setting:
        assert run_main_table.main(argv[:-1]) == 0
    run_setting.assert_not_called()
    with rows_path.open(encoding="utf-8", newline="") as handle:
        recovered_rows = list(csv.DictReader(handle))
    assert [int(row["seed_idx"]) for row in recovered_rows] == [321, 322]


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
