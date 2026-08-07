import pytest

from dilu.evaluation.metrics_aggregator import MetricsAggregator
from dilu.evaluation.physical_metrics import PhysicalMetricsRecorder
from dilu.runtime_episode_setup import create_episode_recorders


def _record_frames(recorder, count, *, speeds=None, rewards=None, crash_last=False):
    speeds = speeds or [10.0] * count
    rewards = rewards or [1.0] * count
    for frame in range(count):
        recorder.record_frame(
            frame,
            {"speed": speeds[frame], "pos": (float(frame), 0.0)},
            action=1,
            reward=rewards[frame],
            crashed=bool(crash_last and frame == count - 1),
            done=bool(frame == count - 1),
        )


def test_expected_frames_drive_continuous_completion_and_threshold_success(tmp_path):
    partial = PhysicalMetricsRecorder(
        1,
        11,
        str(tmp_path),
        expected_total_frames=10,
        success_completion_threshold=0.8,
    )
    _record_frames(partial, 7)

    partial_metrics = partial.calculate_episode_metrics()

    assert partial_metrics.route_completion == pytest.approx(0.7)
    assert partial_metrics.success_completion is False
    assert partial_metrics.safety_qualified_speed == 0.0

    complete = PhysicalMetricsRecorder(
        2,
        12,
        str(tmp_path),
        expected_total_frames=10,
        success_completion_threshold=0.8,
    )
    _record_frames(complete, 8, speeds=[8.0] * 8)

    complete_metrics = complete.calculate_episode_metrics()

    assert complete_metrics.route_completion == pytest.approx(0.8)
    assert complete_metrics.success_completion is True
    assert complete_metrics.safety_qualified_speed == pytest.approx(8.0)


def test_collision_disqualifies_success_even_after_completion_threshold(tmp_path):
    recorder = PhysicalMetricsRecorder(
        1,
        11,
        str(tmp_path),
        expected_total_frames=10,
        success_completion_threshold=0.8,
    )
    _record_frames(recorder, 10, crash_last=True)

    metrics = recorder.calculate_episode_metrics()

    assert metrics.route_completion == 1.0
    assert metrics.collision is True
    assert metrics.success_completion is False
    assert metrics.safety_qualified_speed == 0.0


def test_episode_total_reward_is_distinct_from_per_frame_reward(tmp_path):
    recorder = PhysicalMetricsRecorder(
        1, 11, str(tmp_path), expected_total_frames=3
    )
    _record_frames(
        recorder,
        3,
        rewards=[1.0, -0.5, 2.5],
        speeds=[4.0, 5.0, 6.0],
    )

    metrics = recorder.calculate_episode_metrics()

    assert metrics.episode_total_reward == pytest.approx(3.0)
    assert metrics.mean_reward_per_frame == pytest.approx(1.0)
    assert metrics.avg_reward == pytest.approx(1.0)


def test_missing_completion_denominator_is_explicitly_unavailable(tmp_path):
    recorder = PhysicalMetricsRecorder(1, 11, str(tmp_path))
    _record_frames(recorder, 3)

    metrics = recorder.calculate_episode_metrics()

    assert metrics.route_completion is None
    assert metrics.success_completion is False


def test_aggregator_uses_continuous_completion_and_real_endpoint_metrics(tmp_path):
    aggregator = MetricsAggregator("endpoint", str(tmp_path))
    aggregator.physical_metrics_list = [
        {
            "total_frames": 10,
            "expected_total_frames": 10,
            "route_completion": 1.0,
            "collision": False,
            "success_completion": True,
            "avg_speed": 10.0,
            "episode_total_reward": 20.0,
            "mean_reward_per_frame": 2.0,
            "driving_distance": 100.0,
        },
        {
            "total_frames": 5,
            "expected_total_frames": 10,
            "route_completion": 0.5,
            "collision": False,
            "success_completion": False,
            "avg_speed": 30.0,
            "episode_total_reward": 5.0,
            "mean_reward_per_frame": 1.0,
            "driving_distance": 50.0,
        },
    ]

    metrics = aggregator.calculate_comprehensive_metrics()

    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["avg_route_completion"] == pytest.approx(0.75)
    assert metrics["avg_route_completion"] != metrics["success_rate"]
    assert metrics["route_completion_available"] is True
    assert metrics["avg_episode_reward"] == pytest.approx(12.5)
    assert metrics["avg_reward_per_frame"] == pytest.approx(25.0 / 15.0)
    assert metrics["avg_speed_safety_qualified"] == pytest.approx(5.0)
    assert metrics["avg_speed_all_frames"] == pytest.approx(250.0 / 15.0)
    assert metrics["avg_speed_over_success"] == pytest.approx(10.0)


def test_legacy_rows_recover_total_reward_but_do_not_fake_completion(tmp_path):
    aggregator = MetricsAggregator("legacy", str(tmp_path))
    aggregator.physical_metrics_list = [
        {
            "total_frames": 4,
            "collision": False,
            "success_completion": True,
            "avg_speed": 6.0,
            "avg_reward": 1.5,
        }
    ]

    metrics = aggregator.calculate_comprehensive_metrics()

    assert metrics["success_rate"] == 1.0
    assert metrics["avg_route_completion"] is None
    assert metrics["route_completion_available"] is False
    assert metrics["avg_episode_reward"] == pytest.approx(6.0)
    assert metrics["avg_reward_per_frame"] == pytest.approx(1.5)


def test_runtime_recorder_derives_formal_policy_frame_denominator(tmp_path):
    physical, reasoning = create_episode_recorders(
        0,
        3,
        str(tmp_path),
        {
            "env_type": "highway-v0",
            "simulation_duration": 30,
            "policy_frequency": 10,
            "enable_physical_metrics": True,
            "enable_reasoning_recording": False,
            "success_completion_threshold": 0.999,
        },
    )

    assert reasoning is None
    assert physical.expected_total_frames == 300
    assert physical.success_completion_threshold == pytest.approx(0.999)


def test_disabled_physical_recorder_does_not_require_a_horizon(tmp_path):
    physical, reasoning = create_episode_recorders(
        0,
        3,
        str(tmp_path),
        {
            "env_type": "highway-v0",
            "enable_physical_metrics": False,
            "enable_reasoning_recording": False,
        },
    )

    assert physical is None
    assert reasoning is None


def test_aggregator_recomputes_completion_from_expected_frames(tmp_path):
    aggregator = MetricsAggregator("denominator", str(tmp_path))
    aggregator.physical_metrics_list = [
        {
            "total_frames": 4,
            "expected_total_frames": 10,
            "route_completion": 1.0,
            "success_completion": True,
            "success_completion_threshold": 0.8,
            "success_metric_mode": "completion_threshold",
            "avg_speed": 12.0,
        }
    ]

    metrics = aggregator.calculate_comprehensive_metrics()

    assert metrics["avg_route_completion"] == pytest.approx(0.4)
    assert metrics["success_rate"] == 0.0
    assert metrics["avg_speed_safety_qualified"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expected_total_frames": 0}, "positive integer"),
        ({"expected_total_frames": 10, "success_completion_threshold": 1.1}, "between 0 and 1"),
        ({"expected_total_frames": 10, "success_metric_mode": "unknown"}, "success_metric_mode"),
    ],
)
def test_invalid_endpoint_contract_fails_closed(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        PhysicalMetricsRecorder(1, 2, str(tmp_path), **kwargs)
