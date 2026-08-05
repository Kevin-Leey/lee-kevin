from collections import deque

from dilu.runtime_frame_state import build_frame_driving_state, record_executed_history_frame


def _state():
    return {
        "speed": 24.0,
        "front_speed": 22.0,
        "front_dist": 18.0,
        "lane": 1,
        "total_lanes": 4,
        "nearby_vehicle_count": 3,
        "pos": (10.0, 4.0),
        "ttc": 9.0,
        "thw": 0.75,
    }


def test_current_frame_is_not_exposed_as_prior_history():
    prior = {"speed": 23.0, "ttc": 8.0, "thw": 0.70, "action": 1}
    history = deque([prior], maxlen=6)
    driving_state = build_frame_driving_state(
        _state(),
        "highway",
        {"env_type": "highway-v0", "scenario_type": "highway"},
        history,
        prev_action=1,
        env_available_actions=[0, 1, 2, 3, 4],
    )

    assert list(history) == [prior]
    assert driving_state.history_frames == [prior]


def test_history_records_the_final_executed_action():
    history = deque(maxlen=6)
    record_executed_history_frame(history, _state(), action=4)
    assert list(history) == [{"speed": 24.0, "ttc": 9.0, "thw": 0.75, "action": 4}]
