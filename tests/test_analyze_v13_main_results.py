import subprocess
import sys
from pathlib import Path

import pytest

from tools.analyze_v13_main_results import (
    ENDPOINTS,
    FORMAL_GROUPS,
    _lifecycle,
    analyze,
    canonical_distinct_actuation,
)


def _event(**updates):
    event = {
        "closed_loop_latency_release_event": True,
        "closed_loop_release_action_alignment_evaluated": True,
        "closed_loop_release_action_alignment_pass": True,
        "closed_loop_release_opportunity_rejected": False,
        "closed_loop_release_action_unavailable": False,
        "closed_loop_execution_state_fast_action": 1,
        "closed_loop_latency_executed_action": 4,
    }
    event.update(updates)
    return event


def test_canonical_distinct_actuation_uses_final_executed_action():
    assert canonical_distinct_actuation(_event()) is True
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_latency_executed_action=1)
        )
        is False
    )


def test_canonical_distinct_actuation_requires_evaluated_passing_alignment():
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_action_alignment_evaluated=False)
        )
        is False
    )
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_action_alignment_pass=False)
        )
        is False
    )


def test_canonical_distinct_actuation_rejects_unavailable_or_rejected_release():
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_action_unavailable=True)
        )
        is False
    )
    assert (
        canonical_distinct_actuation(
            _event(closed_loop_release_opportunity_rejected=True)
        )
        is False
    )


def _validated_matrix():
    seeds = (5000, 5001, 5002, 5003)
    matrix = {}
    for group_index, group in enumerate(FORMAL_GROUPS):
        for seed_index, seed in enumerate(seeds):
            values = {
                "success_rate": float(group == "rgd_fixed_policy" or seed_index > 0),
                "collision_rate": float(group != "rgd_fixed_policy" and seed_index == 0),
                "route_completion": 1.0 if group == "rgd_fixed_policy" else 0.9,
                "episode_reward": 100.0 - group_index,
                "driving_distance_m": 500.0 - group_index,
                "safety_qualified_speed_mps": 18.0 - group_index * 0.1,
                "all_frame_speed_mps": 18.0 - group_index * 0.1,
                "algorithm_runtime_ms_per_frame": 20.0 + group_index,
                "request_count": 1 + group_index,
                "valid_response_count": group_index,
                "release_count": group_index,
                "authorized_release_count": 0,
                "distinct_actuation_count": 0,
            }
            assert set(values) == set(ENDPOINTS)
            matrix[(seed, group)] = {
                "group": group,
                "seed": seed,
                **values,
                "terminal_count": group_index,
                "timeout_count": 0,
                "failure_count": 0,
                "dropped_at_episode_end_count": 0,
                "terminal_latencies_s": [1.0] if group_index else [],
            }
    return {"seeds": seeds, "matrix": matrix}


def test_main_analysis_is_seed_paired_and_deterministic():
    first = analyze(_validated_matrix(), draws=500, bootstrap_seed=11)
    second = analyze(_validated_matrix(), draws=500, bootstrap_seed=11)
    assert first == second
    success = next(
        row
        for row in first["paired_contrasts"]
        if row["baseline_group"] == "always_fast"
        and row["endpoint"] == "success_rate"
    )
    assert success["estimate_rgd_minus_baseline"] == pytest.approx(0.25)
    assert success["rgd_wins"] == 1
    assert success["baseline_wins"] == 0
    assert 0.0 <= success["p_value_holm_within_endpoint"] <= 1.0


def test_request_lifecycle_requires_terminal_or_explicit_episode_drop():
    issued = {
        "closed_loop_latency_issuance_event": True,
        "closed_loop_latency_issued_request_id": "r1",
    }
    with pytest.raises(ValueError, match="unterminated"):
        _lifecycle([issued], [])
    accepted = _lifecycle([issued], [{"request_id": "r1"}])
    assert accepted["request_count"] == 1
    assert accepted["dropped_at_episode_end_count"] == 1


def test_analyzer_supports_direct_cli_help():
    completed = subprocess.run(
        [sys.executable, "tools/analyze_v13_main_results.py", "--help"],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--bundle" in completed.stdout
