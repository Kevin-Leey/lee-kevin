import math

import subprocess
import sys

import numpy as np

from tools.analyze_five_arm_factorial import (
    ARM_NAMES,
    METRICS,
    _mcnemar_exact,
    _wilson,
    analyze,
)


def test_five_arm_analyzer_supports_direct_cli_help():
    completed = subprocess.run(
        [sys.executable, "tools/analyze_five_arm_factorial.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--bundle" in completed.stdout


def _validated():
    seeds = (10, 11, 12, 13)
    matrix = {}
    reward_offsets = {
        "fast_only": 0.0,
        "neither": -1.0,
        "query_only": 1.0,
        "release_only": 2.0,
        "full": 4.0,
    }
    for seed in seeds:
        for arm in ARM_NAMES:
            row = {metric: 0.0 for metric in METRICS}
            row.update(
                {
                    "seed": seed,
                    "arm": arm,
                    "collision": float(arm == "neither" and seed == 10),
                    "success_rate": float(not (arm == "neither" and seed == 10)),
                    "route_completion": 1.0,
                    "episode_reward": seed + reward_offsets[arm],
                    "driving_distance": 100.0 + reward_offsets[arm],
                    "avg_speed": 20.0,
                    "runtime_per_frame": 0.01,
                    "candidate_queries": 2.0,
                    "issued_queries": 0.0 if arm == "fast_only" else 1.0,
                    "query_gate_rejections": 2.0 if arm == "fast_only" else 1.0,
                    "release_events": 0.0 if arm == "fast_only" else 1.0,
                    "primitive_distinct_selections": 0.0,
                }
            )
            matrix[(seed, arm)] = row
    return {"seeds": seeds, "matrix": matrix}


def test_five_arm_analysis_is_seed_paired_and_deterministic():
    first = analyze(_validated(), draws=500, bootstrap_seed=17)
    second = analyze(_validated(), draws=500, bootstrap_seed=17)

    assert first == second
    reward = next(
        row
        for row in first["paired_contrasts"]
        if row["contrast"] == "full_minus_fast_only"
        and row["metric"] == "episode_reward"
    )
    assert reward["estimate"] == 4.0
    assert reward["left_wins"] == 4
    assert reward["ties"] == reward["right_wins"] == 0
    assert 0.0 <= reward["p_value_raw"] <= 1.0
    assert 0.0 <= reward["p_value_holm_primary_family"] <= 1.0


def test_binary_inference_helpers_are_bounded_and_exact():
    low, high = _wilson(3, 4)
    assert 0.0 <= low < 0.75 < high <= 1.0
    assert math.isclose(
        _mcnemar_exact(
            left=np.asarray([1.0, 1.0, 1.0, 0.0]),
            right=np.asarray([0.0, 0.0, 0.0, 0.0]),
        ),
        0.25,
    )
