from dilu.evaluation.factorial_replay import COMPONENT_ABLATION_ARMS
from tools.analyze_factorial_interventions import summarize_events
from tools.analyze_v13_component_interventions import _episode_effects


def _event(seed, *, executed, classification, rejected=0, utility=0.2):
    return {
        "arm": "full",
        "seed": seed,
        "candidate_evaluable": 1,
        "first_step_actuator_distinct": 1,
        "executed_first_step_actuator_distinct": executed,
        "release_guard_rejected": rejected,
        "classification": classification,
        "utility_delta": utility,
        "normalized_return_delta": utility,
        "collision_delta": -1 if classification == "beneficial" else 1,
        "progress_delta_m": utility * 10.0,
        "min_ttc_delta_s": utility,
        "mean_abs_jerk_delta_mps3": -utility,
    }


def test_release_summary_reports_coverage_errors_and_endpoint_effects():
    summary = summarize_events(
        [
            _event(1, executed=1, classification="harmful", utility=-0.2),
            _event(2, executed=0, classification="beneficial", rejected=1),
        ],
        seeds=(1, 2),
        draws=40,
        bootstrap_seed=9,
        arms=("full",),
    )
    values = {row["metric"]: row for row in summary}

    assert values["executed_distinct_coverage_of_evaluable_releases"]["estimate"] == 0.5
    assert values["harmful_fraction_of_executed_first_step_interventions"]["estimate"] == 1.0
    assert values["beneficial_fraction_of_release_guard_rejections"]["estimate"] == 1.0
    assert values["missed_beneficial_fraction_of_evaluable_distinct_candidates"]["estimate"] == 0.5
    assert values["collision_delta_per_executed_first_step_intervention"]["estimate"] == 1.0
    assert values["progress_delta_m_per_executed_first_step_intervention"]["estimate"] == -2.0


def _episode_rows():
    rows = []
    for seed, offset in ((10, 0.0), (11, 2.0)):
        values = {
            "full": 10.0 + offset,
            "without_l": 9.0 + offset,
            "without_a": 8.0 + offset,
            "without_h": 7.0 + offset,
            "without_n": 8.0 + offset,
            "without_h_and_n": 3.0 + offset,
        }
        for arm, value in values.items():
            rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "collision": 0.0,
                    "route_completion": 1.0,
                    "episode_reward": value,
                    "driving_distance": value * 10.0,
                    "avg_speed": value,
                }
            )
    return rows


def test_episode_effects_include_the_prespecified_h_by_n_interaction():
    arms = tuple(arm.name for arm in COMPONENT_ABLATION_ARMS)
    effects = _episode_effects(
        _episode_rows(),
        seeds=(10, 11),
        arms=arms,
        draws=40,
        bootstrap_seed=4,
    )
    values = {(row["effect"], row["metric"]): row for row in effects}

    assert values[("full_minus_without_n", "episode_reward")]["estimate"] == 2.0
    assert values[("h_x_n_interaction", "episode_reward")]["estimate"] == -2.0
    assert values[("h_x_n_interaction", "driving_distance")]["estimate"] == -20.0
