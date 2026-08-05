from dilu.evaluation.reporter import _build_runtime_experiment_config
from dilu.driver_agent.reasoning.rgd_support import build_rgd_execution_contract


def test_runtime_manifest_retains_effective_metadrive_configuration():
    core_story = {"rgd_route_score_requires_soft_recoverability": True}
    cfg = {
        "protocol_name": "metadrive-provenance-test",
        "env_type": "metadrive-intersection-v0",
        "scenario_type": "intersection",
        "episodes_num": 1,
        "simulation_duration": 480,
        "metadrive_eval": {
            "metadrive_horizon": 480,
            "metadrive_start_seed_base": 0,
            "metadrive_random_seed": 0,
            "metadrive_traffic_density": 0.4,
            "metadrive_spawn_speed": 6.5,
        },
        "slow_thinking": {"executor": "online_llm", "risk_coupling": {"core_story": core_story}},
        "_rgd_runtime_contract": build_rgd_execution_contract(core_story).to_dict(),
    }
    public = _build_runtime_experiment_config(cfg)
    assert public["metadrive_eval"] == cfg["metadrive_eval"]
