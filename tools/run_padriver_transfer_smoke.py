"""Configure and audit the fixed PaDriver lane-density transfer probe."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from dilu.evaluation.reporter import save_experiment_snapshot


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed zero-added-latency lane-density transfer probe."
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--simulation-duration", type=int, default=30)
    parser.add_argument("--policy-frequency", type=int, default=10)
    parser.add_argument("--simulation-frequency", type=int, default=10)
    parser.add_argument("--preserve-executor-action", action="store_true")
    parser.add_argument("--release-dominance-guard", action="store_true")
    parser.add_argument("--release-dominance-margin", type=float, default=0.0)
    parser.add_argument("--min-observation-frames", type=int, default=None)
    return parser.parse_args(argv)


def _episode_contracts(seed: int, episodes: int) -> list[Dict[str, int]]:
    return [
        {
            "episode_index": index,
            "episode_id": int(seed) * int(episodes) + index,
            "env_seed": int(seed),
        }
        for index in range(int(episodes))
    ]


def _apply_transfer_overrides(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    *,
    lane_count: int,
    density: float,
    seed: int,
) -> Dict[str, Any]:
    """Bind the transfer setting before any environment is constructed."""
    if int(args.episodes) <= 0:
        raise ValueError("episodes must be positive")
    policy_frequency = int(args.policy_frequency)
    simulation_frequency = int(args.simulation_frequency)
    if policy_frequency <= 0 or simulation_frequency <= 0:
        raise ValueError("simulator frequencies must be positive")
    min_observation_frames = (
        int(args.min_observation_frames)
        if args.min_observation_frames is not None
        else int(cfg.get("rgd_min_observation_frames", 2) or 2)
    )
    if min_observation_frames < 1:
        raise ValueError("min observation frames must be positive")

    risk_calibration = {"enable": False}
    replay = {
        "enable": False,
        "extra_latency_s": 0.0,
        "delay_steps": 0,
        "target_systems": ["slow"],
    }
    guard = {
        "enable": bool(args.release_dominance_guard),
        "risk_margin": float(args.release_dominance_margin),
        "require_strict_improvement": True,
        "scope": "query_equals_release_zero_delay",
    }
    contracts = _episode_contracts(int(seed), int(args.episodes))
    effective = {
        "env_type": "highway-v0",
        "lanes_count": int(lane_count),
        "vehicles_density": float(density),
        "vehicle_count": 30,
        "simulation_duration": int(args.simulation_duration),
        "policy_frequency_hz": policy_frequency,
        "simulation_frequency_hz": simulation_frequency,
        "fixed_seed_override": int(seed),
        "resolved_env_seeds": [int(seed)] * int(args.episodes),
        "episode_contracts": contracts,
        "additional_latency_s": 0.0,
        "zero_additional_latency": True,
        "rgd_predicted_slow_latency_s": 0.0,
        "rgd_predicted_slow_latency_source": "lane_transfer_configured_extra_latency",
        "closed_loop_latency_replay": replay,
        "release_dominance_guard": guard,
        "target_lane_projection_enable": False,
        "preserve_executor_action": bool(args.preserve_executor_action),
        "risk_calibration": risk_calibration,
        "rgd_min_observation_frames": min_observation_frames,
    }

    cfg.update(
        {
            "env_type": effective["env_type"],
            "scenario_type": "highway",
            "lanes_count": effective["lanes_count"],
            "vehicles_density": effective["vehicles_density"],
            "vehicles_count": effective["vehicle_count"],
            "simulation_duration": effective["simulation_duration"],
            "policy_frequency": policy_frequency,
            "simulation_frequency": simulation_frequency,
            "fixed_seed_override": int(seed),
            "closed_loop_latency_replay": copy.deepcopy(replay),
            "rgd_predicted_slow_latency_s": 0.0,
            "rgd_predicted_slow_latency_source": effective[
                "rgd_predicted_slow_latency_source"
            ],
            "release_dominance_guard": copy.deepcopy(guard),
            "target_lane_projection_enable": False,
            "preserve_executor_action": effective["preserve_executor_action"],
            "risk_calibration": copy.deepcopy(risk_calibration),
            "rgd_min_observation_frames": min_observation_frames,
            "transfer_effective_config": copy.deepcopy(effective),
        }
    )
    slow = dict(cfg.get("slow_thinking", {}) or {})
    slow["risk_calibration"] = copy.deepcopy(risk_calibration)
    cfg["slow_thinking"] = slow
    protocol = dict(cfg.get("_paper_protocol_config", {}) or {})
    if protocol:
        runtime = copy.deepcopy(cfg)
        runtime.pop("_paper_protocol_config", None)
        protocol["runtime_config"] = runtime
        cfg["_paper_protocol_config"] = protocol
    return effective


def _save_transfer_experiment_snapshot(
    cfg: Mapping[str, Any],
    root: Path,
    seed: int,
    effective: Mapping[str, Any],
) -> None:
    """Write matching public manifests with the executed transfer contract."""
    payload_cfg = copy.deepcopy(dict(cfg))
    payload_cfg["transfer_effective_config"] = copy.deepcopy(dict(effective))
    save_experiment_snapshot(payload_cfg, str(root), int(seed))
    for name in ("runtime_manifest.json", "experiment_snapshot.json"):
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["transfer_effective_config"] = copy.deepcopy(dict(effective))
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _validate_transfer_snapshot_pair(
    root: Path,
    effective: Mapping[str, Any],
) -> None:
    """Fail closed when the two immutable manifests disagree with the setting."""
    manifests = []
    for name in ("runtime_manifest.json", "experiment_snapshot.json"):
        path = root / name
        if not path.is_file():
            raise ValueError(f"missing transfer snapshot: {name}")
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    left, right = manifests
    for field in ("protocol_hash", "config_hash"):
        if left.get(field) != right.get(field):
            raise ValueError(f"transfer snapshot {field} mismatch")
    expected = dict(effective)
    for payload in manifests:
        contract = dict(payload.get("transfer_effective_config", {}) or {})
        if contract != expected:
            for field, value in expected.items():
                if contract.get(field) != value:
                    raise ValueError(f"transfer effective contract drift: {field}")
            raise ValueError("transfer effective contract drift")
        config = dict(payload.get("config", {}) or {})
        for field in (
            "lanes_count",
            "vehicles_density",
            "simulation_duration",
            "fixed_seed_override",
            "preserve_executor_action",
            "rgd_min_observation_frames",
        ):
            if config.get(field) != expected[field]:
                raise ValueError(f"transfer config drift: {field}")
        slow = dict(config.get("slow_thinking", {}) or {})
        if dict(slow.get("risk_calibration", {}) or {}) != expected["risk_calibration"]:
            raise ValueError("transfer config drift: risk_calibration")
    if left.get("config") != right.get("config"):
        raise ValueError("transfer snapshot config mismatch")


def _effective_config_matches_request(
    effective: Mapping[str, Any],
    args: argparse.Namespace,
    lane_count: int,
    density: float,
    seed: int,
    min_observation_frames: int,
) -> bool:
    expected_guard = {
        "enable": bool(args.release_dominance_guard),
        "risk_margin": float(args.release_dominance_margin),
        "require_strict_improvement": True,
        "scope": "query_equals_release_zero_delay",
    }
    return bool(
        str(effective.get("env_type")) == "highway-v0"
        and int(effective.get("lanes_count", -1)) == int(lane_count)
        and float(effective.get("vehicles_density", -1.0)) == float(density)
        and int(effective.get("fixed_seed_override", -1)) == int(seed)
        and bool(effective.get("zero_additional_latency", False))
        and dict(effective.get("closed_loop_latency_replay", {}) or {}).get("enable") is False
        and dict(effective.get("release_dominance_guard", {}) or {}) == expected_guard
        and bool(effective.get("preserve_executor_action", False))
        == bool(args.preserve_executor_action)
        and int(effective.get("rgd_min_observation_frames", -1))
        == int(min_observation_frames)
    )


def _transfer_trace_files_complete(root: Path, *, episodes: int) -> bool:
    """Require one readable physical/reasoning/event triplet per episode."""
    for index in range(int(episodes)):
        paths = (
            root / f"ep_{index}" / f"highway_{index}_physical_frames.json",
            root / f"ep_{index}" / f"highway_{index}_reasoning_records.json",
            root / "event_logs" / f"event_log_highway_{index}_{index}.json",
        )
        try:
            payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if not all(isinstance(payload, Mapping) for payload in payloads):
            return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    raise RuntimeError(
        "the transfer runner requires the main-table execution harness; use its "
        "setting-specific entry point after the runtime configuration is bound"
    )


if __name__ == "__main__":
    raise SystemExit(main())
