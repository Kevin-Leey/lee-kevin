"""Public result-surface builders with explicit null semantics."""

from __future__ import annotations

from typing import Any, Dict, Mapping


_PUBLIC_KEYS = (
    "collision_rate",
    "success_rate",
    "avg_route_completion",
    "avg_episode_reward",
    "avg_driving_distance",
    "avg_speed_safety_qualified",
    "avg_speed_all_frames",
    "avg_runtime_per_frame",
    "budget_normalized_independent_high_risk_utility",
    "independent_selective_routing_gain",
    "risk_conditional_query_recall",
    "queried_frame_risk_precision",
    "high_vs_low_query_rate_difference",
    "rvod_positive_yield",
    "actuation_yield",
    "compute_per_corrective_release",
    "compute_seconds_per_corrective_release",
)


def build_public_comprehensive_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Expose the stable public metric surface without fabricating undefined rates."""
    source = dict(metrics or {})
    return {key: source.get(key) for key in _PUBLIC_KEYS}


__all__ = ["build_public_comprehensive_metrics"]
