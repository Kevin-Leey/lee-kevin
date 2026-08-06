"""Decision records shared by the fast, slow, and RGD route stages."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


def _finite_or_zero(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    return resolved if math.isfinite(resolved) else 0.0


def _integer_key_map(values: Mapping[Any, Any]) -> Dict[int, Any]:
    return {int(key): value for key, value in dict(values or {}).items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


@dataclass
class RouteAmbiguityProfile:
    """Action-cost landscape retained as route diagnostics.

    The profile does not decide the route.  It records the same action domain
    used by the recoverability gate so traces can distinguish physical costs
    from the relative support costs used for diagnostic probability ranking.
    """

    action_probabilities: Dict[int, float]
    action_recovery_costs: Dict[int, float]
    action_support_ranking_costs: Dict[int, float] = field(default_factory=dict)
    probability_cost_source: str = "action_recovery_costs"
    method_version: str = "identifiable_gate_v12"
    gate_action_universe: List[int] = field(default_factory=list)
    fast_executor_action_universe: List[int] = field(default_factory=list)
    action_recovery_cost_parts: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    raw_cost_source: str = "unknown"
    raw_cost_complete: bool = False
    missing_raw_cost_actions: List[int] = field(default_factory=list)
    nonfinite_raw_cost_actions: List[int] = field(default_factory=list)
    ambiguity_best_action: Optional[int] = None
    selected_probability: float = 0.0
    ambiguity_entropy: float = 0.0
    ambiguity_gap: float = 0.0
    evidence_disagreement: float = 0.0
    intervention_risk: float = 0.0
    source_priors: Dict[str, float] = field(default_factory=dict)
    hypothesis_beliefs: Dict[str, float] = field(default_factory=dict)
    executed_action: Optional[int] = None

    def __post_init__(self) -> None:
        self.action_probabilities = {
            action: _finite_or_zero(value)
            for action, value in _integer_key_map(self.action_probabilities).items()
        }
        self.action_recovery_costs = {
            action: float(value)
            for action, value in _integer_key_map(self.action_recovery_costs).items()
        }
        self.action_support_ranking_costs = {
            action: float(value)
            for action, value in _integer_key_map(
                self.action_support_ranking_costs
            ).items()
        }
        self.gate_action_universe = [int(action) for action in self.gate_action_universe]
        self.fast_executor_action_universe = [
            int(action) for action in self.fast_executor_action_universe
        ]
        self.action_recovery_cost_parts = {
            int(action): dict(parts or {})
            for action, parts in dict(self.action_recovery_cost_parts or {}).items()
        }
        self.missing_raw_cost_actions = [
            int(action) for action in self.missing_raw_cost_actions
        ]
        self.nonfinite_raw_cost_actions = [
            int(action) for action in self.nonfinite_raw_cost_actions
        ]
        if self.ambiguity_best_action is not None:
            self.ambiguity_best_action = int(self.ambiguity_best_action)
        if self.executed_action is not None:
            self.executed_action = int(self.executed_action)
        for field_name in (
            "selected_probability",
            "ambiguity_entropy",
            "ambiguity_gap",
            "evidence_disagreement",
            "intervention_risk",
        ):
            setattr(self, field_name, _finite_or_zero(getattr(self, field_name)))
        self.source_priors = {
            str(key): _finite_or_zero(value)
            for key, value in dict(self.source_priors or {}).items()
        }
        self.hypothesis_beliefs = {
            str(key): _finite_or_zero(value)
            for key, value in dict(self.hypothesis_beliefs or {}).items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_probabilities": _json_safe(self.action_probabilities),
            "action_recovery_costs": _json_safe(self.action_recovery_costs),
            "action_support_ranking_costs": _json_safe(
                self.action_support_ranking_costs
            ),
            "probability_cost_source": str(self.probability_cost_source),
            "method_version": str(self.method_version),
            "gate_action_universe": list(self.gate_action_universe),
            "fast_executor_action_universe": list(self.fast_executor_action_universe),
            "action_recovery_cost_parts": _json_safe(
                self.action_recovery_cost_parts
            ),
            "raw_cost_source": str(self.raw_cost_source),
            "raw_cost_complete": bool(self.raw_cost_complete),
            "missing_raw_cost_actions": list(self.missing_raw_cost_actions),
            "nonfinite_raw_cost_actions": list(self.nonfinite_raw_cost_actions),
            "ambiguity_best_action": self.ambiguity_best_action,
            "selected_probability": self.selected_probability,
            "ambiguity_entropy": self.ambiguity_entropy,
            "ambiguity_gap": self.ambiguity_gap,
            "evidence_disagreement": self.evidence_disagreement,
            "intervention_risk": self.intervention_risk,
            "source_priors": _json_safe(self.source_priors),
            "hypothesis_beliefs": _json_safe(self.hypothesis_beliefs),
            "executed_action": self.executed_action,
        }


@dataclass
class RGDDecision:
    """A resolved action and the minimal provenance needed by downstream code."""

    action: int
    reasoning: str
    confidence: float
    system_used: str
    route_label: str
    route_score: float
    ambiguity_profile: Optional[RouteAmbiguityProfile] = None
    thinking_steps: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    agent_opinions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = int(self.action)
        self.reasoning = str(self.reasoning)
        self.confidence = _finite_or_zero(self.confidence)
        self.system_used = str(self.system_used)
        self.route_label = str(self.route_label)
        self.route_score = _finite_or_zero(self.route_score)
        self.thinking_steps = [str(step) for step in list(self.thinking_steps or [])]
        self.stats = dict(self.stats or {})
        self.agent_opinions = dict(self.agent_opinions or {})
        self.latency_ms = max(0.0, _finite_or_zero(self.latency_ms))

    @property
    def rule_name(self) -> Optional[str]:
        value = self.stats.get("rule_name")
        return None if value is None else str(value)

    @property
    def smoothness_override(self) -> bool:
        return bool(self.stats.get("smoothness_override", False))

    @property
    def decision_mode(self) -> str:
        return str(self.stats.get("decision_mode", "hard_rule_shell"))

    @property
    def abstention_applied(self) -> bool:
        return bool(self.stats.get("abstention_applied", False))

    @property
    def top_score_gap(self) -> float:
        return _finite_or_zero(self.stats.get("top_score_gap", 0.0))

    @property
    def calibration_context(self) -> Dict[str, Any]:
        return dict(self.stats.get("calibration_context", {}) or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": int(self.action),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "system_used": self.system_used,
            "route_label": self.route_label,
            "route_score": self.route_score,
            "ambiguity_profile": (
                self.ambiguity_profile.to_dict()
                if self.ambiguity_profile is not None
                else None
            ),
            "thinking_steps": list(self.thinking_steps),
            "stats": _json_safe(self.stats),
            "latency_ms": self.latency_ms,
            "agent_opinions": _json_safe(self.agent_opinions),
        }
