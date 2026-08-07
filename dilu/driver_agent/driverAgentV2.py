"""Primary RGD driver-agent entry point.

This wrapper owns the policy-state boundary between simulator frames.  It does
not apply safety or mutate the environment; the runtime loop performs those
operations and then reports the final executed command through
``record_executed_action``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from dilu.driver_agent.policy_state import (
    DRIVER_POLICY_STATE_SCHEMA,
    validate_driver_policy_state,
    validate_fast_policy_state,
)
from dilu.driver_agent.reasoning.fast_thinker import FastThinker
from dilu.driver_agent.reasoning.rgd_core import RGDOrchestrator
from dilu.driver_agent.reasoning.slow_thinker import SlowThinker
from dilu.latency_contract import bind_latency_contract


class DriverAgentV2:
    """Compose the fast incumbent, slow executor, and RGD orchestrator."""

    def __init__(
        self,
        sce: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        *,
        slow_model: Optional[Any] = None,
        slow_action_provider: Optional[Any] = None,
    ) -> None:
        self.sce = sce
        self.config = self._prepare_runtime_config(config or {})
        fast_cfg = dict(self.config.get("fast_thinking", {}) or {})
        if "lane_change_cooldown" in self.config:
            fast_cfg.setdefault(
                "lane_change_cooldown", self.config["lane_change_cooldown"]
            )
        self.fast_thinker = FastThinker(lane_change_config=fast_cfg)

        slow_cfg = dict(self.config.get("slow_thinking", {}) or {})
        slow_cfg.setdefault(
            "request_timeout_s", self.config.get("QWEN_REQUEST_TIMEOUT_S", 30.0)
        )
        self.slow_thinker = SlowThinker(
            slow_cfg,
            model=slow_model,
            action_provider=slow_action_provider,
        )
        self.orchestrator = RGDOrchestrator(
            self.fast_thinker, self.slow_thinker, self.config
        )
        self._episode_ended = False
        self._closed = False

    def _prepare_runtime_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Bind latency units after the simulator instance has been selected."""
        resolved = copy.deepcopy(dict(config or {}))
        slow_cfg = dict(resolved.get("slow_thinking", {}) or {})
        if "risk_coupling" not in resolved and slow_cfg.get("risk_coupling"):
            resolved["risk_coupling"] = copy.deepcopy(slow_cfg["risk_coupling"])
        env = getattr(getattr(self, "sce", None), "env", None)
        resolved = bind_latency_contract(resolved, env)
        resolved.setdefault(
            "policy_frequency", resolved["_resolved_policy_frequency_hz"]
        )
        return resolved

    def decide(self, state: Any) -> Tuple[int, str, Dict[str, Any]]:
        if self._episode_ended:
            raise RuntimeError("cannot decide after end_episode()")
        decision = self.orchestrator.decide(state)
        metadata = dict(decision.stats or {})
        metadata.setdefault("fast_rule_name", metadata.get("rule_name", "none"))
        metadata.update(
            {
                "system_used": str(decision.system_used),
                "route_label": str(decision.route_label),
                "route_score": float(decision.route_score),
                "confidence": float(decision.confidence),
                "latency_ms": float(decision.latency_ms),
                "proposed_action": int(decision.action),
                "final_action": int(decision.action),
            }
        )
        return int(decision.action), str(decision.reasoning), metadata

    def record_executed_action(self, action: int) -> None:
        self.fast_thinker.record_executed_action(int(action))

    def uses_native_async_policy_pacing(self) -> bool:
        """Return whether this agent executes the live asynchronous RGD path."""
        return bool(
            self.orchestrator._async_slow_path_enabled
            and self.orchestrator._forced_route_system != "fast"
        )

    def _snapshot_policy_state_unchecked(self) -> Dict[str, Any]:
        return {
            "schema": DRIVER_POLICY_STATE_SCHEMA,
            "fast": self.fast_thinker.snapshot_policy_state(),
            "orchestrator": self.orchestrator.snapshot_policy_state(),
        }

    def snapshot_policy_state(self) -> Dict[str, Any]:
        if self.pending_slow_requests():
            raise RuntimeError(
                "cannot create a standalone policy snapshot while a slow request is live"
            )
        return self._snapshot_policy_state_unchecked()

    def snapshot_release_policy_state(self) -> Dict[str, Any]:
        """Capture replayable policy state beside a request-bound descriptor."""
        return self._snapshot_policy_state_unchecked()

    def restore_policy_state(self, snapshot: Dict[str, Any]) -> None:
        if self.pending_slow_requests():
            raise RuntimeError("cannot restore policy state while a slow request is live")
        normalized = validate_driver_policy_state(snapshot)
        validate_fast_policy_state(
            normalized["fast"],
            expected_capacity=self.fast_thinker.action_history_capacity,
        )
        self.orchestrator.restore_policy_state(normalized["orchestrator"])
        self.fast_thinker.restore_policy_state(normalized["fast"])

    def prepare_frame(self, frame: int) -> Optional[Dict[str, Any]]:
        if self._episode_ended:
            return None
        return self.orchestrator.prepare_frame(int(frame))

    def pending_slow_requests(self) -> list[Dict[str, Any]]:
        return self.orchestrator.pending_slow_requests()

    def end_episode(self, reason: str = "episode_end") -> list[Dict[str, Any]]:
        if self._episode_ended:
            return []
        self._episode_ended = True
        return self.orchestrator.end_episode(str(reason))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.end_episode("agent_closed")
        self.slow_thinker.shutdown(wait=False)
