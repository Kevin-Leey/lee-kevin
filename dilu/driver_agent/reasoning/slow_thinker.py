"""Structured slow-path executor for the RGD request stage."""

from __future__ import annotations

import concurrent.futures
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

from dilu.driver_agent.base.prompts import PromptManager
from dilu.driver_agent.base.state import ActionType, DrivingState
from dilu.driver_agent.reasoning.decision import RGDDecision
from dilu.driver_agent.reasoning.llm_utils import LLMUtils


class SlowPathUnavailableError(RuntimeError):
    """Raised when the configured slow executor cannot issue a request."""

    def __init__(self, failure_reason: str) -> None:
        self.failure_reason = str(failure_reason)
        super().__init__(self.failure_reason)


@dataclass(frozen=True)
class SlowRequest:
    request_id: str
    source_frame: int
    release_frame: int
    future: concurrent.futures.Future


class SlowThinker:
    """Issue one structured LLM request without inventing a substitute action.

    ``model`` must expose ``complete(system_prompt, user_prompt)``.  Offline
    studies may instead inject an ``action_provider`` callable, keeping the
    proposal source explicit in the request metadata.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        model: Optional[Any] = None,
        action_provider: Optional[Callable[[DrivingState, Dict[str, Any]], Any]] = None,
    ) -> None:
        self.config = dict(config or {})
        self.model = model
        self.action_provider = action_provider
        self.request_timeout_s = max(
            0.1, float(self.config.get("request_timeout_s", 30.0) or 30.0)
        )
        self.max_workers = max(1, int(self.config.get("max_workers", 1) or 1))
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="dilu-slow"
        )

    def get_available_action_universe(self, state: DrivingState) -> tuple[int, ...]:
        return tuple(int(action) for action in state.get_available_actions())

    def get_runtime_budget_state(self) -> Dict[str, Any]:
        available = self.model is not None or self.action_provider is not None
        return {
            "llm_available": bool(available),
            "llm_invoke_timeout_s": float(self.request_timeout_s),
            "executor": "online_llm" if self.model is not None else "offline_provider",
        }

    @staticmethod
    def _observation_text(state: DrivingState) -> str:
        return (
            f"scenario={state.scenario_type}; speed={state.ego_speed:.2f}; "
            f"front_distance={state.front_distance:.2f}; ttc={state.ttc:.2f}; "
            f"available_actions={state.get_available_actions()}"
        )

    def _execute(self, state: DrivingState, recoverability_context: Dict[str, Any]) -> RGDDecision:
        available = list(self.get_available_action_universe(state))
        if not available:
            raise SlowPathUnavailableError("empty_action_universe")
        if self.action_provider is not None:
            try:
                raw = self.action_provider(state, dict(recoverability_context))
            except SlowPathUnavailableError:
                raise
            except Exception as exc:
                raise SlowPathUnavailableError(
                    f"offline_provider_failure:{type(exc).__name__}"
                ) from exc
            if isinstance(raw, RGDDecision):
                decision = raw
            else:
                action = int(raw)
                decision = RGDDecision(
                    action=action,
                    reasoning="offline slow proposal",
                    confidence=1.0,
                    system_used="slow",
                    route_label="offline_proposal",
                    route_score=0.0,
                )
            if int(decision.action) not in available:
                raise SlowPathUnavailableError("offline_proposal_outside_action_universe")
            decision.stats.setdefault("slow_reasoning_mode", "offline_provider")
            decision.stats.setdefault("slow_reasoning_success", True)
            return decision
        if self.model is None:
            raise SlowPathUnavailableError("slow_executor_unavailable")

        prompt = PromptManager.render(
            "slow_decision",
            observation=self._observation_text(state),
            available_actions=available,
        )
        try:
            response = self.model.complete(
                "Return a concise structured driving decision.", prompt
            )
            parsed = LLMUtils.parse_structured_decision(response, available)
        except Exception as exc:
            raise SlowPathUnavailableError(f"slow_executor_failure:{type(exc).__name__}") from exc

        action = int(parsed["action"])
        if action not in available:
            raise SlowPathUnavailableError("slow_response_outside_action_universe")
        confidence = parsed.get("confidence")
        return RGDDecision(
            action=action,
            reasoning=" | ".join(parsed.get("reason_lines", [])) or "structured slow proposal",
            confidence=float(0.5 if confidence is None else confidence),
            system_used="slow",
            route_label="online_llm",
            route_score=0.0,
            thinking_steps=list(parsed.get("reason_lines", [])),
            stats={
                "slow_reasoning_mode": "online_llm",
                "slow_reasoning_success": True,
                "slow_structured_response": True,
                "slow_response_text": str(response),
            },
        )

    def think(
        self, *, state: DrivingState, recoverability_context: Optional[Dict[str, Any]] = None
    ) -> RGDDecision:
        return self._execute(state, dict(recoverability_context or {}))

    def submit(
        self,
        *,
        request_id: str,
        source_frame: int,
        release_frame: int,
        state: DrivingState,
        recoverability_context: Optional[Dict[str, Any]] = None,
    ) -> SlowRequest:
        if not self.get_runtime_budget_state()["llm_available"]:
            raise SlowPathUnavailableError("slow_executor_unavailable")
        future = self._executor.submit(
            self._execute, state, dict(recoverability_context or {})
        )
        return SlowRequest(
            request_id=str(request_id),
            source_frame=int(source_frame),
            release_frame=int(release_frame),
            future=future,
        )

    def shutdown(self, *, wait: bool = False) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)
