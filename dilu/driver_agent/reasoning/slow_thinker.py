"""Structured slow-path executor for the RGD request stage."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import math
import threading
import time
from dataclasses import dataclass, field
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


def _strict_action_id(value: Any, failure_reason: str) -> int:
    if isinstance(value, bool):
        raise SlowPathUnavailableError(failure_reason)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise SlowPathUnavailableError(failure_reason) from exc
    if not math.isfinite(numeric) or int(numeric) != numeric:
        raise SlowPathUnavailableError(failure_reason)
    return int(numeric)


@dataclass
class SlowRequest:
    request_id: str
    episode_token: str
    source_frame: int
    release_frame: int
    submitted_at_monotonic: float
    deadline_at_monotonic: float
    future: concurrent.futures.Future
    completed_at_monotonic: Optional[float] = None
    completion_clock: Dict[str, float] = field(default_factory=dict, repr=False)


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
        configured_workers = int(self.config.get("max_workers", 1) or 1)
        if configured_workers != 1:
            raise ValueError(
                "SlowThinker requires max_workers=1 under the single-pending-request contract"
            )
        self.max_workers = 1
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="dilu-slow"
        )
        self._active_futures: set[concurrent.futures.Future] = set()
        self._active_lock = threading.Lock()
        self._shutdown = False

    def get_available_action_universe(self, state: DrivingState) -> tuple[int, ...]:
        return tuple(int(action) for action in state.get_available_actions())

    def get_runtime_budget_state(self) -> Dict[str, Any]:
        provider_available = self.model is not None or self.action_provider is not None
        with self._active_lock:
            active = len(self._active_futures)
            shutdown = bool(self._shutdown)
        capacity = max(0, int(self.max_workers) - int(active))
        return {
            "llm_available": bool(
                provider_available and not shutdown and capacity > 0
            ),
            "provider_available": bool(provider_available),
            "executor_capacity_available": bool(capacity > 0 and not shutdown),
            "executor_active_requests": int(active),
            "executor_capacity_remaining": int(capacity),
            "executor_max_workers": int(self.max_workers),
            "llm_invoke_timeout_s": float(self.request_timeout_s),
            "executor": "online_llm" if self.model is not None else "offline_provider",
        }

    @staticmethod
    def _prompt_value(value: Any) -> Any:
        if isinstance(value, float):
            return round(value, 3) if math.isfinite(value) else "unobserved"
        if isinstance(value, dict):
            return {
                str(key): SlowThinker._prompt_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [SlowThinker._prompt_value(item) for item in value]
        if value is None or isinstance(value, (str, int, bool)):
            return value
        return str(value)

    @classmethod
    def _observation_text(
        cls, state: DrivingState, recoverability_context: Dict[str, Any]
    ) -> str:
        assessment = dict(
            recoverability_context.get("recoverability_assessment", {}) or {}
        )
        profile = dict(
            recoverability_context.get("route_ambiguity_profile", {}) or {}
        )
        history = [
            {
                "speed": frame.get("speed", frame.get("ego_speed")),
                "front_distance": frame.get(
                    "front_dist", frame.get("front_distance")
                ),
                "ttc": frame.get("ttc"),
                "executed_action": frame.get("executed_action"),
            }
            for frame in list(state.history_frames or [])[-3:]
        ]
        payload = {
            "scenario": state.scenario_type,
            "ego": {
                "speed_mps": state.ego_speed,
                "lane": state.ego_lane,
                "lane_count": state.total_lanes,
                "heading_rad": state.heading,
            },
            "lead": {
                "distance_m": state.front_distance,
                "speed_mps": state.front_speed,
                "closing_speed_mps": state.front_relative_speed,
                "ttc_s": state.ttc,
                "thw_s": state.thw,
            },
            "left_lane": {
                "legal_and_safe": bool(state.can_change_left),
                "front_distance_m": state.left_front_distance,
                "rear_distance_m": state.left_rear_distance,
                "front_speed_mps": state.left_front_speed,
                "rear_speed_mps": state.left_rear_speed,
            },
            "right_lane": {
                "legal_and_safe": bool(state.can_change_right),
                "front_distance_m": state.right_front_distance,
                "rear_distance_m": state.right_rear_distance,
                "front_speed_mps": state.right_front_speed,
                "rear_speed_mps": state.right_rear_speed,
            },
            "conflict": {
                "nearest_distance_m": state.closest_vehicle_distance,
                "longitudinal_m": state.closest_vehicle_longitudinal,
                "lateral_m": state.closest_vehicle_lateral,
                "closing_speed_mps": state.closest_vehicle_closing_speed,
                "cross_traffic_distance_m": state.cross_traffic_distance,
                "junction_gap": state.junction_gap,
            },
            "available_actions": {
                str(action): ActionType.to_english(action)
                for action in state.get_available_actions()
            },
            "fast_incumbent_action": recoverability_context.get(
                "provisional_action"
            ),
            "recoverability": {
                "feasible_alternatives": assessment.get(
                    "raw_feasible_alternative_actions", []
                ),
                "recovery_window": assessment.get("latency_survival", 0.0),
                "maneuver_breadth": assessment.get(
                    "relative_support_weighted_maneuver_family_breadth", 0.0
                ),
                "corrective_headroom": assessment.get(
                    "corrective_recovery_headroom", 0.0
                ),
                "action_costs": profile.get("action_recovery_costs", {}),
            },
            "recent_history": history,
        }
        return json.dumps(
            cls._prompt_value(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
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
                action = _strict_action_id(
                    raw, "offline_proposal_action_identity_invalid"
                )
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
            observation=self._observation_text(state, recoverability_context),
            available_actions=available,
        )
        try:
            response = self.model.complete(
                "You are a delayed driving decision module. Prioritize collision "
                "avoidance, legal actions, and recoverable progress. Return compact "
                "JSON only; no prose.",
                prompt,
            )
        except Exception as exc:
            raise SlowPathUnavailableError(
                f"slow_executor_failure:{type(exc).__name__}:{str(exc)[:160]}"
            ) from exc
        try:
            parsed = LLMUtils.parse_structured_decision(response, available)
        except Exception as exc:
            raise SlowPathUnavailableError(
                f"slow_response_parse_failure:{type(exc).__name__}:{str(exc)[:160]}"
            ) from exc

        action = _strict_action_id(
            parsed["action"], "slow_response_action_identity_invalid"
        )
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
                "slow_request_attempted": True,
                "slow_request_valid_return": True,
                "slow_request_failed": False,
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
        episode_token: str,
        source_frame: int,
        release_frame: int,
        state: DrivingState,
        recoverability_context: Optional[Dict[str, Any]] = None,
    ) -> SlowRequest:
        resolved_request_id = str(request_id or "").strip()
        resolved_episode_token = str(episode_token or "").strip()
        if not resolved_request_id:
            raise SlowPathUnavailableError("slow_request_id_missing")
        if not resolved_episode_token:
            raise SlowPathUnavailableError("slow_request_episode_token_missing")
        source = int(source_frame)
        release = int(release_frame)
        if source < 0 or release < source:
            raise SlowPathUnavailableError("slow_request_frame_contract_invalid")
        if self.model is None and self.action_provider is None:
            raise SlowPathUnavailableError("slow_executor_unavailable")
        try:
            frozen_state = copy.deepcopy(state)
            frozen_context = copy.deepcopy(dict(recoverability_context or {}))
        except Exception as exc:
            raise SlowPathUnavailableError(
                f"slow_request_snapshot_failure:{type(exc).__name__}"
            ) from exc
        submitted_at = time.perf_counter()
        completion_clock: Dict[str, float] = {}

        def _execute_frozen() -> RGDDecision:
            try:
                return self._execute(frozen_state, frozen_context)
            finally:
                completion_clock["completed_at_monotonic"] = time.perf_counter()

        # Admission and registration are one transaction.  Without this lock,
        # concurrent callers can both observe the single worker as available
        # and queue two requests under the one-pending-request contract.
        with self._active_lock:
            if self._shutdown:
                raise SlowPathUnavailableError("slow_executor_shutdown")
            if len(self._active_futures) >= self.max_workers:
                raise SlowPathUnavailableError("slow_executor_capacity_exhausted")
            try:
                future = self._executor.submit(_execute_frozen)
            except RuntimeError as exc:
                raise SlowPathUnavailableError("slow_executor_shutdown") from exc
            self._active_futures.add(future)
        request = SlowRequest(
            request_id=resolved_request_id,
            episode_token=resolved_episode_token,
            source_frame=source,
            release_frame=release,
            submitted_at_monotonic=float(submitted_at),
            deadline_at_monotonic=float(submitted_at + self.request_timeout_s),
            future=future,
            completion_clock=completion_clock,
        )

        def _mark_complete(completed: concurrent.futures.Future) -> None:
            request.completed_at_monotonic = float(
                completion_clock.get("completed_at_monotonic", time.perf_counter())
            )
            with self._active_lock:
                self._active_futures.discard(completed)

        future.add_done_callback(_mark_complete)
        return request

    @staticmethod
    def is_ready(request: SlowRequest) -> bool:
        """Return whether a request has reached a terminal executor state."""
        return bool(
            request.future.done()
            or time.perf_counter() >= float(request.deadline_at_monotonic)
        )

    @staticmethod
    def poll(request: SlowRequest) -> Optional[RGDDecision]:
        """Return a completed decision without blocking, or ``None`` if pending."""
        now = time.perf_counter()
        if not request.future.done() and now < float(request.deadline_at_monotonic):
            return None
        completed_at = request.completed_at_monotonic
        if completed_at is None and request.future.done():
            completed_at = request.completion_clock.get(
                "completed_at_monotonic", now
            )
            request.completed_at_monotonic = float(completed_at)
        if completed_at is None or float(completed_at) > float(request.deadline_at_monotonic):
            request.future.cancel()
            raise SlowPathUnavailableError("slow_request_timeout")
        try:
            return request.future.result(timeout=0.0)
        except SlowPathUnavailableError:
            raise
        except concurrent.futures.CancelledError as exc:
            raise SlowPathUnavailableError("slow_request_cancelled") from exc
        except Exception as exc:
            raise SlowPathUnavailableError(
                f"slow_executor_failure:{type(exc).__name__}:{str(exc)[:160]}"
            ) from exc

    @staticmethod
    def cancel(request: SlowRequest) -> bool:
        return bool(request.future.cancel())

    def shutdown(self, *, wait: bool = False) -> None:
        with self._active_lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=True)
