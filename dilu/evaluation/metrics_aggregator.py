"""Aggregate episode traces into public RGD result metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_RISK_TTC_SECONDS = 2.0


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _binary_label(value: Any) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and math.isfinite(value) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return value.strip() == "1"
    raise ValueError("corrective_set_nonempty must be a binary label")


class MetricsAggregator:
    """Collect per-episode recorder payloads and request-scoped event traces."""

    def __init__(self, experiment_name: str, result_dir: str) -> None:
        self.experiment_name = str(experiment_name)
        self.result_dir = str(result_dir)
        self.physical_metrics_list: List[Dict[str, Any]] = []
        self.all_reasoning_records: List[Dict[str, Any]] = []
        self.all_event_records: List[Dict[str, Any]] = []
        self.evaluation_metrics_list: List[Dict[str, Any]] = []

    def add_episode(self, *, physical_payload: Any = None, reasoning_payload: Any = None) -> None:
        if isinstance(physical_payload, Mapping):
            metrics = physical_payload.get("metrics", physical_payload)
            if isinstance(metrics, Mapping):
                self.physical_metrics_list.append(dict(metrics))
        if isinstance(reasoning_payload, Mapping):
            records = reasoning_payload.get("analysis_records", reasoning_payload.get("records", []))
            if isinstance(records, list):
                self.all_reasoning_records.extend(dict(item) for item in records if isinstance(item, Mapping))

    def _event_records_from_disk(self) -> List[Dict[str, Any]]:
        root = Path(self.result_dir) / "event_logs"
        rows: List[Dict[str, Any]] = []
        if not root.is_dir():
            return rows
        for path in sorted(root.glob("event_log_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(dict(item) for item in payload.get("events", []) if isinstance(item, Mapping))
        return rows

    def _events(self) -> List[Dict[str, Any]]:
        return list(self.all_event_records) or self._event_records_from_disk()

    @staticmethod
    def _is_queried(record: Mapping[str, Any]) -> bool:
        return bool(record.get("slow_request_attempted", False))

    @staticmethod
    def _is_risk(record: Mapping[str, Any]) -> bool:
        ttc = _finite(record.get("ttc"))
        return ttc is not None and ttc <= _RISK_TTC_SECONDS

    @staticmethod
    def _release_distinct(record: Mapping[str, Any]) -> bool:
        if not bool(record.get("closed_loop_latency_release_event", False)):
            return False
        selected = record.get("release_selected_action")
        comparator = record.get("release_fast_comparator_action")
        if selected is not None and comparator is not None:
            return int(selected) != int(comparator)
        fast = record.get("closed_loop_execution_state_fast_action")
        executed = record.get("closed_loop_latency_executed_action", record.get("final_action"))
        if fast is not None and executed is not None:
            return int(fast) != int(executed)
        return bool(record.get("closed_loop_release_actuation_distinct", False))

    def _selective_allocation_story(self, reasoning_records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        records = [dict(record) for record in list(reasoning_records or [])]
        events = self._events() or records
        risk_events = [record for record in events if self._is_risk(record)]
        low_events = [record for record in events if not self._is_risk(record)]
        queried_events = [record for record in events if self._is_queried(record)]
        queried_risk = [record for record in risk_events if self._is_queried(record)]
        risk_rate = (len(queried_risk) / len(risk_events)) if risk_events else None
        low_rate = (sum(self._is_queried(record) for record in low_events) / len(low_events)) if low_events else None
        precision = (len(queried_risk) / len(queried_events)) if queried_events else None
        released = [record for record in events if bool(record.get("closed_loop_latency_release_event", False))]
        distinct = [record for record in released if self._release_distinct(record)]
        compute_records = [record for record in records if self._is_queried(record)]
        latency_values = [_finite(record.get("inference_latency")) for record in compute_records]
        latency_values = [value for value in latency_values if value is not None]

        labels = [_binary_label(row.get("corrective_set_nonempty")) for row in self.evaluation_metrics_list]
        rv_positive = sum(labels)
        return {
            "selective_allocation_metrics_available": bool(events),
            "risk_frame_count": len(risk_events),
            "queried_frame_count": len(queried_events),
            "queried_risk_frame_count": len(queried_risk),
            "risk_conditional_query_recall": risk_rate,
            "queried_frame_risk_precision": precision,
            "high_vs_low_query_rate_difference": None if risk_rate is None or low_rate is None else risk_rate - low_rate,
            "released_response_count": len(released),
            "distinct_corrective_actuation_count": len(distinct),
            "actuation_yield": (len(distinct) / len(released)) if released else None,
            "effect_distinctness_available": bool(released) and all("closed_loop_release_effect_distinct" in item for item in released),
            "slow_compute_call_count": len(compute_records),
            "slow_compute_seconds_available": bool(latency_values),
            "slow_compute_seconds": float(sum(latency_values)) if latency_values else None,
            "compute_per_corrective_release": (len(compute_records) / len(distinct)) if distinct else None,
            "compute_seconds_per_corrective_release": (sum(latency_values) / len(distinct)) if latency_values and distinct else None,
            "rvod_evaluated_release_count": len(labels),
            "rvod_positive_release_count": int(rv_positive),
            "rvod_positive_yield": (rv_positive / len(labels)) if labels else None,
        }

    def _reasoning_story(self) -> Dict[str, Any]:
        events = self._events()
        issued = [record for record in events if bool(record.get("closed_loop_latency_issuance_event", False))]
        terminal = [record for record in events if bool(record.get("closed_loop_latency_terminal_event", False))]
        outcomes = [str(record.get("closed_loop_latency_terminal_response_outcome", "") or "") for record in terminal]
        slow_attempts = len(issued)
        successes = sum(outcome == "valid" for outcome in outcomes)
        failures = sum(outcome in {"timeout", "failure"} for outcome in outcomes)
        pending = max(0, slow_attempts - len(terminal))
        high_risk_scores = []
        for record in self.all_reasoning_records:
            if self._is_queried(record) and self._is_risk(record):
                ttc = _finite(record.get("ttc"))
                if ttc is not None:
                    high_risk_scores.append(max(0.0, _RISK_TTC_SECONDS - 0.5 * ttc))
        utility = float(sum(high_risk_scores)) if high_risk_scores else 0.0
        return {
            "slow_request_lifecycle_request_scoped": True,
            "slow_attempts": slow_attempts,
            "slow_attempt_successes": int(successes),
            "slow_attempt_failures": int(failures),
            "slow_attempt_terminal_outcomes": len(terminal),
            "slow_attempt_pending": int(pending),
            "budget_normalized_independent_high_risk_utility": utility,
            "independent_selective_routing_gain": utility,
        }

    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        physical = self.physical_metrics_list
        frame_count = sum(int(row.get("total_frames", 0) or 0) for row in physical)
        collision_rate = (sum(bool(row.get("collision", False)) for row in physical) / len(physical)) if physical else 0.0
        success_rate = (sum(bool(row.get("success_completion", False)) for row in physical) / len(physical)) if physical else 0.0
        avg = lambda key: (sum(float(row.get(key, 0.0) or 0.0) for row in physical) / len(physical)) if physical else 0.0
        payload = {
            "experiment_name": self.experiment_name,
            "total_episodes": len(physical),
            "total_frames": frame_count,
            "collision_rate": collision_rate,
            "success_rate": success_rate,
            "avg_speed_safety_qualified": avg("avg_speed"),
            "avg_speed_all_frames": avg("avg_speed"),
            "avg_episode_reward": avg("avg_reward"),
            "avg_driving_distance": avg("driving_distance"),
            "avg_route_completion": success_rate,
            "avg_runtime_per_frame": 0.0,
        }
        payload.update(self._reasoning_story())
        payload.update(self._selective_allocation_story(self.all_reasoning_records))
        return payload


__all__ = ["MetricsAggregator"]
