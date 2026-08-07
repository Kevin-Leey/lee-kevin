"""Physical telemetry recording for one simulator episode."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


@dataclass
class PhysicalMetrics:
    episode_id: int
    seed: int
    total_frames: int
    collision: bool
    success_completion: bool
    avg_speed: float
    avg_reward: float
    driving_distance: float
    expected_total_frames: Optional[int] = None
    route_completion: Optional[float] = None
    success_completion_threshold: float = 0.95
    success_metric_mode: str = "completion_threshold"
    safety_qualified_speed: float = 0.0
    episode_total_reward: float = 0.0
    mean_reward_per_frame: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhysicalMetricsRecorder:
    """Record a compact, backend-neutral physical trace."""

    def __init__(
        self,
        episode_id: int,
        seed: int,
        result_folder: str,
        *,
        step_seconds: float = 1.0,
        expected_total_frames: Optional[int] = None,
        success_completion_threshold: float = 0.95,
        success_metric_mode: str = "completion_threshold",
        env_type: str = "",
        published_reward_vmax_mps: float = 0.0,
    ) -> None:
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.result_folder = str(result_folder)
        self.step_seconds = max(0.0, float(step_seconds))
        if expected_total_frames is None:
            self.expected_total_frames = None
        else:
            if isinstance(expected_total_frames, bool):
                raise ValueError("expected_total_frames must be a positive integer")
            try:
                numeric_expected_frames = float(expected_total_frames)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "expected_total_frames must be a positive integer"
                ) from exc
            if (
                not math.isfinite(numeric_expected_frames)
                or not numeric_expected_frames.is_integer()
                or numeric_expected_frames <= 0.0
            ):
                raise ValueError("expected_total_frames must be a positive integer")
            self.expected_total_frames = int(numeric_expected_frames)
        self.success_completion_threshold = float(success_completion_threshold)
        if not math.isfinite(self.success_completion_threshold) or not (
            0.0 <= self.success_completion_threshold <= 1.0
        ):
            raise ValueError("success_completion_threshold must be between 0 and 1")
        self.success_metric_mode = str(success_metric_mode or "").strip().lower()
        if self.success_metric_mode not in {"completion_threshold", "collision_free"}:
            raise ValueError(
                "success_metric_mode must be 'completion_threshold' or 'collision_free'"
            )
        self.env_type = str(env_type)
        self.published_reward_vmax_mps = float(published_reward_vmax_mps)
        self.frames: List[Dict[str, Any]] = []

    def record_frame(
        self,
        frame_id: int,
        state: Mapping[str, Any],
        *,
        action: int,
        reward: float = 0.0,
        crashed: bool = False,
        done: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = dict(state or {})
        position = payload.get("pos", payload.get("position", (float("nan"), float("nan"))))
        try:
            position_x, position_y = _finite_or_none(position[0]), _finite_or_none(position[1])
        except (IndexError, TypeError, ValueError):
            position_x = position_y = None
        row = {
            "frame_id": int(frame_id),
            "speed": float(payload.get("speed", payload.get("ego_speed", 0.0)) or 0.0),
            "ttc": _finite_or_none(payload.get("ttc", float("inf"))),
            "thw": _finite_or_none(payload.get("thw", float("inf"))),
            "position_x": position_x,
            "position_y": position_y,
            "action_id": int(action),
            "reward": float(reward),
            "crashed": bool(crashed),
            "done": bool(done),
            "info": _json_safe(dict(info or {})),
        }
        self.frames.append(row)
        return row

    def calculate_episode_metrics(self) -> PhysicalMetrics:
        speeds = [float(row["speed"]) for row in self.frames if math.isfinite(float(row["speed"]))]
        rewards = [float(row["reward"]) for row in self.frames]
        collision = any(bool(row["crashed"]) for row in self.frames)
        total_frames = len(self.frames)
        route_completion = (
            min(1.0, float(total_frames) / float(self.expected_total_frames))
            if self.expected_total_frames is not None
            else None
        )
        if self.success_metric_mode == "completion_threshold":
            success_completion = bool(
                not collision
                and route_completion is not None
                and route_completion >= self.success_completion_threshold
            )
        else:
            # Retain the explicitly requested legacy collision-free mode while
            # making the formal completion-threshold mode denominator-based.
            success_completion = bool(not collision and total_frames > 0)
        avg_speed = float(sum(speeds) / len(speeds)) if speeds else 0.0
        episode_total_reward = float(sum(rewards)) if rewards else 0.0
        mean_reward_per_frame = (
            float(episode_total_reward / len(rewards)) if rewards else 0.0
        )
        if len(self.frames) >= 2:
            start = self.frames[0]
            end = self.frames[-1]
            try:
                distance = math.hypot(
                    float(end["position_x"]) - float(start["position_x"]),
                    float(end["position_y"]) - float(start["position_y"]),
                )
            except (TypeError, ValueError):
                distance = float("nan")
            if not math.isfinite(distance):
                distance = sum(max(0.0, speed) * self.step_seconds for speed in speeds)
        else:
            distance = sum(max(0.0, speed) * self.step_seconds for speed in speeds)
        return PhysicalMetrics(
            episode_id=self.episode_id,
            seed=self.seed,
            total_frames=total_frames,
            expected_total_frames=self.expected_total_frames,
            route_completion=route_completion,
            collision=bool(collision),
            success_completion=success_completion,
            success_completion_threshold=self.success_completion_threshold,
            success_metric_mode=self.success_metric_mode,
            avg_speed=avg_speed,
            safety_qualified_speed=avg_speed if success_completion else 0.0,
            episode_total_reward=episode_total_reward,
            mean_reward_per_frame=mean_reward_per_frame,
            # Backward-compatible alias: historically avg_reward meant the
            # per-frame mean, despite the aggregate calling it episode reward.
            avg_reward=mean_reward_per_frame,
            driving_distance=float(distance),
        )

    def save(self) -> Dict[str, Any]:
        root = Path(self.result_folder)
        root.mkdir(parents=True, exist_ok=True)
        metrics = self.calculate_episode_metrics().to_dict()
        payload = {"episode_id": self.episode_id, "frames": self.frames, "metrics": metrics}
        path = root / f"physical_frames_{self.episode_id}.json"
        path.write_text(
            json.dumps(_json_safe(payload), ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return payload


__all__ = ["PhysicalMetrics", "PhysicalMetricsRecorder"]
