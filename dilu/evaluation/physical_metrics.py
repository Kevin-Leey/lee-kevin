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
        success_completion_threshold: float = 0.95,
        success_metric_mode: str = "completion_threshold",
        env_type: str = "",
        published_reward_vmax_mps: float = 0.0,
    ) -> None:
        self.episode_id = int(episode_id)
        self.seed = int(seed)
        self.result_folder = str(result_folder)
        self.step_seconds = max(0.0, float(step_seconds))
        self.success_completion_threshold = float(success_completion_threshold)
        self.success_metric_mode = str(success_metric_mode)
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
            total_frames=len(self.frames),
            collision=bool(collision),
            success_completion=bool(not collision and bool(self.frames)),
            avg_speed=float(sum(speeds) / len(speeds)) if speeds else 0.0,
            avg_reward=float(sum(rewards) / len(rewards)) if rewards else 0.0,
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
