"""Read-only state-memory retrieval for optional fast-path support."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SupportMemoryExample:
    """One retrieved memory item with its action-domain validity."""

    memory_id: int
    action: int
    similarity: float
    action_available: bool
    payload: Dict[str, Any]


def resolve_memory_path(value: str) -> Optional[Path]:
    """Resolve a configured repository-relative memory location.

    The function returns a path even when it does not exist so the retriever can
    retain the configured location while explicitly exposing an empty memory
    result.  An absent configuration remains ``None`` rather than the current
    working directory.
    """

    raw = str(value or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    root = Path(__file__).resolve().parents[3]
    return root / candidate


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return float(default)
    return resolved if math.isfinite(resolved) else float(default)


def _unit_vector(values: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _state_features(state: Any) -> List[float]:
    def bounded(value: Any, scale: float) -> float:
        parsed = _finite(value, scale)
        return min(1.0, max(0.0, parsed / scale))

    return _unit_vector(
        [
            bounded(getattr(state, "ego_speed", 0.0), 35.0),
            bounded(getattr(state, "front_distance", 100.0), 100.0),
            bounded(getattr(state, "ttc", 12.0), 12.0),
            bounded(getattr(state, "thw", 4.0), 4.0),
            bounded(getattr(state, "ego_lane", 0.0), 6.0),
        ]
    )


def _entry_features(payload: Mapping[str, Any]) -> Optional[List[float]]:
    embedding = payload.get("embedding")
    if isinstance(embedding, Sequence) and not isinstance(embedding, (str, bytes)):
        values = [_finite(value) for value in embedding]
        if values:
            return _unit_vector(values)
    state = payload.get("state")
    if not isinstance(state, Mapping):
        state = payload.get("observation")
    if not isinstance(state, Mapping):
        state = payload
    feature_keys = ("ego_speed", "front_distance", "ttc", "thw", "ego_lane")
    aliases = {
        "ego_speed": "speed",
        "front_distance": "front_dist",
    }
    if not any(key in state or aliases.get(key) in state for key in feature_keys):
        return None
    values = [
        _finite(state.get(key, state.get(aliases.get(key, ""), 0.0)))
        for key in feature_keys
    ]
    scales = (35.0, 100.0, 12.0, 4.0, 6.0)
    bounded = [min(1.0, max(0.0, value / scale)) for value, scale in zip(values, scales)]
    return _unit_vector(bounded)


class StateMemoryRetriever:
    """Load explicit memory records and rank them by state similarity."""

    def __init__(self, path: Optional[Path], top_k: int = 1) -> None:
        self.path = None if path is None else Path(path)
        self.top_k = max(1, int(top_k))
        self._records = self._load_records()

    @staticmethod
    def _normalise_record(raw: Any, fallback_id: int) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, Mapping):
            return None
        payload = dict(raw)
        try:
            action = int(payload.get("action"))
        except (TypeError, ValueError):
            return None
        try:
            memory_id = int(payload.get("memory_id", payload.get("id", fallback_id)))
        except (TypeError, ValueError):
            return None
        payload["memory_id"] = memory_id
        payload["action"] = action
        if _entry_features(payload) is None:
            return None
        return payload

    def _load_json_records(self, path: Path) -> List[Dict[str, Any]]:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            records = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError:
                    return []
                records.append(item)
            parsed = records
        if isinstance(parsed, Mapping):
            parsed = parsed.get("records", parsed.get("memories", []))
        if not isinstance(parsed, list):
            return []
        normalized = []
        for index, item in enumerate(parsed):
            record = self._normalise_record(item, index)
            if record is not None:
                normalized.append(record)
        return normalized

    def _load_sqlite_records(self, path: Path) -> List[Dict[str, Any]]:
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        except sqlite3.Error:
            return []
        try:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            for (name,) in tables:
                if str(name).lower() not in {"memories", "state_memory", "records"}:
                    continue
                columns = [row[1] for row in connection.execute(f"PRAGMA table_info({name})")]
                if "action" not in columns:
                    continue
                rows = connection.execute(f"SELECT * FROM {name}").fetchall()
                records = []
                for index, row in enumerate(rows):
                    payload = dict(zip(columns, row))
                    for key in ("state", "observation", "embedding", "payload"):
                        value = payload.get(key)
                        if isinstance(value, str):
                            try:
                                payload[key] = json.loads(value)
                            except json.JSONDecodeError:
                                pass
                    if isinstance(payload.get("payload"), Mapping):
                        merged = dict(payload["payload"])
                        merged.update({key: value for key, value in payload.items() if key != "payload"})
                        payload = merged
                    record = self._normalise_record(payload, index)
                    if record is not None:
                        records.append(record)
                return records
        except sqlite3.Error:
            return []
        finally:
            connection.close()
        return []

    def _load_records(self) -> List[Dict[str, Any]]:
        if self.path is None or not self.path.is_file():
            return []
        suffix = self.path.suffix.lower()
        if suffix in {".json", ".jsonl"}:
            return self._load_json_records(self.path)
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            return self._load_sqlite_records(self.path)
        return []

    def retrieve(self, state: Any, available: Iterable[Any]) -> List[SupportMemoryExample]:
        """Return at most ``top_k`` actual records, ranked deterministically."""

        if not self._records:
            return []
        query = _state_features(state)
        available_actions = {int(action) for action in available}
        ranked = []
        for record in self._records:
            features = _entry_features(record)
            if features is None or len(features) != len(query):
                continue
            similarity = max(0.0, min(1.0, sum(a * b for a, b in zip(query, features))))
            ranked.append(
                SupportMemoryExample(
                    memory_id=int(record["memory_id"]),
                    action=int(record["action"]),
                    similarity=float(similarity),
                    action_available=int(record["action"]) in available_actions,
                    payload=dict(record),
                )
            )
        ranked.sort(key=lambda item: (-item.similarity, item.memory_id))
        return ranked[: self.top_k]
