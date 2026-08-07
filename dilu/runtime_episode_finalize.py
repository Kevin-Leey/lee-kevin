"""Persist episode traces after validating delayed-response provenance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from dilu.evaluation.release_snapshot import save_release_snapshot_bundle
from dilu.evaluation.reporter import save_episode_results
from dilu.runtime_frame_trace import EVENT_SCHEMA_VERSION


_SUPPORTED_EVENT_SCHEMAS = {"rgd_event_log_v2", "rgd_event_log_v3"}


def _request_id(event: Mapping[str, Any]) -> str:
    return str(event.get("closed_loop_latency_terminal_request_id", "") or "").strip()


def _validate_release_snapshots(
    events: Iterable[Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    *,
    required: bool,
) -> list[str]:
    release_ids: list[str] = []
    for event in events:
        if not bool(event.get("closed_loop_latency_release_event", False)):
            continue
        request_id = _request_id(event)
        if not request_id:
            raise RuntimeError("release event is missing a non-empty terminal request ID")
        if request_id in release_ids:
            raise RuntimeError("duplicate release terminal request ID")
        release_ids.append(request_id)

    if required and set(release_ids) != {str(key) for key in snapshots}:
        raise RuntimeError("release snapshot coverage mismatch")
    return release_ids


def _dropped_pending_rows(pending: Optional[Iterable[Mapping[str, Any]]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for item in list(pending or []):
        request_id = str(dict(item).get("request_id", "") or "")
        source = dict(item)
        row: Dict[str, Any] = {
            "request_id": request_id,
            "source_frame": source.get("source_frame"),
            "terminal_outcome": "dropped_at_episode_end",
        }
        for key in (
            "episode_token",
            "release_frame",
            "response_outcome",
            "drop_reason",
            "future_cancelled",
            "native_async",
        ):
            if key in source:
                row[key] = source[key]
        rows.append(row)
    return rows


def finalize_episode_outputs(
    *,
    ep: int,
    cfg: Mapping[str, Any],
    agent: Any,
    metrics_agg: Any,
    docs: Iterable[Any],
    collision_frame: int,
    result_dir: str,
    prefix: str,
    frame_runtimes: Iterable[float],
    phys_rec: Any,
    reas_rec: Any,
    event_log: Iterable[Mapping[str, Any]],
    pending_latency_queue: Optional[Iterable[Mapping[str, Any]]] = None,
    release_snapshots: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Write one self-contained episode bundle and delegate aggregate storage."""
    del docs
    runtime_cfg = dict(cfg or {})
    schema_version = str(runtime_cfg.get("event_log_schema_version", EVENT_SCHEMA_VERSION) or "")
    if schema_version not in _SUPPORTED_EVENT_SCHEMAS:
        raise RuntimeError(f"unsupported event-log schema: {schema_version!r}")

    events = [dict(event) for event in list(event_log or [])]
    snapshots = dict(release_snapshots or {})
    _validate_release_snapshots(
        events,
        snapshots,
        required=bool(runtime_cfg.get("require_release_snapshot_on_release", False)),
    )
    pending = [dict(item) for item in list(pending_latency_queue or [])]
    end_episode = getattr(agent, "end_episode", None)
    if callable(end_episode):
        pending.extend(dict(item) for item in list(end_episode("episode_finalize") or []))
    dropped = _dropped_pending_rows(pending)

    root = Path(result_dir)
    event_dir = root / "event_logs"
    event_dir.mkdir(parents=True, exist_ok=True)
    release_bundle = release_manifest = None
    release_digest = None
    if snapshots:
        bundle_path, manifest_path, release_digest = save_release_snapshot_bundle(
            snapshots,
            root / "release_snapshots",
            prefix=str(prefix),
            episode_id=int(ep),
        )
        release_bundle = str(bundle_path.relative_to(root)).replace("\\", "/")
        release_manifest = str(manifest_path.relative_to(root)).replace("\\", "/")

    event_payload = {
        "schema_version": schema_version,
        "episode_id": int(ep),
        "prefix": str(prefix),
        "events": events,
        "pending_releases_dropped_at_episode_end": dropped,
        "release_snapshot_count": len(snapshots),
        "release_snapshot_bundle": release_bundle,
        "release_snapshot_manifest": release_manifest,
        "release_snapshot_bundle_sha256": release_digest,
    }
    event_path = event_dir / f"event_log_{prefix}_{int(ep)}.json"
    event_path.write_text(
        json.dumps(event_payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    return save_episode_results(
        ep=int(ep),
        cfg=runtime_cfg,
        metrics_agg=metrics_agg,
        collision_frame=int(collision_frame),
        result_dir=str(root),
        prefix=str(prefix),
        frame_runtimes=list(frame_runtimes or []),
        phys_rec=phys_rec,
        reas_rec=reas_rec,
        event_log_path=str(event_path),
        event_payload=event_payload,
    )


__all__ = ["finalize_episode_outputs"]
