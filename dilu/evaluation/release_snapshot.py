"""Versioned online release-state snapshots for matched replay.

The snapshot is captured immediately before the release-frame policy decision.
It therefore preserves the simulator state, observation, executed-action history,
and complete allowlisted policy state that jointly determine the matched Fast
continuation. Request metadata binds the snapshot to one asynchronous response.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from dilu.driver_agent.policy_state import (
    DRIVER_POLICY_STATE_SCHEMA,
    policy_state_sha256,
    validate_driver_policy_state,
)


RELEASE_SNAPSHOT_SCHEMA = "rgd_release_snapshot_v2"
RELEASE_SNAPSHOT_BUNDLE_SCHEMA = "rgd_release_snapshot_bundle_v1"
RELEASE_SNAPSHOT_CAPTURE_STAGE = "pre_release_frame_policy_decision"


@dataclass
class ReleaseSnapshot:
    frame: int
    env: Any
    obs: Any
    fast_state: Dict[str, Any]
    history: collections.deque
    previous_action: int
    policy_state_schema: Optional[str] = None
    policy_state: Optional[Dict[str, Any]] = None
    policy_state_sha256: Optional[str] = None
    schema: str = RELEASE_SNAPSHOT_SCHEMA
    request_id: str = ""
    source_frame: Optional[int] = None
    scheduled_release_frame: Optional[int] = None
    capture_stage: str = RELEASE_SNAPSHOT_CAPTURE_STAGE
    snapshot_identity_sha256: str = ""


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_payload(snapshot: ReleaseSnapshot) -> Dict[str, Any]:
    return {
        "schema": str(snapshot.schema),
        "capture_stage": str(snapshot.capture_stage),
        "frame": int(snapshot.frame),
        "request_id": str(snapshot.request_id or ""),
        "source_frame": (
            None if snapshot.source_frame is None else int(snapshot.source_frame)
        ),
        "scheduled_release_frame": (
            None
            if snapshot.scheduled_release_frame is None
            else int(snapshot.scheduled_release_frame)
        ),
        "previous_action": int(snapshot.previous_action),
        "policy_state_schema": str(snapshot.policy_state_schema or ""),
        "policy_state_sha256": str(snapshot.policy_state_sha256 or ""),
    }


def capture_release_snapshot(
    agent: Any,
    *,
    frame: int,
    env: Any,
    obs: Any,
    history: collections.deque,
    previous_action: int,
    pending_request: Optional[Mapping[str, Any]] = None,
) -> ReleaseSnapshot:
    """Capture an authenticated replay target before a queued release is handled."""
    policy_state = agent.snapshot_policy_state()
    request = dict(pending_request or {})
    snapshot = ReleaseSnapshot(
        frame=int(frame),
        env=copy.deepcopy(env),
        obs=copy.deepcopy(obs),
        fast_state=copy.deepcopy(agent.fast_thinker.snapshot_runtime_state()),
        history=copy.deepcopy(history),
        previous_action=int(previous_action),
        policy_state_schema=DRIVER_POLICY_STATE_SCHEMA,
        policy_state=policy_state,
        policy_state_sha256=policy_state_sha256(policy_state),
        request_id=str(request.get("request_id", "") or ""),
        source_frame=(
            None
            if request.get("source_frame") is None
            else int(request["source_frame"])
        ),
        scheduled_release_frame=(
            None
            if request.get("release_frame") is None
            else int(request["release_frame"])
        ),
    )
    snapshot.snapshot_identity_sha256 = _canonical_sha256(
        _identity_payload(snapshot)
    )
    return snapshot


def validate_release_snapshot_policy_state(
    snapshot: ReleaseSnapshot,
    *,
    context: str,
) -> Dict[str, Any]:
    """Authenticate the complete policy state and snapshot/request identity."""
    if getattr(snapshot, "schema", RELEASE_SNAPSHOT_SCHEMA) != RELEASE_SNAPSHOT_SCHEMA:
        raise ValueError(f"{context}: release-snapshot schema drift")
    if getattr(snapshot, "capture_stage", RELEASE_SNAPSHOT_CAPTURE_STAGE) != RELEASE_SNAPSHOT_CAPTURE_STAGE:
        raise ValueError(f"{context}: invalid release-snapshot capture stage")
    if getattr(snapshot, "policy_state_schema", None) != DRIVER_POLICY_STATE_SCHEMA:
        raise ValueError(f"{context}: policy-state schema drift")
    try:
        state = validate_driver_policy_state(
            getattr(snapshot, "policy_state", None)
        )
    except ValueError as exc:
        raise ValueError(f"{context}: invalid policy state: {exc}") from exc

    recorded = str(getattr(snapshot, "policy_state_sha256", "") or "")
    if len(recorded) != 64 or any(
        character not in "0123456789abcdef" for character in recorded
    ):
        raise ValueError(f"{context}: invalid policy-state SHA256")
    if recorded != policy_state_sha256(state):
        raise ValueError(f"{context}: policy-state SHA256 mismatch")

    identity = str(getattr(snapshot, "snapshot_identity_sha256", "") or "")
    if identity:
        expected = _canonical_sha256(_identity_payload(snapshot))
        if identity != expected:
            raise ValueError(f"{context}: release-snapshot identity mismatch")
    return state


def snapshot_manifest_row(snapshot: ReleaseSnapshot) -> Dict[str, Any]:
    validate_release_snapshot_policy_state(
        snapshot,
        context=f"release snapshot {snapshot.request_id or snapshot.frame}",
    )
    return {
        **_identity_payload(snapshot),
        "snapshot_identity_sha256": str(
            snapshot.snapshot_identity_sha256
            or _canonical_sha256(_identity_payload(snapshot))
        ),
    }


def save_release_snapshot_bundle(
    snapshots: Mapping[str, ReleaseSnapshot],
    output_dir: Path,
    *,
    prefix: str,
    episode_id: int,
) -> Tuple[Path, Path, str]:
    """Atomically persist request-keyed snapshots and a JSON audit manifest."""
    normalized: Dict[str, ReleaseSnapshot] = {}
    rows = []
    for key, snapshot in sorted(snapshots.items(), key=lambda item: str(item[0])):
        request_id = str(key or "")
        if not request_id:
            raise ValueError("online release snapshot has an empty request ID")
        if str(snapshot.request_id or "") != request_id:
            raise ValueError(
                f"release snapshot key/request mismatch: {request_id!r} != "
                f"{snapshot.request_id!r}"
            )
        normalized[request_id] = snapshot
        rows.append(snapshot_manifest_row(snapshot))

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"release_snapshots_{prefix}_{int(episode_id)}"
    bundle_path = output_dir / f"{stem}.pkl"
    manifest_path = output_dir / f"{stem}.json"
    bundle_payload = {
        "schema": RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
        "episode_id": int(episode_id),
        "snapshots": normalized,
    }
    bundle_bytes = pickle.dumps(bundle_payload, protocol=pickle.HIGHEST_PROTOCOL)
    bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()
    manifest = {
        "schema": RELEASE_SNAPSHOT_BUNDLE_SCHEMA,
        "episode_id": int(episode_id),
        "snapshot_count": len(rows),
        "bundle_file": bundle_path.name,
        "bundle_sha256": bundle_sha256,
        "snapshots": rows,
    }

    bundle_tmp = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
    manifest_tmp = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    bundle_tmp.write_bytes(bundle_bytes)
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(bundle_tmp), str(bundle_path))
    os.replace(str(manifest_tmp), str(manifest_path))
    return bundle_path, manifest_path, bundle_sha256

