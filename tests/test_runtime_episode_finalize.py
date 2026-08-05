import json
from pathlib import Path

import pytest

import dilu.runtime_episode_finalize as episode_finalize


def _finalize(
    tmp_path,
    *,
    events,
    snapshots,
    pending_latency_queue=None,
    monkeypatch=None,
    cfg_overrides=None,
):
    if monkeypatch is not None:
        monkeypatch.setattr(
            episode_finalize,
            "save_release_snapshot_bundle",
            lambda _snapshots, output_dir, *, prefix, episode_id: (
                Path(output_dir) / f"{prefix}_{episode_id}.pkl",
                Path(output_dir) / f"{prefix}_{episode_id}.json",
                "bundle-sha256",
            ),
        )
        monkeypatch.setattr(
            episode_finalize,
            "save_episode_results",
            lambda **_kwargs: {"saved": True},
        )
    return episode_finalize.finalize_episode_outputs(
        ep=0,
        cfg={
            "require_release_snapshot_on_release": True,
            **dict(cfg_overrides or {}),
        },
        agent=None,
        metrics_agg=None,
        docs=[],
        collision_frame=-1,
        result_dir=str(tmp_path),
        prefix="contract",
        frame_runtimes=[],
        phys_rec=None,
        reas_rec=None,
        event_log=events,
        pending_latency_queue=pending_latency_queue,
        release_snapshots=snapshots,
    )


def _release_event(request_id):
    return {
        "closed_loop_latency_release_event": True,
        "closed_loop_latency_terminal_request_id": request_id,
    }


def test_finalize_rejects_release_without_terminal_request_id(tmp_path):
    event = {
        **_release_event(""),
        "closed_loop_latency_request_id": "legacy-fallback-must-not-be-used",
    }

    with pytest.raises(RuntimeError, match="non-empty terminal request ID"):
        _finalize(tmp_path, events=[event], snapshots={})


def test_finalize_rejects_duplicate_release_terminal_request_id(tmp_path):
    with pytest.raises(RuntimeError, match="duplicate release terminal request ID"):
        _finalize(
            tmp_path,
            events=[_release_event("request-a"), _release_event("request-a")],
            snapshots={"request-a": object()},
        )


def test_finalize_requires_a_snapshot_bijection_for_release_events(
    tmp_path, monkeypatch
):
    result = _finalize(
        tmp_path,
        events=[_release_event("request-a"), _release_event("request-b")],
        snapshots={"request-a": object(), "request-b": object()},
        monkeypatch=monkeypatch,
    )

    assert result == {"saved": True}
    event_bundle = json.loads(
        (tmp_path / "event_logs" / "event_log_contract_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert event_bundle["release_snapshot_count"] == 2
    assert event_bundle["release_snapshot_bundle"] == (
        "release_snapshots/contract_0.pkl"
    )
    assert event_bundle["release_snapshot_manifest"] == (
        "release_snapshots/contract_0.json"
    )
    assert event_bundle["release_snapshot_bundle_sha256"] == "bundle-sha256"


def test_finalize_writes_empty_snapshot_provenance_without_releases(
    tmp_path, monkeypatch
):
    _finalize(
        tmp_path,
        events=[],
        snapshots={},
        monkeypatch=monkeypatch,
    )

    event_bundle = json.loads(
        (tmp_path / "event_logs" / "event_log_contract_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert event_bundle["release_snapshot_count"] == 0
    assert event_bundle["release_snapshot_bundle"] is None
    assert event_bundle["release_snapshot_manifest"] is None
    assert event_bundle["release_snapshot_bundle_sha256"] is None


def test_finalize_marks_pending_requests_as_dropped_at_episode_end(
    tmp_path, monkeypatch
):
    _finalize(
        tmp_path,
        events=[],
        snapshots={},
        pending_latency_queue=[{"request_id": "request-a", "source_frame": 4}],
        monkeypatch=monkeypatch,
    )

    event_bundle = json.loads(
        (tmp_path / "event_logs" / "event_log_contract_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert event_bundle["pending_releases_dropped_at_episode_end"] == [
        {
            "request_id": "request-a",
            "source_frame": 4,
            "terminal_outcome": "dropped_at_episode_end",
        }
    ]


def test_finalize_writes_requested_supported_event_schema(tmp_path, monkeypatch):
    _finalize(
        tmp_path,
        events=[],
        snapshots={},
        monkeypatch=monkeypatch,
        cfg_overrides={"event_log_schema_version": "rgd_event_log_v3"},
    )

    event_bundle = json.loads(
        (tmp_path / "event_logs" / "event_log_contract_0.json").read_text(
            encoding="utf-8"
        )
    )
    assert event_bundle["schema_version"] == "rgd_event_log_v3"


def test_finalize_rejects_unsupported_event_schema(tmp_path):
    with pytest.raises(RuntimeError, match="unsupported event-log schema"):
        _finalize(
            tmp_path,
            events=[],
            snapshots={},
            cfg_overrides={"event_log_schema_version": "unknown"},
        )


def test_finalize_rejects_non_bijective_snapshot_coverage(tmp_path):
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        _finalize(
            tmp_path,
            events=[_release_event("request-a")],
            snapshots={"request-b": object()},
        )
