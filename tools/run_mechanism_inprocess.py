"""Snapshot-provenance helpers for the in-process v12 mechanism acquisition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from dilu.driver_agent.policy_state import DRIVER_POLICY_STATE_SCHEMA


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PRODUCER_PATH = Path(__file__).resolve()
BASE_CONFIG_PATH = REPO_ROOT / "config.yaml"


def _sha256(path: Path) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_snapshot_acquisition_provenance(
    result_dir: Path,
    protocol_path: Path,
    artifacts: Mapping[str, Path],
) -> dict:
    """Bind historical traces and snapshot bundles to their acquisition code."""
    required = ("reasoning", "physical", "snapshot_bundle")
    paths = {label: Path(artifacts[label]).resolve() for label in required}
    provenance = {
        "schema_version": 2,
        "policy_state_schema": DRIVER_POLICY_STATE_SCHEMA,
        "policy_state_integrity": "canonical_json_sha256",
        "producer_path": str(SNAPSHOT_PRODUCER_PATH),
        "producer_sha256": _sha256(SNAPSHOT_PRODUCER_PATH),
        "base_config_path": str(BASE_CONFIG_PATH.resolve()),
        "base_config_sha256": _sha256(BASE_CONFIG_PATH),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": _sha256(Path(protocol_path)),
        "artifact_hashes": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
    }
    root = Path(result_dir)
    for name in ("experiment_snapshot.json", "runtime_manifest.json"):
        path = root / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["snapshot_acquisition"] = provenance
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return provenance
