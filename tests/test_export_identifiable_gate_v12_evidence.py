from __future__ import annotations

import json

import pytest

from tools import export_identifiable_gate_v12_evidence as exporter
from tools import audit_tvt_manuscript


def _validation_manifest() -> dict:
    source = {
        "calibration_manifest_sha256": "a" * 64,
        "go_no_go_manifest_sha256": "b" * 64,
        "trace_semantic_hash": "c" * 64,
        "trace_raw_file_set_hash": "d" * 64,
        "branch_label_semantic_hash": "e" * 64,
        "branch_label_raw_sha256": "f" * 64,
        "lock_sha256": "2" * 64,
        "protocol_sha256": "1" * 64,
        "selector_sha256": "3" * 64,
        "gate_support_sha256": "4" * 64,
        "holdout_authorization_id": "authorization-1",
        "holdout_authorization_sha256": "5" * 64,
        "holdout_final_stage": "consumed",
    }
    acceptance = {
        "scope": "confirmatory_holdout",
        "validation_evaluated": True,
        "validation_passed": True,
        "paper_facing_passed": True,
        "passed": True,
    }
    manifest = {
        "schema": exporter.VALIDATION_SCHEMA,
        "artifact_role": "validation_locked_analysis",
        "partition": "validation",
        "method_version": exporter.METHOD_VERSION,
        "seed_block": list(exporter.CONFIRMATORY_SEEDS),
        "locked_thresholds": {
            "candidate_id": "L05_A10_H15_I20",
            "lambda_L": 0.05,
            "lambda_A": 0.10,
            "lambda_H": 0.15,
            "lambda_I": 0.20,
        },
        "parameter_search_performed": False,
        "metrics": {"full": {}},
        "geometry": {"passed": True},
        "source": source,
        "acceptance": acceptance,
        "paper_acceptance": dict(acceptance),
    }
    _refresh_analysis_digest(manifest)
    return manifest


def _refresh_analysis_digest(manifest: dict) -> None:
    manifest["analysis_digest"] = exporter._semantic_hash(
        {
            "partition": "validation",
            "seed_block": list(exporter.CONFIRMATORY_SEEDS),
            "locked_thresholds": manifest["locked_thresholds"],
            "source": manifest["source"],
            "metrics": manifest["metrics"],
            "geometry": manifest["geometry"],
            "acceptance": manifest["acceptance"],
        }
    )


def _validate(manifest: dict) -> None:
    exporter.validate_validation_manifest(
        manifest,
        protocol_sha256="1" * 64,
        lock_sha256="2" * 64,
    )


def test_validation_manifest_passes_only_for_confirmatory_paper_acceptance():
    _validate(_validation_manifest())


def test_validation_manifest_rejects_failed_acceptance_even_with_fresh_digest():
    manifest = _validation_manifest()
    manifest["acceptance"]["validation_passed"] = False
    manifest["acceptance"]["paper_facing_passed"] = False
    manifest["acceptance"]["passed"] = False
    manifest["paper_acceptance"] = dict(manifest["acceptance"])
    _refresh_analysis_digest(manifest)

    with pytest.raises(ValueError, match="validation acceptance failed"):
        _validate(manifest)


def test_validation_manifest_rejects_seed_tamper():
    manifest = _validation_manifest()
    manifest["seed_block"][-1] = 3030

    with pytest.raises(ValueError, match="exactly 3000-3029"):
        _validate(manifest)


def test_validation_manifest_rejects_schema_tamper():
    manifest = _validation_manifest()
    manifest["schema"] = "identifiable_gate_v12_locked_analysis_v0"

    with pytest.raises(ValueError, match="schema drift"):
        _validate(manifest)


def test_registered_input_hash_tamper_fails_closed(tmp_path):
    source = tmp_path / "validation.json"
    source.write_text(json.dumps({"passed": True}) + "\n", encoding="utf-8")
    provenance = exporter._build_provenance({"validation_manifest": source})
    exporter._revalidate_provenance(provenance)

    source.write_text(json.dumps({"passed": False}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed during export"):
        exporter._revalidate_provenance(provenance)


def test_branch_manifest_rejects_payload_hash_tamper(tmp_path):
    trace_root = tmp_path / "traces"
    trace_root.mkdir()
    validation = _validation_manifest()
    source = validation["source"]
    manifest = {
        "schema": exporter.BRANCH_SCHEMA,
        "status": "complete",
        "artifact_role": "confirmatory_holdout_branch_labels",
        "partition": "confirmatory_holdout",
        "v12_partition": "validation",
        "method_version": exporter.METHOD_VERSION,
        "gate_selection_performed": False,
        "exact_action_provenance": "exact",
        "seeds": list(exporter.CONFIRMATORY_SEEDS),
        "trace_root": str(trace_root.resolve()),
        "authorization_id": source["holdout_authorization_id"],
        "authorization_sha256": source["holdout_authorization_sha256"],
    }
    manifest["manifest_payload_hash"] = exporter._payload_sha256(
        manifest, "manifest_payload_hash"
    )
    exporter._validate_branch_manifest(
        manifest,
        trace_root=trace_root,
        validation_source=source,
    )

    manifest["tampered_after_signing"] = True
    with pytest.raises(ValueError, match="payload hash drift"):
        exporter._validate_branch_manifest(
            manifest,
            trace_root=trace_root,
            validation_source=source,
        )


def test_manuscript_audit_dispatches_v12_bundle_without_relaxing_v11_verifier(
    tmp_path, monkeypatch
):
    verified = []
    monkeypatch.setattr(
        exporter,
        "verify_published_bundle",
        lambda root: verified.append(root.resolve()),
    )
    resolved = {
        "release_rollout": {"root": tmp_path, "manifest": {}},
        "component_ablation": {
            "root": tmp_path,
            "manifest": {"schema": exporter.COMPONENT_SCHEMA},
        },
    }

    errors, _ = audit_tvt_manuscript.audit_result_shapes(
        {"tvt_submission_contract": {}}, resolved
    )

    assert errors == []
    assert verified == [tmp_path.resolve()]
