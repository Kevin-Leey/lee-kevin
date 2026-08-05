import json
from pathlib import Path

import pytest
import yaml

from tools.analyze_rgd_latency_surface import DEPRECATION_MESSAGE, main as latency_surface_main
from tools.audit_tvt_manuscript import audit_evidence_provenance


ROOT = Path(__file__).resolve().parents[1]


def _submission(*, status="current", manifest_method="support_breadth_v11"):
    return {
        "rgd_method_version": "support_breadth_v11",
        "prior_paper_result_method_version": "raw_affordance_v10",
        "component_validation_seeds": "1000-1001",
        "evidence_artifacts": {
            "compatibility": {
                "required_method_version": "support_breadth_v11",
                "prior_method_version": "raw_affordance_v10",
                "cross_method_version_policy": "reject",
                "missing_manifest_method_version_policy": "reject",
            },
            "artifacts": {
                "component_ablation": {
                    "paper_facing": True,
                    "status": status,
                    "method_version": "support_breadth_v11",
                    "artifact": "evidence/component",
                    "manifest": "manifest.json",
                    "data_files": ["summary.csv"],
                    "seed_contract": "component_validation_seeds",
                    "required_manifest_values": {
                        "method_version": "support_breadth_v11",
                        "alternative_metric_source": "action_support_ranking_costs",
                        "absolute_alternative_feasibility_non_ablatable": True,
                        "legal_action_provenance": "exact",
                        "paper_acceptance.metric_passed": True,
                        "paper_acceptance.passed": True,
                    },
                }
            },
        },
        "_manifest_method": manifest_method,
    }


def _write_artifact(root: Path, submission, *, seeds=(1000, 1001), include_method=True):
    artifact = root / "evidence" / "component"
    artifact.mkdir(parents=True)
    manifest = {
        "seeds": list(seeds),
        "alternative_metric_source": "action_support_ranking_costs",
        "absolute_alternative_feasibility_non_ablatable": True,
        "legal_action_provenance": "exact",
        "paper_acceptance": {"metric_passed": True, "passed": True},
    }
    if include_method:
        manifest["method_version"] = submission.pop("_manifest_method")
    else:
        submission.pop("_manifest_method")
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (artifact / "summary.csv").write_text("arm\nRGD\n", encoding="utf-8")


def test_current_v11_artifact_with_locked_cohort_is_accepted(tmp_path):
    submission = _submission()
    _write_artifact(tmp_path, submission)

    errors, resolved = audit_evidence_provenance(submission, tmp_path)

    assert errors == []
    assert set(resolved) == {"component_ablation"}


def test_legacy_v10_manifest_is_rejected_even_under_a_current_contract(tmp_path):
    submission = _submission(manifest_method="raw_affordance_v10")
    _write_artifact(tmp_path, submission)

    errors, resolved = audit_evidence_provenance(submission, tmp_path)

    assert any("legacy raw_affordance_v10" in error for error in errors)
    assert resolved == {}


def test_unversioned_manifest_fails_closed(tmp_path):
    submission = _submission()
    _write_artifact(tmp_path, submission, include_method=False)

    errors, resolved = audit_evidence_provenance(submission, tmp_path)

    assert any("omits method_version" in error for error in errors)
    assert resolved == {}


def test_seed_cohort_must_match_the_protocol_partition(tmp_path):
    submission = _submission()
    _write_artifact(tmp_path, submission, seeds=(1000, 1002))

    errors, resolved = audit_evidence_provenance(submission, tmp_path)

    assert any("seed cohort differs" in error for error in errors)
    assert resolved == {}


def test_pending_paper_artifact_is_an_explicit_audit_failure(tmp_path):
    submission = _submission(status="pending_v11_rerun")
    submission.pop("_manifest_method")

    errors, resolved = audit_evidence_provenance(submission, tmp_path)

    assert resolved == {}
    assert any("not current" in error and "cannot be mixed" in error for error in errors)


def test_deprecated_latency_surface_never_writes_output(tmp_path, capsys):
    output = tmp_path / "paper_facing_latency.csv"
    with pytest.raises(SystemExit) as exc_info:
        latency_surface_main(
            ["--fast-root", str(tmp_path / "traces"), "--output", str(output)]
        )

    assert exc_info.value.code == 2
    assert not output.exists()
    assert DEPRECATION_MESSAGE in capsys.readouterr().err


def test_formal_protocol_declares_cross_version_rejection():
    protocol = yaml.safe_load((ROOT / "formal_protocol.yaml").read_text(encoding="utf-8"))
    submission = protocol["tvt_submission_contract"]
    compatibility = submission["evidence_artifacts"]["compatibility"]

    assert compatibility["required_method_version"] == submission["rgd_method_version"]
    assert compatibility["prior_method_version"] == submission["prior_paper_result_method_version"]
    assert compatibility["cross_method_version_policy"] == "reject"
    assert compatibility["missing_manifest_method_version_policy"] == "reject"


def test_audit_tools_do_not_embed_legacy_result_paths_or_thresholds():
    audit_source = (ROOT / "tools" / "audit_tvt_manuscript.py").read_text(encoding="utf-8")
    latency_source = (ROOT / "tools" / "analyze_rgd_latency_surface.py").read_text(encoding="utf-8")
    combined = audit_source + latency_source

    assert "tvt_final_20260718" not in combined
    assert "component_ablation_v1" not in combined
    assert "0.16" not in combined
    assert "range(160, 190)" not in combined
