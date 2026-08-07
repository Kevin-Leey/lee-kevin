import json
from copy import deepcopy
from pathlib import Path

import pytest

import tools.calibrate_identifiable_gate_v12 as calibration
from tools.v12_floor_overlay import (
    APPLIED_STATUS,
    DEFAULT_LOCK_PATH,
    DEFAULT_PROTOCOL_PATH,
    FLOOR_FIELDS,
    FLOOR_SELECTION_SOURCE,
    PROTOCOL_PLACEHOLDER_STATUS,
    apply_floor_overlay,
    assert_floor_overlay_applied,
    canonical_sha256,
    classify_v12_partition,
    create_floor_overlay,
    enforce_v12_floor_overlay_contract,
    load_optional_verified_floor_overlay,
    load_verified_floor_overlay,
)
from tools.analyze_release_state_rollouts import _build_fast_config
from tools.run_main_table_runtime import (
    build_group_config,
    load_formal_base_config,
    load_formal_protocol,
)
from dilu.evaluation.reporter import build_experiment_identity


def _write_calibration_manifest(path: Path) -> dict:
    spec = calibration.load_spec(DEFAULT_LOCK_PATH)
    threshold = calibration.Thresholds(5, 55, 10, 20)
    selected = {"candidate_id": threshold.candidate_id, **threshold.as_floats()}
    source = {
        "lock_sha256": calibration._sha256(DEFAULT_LOCK_PATH),
        "protocol_sha256": calibration._sha256(DEFAULT_PROTOCOL_PATH),
        "selector_sha256": calibration._sha256(Path(calibration.__file__)),
        "gate_support_sha256": calibration._sha256(calibration.GATE_SUPPORT_PATH),
        "trace_semantic_hash": "1" * 64,
        "branch_label_semantic_hash": "2" * 64,
    }
    digest_payload = {
        "lock_sha256": source["lock_sha256"],
        "protocol_sha256": source["protocol_sha256"],
        "gate_support_sha256": source["gate_support_sha256"],
        "trace_semantic_hash": source["trace_semantic_hash"],
        "label_semantic_hash": source["branch_label_semantic_hash"],
        "selected": selected,
    }
    manifest = {
        "schema": "identifiable_gate_v12_calibration_selection_v1",
        "artifact_role": "calibration_lock",
        "method_version": spec.method_version,
        "lock_id": spec.lock_id,
        "calibration_seed_block": {"seeds": list(spec.seeds)},
        "candidate_space_hash": calibration._semantic_hash(
            [candidate.candidate_id for candidate in calibration.candidates(spec)]
        ),
        "source": source,
        "selected": selected,
        "calibration_constraints_satisfied": True,
        "paper_acceptance": {
            "scope": "calibration_only",
            "validation_evaluated": False,
            "validation_passed": None,
            "paper_facing_passed": False,
        },
        "selection_digest": calibration._semantic_hash(digest_payload),
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _placeholder_config() -> dict:
    return {
        "slow_thinking": {
            "risk_coupling": {
                "core_story": {
                    **{field: 0.20 for field in FLOOR_FIELDS},
                    "v12_floor_status": PROTOCOL_PLACEHOLDER_STATUS,
                }
            }
        }
    }


@pytest.fixture
def overlay_bundle(tmp_path):
    calibration_path = tmp_path / "v12_calibration_manifest.json"
    manifest = _write_calibration_manifest(calibration_path)
    overlay_path = tmp_path / "v12_runtime_floor_overlay.json"
    verified = create_floor_overlay(
        overlay_path,
        calibration_manifest_path=calibration_path,
        protocol_path=DEFAULT_PROTOCOL_PATH,
        lock_path=DEFAULT_LOCK_PATH,
    )
    return calibration_path, manifest, overlay_path, verified


def test_verified_overlay_applies_exact_tuple_and_runtime_provenance(overlay_bundle):
    calibration_path, manifest, overlay_path, verified = overlay_bundle
    assert verified.path == overlay_path.resolve()
    assert verified.calibration_manifest_path == calibration_path.resolve()
    assert verified.selection_digest == manifest["selection_digest"]
    assert verified.floors == {
        "rgd_latency_survival_floor": 0.05,
        "rgd_maneuver_breadth_floor": 0.55,
        "rgd_corrective_headroom_floor": 0.10,
        "rgd_state_need_floor": 0.20,
    }

    cfg = apply_floor_overlay(_placeholder_config(), verified)
    core = cfg["slow_thinking"]["risk_coupling"]["core_story"]
    assert core["v12_floor_status"] == APPLIED_STATUS
    for field, value in verified.floors.items():
        assert core[field] == value
        assert core[f"{field}_source"] == FLOOR_SELECTION_SOURCE
    assert core["rgd_floor_calibration_manifest_sha256"] == verified.calibration_manifest_sha256
    assert core["rgd_floor_selection_digest"] == verified.selection_digest
    assert core["rgd_floor_overlay_sha256"] == verified.raw_sha256
    assert cfg["_v12_floor_overlay"]["candidate_id"] == verified.candidate_id
    assert_floor_overlay_applied(cfg, verified)


def test_optional_overlay_loader_requires_the_authenticated_input_pair(overlay_bundle):
    calibration_path, _, overlay_path, verified = overlay_bundle
    assert (
        load_optional_verified_floor_overlay(
            None,
            calibration_manifest_path=None,
            protocol_path=DEFAULT_PROTOCOL_PATH,
            lock_path=DEFAULT_LOCK_PATH,
        )
        is None
    )
    with pytest.raises(ValueError, match="supplied together"):
        load_optional_verified_floor_overlay(
            overlay_path,
            calibration_manifest_path=None,
            protocol_path=DEFAULT_PROTOCOL_PATH,
            lock_path=DEFAULT_LOCK_PATH,
        )
    loaded = load_optional_verified_floor_overlay(
        overlay_path,
        calibration_manifest_path=calibration_path,
        protocol_path=DEFAULT_PROTOCOL_PATH,
        lock_path=DEFAULT_LOCK_PATH,
    )
    assert loaded is not None
    assert loaded.runtime_binding == verified.runtime_binding


def test_branch_fast_config_consumes_overlay_before_protocol_snapshot(
    overlay_bundle, tmp_path
):
    _, _, _, verified = overlay_bundle
    cfg = _build_fast_config(
        DEFAULT_PROTOCOL_PATH,
        2040,
        tmp_path / "branch",
        verified_floor_overlay=verified,
    )
    assert_floor_overlay_applied(cfg, verified)
    embedded = cfg["_paper_protocol_config"]["runtime_config"]
    assert embedded["_v12_floor_overlay"] == cfg["_v12_floor_overlay"]
    assert (
        embedded["slow_thinking"]["risk_coupling"]["core_story"]
        ["v12_floor_status"]
        == APPLIED_STATUS
    )


def test_overlay_rejects_raw_tampering_even_when_calibration_is_unchanged(overlay_bundle):
    calibration_path, _, overlay_path, _ = overlay_bundle
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    payload["floors"]["rgd_latency_survival_floor"]["value"] = 0.90
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        load_verified_floor_overlay(
            overlay_path,
            calibration_manifest_path=calibration_path,
            protocol_path=DEFAULT_PROTOCOL_PATH,
            lock_path=DEFAULT_LOCK_PATH,
        )


def test_overlay_rejects_semantic_tampering_after_attacker_rehashes_payload(overlay_bundle):
    calibration_path, _, overlay_path, _ = overlay_bundle
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    payload["floors"]["rgd_latency_survival_floor"]["value"] = 0.90
    payload.pop("overlay_payload_sha256")
    payload["overlay_payload_sha256"] = canonical_sha256(payload)
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the verified calibration selection"):
        load_verified_floor_overlay(
            overlay_path,
            calibration_manifest_path=calibration_path,
            protocol_path=DEFAULT_PROTOCOL_PATH,
            lock_path=DEFAULT_LOCK_PATH,
        )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda cfg: cfg["slow_thinking"]["risk_coupling"]["core_story"].update(
                {"rgd_latency_survival_floor": 0.25}
            ),
            "protocol/CLI floor override",
        ),
        (
            lambda cfg: cfg["slow_thinking"]["risk_coupling"]["core_story"].update(
                {"v12_floor_status": "edited"}
            ),
            "registered v12 floor status",
        ),
    ],
)
def test_formal_apply_rejects_protocol_or_cli_override(overlay_bundle, mutate, message):
    _, _, _, verified = overlay_bundle
    cfg = _placeholder_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        apply_floor_overlay(cfg, verified)


def test_runtime_assertion_rejects_post_apply_floor_override(overlay_bundle):
    _, _, _, verified = overlay_bundle
    cfg = apply_floor_overlay(_placeholder_config(), verified)
    tampered = deepcopy(cfg)
    tampered["slow_thinking"]["risk_coupling"]["core_story"][
        "rgd_corrective_headroom_floor"
    ] = 0.99
    with pytest.raises(ValueError, match="runtime floor drift"):
        assert_floor_overlay_applied(tampered, verified)


def test_overlay_is_single_use_for_a_config_object(overlay_bundle):
    _, _, _, verified = overlay_bundle
    cfg = apply_floor_overlay(_placeholder_config(), verified)
    with pytest.raises(ValueError, match="already contains a floor overlay"):
        apply_floor_overlay(cfg, verified)


def test_formal_group_and_manifest_keep_the_same_overlay_binding(overlay_bundle, tmp_path):
    _, _, _, verified = overlay_bundle
    protocol = load_formal_protocol(DEFAULT_PROTOCOL_PATH)
    base = load_formal_base_config(protocol)
    base = apply_floor_overlay(base, verified)
    group_cfg = dict(protocol["groups"]["random_budget"])
    cfg = build_group_config(
        base,
        "random_budget",
        group_cfg,
        "highway-v0",
        1,
        tmp_path / "run",
        protocol,
    )
    assert_floor_overlay_applied(cfg, verified)

    identity = build_experiment_identity(cfg, 4000)
    protocol_manifest = identity["protocol_manifest"]
    assert protocol_manifest["v12_floor_overlay"]["floor_overlay_sha256"] == verified.raw_sha256
    assert (
        protocol_manifest["config"]["v12_floor_overlay"]["calibration_selection_digest"]
        == verified.selection_digest
    )


@pytest.mark.parametrize(
    "seed, partition",
    [(2000, "calibration"), (2040, "go_no_go"), (3000, "confirmatory_holdout"), (4000, "main")],
)
def test_common_partition_classifier_accepts_locked_subsets(seed, partition):
    assert classify_v12_partition([seed, seed + 1]) == partition


def test_common_partition_contract_requires_overlay_after_calibration(overlay_bundle):
    _, _, _, verified = overlay_bundle
    protocol_name = "rgd_tvt_identifiable_gate_v12"
    assert enforce_v12_floor_overlay_contract(protocol_name, [2000, 2001], None) == "calibration"
    with pytest.raises(ValueError, match="cannot consume"):
        enforce_v12_floor_overlay_contract(protocol_name, [2000], verified)
    for seeds, partition in [
        ([2040], "go_no_go"),
        ([3000, 3029], "confirmatory_holdout"),
        ([4000, 4029], "main"),
    ]:
        with pytest.raises(ValueError, match="requires a verified"):
            enforce_v12_floor_overlay_contract(protocol_name, seeds, None)
        assert enforce_v12_floor_overlay_contract(protocol_name, seeds, verified) == partition


def test_common_partition_contract_rejects_unknown_or_cross_partition_seed_requests():
    protocol_name = "rgd_tvt_identifiable_gate_v12"
    with pytest.raises(ValueError, match="outside one frozen partition"):
        enforce_v12_floor_overlay_contract(protocol_name, [2060], None)
    with pytest.raises(ValueError, match="spans partitions"):
        enforce_v12_floor_overlay_contract(protocol_name, [2039, 2040], None)
