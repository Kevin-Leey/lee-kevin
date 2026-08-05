import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dilu.evaluation.decision_trace import build_decision_meta
from dilu.evaluation.discrepancy_query_gate import (
    DISCREPANCY_GATE_VERSION,
    FEATURE_NAMES,
    DiscrepancyQueryGate,
    canonical_json_sha256,
    extract_query_features,
    validate_discrepancy_artifact,
)
from dilu.evaluation.factorial_replay import (
    ProposalRecord,
    ProposalReplayAgent,
)
from dilu.runtime_frame_trace import build_episode_event
from tools import train_discrepancy_query_gate as training
from tools.run_discrepancy_query_baseline import (
    DISCREPANCY_ARMS,
    DISCREPANCY_BASELINE_VERSION,
)


def _query_metadata(**extra):
    payload = {
        "rgd_execution_route_score": 0.71,
        "recoverability_recovery_window": 0.42,
        "recoverability_post_latency_opportunity": 0.31,
        "recoverability_relative_support_weighted_maneuver_family_breadth": 0.64,
        "recoverability_relative_corrective_headroom": 0.18,
        "recoverability_action_cost_entropy": 0.76,
        "recoverability_absolute_recovery_depth": 0.83,
        "reasoning_latency_pressure": 0.58,
        "confidence": 0.67,
        "recoverability_pre_screen_trigger": True,
        "recoverability_gate_domain_valid": True,
        "recoverability_absolute_alternative_feasible": True,
        "recoverability_gate": {"serial_gate_pass": False},
    }
    payload.update(extra)
    return payload


def _synthetic_source(seed, include_discrepancy_label):
    records = []
    for frame in range(4):
        base = (seed % 100) * 0.013 + frame * 0.17
        features = np.asarray(
            [base + index * 0.019 for index in range(len(FEATURE_NAMES))],
            dtype=np.float64,
        )
        row = {
            "seed": seed,
            "frame": frame,
            "features": features,
            "rgd_query_exposed": frame == (seed % 4),
        }
        if include_discrepancy_label:
            row["label"] = int((seed + frame) % 3 != 0)
        records.append(row)
    digest = hashlib.sha256(str(seed).encode("ascii")).hexdigest()
    provenance = {
        "seed": seed,
        "event_log": {"path": f"seed_{seed}/event.json", "sha256": digest},
        "reasoning_trace": {"path": f"seed_{seed}/reasoning.json", "sha256": digest},
        "experiment_snapshot": {"path": f"seed_{seed}/snapshot.json", "sha256": digest},
        "record_count": len(records),
        "discrepancy_label_read": include_discrepancy_label,
    }
    return records, provenance


@pytest.fixture
def synthetic_artifact(monkeypatch):
    calls = []

    def fake_loader(source_root, seed, *, include_discrepancy_label):
        calls.append((seed, include_discrepancy_label))
        return _synthetic_source(seed, include_discrepancy_label)

    monkeypatch.setattr(training, "_load_seed_records", fake_loader)
    artifact = training.train_discrepancy_artifact(
        Path("synthetic_always_slow"),
        fit_seeds=(10, 11, 12),
        calibration_seeds=(20, 21),
    )
    return artifact, calls


def test_feature_extractor_ignores_all_proposal_and_outcome_fields():
    clean = _query_metadata()
    contaminated_a = {
        **clean,
        "query_state_slow_released_action": 0,
        "factorial_shared_raw_slow_action": 4,
        "factorial_shared_response_sha256": "a" * 64,
        "factorial_shared_response_outcome": "valid",
        "closed_loop_latency_release_event": True,
        "collision": True,
        "episode_reward": -999.0,
    }
    contaminated_b = {
        **clean,
        "query_state_slow_released_action": 4,
        "factorial_shared_raw_slow_action": 0,
        "factorial_shared_response_sha256": "b" * 64,
        "factorial_shared_response_outcome": "timeout",
        "closed_loop_latency_release_event": False,
        "collision": False,
        "episode_reward": 999.0,
    }

    expected = extract_query_features(clean, fast_action=1)

    np.testing.assert_array_equal(
        extract_query_features(contaminated_a, fast_action=1), expected
    )
    np.testing.assert_array_equal(
        extract_query_features(contaminated_b, fast_action=1), expected
    )


def test_training_is_deterministic_and_calibration_never_reads_slow_labels(
    monkeypatch,
):
    calls = []

    def fake_loader(source_root, seed, *, include_discrepancy_label):
        calls.append((seed, include_discrepancy_label))
        return _synthetic_source(seed, include_discrepancy_label)

    monkeypatch.setattr(training, "_load_seed_records", fake_loader)
    kwargs = {
        "fit_seeds": (10, 11, 12),
        "calibration_seeds": (20, 21),
    }
    first = training.train_discrepancy_artifact(Path("synthetic"), **kwargs)
    second = training.train_discrepancy_artifact(Path("synthetic"), **kwargs)

    assert first == second
    assert first["artifact_sha256"] == canonical_json_sha256(
        {key: value for key, value in first.items() if key != "artifact_sha256"}
    )
    assert first["seed_split"]["fit_seeds"] == [10, 11, 12]
    assert first["seed_split"]["calibration_seeds"] == [20, 21]
    assert first["calibration"]["target_invocations"] == first["calibration"][
        "achieved_invocations"
    ]
    assert all(include for seed, include in calls if seed in {10, 11, 12})
    assert not any(include for seed, include in calls if seed in {20, 21})


def test_calibration_threshold_matches_exact_count_and_rejects_boundary_ties():
    scores = np.asarray([0.9, 0.7, 0.4, 0.1])
    threshold = training._calibrated_threshold(scores, 2)

    assert threshold == pytest.approx(0.55)
    assert int(np.sum(scores >= threshold)) == 2
    with pytest.raises(RuntimeError, match="tie at the RGD exposure boundary"):
        training._calibrated_threshold(np.asarray([0.9, 0.7, 0.7, 0.1]), 2)


def test_artifact_validation_fails_closed_on_hash_or_seed_split_drift(
    synthetic_artifact,
):
    artifact, _ = synthetic_artifact
    tampered = copy.deepcopy(artifact)
    tampered["model"]["logistic_regression"]["intercept"] += 0.1
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_discrepancy_artifact(tampered)

    overlap = copy.deepcopy(artifact)
    overlap["seed_split"]["calibration_seeds"][0] = overlap["seed_split"][
        "fit_seeds"
    ][0]
    overlap["artifact_sha256"] = canonical_json_sha256(
        {key: value for key, value in overlap.items() if key != "artifact_sha256"}
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_discrepancy_artifact(overlap)


def test_runtime_probability_is_invariant_to_frozen_proposal_content(
    synthetic_artifact,
):
    artifact, _ = synthetic_artifact
    gate = DiscrepancyQueryGate(artifact)
    clean = _query_metadata()
    leaked_a = {**clean, "query_state_slow_released_action": 0, "slow_response": "A"}
    leaked_b = {**clean, "query_state_slow_released_action": 4, "slow_response": "B"}

    first = gate.probability(leaked_a, fast_action=1)
    second = gate.probability(leaked_b, fast_action=1)

    assert first == second
    assert 0.0 <= first <= 1.0


class _State:
    legal_actions = tuple(range(5))


class _Inner:
    def __init__(self):
        self.fast_thinker = SimpleNamespace()
        self.external_requests = 0
        self.orchestrator = SimpleNamespace(
            record_external_slow_request=self._record_request
        )

    def _record_request(self):
        self.external_requests += 1

    def decide(self, state):
        return 1, "fast", _query_metadata()

    def snapshot_policy_state(self):
        return {}

    def restore_policy_state(self, snapshot):
        return None

    def record_executed_action(self, action):
        return None


def test_baseline_arms_share_proposal_identity_and_gate_decision(synthetic_artifact):
    artifact, _ = synthetic_artifact
    proposal = ProposalRecord(
        seed=5000,
        source_frame=0,
        request_id="factorial:5000:0:00",
        raw_slow_action=4,
        latency_steps=17,
        response_text="frozen slow response",
        response_sha256="a" * 64,
    )
    observed = []
    for arm in DISCREPANCY_ARMS:
        agent = ProposalReplayAgent(
            _Inner(),
            {0: proposal},
            arm=arm,
            bank_sha256="shared-bank",
            query_admission_policy=DiscrepancyQueryGate(artifact),
        )
        _, _, metadata = agent.decide(_State())
        observed.append(
            (
                metadata["factorial_candidate_request_id"],
                metadata["factorial_shared_latency_steps"],
                metadata["discrepancy_gate_probability"],
                metadata["discrepancy_gate_admit"],
            )
        )
        assert metadata["discrepancy_gate_query_inputs_only"] is True
        assert metadata["discrepancy_gate_version"] == DISCREPANCY_GATE_VERSION

    assert len(set(observed)) == 1
    assert observed[0][0:2] == (proposal.request_id, proposal.latency_steps)
    assert {
        arm.name: (arm.query_gate_enabled, arm.release_guard_enabled)
        for arm in DISCREPANCY_ARMS
    } == {
        "discrepancy_only": (True, False),
        "discrepancy_release": (True, True),
    }
    assert DISCREPANCY_BASELINE_VERSION == "rgd_discrepancy_query_baseline_v1"


def test_discrepancy_audit_survives_decision_and_event_tracing():
    raw = {
        "discrepancy_gate_version": DISCREPANCY_GATE_VERSION,
        "discrepancy_gate_artifact_sha256": "a" * 64,
        "discrepancy_gate_feature_schema_version": "rgd_query_observables_v1",
        "discrepancy_gate_probability": 0.73,
        "discrepancy_gate_threshold": 0.61,
        "discrepancy_gate_admit": True,
        "discrepancy_gate_query_inputs_only": True,
    }
    traced = build_decision_meta(raw, proposed_action=1, final_action=1)
    event = build_episode_event(
        0,
        {},
        traced,
        {"done": False, "term": False, "trunc": False},
    )

    for payload in (traced, event):
        assert payload["discrepancy_gate_probability"] == pytest.approx(0.73)
        assert payload["discrepancy_gate_threshold"] == pytest.approx(0.61)
        assert payload["discrepancy_gate_admit"] is True
        assert payload["discrepancy_gate_query_inputs_only"] is True

