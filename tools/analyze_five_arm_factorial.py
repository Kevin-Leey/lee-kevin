"""Audit and analyze the formal five-arm shared-proposal RGD experiment.

The original four arms identify the query-gate and release-guard main effects.
The fifth arm is a genuine Fast-only control that observes the same candidate
schedule but never issues a slow request.  Simulator seed is the independent
unit for every interval and contrast.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dilu.evaluation.factorial_replay import (  # noqa: E402
    FACTORIAL_REPLAY_VERSION,
    FORMAL_FACTORIAL_ARMS,
)


ANALYSIS_VERSION = "rgd_five_arm_paired_analysis_v1"
ARM_NAMES = tuple(arm.name for arm in FORMAL_FACTORIAL_ARMS)
ARM_BY_NAME = {arm.name: arm for arm in FORMAL_FACTORIAL_ARMS}
METRICS = (
    "collision",
    "success_rate",
    "route_completion",
    "episode_reward",
    "driving_distance",
    "avg_speed",
    "runtime_per_frame",
    "candidate_queries",
    "issued_queries",
    "query_gate_rejections",
    "release_events",
    "primitive_distinct_selections",
)
BINARY_METRICS = frozenset({"collision", "success_rate"})
INTEGER_METRICS = frozenset(
    {
        "collision",
        "candidate_queries",
        "issued_queries",
        "query_gate_rejections",
        "release_events",
        "primitive_distinct_selections",
    }
)
CONTRASTS = {
    "full_minus_fast_only": ("full", "fast_only"),
    "neither_minus_fast_only": ("neither", "fast_only"),
    "full_minus_neither": ("full", "neither"),
    "full_minus_query_only": ("full", "query_only"),
    "full_minus_release_only": ("full", "release_only"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return dict(payload)


def _read_csv(path: Path) -> list[Dict[str, str]]:
    require(path.is_file(), f"missing CSV artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    require(bool(rows), f"empty CSV artifact: {path}")
    return rows


def _number(value: Any, field: str) -> float:
    require(not isinstance(value, bool), f"{field} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    require(math.isfinite(parsed), f"non-finite {field}: {value!r}")
    return parsed


def _integer(value: Any, field: str) -> int:
    parsed = _number(value, field)
    require(parsed == int(parsed), f"non-integral {field}: {value!r}")
    return int(parsed)


def _boolean(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    require(text in {"true", "false"}, f"invalid boolean {field}: {value!r}")
    return text == "true"


def _resolve_within(root: Path, relative: Any, field: str) -> Path:
    candidate = Path(str(relative or ""))
    require(str(candidate) not in {"", "."} and not candidate.is_absolute(), f"invalid {field}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{field} escapes its declared root") from exc
    return resolved


def _validate_proposal_bank(manifest: Mapping[str, Any]) -> tuple[str, Dict[int, Dict[str, Dict[str, Any]]]]:
    payload = manifest.get("bank_payload")
    require(isinstance(payload, list) and payload, "proposal bank payload is empty")
    digest = str(manifest.get("bank_sha256", "") or "")
    require(len(digest) == 64 and _sha256_json(payload) == digest, "proposal-bank hash drift")
    records_by_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for block in payload:
        require(isinstance(block, Mapping), "proposal seed block is not an object")
        seed = _integer(block.get("seed"), "proposal seed")
        require(seed not in records_by_seed, f"duplicate proposal seed {seed}")
        records: Dict[str, Dict[str, Any]] = {}
        frames = set()
        for raw in list(block.get("records", []) or []):
            require(isinstance(raw, Mapping), f"seed {seed}: proposal record is not an object")
            record = dict(raw)
            request_id = str(record.get("request_id", "") or "")
            frame = _integer(record.get("source_frame"), f"seed {seed} source frame")
            require(request_id and request_id not in records, f"seed {seed}: duplicate request ID")
            require(frame not in frames, f"seed {seed}: duplicate source frame")
            response = str(record.get("response_text", "") or "")
            require(
                hashlib.sha256(response.encode("utf-8")).hexdigest()
                == str(record.get("response_sha256", "") or ""),
                f"seed {seed}: proposal response hash drift",
            )
            records[request_id] = record
            frames.add(frame)
        require(bool(records), f"seed {seed}: proposal block is empty")
        records_by_seed[seed] = records
    require(len(records_by_seed) == int(manifest.get("seed_count", -1)), "proposal seed count drift")
    require(
        sum(len(records) for records in records_by_seed.values())
        == int(manifest.get("proposal_count", -1)),
        "proposal count drift",
    )
    source_root = Path(str(manifest.get("source_root", "") or "")).resolve()
    require(source_root.is_dir(), "proposal source root is unavailable")
    artifacts = list(manifest.get("source_artifacts", []) or [])
    require(len(artifacts) == len(records_by_seed), "proposal source-artifact coverage drift")
    for item in artifacts:
        seed = _integer(item.get("seed"), "source artifact seed")
        require(seed in records_by_seed, f"unexpected source artifact seed {seed}")
        for label in ("event_log", "reasoning_trace", "experiment_snapshot"):
            spec = item.get(label)
            require(isinstance(spec, Mapping), f"seed {seed}: missing {label}")
            path = _resolve_within(source_root, spec.get("path"), f"seed {seed} {label}")
            require(path.is_file(), f"seed {seed}: missing authenticated {label}")
            require(_sha256_file(path) == spec.get("sha256"), f"seed {seed}: {label} hash drift")
    return digest, records_by_seed


def _event_counts(
    bundle: Path,
    arm: str,
    seed: int,
    proposal_records: Mapping[str, Mapping[str, Any]],
) -> Dict[str, int]:
    event_paths = sorted((bundle / arm / f"seed_{seed}" / "event_logs").glob("event_log_*.json"))
    require(len(event_paths) == 1, f"{arm}/{seed}: expected one event log")
    payload = _read_json(event_paths[0])
    events = list(payload.get("events", []) or [])
    require(len(events) == int(payload.get("event_count", -1)), f"{arm}/{seed}: event count drift")
    candidate_ids = []
    issued_ids = []
    rejected = 0
    releases = 0
    for index, raw in enumerate(events):
        require(isinstance(raw, Mapping), f"{arm}/{seed}/{index}: event is not an object")
        event = dict(raw)
        require(str(event.get("factorial_arm", "") or "") == arm, f"{arm}/{seed}/{index}: arm drift")
        if bool(event.get("factorial_candidate_query", False)):
            request_id = str(event.get("factorial_candidate_request_id", "") or "")
            require(request_id in proposal_records, f"{arm}/{seed}/{index}: unknown candidate")
            candidate_ids.append(request_id)
            if bool(event.get("factorial_query_issued", False)):
                issued_ids.append(request_id)
            else:
                reason = str(event.get("factorial_query_rejection_reason", "") or "")
                require(
                    reason in {"query_gate_failed", "fast_only_control"},
                    f"{arm}/{seed}/{index}: rejection provenance drift",
                )
                rejected += 1
        releases += int(bool(event.get("closed_loop_latency_release_event", False)))
    require(len(candidate_ids) == len(set(candidate_ids)), f"{arm}/{seed}: duplicate candidates")
    require(len(issued_ids) == len(set(issued_ids)), f"{arm}/{seed}: duplicate issuance")
    return {
        "candidate_queries": len(candidate_ids),
        "issued_queries": len(issued_ids),
        "query_gate_rejections": rejected,
        "release_events": releases,
        "snapshot_count": int(payload.get("release_snapshot_count", 0) or 0),
    }


def validate_bundle(bundle: Path) -> Dict[str, Any]:
    bundle = Path(bundle).resolve()
    run_manifest = _read_json(bundle / "factorial_run_manifest.json")
    proposal_manifest = _read_json(bundle / "proposal_bank_manifest.json")
    require(run_manifest.get("factorial_design") == "five_arm", "bundle is not a five-arm design")
    require(
        run_manifest.get("factorial_replay_version") == FACTORIAL_REPLAY_VERSION,
        "factorial replay version drift",
    )
    require(run_manifest.get("arms") == [asdict(arm) for arm in FORMAL_FACTORIAL_ARMS], "arm contract drift")
    bank_digest, proposals_by_seed = _validate_proposal_bank(proposal_manifest)
    require(run_manifest.get("proposal_bank_sha256") == bank_digest, "run/proposal hash drift")
    seed_start = _integer(run_manifest.get("seed_start"), "seed start")
    seed_count = _integer(run_manifest.get("seed_count"), "seed count")
    seeds = tuple(range(seed_start, seed_start + seed_count))
    require(set(proposals_by_seed) == set(seeds), "proposal seed cohort drift")
    rows = _read_csv(bundle / "factorial_episode_results.csv")
    require(len(rows) == len(seeds) * len(ARM_NAMES), "five-arm result matrix size drift")
    matrix: Dict[tuple[int, str], Dict[str, Any]] = {}
    for raw in rows:
        seed = _integer(raw.get("seed"), "row seed")
        arm = str(raw.get("arm", "") or "")
        require(seed in seeds and arm in ARM_BY_NAME, f"unexpected cell {seed}/{arm}")
        require((seed, arm) not in matrix, f"duplicate cell {seed}/{arm}")
        spec = ARM_BY_NAME[arm]
        require(_boolean(raw.get("query_gate_enabled"), "query flag") is spec.query_gate_enabled, f"{seed}/{arm}: query flag drift")
        require(_boolean(raw.get("release_guard_enabled"), "release flag") is spec.release_guard_enabled, f"{seed}/{arm}: release flag drift")
        require(raw.get("proposal_bank_sha256") == bank_digest, f"{seed}/{arm}: proposal hash drift")
        row: Dict[str, Any] = {"seed": seed, "arm": arm}
        for metric in METRICS:
            value = _number(raw.get(metric), f"{seed}/{arm} {metric}")
            if metric in INTEGER_METRICS:
                require(value == int(value) and value >= 0, f"{seed}/{arm}: invalid {metric}")
                value = int(value)
            row[metric] = value
        snapshot_count = _integer(raw.get("snapshot_count"), f"{seed}/{arm} snapshot count")
        require(row["candidate_queries"] == row["issued_queries"] + row["query_gate_rejections"], f"{seed}/{arm}: query accounting drift")
        require(row["release_events"] == snapshot_count, f"{seed}/{arm}: release snapshot drift")
        if arm == "fast_only":
            require(row["issued_queries"] == row["release_events"] == snapshot_count == 0, f"{seed}/{arm}: slow path was not disabled")
            require(row["query_gate_rejections"] == row["candidate_queries"], f"{seed}/{arm}: candidate suppression drift")
        observed = _event_counts(bundle, arm, seed, proposals_by_seed[seed])
        for field in observed:
            expected = snapshot_count if field == "snapshot_count" else row[field]
            require(observed[field] == expected, f"{seed}/{arm}: event/{field} drift")
        matrix[(seed, arm)] = row
    expected_cells = {(seed, arm) for seed in seeds for arm in ARM_NAMES}
    require(set(matrix) == expected_cells, "five-arm matrix is incomplete")
    orders = list(run_manifest.get("randomized_block_run_order", []) or [])
    require(len(orders) == len(expected_cells), "randomized order coverage drift")
    for seed in seeds:
        block = [item for item in orders if int(item.get("seed", -1)) == seed]
        require({str(item.get("arm")) for item in block} == set(ARM_NAMES), f"seed {seed}: order arm drift")
        require({int(item.get("order", -1)) for item in block} == set(range(len(ARM_NAMES))), f"seed {seed}: invalid arm order")
    return {
        "bundle": bundle,
        "run_manifest": run_manifest,
        "proposal_manifest": proposal_manifest,
        "bank_sha256": bank_digest,
        "seeds": seeds,
        "matrix": matrix,
    }


def _bootstrap_mean(values: np.ndarray, indices: np.ndarray) -> tuple[float, float, float]:
    point = float(np.mean(values))
    draws = np.mean(values[indices], axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    require(total > 0, "Wilson interval requires a positive denominator")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _mcnemar_exact(left: np.ndarray, right: np.ndarray) -> float:
    discordant_left = int(np.sum((left == 1.0) & (right == 0.0)))
    discordant_right = int(np.sum((left == 0.0) & (right == 1.0)))
    total = discordant_left + discordant_right
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(0, min(discordant_left, discordant_right) + 1))
    return min(1.0, 2.0 * tail / (2.0**total))


def _sign_flip_pvalue(differences: np.ndarray, *, draws: int, seed: int) -> float:
    if np.allclose(differences, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(float(np.mean(differences)))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, differences.size))
    permuted = np.abs(np.mean(signs * differences, axis=1))
    return float((1 + np.sum(permuted >= observed - 1e-15)) / (draws + 1))


def _holm(rows: list[Dict[str, Any]]) -> None:
    eligible = [row for row in rows if row["contrast"] == "full_minus_fast_only"]
    ordered = sorted(eligible, key=lambda row: float(row["p_value_raw"]))
    running = 0.0
    count = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (count - index) * float(row["p_value_raw"]))
        running = max(running, adjusted)
        row["p_value_holm_primary_family"] = running
    for row in rows:
        row.setdefault("p_value_holm_primary_family", "")


def analyze(validated: Mapping[str, Any], *, draws: int, bootstrap_seed: int) -> Dict[str, Any]:
    require(draws > 0, "bootstrap draws must be positive")
    seeds = tuple(validated["seeds"])
    matrix = dict(validated["matrix"])
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, len(seeds), size=(int(draws), len(seeds)))
    arm_rows: list[Dict[str, Any]] = []
    for arm in ARM_NAMES:
        for metric in METRICS:
            values = np.asarray([matrix[(seed, arm)][metric] for seed in seeds], dtype=float)
            point, low, high = _bootstrap_mean(values, indices)
            wilson_low = wilson_high = ""
            if metric in BINARY_METRICS:
                wilson_low, wilson_high = _wilson(int(np.sum(values)), len(values))
            arm_rows.append(
                {
                    "arm": arm,
                    "metric": metric,
                    "mean": point,
                    "ci_low": low,
                    "ci_high": high,
                    "wilson_low": wilson_low,
                    "wilson_high": wilson_high,
                    "n_seeds": len(seeds),
                    "bootstrap_draws": int(draws),
                }
            )
    contrasts: list[Dict[str, Any]] = []
    for contrast_index, (name, (left_arm, right_arm)) in enumerate(CONTRASTS.items()):
        for metric_index, metric in enumerate(METRICS):
            left = np.asarray([matrix[(seed, left_arm)][metric] for seed in seeds], dtype=float)
            right = np.asarray([matrix[(seed, right_arm)][metric] for seed in seeds], dtype=float)
            differences = left - right
            point, low, high = _bootstrap_mean(differences, indices)
            sd = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
            if metric in BINARY_METRICS:
                p_value = _mcnemar_exact(left, right)
                test = "exact_mcnemar"
            else:
                p_value = _sign_flip_pvalue(
                    differences,
                    draws=int(draws),
                    seed=int(bootstrap_seed) + contrast_index * 1009 + metric_index,
                )
                test = "paired_sign_flip"
            contrasts.append(
                {
                    "contrast": name,
                    "left_arm": left_arm,
                    "right_arm": right_arm,
                    "metric": metric,
                    "estimate": point,
                    "ci_low": low,
                    "ci_high": high,
                    "paired_standardized_effect_dz": point / sd if sd > 0.0 else "",
                    "left_wins": int(np.sum(differences > 0.0)),
                    "ties": int(np.sum(differences == 0.0)),
                    "right_wins": int(np.sum(differences < 0.0)),
                    "p_value_raw": p_value,
                    "test": test,
                    "n_seed_pairs": len(seeds),
                }
            )
    _holm(contrasts)
    return {"arm_summaries": arm_rows, "paired_contrasts": contrasts}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260807)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    validated = validate_bundle(args.bundle)
    analysis = analyze(
        validated,
        draws=int(args.draws),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_path = output_dir / "five_arm_arm_summary.csv"
    contrast_path = output_dir / "five_arm_paired_contrasts.csv"
    _write_csv(arm_path, analysis["arm_summaries"])
    _write_csv(contrast_path, analysis["paired_contrasts"])
    audit = {
        "schema": "rgd_five_arm_audit_v1",
        "accepted": True,
        "analysis_version": ANALYSIS_VERSION,
        "bundle": str(Path(args.bundle).resolve()),
        "factorial_replay_version": FACTORIAL_REPLAY_VERSION,
        "proposal_bank_sha256": validated["bank_sha256"],
        "seeds": list(validated["seeds"]),
        "arms": list(ARM_NAMES),
        "matrix_cells": len(validated["matrix"]),
        "bootstrap": {
            "unit": "simulator_seed",
            "draws": int(args.draws),
            "seed": int(args.bootstrap_seed),
        },
        "inference": {
            "binary_pair_test": "exact McNemar",
            "continuous_pair_test": "paired sign-flip randomization",
            "primary_family": "full RGD minus Fast-only across declared metrics",
            "multiplicity": "Holm",
            "proportion_interval": "Wilson 95%",
        },
        "outputs": {
            arm_path.name: _sha256_file(arm_path),
            contrast_path.name: _sha256_file(contrast_path),
        },
    }
    audit_path = output_dir / "five_arm_analysis_manifest.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"accepted": True, "manifest": str(audit_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
