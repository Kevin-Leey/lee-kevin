"""Analyze latency-prediction error in the paired query/release factorial.

The simulator seed is the independent unit. Request-level rows are retained for
auditability, but every confidence interval resamples complete seed blocks.
Scheduled latency is read from the proposal bank, never from the gate's fixed
prediction field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


ARMS = ("full", "query_only", "release_only", "neither")
RATE_METRICS = {
    "issue_rate": ("issued", "candidates"),
    "release_rate_given_issue": ("released", "issued"),
    "timeout_rate_given_issue": ("timeouts", "issued"),
    "distinct_selection_rate_given_release": ("distinct_selections", "released"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Sequence[float], q: float) -> float:
    require(bool(values), "cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def flatten_bank(root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    manifest_path = root / "proposal_bank_manifest.json"
    manifest = read_json(manifest_path)
    rows: List[Dict[str, Any]] = []
    request_ids = set()
    for seed_block in manifest.get("bank_payload", []):
        seed = int(seed_block["seed"])
        for raw in seed_block.get("records", []):
            row = dict(raw)
            row["seed"] = seed
            row["source_frame"] = int(row["source_frame"])
            row["latency_steps"] = int(row["latency_steps"])
            request_id = str(row["request_id"])
            require(request_id not in request_ids, f"duplicate request id: {request_id}")
            request_ids.add(request_id)
            rows.append(row)
    require(len(rows) == int(manifest["proposal_count"]), "proposal count mismatch")
    require(len({row["seed"] for row in rows}) == int(manifest["seed_count"]), "seed count mismatch")
    return manifest, sorted(rows, key=lambda row: (row["seed"], row["source_frame"]))


def reasoning_path(root: Path, arm: str, seed: int) -> Path:
    paths = sorted((root / arm / f"seed_{seed}").glob("ep_*/*_reasoning_records.json"))
    require(len(paths) == 1, f"expected one reasoning trace for {arm} seed {seed}")
    return paths[0]


def extract_lifecycle_events(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = read_json(path)
    records = payload.get("analysis_records", [])
    events: Dict[str, Dict[str, Any]] = {}
    for record in records:
        lifecycle = (
            record.get("rgd_subordinate_diagnostics", {}).get("release_lifecycle", {})
        )
        request_id = str(
            lifecycle.get("closed_loop_latency_request_id")
            or lifecycle.get("closed_loop_latency_issued_request_id")
            or lifecycle.get("closed_loop_latency_terminal_request_id")
            or ""
        )
        if not request_id:
            continue
        event = events.setdefault(
            request_id,
            {
                "issued": False,
                "released": False,
                "timeout": False,
                "failure": False,
                "terminal": False,
                "release_selection_distinct": False,
                "release_actuation_distinct": False,
            },
        )
        frame_id = int(record.get("frame_id", -1))
        if lifecycle.get("closed_loop_latency_issuance_event") is True:
            event.update(
                {
                    "issued": True,
                    "issue_frame": frame_id,
                    "source_frame": int(lifecycle.get("closed_loop_latency_source_frame", frame_id)),
                    "scheduled_release_frame": lifecycle.get(
                        "closed_loop_latency_scheduled_release_frame"
                    ),
                    "issued_response_outcome": str(
                        lifecycle.get("closed_loop_latency_issued_response_outcome") or ""
                    ),
                }
            )
        if lifecycle.get("closed_loop_latency_release_event") is True:
            event.update(
                {
                    "released": True,
                    "release_frame": frame_id,
                    "realized_steps": lifecycle.get("closed_loop_latency_realized_steps"),
                    "release_selected_action": lifecycle.get("release_selected_action"),
                    "release_fast_action": lifecycle.get("release_fast_comparator_action"),
                    "release_selection_distinct": bool(
                        lifecycle.get("release_selection_distinct")
                    ),
                    "release_actuation_distinct": bool(
                        lifecycle.get("closed_loop_release_actuation_distinct")
                    ),
                    "release_alignment_pass": bool(
                        lifecycle.get("closed_loop_release_action_alignment_pass")
                    ),
                }
            )
        if lifecycle.get("closed_loop_latency_timeout_event") is True:
            event["timeout"] = True
        if lifecycle.get("closed_loop_latency_failure_event") is True:
            event["failure"] = True
        if lifecycle.get("closed_loop_latency_terminal_event") is True:
            event["terminal"] = True
            event["terminal_outcome"] = str(
                lifecycle.get("closed_loop_latency_terminal_outcome") or ""
            )
    return events


def predicted_steps(root: Path, arm: str, seed: int) -> int:
    manifest = read_json(root / arm / f"seed_{seed}" / "runtime_manifest.json")
    replay = manifest.get("config", {}).get("closed_loop_latency_replay", {})
    steps = int(replay.get("delay_steps", -1))
    require(steps >= 0, f"missing predicted delay steps for {arm} seed {seed}")
    return steps


def load_episode_rows(root: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with (root / "factorial_episode_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            key = (str(row["arm"]), int(row["seed"]))
            require(key not in rows, f"duplicate factorial episode row: {key}")
            rows[key] = dict(row)
    return rows


def classify_status(bank_outcome: str, event: Mapping[str, Any]) -> str:
    if not event.get("issued"):
        return "not_issued"
    if event.get("released"):
        return "released"
    if event.get("timeout") or bank_outcome == "timeout":
        return "timeout"
    if event.get("failure") or bank_outcome == "failure":
        return "failure"
    return "pending"


def error_direction(error_steps: int) -> str:
    if error_steps < 0:
        return "earlier_than_predicted"
    if error_steps > 0:
        return "later_than_predicted"
    return "matched_prediction"


def error_magnitude_bin(error_seconds: float) -> str:
    value = abs(error_seconds)
    if value <= 0.5:
        return "abs_error_le_0.5s"
    if value <= 1.0:
        return "abs_error_0.5_to_1.0s"
    return "abs_error_gt_1.0s"


def build_request_rows(
    root: Path,
    bank_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    seeds = sorted({int(row["seed"]) for row in bank_rows})
    event_cache: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = {}
    prediction_cache: Dict[Tuple[str, int], int] = {}
    for arm in ARMS:
        for seed in seeds:
            event_cache[(arm, seed)] = extract_lifecycle_events(
                reasoning_path(root, arm, seed)
            )
            prediction_cache[(arm, seed)] = predicted_steps(root, arm, seed)

    rows: List[Dict[str, Any]] = []
    seed_summary: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for arm in ARMS:
        for seed in seeds:
            candidates = [row for row in bank_rows if int(row["seed"]) == seed]
            predictions = {prediction_cache[(arm, seed)]}
            require(len(predictions) == 1, f"prediction drift for {arm} seed {seed}")
            prediction = predictions.pop()
            signed_errors = [int(row["latency_steps"]) - prediction for row in candidates]
            seed_summary[(arm, seed)] = {
                "arm": arm,
                "seed": seed,
                "candidate_count": len(candidates),
                "predicted_latency_steps": prediction,
                "predicted_latency_s": prediction / 10.0,
                "mean_signed_error_s": sum(signed_errors) / (10.0 * len(signed_errors)),
                "mean_absolute_error_s": sum(abs(value) for value in signed_errors)
                / (10.0 * len(signed_errors)),
                "earlier_fraction": sum(value < 0 for value in signed_errors)
                / len(signed_errors),
                "later_fraction": sum(value > 0 for value in signed_errors)
                / len(signed_errors),
            }
            events = event_cache[(arm, seed)]
            for candidate in candidates:
                request_id = str(candidate["request_id"])
                event = events.get(request_id, {})
                steps = int(candidate["latency_steps"])
                error_steps = steps - prediction
                status = classify_status(str(candidate["outcome"]), event)
                row = {
                    "arm": arm,
                    "seed": seed,
                    "request_id": request_id,
                    "source_frame": int(candidate["source_frame"]),
                    "predicted_latency_steps": prediction,
                    "predicted_latency_s": prediction / 10.0,
                    "scheduled_latency_steps": steps,
                    "scheduled_latency_s": steps / 10.0,
                    "signed_error_steps": error_steps,
                    "signed_error_s": error_steps / 10.0,
                    "absolute_error_s": abs(error_steps) / 10.0,
                    "error_direction": error_direction(error_steps),
                    "error_magnitude_bin": error_magnitude_bin(error_steps / 10.0),
                    "bank_outcome": str(candidate["outcome"]),
                    "raw_slow_action": int(candidate["raw_slow_action"]),
                    "issued": int(bool(event.get("issued"))),
                    "terminal_status": status,
                    "released": int(status == "released"),
                    "timeout": int(status == "timeout"),
                    "failure": int(status == "failure"),
                    "pending": int(status == "pending"),
                    "release_selection_distinct": int(
                        bool(event.get("release_selection_distinct"))
                    ),
                    "release_actuation_distinct": int(
                        bool(event.get("release_actuation_distinct"))
                    ),
                    "release_alignment_pass": int(
                        bool(event.get("release_alignment_pass"))
                    ),
                    "realized_steps": event.get("realized_steps", ""),
                }
                rows.append(row)
    return rows, seed_summary


def aggregate_seed_strata(
    rows: Sequence[Mapping[str, Any]],
    stratum_field: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, Any], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row["arm"]), int(row["seed"]), row[stratum_field])
        item = grouped.setdefault(
            key,
            {
                "arm": key[0],
                "seed": key[1],
                stratum_field: key[2],
                "candidates": 0,
                "issued": 0,
                "released": 0,
                "timeouts": 0,
                "failures": 0,
                "pending": 0,
                "distinct_selections": 0,
                "distinct_actuations": 0,
            },
        )
        item["candidates"] += 1
        item["issued"] += int(row["issued"])
        item["released"] += int(row["released"])
        item["timeouts"] += int(row["timeout"])
        item["failures"] += int(row["failure"])
        item["pending"] += int(row["pending"])
        item["distinct_selections"] += int(row["release_selection_distinct"])
        item["distinct_actuations"] += int(row["release_actuation_distinct"])
    return sorted(grouped.values(), key=lambda row: (row["arm"], row[stratum_field], row["seed"]))


def bootstrap_ratio(
    seed_rows: Sequence[Mapping[str, Any]],
    numerator: str,
    denominator: str,
    *,
    draws: int,
    rng: random.Random,
) -> Tuple[float, float]:
    by_seed = list(seed_rows)
    require(bool(by_seed), "empty seed block")
    samples: List[float] = []
    for _ in range(draws):
        chosen = [by_seed[rng.randrange(len(by_seed))] for _ in by_seed]
        den = sum(float(row[denominator]) for row in chosen)
        if den <= 0:
            continue
        num = sum(float(row[numerator]) for row in chosen)
        samples.append(num / den)
    require(bool(samples), f"zero bootstrap denominator for {numerator}/{denominator}")
    return quantile(samples, 0.025), quantile(samples, 0.975)


def summarize_strata(
    seed_rows: Sequence[Mapping[str, Any]],
    stratum_field: str,
    *,
    draws: int,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, Any], List[Mapping[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["arm"]), row[stratum_field])].append(row)
    output: List[Dict[str, Any]] = []
    for group_index, ((arm, stratum), rows) in enumerate(sorted(grouped.items())):
        item: Dict[str, Any] = {
            "arm": arm,
            stratum_field: stratum,
            "n_seeds_with_candidates": len(rows),
        }
        for field in (
            "candidates",
            "issued",
            "released",
            "timeouts",
            "failures",
            "pending",
            "distinct_selections",
            "distinct_actuations",
        ):
            item[field] = int(sum(int(row[field]) for row in rows))
        for metric_index, (name, (numerator, denominator)) in enumerate(
            RATE_METRICS.items()
        ):
            den = float(item[denominator])
            item[name] = float(item[numerator]) / den if den > 0 else ""
            if den > 0:
                low, high = bootstrap_ratio(
                    rows,
                    numerator,
                    denominator,
                    draws=draws,
                    rng=random.Random(
                        bootstrap_seed + group_index * 101 + metric_index
                    ),
                )
                item[f"{name}_ci_low"] = low
                item[f"{name}_ci_high"] = high
            else:
                item[f"{name}_ci_low"] = ""
                item[f"{name}_ci_high"] = ""
        output.append(item)
    return output


def compare_banks(
    frozen_rows: Sequence[Mapping[str, Any]],
    stress_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    frozen = {str(row["request_id"]): row for row in frozen_rows}
    stress = {str(row["request_id"]): row for row in stress_rows}
    require(set(frozen) == set(stress), "frozen and stress request ids differ")
    rows: List[Dict[str, Any]] = []
    for request_id in sorted(frozen):
        left = frozen[request_id]
        right = stress[request_id]
        same_proposal = all(
            left[field] == right[field]
            for field in (
                "seed",
                "source_frame",
                "raw_slow_action",
                "response_text",
                "response_sha256",
            )
        )
        rows.append(
            {
                "request_id": request_id,
                "seed": int(left["seed"]),
                "source_frame": int(left["source_frame"]),
                "same_proposal_payload": int(same_proposal),
                "frozen_latency_steps": int(left["latency_steps"]),
                "stress_latency_steps": int(right["latency_steps"]),
                "frozen_outcome": str(left["outcome"]),
                "stress_outcome": str(right["outcome"]),
            }
        )
    require(all(row["same_proposal_payload"] == 1 for row in rows), "proposal payload drift")
    return rows


def merge_seed_outcomes(
    summaries: Mapping[Tuple[str, int], Dict[str, Any]],
    episode_rows: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for key, summary in sorted(summaries.items()):
        require(key in episode_rows, f"missing episode row: {key}")
        episode = episode_rows[key]
        item = dict(summary)
        for field in (
            "issued_queries",
            "release_events",
            "primitive_distinct_selections",
            "collision",
            "success_rate",
            "route_completion",
            "episode_reward",
            "driving_distance",
            "avg_speed",
        ):
            item[field] = float(episode[field])
        output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stress-root",
        default="results/rgd_factorial_confirmatory_20260731/stress_v5",
    )
    parser.add_argument(
        "--frozen-root",
        default="results/rgd_factorial_confirmatory_20260731/frozen_v5",
    )
    parser.add_argument(
        "--output-dir",
        default="results/rgd_factorial_confirmatory_20260731/latency_error_analysis",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260801)
    args = parser.parse_args()

    stress_root = Path(args.stress_root).resolve()
    frozen_root = Path(args.frozen_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stress_manifest, stress_bank = flatten_bank(stress_root)
    frozen_manifest, frozen_bank = flatten_bank(frozen_root)
    require(stress_manifest.get("latency_profile") == "stress", "wrong stress profile")
    require(frozen_manifest.get("latency_profile") == "frozen", "wrong frozen profile")

    bank_comparison = compare_banks(frozen_bank, stress_bank)
    request_rows, seed_exposure = build_request_rows(stress_root, stress_bank)
    episode_rows = load_episode_rows(stress_root)
    seed_outcomes = merge_seed_outcomes(seed_exposure, episode_rows)

    latency_seed_rows = aggregate_seed_strata(request_rows, "scheduled_latency_s")
    latency_summary = summarize_strata(
        latency_seed_rows,
        "scheduled_latency_s",
        draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    direction_seed_rows = aggregate_seed_strata(request_rows, "error_direction")
    direction_summary = summarize_strata(
        direction_seed_rows,
        "error_direction",
        draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed + 10000,
    )
    magnitude_seed_rows = aggregate_seed_strata(request_rows, "error_magnitude_bin")
    magnitude_summary = summarize_strata(
        magnitude_seed_rows,
        "error_magnitude_bin",
        draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed + 20000,
    )

    write_csv(output_dir / "proposal_bank_comparison.csv", bank_comparison)
    write_csv(output_dir / "stress_request_lifecycle.csv", request_rows)
    write_csv(output_dir / "stress_latency_seed_strata.csv", latency_seed_rows)
    write_csv(output_dir / "stress_latency_stratified_summary.csv", latency_summary)
    write_csv(output_dir / "stress_error_direction_seed_strata.csv", direction_seed_rows)
    write_csv(output_dir / "stress_error_direction_summary.csv", direction_summary)
    write_csv(output_dir / "stress_error_magnitude_seed_strata.csv", magnitude_seed_rows)
    write_csv(output_dir / "stress_error_magnitude_summary.csv", magnitude_summary)
    write_csv(output_dir / "stress_seed_exposure_outcomes.csv", seed_outcomes)

    latency_counts = defaultdict(int)
    outcome_counts = defaultdict(int)
    for row in stress_bank:
        latency_counts[int(row["latency_steps"])] += 1
        outcome_counts[str(row["outcome"])] += 1
    predictions = {
        int(row["predicted_latency_steps"])
        for row in seed_outcomes
    }
    require(len(predictions) == 1, "gate latency prediction is not constant")
    prediction = predictions.pop()
    manifest = {
        "schema": "rgd_factorial_latency_error_analysis_v1",
        "accepted": True,
        "independent_unit": "simulator_seed",
        "request_rows_are_technical_observations": True,
        "bootstrap": {
            "method": "percentile cluster bootstrap",
            "cluster": "simulator_seed",
            "draws": args.bootstrap_draws,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
        },
        "prediction": {
            "steps": prediction,
            "seconds": prediction / 10.0,
            "policy_frequency_hz": 10,
        },
        "stress_schedule_counts": {
            str(steps): count for steps, count in sorted(latency_counts.items())
        },
        "stress_outcome_counts": dict(sorted(outcome_counts.items())),
        "proposal_count": len(stress_bank),
        "seed_count": len({int(row["seed"]) for row in stress_bank}),
        "frozen_bank_sha256": str(frozen_manifest["bank_sha256"]),
        "stress_bank_sha256": str(stress_manifest["bank_sha256"]),
        "proposal_payloads_identical": True,
        "input_sha256": {
            "stress_proposal_bank_manifest.json": sha256(
                stress_root / "proposal_bank_manifest.json"
            ),
            "stress_factorial_episode_results.csv": sha256(
                stress_root / "factorial_episode_results.csv"
            ),
            "frozen_proposal_bank_manifest.json": sha256(
                frozen_root / "proposal_bank_manifest.json"
            ),
        },
    }
    write_json(output_dir / "latency_error_analysis_manifest.json", manifest)
    print(f"Wrote latency-error analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
