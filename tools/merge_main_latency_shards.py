"""Merge independently executed main-table allocator shards with strict coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


EXPECTED_GROUPS = {
    "rgd_fixed_policy",
    "always_fast",
    "random_budget",
    "uncertainty_budget",
    "risk_budget",
}
EXPECTED_SEEDS = set(range(100, 130))


def parse_source(value: str) -> Tuple[str, Path]:
    try:
        group, raw_path = value.split("|", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--source must be GROUP|BUNDLE_PATH") from exc
    return group, Path(raw_path)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    groups = [group for group, _ in args.source]
    if set(groups) != EXPECTED_GROUPS or len(groups) != len(EXPECTED_GROUPS):
        raise RuntimeError(f"sources must select each main allocator once: {sorted(EXPECTED_GROUPS)}")

    output_rows: List[Dict[str, str]] = []
    provenance = []
    for group, bundle in args.source:
        source_path = bundle / "closed_loop_latency_sweep_rows.csv"
        selected = [row for row in read_rows(source_path) if row.get("group") == group]
        seeds = {int(float(row["seed_idx"])) for row in selected}
        latencies = {float(row.get("closed_loop_latency_extra_s", 0.0) or 0.0) for row in selected}
        if len(selected) != 30 or seeds != EXPECTED_SEEDS or latencies != {1.7}:
            raise RuntimeError(
                f"invalid main shard {group}: rows={len(selected)}, seeds={sorted(seeds)}, latencies={sorted(latencies)}"
            )
        output_rows.extend(selected)
        provenance.append(
            {
                "group": group,
                "source": source_path.as_posix(),
                "source_sha256": sha256(source_path),
                "selected_rows": len(selected),
            }
        )

    keys = {(row["group"], int(float(row["seed_idx"]))) for row in output_rows}
    if len(output_rows) != 150 or len(keys) != 150:
        raise RuntimeError(f"expected 150 unique main rows, found rows={len(output_rows)}, keys={len(keys)}")
    output_rows.sort(key=lambda row: (row["group"], int(float(row["seed_idx"]))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "closed_loop_latency_sweep_rows.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (args.output_dir / "canonical_main_manifest.json").write_text(
        json.dumps(
            {
                "kind": "canonical_main_latency_input_v1",
                "groups": sorted(EXPECTED_GROUPS),
                "seeds": [100, 129],
                "latency_s": 1.7,
                "rows": len(output_rows),
                "selection": provenance,
                "row_file_sha256": sha256(output_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output_path), "rows": len(output_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
