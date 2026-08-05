"""Merge validated RGD-only lane shards into analyzer-ready bundles.

The transfer runner writes one CSV per shard.  This helper selects exactly one
complete 30-seed source for every RGD lane--density cell, preserving raw row
and source-hash provenance.  It deliberately fails closed on overlaps,
missing seeds, mixed source hashes, or non-zero replay delay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(bundle: Path) -> Tuple[Path, List[Dict[str, str]]]:
    paths = sorted(bundle.rglob("padriver_transfer_sweep_rows.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one sweep CSV under {bundle}, found {len(paths)}")
    with paths[0].open("r", encoding="utf-8-sig", newline="") as f:
        return paths[0], list(csv.DictReader(f))


def row_key(row: Dict[str, str]) -> Tuple[int, float, int]:
    return (
        int(float(row["transfer_lanes_count"])),
        float(row["transfer_vehicles_density"]),
        int(float(row["seed_idx"])),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane2", type=Path, action="append", required=True, help="bundle containing density-2 rows; repeat for lanes 4/5/6")
    parser.add_argument("--lane3", type=Path, action="append", required=True, help="bundle containing density-3 rows; repeat for lanes 4/5/6")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected: List[Dict[str, str]] = []
    provenance = []
    hashes = set()
    seen = set()
    for bundle, density in [(path, 2.0) for path in args.lane2] + [(path, 3.0) for path in args.lane3]:
        source, rows = read_rows(bundle)
        source_hashes = {str(row.get("source_hash", "")) for row in rows}
        if len(source_hashes) != 1 or not next(iter(source_hashes), ""):
            raise RuntimeError(f"mixed or missing source hash in {source}")
        hashes.update(source_hashes)
        chosen = [
            row for row in rows
            if str(row.get("group")) == "rgd_fixed_policy"
            and abs(float(row.get("transfer_vehicles_density", "nan")) - density) < 1e-9
        ]
        expected_lanes = {4, 5, 6}
        if len(chosen) != 30:
            raise RuntimeError(f"expected 30 RGD rows at density {density:g} per lane shard, found {len(chosen)} in {source}")
        lane_ids = {int(float(row["transfer_lanes_count"])) for row in chosen}
        if len(lane_ids) != 1 or not lane_ids.issubset(expected_lanes):
            raise RuntimeError(f"source {source} must contain exactly one of lanes 4/5/6, found {sorted(lane_ids)}")
        for row in chosen:
            key = row_key(row)
            if key in seen:
                raise RuntimeError(f"duplicate RGD key: {key}")
            seen.add(key)
            selected.append(row)
        provenance.append({
            "density": density,
            "source": source.as_posix(),
            "source_sha256": sha256(source),
            "rows": len(chosen),
        })

    expected = {(lanes, density, seed) for lanes in (4, 5, 6) for density in (2.0, 3.0) for seed in range(30)}
    if {row_key(row) for row in selected} != expected:
        raise RuntimeError("selected rows do not form the complete 180-run RGD factorial")
    if len(hashes) != 1:
        raise RuntimeError(f"mixed runtime source hashes: {sorted(hashes)}")

    selected.sort(key=row_key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "padriver_transfer_sweep_rows.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    (args.output_dir / "rgd_targeted_merge_manifest.json").write_text(
        json.dumps({
            "kind": "rgd_targeted_lane_density_merge_v1",
            "rows": len(selected),
            "source_hash": next(iter(hashes)),
            "provenance": provenance,
            "output_sha256": sha256(output),
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "rows": len(selected), "source_hash": next(iter(hashes))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
