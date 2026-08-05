"""Build one auditable canonical input from non-overlapping transfer shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


EXPECTED_GROUPS = ("rgd_fixed_policy", "risk_budget", "always_fast")
EXPECTED_LANES = (4, 5, 6)
EXPECTED_DENSITIES = (2.0, 3.0)
EXPECTED_SEEDS = set(range(30))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_source(value: str) -> Tuple[str, int, float, Path]:
    try:
        group, lanes, density, raw_path = value.split("|", 3)
        return group, int(lanes), float(density), Path(raw_path)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--source must be GROUP|LANES|DENSITY|BUNDLE_PATH"
        ) from exc


def key(row: Dict[str, str]) -> Tuple[str, int, float, int]:
    return (
        str(row["group"]),
        int(float(row["transfer_lanes_count"])),
        float(row["transfer_vehicles_density"]),
        int(float(row["seed_idx"])),
    )


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty canonical input")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source,
        help="GROUP|LANES|DENSITY|BUNDLE_PATH; repeat for each canonical setting.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    expected = {
        (group, lanes, density)
        for group in EXPECTED_GROUPS
        for lanes in EXPECTED_LANES
        for density in EXPECTED_DENSITIES
    }
    selected_settings = {(group, lanes, density) for group, lanes, density, _ in args.source}
    if selected_settings != expected or len(args.source) != len(expected):
        missing = sorted(expected - selected_settings)
        duplicate = len(args.source) - len(selected_settings)
        raise RuntimeError(f"sources must cover each setting once; missing={missing}, duplicate_specs={duplicate}")

    canonical: List[Dict[str, str]] = []
    provenance: List[Dict[str, Any]] = []
    seen_keys = set()
    for group, lanes, density, bundle in sorted(args.source):
        source_path = bundle / "padriver_transfer_sweep_rows.csv"
        rows = [
            row
            for row in read_rows(source_path)
            if key(row)[:3] == (group, lanes, density)
        ]
        seeds = {key(row)[3] for row in rows}
        if seeds != EXPECTED_SEEDS or len(rows) != 30:
            raise RuntimeError(
                f"{source_path} does not provide exactly seeds 0--29 for {(group, lanes, density)}: {sorted(seeds)}"
            )
        for row in rows:
            row_key = key(row)
            if row_key in seen_keys:
                raise RuntimeError(f"duplicate canonical run key: {row_key}")
            seen_keys.add(row_key)
            canonical.append(row)
        provenance.append(
            {
                "group": group,
                "lanes_count": lanes,
                "vehicles_density": density,
                "source": source_path.as_posix(),
                "source_sha256": sha256(source_path),
                "selected_rows": len(rows),
            }
        )

    if len(canonical) != 540 or len(seen_keys) != 540:
        raise RuntimeError(f"expected 540 canonical rows, found {len(canonical)}")
    canonical.sort(key=key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "padriver_transfer_sweep_rows.csv"
    write_csv(output, canonical)
    (args.output_dir / "canonical_transfer_manifest.json").write_text(
        json.dumps(
            {
                "kind": "canonical_lane_density_transfer_input_v1",
                "row_count": len(canonical),
                "selection": provenance,
                "row_file_sha256": sha256(output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(canonical)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
