"""Merge disjoint baseline seed shards into one analyzer bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(row):
    return (
        str(row["group"]),
        int(float(row["transfer_lanes_count"])),
        float(row["transfer_vehicles_density"]),
        int(float(row["seed_idx"])),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", action="append", required=True, help="GROUP|BUNDLE_PATH; select one allocator from each shard")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    provenance = []
    hashes = set()
    seen = set()
    for spec in args.source:
        try:
            group_name, raw_bundle = spec.split("|", 1)
        except ValueError as exc:
            raise RuntimeError(f"invalid --source {spec!r}; expected GROUP|BUNDLE_PATH") from exc
        if group_name not in {"risk_budget", "always_fast"}:
            raise RuntimeError(f"unsupported baseline group: {group_name}")
        bundle = Path(raw_bundle)
        files = sorted(bundle.rglob("padriver_transfer_sweep_rows.csv"))
        if len(files) != 1:
            raise RuntimeError(f"expected one CSV under {bundle}, found {len(files)}")
        source = files[0]
        with source.open("r", encoding="utf-8-sig", newline="") as f:
            shard = list(csv.DictReader(f))
        source_hashes = {str(r.get("source_hash", "")) for r in shard}
        if len(source_hashes) != 1 or not next(iter(source_hashes), ""):
            raise RuntimeError(f"mixed/missing source hash in {source}")
        hashes.update(source_hashes)
        shard = [row for row in shard if str(row.get("group")) == group_name]
        if not shard:
            raise RuntimeError(f"no {group_name} rows in {source}")
        for row in shard:
            k = key(row)
            if k in seen:
                raise RuntimeError(f"duplicate baseline key: {k}")
            seen.add(k)
            rows.append(row)
        provenance.append({"group": group_name, "source": source.as_posix(), "rows": len(shard), "sha256": sha256(source)})

    expected = {
        (group, lanes, density, seed)
        for group in ("risk_budget", "always_fast")
        for lanes in (4, 5, 6)
        for density in (2.0, 3.0)
        for seed in range(30)
    }
    if set(seen) != expected or len(rows) != 360:
        raise RuntimeError(f"expected 360 complete baseline rows, found {len(rows)}; missing={len(expected-set(seen))}")
    if len(hashes) != 1:
        raise RuntimeError(f"mixed runtime source hashes: {sorted(hashes)}")
    rows.sort(key=key)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "padriver_transfer_sweep_rows.csv"
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "baseline_seed_merge_manifest.json").write_text(
        json.dumps({"kind": "baseline_seed_shard_merge_v1", "rows": len(rows), "source_hash": next(iter(hashes)), "provenance": provenance, "output_sha256": sha256(out)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(out), "rows": len(rows), "source_hash": next(iter(hashes))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
