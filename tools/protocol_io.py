import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import csv
import json
import yaml


def canonical_json_sha256(payload: Any) -> str:
    """Return a stable content hash for a JSON-serializable protocol object."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_inclusive_int_range(value: Any, *, field_name: str) -> Tuple[int, int]:
    """Parse the ``start-end`` notation used by formal protocol seed partitions."""
    text = str(value or "").strip()
    parts = text.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"{field_name} must use inclusive start-end notation")
    try:
        start, end = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{field_name} must contain integer bounds") from exc
    if start > end:
        raise ValueError(f"{field_name} start must not exceed end")
    return start, end


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    import time as _time
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for _attempt in range(20):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            _time.sleep(0.5)
    os.replace(tmp_path, path)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))
