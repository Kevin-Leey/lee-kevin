"""Build a portable, deterministic TVT artifact archive and checksums."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import List, Tuple


MANIFEST_NAME = "MANIFEST.sha256"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def artifact_files(root: Path, *, include_manifest: bool) -> List[Tuple[Path, str]]:
    files: List[Tuple[Path, str]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        if "__pycache__" in parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if not include_manifest and relative == MANIFEST_NAME:
            continue
        files.append((path, relative))
    return sorted(files, key=lambda item: item[1])


def write_manifest(root: Path) -> int:
    lines = [
        f"{sha256_bytes(path.read_bytes())}  {relative}"
        for path, relative in artifact_files(root, include_manifest=False)
    ]
    (root / MANIFEST_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_zip(root: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = artifact_files(root, include_manifest=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in files:
            info = zipfile.ZipInfo(f"{root.name}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path)
    parser.add_argument("--checksum", required=True, type=Path)
    args = parser.parse_args()

    root = args.artifact_root.resolve()
    zip_path = args.zip_path.resolve()
    checksum_path = args.checksum.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    try:
        zip_path.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("ZIP must be outside the artifact root")

    manifest_count = write_manifest(root)
    zip_count = write_zip(root, zip_path)
    digest = sha256_bytes(zip_path.read_bytes())
    checksum_path.write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print(
        f"PASS: wrote {manifest_count} manifest entries, {zip_count} ZIP files, "
        f"and SHA-256 {digest}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
