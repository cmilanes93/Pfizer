"""Verify original input files without performing data analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--manifest", type=Path, default=root / "data/input_manifest.json")
    args = parser.parse_args()
    if not args.manifest.is_file():
        print(f"Input manifest missing: {args.manifest}. Restore the original input files and manifest.")
        return 1
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        entries = manifest["files"]
        if manifest.get("schema_version") != 1 or len(entries) != 5:
            raise ValueError("Expected schema version 1 and five original files.")
        seen = set()
        failures = []
        for entry in entries:
            relative = entry["path"]
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or relative in seen:
                raise ValueError("Invalid or duplicate input path.")
            seen.add(relative)
            if not path.is_file():
                failures.append(f"Missing: {relative}")
                continue
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if path.stat().st_size != entry["size_bytes"] or digest != entry["sha256"]:
                failures.append(f"Changed: {relative}")
        if failures:
            print("\n".join(failures))
            return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Input verification failed: {exc}")
        return 1
    print(f"Verified {len(entries)} original files: sizes and SHA-256 hashes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
