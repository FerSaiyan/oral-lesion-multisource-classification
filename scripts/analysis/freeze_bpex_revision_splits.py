#!/usr/bin/env python3
"""Copy the audited BPEX v2 manifests into a versioned, hash-checked snapshot.

This never reproduces upstream DVC stages or changes the legacy manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from audit_bpex_revision_splits import audit


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/bpex_revision_v2"))
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()

    if args.snapshot.exists() or args.record.exists():
        raise FileExistsError("Freeze destination exists; use a new version rather than overwrite it")
    source = args.source.resolve()
    report = audit(source)
    if not report["checks"]["all_passed"]:
        raise RuntimeError("Split audit failed; refusing to freeze")
    expected = report["csv_sha256"]
    actual = {p.name: digest(p) for p in source.glob("*.csv")}
    if actual != expected:
        raise RuntimeError("Source hashes do not match the split audit")

    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, args.snapshot)
    copied = {p.name: digest(p) for p in args.snapshot.glob("*.csv")}
    if copied != expected:
        shutil.rmtree(args.snapshot)
        raise RuntimeError("Snapshot hashes differ from the source")

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "snapshot": str(args.snapshot),
        "clinical_adjudication": "not performed as of this freeze",
        "checks": report["checks"],
        "csv_sha256": expected,
        "partition_rows": {
            name: {part: detail["rows"] for part, detail in variant["splits"].items()}
            for name, variant in report["variants"].items()
        },
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Frozen {len(expected)} checked CSVs at {args.snapshot}; record: {args.record}")


if __name__ == "__main__":
    main()
