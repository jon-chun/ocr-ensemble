"""Write-temp/validate/fsync/atomic-rename discipline.

A stage writes to a temporary path, validates the full artifact, fsyncs it,
and atomically renames it to its final path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from ocr_ensemble.identity import normalize_for_json

T = TypeVar("T")


def write_jsonl_atomic(
    path: Path,
    records: Iterable[dict],
    *,
    validate: Callable[[dict], None] | None = None,
) -> None:
    """Write ``records`` as JSON Lines to ``path`` using the sealed-store
    discipline: write to a sibling temp file, validate every line, fsync,
    then atomically rename over the final path.

    Each record is passed through ``normalize_for_json`` first, so a
    ``Decimal`` money field serializes as its exact string form (matching
    ``canonical_json_bytes``'s hashing rule) instead of raising or silently
    losing precision through a JSON number. Reading a record back yields that
    string, not a ``Decimal``; callers that need the typed value re-hydrate
    it explicitly (storage stays schema-agnostic on purpose).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for record in records:
            if validate is not None:
                validate(record)
            f.write(json.dumps(normalize_for_json(record), sort_keys=True, ensure_ascii=False))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json_atomic(path: Path, record: dict) -> None:
    """Same write-temp/validate/fsync/atomic-rename discipline as
    ``write_jsonl_atomic``, for a single-document store (e.g.
    ``run_manifest.json``, ``effective_gt_snapshot.json``) rather than a
    JSON-Lines store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(normalize_for_json(record), sort_keys=True, ensure_ascii=False, indent=2))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl_atomic(
    path: Path,
    record: dict,
    *,
    validate: Callable[[dict], None] | None = None,
) -> None:
    """Append one record to an append-only journal (single
    writer, fsync each event before acknowledging it). Reuses the sealed-store
    write-temp/validate/fsync/atomic-rename discipline rather than opening
    ``path`` in append mode, so a crash mid-write can never leave a partially
    written final line for a reader to trip over.
    """
    existing = read_jsonl(path)
    write_jsonl_atomic(path, existing + [record], validate=validate)
