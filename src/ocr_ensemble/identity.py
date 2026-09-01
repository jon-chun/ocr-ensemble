"""Canonical serialization and the two-hash identity scheme.

Every canonical record type computes two different SHA-256 hashes:

- a semantic identifier, over only that record type's declared ``identity_fields``
  preimage; and
- ``record_sha256``, over the complete serialized record except ``record_sha256``
  itself, so it protects timestamps and other audit metadata that never affect
  semantic identity.

Both use RFC 8785/JCS canonical JSON: sorted object keys, no insignificant
whitespace, UTF-8 bytes hashed directly.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any, Mapping, Sequence

CONTENT_ID_PREFIX_SEPARATOR = ":"


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize ``value`` as RFC 8785/JCS-style canonical JSON bytes.

    Object keys are sorted lexicographically and nested structures are handled
    recursively. Sets are not accepted directly: callers must sort a set into a
    tuple before including it in a preimage -- sets are sorted
    lexicographically before hashing. ``Decimal`` (money fields, e.g.
    ``maximum_exposure_usd``) is serialized as its exact string
    form, never as a JSON number -- a float round-trip would silently corrupt
    the semantic ID and the record hash for a dollar amount.
    """
    normalized = _normalize(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def normalize_for_json(value: Any) -> Any:
    """Public entry point for the ``Decimal``/tuple/set normalization rules
    ``canonical_json_bytes`` applies, so storage code that writes plain
    ``json.dumps`` (not the canonical-hashing path) still encodes money and
    other special types consistently -- one rule set, not two.
    """
    return _normalize(value)


def _normalize(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, set):
        raise TypeError(
            "sets must be sorted into a tuple before canonicalization: "
            "sets are sorted lexicographically before hashing"
        )
    return value


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_preimage(preimage: Mapping[str, Any]) -> str:
    """SHA-256 over the canonical JSON encoding of a semantic-identity preimage."""
    return sha256_hex(canonical_json_bytes(preimage))


def content_id(type_name: str, digest_hex: str) -> str:
    """Format a content identifier as ``<type>_sha256:<64-lowercase-hex>``."""
    return f"{type_name}_sha256{CONTENT_ID_PREFIX_SEPARATOR}{digest_hex.lower()}"


def record_sha256(record: Mapping[str, Any], *, exclude: Sequence[str] = ("record_sha256",)) -> str:
    """Full-record hash: every field except ``record_sha256`` itself (and any
    caller-declared extra exclusions, e.g. artifact URIs for types that also
    exclude those from the hash — see each record's own preimage rule)."""
    filtered = {k: v for k, v in record.items() if k not in exclude}
    return sha256_hex(canonical_json_bytes(filtered))
