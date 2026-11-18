"""Canonical JSON bytes and SHA-256 digests for artifact metadata."""

from __future__ import annotations

import hashlib
import json


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
