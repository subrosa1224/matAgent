"""Stable request fingerprints (M2)."""

import hashlib

import orjson

from materials_screening.models import ScreeningRequest


def request_fingerprint(request: ScreeningRequest) -> str:
    """Return a stable SHA-256 fingerprint of the canonical request JSON."""
    canonical = orjson.dumps(
        request.model_dump(mode="json", exclude_none=True),
        option=orjson.OPT_SORT_KEYS,
    )
    return hashlib.sha256(canonical).hexdigest()
