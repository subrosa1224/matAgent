"""Evidence validation for experiment matrices and author claims."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from .models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from .rag import ChunkRecord, VectorStore


def validate_matrix_evidence(
    store: VectorStore,
    *,
    document_id: str,
    groups: Sequence[ExperimentalGroup],
    measurements: Sequence[ExperimentalMeasurement],
) -> None:
    group_by_id = {group.group_id: group for group in groups}
    if len(group_by_id) != len(groups):
        raise ValueError("duplicate experimental group id")
    chunk_ids = {group.chunk_id for group in groups}
    chunk_ids.update(item.chunk_id for item in measurements)
    chunks = {chunk.chunk_id: chunk for chunk in store.get_chunks(tuple(chunk_ids))}
    for group in groups:
        chunk = chunks.get(group.chunk_id)
        _validate_provenance(
            document_id,
            group.source_quote,
            group.source_text_sha256,
            group.page_from,
            group.page_to,
            chunk,
        )
        for evidence_value in (
            *group.variables.values(),
            *group.conditions.values(),
        ):
            if _normalize(evidence_value) not in _normalize(group.source_quote):
                raise ValueError(
                    "group variable or condition is absent from evidence: "
                    f"{group.group_id}"
                )
    for item in measurements:
        parent_group = group_by_id.get(item.group_id)
        if parent_group is None:
            raise ValueError(f"measurement references unknown group: {item.group_id}")
        if item.document_id != document_id or parent_group.document_id != document_id:
            raise ValueError("matrix item belongs to a different document")
        chunk = chunks.get(item.chunk_id)
        _validate_provenance(
            document_id,
            item.source_quote,
            item.source_text_sha256,
            item.page_from,
            item.page_to,
            chunk,
        )
        if not measurement_value_present(item.value_text, item.unit, item.source_quote):
            raise ValueError(
                "measurement value is absent from evidence: "
                f"{item.measurement_id} ({item.metric}={item.value_text})"
            )
        if item.unit and not unit_present_in_evidence(item.unit, item.source_quote):
            raise ValueError(
                f"measurement unit is absent from evidence: {item.measurement_id}"
            )
        for optional_value in (item.uncertainty_text,):
            if optional_value and _normalize(optional_value) not in _normalize(
                item.source_quote
            ):
                raise ValueError(
                    f"measurement unit or uncertainty is absent from evidence: "
                    f"{item.measurement_id}"
                )


def measurement_value_present(value: str, unit: str | None, evidence: str) -> bool:
    """A contiguous value/unit span; only declared unit typography may differ."""
    literal = (
        re.search(
            rf"(?<![a-z0-9])(?<!\d\.){re.escape(_normalize(value))}(?![a-z0-9]|\.\d)",
            _normalize(evidence),
        )
        is not None
    )
    if not unit:
        return literal

    def typography(text):
        return (
            unicodedata.normalize("NFKC", text)
            .replace("−", "-")
            .replace("–", "-")
            .replace("^", "")
        )

    match = re.fullmatch(
        r"\s*([+-]?\d+(?:\.\d+)?)\s*([A-Za-zµμΩ°%].*?)\s*", typography(value)
    )
    if match is None:
        return literal
    if re.sub(r"\s+", "", match.group(2)) != re.sub(r"\s+", "", typography(unit)):
        return False
    tokens = re.findall(r"[A-Za-zµμΩ°%]+|\d+|[^\s]", typography(unit).strip())
    if not tokens:
        return False
    pattern = re.escape(tokens[0])
    for previous, current in zip(tokens, tokens[1:], strict=False):
        pattern += (
            r"\s+" if previous.isalpha() and current.isalpha() else r"\s*"
        ) + re.escape(current)
    return (
        re.search(
            r"(?<![A-Za-z0-9µμΩ.+-])"
            + re.escape(match.group(1))
            + r"\s*"
            + pattern
            + r"(?![A-Za-z0-9µμΩ]|\.\d)",
            typography(evidence),
        )
        is not None
    )


def unit_present_in_evidence(unit: str, evidence: str) -> bool:
    """Match unit typography, never scale, case, or a different exponent.

    PDF extraction commonly separates an exponent from its unit. Normalize
    Unicode superscripts and minus glyphs for this check only; quotes, values,
    and their provenance remain untouched. Keep alphabetic prefixes atomic so
    ms cannot match m s, or AW-1 match mAW-1.
    """

    def typography(value: str) -> str:
        return (
            unicodedata.normalize("NFKC", value)
            .replace("−", "-")
            .replace("–", "-")
            .replace("^", "")
        )

    tokens = re.findall(r"[A-Za-zµμΩ°%]+|\d+|[^\s]", typography(unit).strip())
    if not tokens:
        return False
    # Only a standalone percentage may immediately follow a numeric token.
    # Keep compound units (wt%, mol%, etc.) and alphabetic prefixes unchanged.
    if tokens == ["%"] and re.search(
        r"(?<![A-Za-z0-9µμΩ.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?%(?![A-Za-z0-9µμΩ])",
        typography(evidence),
    ):
        return True
    pattern = re.escape(tokens[0])
    for previous, current in zip(tokens, tokens[1:], strict=False):
        separator = r"\s+" if previous.isalpha() and current.isalpha() else r"\s*"
        pattern += separator + re.escape(current)
    return (
        re.search(
            r"(?<![A-Za-z0-9µμΩ])" + pattern + r"(?![A-Za-z0-9µμΩ])",
            typography(evidence),
        )
        is not None
    )


def validate_claim_evidence(store: VectorStore, claim: PaperClaim) -> None:
    chunks = store.get_chunks((claim.chunk_id,))
    chunk = chunks[0] if chunks else None
    _validate_provenance(
        claim.document_id,
        claim.source_quote,
        claim.source_text_sha256,
        claim.page_from,
        claim.page_to,
        chunk,
    )
    if _normalize(claim.claim_text) not in _normalize(claim.source_quote):
        raise ValueError("claim_text must be an exact span of source_quote")


def _validate_quote(
    document_id: str, source_quote: str, chunk: ChunkRecord | None
) -> None:
    if chunk is None:
        raise ValueError("evidence chunk does not exist")
    if chunk.document_id != document_id:
        raise ValueError("evidence chunk belongs to a different document")
    if _normalize(source_quote) not in _normalize(chunk.text):
        raise ValueError("source_quote is not present in the evidence chunk")


def _validate_provenance(
    document_id: str,
    source_quote: str,
    source_text_sha256: str,
    page_from: int,
    page_to: int,
    chunk: ChunkRecord | None,
) -> None:
    _validate_quote(document_id, source_quote, chunk)
    assert chunk is not None
    if source_text_sha256 != chunk.text_sha256:
        raise ValueError("evidence hash does not match the source chunk")
    if page_from != chunk.page_from or page_to != chunk.page_to:
        raise ValueError("evidence pages do not match the source chunk")


def _normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("×", "x")
    return re.sub(r"\s+", " ", value).strip().casefold()
