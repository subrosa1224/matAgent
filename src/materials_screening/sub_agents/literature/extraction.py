"""Evidence-validated experimental-data extraction and review staging."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from .models import (
    ExperimentalDataCandidate,
    ExperimentalDataRow,
    ExtractExperimentalDataOutput,
)
from .rag import ChunkRecord, VectorStore


class ExperimentalExtractionService:
    def __init__(self, store: VectorStore) -> None:
        self.store = store

    def validate_and_save(
        self,
        *,
        document_id: str,
        candidates: Sequence[ExperimentalDataCandidate],
    ) -> ExtractExperimentalDataOutput:
        chunk_ids = tuple(dict.fromkeys(row.chunk_id for row in candidates))
        chunks = {chunk.chunk_id: chunk for chunk in self.store.get_chunks(chunk_ids)}
        rows: list[ExperimentalDataRow] = []
        warnings: list[str] = []
        for index, candidate in enumerate(candidates):
            chunk = chunks.get(candidate.chunk_id)
            reason = _rejection_reason(document_id, candidate, chunk)
            if reason:
                warnings.append(f"row {index + 1} rejected: {reason}")
                continue
            assert chunk is not None
            identity = "|".join(
                (
                    document_id,
                    candidate.chunk_id,
                    candidate.material,
                    candidate.variable_name,
                    candidate.variable_value,
                    candidate.performance_metric,
                    candidate.performance_value,
                    _normalize(candidate.source_quote),
                )
            )
            rows.append(
                ExperimentalDataRow(
                    fact_id=f"fact-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
                    document_id=document_id,
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    page_from=chunk.page_from,
                    page_to=chunk.page_to,
                    material=candidate.material,
                    variable_name=candidate.variable_name,
                    variable_value=candidate.variable_value,
                    performance_metric=candidate.performance_metric,
                    performance_value=candidate.performance_value,
                    conditions=candidate.conditions,
                    source_quote=candidate.source_quote,
                    source_text_sha256=chunk.text_sha256,
                )
            )
        self.store.save_experimental_facts(rows)
        return ExtractExperimentalDataOutput(
            document_id=document_id,
            rows=tuple(rows),
            rejected_count=len(candidates) - len(rows),
            warnings=tuple(warnings),
        )


def _rejection_reason(
    document_id: str,
    candidate: ExperimentalDataCandidate,
    chunk: ChunkRecord | None,
) -> str | None:
    if chunk is None:
        return "chunk does not exist"
    if chunk.document_id != document_id:
        return "chunk belongs to a different document"
    normalized_text = _normalize(chunk.text)
    normalized_quote = _normalize(candidate.source_quote)
    if normalized_quote not in normalized_text:
        return "source_quote is not present in the evidence chunk"
    if _normalize(candidate.variable_value) not in normalized_quote:
        return "variable_value is not present in source_quote"
    if _normalize(candidate.performance_value) not in normalized_quote:
        return "performance_value is not present in source_quote"
    return None


def _normalize(value: str) -> str:
    value = value.replace("−", "-").replace("–", "-").replace("×", "x")
    return re.sub(r"\s+", " ", value).strip().casefold()
