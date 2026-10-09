from __future__ import annotations

import hashlib
from collections.abc import Sequence

from materials_screening.sub_agents.literature.extraction import (
    ExperimentalExtractionService,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalDataCandidate,
    ExperimentalDataRow,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord, VectorHit


class ExtractionStore:
    def __init__(self, chunks: Sequence[ChunkRecord]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.saved: list[ExperimentalDataRow] = []

    def document_exists(self, document_id: str) -> bool:
        return True

    def upsert(self, **kwargs: object) -> None:
        raise AssertionError("not used")

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[VectorHit]:
        return []

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]:
        return [self.chunks[value] for value in chunk_ids if value in self.chunks]

    def save_experimental_facts(self, rows: Sequence[ExperimentalDataRow]) -> None:
        self.saved.extend(rows)


def _chunk() -> ChunkRecord:
    text = (
        "The polyhedral structure with a porosity of 70 % had the highest rate "
        "of new bone formation. "
        "A specific surface area of 10.49–10.69 mm2 mm−3 and a permeability "
        "of 3.74 × 10−9 m2 promotes osteogenic differentiation."
    )
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id="paper-1",
        page_from=11,
        page_to=11,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_valid_candidate_is_staged_for_review() -> None:
    chunk = _chunk()
    store = ExtractionStore((chunk,))
    service = ExperimentalExtractionService(store)
    result = service.validate_and_save(
        document_id="doc-1",
        candidates=(
            ExperimentalDataCandidate(
                chunk_id="chunk-1",
                material="calcium phosphate bioceramic",
                variable_name="specific surface area",
                variable_value="10.49–10.69 mm2 mm−3",
                performance_metric="osteogenic differentiation",
                performance_value="promotes",
                source_quote=chunk.text,
            ),
        ),
    )
    assert result.rejected_count == 0
    assert len(result.rows) == 1
    assert result.rows[0].llm_extracted is True
    assert result.rows[0].review_status == "pending"
    assert result.rows[0].page_from == 11
    assert store.saved == list(result.rows)


def test_porosity_percentage_and_performance_are_staged_separately() -> None:
    chunk = _chunk()
    quote = (
        "The polyhedral structure with a porosity of 70 % had the highest rate "
        "of new bone formation"
    )
    store = ExtractionStore((chunk,))
    result = ExperimentalExtractionService(store).validate_and_save(
        document_id="doc-1",
        candidates=(
            ExperimentalDataCandidate(
                chunk_id="chunk-1",
                material="polyhedral CaP ceramic scaffold",
                variable_name="porosity",
                variable_value="70 %",
                performance_metric="new bone formation",
                performance_value="highest rate",
                source_quote=quote,
            ),
        ),
    )

    assert result.rejected_count == 0
    assert len(result.rows) == 1
    assert result.rows[0].variable_name == "porosity"
    assert result.rows[0].variable_value == "70 %"
    assert result.rows[0].performance_value == "highest rate"


def test_candidate_without_exact_evidence_is_rejected() -> None:
    store = ExtractionStore((_chunk(),))
    result = ExperimentalExtractionService(store).validate_and_save(
        document_id="doc-1",
        candidates=(
            ExperimentalDataCandidate(
                chunk_id="chunk-1",
                material="bioceramic",
                variable_name="porosity",
                variable_value="99%",
                performance_metric="bone formation",
                performance_value="excellent",
                source_quote="Porosity was 99% and bone formation was excellent.",
            ),
        ),
    )
    assert result.rows == ()
    assert result.rejected_count == 1
    assert "source_quote" in result.warnings[0]
    assert store.saved == []
