from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import pytest

from materials_screening.sub_agents.literature.models import RagRetrieveInput
from materials_screening.sub_agents.literature.rag import (
    ChunkRecord,
    LiteratureRagService,
    ParsedDocument,
    ParsedPage,
    VectorHit,
    _normalize_page_text,
)


class FakeParser:
    def parse(self, path: Path) -> ParsedDocument:
        return ParsedDocument(
            page_count=2,
            pages=(
                ParsedPage(1, "TiO2 photocatalysis " * 40),
                ParsedPage(2, "Doping improves activity " * 30),
            ),
        )


class FakeEmbeddings:
    model_id = "fake-bge-m3"
    dimensions = 4

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, float(index)] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


class FakeReranker:
    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(index) for index, _ in enumerate(texts)]


class FakeStore:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[ChunkRecord, ...]] = {}

    def document_exists(self, document_id: str) -> bool:
        return document_id in self.documents

    def upsert(
        self,
        *,
        document_id: str,
        file_sha256: str,
        file_name: str,
        paper_id: str | None,
        embedding_model: str,
        chunks: Sequence[ChunkRecord],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        assert len(chunks) == len(embeddings)
        self.documents[document_id] = tuple(chunks)

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[VectorHit]:
        chunks = [chunk for values in self.documents.values() for chunk in values]
        if paper_ids:
            chunks = [chunk for chunk in chunks if chunk.paper_id in paper_ids]
        return [
            VectorHit(chunk=chunk, distance=float(index) / 10)
            for index, chunk in enumerate(chunks[:limit])
        ]


def _service(root: Path, store: FakeStore | None = None) -> LiteratureRagService:
    return LiteratureRagService(
        parser=FakeParser(),
        embeddings=FakeEmbeddings(),
        store=store or FakeStore(),
        reranker=FakeReranker(),
        ingestion_roots=(root,),
        chunk_chars=300,
        overlap_chars=50,
    )


def _pdf(root: Path, name: str = "paper.pdf") -> Path:
    path = root / name
    path.write_bytes(b"%PDF-1.7\nfixture")
    return path


def test_ingestion_is_authorized_traceable_and_idempotent(tmp_path: Path) -> None:
    store = FakeStore()
    service = _service(tmp_path, store)
    path = _pdf(tmp_path)
    first = service.ingest((str(path),), "paper-1")[0]
    second = service.ingest((str(path),), "paper-1")[0]
    assert first.status == "indexed"
    assert second.status == "already_indexed"
    assert first.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert first.page_count == 2
    assert first.chunk_count > 1
    assert all(
        chunk.page_from in {1, 2} for chunk in store.documents[first.document_id]
    )


def test_ingestion_rejects_outside_root_and_fake_pdf(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = _pdf(tmp_path, "outside.pdf")
    with pytest.raises(ValueError, match="PDF_NOT_AUTHORIZED"):
        _service(allowed).ingest((str(outside),), None)
    fake = allowed / "fake.pdf"
    fake.write_text("not a pdf", encoding="utf-8")
    with pytest.raises(ValueError, match="PDF_PARSE_FAILED"):
        _service(allowed).ingest((str(fake),), None)


def test_retrieve_filters_reranks_and_preserves_evidence(tmp_path: Path) -> None:
    store = FakeStore()
    service = _service(tmp_path, store)
    service.ingest((str(_pdf(tmp_path)),), "paper-1")
    result = service.retrieve(
        RagRetrieveInput(
            query="doping activity", paper_ids=("paper-1",), top_n=20, top_k=3
        )
    )
    assert result.returned_count == 3
    assert result.chunks[0].rerank_score > result.chunks[-1].rerank_score
    assert all(chunk.paper_id == "paper-1" for chunk in result.chunks)
    assert all(chunk.text_sha256 for chunk in result.chunks)


def test_retrieve_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        RagRetrieveInput(query="test", top_n=2, top_k=3)


def test_pdf_text_normalization_removes_postgres_forbidden_nul_bytes() -> None:
    normalized = _normalize_page_text("hydroxy\x00apatite\nresult")

    assert normalized == "hydroxyapatite\nresult"
    assert "\x00" not in normalized
