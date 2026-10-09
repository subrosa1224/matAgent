"""Authorized PDF ingestion and provider-neutral RAG services."""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import (
    ExperimentalDataRow,
    IngestedDocument,
    RagRetrieveInput,
    RagRetrieveOutput,
    RetrievedChunk,
)


class RagConfigurationError(RuntimeError):
    """Raised when an optional PDF, embedding, or vector dependency is absent."""


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    page_count: int
    pages: tuple[ParsedPage, ...]


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    paper_id: str | None
    page_from: int
    page_to: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class VectorHit:
    chunk: ChunkRecord
    distance: float


class PdfParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


class EmbeddingProvider(Protocol):
    model_id: str
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class RerankerProvider(Protocol):
    def rerank(self, query: str, texts: Sequence[str]) -> list[float]: ...


class VectorStore(Protocol):
    def document_exists(self, document_id: str) -> bool: ...

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
    ) -> None: ...

    def search(
        self,
        query_embedding: Sequence[float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[VectorHit]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]: ...

    def get_document_chunks(self, document_id: str) -> list[ChunkRecord]: ...

    def save_experimental_facts(self, rows: Sequence[ExperimentalDataRow]) -> None: ...


class PyMuPdfParser:
    """Lazy PyMuPDF parser so metadata-only installations still import."""

    def __init__(self, *, max_pages: int = 500) -> None:
        self.max_pages = max_pages

    def parse(self, path: Path) -> ParsedDocument:
        try:
            pymupdf = importlib.import_module("pymupdf")
        except ImportError as exc:
            raise RagConfigurationError(
                "PyMuPDF is not installed; install the literature extra"
            ) from exc
        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise ValueError("PDF_PARSE_FAILED: unable to open PDF") from exc
        try:
            if document.page_count < 1:
                raise ValueError("PDF_PARSE_FAILED: PDF contains no pages")
            if document.page_count > self.max_pages:
                raise ValueError("PDF_TOO_LARGE: PDF exceeds the page limit")
            pages = tuple(
                ParsedPage(
                    page_number=index + 1,
                    text=_normalize_page_text(document.load_page(index).get_text()),
                )
                for index in range(document.page_count)
            )
        finally:
            document.close()
        if not any(page.text for page in pages):
            raise ValueError("OCR_REQUIRED: PDF has no extractable text")
        return ParsedDocument(page_count=len(pages), pages=pages)


def infer_pdf_title(path: Path) -> str | None:
    """Read a likely article title without invoking an LLM."""
    try:
        pymupdf = importlib.import_module("pymupdf")
        document = pymupdf.open(path)
    except Exception:
        return None
    try:
        metadata_title = " ".join(str(document.metadata.get("title") or "").split())
        if len(metadata_title) >= 12 and metadata_title.casefold() not in {
            path.name.casefold(),
            path.stem.casefold(),
        }:
            return metadata_title
        page = document.load_page(0)
        spans = [
            span
            for block in page.get_text("dict").get("blocks", ())
            for line in block.get("lines", ())
            for span in line.get("spans", ())
            if str(span.get("text", "")).strip()
        ]
        if not spans:
            return None
        largest = max(float(span.get("size", 0)) for span in spans)
        title_parts = [
            " ".join(str(span["text"]).split())
            for span in spans
            if float(span.get("size", 0)) >= largest * 0.9
            and float(span.get("bbox", (0, 0, 0, 0))[1]) < page.rect.height * 0.45
        ]
        inferred = " ".join(part for part in title_parts if part)
        return inferred if 12 <= len(inferred) <= 500 else None
    finally:
        document.close()


class BgeM3EmbeddingProvider:
    model_id = "BAAI/bge-m3"
    dimensions = 1024

    def __init__(self, *, revision: str | None = None, device: str = "cpu") -> None:
        try:
            sentence_transformers = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise RagConfigurationError(
                "sentence-transformers is not installed; install the rag extra"
            ) from exc
        self._model = sentence_transformers.SentenceTransformer(
            self.model_id, revision=revision, device=device
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [self._validate_vector(row.tolist()) for row in values]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector):
            raise ValueError("embedding model returned an invalid vector")
        return vector


class FlagEmbeddingReranker:
    def __init__(
        self,
        *,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ) -> None:
        try:
            flag_embedding = importlib.import_module("FlagEmbedding")
        except ImportError as exc:
            raise RagConfigurationError(
                "FlagEmbedding is not installed; install the rag extra"
            ) from exc
        self._model = flag_embedding.FlagReranker(model_id, use_fp16=device != "cpu")

    def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self._model.compute_score([[query, text] for text in texts])
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]


class LiteratureRagService:
    def __init__(
        self,
        *,
        parser: PdfParser,
        embeddings: EmbeddingProvider,
        store: VectorStore,
        reranker: RerankerProvider | None = None,
        ingestion_roots: Sequence[Path],
        max_pdf_bytes: int = 50 * 1024 * 1024,
        chunk_chars: int = 3200,
        overlap_chars: int = 400,
    ) -> None:
        self.parser = parser
        self.embeddings = embeddings
        self.store = store
        self.reranker = reranker
        self.ingestion_roots = tuple(root.resolve() for root in ingestion_roots)
        self.max_pdf_bytes = max_pdf_bytes
        self.chunk_chars = chunk_chars
        self.overlap_chars = overlap_chars
        if chunk_chars < 200 or overlap_chars < 0 or overlap_chars >= chunk_chars:
            raise ValueError("invalid chunk configuration")

    def ingest(
        self, paths: Sequence[str], paper_id: str | None
    ) -> tuple[IngestedDocument, ...]:
        return tuple(
            self._ingest_one(self._authorize_path(value), paper_id) for value in paths
        )

    def retrieve(self, request: RagRetrieveInput) -> RagRetrieveOutput:
        query_vector = self.embeddings.embed_query(request.query)
        self._validate_vectors((query_vector,))
        hits = self.store.search(
            query_vector, paper_ids=request.paper_ids, limit=request.top_n
        )
        ranked: list[tuple[VectorHit, float | None]]
        if self.reranker and hits:
            scores = self.reranker.rerank(
                request.query, [hit.chunk.text for hit in hits]
            )
            if len(scores) != len(hits):
                raise ValueError("reranker returned an invalid score count")
            scored = sorted(
                zip(hits, scores, strict=True),
                key=lambda item: (-item[1], item[0].distance, item[0].chunk.chunk_id),
            )
            ranked = list(scored)
        else:
            ranked = [(hit, None) for hit in hits]
        chunks = tuple(
            RetrievedChunk(
                chunk_id=hit.chunk.chunk_id,
                document_id=hit.chunk.document_id,
                paper_id=hit.chunk.paper_id,
                page_from=hit.chunk.page_from,
                page_to=hit.chunk.page_to,
                text=hit.chunk.text,
                text_sha256=hit.chunk.text_sha256,
                vector_distance=hit.distance,
                rerank_score=score,
            )
            for hit, score in ranked[: request.top_k]
        )
        return RagRetrieveOutput(
            query=request.query, chunks=chunks, returned_count=len(chunks)
        )

    def _authorize_path(self, value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not any(
            path == root or root in path.parents for root in self.ingestion_roots
        ):
            raise ValueError("PDF_NOT_AUTHORIZED: path is outside ingestion roots")
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise ValueError("PDF_NOT_FOUND: authorized PDF file does not exist")
        if path.stat().st_size > self.max_pdf_bytes:
            raise ValueError("PDF_TOO_LARGE: PDF exceeds the byte limit")
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError("PDF_PARSE_FAILED: file does not have a PDF signature")
        return path

    def _ingest_one(self, path: Path, paper_id: str | None) -> IngestedDocument:
        file_bytes = path.read_bytes()
        file_sha = hashlib.sha256(file_bytes).hexdigest()
        document_id = f"doc-{file_sha[:24]}"
        if self.store.document_exists(document_id):
            parsed = self.parser.parse(path)
            chunks = _chunk_document(
                parsed, document_id, paper_id, self.chunk_chars, self.overlap_chars
            )
            return IngestedDocument(
                document_id=document_id,
                paper_id=paper_id,
                file_name=path.name,
                sha256=file_sha,
                page_count=parsed.page_count,
                chunk_count=len(chunks),
                embedding_model=self.embeddings.model_id,
                status="already_indexed",
            )
        parsed = self.parser.parse(path)
        chunks = _chunk_document(
            parsed, document_id, paper_id, self.chunk_chars, self.overlap_chars
        )
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        self._validate_vectors(vectors)
        self.store.upsert(
            document_id=document_id,
            file_sha256=file_sha,
            file_name=path.name,
            paper_id=paper_id,
            embedding_model=self.embeddings.model_id,
            chunks=chunks,
            embeddings=vectors,
        )
        return IngestedDocument(
            document_id=document_id,
            paper_id=paper_id,
            file_name=path.name,
            sha256=file_sha,
            page_count=parsed.page_count,
            chunk_count=len(chunks),
            embedding_model=self.embeddings.model_id,
            status="indexed",
        )

    def _validate_vectors(self, vectors: Sequence[Sequence[float]]) -> None:
        if any(len(vector) != self.embeddings.dimensions for vector in vectors):
            raise ValueError("embedding vector dimension mismatch")
        if any(not all(math.isfinite(value) for value in vector) for vector in vectors):
            raise ValueError("embedding vector contains a non-finite value")


def parse_pdf_chunks(
    path: Path,
    *,
    parser: PdfParser,
    document_id: str,
    paper_id: str | None = None,
    chunk_chars: int = 3200,
    overlap_chars: int = 400,
) -> tuple[ChunkRecord, ...]:
    """Create deterministic evidence chunks without requiring a vector database."""
    if chunk_chars < 200 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("invalid chunk configuration")
    return _chunk_document(
        parser.parse(path),
        document_id,
        paper_id,
        chunk_chars,
        overlap_chars,
    )


def _normalize_page_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\u00ad", "").replace("\r", "\n")
    value = re.sub(r"(?<=\w)-\n(?=\w)", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _chunk_document(
    document: ParsedDocument,
    document_id: str,
    paper_id: str | None,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[ChunkRecord, ...]:
    chunks: list[ChunkRecord] = []
    for page in document.pages:
        text = page.text
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + chunk_chars // 2, end)
                if boundary > start:
                    end = boundary
            chunk_text = text[start:end].strip()
            if chunk_text:
                digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                identity = f"{document_id}:{page.page_number}:{start}:{digest}".encode()
                chunk_id = f"chunk-{hashlib.sha256(identity).hexdigest()[:24]}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        paper_id=paper_id,
                        page_from=page.page_number,
                        page_to=page.page_number,
                        text=chunk_text,
                        text_sha256=digest,
                    )
                )
            if end >= len(text):
                break
            start = max(end - overlap_chars, start + 1)
    if not chunks:
        raise ValueError("OCR_REQUIRED: PDF has no indexable text")
    return tuple(chunks)
