"""Run a real PDF -> pgvector -> reranker LiteratureAgent smoke test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.models import RagRetrieveInput
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)
from materials_screening.sub_agents.literature.rag import (
    BgeM3EmbeddingProvider,
    FlagEmbeddingReranker,
    LiteratureRagService,
    PyMuPdfParser,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--query",
        default=(
            "optimal structural characteristics osteoinductivity bioceramics "
            "high-throughput screening machine learning"
        ),
    )
    args = parser.parse_args()
    load_dotenv()

    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    root_values = os.getenv("LITERATURE_INGEST_ROOTS", "").strip()
    if not database_url or not root_values:
        raise SystemExit("literature database URL and ingestion roots are required")
    roots = tuple(
        Path(value.strip()) for value in root_values.split(os.pathsep) if value.strip()
    )
    device = os.getenv("LITERATURE_MODEL_DEVICE", "cpu")
    embeddings = BgeM3EmbeddingProvider(
        revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
        device=device,
    )
    store = PgVectorLiteratureStore(database_url, dimensions=embeddings.dimensions)
    service = LiteratureRagService(
        parser=PyMuPdfParser(
            max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
        ),
        embeddings=embeddings,
        store=store,
        ingestion_roots=roots,
        max_pdf_bytes=int(os.getenv("LITERATURE_MAX_PDF_MB", "50")) * 1024 * 1024,
    )
    print("ingesting PDF...", flush=True)
    document = service.ingest((str(args.pdf),), None)[0]
    print(json.dumps(document.model_dump(mode="json"), ensure_ascii=False), flush=True)

    print("loading reranker...", flush=True)
    service.reranker = FlagEmbeddingReranker(device=device)
    print("retrieving evidence...", flush=True)
    result = service.retrieve(RagRetrieveInput(query=args.query, top_n=10, top_k=5))
    payload = {
        "query": result.query,
        "returned_count": result.returned_count,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "pages": [chunk.page_from, chunk.page_to],
                "vector_distance": chunk.vector_distance,
                "rerank_score": chunk.rerank_score,
                "text_preview": " ".join(chunk.text.split())[:500],
            }
            for chunk in result.chunks
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if result.returned_count < 1:
        raise SystemExit("RAG smoke test returned no evidence")


if __name__ == "__main__":
    main()
