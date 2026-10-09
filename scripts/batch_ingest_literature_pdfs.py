"""Batch-ingest authorized PDFs with one embedding-model load."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)
from materials_screening.sub_agents.literature.rag import (
    BgeM3EmbeddingProvider,
    LiteratureRagService,
    PyMuPdfParser,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    raw_roots = os.getenv("LITERATURE_INGEST_ROOTS", "").strip()
    if not database_url or not raw_roots:
        raise SystemExit("literature database URL and ingestion roots are required")
    roots = tuple(
        Path(value.strip()).resolve()
        for value in raw_roots.split(os.pathsep)
        if value.strip()
    )
    paths = tuple(
        str(path)
        for root in roots
        for path in sorted(root.glob("*.pdf"))
        if path.is_file()
    )
    if not paths:
        raise SystemExit("no PDF files found in configured ingestion roots")
    print(f"loading embedding model for {len(paths)} PDFs...", flush=True)
    embeddings = BgeM3EmbeddingProvider(
        revision=os.getenv("LITERATURE_EMBEDDING_REVISION") or None,
        device=os.getenv("LITERATURE_MODEL_DEVICE", "cpu"),
    )
    service = LiteratureRagService(
        parser=PyMuPdfParser(
            max_pages=int(os.getenv("LITERATURE_MAX_PDF_PAGES", "500"))
        ),
        embeddings=embeddings,
        store=PgVectorLiteratureStore(database_url, dimensions=embeddings.dimensions),
        ingestion_roots=roots,
        max_pdf_bytes=int(os.getenv("LITERATURE_MAX_PDF_MB", "50")) * 1024 * 1024,
    )
    print("ingesting PDFs...", flush=True)
    documents = service.ingest(paths, None)
    print(
        json.dumps(
            [document.model_dump(mode="json") for document in documents],
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
