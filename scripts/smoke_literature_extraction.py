"""Stage evidence-validated experimental facts from the real smoke-test PDF."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from materials_screening.sub_agents.literature.extraction import (
    ExperimentalExtractionService,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalDataCandidate,
)
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)

DOCUMENT_ID = "doc-6be19aada0b95fe176f931ee"
CHUNK_ID = "chunk-ecbe2fd0cce2de6097987d80"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("loading configuration...", flush=True)
    load_dotenv()
    database_url = os.getenv("LITERATURE_DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit("LITERATURE_DATABASE_URL is required")
    print("loading evidence chunk...", flush=True)
    store = PgVectorLiteratureStore(database_url)
    parameter_chunks = store.get_chunks((CHUNK_ID,))
    if len(parameter_chunks) != 1:
        raise SystemExit("expected smoke-test evidence chunk is missing")
    chunk = parameter_chunks[0]
    document_chunks = store.get_document_chunks(DOCUMENT_ID)
    porosity_chunk = next(
        (
            item
            for item in document_chunks
            if "porosity of 70 %" in item.text.casefold()
        ),
        None,
    )
    if porosity_chunk is None:
        raise SystemExit("expected porosity evidence chunk is missing")
    porosity_text = porosity_chunk.text
    porosity_start = porosity_text.casefold().find("the polyhedral structure")
    porosity_marker = "new bone formation"
    porosity_end = porosity_text.casefold().find(porosity_marker, porosity_start)
    start = chunk.text.find("10.49")
    marker = "osteogenic differentiation"
    end = chunk.text.find(marker, start)
    if porosity_start < 0 or porosity_end < 0 or start < 0 or end < 0:
        raise SystemExit("expected experimental sentence is missing")
    porosity_quote = porosity_text[
        porosity_start : porosity_end + len(porosity_marker)
    ]
    quote = chunk.text[start : end + len(marker)]
    candidates = (
        ExperimentalDataCandidate(
            chunk_id=porosity_chunk.chunk_id,
            material="polyhedral CaP ceramic scaffold",
            variable_name="porosity",
            variable_value="70 %",
            performance_metric="new bone formation",
            performance_value="highest rate",
            source_quote=porosity_quote,
        ),
        ExperimentalDataCandidate(
            chunk_id=CHUNK_ID,
            material="calcium phosphate bioceramic scaffold",
            variable_name="specific surface area",
            variable_value="10.49–10.69 mm2 mm−3",
            performance_metric="osteogenic differentiation",
            performance_value="promotes",
            source_quote=quote,
        ),
        ExperimentalDataCandidate(
            chunk_id=CHUNK_ID,
            material="calcium phosphate bioceramic scaffold",
            variable_name="permeability",
            variable_value="3.74 × 10−9 m2",
            performance_metric="osteogenic differentiation",
            performance_value="promotes",
            source_quote=quote,
        ),
    )
    print("validating and staging facts...", flush=True)
    result = ExperimentalExtractionService(store).validate_and_save(
        document_id=DOCUMENT_ID,
        candidates=candidates,
    )
    print(result.model_dump_json(indent=2), flush=True)
    if result.rejected_count or len(result.rows) != 3:
        raise SystemExit("real experimental extraction smoke test failed")


if __name__ == "__main__":
    main()
