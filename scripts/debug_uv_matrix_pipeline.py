"""Read-only diagnostic of the existing matrix extractor on one supplied PDF.

Does not save, approve, or replace any matrix records. Prints bounded model
candidate traces and validator warnings; never prints connection credentials.
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

from materials_screening.planner.settings import Settings
from materials_screening.llm.factory import create_llm_provider
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
)
from materials_screening.sub_agents.literature.pgvector_store import (
    PgVectorLiteratureStore,
)


class CapturingProvider:
    def __init__(self, provider: object) -> None:
        self.provider = provider
        self.batches: list[dict[str, object]] = []

    def generate_structured(self, **kwargs: object) -> object:
        result = self.provider.generate_structured(**kwargs)
        parsed = result.parsed.model_dump(mode="json")
        self.batches.append(parsed)
        print(
            json.dumps(
                {
                    "batch": len(self.batches),
                    "groups": len(parsed.get("groups", [])),
                    "measurements": len(parsed.get("measurements", [])),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    settings = Settings()
    model = os.getenv("LITERATURE_EXTRACTION_MODEL", "intern-s1-mini").strip()
    provider = CapturingProvider(
        create_llm_provider(
            settings.model_copy(update={"intern_model": model or settings.intern_model})
        )
    )
    store = PgVectorLiteratureStore(os.environ["LITERATURE_DATABASE_URL"])
    document_id = "doc-fcc732f960a270e9176ab5c0"
    chunks = store.get_document_chunks(document_id)
    result = AutomatedMatrixExtractor(provider, store).extract(
        document_id=document_id,
        chunks=chunks,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    print(
        json.dumps(
            {
                "document_id": document_id,
                "chunks": len(chunks),
                "batches": provider.batches,
                "accepted_groups": len(result.groups),
                "accepted_measurements": [
                    item.model_dump(mode="json") for item in result.measurements
                ],
                "warnings": result.warnings,
                "persisted": False,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
