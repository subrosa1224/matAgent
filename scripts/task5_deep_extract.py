"""Extract pending, evidence-linked experiment matrices for task 5 papers."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from materials_screening.llm.factory import create_llm_provider
from materials_screening.planner.settings import Settings
from materials_screening.sub_agents.literature.batch import document_id_for_pdf
from materials_screening.sub_agents.literature.matrix_automation import (
    AutomatedMatrixExtractor,
)
from materials_screening.sub_agents.literature.rag import (
    ChunkRecord,
    PyMuPdfParser,
    parse_pdf_chunks,
)

PAPERS = (
    "J_S_Earl_2006_J._Phys.__Conf._Ser._26_268.pdf",
    "cg801353n.pdf",
    (
        "Bioinorganic Chemistry and Applications - 2022 - Szterner - The Synthesis "
        "of Hydroxyapatite by Hydrothermal Process with.pdf"
    ),
)


class LocalChunkStore:
    """Minimal evidence store used by the existing matrix validator."""

    def __init__(self, chunks: Sequence[ChunkRecord]) -> None:
        self._chunks = {chunk.chunk_id: chunk for chunk in chunks}

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[ChunkRecord]:
        return [self._chunks[item] for item in chunk_ids if item in self._chunks]


def _dump_models(items: Sequence[object]) -> list[object]:
    return [item.model_dump(mode="json") for item in items]  # type: ignore[attr-defined]


def main() -> None:
    load_dotenv()
    settings = Settings()
    extraction_model = os.getenv(
        "LITERATURE_EXTRACTION_MODEL", "intern-s1-mini"
    ).strip()
    provider = create_llm_provider(
        settings.model_copy(
            update={
                "intern_model": extraction_model or settings.intern_model,
                "intern_thinking_mode": False,
            }
        )
    )
    parser = PyMuPdfParser()
    source_root = Path(r"D:\Downloads")
    output_dir = Path("outputs/task5_nha_trial/deep_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, object]] = []
    for index, name in enumerate(PAPERS, 1):
        path = source_root / name
        document_id = document_id_for_pdf(path)
        chunks = parse_pdf_chunks(path, parser=parser, document_id=document_id)
        print(f"[{index}/{len(PAPERS)}] 深度抽取 {name}", flush=True)
        extraction = AutomatedMatrixExtractor(
            provider,
            LocalChunkStore(chunks),  # type: ignore[arg-type]
        ).extract(
            document_id=document_id,
            chunks=chunks,
            max_output_tokens=min(settings.llm_max_output_tokens, 8192),
            max_evidence_chars=36000,
            batch_chars=9000,
        )
        result = {
            "document_id": document_id,
            "source_pdf": str(path),
            "groups": _dump_models(extraction.groups),
            "measurements": _dump_models(extraction.measurements),
            "comparisons": _dump_models(extraction.comparisons),
            "claims": _dump_models(extraction.claims),
            "claim_evidence_links": _dump_models(extraction.claim_evidence_links),
            "warnings": list(extraction.warnings),
        }
        target = output_dir / f"{document_id}.json"
        target.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary.append(
            {
                "document_id": document_id,
                "source_pdf": str(path),
                "result_path": str(target.resolve()),
                "group_count": len(extraction.groups),
                "measurement_count": len(extraction.measurements),
                "claim_count": len(extraction.claims),
                "warning_count": len(extraction.warnings),
            }
        )
        print(
            "  -> "
            f"groups={len(extraction.groups)}, "
            f"measurements={len(extraction.measurements)}, "
            f"claims={len(extraction.claims)}, "
            f"warnings={len(extraction.warnings)}",
            flush=True,
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
