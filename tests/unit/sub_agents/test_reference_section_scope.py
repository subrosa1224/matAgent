import hashlib
from pathlib import Path
from typing import Any

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.sub_agents.literature.automation import (
    AutomatedDossierExtractor,
)
from materials_screening.sub_agents.literature.dossier import (
    DossierExtractionBatch,
    DossierItem,
    DossierItemCandidate,
    DossierSummaryTranslation,
    DossierTranslationBatch,
    PaperDossier,
    PaperDossierStore,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReportService,
)


def chunk(text: str, page: int) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"chunk-{page}",
        document_id="doc-1",
        paper_id=None,
        text=text,
        page_from=page,
        page_to=page,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


OWN = "In this work, oxide samples were prepared at 450 C."
REFERENCE = (
    "24. Smith, J. Molten salt synthesis of oxide nanorods. Journal 2013, 5, 10-20."
)
CHUNKS = (
    chunk(
        OWN + "\nReferences\n23. Doe, A. Hydrothermal synthesis. Journal 2012, 4, 1-9.",
        7,
    ),
    chunk("Journal 2025, 10, 12 of 12\n" + REFERENCE, 8),
)


class Llm:
    def generate_structured(self, **kwargs: Any) -> StructuredProviderResponse[Any]:
        parsed = DossierExtractionBatch(
            items=tuple(
                DossierItemCandidate(
                    category="preparation",
                    summary=summary,
                    source_quote=quote,
                    chunk_id=identifier,
                )
                for summary, quote, identifier in (
                    ("本文样品在450 C制备。", OWN, "chunk-7"),
                    ("熔盐法制备氧化物纳米棒。", REFERENCE, "chunk-8"),
                )
            )
            if "Target category: preparation" in kwargs["user_text"]
            else ()
        )
        if kwargs["output_model"] is DossierTranslationBatch:
            parsed = DossierTranslationBatch(
                items=(
                    DossierSummaryTranslation(
                        item_key="item-0", chinese_summary="本文样品在450 C制备。"
                    ),
                )
            )
        return StructuredProviderResponse(
            parsed=parsed,
            provider="fake",
            model="fake",
            request_id="test",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


def test_dossier_excludes_reference_continuation_but_keeps_preceding_method() -> None:
    dossier, warnings = AutomatedDossierExtractor(Llm()).extract(
        document_id="doc-1",
        title="Paper",
        chunks=CHUNKS,
    )
    assert any(item.source_quote == OWN for item in dossier.items)
    assert all(item.source_quote != REFERENCE for item in dossier.items)
    assert any("reference section" in warning for warning in warnings)


def test_old_dossier_report_filters_reference_without_overwriting(
    tmp_path: Path,
) -> None:
    pending = PaperDossierStore(tmp_path / "pending")
    original = PaperDossier(
        document_id="doc-1",
        title="Paper",
        extraction_method="llm",
        items=tuple(
            DossierItem(
                category="preparation",
                summary=summary,
                source_quote=quote,
                chunk_id=identifier,
                page=page,
                risk_level="low",
            )
            for summary, quote, identifier, page in (
                ("本文样品在450 C制备。", OWN, "chunk-7", 7),
                ("熔盐法制备氧化物纳米棒。", REFERENCE, "chunk-8", 8),
            )
        ),
    )
    pending.save_pending(original)

    class Store:
        def get_document_chunks(self, _: str) -> tuple[ChunkRecord, ...]:
            return CHUNKS

        def get_document_metadata(self, _: str) -> None:
            return None

    report = LiteratureUserReportService(
        metadata_store=Store(),
        pending_store=pending,
        approved_store=PaperDossierStore(tmp_path / "approved"),
    ).build(topic="材料制备", document_ids=("doc-1",))
    assert [item.source_quote for item in report.papers[0].evidence] == [OWN]
    assert report.papers[0].excluded_unsafe_count == 1
    assert pending.load("doc-1") == original


def test_reference_heading_is_not_a_prose_mention_and_appendix_resumes_body() -> None:
    from materials_screening.sub_agents.literature.evidence_scope import (
        is_reference_evidence,
    )

    body = chunk("The references below discuss synthesis.\n" + OWN, 2)
    appendix = chunk("Appendix A\n" + OWN, 9)
    records = (body, *CHUNKS, appendix)
    assert not is_reference_evidence(OWN, body, records)
    assert is_reference_evidence(REFERENCE, CHUNKS[1], tuple(reversed(records)))
    assert not is_reference_evidence(OWN, appendix, records)


def test_quote_spanning_reference_heading_cannot_launder_citation() -> None:
    from materials_screening.sub_agents.literature.evidence_scope import (
        is_reference_evidence,
    )

    assert is_reference_evidence(CHUNKS[0].text, CHUNKS[0], CHUNKS)
