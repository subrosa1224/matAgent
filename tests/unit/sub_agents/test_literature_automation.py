from __future__ import annotations

from typing import Any

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.sub_agents.literature.automation import (
    AutomatedDossierExtractor,
    _category_evidence_consistent,
    _ground_quote,
    plan_dossier_tasks,
)
from materials_screening.sub_agents.literature.dossier import (
    DossierExtractionBatch,
    DossierItemCandidate,
    DossierSummaryTranslation,
    DossierTranslationBatch,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


class FakeLlm:
    def __init__(
        self,
        batch: DossierExtractionBatch,
        *,
        translation: str = "70 %孔隙率支架的骨体积分数为14.2 %。",
    ) -> None:
        self.batch = batch
        self.translation = translation

    def generate_structured(self, **_: Any) -> StructuredProviderResponse[Any]:
        if _["output_model"] is DossierTranslationBatch:
            parsed: Any = DossierTranslationBatch(
                items=(
                    DossierSummaryTranslation(
                        item_key="item-0",
                        chinese_summary=self.translation,
                    ),
                )
            )
            return StructuredProviderResponse(
                parsed=parsed,
                provider="fake",
                model="fake",
                request_id="translation-1",
                latency_ms=0,
                input_tokens=10,
                output_tokens=10,
                reasoning_tokens=0,
                raw_output_sha256="d" * 64,
            )
        user_text = str(_.get("user_text", ""))
        target = next(
            (
                item
                for item in self.batch.items
                if f"Target category: {item.category}" in user_text
            ),
            None,
        )
        return StructuredProviderResponse(
            parsed=DossierExtractionBatch(items=(target,) if target else ()),
            provider="fake",
            model="fake",
            request_id="request-1",
            latency_ms=0,
            input_tokens=10,
            output_tokens=10,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


def _chunk() -> ChunkRecord:
    text = "The 70 % porous scaffold reached a bone volume of 14.2 %."
    return ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=7,
        page_to=7,
        text=text,
        text_sha256="b" * 64,
    )


def test_automated_dossier_accepts_verbatim_number_grounded_item() -> None:
    candidate = DossierItemCandidate(
        category="performance_result",
        summary="70 %孔隙率支架的骨体积分数为14.2 %。",
        source_quote=_chunk().text,
        chunk_id="chunk-1",
    )
    dossier, warnings = AutomatedDossierExtractor(
        FakeLlm(DossierExtractionBatch(items=(candidate,)))
    ).extract(document_id="doc-1", title="Paper", chunks=(_chunk(),))

    assert len(dossier.items) == 1
    assert dossier.items[0].risk_level == "medium"
    assert dossier.items[0].llm_extracted is True
    assert dossier.items[0].document_id == "doc-1"
    assert dossier.items[0].page_to == 7
    assert dossier.items[0].source_text_sha256 == "b" * 64
    assert dossier.review_status == "pending"
    assert dossier.extraction_version == "task-retrieval-v3-risk-calibrated"
    assert round(dossier.completeness_score or 0, 2) == 0.1
    assert "性能结果" in (dossier.narrative_summary or "")
    assert "research_problem" in dossier.missing_core_categories
    assert len(warnings) == 1
    assert warnings[0].startswith("missing core dossier categories")


def test_exact_qualitative_evidence_can_remain_low_risk_after_translation() -> None:
    text = (
        "The study investigates how scaffold architecture influences "
        "osteogenic differentiation and bone regeneration."
    )
    chunk = ChunkRecord(
        chunk_id="chunk-low",
        document_id="doc-1",
        paper_id=None,
        page_from=1,
        page_to=1,
        text=text,
        text_sha256="e" * 64,
    )
    candidate = DossierItemCandidate(
        category="research_problem",
        summary="The study investigates scaffold architecture and osteogenesis.",
        source_quote=text,
        chunk_id="chunk-low",
    )
    dossier, _ = AutomatedDossierExtractor(
        FakeLlm(
            DossierExtractionBatch(items=(candidate,)),
            translation="研究支架结构如何影响成骨分化和骨再生。",
        )
    ).extract(document_id="doc-1", title="Paper", chunks=(chunk,))

    assert dossier.items[0].risk_level == "low"
    assert dossier.items[0].confidence == 0.9


def test_automated_dossier_replaces_ungrounded_number_with_quote() -> None:
    candidate = DossierItemCandidate(
        category="performance_result",
        summary="70 %孔隙率支架的骨体积分数为99 %。",
        source_quote=_chunk().text,
        chunk_id="chunk-1",
    )
    dossier, warnings = AutomatedDossierExtractor(
        FakeLlm(DossierExtractionBatch(items=(candidate,)))
    ).extract(document_id="doc-1", title="Paper", chunks=(_chunk(),))

    assert dossier.items[0].summary == "70 %孔隙率支架的骨体积分数为14.2 %。"
    assert dossier.items[0].risk_level == "high"
    assert "replaced with source_quote" in warnings[0]


def test_task_planner_covers_every_category_and_prioritizes_evidence() -> None:
    method = ChunkRecord(
        chunk_id="chunk-method",
        document_id="doc-1",
        paper_id=None,
        page_from=5,
        page_to=5,
        text="Experimental methods: samples were annealed at 500 °C for 30 min.",
        text_sha256="c" * 64,
    )
    tasks = plan_dossier_tasks((_chunk(), method), chunks_per_task=1)

    assert len(tasks) == 12
    assert len({task.category for task in tasks}) == 12
    preparation = next(task for task in tasks if task.category == "preparation")
    assert preparation.chunks[0].chunk_id == "chunk-method"


def test_ground_quote_repairs_only_minor_pdf_noise() -> None:
    source = "The scaffold achieved 70% porosity. Bone formation increased."

    assert (
        _ground_quote("The scaffold achieved 70 % porosity.", source)
        == "The scaffold achieved 70% porosity."
    )
    assert _ground_quote("An unrelated invented conclusion.", source) is None


def test_unsafe_translation_becomes_chinese_review_notice() -> None:
    candidate = DossierItemCandidate(
        category="performance_result",
        summary=_chunk().text,
        source_quote=_chunk().text,
        chunk_id="chunk-1",
    )
    dossier, warnings = AutomatedDossierExtractor(
        FakeLlm(
            DossierExtractionBatch(items=(candidate,)),
            translation="该支架达到99 %骨体积分数。",
        )
    ).extract(document_id="doc-1", title="Paper", chunks=(_chunk(),))

    assert dossier.items[0].summary.startswith("该候选摘要未能安全翻译")
    assert dossier.items[0].risk_level == "high"
    assert any("unsafe translation" in warning for warning in warnings)


def test_category_filter_removes_captions_and_author_contributions() -> None:
    assert not _category_evidence_consistent(
        "characterization", "Fig. S1. Cross-sectional SEM image of the cell."
    )
    assert not _category_evidence_consistent(
        "characterization", "XRD measurements and analyses were performed by P.G."
    )
    assert not _category_evidence_consistent(
        "device_fabrication",
        "Top-view SEM images of films on TiO2/FTO with varying ratios.",
    )
    assert _category_evidence_consistent(
        "device_fabrication",
        "The electrode was fabricated by spin coating and annealed at 500 C.",
    )
