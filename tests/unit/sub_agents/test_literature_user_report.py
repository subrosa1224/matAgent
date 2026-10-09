from __future__ import annotations

from pathlib import Path
from typing import Any

from materials_screening.llm.base import StructuredProviderResponse
from materials_screening.sub_agents.literature.dossier import (
    DossierItem,
    PaperDossier,
    PaperDossierStore,
)
from materials_screening.sub_agents.literature.models import (
    LiteratureDocumentMetadata,
)
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReportService,
    LiteratureUserReportStore,
    LiteratureUserReportSynthesizer,
    UserReportNarrative,
)


class MetadataStore:
    def get_document_metadata(
        self, document_id: str
    ) -> LiteratureDocumentMetadata | None:
        return LiteratureDocumentMetadata(
            document_id=document_id,
            file_name="paper.pdf",
            title="Formal paper title",
            doi="10.1/example",
            year=2025,
        )


class NarrativeLlm:
    def __init__(self, narrative: UserReportNarrative) -> None:
        self.narrative = narrative
        self.calls = 0

    def generate_structured(self, **_: Any) -> StructuredProviderResponse[Any]:
        self.calls += 1
        return StructuredProviderResponse(
            parsed=self.narrative,
            provider="fake",
            model="fake",
            request_id="narrative-1",
            latency_ms=0,
            input_tokens=1,
            output_tokens=1,
            reasoning_tokens=0,
            raw_output_sha256="a" * 64,
        )


def _item(
    category: str, summary: str, *, risk: str, page: int
) -> DossierItem:
    return DossierItem.model_validate(
        {
            "category": category,
            "summary": summary,
            "source_quote": f"Verbatim evidence for {summary}",
            "chunk_id": f"chunk-{page}",
            "page": page,
            "risk_level": risk,
        }
    )


def test_user_report_uses_pending_dossier_and_filters_risk(tmp_path: Path) -> None:
    pending = PaperDossierStore(tmp_path / "pending")
    approved = PaperDossierStore(tmp_path / "approved")
    pending.save_pending(
        PaperDossier(
            document_id="doc-1",
            title="paper-file-name",
            extraction_method="llm",
            items=(
                _item(
                    "research_problem",
                    "研究多孔结构对成骨的影响。",
                    risk="low",
                    page=1,
                ),
                _item(
                    "performance_result",
                    "支架表现出更好的成骨性能。",
                    risk="medium",
                    page=6,
                ),
                _item(
                    "mechanism",
                    "未经可靠证据支持的机理。",
                    risk="high",
                    page=7,
                ),
                _item(
                    "materials",
                    "该候选摘要未能安全翻译，请人工核对。",
                    risk="high",
                    page=8,
                ),
            ),
        )
    )
    report = LiteratureUserReportService(
        metadata_store=MetadataStore(),
        pending_store=pending,
        approved_store=approved,
    ).build(topic="多孔支架成骨", document_ids=("doc-1",))

    paper = report.papers[0]
    assert paper.title == "Formal paper title"
    assert paper.dossier_status == "pending"
    assert paper.low_risk_count == 1
    assert paper.medium_risk_count == 1
    assert paper.excluded_high_risk_count == 1
    assert paper.excluded_unsafe_count == 1
    assert [item.display_status for item in paper.evidence] == [
        "reliable",
        "check_recommended",
    ]


def test_user_report_store_persists_json(tmp_path: Path) -> None:
    pending = PaperDossierStore(tmp_path / "pending")
    pending.save_pending(
        PaperDossier(
            document_id="doc-1",
            title="Paper",
            extraction_method="llm",
            items=(
                _item(
                    "research_problem",
                    "研究问题。",
                    risk="low",
                    page=1,
                ),
            ),
        )
    )
    report = LiteratureUserReportService(
        metadata_store=MetadataStore(),
        pending_store=pending,
        approved_store=PaperDossierStore(tmp_path / "approved"),
    ).build(topic="主题", document_ids=("doc-1",))

    target = LiteratureUserReportStore(tmp_path / "reports").save(report)

    assert target.exists()
    assert "Formal paper title" in target.read_text(encoding="utf-8")


def test_user_report_synthesizes_once_with_grounded_citations(
    tmp_path: Path,
) -> None:
    pending = PaperDossierStore(tmp_path / "pending")
    pending.save_pending(
        PaperDossier(
            document_id="doc-1",
            title="Paper",
            extraction_method="llm",
            items=(
                _item(
                    "research_problem",
                    "研究支架结构对成骨的影响。",
                    risk="low",
                    page=1,
                ),
            ),
        )
    )
    report = LiteratureUserReportService(
        metadata_store=MetadataStore(),
        pending_store=pending,
        approved_store=PaperDossierStore(tmp_path / "approved"),
    ).build(topic="多孔支架成骨", document_ids=("doc-1",))
    evidence_id = report.papers[0].evidence[0].evidence_id
    llm = NarrativeLlm(
        UserReportNarrative(
            markdown=(
                "## 主题概述\n该论文研究支架结构对成骨的影响。"
                f"[{evidence_id}]\n"
                "## 逐篇解读\n研究聚焦支架结构与成骨。"
                f"[{evidence_id}]\n"
                "## 共同结论\n结构是重要因素。"
                f"[{evidence_id}]\n"
                "## 关键差异\n目前仅纳入一篇论文。"
                f"[{evidence_id}]\n"
                "## 设计启示\n应关注支架结构。"
                f"[{evidence_id}]\n"
                "## 适用边界\n结论证据范围有限。"
                f"[{evidence_id}]"
            )
        )
    )

    synthesized = LiteratureUserReportSynthesizer(llm).synthesize(report)

    assert llm.calls == 1
    assert synthesized.narrative is not None
    assert evidence_id in synthesized.narrative.markdown


def test_user_report_drops_sentence_with_number_without_cited_evidence(
    tmp_path: Path,
) -> None:
    pending = PaperDossierStore(tmp_path / "pending")
    pending.save_pending(
        PaperDossier(
            document_id="doc-1",
            title="Paper",
            extraction_method="llm",
            items=(
                _item(
                    "research_problem",
                    "研究支架结构对成骨的影响。",
                    risk="low",
                    page=1,
                ),
            ),
        )
    )
    report = LiteratureUserReportService(
        metadata_store=MetadataStore(),
        pending_store=pending,
        approved_store=PaperDossierStore(tmp_path / "approved"),
    ).build(topic="多孔支架成骨", document_ids=("doc-1",))
    evidence_id = report.papers[0].evidence[0].evidence_id
    llm = NarrativeLlm(
        UserReportNarrative(
            markdown=(
                "## 主题概述\n该材料性能提高了99%。"
                f"[{evidence_id}]\n"
                "## 逐篇解读\n研究支架结构对成骨的影响。"
                f"[{evidence_id}]\n"
                "## 共同结论\n结构是重要因素。"
                f"[{evidence_id}]\n"
                "## 关键差异\n目前仅纳入一篇论文。"
                f"[{evidence_id}]\n"
                "## 设计启示\n应关注支架结构。"
                f"[{evidence_id}]\n"
                "## 适用边界\n结论证据范围有限。"
                f"[{evidence_id}]"
            )
        )
    )

    synthesized = LiteratureUserReportSynthesizer(llm).synthesize(report)

    assert synthesized.narrative is not None
    assert "99" not in synthesized.narrative.markdown
    assert "研究支架结构" in synthesized.narrative.markdown
