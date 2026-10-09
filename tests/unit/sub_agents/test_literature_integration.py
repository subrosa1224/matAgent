from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from materials_screening.sub_agents.literature.dossier import (
    DossierItem,
    PaperDossier,
)
from materials_screening.sub_agents.literature.integration import (
    LiteratureIntegrationService,
    _sanitize_dossier,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalDataRow,
    KnowledgeEdge,
    LiteratureDocumentMetadata,
    LiteratureSearchOutput,
    PaperProvenance,
    PaperRecord,
)
from materials_screening.sub_agents.literature.store import LiteratureQueryStore


class KnowledgeStore:
    def __init__(self, fact: ExperimentalDataRow, edge: KnowledgeEdge) -> None:
        self.fact = fact
        self.edge = edge

    def get_document_metadata(
        self, document_id: str
    ) -> LiteratureDocumentMetadata | None:
        return None

    def list_experimental_facts(
        self, *, status: str | None = None, document_id: str | None = None
    ) -> list[ExperimentalDataRow]:
        return [self.fact] if status == "approved" and document_id == "doc-1" else []

    def list_knowledge_edges(
        self, *, document_id: str | None = None
    ) -> list[KnowledgeEdge]:
        return [self.edge] if document_id == "doc-1" else []

    def load_approved_matrix(
        self, document_id: str
    ) -> tuple[list[object], list[object], list[object], list[object], list[object]]:
        return [], [], [], [], []


def test_assemble_strict_result_from_saved_and_approved_evidence(
    tmp_path: Path,
) -> None:
    now = datetime.now(UTC)
    query_store = LiteratureQueryStore(tmp_path)
    query_store.save(
        LiteratureSearchOutput(
            query_id="lit-abc123",
            provider="openalex",
            papers=(
                PaperRecord(
                    paper_id="W1",
                    title="CaP scaffold study",
                    doi="10.1/example",
                    year=2025,
                    abstract="A concise evidence-grounded abstract.",
                    provenance=(
                        PaperProvenance(
                            provider="openalex",
                            provider_id="W1",
                            retrieved_at=now,
                            raw_record_sha256="a" * 64,
                        ),
                    ),
                ),
            ),
            returned_count=1,
        )
    )
    fact = ExperimentalDataRow(
        fact_id="fact-1",
        document_id="doc-1",
        chunk_id="chunk-1",
        page_from=11,
        page_to=11,
        material="CaP scaffold",
        variable_name="porosity",
        variable_value="70 %",
        performance_metric="new bone formation",
        performance_value="highest rate",
        source_quote="porosity of 70 % had the highest rate",
        source_text_sha256="b" * 64,
        review_status="approved",
    )
    edge = KnowledgeEdge(
        edge_id="edge-1",
        fact_id="fact-1",
        document_id="doc-1",
        subject="CaP scaffold",
        predicate="has_parameter_performance_relation",
        object="porosity=70 % -> new bone formation=highest rate",
        variable_name="porosity",
        variable_value="70 %",
        performance_metric="new bone formation",
        performance_value="highest rate",
        chunk_id="chunk-1",
        page_from=11,
        page_to=11,
        source_quote=fact.source_quote,
        source_text_sha256=fact.source_text_sha256,
        created_at=now,
    )
    result = LiteratureIntegrationService(
        query_store, KnowledgeStore(fact, edge)
    ).assemble(query_ids=("lit-abc123",), document_id="doc-1")

    assert result.papers[0].key_findings == ("A concise evidence-grounded abstract.",)
    assert result.data_tables[0].rows == (fact,)
    assert result.kp_edges == (edge,)
    assert "1 approved legacy facts" in result.synthesis_summary
    assert result.warnings == ()


def test_report_dossier_filters_translation_placeholders_and_microscopy() -> None:
    dossier = PaperDossier(
        document_id="doc-1",
        title="Paper",
        extraction_method="llm",
        review_status="approved",
        items=(
            DossierItem(
                category="materials",
                summary="该候选摘要未能安全翻译，请人工核对。",
                source_quote="source",
                chunk_id="chunk-1",
                page=1,
            ),
            DossierItem(
                category="optical_result",
                summary="使用光学显微镜观察染色切片。",
                source_quote="source",
                chunk_id="chunk-2",
                page=2,
            ),
            DossierItem(
                category="performance_result",
                summary="支架促进新骨形成。",
                source_quote="source",
                chunk_id="chunk-3",
                page=3,
            ),
            DossierItem(
                category="materials",
                summary="颗粒的比表面积为-15.31 m²/kg。",
                source_quote="source",
                chunk_id="chunk-4",
                page=4,
            ),
        ),
    )

    sanitized = _sanitize_dossier(dossier)

    assert [item.category for item in sanitized.items] == ["performance_result"]
