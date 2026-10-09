"""Deterministic assembly of the LiteratureAgent knowledge package."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from .dossier import PaperDossier
from .models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalDataRow,
    ExperimentalDataTable,
    ExperimentalGroup,
    ExperimentalMeasurement,
    ExperimentMatrixDataTable,
    KnowledgeEdge,
    LiteratureDocumentMetadata,
    LiteraturePaperSummary,
    LiteratureResult,
    PaperClaim,
    PaperRecord,
)
from .store import LiteratureQueryStore


class KnowledgeResultStore(Protocol):
    def get_document_metadata(
        self, document_id: str
    ) -> LiteratureDocumentMetadata | None: ...

    def list_experimental_facts(
        self, *, status: str | None = None, document_id: str | None = None
    ) -> list[ExperimentalDataRow]: ...

    def list_knowledge_edges(
        self, *, document_id: str | None = None
    ) -> list[KnowledgeEdge]: ...

    def load_approved_matrix(
        self, document_id: str
    ) -> tuple[
        list[ExperimentalGroup],
        list[ExperimentalMeasurement],
        list[ExperimentalComparison],
        list[PaperClaim],
        list[ClaimEvidenceLink],
    ]: ...


class DossierResultStore(Protocol):
    def load(self, document_id: str) -> PaperDossier | None: ...


class LiteratureIntegrationService:
    def __init__(
        self,
        query_store: LiteratureQueryStore,
        knowledge_store: KnowledgeResultStore,
        dossier_store: DossierResultStore | None = None,
    ) -> None:
        self.query_store = query_store
        self.knowledge_store = knowledge_store
        self.dossier_store = dossier_store

    def assemble(
        self, *, query_ids: Sequence[str], document_id: str | None
    ) -> LiteratureResult:
        papers = self._load_papers(query_ids)
        facts = self.knowledge_store.list_experimental_facts(
            status="approved", document_id=document_id
        )
        edges = self.knowledge_store.list_knowledge_edges(document_id=document_id)
        matrix = (
            self.knowledge_store.load_approved_matrix(document_id)
            if document_id is not None
            else ([], [], [], [], [])
        )
        groups, measurements, comparisons, claims, claim_links = matrix
        dossier = (
            self.dossier_store.load(document_id)
            if self.dossier_store is not None and document_id is not None
            else None
        )
        approved_dossiers: tuple[PaperDossier, ...] = (
            (sanitize_dossier_for_report(dossier),)
            if dossier is not None and dossier.review_status == "approved"
            else ()
        )
        edges.extend(
            _matrix_knowledge_edges(
                document_id=document_id,
                groups=groups,
                measurements=measurements,
                comparisons=comparisons,
                links=claim_links,
            )
        )
        warnings: list[str] = []
        if query_ids and not papers:
            warnings.append("No papers were present in the requested query snapshots.")
        if document_id and not facts and not measurements and not approved_dossiers:
            warnings.append("No human-approved experimental evidence was found.")
        if facts and not edges:
            warnings.append(
                "Approved facts exist but knowledge edges have not been built."
            )
        if document_id is not None and measurements:
            tables: tuple[ExperimentalDataTable | ExperimentMatrixDataTable, ...] = (
                ExperimentMatrixDataTable(
                    title="Human-approved experiment matrix",
                    document_id=document_id,
                    groups=tuple(groups),
                    measurements=tuple(measurements),
                    comparisons=tuple(comparisons),
                    claims=tuple(claims),
                    claim_evidence_links=tuple(claim_links),
                ),
            )
        elif document_id is not None and facts:
            tables = (
                ExperimentalDataTable(
                    title="Human-approved experimental data",
                    document_id=document_id,
                    rows=tuple(facts),
                ),
            )
        else:
            tables = ()
        paper_summaries = [_paper_summary(paper) for paper in papers]
        metadata = (
            self.knowledge_store.get_document_metadata(document_id)
            if document_id is not None
            else None
        )
        if metadata is not None and metadata.title and approved_dossiers:
            approved_dossiers = tuple(
                item.model_copy(update={"title": metadata.title})
                for item in approved_dossiers
            )
        if metadata is not None and not _metadata_already_present(metadata, papers):
            findings = tuple(
                f"{edge.variable_name}={edge.variable_value} -> "
                f"{edge.performance_metric}={edge.performance_value}"
                for edge in edges
            )
            paper_summaries.append(
                LiteraturePaperSummary(
                    title=(
                        metadata.title
                        or (
                            approved_dossiers[0].title
                            if approved_dossiers
                            else metadata.file_name.removesuffix(".pdf")
                        )
                    ),
                    doi=metadata.doi,
                    year=metadata.year,
                    key_findings=findings,
                )
            )
        return LiteratureResult(
            papers=tuple(paper_summaries),
            synthesis_summary=_synthesis(
                paper_summaries,
                facts,
                measurements,
                comparisons,
                edges,
                approved_dossiers,
            ),
            data_tables=tables,
            kp_edges=tuple(edges),
            paper_dossiers=approved_dossiers,
            warnings=tuple(warnings),
        )

    def assemble_many(
        self, *, query_ids: Sequence[str], document_ids: Sequence[str]
    ) -> LiteratureResult:
        """Assemble a topic-level package from multiple reviewed documents."""
        paper_summaries = [
            _paper_summary(paper) for paper in self._load_papers(query_ids)
        ]
        tables: list[ExperimentalDataTable | ExperimentMatrixDataTable] = []
        edges: list[KnowledgeEdge] = []
        dossiers: list[PaperDossier] = []
        warnings: list[str] = []
        for document_id in dict.fromkeys(document_ids):
            result = self.assemble(query_ids=(), document_id=document_id)
            paper_summaries.extend(result.papers)
            tables.extend(result.data_tables)
            edges.extend(result.kp_edges)
            dossiers.extend(result.paper_dossiers)
            warnings.extend(f"{document_id}: {warning}" for warning in result.warnings)
        unique_papers: list[LiteraturePaperSummary] = []
        seen_papers: set[tuple[str | None, str, int | None]] = set()
        for paper in paper_summaries:
            key = (paper.doi, paper.title, paper.year)
            if key not in seen_papers:
                seen_papers.add(key)
                unique_papers.append(paper)
        if query_ids and not paper_summaries:
            warnings.append("No papers were present in the requested query snapshots.")
        synthesis = (
            f"Integrated {len(unique_papers)} papers into {len(tables)} approved "
            f"evidence tables, {len(dossiers)} approved paper dossiers, and "
            f"{len(edges)} provenance-bound knowledge edges."
        )
        if edges:
            relations = "; ".join(f"{edge.subject}: {edge.object}" for edge in edges)
            synthesis += f" Approved evidence indicates: {relations}."
        return LiteratureResult(
            papers=tuple(unique_papers),
            synthesis_summary=synthesis,
            data_tables=tuple(tables),
            kp_edges=tuple(edges),
            paper_dossiers=tuple(dossiers),
            warnings=tuple(warnings),
        )

    def _load_papers(self, query_ids: Sequence[str]) -> list[PaperRecord]:
        papers: list[PaperRecord] = []
        seen: set[str] = set()
        for query_id in dict.fromkeys(query_ids):
            result = self.query_store.load(query_id)
            for paper in result.papers:
                key = f"doi:{paper.doi}" if paper.doi else f"id:{paper.paper_id}"
                if key not in seen:
                    seen.add(key)
                    papers.append(paper)
        return papers


def _matrix_knowledge_edges(
    *,
    document_id: str | None,
    groups: Sequence[ExperimentalGroup],
    measurements: Sequence[ExperimentalMeasurement],
    comparisons: Sequence[ExperimentalComparison],
    links: Sequence[ClaimEvidenceLink],
) -> list[KnowledgeEdge]:
    """Project approved, supported matrix claims into provenance-bound edges."""
    if document_id is None:
        return []
    groups_by_id = {row.group_id: row for row in groups}
    measurements_by_id = {row.measurement_id: row for row in measurements}
    comparisons_by_id = {row.comparison_id: row for row in comparisons}
    edges: list[KnowledgeEdge] = []
    for link in links:
        if link.assessment != "supported":
            continue
        comparison = (
            comparisons_by_id.get(link.evidence_id)
            if link.evidence_type == "comparison"
            else None
        )
        target = (
            measurements_by_id.get(comparison.target_measurement_id)
            if comparison is not None and comparison.target_measurement_id is not None
            else measurements_by_id.get(link.evidence_id)
            if link.evidence_type == "measurement"
            else None
        )
        group = groups_by_id.get(target.group_id) if target is not None else None
        if target is None or group is None:
            continue
        variable_name, variable_value = (
            next(iter(group.variables.items()))
            if group.variables
            else ("experimental_group", group.label)
        )
        metric = comparison.metric if comparison is not None else target.metric
        unit = (comparison.unit if comparison is not None else target.unit) or ""
        target_value = (
            comparison.target_value
            if comparison is not None
            else target.numeric_value
            if target.numeric_value is not None
            else target.value_text
        )
        performance_value = f"{target_value}{unit}"
        if comparison is not None:
            baseline_value = f"{comparison.baseline_value}{unit}"
            relation_value = f"{baseline_value} to {performance_value}"
        else:
            relation_value = performance_value
        object_value = (
            f"{variable_name}={variable_value} -> {metric}={relation_value}"
        )
        conditions = "; ".join(
            f"{key}={value}" for key, value in group.conditions.items()
        )
        edge_id = "edge-" + hashlib.sha256(link.link_id.encode()).hexdigest()[:24]
        edges.append(
            KnowledgeEdge(
                edge_id=edge_id,
                fact_id=link.link_id,
                document_id=document_id,
                subject=group.material,
                predicate="has_parameter_performance_relation",
                object=object_value,
                variable_name=variable_name,
                variable_value=variable_value,
                performance_metric=metric,
                performance_value=performance_value,
                conditions=conditions or None,
                chunk_id=target.chunk_id,
                page_from=target.page_from,
                page_to=target.page_to,
                source_quote=target.source_quote,
                source_text_sha256=target.source_text_sha256,
                created_at=datetime.now(UTC),
            )
        )
    return edges


def _paper_summary(paper: PaperRecord) -> LiteraturePaperSummary:
    findings = (paper.abstract[:600],) if paper.abstract else ()
    return LiteraturePaperSummary(
        title=paper.title,
        doi=paper.doi,
        year=paper.year,
        key_findings=findings,
    )


def sanitize_dossier_for_report(dossier: PaperDossier) -> PaperDossier:
    """Remove non-deliverable placeholders and obvious category mismatches."""
    safe_items = tuple(item for item in dossier.items if _report_safe_item(item))
    return dossier.model_copy(update={"items": safe_items})


_sanitize_dossier = sanitize_dossier_for_report


def _report_safe_item(item: object) -> bool:
    summary = str(getattr(item, "summary", ""))
    if "未能安全翻译" in summary:
        return False
    if "比表面积" in summary and re.search(r"(?<!\d)-\d+(?:\.\d+)?", summary):
        return False
    if getattr(item, "category", None) != "optical_result":
        return True
    lowered = summary.casefold()
    microscopy_markers = (
        "光学显微镜",
        "optical microscope",
        "染色",
        "显微视野",
    )
    actual_optical_markers = (
        "带隙",
        "吸收",
        "透射",
        "反射",
        "荧光光谱",
        "bandgap",
        "absorption",
        "photoluminescence",
    )
    return not (
        any(marker in lowered for marker in microscopy_markers)
        and not any(marker in lowered for marker in actual_optical_markers)
    )


def _metadata_already_present(
    metadata: LiteratureDocumentMetadata, papers: Sequence[PaperRecord]
) -> bool:
    return bool(
        (metadata.doi and any(paper.doi == metadata.doi for paper in papers))
        or (
            metadata.paper_id
            and any(paper.paper_id == metadata.paper_id for paper in papers)
        )
    )


def _synthesis(
    papers: Sequence[object],
    facts: Sequence[ExperimentalDataRow],
    measurements: Sequence[ExperimentalMeasurement],
    comparisons: Sequence[ExperimentalComparison],
    edges: Sequence[KnowledgeEdge],
    dossiers: Sequence[PaperDossier],
) -> str:
    parts = [
        f"Integrated {len(papers)} papers, {len(facts)} approved legacy facts, "
        f"{len(measurements)} approved measurements, {len(comparisons)} "
        f"group comparisons, {len(dossiers)} approved paper dossiers, and "
        f"{len(edges)} provenance-bound knowledge edges."
    ]
    if edges:
        relations = "; ".join(f"{edge.subject}: {edge.object}" for edge in edges)
        parts.append(f"Approved evidence indicates: {relations}.")
    return " ".join(parts)
