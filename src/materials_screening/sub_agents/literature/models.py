"""Strict contracts for literature retrieval and synthesis inputs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .dossier import PaperDossier


class LiteratureSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    topic: str = Field(min_length=1, max_length=1000)
    research_question: str | None = Field(default=None, min_length=1, max_length=4000)
    material_keywords: tuple[str, ...] = Field(default=(), max_length=30)
    year_from: int | None = Field(default=None, ge=1900, le=2100)
    year_to: int | None = Field(default=None, ge=1900, le=2100)
    max_papers: int = Field(default=20, ge=1, le=100)
    sort_mode: Literal["balanced", "recent", "relevance"] = "balanced"

    @field_validator("topic")
    @classmethod
    def normalize_topic(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("topic must not be blank")
        return normalized

    @field_validator("material_keywords")
    @classmethod
    def normalize_keywords(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(" ".join(v.split()) for v in values if v.strip())
        )
        if any(len(value) > 100 for value in normalized):
            raise ValueError("material keywords must not exceed 100 characters")
        return normalized

    @model_validator(mode="after")
    def validate_years(self) -> Self:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from must not exceed year_to")
        return self


class ExpandedQuery(BaseModel):
    """Bounded, reproducible expansion of a user's literature topic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_topic: str
    research_question: str | None = None
    normalized_materials: tuple[str, ...] = Field(default=(), max_length=30)
    synonyms: tuple[str, ...] = Field(default=(), max_length=30)
    performance_terms: tuple[str, ...] = Field(default=(), max_length=20)
    process_terms: tuple[str, ...] = Field(default=(), max_length=20)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=6)
    expansion_version: str = "deterministic-v1"


class ProviderSearchStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openalex", "semantic_scholar"]
    status: Literal["ok", "degraded"]
    queries_attempted: int = Field(ge=0, le=6)
    records_returned: int = Field(ge=0)
    warning: str | None = None
    search_queries: tuple[str, ...] = Field(default=(), max_length=6)
    submitted_materials: tuple[str, ...] = Field(default=(), max_length=30)
    failed_materials: tuple[str, ...] = Field(default=(), max_length=30)
    unqueried_materials: tuple[str, ...] = Field(default=(), max_length=30)
    queries_at_record_limit: int = Field(default=0, ge=0, le=6)


class PaperProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openalex", "semantic_scholar"]
    provider_id: str
    retrieved_at: datetime
    raw_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paper_id: str
    title: str
    doi: str | None = None
    year: int | None = None
    authors: tuple[str, ...] = ()
    venue: str | None = None
    abstract: str | None = None
    cited_by_count: int | None = Field(default=None, ge=0)
    open_access: bool | None = None
    landing_page_url: str | None = None
    provenance: tuple[PaperProvenance, ...]
    selection_reason: str | None = None
    relevance_level: Literal["core", "high", "extended"] | None = None
    relevance_score: float | None = Field(default=None, ge=0)
    matched_concepts: tuple[str, ...] = ()
    missing_concepts: tuple[str, ...] = ()
    application_evidence_grade: Literal["A", "B", "C"] | None = None
    application_evidence_reason: str | None = None
    access_status: Literal[
        "metadata_only", "abstract_available", "open_access_reported"
    ] = "metadata_only"


class LiteratureSearchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    provider: Literal["openalex", "semantic_scholar", "unified"]
    papers: tuple[PaperRecord, ...]
    returned_count: int = Field(ge=0)
    expanded_query: ExpandedQuery | None = None
    provider_statuses: tuple[ProviderSearchStatus, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieval_mode: Literal["fresh", "cache", "mixed", "unknown"] = "unknown"


class CandidateLiteratureScreenInput(BaseModel):
    """Bounded application-evidence pre-screen for a larger material pool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    materials: tuple[str, ...] = Field(min_length=1, max_length=100)
    supplementary_materials: tuple[str, ...] = Field(default=(), max_length=50)
    application: str = Field(default="UV photodetector", min_length=1, max_length=100)
    final_limit: int = Field(default=5, ge=1, le=20)
    papers_per_candidate: int = Field(default=5, ge=1, le=10)

    @field_validator("materials")
    @classmethod
    def normalize_materials(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys("".join(value.split()) for value in values if value.strip())
        )
        if not normalized:
            raise ValueError("materials must not be empty")
        if any(len(value) > 100 for value in normalized):
            raise ValueError("material formulas must not exceed 100 characters")
        return normalized

    @model_validator(mode="after")
    def validate_pool_membership(self) -> Self:
        if not set(self.supplementary_materials).issubset(self.materials):
            raise ValueError(
                "supplementary materials must belong to the candidate pool"
            )
        return self


class CandidateEvidenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    formula: str
    original_rank: int = Field(ge=1)
    evidence_grade: Literal["A", "B", "C", "NONE"]
    pool: Literal["strict", "supplementary"] = "strict"
    retrieval_status: Literal["ok", "error", "not_attempted"] = "ok"
    matching_paper_count: int = Field(ge=0)
    source_query_id: str | None = None
    source_created_at: datetime | None = None
    papers: tuple[PaperRecord, ...] = Field(default=(), max_length=10)
    note: str


class CandidateLiteratureScreenOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    screening_id: str = Field(pattern=r"^lit-screen-[0-9a-f]{24}$")
    application: str
    candidate_count: int = Field(ge=1, le=100)
    qualifying_candidate_count: int = Field(ge=0, le=100)
    candidates: tuple[CandidateEvidenceSummary, ...] = Field(max_length=100)
    finalists: tuple[CandidateEvidenceSummary, ...] = Field(max_length=20)
    supplementary_finalists: tuple[CandidateEvidenceSummary, ...] = Field(
        default=(), max_length=20
    )
    download_candidates: tuple[PaperRecord, ...] = Field(max_length=200)
    provider: str
    queries_attempted: int = Field(ge=0, le=200)
    retrieval_complete: bool = True
    final_limit: int = Field(default=5, ge=1, le=20)
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    retrieval_mode: Literal["fresh", "cache", "mixed", "unknown"] = "unknown"
    cache_hits: int = Field(default=0, ge=0, le=200)


class IngestDocumentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    paths: tuple[str, ...] = Field(min_length=1, max_length=20)
    paper_id: str | None = Field(default=None, max_length=128)

    @field_validator("paths")
    @classmethod
    def validate_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            candidate = value.strip()
            if not candidate:
                raise ValueError("document path must not be blank")
            if "\x00" in candidate or Path(candidate).suffix.lower() != ".pdf":
                raise ValueError("only PDF paths are accepted")
            normalized.append(candidate)
        return tuple(dict.fromkeys(normalized))


class IngestedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    paper_id: str | None = None
    file_name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    embedding_model: str
    status: Literal["indexed", "already_indexed"]


class IngestDocumentsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: tuple[IngestedDocument, ...]
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None


class RagRetrieveInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=2000)
    paper_ids: tuple[str, ...] = Field(default=(), max_length=50)
    top_n: int = Field(default=50, ge=1, le=200)
    top_k: int = Field(default=8, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.top_k > self.top_n:
            raise ValueError("top_k must not exceed top_n")
        return self


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str
    paper_id: str | None = None
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    text: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vector_distance: float = Field(ge=0)
    rerank_score: float | None = None


class RagRetrieveOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    chunks: tuple[RetrievedChunk, ...]
    returned_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None


class ExperimentalDataCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1, max_length=128)
    material: str = Field(min_length=1, max_length=300)
    variable_name: str = Field(min_length=1, max_length=200)
    variable_value: str = Field(min_length=1, max_length=200)
    performance_metric: str = Field(min_length=1, max_length=200)
    performance_value: str = Field(min_length=1, max_length=200)
    conditions: str | None = Field(default=None, max_length=1000)
    source_quote: str = Field(min_length=1, max_length=1500)


class ExtractExperimentalDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1, max_length=128)
    rows: tuple[ExperimentalDataCandidate, ...] = Field(min_length=1, max_length=100)


class ExperimentalDataRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str
    document_id: str
    paper_id: str | None = None
    chunk_id: str
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    material: str
    variable_name: str
    variable_value: str
    performance_metric: str
    performance_value: str
    conditions: str | None = None
    source_quote: str
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    llm_extracted: Literal[True] = True
    review_status: Literal["pending", "approved", "rejected"] = "pending"


class ExtractExperimentalDataOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    rows: tuple[ExperimentalDataRow, ...]
    rejected_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None


class ExperimentalFactReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    fact_id: str
    previous_status: Literal["pending", "approved", "rejected"]
    decision: Literal["approved", "rejected"]
    reviewer: str
    reason: str | None = None
    reviewed_at: datetime


class KnowledgeEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    edge_id: str
    fact_id: str
    document_id: str
    paper_id: str | None = None
    subject: str
    predicate: Literal["has_parameter_performance_relation"]
    object: str
    variable_name: str
    variable_value: str
    performance_metric: str
    performance_value: str
    conditions: str | None = None
    chunk_id: str
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    source_quote: str
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_status: Literal["approved"] = "approved"
    created_at: datetime


class AssembleLiteratureResultInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query_ids: tuple[str, ...] = Field(default=(), max_length=20)
    document_id: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def require_evidence_source(self) -> Self:
        if not self.query_ids and self.document_id is None:
            raise ValueError("at least one query_id or document_id is required")
        return self


class LiteraturePaperSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    doi: str | None = None
    year: int | None = None
    key_findings: tuple[str, ...] = ()


class LiteratureDocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    file_name: str
    paper_id: str | None = None
    title: str | None = None
    doi: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100)


class ExperimentalDataTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    document_id: str
    rows: tuple[ExperimentalDataRow, ...]


class ExperimentMatrixDataTable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    document_id: str
    groups: tuple[ExperimentalGroup, ...]
    measurements: tuple[ExperimentalMeasurement, ...]
    comparisons: tuple[ExperimentalComparison, ...]
    claims: tuple[PaperClaim, ...]
    claim_evidence_links: tuple[ClaimEvidenceLink, ...]


class LiteratureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    papers: tuple[LiteraturePaperSummary, ...]
    synthesis_summary: str
    data_tables: tuple[ExperimentalDataTable | ExperimentMatrixDataTable, ...]
    kp_edges: tuple[KnowledgeEdge, ...]
    paper_dossiers: tuple[PaperDossier, ...] = ()
    warnings: tuple[str, ...] = ()
    evidence_id: str | None = None


ReviewStatus = Literal["pending", "approved", "rejected"]


class ExperimentalGroup(BaseModel):
    """One real sample or experimental/control group reported by a paper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    group_id: str
    document_id: str
    label: str
    role: Literal["control", "treatment", "reference", "unknown"]
    material: str
    variables: dict[str, str] = Field(default_factory=dict)
    conditions: dict[str, str] = Field(default_factory=dict)
    source_quote: str
    chunk_id: str
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    llm_extracted: bool = True
    extraction_method: Literal["llm", "table_parser", "manual"] = "llm"
    review_status: ReviewStatus = "pending"


class ExperimentalMeasurement(BaseModel):
    """One measured outcome attached to an experimental group."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_id: str
    group_id: str
    document_id: str
    metric: str
    value_text: str
    numeric_value: float | None = None
    unit: str | None = None
    uncertainty_text: str | None = None
    sample_size: int | None = Field(default=None, ge=1)
    source_quote: str
    chunk_id: str
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    llm_extracted: bool = True
    extraction_method: Literal["llm", "table_parser", "manual"] = "llm"
    review_status: ReviewStatus = "pending"


class ExperimentalComparison(BaseModel):
    """Deterministically computed or author-reported group comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: str
    document_id: str
    baseline_group_id: str
    target_group_id: str
    metric: str
    baseline_measurement_id: str | None = None
    target_measurement_id: str | None = None
    baseline_value: float | None = None
    target_value: float | None = None
    absolute_change: float | None = None
    relative_change_percent: float | None = None
    direction: Literal["increase", "decrease", "unchanged", "not_computable"]
    provenance_type: Literal["reported", "calculated"]
    reported_text: str | None = None
    unit: str | None = None
    review_status: ReviewStatus = "pending"


class PaperClaim(BaseModel):
    """An author claim from abstract, conclusion, or results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    document_id: str
    claim_text: str
    source_section: Literal["abstract", "conclusion", "results", "discussion"]
    source_quote: str
    chunk_id: str
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)
    source_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    llm_extracted: Literal[True] = True
    review_status: ReviewStatus = "pending"


class ClaimEvidenceLink(BaseModel):
    """Assessment linking an author claim to body/table evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    link_id: str
    claim_id: str
    document_id: str
    evidence_type: Literal["group", "measurement", "comparison", "text_chunk"]
    evidence_id: str
    assessment: Literal[
        "supported",
        "partially_supported",
        "unsupported",
        "contradicted",
        "not_verifiable",
    ]
    explanation: str
    assessment_method: Literal["rule", "llm", "human"]
    review_status: ReviewStatus = "pending"


class MatrixReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    decision: Literal["approved", "rejected"]
    reviewer: str
    reason: str | None = None
    reviewed_counts: dict[str, int]
    review_event_ids: tuple[str, ...]
    reviewed_at: datetime
