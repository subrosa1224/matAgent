"""Immutable candidate, decision, claim, and evidence ledger for RA-2."""

from __future__ import annotations

import math
import os
import re
import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from materials_screening.research.errors import (
    ProjectConflictError,
    ResearchProjectError,
)
from materials_screening.research.material_identity import (
    IdentityUse,
    MaterialIdentityAssessmentRecord,
)
from materials_screening.research.models import SourceRole
from materials_screening.research.property_registry import (
    PropertyRegistry,
    default_property_registry,
)

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$"
_SHA256 = r"^(?:sha256:)?[0-9a-f]{64}$"


class CandidateSourceKind(StrEnum):
    MATERIALS_DATABASE = "materials_database"
    USER_DATA = "user_data"


class HardConstraintStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTED = "conflicted"
    NOT_COMPARABLE = "not_comparable"
    INSUFFICIENT = "insufficient"


class CandidateTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    UNRANKED = "unranked"


class EvidenceSourceKind(StrEnum):
    MATERIALS_DATABASE = "materials_database"
    LITERATURE_FULL_TEXT = "literature_full_text"
    LITERATURE_ABSTRACT = "literature_abstract"
    USER_EXPERIMENT = "user_experiment"
    DATA_ANALYSIS = "data_analysis"


class EvidenceReviewStatus(StrEnum):
    VALIDATED = "validated"
    HUMAN_REVIEWED = "human_reviewed"
    PENDING = "pending"
    REJECTED = "rejected"


class ClaimKind(StrEnum):
    PROPERTY_VALUE = "property_value"
    PHASE_IDENTITY = "phase_identity"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    LIMITATION = "limitation"
    INTERPRETATION = "interpretation"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT_ONLY = "context_only"


class EvidenceLocator(BaseModel):
    """Structured locator; requirements depend on source kind."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    source_material_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    database_version: str | None = Field(default=None, max_length=100)
    document_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    paper_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    doi: str | None = Field(default=None, max_length=300)
    chunk_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    page_from: int | None = Field(default=None, ge=1)
    page_to: int | None = Field(default=None, ge=1)
    dataset_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    analysis_id: str | None = Field(default=None, pattern=_ID_PATTERN)

    @model_validator(mode="after")
    def page_range_is_ordered(self) -> Self:
        if (self.page_from is None) != (self.page_to is None):
            raise ValueError("page locator requires both page_from and page_to")
        if (
            self.page_from is not None
            and self.page_to is not None
            and self.page_from > self.page_to
        ):
            raise ValueError("page_from cannot exceed page_to")
        return self


class PropertyValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    property_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,127}$")
    value: float | int | str | bool
    unit: str | None = Field(default=None, max_length=64)
    source_role: SourceRole
    evidence_id: str = Field(pattern=_ID_PATTERN)
    method: str | None = Field(default=None, max_length=500)
    conditions: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value")
    @classmethod
    def value_is_finite(
        cls, value: float | int | str | bool
    ) -> float | int | str | bool:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("property value must be finite")
        return value

    @field_validator("conditions")
    @classmethod
    def conditions_are_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json(value)
        return value


class MaterialCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=_ID_PATTERN)
    project_id: str = Field(pattern=_ID_PATTERN)
    source_kind: CandidateSourceKind
    source_material_id: str = Field(pattern=_ID_PATTERN)
    formula: str = Field(min_length=1, max_length=300)
    chemsys: str = Field(min_length=1, max_length=300)
    structure_fingerprint: str | None = Field(default=None, pattern=_SHA256)
    space_group_number: int | None = Field(default=None, ge=1, le=230)
    property_values: tuple[PropertyValue, ...] = Field(default=(), max_length=100)
    provenance: dict[str, Any]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("provenance")
    @classmethod
    def provenance_is_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_json(value)
        return value

    @model_validator(mode="after")
    def property_values_are_unique(self) -> Self:
        ids = [item.property_id for item in self.property_values]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate property ids must be unique")
        return self


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    decision_id: str = Field(pattern=_ID_PATTERN)
    project_id: str = Field(pattern=_ID_PATTERN)
    candidate_id: str = Field(pattern=_ID_PATTERN)
    hard_constraint_status: HardConstraintStatus
    exclusion_reasons: tuple[str, ...] = Field(default=(), max_length=50)
    ranking_score: float | None = None
    ranking_components: dict[str, float] = Field(default_factory=dict)
    evidence_status: EvidenceStatus = EvidenceStatus.INSUFFICIENT
    final_tier: CandidateTier = CandidateTier.UNRANKED
    rationale: tuple[str, ...] = Field(default=(), max_length=50)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=200)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def decision_is_consistent(self) -> Self:
        if self.hard_constraint_status is HardConstraintStatus.FAILED:
            if not self.exclusion_reasons:
                raise ValueError("failed candidate requires an exclusion reason")
            if self.final_tier not in {CandidateTier.D, CandidateTier.UNRANKED}:
                raise ValueError("failed candidate cannot receive tier A, B, or C")
        numbers = [*self.ranking_components.values()]
        if self.ranking_score is not None:
            numbers.append(self.ranking_score)
        if any(not math.isfinite(value) for value in numbers):
            raise ValueError("ranking numbers must be finite")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("decision evidence ids must be unique")
        return self


class ScreeningEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=_ID_PATTERN)
    project_id: str = Field(pattern=_ID_PATTERN)
    candidate_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    source_kind: EvidenceSourceKind
    source_id: str = Field(pattern=_ID_PATTERN)
    locator: EvidenceLocator
    property_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$"
    )
    value: float | int | str | bool | None = None
    unit: str | None = Field(default=None, max_length=64)
    method: str | None = Field(default=None, max_length=500)
    conditions: dict[str, Any] = Field(default_factory=dict)
    source_excerpt: str | None = Field(default=None, max_length=4000)
    source_text_sha256: str | None = Field(default=None, pattern=_SHA256)
    review_status: EvidenceReviewStatus
    limitations: tuple[str, ...] = Field(default=(), max_length=50)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("conditions")
    @classmethod
    def evidence_conditions_are_json_safe(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        _validate_json(value)
        return value

    @model_validator(mode="after")
    def validate_source_and_value(self) -> Self:
        _validate_locator(self.source_kind, self.locator)
        if self.property_id is None and (
            self.value is not None or self.unit is not None
        ):
            raise ValueError("value and unit require property_id")
        if self.property_id is not None and self.value is None:
            raise ValueError("property evidence requires a value")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("evidence value must be finite")
        if self.source_kind in {
            EvidenceSourceKind.LITERATURE_FULL_TEXT,
            EvidenceSourceKind.LITERATURE_ABSTRACT,
        } and not self.source_excerpt:
            raise ValueError("literature evidence requires a source excerpt")
        if (
            self.source_kind is EvidenceSourceKind.LITERATURE_FULL_TEXT
            and self.source_text_sha256 is None
        ):
            raise ValueError("full-text evidence requires source text hash")
        return self


class ScreeningClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=_ID_PATTERN)
    project_id: str = Field(pattern=_ID_PATTERN)
    candidate_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    kind: ClaimKind
    statement: str = Field(min_length=1, max_length=4000)
    property_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def property_claim_names_property(self) -> Self:
        if self.kind is ClaimKind.PROPERTY_VALUE and self.property_id is None:
            raise ValueError("property-value claim requires property_id")
        return self


class ClaimEvidenceLink(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    link_id: str = Field(pattern=_ID_PATTERN)
    project_id: str = Field(pattern=_ID_PATTERN)
    claim_id: str = Field(pattern=_ID_PATTERN)
    evidence_id: str = Field(pattern=_ID_PATTERN)
    stance: EvidenceStance
    identity_assessment_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    explanation: str = Field(min_length=1, max_length=2000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


LedgerRecord = (
    MaterialCandidate
    | CandidateDecision
    | ScreeningEvidence
    | ScreeningClaim
    | ClaimEvidenceLink
    | MaterialIdentityAssessmentRecord
)


class CandidateEvidenceLedger:
    """Append-only store that validates every cross-record reference."""

    def __init__(
        self, root: Path, *, registry: PropertyRegistry | None = None
    ) -> None:
        self._root = root.resolve()
        self._registry = registry or default_property_registry()
        self._lock = threading.RLock()

    def add_evidence(self, evidence: ScreeningEvidence) -> ScreeningEvidence:
        if (
            evidence.property_id is not None
            and self._registry.resolve(evidence.property_id) is None
        ):
            raise ResearchProjectError(
                "UNREGISTERED_PROPERTY_EVIDENCE",
                "unregistered property cannot enter the evidence ledger",
            )
        if evidence.candidate_id is not None:
            candidate = self.get_candidate(evidence.project_id, evidence.candidate_id)
            if candidate.project_id != evidence.project_id:
                raise ResearchProjectError(
                    "PROJECT_MISMATCH", "candidate project mismatch"
                )
        self._write(evidence.project_id, "evidence", evidence.evidence_id, evidence)
        return evidence

    def add_candidate_bundle(
        self,
        candidate: MaterialCandidate,
        evidence: tuple[ScreeningEvidence, ...],
    ) -> MaterialCandidate:
        """Validate and persist property evidence before its candidate record."""
        expected = {value.evidence_id for value in candidate.property_values}
        supplied = {item.evidence_id for item in evidence}
        if expected != supplied:
            raise ResearchProjectError(
                "INCOMPLETE_CANDIDATE_EVIDENCE",
                "candidate bundle must contain exactly its property evidence",
            )
        for item in evidence:
            if (
                item.project_id != candidate.project_id
                or item.candidate_id != candidate.candidate_id
            ):
                raise ResearchProjectError(
                    "EVIDENCE_CANDIDATE_MISMATCH",
                    "bundled evidence must belong to the candidate",
                )
        by_id = {item.evidence_id: item for item in evidence}
        for value in candidate.property_values:
            item = by_id[value.evidence_id]
            if (
                item.property_id != value.property_id
                or item.value != value.value
                or item.unit != value.unit
            ):
                raise ResearchProjectError(
                    "PROPERTY_EVIDENCE_MISMATCH",
                    "candidate property does not match its bundled evidence",
                )
        with self._lock:
            for item in evidence:
                self._write(item.project_id, "evidence", item.evidence_id, item)
            return self.add_candidate(candidate)

    def add_candidate(self, candidate: MaterialCandidate) -> MaterialCandidate:
        for value in candidate.property_values:
            definition = self._registry.require(value.property_id)
            if (
                value.unit != definition.canonical_unit
                or value.source_role not in definition.source_roles
            ):
                raise ResearchProjectError(
                    "CANDIDATE_PROPERTY_CONTRACT_MISMATCH",
                    "candidate property unit or source role violates its registry",
                )
            evidence = self.get_evidence(candidate.project_id, value.evidence_id)
            if evidence.candidate_id != candidate.candidate_id:
                raise ResearchProjectError(
                    "EVIDENCE_CANDIDATE_MISMATCH",
                    "candidate property evidence belongs to another candidate",
                )
            if (
                evidence.property_id != value.property_id
                or evidence.value != value.value
                or evidence.unit != value.unit
            ):
                raise ResearchProjectError(
                    "PROPERTY_EVIDENCE_MISMATCH",
                    "candidate property does not match its evidence",
                )
        self._write(
            candidate.project_id,
            "candidates",
            candidate.candidate_id,
            candidate,
        )
        return candidate

    def add_claim(self, claim: ScreeningClaim) -> ScreeningClaim:
        if (
            claim.property_id is not None
            and self._registry.resolve(claim.property_id) is None
        ):
            raise ResearchProjectError(
                "UNREGISTERED_CLAIM_PROPERTY",
                "unregistered property cannot enter a claim",
            )
        if claim.candidate_id is not None:
            self.get_candidate(claim.project_id, claim.candidate_id)
        self._write(claim.project_id, "claims", claim.claim_id, claim)
        return claim

    def add_link(self, link: ClaimEvidenceLink) -> ClaimEvidenceLink:
        claim = self.get_claim(link.project_id, link.claim_id)
        evidence = self.get_evidence(link.project_id, link.evidence_id)
        if (
            claim.candidate_id is not None
            and evidence.candidate_id is not None
            and claim.candidate_id != evidence.candidate_id
        ):
            raise ResearchProjectError(
                "CLAIM_EVIDENCE_CANDIDATE_MISMATCH",
                "claim and evidence refer to different candidates",
            )
        if link.stance in {EvidenceStance.SUPPORTS, EvidenceStance.CONTRADICTS}:
            if evidence.review_status not in {
                EvidenceReviewStatus.VALIDATED,
                EvidenceReviewStatus.HUMAN_REVIEWED,
            }:
                raise ResearchProjectError(
                    "UNVALIDATED_EVIDENCE_LINK",
                    "pending or rejected evidence cannot support or contradict a claim",
                )
            if (
                claim.kind is ClaimKind.PROPERTY_VALUE
                and claim.property_id != evidence.property_id
            ):
                raise ResearchProjectError(
                    "CLAIM_PROPERTY_MISMATCH",
                    "property claim and evidence refer to different properties",
                )
            if evidence.source_kind in {
                EvidenceSourceKind.LITERATURE_FULL_TEXT,
                EvidenceSourceKind.LITERATURE_ABSTRACT,
                EvidenceSourceKind.USER_EXPERIMENT,
            }:
                if link.identity_assessment_id is None:
                    raise ResearchProjectError(
                        "IDENTITY_ASSESSMENT_REQUIRED",
                        "external material evidence requires an identity assessment",
                    )
                identity = self.get_identity(
                    link.project_id, link.identity_assessment_id
                )
                if (
                    identity.candidate_id != claim.candidate_id
                    or identity.evidence_id != evidence.evidence_id
                ):
                    raise ResearchProjectError(
                        "IDENTITY_ASSESSMENT_MISMATCH",
                        "identity assessment does not link this claim and evidence",
                    )
                if identity.identity_use is not IdentityUse.CANDIDATE_PROPERTY:
                    raise ResearchProjectError(
                        "IDENTITY_USE_BLOCKS_CLAIM",
                        "background or blocked identity cannot support a "
                        "candidate claim",
                    )
                if (
                    identity.requires_human_review
                    and identity.review_status != "human_reviewed"
                ):
                    raise ResearchProjectError(
                        "IDENTITY_REVIEW_REQUIRED",
                        "identity assessment requires human review before claim use",
                    )
        self._write(link.project_id, "links", link.link_id, link)
        return link

    def add_identity(
        self, identity: MaterialIdentityAssessmentRecord
    ) -> MaterialIdentityAssessmentRecord:
        self.get_candidate(identity.project_id, identity.candidate_id)
        evidence = self.get_evidence(identity.project_id, identity.evidence_id)
        if evidence.candidate_id != identity.candidate_id:
            raise ResearchProjectError(
                "IDENTITY_EVIDENCE_CANDIDATE_MISMATCH",
                "identity evidence belongs to another candidate",
            )
        self._write(
            identity.project_id,
            "identities",
            identity.identity_id,
            identity,
        )
        return identity

    def add_decision(self, decision: CandidateDecision) -> CandidateDecision:
        self.get_candidate(decision.project_id, decision.candidate_id)
        for evidence_id in decision.evidence_ids:
            evidence = self.get_evidence(decision.project_id, evidence_id)
            if (
                evidence.candidate_id is not None
                and evidence.candidate_id != decision.candidate_id
            ):
                raise ResearchProjectError(
                    "DECISION_EVIDENCE_CANDIDATE_MISMATCH",
                    "decision evidence belongs to another candidate",
                )
            if evidence.review_status is EvidenceReviewStatus.REJECTED:
                raise ResearchProjectError(
                    "REJECTED_DECISION_EVIDENCE",
                    "rejected evidence cannot be used in a candidate decision",
                )
        self._write(decision.project_id, "decisions", decision.decision_id, decision)
        return decision

    def get_candidate(self, project_id: str, candidate_id: str) -> MaterialCandidate:
        return MaterialCandidate.model_validate_json(
            self._read(project_id, "candidates", candidate_id)
        )

    def get_evidence(self, project_id: str, evidence_id: str) -> ScreeningEvidence:
        return ScreeningEvidence.model_validate_json(
            self._read(project_id, "evidence", evidence_id)
        )

    def get_claim(self, project_id: str, claim_id: str) -> ScreeningClaim:
        return ScreeningClaim.model_validate_json(
            self._read(project_id, "claims", claim_id)
        )

    def get_decision(
        self, project_id: str, decision_id: str
    ) -> CandidateDecision:
        return CandidateDecision.model_validate_json(
            self._read(project_id, "decisions", decision_id)
        )

    def get_link(self, project_id: str, link_id: str) -> ClaimEvidenceLink:
        return ClaimEvidenceLink.model_validate_json(
            self._read(project_id, "links", link_id)
        )

    def get_identity(
        self, project_id: str, identity_id: str
    ) -> MaterialIdentityAssessmentRecord:
        return MaterialIdentityAssessmentRecord.model_validate_json(
            self._read(project_id, "identities", identity_id)
        )

    def _read(self, project_id: str, kind: str, record_id: str) -> str:
        path = self._path(project_id, kind, record_id)
        with self._lock:
            if not path.is_file():
                raise KeyError(f"unknown {kind} record: {record_id!r}")
            return path.read_text(encoding="utf-8")

    def _write(
        self, project_id: str, kind: str, record_id: str, record: LedgerRecord
    ) -> None:
        path = self._path(project_id, kind, record_id)
        content = record.model_dump_json(indent=2)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.read_text(encoding="utf-8") == content:
                    return
                raise ProjectConflictError(
                    "IMMUTABLE_LEDGER_RECORD",
                    f"ledger record already exists: {record_id}",
                )
            temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
            try:
                temporary.write_text(content, encoding="utf-8")
                if path.exists():
                    raise ProjectConflictError(
                        "IMMUTABLE_LEDGER_RECORD",
                        f"ledger record already exists: {record_id}",
                    )
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def _path(self, project_id: str, kind: str, record_id: str) -> Path:
        if not re.fullmatch(_ID_PATTERN, project_id) or not re.fullmatch(
            _ID_PATTERN, record_id
        ):
            raise ResearchProjectError("INVALID_LEDGER_ID", "invalid ledger id")
        if kind not in {
            "candidates",
            "decisions",
            "evidence",
            "claims",
            "links",
            "identities",
        }:
            raise ResearchProjectError("INVALID_LEDGER_KIND", "invalid ledger kind")
        project_root = (self._root / project_id).resolve()
        if project_root.parent != self._root:
            raise ResearchProjectError("INVALID_LEDGER_ID", "invalid ledger path")
        path = (project_root / kind / f"{record_id}.json").resolve()
        if path.parent != (project_root / kind).resolve():
            raise ResearchProjectError("INVALID_LEDGER_ID", "invalid ledger path")
        return path


def _validate_locator(kind: EvidenceSourceKind, locator: EvidenceLocator) -> None:
    if kind is EvidenceSourceKind.MATERIALS_DATABASE:
        if locator.query_id is None or locator.source_material_id is None:
            raise ValueError("database evidence requires query and material ids")
    elif kind is EvidenceSourceKind.LITERATURE_FULL_TEXT:
        if (
            locator.document_id is None
            or locator.chunk_id is None
            or locator.page_from is None
        ):
            raise ValueError("full-text evidence requires document, chunk, and pages")
    elif kind is EvidenceSourceKind.LITERATURE_ABSTRACT:
        if locator.paper_id is None and locator.doi is None:
            raise ValueError("abstract evidence requires paper id or DOI")
    elif kind is EvidenceSourceKind.USER_EXPERIMENT:
        if locator.dataset_id is None:
            raise ValueError("user experiment evidence requires dataset id")
    elif kind is EvidenceSourceKind.DATA_ANALYSIS and (
        locator.dataset_id is None or locator.analysis_id is None
    ):
        raise ValueError("analysis evidence requires dataset and analysis ids")


def _validate_json(value: Any) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON metadata must not contain NaN or infinity")
        return
    if isinstance(value, list | tuple):
        for item in value:
            _validate_json(item)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON metadata keys must be strings")
        for item in value.values():
            _validate_json(item)
        return
    raise ValueError(f"unsupported JSON metadata type: {type(value).__name__}")
