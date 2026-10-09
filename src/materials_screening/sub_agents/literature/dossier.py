"""Validated local artifacts for evidence-grounded single-paper dossiers."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DossierItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Literal[
        "research_problem",
        "innovation",
        "materials",
        "preparation",
        "device_fabrication",
        "characterization",
        "structural_result",
        "optical_result",
        "performance_result",
        "mechanism",
        "limitations",
        "reproducibility",
    ]
    summary: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    page: int = Field(ge=1)
    document_id: str | None = None
    page_to: int | None = Field(default=None, ge=1)
    source_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    confidence: float = Field(default=1.0, ge=0, le=1)
    risk_level: Literal["low", "medium", "high"] = "low"
    llm_extracted: bool = False


class DossierItemCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    category: Literal[
        "research_problem",
        "innovation",
        "materials",
        "preparation",
        "device_fabrication",
        "characterization",
        "structural_result",
        "optical_result",
        "performance_result",
        "mechanism",
        "limitations",
        "reproducibility",
    ]
    summary: str = Field(min_length=1)
    source_quote: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class DossierExtractionBatch(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    items: tuple[DossierItemCandidate, ...] = Field(max_length=30)


class DossierSummaryTranslation(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    item_key: str = Field(min_length=1)
    chinese_summary: str = Field(min_length=1)


class DossierTranslationBatch(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    items: tuple[DossierSummaryTranslation, ...] = Field(max_length=30)


class PaperDossier(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    title: str
    extraction_method: Literal["manual_pilot", "llm", "hybrid"]
    review_status: Literal["pending", "approved", "rejected"] = "pending"
    items: tuple[DossierItem, ...]
    warnings: tuple[str, ...] = ()
    extraction_version: str | None = None
    narrative_summary: str | None = None
    covered_categories: tuple[str, ...] = ()
    missing_core_categories: tuple[str, ...] = ()
    completeness_score: float | None = Field(default=None, ge=0, le=1)


class DossierReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    review_id: str
    document_id: str
    previous_status: Literal["pending", "approved", "rejected"]
    decision: Literal["approved", "rejected"]
    reviewer: str
    reason: str | None = None
    reviewed_at: datetime


class PaperDossierStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def load(self, document_id: str) -> PaperDossier | None:
        path = (self.root / f"{document_id}.json").resolve()
        if path.parent != self.root or not path.exists():
            return None
        return PaperDossier.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def review(
        self,
        *,
        document_id: str,
        decision: Literal["approved", "rejected"],
        reviewer: str,
        reason: str | None = None,
    ) -> DossierReview:
        dossier = self.load(document_id)
        if dossier is None:
            raise ValueError("paper dossier has not been generated")
        reviewer = reviewer.strip()
        if not reviewer:
            raise ValueError("reviewer must not be blank")
        reviewed_at = datetime.now(UTC)
        identity = f"{document_id}|{decision}|{reviewer}|{reviewed_at.isoformat()}"
        review = DossierReview(
            review_id="dossier-review-"
            + hashlib.sha256(identity.encode()).hexdigest()[:24],
            document_id=document_id,
            previous_status=dossier.review_status,
            decision=decision,
            reviewer=reviewer,
            reason=reason,
            reviewed_at=reviewed_at,
        )
        target = (self.root / f"{document_id}.json").resolve()
        updated = dossier.model_copy(update={"review_status": decision})
        self._atomic_write(target, updated.model_dump_json(indent=2))
        review_root = self.root.parent / "literature_dossier_reviews" / document_id
        review_root.mkdir(parents=True, exist_ok=True)
        event_path = review_root / f"{review.review_id}.json"
        with event_path.open("x", encoding="utf-8") as handle:
            handle.write(review.model_dump_json(indent=2))
        return review

    def save_pending(self, dossier: PaperDossier) -> Path:
        if dossier.review_status != "pending":
            raise ValueError("automated dossier must be staged as pending")
        target = (self.root / f"{dossier.document_id}.json").resolve()
        if target.parent != self.root:
            raise ValueError("invalid dossier document id")
        existing = self.load(dossier.document_id)
        if existing is not None and existing.review_status == "approved":
            raise ValueError("approved dossier cannot be overwritten")
        self._atomic_write(target, dossier.model_dump_json(indent=2))
        return target

    def save_reviewed(self, dossier: PaperDossier) -> Path:
        """Publish a human-approved dossier into the trusted dossier store."""
        if dossier.review_status != "approved":
            raise ValueError("only an approved dossier can be published")
        target = (self.root / f"{dossier.document_id}.json").resolve()
        if target.parent != self.root:
            raise ValueError("invalid dossier document id")
        self._atomic_write(target, dossier.model_dump_json(indent=2))
        return target

    @staticmethod
    def _atomic_write(target: Path, payload: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(f"{target.suffix}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
