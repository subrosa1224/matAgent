"""Persistent batch summaries for user-directed multi-PDF analysis."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BatchPaperResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    file_name: str
    title: str
    status: Literal["completed", "partial", "failed"]
    dossier_status: Literal["pending", "approved", "rejected", "missing"]
    dossier_items: int = Field(ge=0)
    completeness_score: float | None = Field(default=None, ge=0, le=1)
    matrix_groups: int = Field(ge=0)
    matrix_measurements: int = Field(ge=0)
    matrix_claims: int = Field(default=0, ge=0)
    claim_checks: int = Field(default=0, ge=0)
    verified_claim_checks: int = Field(default=0, ge=0)
    pending_measurements: int = Field(ge=0)
    warnings: tuple[str, ...] = ()
    error: str | None = None


class LiteratureBatchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_id: str
    created_at: datetime
    papers: tuple[BatchPaperResult, ...]

    @property
    def completed_count(self) -> int:
        return sum(row.status == "completed" for row in self.papers)

    @property
    def attention_count(self) -> int:
        return sum(row.status != "completed" for row in self.papers)

    @property
    def review_required_count(self) -> int:
        return sum(
            row.dossier_status == "pending" or row.pending_measurements > 0
            for row in self.papers
        )


class LiteratureBatchStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(self, papers: tuple[BatchPaperResult, ...]) -> LiteratureBatchReport:
        if not papers:
            raise ValueError("a literature batch requires at least one paper")
        identity = "|".join(sorted(row.document_id for row in papers))
        batch_id = f"lit-batch-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        report = LiteratureBatchReport(
            batch_id=batch_id,
            created_at=datetime.now(UTC),
            papers=papers,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        target = (self.root / f"{batch_id}.json").resolve()
        if target.parent != self.root:
            raise ValueError("invalid literature batch id")
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)
        return report

    def load(self, batch_id: str) -> LiteratureBatchReport | None:
        target = (self.root / f"{batch_id}.json").resolve()
        if target.parent != self.root or not target.exists():
            return None
        return LiteratureBatchReport.model_validate(
            json.loads(target.read_text(encoding="utf-8"))
        )


def document_id_for_pdf(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"doc-{digest[:24]}"


def missing_index_paths(
    paths: tuple[Path, ...], document_exists: Callable[[str], bool]
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in paths
        if not document_exists(document_id_for_pdf(path))
    )


def should_extract_matrix(counts: dict[str, int], *, retry_incomplete: bool) -> bool:
    if not any(counts.values()):
        return True
    has_reviewable_evidence = (
        counts.get("measurements", 0) > 0
        or counts.get("verified_claim_checks", 0) > 0
    )
    return retry_incomplete and not has_reviewable_evidence


def matrix_analysis_complete(counts: dict[str, int]) -> bool:
    """Accept quantitative matrices or evidence-checked qualitative claims."""
    return (
        counts.get("measurements", 0) > 0
        or counts.get("verified_claim_checks", 0) > 0
    )
