"""Append-only, task-bound extraction checkpoints; no canonical matrix writes."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.sub_agents.literature.matrix_automation import (
    MatrixExtractionBatch,
    MatrixExtractionDiagnostics,
    PendingMatrixExtraction,
)
from materials_screening.sub_agents.literature.models import (
    ClaimEvidenceLink,
    ExperimentalComparison,
    ExperimentalGroup,
    ExperimentalMeasurement,
    PaperClaim,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord

_ID = r"^snapshot-[a-f0-9]{32}$"
_SHA = r"^[a-f0-9]{64}$"


class SnapshotReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_id: str = Field(pattern=_ID)
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    content_sha256: str = Field(pattern=_SHA)


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk_id: str
    text_sha256: str = Field(pattern=_SHA)
    page_from: int = Field(ge=1)
    page_to: int = Field(ge=1)


def source_identities(chunks):
    return tuple(
        SourceIdentity(
            chunk_id=row.chunk_id,
            text_sha256=row.text_sha256,
            page_from=row.page_from,
            page_to=row.page_to,
        )
        for row in sorted(chunks, key=lambda row: row.chunk_id)
    )


class MatrixPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    groups: tuple[ExperimentalGroup, ...] = ()
    measurements: tuple[ExperimentalMeasurement, ...] = ()
    comparisons: tuple[ExperimentalComparison, ...] = ()
    claims: tuple[PaperClaim, ...] = ()
    claim_evidence_links: tuple[ClaimEvidenceLink, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: MatrixExtractionDiagnostics

    @classmethod
    def from_result(cls, result: PendingMatrixExtraction):
        return cls(**{name: getattr(result, name) for name in cls.model_fields})


class ExtractionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    snapshot_id: str = Field(
        default_factory=lambda: "snapshot-" + uuid4().hex, pattern=_ID
    )
    attempt_id: str = Field(pattern=r"^attempt-[a-f0-9]{32}$")
    parent_snapshot_id: str | None = Field(default=None, pattern=_ID)
    task_id: str = Field(pattern=r"^task-fulltext-[a-f0-9]{32}$")
    conversation_id: str
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    pdf_sha256: str = Field(pattern=_SHA)
    title: str
    version: Literal["master-matrix-v1"] = "master-matrix-v1"
    policy_sha256: str = Field(pattern=_SHA)
    original_question: str
    user_instructions: tuple[str, ...] = ()
    model_profile: str
    source_identities: tuple[SourceIdentity, ...]
    planned_batches: int = Field(ge=1, le=100)
    batches: dict[int, MatrixExtractionBatch] = Field(default_factory=dict)
    status: Literal["collecting", "complete", "partial", "failed", "cancelled"]
    matrix: MatrixPayload | None = None
    warnings: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def isolated_pending(self):
        if self.document_id != "doc-" + self.pdf_sha256[:24]:
            raise ValueError("Snapshot document and PDF identity differ")
        if len({row.chunk_id for row in self.source_identities}) != len(
            self.source_identities
        ):
            raise ValueError("Duplicate source identity")
        if any(
            index not in range(1, self.planned_batches + 1) for index in self.batches
        ):
            raise ValueError("Batch is outside evidence plan")
        if self.status == "complete" and (
            self.matrix is None or len(self.batches) != self.planned_batches
        ):
            raise ValueError("Incomplete extraction cannot be a complete snapshot")
        if self.matrix is not None:
            for name in (
                "groups",
                "measurements",
                "comparisons",
                "claims",
                "claim_evidence_links",
            ):
                if any(
                    row.document_id != self.document_id
                    or row.review_status != "pending"
                    for row in getattr(self.matrix, name)
                ):
                    raise ValueError(
                        "New snapshots contain only their own pending records"
                    )
        return self


class ExtractionSnapshotStore:
    def __init__(self, root: Path, *, max_bytes=16 * 1024 * 1024):
        self.root = root.resolve()
        self.max_bytes = max_bytes

    def _path(self, reference):
        # Reference validation precedes path interpolation, including on callers
        # that bypassed pydantic validation using model_copy.
        reference = SnapshotReference.model_validate(reference.model_dump())
        path = (self.root / f"{reference.snapshot_id}.json").resolve()
        if path.parent != self.root:
            raise ValueError("Invalid snapshot path")
        return path

    def save(self, snapshot: ExtractionSnapshot) -> SnapshotReference:
        snapshot = ExtractionSnapshot.model_validate(snapshot.model_dump())
        content = snapshot.model_dump_json(indent=2).encode("utf-8")
        if len(content) > self.max_bytes:
            raise ValueError("Snapshot exceeds bounded storage limit")
        reference = SnapshotReference(
            snapshot_id=snapshot.snapshot_id,
            document_id=snapshot.document_id,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )
        path = self._path(reference)
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(
                    "Refusing to overwrite an immutable snapshot"
                ) from None
        return reference

    def load(self, reference: SnapshotReference) -> ExtractionSnapshot:
        path = self._path(reference)
        with path.open("rb") as handle:
            content = handle.read(self.max_bytes + 1)
        if (
            len(content) > self.max_bytes
            or hashlib.sha256(content).hexdigest() != reference.content_sha256
        ):
            raise ValueError("Snapshot bytes do not match checkpoint reference")
        snapshot = ExtractionSnapshot.model_validate_json(content)
        if (
            snapshot.snapshot_id != reference.snapshot_id
            or snapshot.document_id != reference.document_id
        ):
            raise ValueError("Snapshot identity differs from checkpoint reference")
        return snapshot


class CheckedChunkStore:
    """Only evidence methods; canonical matrix save/load methods do not exist."""

    def __init__(self, chunks: tuple[ChunkRecord, ...]):
        self.chunks = chunks

    def get_chunks(self, chunk_ids):
        return [row for row in self.chunks if row.chunk_id in chunk_ids]
