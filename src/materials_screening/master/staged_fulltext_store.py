"""Versioned append-only stage checkpoints, separate from legacy v1 snapshots."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from .fulltext_conditions import ConditionPlan
from .fulltext_snapshots import SourceIdentity
from .staged_fulltext_evidence import VerifiedMetric, VerifiedSample
from .staged_fulltext_repairs import RepairFeedback


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StagedReference(_Frozen):
    record_id: str = Field(pattern=r"^staged-[a-f0-9]{32}$")
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class UnresolvedEvidence(_Frozen):
    stage: Literal["inventory", "metrics", "conditions"]
    source_id: str
    metric: str | None = None
    reason: Literal[
        "oversized_source",
        "table_layout_unverified",
        "candidate_rejected",
        "ambiguous_identity",
        "conflicting_condition",
        "sample_binding_unverified",
        "prior_report_scope_unverified",
    ]


class RejectedCandidate(_Frozen):
    candidate: dict[str, str]
    reason: str = Field(min_length=1, max_length=240)


class SavedStep(_Frozen):
    stage: Literal["inventory", "metrics", "conditions"]
    status: Literal["done", "retry_pending", "failed"]
    attempts: int = Field(ge=0, le=2)
    samples: tuple[VerifiedSample, ...] = ()
    measurements: tuple[VerifiedMetric, ...] = ()
    conditions: ConditionPlan | None = None
    rejected: int = Field(default=0, ge=0)
    rejected_details: tuple[RejectedCandidate, ...] = Field(default=(), max_length=24)
    error_kind: str | None = None
    repair_feedback: RepairFeedback | None = None
    returned_model_profile: str | None = None

    @model_serializer(mode="wrap")
    def serialize_feedback(self, handler):
        # Preserve legacy canonical bytes/signatures when this optional new
        # metadata is absent. Never rewrite old snapshots or relax SHA checks.
        values = handler(self)
        if self.repair_feedback is None:
            values.pop("repair_feedback", None)
        return values


class StagedExtractionRecord(_Frozen):
    record_id: str = Field(
        default_factory=lambda: "staged-" + uuid4().hex,
        pattern=r"^staged-[a-f0-9]{32}$",
    )
    parent_record_id: str | None = Field(default=None, pattern=r"^staged-[a-f0-9]{32}$")
    attempt_id: str = Field(default_factory=lambda: "attempt-" + uuid4().hex)
    version: Literal["master-staged-extraction-v1"] = "master-staged-extraction-v1"
    task_id: str = Field(pattern=r"^task-fulltext-[a-f0-9]{32}$")
    conversation_id: str
    document_id: str = Field(pattern=r"^doc-[a-f0-9]{24}$")
    pdf_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    title: str
    requirements: str
    requested_metrics: tuple[str, ...]
    model_profile: str
    policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_identities: tuple[SourceIdentity, ...]
    steps: dict[str, SavedStep] = Field(default_factory=dict)
    samples: tuple[VerifiedSample, ...] = ()
    measurements: tuple[VerifiedMetric, ...] = ()
    condition_plans: tuple[ConditionPlan, ...] = ()
    unresolved: tuple[UnresolvedEvidence, ...] = ()
    coverage: dict[
        str, Literal["verified", "pending_verification", "not_found_current_evidence"]
    ] = Field(default_factory=dict)
    status: Literal[
        "processing", "requests_complete", "partial", "failed", "cancelled"
    ] = "processing"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def identities(self):
        if self.document_id != "doc-" + self.pdf_sha256[:24]:
            raise ValueError("Stage PDF identity differs")
        if len({r.chunk_id for r in self.source_identities}) != len(
            self.source_identities
        ):
            raise ValueError("Duplicate staged source identity")
        if len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate staged sample identity")
        if len({m.measurement_id for m in self.measurements}) != len(self.measurements):
            raise ValueError("Duplicate staged metric identity")
        if any(s.document_id != self.document_id for s in self.samples) or any(
            m.document_id != self.document_id for m in self.measurements
        ):
            raise ValueError("Foreign document in stage record")
        if not set(self.coverage).issubset(self.requested_metrics):
            raise ValueError("Coverage outside requested metrics")
        return self


class StagedExtractionStore:
    def __init__(self, root: Path, max_bytes=16 * 1024 * 1024):
        self.root, self.max_bytes = root.resolve(), max_bytes

    def _path(self, ref):
        ref = StagedReference.model_validate(ref.model_dump())
        path = (self.root / (ref.record_id + ".json")).resolve()
        if path.parent != self.root:
            raise ValueError("Invalid stage path")
        return path

    def save(self, record):
        record = StagedExtractionRecord.model_validate(record.model_dump())
        content = record.model_dump_json(indent=2).encode()
        if len(content) > self.max_bytes:
            raise ValueError("Stage record exceeds bounded store")
        ref = StagedReference(
            record_id=record.record_id,
            document_id=record.document_id,
            content_sha256=hashlib.sha256(content).hexdigest(),
        )
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(ref)
        try:
            with path.open("xb") as handle:
                handle.write(content)
                handle.flush()
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError("Cannot overwrite immutable stage record") from None
        return ref

    def load(self, ref):
        ref = StagedReference.model_validate(ref.model_dump())
        with self._path(ref).open("rb") as handle:
            content = handle.read(self.max_bytes + 1)
        if (
            len(content) > self.max_bytes
            or hashlib.sha256(content).hexdigest() != ref.content_sha256
        ):
            raise ValueError("Stage bytes differ from checkpoint reference")
        record = StagedExtractionRecord.model_validate_json(content)
        if (record.record_id, record.document_id) != (ref.record_id, ref.document_id):
            raise ValueError("Stage identity differs from reference")
        return record
