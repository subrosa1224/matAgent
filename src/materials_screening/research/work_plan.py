"""Persisted bounded work plans for the RA-3 screening orchestrator."""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.research.errors import ProjectConflictError
from materials_screening.research.models import ProjectStatus, ScreeningProject

_ID = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,247}$"


class WorkKind(StrEnum):
    DATABASE_QUERY = "database_query"
    HARD_FILTER = "hard_filter"
    RANK_CANDIDATES = "rank_candidates"
    SELECT_QUICK_REVIEW = "select_quick_review"
    LITERATURE_QUICK_REVIEW = "literature_quick_review"
    SELECT_FULL_REVIEW = "select_full_review"
    LITERATURE_FULL_REVIEW = "literature_full_review"
    DATA_ANALYSIS = "data_analysis"
    SUPPLEMENTAL_REVIEW = "supplemental_review"


class WorkStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class OrchestrationStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    PAUSED_FAILURE = "paused_failure"
    EVIDENCE_READY = "evidence_ready"
    CANCELLED = "cancelled"


class ScreeningWorkItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    work_id: str = Field(pattern=_ID)
    kind: WorkKind
    agent_name: str
    objective: str = Field(min_length=1, max_length=1000)
    depends_on: tuple[str, ...] = Field(default=(), max_length=20)
    status: WorkStatus = WorkStatus.PENDING
    attempt_count: int = Field(default=0, ge=0, le=2)
    max_attempts: int = Field(default=1, ge=1, le=2)
    output_refs: tuple[str, ...] = Field(default=(), max_length=1000)
    error_code: str | None = Field(default=None, max_length=100)
    error_message: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def attempts_stay_bounded(self) -> Self:
        if self.attempt_count > self.max_attempts:
            raise ValueError("attempt_count cannot exceed max_attempts")
        if self.work_id in self.depends_on:
            raise ValueError("work item cannot depend on itself")
        return self


class FilterOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=_ID)
    status: str = Field(pattern=r"^(passed|failed|unknown)$")
    reasons: tuple[str, ...] = Field(default=(), max_length=100)


class RankedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=_ID)
    rank: int = Field(ge=1, le=500)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    components: dict[str, float] = Field(default_factory=dict)


class ScreeningRunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=_ID)
    project_id: str = Field(pattern=_ID)
    project_revision: int = Field(ge=1)
    project_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: OrchestrationStatus = OrchestrationStatus.PLANNED
    revision: int = Field(default=1, ge=1)
    work_items: tuple[ScreeningWorkItem, ...] = Field(min_length=1, max_length=12)
    candidate_ids: tuple[str, ...] = Field(default=(), max_length=500)
    filter_outcomes: tuple[FilterOutcome, ...] = Field(default=(), max_length=500)
    ranked_candidates: tuple[RankedCandidate, ...] = Field(default=(), max_length=500)
    shortlist_ids: tuple[str, ...] = Field(default=(), max_length=100)
    quick_review_ids: tuple[str, ...] = Field(default=(), max_length=20)
    full_review_ids: tuple[str, ...] = Field(default=(), max_length=5)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=5000)
    claim_ids: tuple[str, ...] = Field(default=(), max_length=2000)
    supplemental_used: bool = False
    warnings: tuple[str, ...] = Field(default=(), max_length=500)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def plan_references_are_valid(self) -> Self:
        ids = [item.work_id for item in self.work_items]
        if len(ids) != len(set(ids)):
            raise ValueError("work ids must be unique")
        known = set(ids)
        if any(
            dependency not in known
            for item in self.work_items
            for dependency in item.depends_on
        ):
            raise ValueError("work dependency references an unknown work item")
        for values in (
            self.candidate_ids,
            self.shortlist_ids,
            self.quick_review_ids,
            self.full_review_ids,
            self.evidence_ids,
            self.claim_ids,
        ):
            if len(values) != len(set(values)):
                raise ValueError("run references must be unique")
        return self


def build_screening_plan(project: ScreeningProject) -> ScreeningRunState:
    """Build the fixed, bounded RA-3 dependency graph for a confirmed contract."""
    if (
        project.status is not ProjectStatus.CONFIRMED
        or project.confirmed_revision != project.revision
    ):
        raise ValueError("only the currently confirmed project revision can be planned")
    digest = hashlib.sha256(project.model_dump_json().encode()).hexdigest()
    run_id = f"run-{digest[:24]}"
    specs = (
        (WorkKind.DATABASE_QUERY, "materials_database", "生成数据库候选", (), 2),
        (WorkKind.HARD_FILTER, "orchestrator", "重新执行全部硬约束", (0,), 1),
        (WorkKind.RANK_CANDIDATES, "orchestrator", "计算可解释排序", (1,), 1),
        (WorkKind.SELECT_QUICK_REVIEW, "orchestrator", "选择快速文献核验候选", (2,), 1),
        (WorkKind.LITERATURE_QUICK_REVIEW, "literature", "执行快速文献核验", (3,), 2),
        (WorkKind.SELECT_FULL_REVIEW, "orchestrator", "选择全文深度核验候选", (4,), 1),
        (WorkKind.LITERATURE_FULL_REVIEW, "literature", "执行全文深度核验", (5,), 2),
        (WorkKind.DATA_ANALYSIS, "data_analysis", "可选候选数据分析", (6,), 1),
    )
    work_ids = tuple(
        f"work-{hashlib.sha256(f'{run_id}|{kind.value}'.encode()).hexdigest()[:24]}"
        for kind, *_ in specs
    )
    items = tuple(
        ScreeningWorkItem(
            work_id=work_ids[index],
            kind=kind,
            agent_name=agent,
            objective=objective,
            depends_on=tuple(work_ids[item] for item in dependency_indexes),
            max_attempts=max_attempts,
        )
        for index, (
            kind,
            agent,
            objective,
            dependency_indexes,
            max_attempts,
        ) in enumerate(specs)
    )
    return ScreeningRunState(
        run_id=run_id,
        project_id=project.project_id,
        project_revision=project.revision,
        project_contract_sha256=digest,
        work_items=items,
    )


class ScreeningRunStore:
    """Optimistic, immutable-revision persistence for orchestration state."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._lock = threading.RLock()

    def create(self, state: ScreeningRunState) -> ScreeningRunState:
        with self._lock:
            current = self._path(state.run_id, "current.json")
            if current.exists():
                existing = self.get(state.run_id)
                if existing == state:
                    return existing
                raise ProjectConflictError("RUN_EXISTS", "screening run already exists")
            self._persist(state)
            return state

    def get(self, run_id: str) -> ScreeningRunState:
        path = self._path(run_id, "current.json")
        with self._lock:
            if not path.is_file():
                raise KeyError(f"unknown screening run: {run_id!r}")
            return ScreeningRunState.model_validate_json(
                path.read_text(encoding="utf-8")
            )

    def save(
        self, state: ScreeningRunState, *, expected_revision: int
    ) -> ScreeningRunState:
        with self._lock:
            current = self.get(state.run_id)
            if current.revision != expected_revision:
                raise ProjectConflictError(
                    "STALE_RUN_REVISION",
                    f"expected run revision {expected_revision}, "
                    f"current is {current.revision}",
                )
            if state.revision != expected_revision + 1:
                raise ProjectConflictError(
                    "INVALID_RUN_REVISION", "run revision must increment exactly once"
                )
            if (
                state.project_id != current.project_id
                or state.project_revision != current.project_revision
                or state.project_contract_sha256 != current.project_contract_sha256
                or state.created_at != current.created_at
            ):
                raise ProjectConflictError(
                    "IMMUTABLE_RUN_FIELD", "run identity and contract cannot change"
                )
            self._persist(state)
            return state

    def _persist(self, state: ScreeningRunState) -> None:
        revision_name = f"revisions/{state.revision:06d}.json"
        revision_path = self._path(state.run_id, revision_name)
        if revision_path.exists():
            raise ProjectConflictError(
                "RUN_REVISION_EXISTS", "immutable run revision already exists"
            )
        content = state.model_dump_json(indent=2)
        _atomic_write(revision_path, content, overwrite=False)
        _atomic_write(self._path(state.run_id, "current.json"), content)

    def _path(self, run_id: str, relative: str) -> Path:
        if not run_id or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-:"
            for char in run_id
        ):
            raise ValueError("invalid run id")
        run_root = (self._root / run_id).resolve()
        if run_root.parent != self._root:
            raise ValueError("invalid run path")
        path = (run_root / relative).resolve()
        if run_root not in path.parents:
            raise ValueError("invalid run path")
        return path


def _atomic_write(path: Path, content: str, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        if not overwrite and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
