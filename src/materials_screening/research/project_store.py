"""Immutable-revision file store for screening projects and decision logs."""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from materials_screening.research.capability_gate import CapabilityAssessment
from materials_screening.research.errors import (
    ProjectConflictError,
    ResearchProjectError,
)
from materials_screening.research.models import (
    DecisionLogEntry,
    ProjectStatus,
    ScreeningProject,
)
from materials_screening.research.state_machine import ProjectStateMachine


class ScreeningProjectStore:
    """Persist current state plus immutable revision and decision history."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._root = root.resolve()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._machine = ProjectStateMachine(clock=self._clock)
        self._lock = threading.RLock()

    def create(
        self,
        project: ScreeningProject,
        *,
        actor: str,
        reason: str,
    ) -> ScreeningProject:
        if project.revision != 1 or project.status is not ProjectStatus.DRAFT:
            raise ResearchProjectError(
                "INVALID_INITIAL_PROJECT",
                "new project must be draft revision 1",
            )
        project_dir = self._project_dir(project.project_id)
        with self._lock:
            current = project_dir / "current.json"
            if current.exists():
                raise ProjectConflictError(
                    "PROJECT_EXISTS", f"project already exists: {project.project_id}"
                )
            self._persist(
                project,
                DecisionLogEntry(
                    project_id=project.project_id,
                    revision=project.revision,
                    action="project_created",
                    actor=actor,
                    reason=reason,
                    timestamp=self._clock(),
                ),
            )
        return project

    def get(self, project_id: str) -> ScreeningProject:
        current = self._project_dir(project_id) / "current.json"
        with self._lock:
            if not current.is_file():
                raise KeyError(f"unknown screening project: {project_id!r}")
            return ScreeningProject.model_validate_json(
                current.read_text(encoding="utf-8")
            )

    def save_revision(
        self,
        project: ScreeningProject,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
    ) -> ScreeningProject:
        with self._lock:
            current = self.get(project.project_id)
            if current.revision != expected_revision:
                raise ProjectConflictError(
                    "STALE_REVISION",
                    f"expected revision {expected_revision}, "
                    f"current is {current.revision}",
                )
            if project.revision != expected_revision + 1:
                raise ProjectConflictError(
                    "INVALID_NEXT_REVISION",
                    "saved project revision must increment exactly once",
                )
            if project.created_at != current.created_at:
                raise ProjectConflictError(
                    "IMMUTABLE_PROJECT_FIELD", "created_at cannot be changed"
                )
            if (
                current.status is not ProjectStatus.DRAFT
                or project.status is not ProjectStatus.DRAFT
                or project.confirmed_revision is not None
            ):
                raise ProjectConflictError(
                    "CONTRACT_NOT_EDITABLE",
                    "contract revisions require draft state and clear confirmation",
                )
            self._commit_revision(
                project,
                DecisionLogEntry(
                    project_id=project.project_id,
                    revision=project.revision,
                    action="contract_revised",
                    actor=actor,
                    reason=reason,
                    timestamp=self._clock(),
                    details={
                        "from_status": current.status.value,
                        "to_status": project.status.value,
                    },
                ),
            )
            return project

    def transition(
        self,
        project_id: str,
        target: ProjectStatus,
        *,
        expected_revision: int,
        actor: str,
        reason: str,
        assessment: CapabilityAssessment | None = None,
    ) -> ScreeningProject:
        with self._lock:
            current = self.get(project_id)
            updated = self._machine.transition(
                current,
                target,
                expected_revision=expected_revision,
                assessment=assessment,
            )
            self._commit_revision(
                updated,
                DecisionLogEntry(
                    project_id=updated.project_id,
                    revision=updated.revision,
                    action="status_transition",
                    actor=actor,
                    reason=reason,
                    timestamp=self._clock(),
                    details={
                        "from_status": current.status.value,
                        "to_status": updated.status.value,
                    },
                ),
            )
            return updated

    def list_decisions(self, project_id: str) -> tuple[DecisionLogEntry, ...]:
        path = self._project_dir(project_id) / "decision_log.jsonl"
        with self._lock:
            if not path.is_file():
                if not (self._project_dir(project_id) / "current.json").is_file():
                    raise KeyError(f"unknown screening project: {project_id!r}")
                return ()
            return tuple(
                DecisionLogEntry.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )

    def get_revision(self, project_id: str, revision: int) -> ScreeningProject:
        path = self._project_dir(project_id) / "revisions" / f"{revision:06d}.json"
        with self._lock:
            if not path.is_file():
                raise KeyError(f"unknown project revision: {project_id!r} r{revision}")
            return ScreeningProject.model_validate_json(
                path.read_text(encoding="utf-8")
            )

    def _project_dir(self, project_id: str) -> Path:
        allowed = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        )
        if not project_id or any(character not in allowed for character in project_id):
            raise ResearchProjectError("INVALID_PROJECT_ID", "invalid project id")
        path = (self._root / project_id).resolve()
        if path.parent != self._root:
            raise ResearchProjectError("INVALID_PROJECT_ID", "invalid project path")
        return path

    def _persist(self, project: ScreeningProject, decision: DecisionLogEntry) -> None:
        project_dir = self._project_dir(project.project_id)
        revisions = project_dir / "revisions"
        revision_path = revisions / f"{project.revision:06d}.json"
        current_path = project_dir / "current.json"
        log_path = project_dir / "decision_log.jsonl"
        project_dir.mkdir(parents=True, exist_ok=True)
        revisions.mkdir(parents=True, exist_ok=True)
        if revision_path.exists():
            raise ProjectConflictError(
                "REVISION_EXISTS",
                f"immutable revision already exists: {project.revision}",
            )
        project_json = project.model_dump_json(indent=2)
        previous_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        new_log = previous_log + decision.model_dump_json() + "\n"
        _atomic_write(revision_path, project_json, overwrite=False)
        _atomic_write(current_path, project_json)
        _atomic_write(log_path, new_log)

    def _commit_revision(
        self, project: ScreeningProject, decision: DecisionLogEntry
    ) -> None:
        self._persist(project, decision)


def _atomic_write(path: Path, content: str, *, overwrite: bool = True) -> None:
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
