"""Explicit project state transitions with confirmation and revision guards."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from materials_screening.research.capability_gate import CapabilityAssessment
from materials_screening.research.errors import (
    ProjectConflictError,
    ProjectTransitionError,
)
from materials_screening.research.models import ProjectStatus, ScreeningProject

_ALLOWED: dict[ProjectStatus, frozenset[ProjectStatus]] = {
    ProjectStatus.DRAFT: frozenset(
        {ProjectStatus.AWAITING_CONFIRMATION, ProjectStatus.CANCELLED}
    ),
    ProjectStatus.AWAITING_CONFIRMATION: frozenset(
        {ProjectStatus.DRAFT, ProjectStatus.CONFIRMED, ProjectStatus.CANCELLED}
    ),
    ProjectStatus.CONFIRMED: frozenset(
        {ProjectStatus.DRAFT, ProjectStatus.RUNNING, ProjectStatus.CANCELLED}
    ),
    ProjectStatus.RUNNING: frozenset(
        {ProjectStatus.COMPLETED, ProjectStatus.FAILED, ProjectStatus.CANCELLED}
    ),
    ProjectStatus.FAILED: frozenset({ProjectStatus.DRAFT}),
    ProjectStatus.COMPLETED: frozenset(),
    ProjectStatus.CANCELLED: frozenset(),
}


class ProjectStateMachine:
    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))

    def transition(
        self,
        project: ScreeningProject,
        target: ProjectStatus,
        *,
        expected_revision: int,
        assessment: CapabilityAssessment | None = None,
    ) -> ScreeningProject:
        if project.revision != expected_revision:
            raise ProjectConflictError(
                "STALE_REVISION",
                f"expected revision {expected_revision}, current is {project.revision}",
            )
        if target not in _ALLOWED[project.status]:
            raise ProjectTransitionError(
                "INVALID_STATE_TRANSITION",
                f"cannot transition from {project.status.value} to {target.value}",
            )
        if target in {
            ProjectStatus.AWAITING_CONFIRMATION,
            ProjectStatus.CONFIRMED,
        } and (
            assessment is None
            or assessment.project_id != project.project_id
            or assessment.revision != project.revision
            or not assessment.executable
        ):
            raise ProjectTransitionError(
                "CAPABILITY_GATE_NOT_PASSED",
                "a supported assessment for the current revision is required",
            )
        if target is ProjectStatus.RUNNING and (
            project.status is not ProjectStatus.CONFIRMED
            or project.confirmed_revision != project.revision
        ):
            raise ProjectTransitionError(
                "PROJECT_NOT_CONFIRMED",
                "only the currently confirmed revision may run",
            )

        next_revision = project.revision + 1
        confirmed_revision = project.confirmed_revision
        if target is ProjectStatus.CONFIRMED:
            confirmed_revision = next_revision
        elif target is ProjectStatus.DRAFT:
            confirmed_revision = None
        return project.model_copy(
            update={
                "status": target,
                "revision": next_revision,
                "confirmed_revision": confirmed_revision,
                "updated_at": self._clock(),
            }
        )
