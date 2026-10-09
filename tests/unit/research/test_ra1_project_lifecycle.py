from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from materials_screening.research import (
    CriterionOperator,
    CriterionRole,
    MaterialScope,
    MissingValuePolicy,
    ProjectConflictError,
    ProjectStatus,
    ProjectTransitionError,
    PropertyCapabilityGate,
    ScreeningCriterion,
    ScreeningProject,
    ScreeningProjectStore,
    default_property_registry,
)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 9, 2, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value += timedelta(seconds=1)
        return self.value


def _project(now: datetime) -> ScreeningProject:
    return ScreeningProject(
        project_id="wide-gap-oxide",
        title="稳定宽禁带氧化物",
        research_question="哪些稳定氧化物的计算带隙不低于 3 eV？",
        material_scope=MaterialScope(required_elements=("O",)),
        criteria=(
            ScreeningCriterion(
                property_id="band_gap_ev",
                role=CriterionRole.HARD_FILTER,
                operator=CriterionOperator.GTE,
                values=(3.0,),
                unit="eV",
                missing_policy=MissingValuePolicy.EXCLUDE,
            ),
        ),
        created_at=now,
        updated_at=now,
    )


def _store(tmp_path: Path) -> tuple[ScreeningProjectStore, ScreeningProject]:
    clock = Clock()
    project = _project(clock())
    store = ScreeningProjectStore(tmp_path / "projects", clock=clock)
    store.create(project, actor="user", reason="建立筛选任务")
    return store, project


def test_store_preserves_immutable_revisions_and_decision_log(tmp_path: Path) -> None:
    store, project = _store(tmp_path)
    revised = project.model_copy(
        update={
            "revision": 2,
            "title": "稳定宽禁带晶态氧化物",
            "updated_at": datetime(2026, 9, 2, 0, 0, 5, tzinfo=UTC),
        }
    )

    store.save_revision(
        revised,
        expected_revision=1,
        actor="user",
        reason="明确晶态范围",
    )

    assert store.get("wide-gap-oxide").revision == 2
    assert store.get_revision("wide-gap-oxide", 1).title == "稳定宽禁带氧化物"
    assert store.get_revision("wide-gap-oxide", 2).title == revised.title
    assert [entry.action for entry in store.list_decisions("wide-gap-oxide")] == [
        "project_created",
        "contract_revised",
    ]


def test_stale_revision_overwrite_is_rejected(tmp_path: Path) -> None:
    store, project = _store(tmp_path)
    revised = project.model_copy(update={"revision": 2, "title": "first edit"})
    store.save_revision(
        revised, expected_revision=1, actor="user", reason="first edit"
    )

    with pytest.raises(ProjectConflictError) as exc_info:
        store.save_revision(
            project.model_copy(update={"revision": 2, "title": "stale edit"}),
            expected_revision=1,
            actor="user",
            reason="stale edit",
        )

    assert exc_info.value.code == "STALE_REVISION"


def test_unconfirmed_project_cannot_run(tmp_path: Path) -> None:
    store, _ = _store(tmp_path)

    with pytest.raises(ProjectTransitionError) as exc_info:
        store.transition(
            "wide-gap-oxide",
            ProjectStatus.RUNNING,
            expected_revision=1,
            actor="system",
            reason="attempt execution",
        )

    assert exc_info.value.code == "INVALID_STATE_TRANSITION"


def test_confirmation_requires_current_supported_assessment(tmp_path: Path) -> None:
    store, project = _store(tmp_path)

    with pytest.raises(ProjectTransitionError) as exc_info:
        store.transition(
            project.project_id,
            ProjectStatus.AWAITING_CONFIRMATION,
            expected_revision=1,
            actor="system",
            reason="prepare confirmation",
        )

    assert exc_info.value.code == "CAPABILITY_GATE_NOT_PASSED"


def test_supported_contract_can_be_confirmed_then_run(tmp_path: Path) -> None:
    store, project = _store(tmp_path)
    gate = PropertyCapabilityGate(default_property_registry())

    awaiting = store.transition(
        project.project_id,
        ProjectStatus.AWAITING_CONFIRMATION,
        expected_revision=1,
        assessment=gate.assess(project),
        actor="system",
        reason="能力检查通过",
    )
    confirmed = store.transition(
        project.project_id,
        ProjectStatus.CONFIRMED,
        expected_revision=2,
        assessment=gate.assess(awaiting),
        actor="user",
        reason="确认筛选合同",
    )
    running = store.transition(
        project.project_id,
        ProjectStatus.RUNNING,
        expected_revision=3,
        actor="system",
        reason="开始执行",
    )

    assert awaiting.status is ProjectStatus.AWAITING_CONFIRMATION
    assert confirmed.confirmed_revision == confirmed.revision == 3
    assert running.status is ProjectStatus.RUNNING
    assert running.revision == 4
    assert len(store.list_decisions(project.project_id)) == 4


def test_contract_cannot_be_edited_after_confirmation_flow_starts(
    tmp_path: Path,
) -> None:
    store, project = _store(tmp_path)
    assessment = PropertyCapabilityGate(default_property_registry()).assess(project)
    awaiting = store.transition(
        project.project_id,
        ProjectStatus.AWAITING_CONFIRMATION,
        expected_revision=1,
        assessment=assessment,
        actor="system",
        reason="ready",
    )

    with pytest.raises(ProjectConflictError) as exc_info:
        store.save_revision(
            awaiting.model_copy(update={"revision": 3, "title": "illegal edit"}),
            expected_revision=2,
            actor="user",
            reason="edit without returning to draft",
        )

    assert exc_info.value.code == "CONTRACT_NOT_EDITABLE"


def test_invalid_terminal_transition_is_rejected(tmp_path: Path) -> None:
    store, project = _store(tmp_path)
    cancelled = store.transition(
        project.project_id,
        ProjectStatus.CANCELLED,
        expected_revision=1,
        actor="user",
        reason="cancel",
    )

    with pytest.raises(ProjectTransitionError) as exc_info:
        store.transition(
            project.project_id,
            ProjectStatus.DRAFT,
            expected_revision=cancelled.revision,
            actor="user",
            reason="attempt reopen",
        )

    assert exc_info.value.code == "INVALID_STATE_TRANSITION"


def test_project_id_cannot_escape_store_root(tmp_path: Path) -> None:
    store = ScreeningProjectStore(tmp_path / "projects")

    with pytest.raises(ValueError, match="invalid project id"):
        store.get("../outside")
