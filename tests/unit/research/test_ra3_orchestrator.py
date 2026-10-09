from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from materials_screening.research import (
    CandidateEvidenceLedger,
    CriterionOperator,
    CriterionRole,
    MaterialCandidate,
    MaterialScope,
    MissingValuePolicy,
    OrchestrationStatus,
    ProjectResourceLimits,
    ProjectStatus,
    ResearchProjectError,
    ScreeningCriterion,
    ScreeningOrchestrator,
    ScreeningProject,
    ScreeningRunStore,
    WorkKind,
    WorkStatus,
)
from materials_screening.research.adapters import MaterialsDatabaseReadAdapter
from materials_screening.research.adapters.models import AdapterBatch
from materials_screening.research.property_registry import default_property_registry
from materials_screening.sub_agents.materials_database.models import (
    QueryResultReference,
)

ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "tests" / "fixtures" / "research" / "benchmark_v1"
UNIT = {
    "band_gap_ev": "eV",
    "energy_above_hull_ev_atom": "eV/atom",
    "density_g_cm3": "g/cm^3",
}


class SnapshotSource:
    def __init__(self) -> None:
        path = BENCHMARK / "database_snapshots" / "materials_project_oxide_300.jsonl"
        self.rows = tuple(
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        )
        self.calls: list[str] = []
        self._adapter = MaterialsDatabaseReadAdapter(default_property_registry())

    def fetch_candidates(
        self, project: ScreeningProject, *, idempotency_key: str
    ) -> AdapterBatch:
        self.calls.append(idempotency_key)
        property_ids = {
            default_property_registry().require(item.property_id).property_id
            for item in project.criteria
        }
        fields = {
            "material_id",
            "formula_pretty",
            "chemsys",
            "elements",
            "spacegroup_number",
            "structure_hash",
            *property_ids,
        }
        rows = tuple(
            {key: value for key, value in row.items() if key in fields}
            for row in self.rows
        )
        query_id = "query-" + hashlib.sha256(idempotency_key.encode()).hexdigest()[:24]
        result = QueryResultReference(
            query_id=query_id,
            source="materials_project",
            matched_count=len(rows),
            returned_count=len(rows),
            fields=tuple(sorted(fields)),
            materials=rows,
            created_at=datetime(2026, 9, 3, tzinfo=UTC),
        )
        return self._adapter.normalize(
            project.project_id,
            result,
            database_version="2026.04.13",
        )


class RecordingLiterature:
    def __init__(self, *, fail_quick_once: bool = False) -> None:
        self.quick_calls: list[tuple[str, tuple[str, ...]]] = []
        self.full_calls: list[tuple[str, tuple[str, ...]]] = []
        self.supplemental_calls: list[str] = []
        self._fail_quick_once = fail_quick_once

    def quick_review(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        *,
        idempotency_key: str,
    ) -> AdapterBatch:
        del project
        ids = tuple(item.candidate_id for item in candidates)
        self.quick_calls.append((idempotency_key, ids))
        if self._fail_quick_once:
            self._fail_quick_once = False
            raise RuntimeError("temporary literature failure")
        return AdapterBatch()

    def full_review(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        *,
        idempotency_key: str,
    ) -> AdapterBatch:
        del project
        ids = tuple(item.candidate_id for item in candidates)
        self.full_calls.append((idempotency_key, ids))
        return AdapterBatch()

    def supplemental_review(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        objective: str,
        *,
        idempotency_key: str,
    ) -> AdapterBatch:
        del project, candidates, objective
        self.supplemental_calls.append(idempotency_key)
        return AdapterBatch()


def _criterion(
    property_id: str,
    operator: CriterionOperator,
    values: tuple[float | bool | str, ...],
    *,
    role: CriterionRole,
) -> ScreeningCriterion:
    return ScreeningCriterion(
        property_id=property_id,
        role=role,
        operator=operator,
        values=values,
        unit=UNIT.get(property_id),
        missing_policy=MissingValuePolicy.EXCLUDE,
        weight=1.0 if role is CriterionRole.SOFT_RANK else None,
    )


def _project_from_case(case: dict[str, object]) -> ScreeningProject:
    constraints = dict(case["constraints"])
    criteria: list[ScreeningCriterion] = []
    for property_id in (
        "band_gap_ev",
        "energy_above_hull_ev_atom",
        "density_g_cm3",
    ):
        bounds = constraints.get(property_id)
        if not isinstance(bounds, list):
            continue
        low, high = bounds
        if low is not None and high is not None:
            operator = CriterionOperator.BETWEEN
            values = (float(low), float(high))
        elif low is not None:
            operator = CriterionOperator.GTE
            values = (float(low),)
        else:
            operator = CriterionOperator.LTE
            values = (float(high),)
        criteria.append(
            _criterion(
                property_id,
                operator,
                values,
                role=CriterionRole.HARD_FILTER,
            )
        )
    for property_id in ("is_stable", "is_metal", "theoretical"):
        if property_id in constraints:
            criteria.append(
                _criterion(
                    property_id,
                    CriterionOperator.EQ,
                    (bool(constraints[property_id]),),
                    role=CriterionRole.HARD_FILTER,
                )
            )
    sort = dict(case["sort"])
    direction = sort["direction"]
    if direction == "target":
        operator = CriterionOperator.TARGET
        values = (float(sort["target"]),)
    else:
        operator = (
            CriterionOperator.PREFER_MIN
            if direction == "asc"
            else CriterionOperator.PREFER_MAX
        )
        values = (0.0,)
    if sort["field"] != "material_id":
        criteria.append(
            _criterion(
                str(sort["field"]),
                operator,
                values,
                role=CriterionRole.SOFT_RANK,
            )
        )
    limit = int(case["limit"])
    scope = MaterialScope(
        required_elements=tuple(constraints.get("required_elements", ())),
        excluded_elements=tuple(constraints.get("excluded_elements", ())),
        formulas=(
            (str(constraints["formula_pretty"]),)
            if "formula_pretty" in constraints
            else ()
        ),
    )
    return ScreeningProject(
        project_id=f"project-{case['case_id']}",
        title=str(case["question"]),
        research_question=str(case["question"]),
        material_scope=scope,
        criteria=tuple(criteria),
        resource_limits=ProjectResourceLimits(
            candidate_snapshot_limit=500,
            shortlist_limit=limit,
            quick_literature_limit=min(20, limit),
            deep_literature_limit=min(5, limit),
        ),
        status=ProjectStatus.CONFIRMED,
        revision=2,
        confirmed_revision=2,
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        updated_at=datetime(2026, 9, 3, tzinfo=UTC),
    )


def _orchestrator(
    tmp_path: Path,
    source: SnapshotSource,
    literature: RecordingLiterature | None,
) -> ScreeningOrchestrator:
    return ScreeningOrchestrator(
        run_store=ScreeningRunStore(tmp_path / "runs"),
        ledger=CandidateEvidenceLedger(tmp_path / "ledger"),
        candidate_source=source,
        literature=literature,
    )


def _cases() -> dict[str, dict[str, object]]:
    items = json.loads(
        (BENCHMARK / "screening_cases" / "cases.json").read_text(encoding="utf-8")
    )
    return {item["case_id"]: item for item in items}


def _tasks() -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(
            (BENCHMARK / "end_to_end_tasks" / "tasks.json").read_text(
                encoding="utf-8"
            )
        )
    )


@pytest.mark.parametrize("task", _tasks(), ids=lambda task: str(task["task_id"]))
def test_three_gold_drafts_form_reproducible_candidate_funnels(
    tmp_path: Path, task: dict[str, object]
) -> None:
    case = _cases()[str(task["screening_case_id"])]
    project = _project_from_case(case)
    source = SnapshotSource()
    literature = RecordingLiterature()
    orchestrator = _orchestrator(tmp_path, source, literature)
    planned = orchestrator.plan(project)

    result = orchestrator.execute(project, planned.run_id)

    assert result.status is OrchestrationStatus.EVIDENCE_READY
    assert all(
        item.status in {WorkStatus.COMPLETED, WorkStatus.SKIPPED}
        for item in result.work_items
    )
    ledger = CandidateEvidenceLedger(tmp_path / "ledger")
    ranked_material_ids = tuple(
        ledger.get_candidate(project.project_id, item.candidate_id).source_material_id
        for item in result.ranked_candidates
    )
    expected = tuple(task["database_expected"]["expected_all_material_ids"])
    assert ranked_material_ids == expected
    outcome_by_id = {item.candidate_id: item for item in result.filter_outcomes}
    assert all(
        outcome_by_id[item.candidate_id].status == "passed"
        for item in result.ranked_candidates
    )
    shortlist_material_ids = tuple(
        ledger.get_candidate(project.project_id, candidate_id).source_material_id
        for candidate_id in result.shortlist_ids
    )
    assert shortlist_material_ids == tuple(
        task["database_expected"]["expected_material_ids"]
    )
    assert len(result.shortlist_ids) <= project.resource_limits.shortlist_limit
    assert len(result.quick_review_ids) <= 20
    assert len(result.full_review_ids) <= 5
    assert len(source.calls) == len(set(source.calls)) == 1
    assert len(literature.quick_calls) == 1
    assert len(literature.full_calls) == 1


def test_failure_preserves_candidates_and_retry_skips_completed_work(
    tmp_path: Path,
) -> None:
    project = _project_from_case(_cases()["screen-001"])
    source = SnapshotSource()
    literature = RecordingLiterature(fail_quick_once=True)
    orchestrator = _orchestrator(tmp_path, source, literature)
    planned = orchestrator.plan(project)

    failed = orchestrator.execute(project, planned.run_id)

    assert failed.status is OrchestrationStatus.PAUSED_FAILURE
    assert len(failed.candidate_ids) == 300
    preserved_evidence = failed.evidence_ids
    quick = next(
        item
        for item in failed.work_items
        if item.kind is WorkKind.LITERATURE_QUICK_REVIEW
    )
    assert quick.status is WorkStatus.FAILED
    resumed = orchestrator.retry_failed(project, failed.run_id)
    assert resumed.status is OrchestrationStatus.EVIDENCE_READY
    assert resumed.evidence_ids == preserved_evidence
    assert len(source.calls) == 1
    assert len(literature.quick_calls) == 2
    assert literature.quick_calls[0][0] == literature.quick_calls[1][0]
    assert len(literature.full_calls) == 1

    repeated = orchestrator.execute(project, resumed.run_id)
    assert repeated == resumed
    assert len(source.calls) == 1
    assert len(literature.full_calls) == 1


def test_zero_results_do_not_relax_contract_or_call_literature(tmp_path: Path) -> None:
    project = _project_from_case(_cases()["screen-013"])
    source = SnapshotSource()
    literature = RecordingLiterature()
    orchestrator = _orchestrator(tmp_path, source, literature)

    result = orchestrator.execute(project, orchestrator.plan(project).run_id)

    assert result.status is OrchestrationStatus.EVIDENCE_READY
    assert not result.ranked_candidates
    assert not result.shortlist_ids
    assert all(outcome.status == "failed" for outcome in result.filter_outcomes)
    assert not literature.quick_calls
    assert not literature.full_calls
    assert len(source.calls) == 1


def test_only_one_supplemental_review_is_allowed(tmp_path: Path) -> None:
    project = _project_from_case(_cases()["screen-001"])
    source = SnapshotSource()
    literature = RecordingLiterature()
    orchestrator = _orchestrator(tmp_path, source, literature)
    initial = orchestrator.execute(project, orchestrator.plan(project).run_id)

    supplemented = orchestrator.request_supplemental_review(
        project,
        initial.run_id,
        objective="补充核验首选候选的实验物相。",
    )

    assert supplemented.status is OrchestrationStatus.EVIDENCE_READY
    assert supplemented.supplemental_used
    assert len(literature.supplemental_calls) == 1
    with pytest.raises(ResearchProjectError) as exc_info:
        orchestrator.request_supplemental_review(
            project,
            supplemented.run_id,
            objective="再次检索",
        )
    assert exc_info.value.code == "SUPPLEMENTAL_LIMIT_REACHED"


def test_run_rejects_modified_or_unconfirmed_contract(tmp_path: Path) -> None:
    project = _project_from_case(_cases()["screen-001"])
    source = SnapshotSource()
    orchestrator = _orchestrator(tmp_path, source, None)
    planned = orchestrator.plan(project)

    changed = project.model_copy(update={"title": "changed after confirmation"})
    with pytest.raises(ResearchProjectError) as exc_info:
        orchestrator.execute(changed, planned.run_id)

    assert exc_info.value.code == "PROJECT_CONTRACT_MISMATCH"
