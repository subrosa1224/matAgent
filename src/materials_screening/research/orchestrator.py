"""Bounded, resumable RA-3 screening orchestrator and candidate funnel."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from pymatgen.core.composition import Composition

from materials_screening.research.adapters.models import AdapterBatch
from materials_screening.research.candidate_ledger import (
    CandidateDecision,
    CandidateEvidenceLedger,
    CandidateTier,
    HardConstraintStatus,
    MaterialCandidate,
)
from materials_screening.research.errors import ResearchProjectError
from materials_screening.research.models import (
    CriterionOperator,
    CriterionRole,
    MissingValuePolicy,
    ProjectStatus,
    ScreeningCriterion,
    ScreeningProject,
)
from materials_screening.research.property_registry import (
    PropertyRegistry,
    default_property_registry,
)
from materials_screening.research.work_plan import (
    FilterOutcome,
    OrchestrationStatus,
    RankedCandidate,
    ScreeningRunState,
    ScreeningRunStore,
    ScreeningWorkItem,
    WorkKind,
    WorkStatus,
    build_screening_plan,
)


class CandidateSourcePort(Protocol):
    """Read candidates produced by a database operation."""

    def fetch_candidates(
        self, project: ScreeningProject, *, idempotency_key: str
    ) -> AdapterBatch: ...


class LiteratureReviewPort(Protocol):
    """Run bounded literature review for explicitly selected candidates."""

    def quick_review(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        *,
        idempotency_key: str,
    ) -> AdapterBatch: ...

    def full_review(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        *,
        idempotency_key: str,
    ) -> AdapterBatch: ...

    def supplemental_review(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        objective: str,
        *,
        idempotency_key: str,
    ) -> AdapterBatch: ...


class CandidateAnalysisPort(Protocol):
    """Analyze an already-frozen candidate set without changing the contract."""

    def analyze(
        self,
        project: ScreeningProject,
        candidates: tuple[MaterialCandidate, ...],
        *,
        idempotency_key: str,
    ) -> AdapterBatch: ...


@dataclass(frozen=True)
class _HandlerResult:
    updates: Mapping[str, Any]
    output_refs: tuple[str, ...] = ()
    skipped: bool = False


class ScreeningOrchestrator:
    """Execute a fixed plan while keeping all domain facts in adapter outputs."""

    def __init__(
        self,
        *,
        run_store: ScreeningRunStore,
        ledger: CandidateEvidenceLedger,
        candidate_source: CandidateSourcePort,
        literature: LiteratureReviewPort | None = None,
        data_analysis: CandidateAnalysisPort | None = None,
        registry: PropertyRegistry | None = None,
    ) -> None:
        self._runs = run_store
        self._ledger = ledger
        self._source = candidate_source
        self._literature = literature
        self._analysis = data_analysis
        self._registry = registry or default_property_registry()

    def plan(self, project: ScreeningProject) -> ScreeningRunState:
        return self._runs.create(build_screening_plan(project))

    def execute(self, project: ScreeningProject, run_id: str) -> ScreeningRunState:
        state = self._runs.get(run_id)
        self._validate_project(project, state)
        while True:
            state = self._runs.get(run_id)
            failed = next(
                (item for item in state.work_items if item.status is WorkStatus.FAILED),
                None,
            )
            if failed is not None:
                return state
            pending = self._next_ready(state)
            if pending is None:
                terminal = all(
                    item.status in {WorkStatus.COMPLETED, WorkStatus.SKIPPED}
                    for item in state.work_items
                )
                if terminal and state.status is not OrchestrationStatus.EVIDENCE_READY:
                    state = self._save_state(
                        state,
                        status=OrchestrationStatus.EVIDENCE_READY,
                    )
                return state
            state = self._start_item(state, pending.work_id)
            running = self._item(state, pending.work_id)
            try:
                result = self._handle(project, state, running)
            except Exception as exc:  # deterministic pause boundary
                return self._fail_item(state, running.work_id, exc)
            state = self._finish_item(state, running.work_id, result)

    def retry_failed(self, project: ScreeningProject, run_id: str) -> ScreeningRunState:
        state = self._runs.get(run_id)
        self._validate_project(project, state)
        failed = next(
            (item for item in state.work_items if item.status is WorkStatus.FAILED),
            None,
        )
        if failed is None:
            return self.execute(project, run_id)
        if failed.attempt_count >= failed.max_attempts:
            raise ResearchProjectError(
                "RETRY_LIMIT_REACHED",
                f"work item {failed.work_id} reached its retry limit",
            )
        reset = failed.model_copy(
            update={
                "status": WorkStatus.PENDING,
                "error_code": None,
                "error_message": None,
            }
        )
        state = self._replace_item(
            state,
            reset,
            status=OrchestrationStatus.RUNNING,
        )
        return self.execute(project, run_id)

    def request_supplemental_review(
        self,
        project: ScreeningProject,
        run_id: str,
        *,
        objective: str,
    ) -> ScreeningRunState:
        state = self._runs.get(run_id)
        self._validate_project(project, state)
        if state.supplemental_used:
            raise ResearchProjectError(
                "SUPPLEMENTAL_LIMIT_REACHED",
                "a project may run at most one supplemental review",
            )
        if state.status is not OrchestrationStatus.EVIDENCE_READY:
            raise ResearchProjectError(
                "SUPPLEMENTAL_NOT_READY",
                "supplemental review requires the initial funnel to finish",
            )
        clean_objective = " ".join(objective.split())
        if not clean_objective or len(clean_objective) > 1000:
            raise ValueError("supplemental objective must be 1-1000 characters")
        work_id = _stable_id("work", run_id, "supplemental")
        dependency = state.work_items[-1].work_id
        item = ScreeningWorkItem(
            work_id=work_id,
            kind=WorkKind.SUPPLEMENTAL_REVIEW,
            agent_name="literature",
            objective=clean_objective,
            depends_on=(dependency,),
            max_attempts=2,
        )
        state = self._save_state(
            state,
            work_items=(*state.work_items, item),
            supplemental_used=True,
            status=OrchestrationStatus.RUNNING,
        )
        return self.execute(project, state.run_id)

    def _handle(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        handlers = {
            WorkKind.DATABASE_QUERY: self._query,
            WorkKind.HARD_FILTER: self._hard_filter,
            WorkKind.RANK_CANDIDATES: self._rank,
            WorkKind.SELECT_QUICK_REVIEW: self._select_quick,
            WorkKind.LITERATURE_QUICK_REVIEW: self._quick_review,
            WorkKind.SELECT_FULL_REVIEW: self._select_full,
            WorkKind.LITERATURE_FULL_REVIEW: self._full_review,
            WorkKind.DATA_ANALYSIS: self._analyze,
            WorkKind.SUPPLEMENTAL_REVIEW: self._supplemental_review,
        }
        return handlers[item.kind](project, state, item)

    def _query(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        batch = self._source.fetch_candidates(project, idempotency_key=item.work_id)
        if len(batch.candidates) > project.resource_limits.candidate_snapshot_limit:
            raise ResearchProjectError(
                "CANDIDATE_LIMIT_EXCEEDED",
                "candidate source exceeded the confirmed snapshot limit",
            )
        refs = self._persist_batch(batch)
        candidate_ids = tuple(candidate.candidate_id for candidate in batch.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ResearchProjectError(
                "DUPLICATE_CANDIDATE", "candidate source returned duplicate ids"
            )
        return _HandlerResult(
            updates={
                "candidate_ids": candidate_ids,
                "evidence_ids": _unique((*state.evidence_ids, *refs[0])),
                "claim_ids": _unique((*state.claim_ids, *refs[1])),
                "warnings": _unique((*state.warnings, *batch.warnings)),
            },
            output_refs=candidate_ids,
        )

    def _hard_filter(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        del item
        outcomes = tuple(
            self._filter_candidate(
                project, self._candidate(project.project_id, candidate_id)
            )
            for candidate_id in state.candidate_ids
        )
        return _HandlerResult(
            updates={"filter_outcomes": outcomes},
            output_refs=tuple(outcome.candidate_id for outcome in outcomes),
        )

    def _rank(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        del item
        status_by_id = {
            outcome.candidate_id: outcome for outcome in state.filter_outcomes
        }
        passed = [
            self._candidate(project.project_id, candidate_id)
            for candidate_id in state.candidate_ids
            if status_by_id[candidate_id].status == "passed"
        ]
        ranked, warnings = self._rank_candidates(project, passed)
        ranking_by_id = {item.candidate_id: item for item in ranked}
        decision_ids: list[str] = []
        for candidate_id in state.candidate_ids:
            outcome = status_by_id[candidate_id]
            ranking = ranking_by_id.get(candidate_id)
            decision = CandidateDecision(
                decision_id=_stable_id("decision", state.run_id, candidate_id),
                project_id=project.project_id,
                candidate_id=candidate_id,
                hard_constraint_status=HardConstraintStatus(outcome.status),
                exclusion_reasons=(
                    outcome.reasons if outcome.status == "failed" else ()
                ),
                ranking_score=None if ranking is None else ranking.score,
                ranking_components={} if ranking is None else ranking.components,
                final_tier=CandidateTier.UNRANKED,
                rationale=outcome.reasons,
                evidence_ids=tuple(
                    value.evidence_id
                    for value in self._candidate(
                        project.project_id, candidate_id
                    ).property_values
                ),
                created_at=state.created_at,
            )
            self._ledger.add_decision(decision)
            decision_ids.append(decision.decision_id)
        shortlist = tuple(
            item.candidate_id
            for item in ranked[: project.resource_limits.shortlist_limit]
        )
        return _HandlerResult(
            updates={
                "ranked_candidates": ranked,
                "shortlist_ids": shortlist,
                "warnings": _unique((*state.warnings, *warnings)),
            },
            output_refs=tuple(decision_ids),
        )

    def _select_quick(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        del item
        selected = state.shortlist_ids[: project.resource_limits.quick_literature_limit]
        return _HandlerResult(
            updates={"quick_review_ids": selected},
            output_refs=selected,
        )

    def _quick_review(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        candidates = self._candidates(project.project_id, state.quick_review_ids)
        if not candidates or self._literature is None:
            warning = (
                () if not candidates else ("literature quick-review port unavailable",)
            )
            return _HandlerResult(
                updates={"warnings": _unique((*state.warnings, *warning))},
                skipped=True,
            )
        batch = self._literature.quick_review(
            project, candidates, idempotency_key=item.work_id
        )
        return self._batch_result(state, batch)

    def _select_full(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        del item
        selected = state.quick_review_ids[
            : project.resource_limits.deep_literature_limit
        ]
        return _HandlerResult(
            updates={"full_review_ids": selected},
            output_refs=selected,
        )

    def _full_review(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        candidates = self._candidates(project.project_id, state.full_review_ids)
        if not candidates or self._literature is None:
            warning = (
                () if not candidates else ("literature full-review port unavailable",)
            )
            return _HandlerResult(
                updates={"warnings": _unique((*state.warnings, *warning))},
                skipped=True,
            )
        batch = self._literature.full_review(
            project, candidates, idempotency_key=item.work_id
        )
        return self._batch_result(state, batch)

    def _analyze(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        if self._analysis is None or not state.shortlist_ids:
            return _HandlerResult(updates={}, skipped=True)
        batch = self._analysis.analyze(
            project,
            self._candidates(project.project_id, state.shortlist_ids),
            idempotency_key=item.work_id,
        )
        return self._batch_result(state, batch)

    def _supplemental_review(
        self,
        project: ScreeningProject,
        state: ScreeningRunState,
        item: ScreeningWorkItem,
    ) -> _HandlerResult:
        if self._literature is None:
            return _HandlerResult(
                updates={
                    "warnings": _unique(
                        (*state.warnings, "literature supplemental port unavailable")
                    )
                },
                skipped=True,
            )
        batch = self._literature.supplemental_review(
            project,
            self._candidates(project.project_id, state.full_review_ids),
            item.objective,
            idempotency_key=item.work_id,
        )
        return self._batch_result(state, batch)

    def _batch_result(
        self, state: ScreeningRunState, batch: AdapterBatch
    ) -> _HandlerResult:
        evidence_ids, claim_ids = self._persist_batch(batch)
        return _HandlerResult(
            updates={
                "evidence_ids": _unique((*state.evidence_ids, *evidence_ids)),
                "claim_ids": _unique((*state.claim_ids, *claim_ids)),
                "warnings": _unique((*state.warnings, *batch.warnings)),
            },
            output_refs=(*evidence_ids, *claim_ids),
        )

    def _persist_batch(
        self, batch: AdapterBatch
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        evidence_by_candidate: dict[str, list[Any]] = {}
        for evidence in batch.evidence:
            if evidence.candidate_id is not None:
                evidence_by_candidate.setdefault(evidence.candidate_id, []).append(
                    evidence
                )
        consumed: set[str] = set()
        for candidate in batch.candidates:
            needed = {value.evidence_id for value in candidate.property_values}
            bundled = tuple(
                evidence
                for evidence in evidence_by_candidate.get(candidate.candidate_id, [])
                if evidence.evidence_id in needed
            )
            self._ledger.add_candidate_bundle(candidate, bundled)
            consumed.update(item.evidence_id for item in bundled)
        for evidence in batch.evidence:
            if evidence.evidence_id not in consumed:
                self._ledger.add_evidence(evidence)
        for claim in batch.claims:
            self._ledger.add_claim(claim)
        for link in batch.links:
            self._ledger.add_link(link)
        return (
            tuple(evidence.evidence_id for evidence in batch.evidence),
            tuple(claim.claim_id for claim in batch.claims),
        )

    def _filter_candidate(
        self, project: ScreeningProject, candidate: MaterialCandidate
    ) -> FilterOutcome:
        failures: list[str] = []
        unknown: list[str] = []
        scope = project.material_scope
        elements = set(candidate.chemsys.split("-"))
        if not set(scope.required_elements).issubset(elements):
            failures.append("missing_required_element")
        if set(scope.excluded_elements) & elements:
            failures.append("contains_excluded_element")
        if scope.chemsys and candidate.chemsys != scope.chemsys:
            failures.append("chemsys_mismatch")
        if scope.formulas and _formula(candidate.formula) not in {
            _formula(formula) for formula in scope.formulas
        }:
            failures.append("formula_mismatch")
        if (
            scope.material_ids
            and candidate.source_material_id not in scope.material_ids
        ):
            failures.append("material_id_mismatch")
        values = {value.property_id: value.value for value in candidate.property_values}
        for criterion in project.criteria:
            if criterion.role is not CriterionRole.HARD_FILTER:
                continue
            property_id = self._registry.require(criterion.property_id).property_id
            if property_id not in values:
                if criterion.missing_policy is MissingValuePolicy.FAIL_PROJECT:
                    raise ResearchProjectError(
                        "REQUIRED_PROPERTY_MISSING",
                        f"{candidate.candidate_id} lacks {property_id}",
                    )
                target = (
                    unknown
                    if criterion.missing_policy is MissingValuePolicy.KEEP_WITH_WARNING
                    else failures
                )
                target.append(f"{property_id}_missing")
                continue
            if not _matches(values[property_id], criterion):
                failures.append(f"{property_id}_{criterion.operator.value}_failed")
        if failures:
            return FilterOutcome(
                candidate_id=candidate.candidate_id,
                status="failed",
                reasons=tuple(failures),
            )
        if unknown:
            return FilterOutcome(
                candidate_id=candidate.candidate_id,
                status="unknown",
                reasons=tuple(unknown),
            )
        return FilterOutcome(candidate_id=candidate.candidate_id, status="passed")

    def _rank_candidates(
        self,
        project: ScreeningProject,
        candidates: list[MaterialCandidate],
    ) -> tuple[tuple[RankedCandidate, ...], tuple[str, ...]]:
        objectives = tuple(
            criterion
            for criterion in project.criteria
            if criterion.role is CriterionRole.SOFT_RANK
        )
        if not objectives:
            ordered = sorted(
                candidates, key=lambda candidate: candidate.source_material_id
            )
            return (
                tuple(
                    RankedCandidate(candidate_id=item.candidate_id, rank=index, score=0)
                    for index, item in enumerate(ordered, start=1)
                ),
                (),
            )
        values_by_candidate = {
            candidate.candidate_id: {
                value.property_id: value.value for value in candidate.property_values
            }
            for candidate in candidates
        }
        warnings: list[str] = []
        rankable: list[MaterialCandidate] = []
        for candidate in candidates:
            missing_excluded = False
            for criterion in objectives:
                property_id = self._registry.require(criterion.property_id).property_id
                if property_id in values_by_candidate[candidate.candidate_id]:
                    continue
                if criterion.missing_policy is MissingValuePolicy.FAIL_PROJECT:
                    raise ResearchProjectError(
                        "RANKING_PROPERTY_MISSING",
                        f"{candidate.candidate_id} lacks {property_id}",
                    )
                if criterion.missing_policy is MissingValuePolicy.EXCLUDE:
                    missing_excluded = True
                warnings.append(
                    f"{candidate.candidate_id}: ranking value {property_id} missing"
                )
            if not missing_excluded:
                rankable.append(candidate)
        ranges = self._ranking_ranges(rankable, objectives, values_by_candidate)
        scored: list[tuple[str, str, float, dict[str, float]]] = []
        total_weight = sum(float(criterion.weight or 0) for criterion in objectives)
        for candidate in rankable:
            components: dict[str, float] = {}
            weighted = 0.0
            for index, criterion in enumerate(objectives):
                property_id = self._registry.require(criterion.property_id).property_id
                raw = values_by_candidate[candidate.candidate_id].get(property_id)
                component = self._component(raw, criterion, ranges[index])
                key = f"{property_id}:{index}"
                components[key] = round(component, 8)
                weighted += component * float(criterion.weight or 0)
            score = 0.0 if total_weight == 0 else weighted / total_weight
            scored.append(
                (
                    candidate.candidate_id,
                    candidate.source_material_id,
                    round(score, 8),
                    components,
                )
            )
        scored.sort(key=lambda item: (-item[2], item[1]))
        return (
            tuple(
                RankedCandidate(
                    candidate_id=candidate_id,
                    rank=index,
                    score=score,
                    components=components,
                )
                for index, (candidate_id, _, score, components) in enumerate(
                    scored, start=1
                )
            ),
            tuple(warnings),
        )

    def _ranking_ranges(
        self,
        candidates: list[MaterialCandidate],
        objectives: tuple[ScreeningCriterion, ...],
        values: dict[str, dict[str, Any]],
    ) -> tuple[tuple[float, float] | None, ...]:
        ranges: list[tuple[float, float] | None] = []
        for criterion in objectives:
            property_id = self._registry.require(criterion.property_id).property_id
            numeric: list[float] = []
            for candidate in candidates:
                value = values[candidate.candidate_id].get(property_id)
                if _is_number(value):
                    numeric.append(float(cast(float | int, value)))
            ranges.append((min(numeric), max(numeric)) if numeric else None)
        return tuple(ranges)

    @staticmethod
    def _component(
        raw: Any,
        criterion: ScreeningCriterion,
        bounds: tuple[float, float] | None,
    ) -> float:
        if not _is_number(raw) or bounds is None:
            return 0.0
        value = float(raw)
        low, high = bounds
        span = high - low
        if criterion.operator is CriterionOperator.PREFER_MIN:
            return 1.0 if span == 0 else (high - value) / span
        if criterion.operator is CriterionOperator.PREFER_MAX:
            return 1.0 if span == 0 else (value - low) / span
        if criterion.operator is CriterionOperator.TARGET:
            target = float(criterion.values[0])
            distance = max(abs(low - target), abs(high - target))
            return (
                1.0 if distance == 0 else max(0.0, 1 - abs(value - target) / distance)
            )
        raise ResearchProjectError(
            "INVALID_RANKING_OPERATOR", "unsupported ranking operator"
        )

    def _persist_state_updates(
        self, state: ScreeningRunState, updates: Mapping[str, Any]
    ) -> ScreeningRunState:
        return self._save_state(state, **dict(updates))

    def _start_item(self, state: ScreeningRunState, work_id: str) -> ScreeningRunState:
        item = self._item(state, work_id)
        running = item.model_copy(
            update={
                "status": WorkStatus.RUNNING,
                "attempt_count": item.attempt_count + 1,
            }
        )
        return self._replace_item(state, running, status=OrchestrationStatus.RUNNING)

    def _finish_item(
        self,
        state: ScreeningRunState,
        work_id: str,
        result: _HandlerResult,
    ) -> ScreeningRunState:
        state = self._persist_state_updates(state, result.updates)
        item = self._item(state, work_id)
        completed = item.model_copy(
            update={
                "status": WorkStatus.SKIPPED
                if result.skipped
                else WorkStatus.COMPLETED,
                "output_refs": result.output_refs,
                "error_code": None,
                "error_message": None,
            }
        )
        return self._replace_item(state, completed)

    def _fail_item(
        self, state: ScreeningRunState, work_id: str, error: Exception
    ) -> ScreeningRunState:
        item = self._item(state, work_id)
        failed = item.model_copy(
            update={
                "status": WorkStatus.FAILED,
                "error_code": getattr(error, "code", type(error).__name__),
                "error_message": str(error)[:1000] or type(error).__name__,
            }
        )
        return self._replace_item(
            state,
            failed,
            status=OrchestrationStatus.PAUSED_FAILURE,
        )

    def _replace_item(
        self,
        state: ScreeningRunState,
        replacement: ScreeningWorkItem,
        *,
        status: OrchestrationStatus | None = None,
    ) -> ScreeningRunState:
        items = tuple(
            replacement if item.work_id == replacement.work_id else item
            for item in state.work_items
        )
        return self._save_state(
            state,
            work_items=items,
            status=status or state.status,
        )

    def _save_state(
        self, state: ScreeningRunState, **updates: Any
    ) -> ScreeningRunState:
        payload = state.model_dump(mode="python")
        payload.update(updates)
        payload.update(
            revision=state.revision + 1,
            updated_at=datetime.now(UTC),
        )
        updated = ScreeningRunState.model_validate(payload)
        return self._runs.save(updated, expected_revision=state.revision)

    @staticmethod
    def _next_ready(state: ScreeningRunState) -> ScreeningWorkItem | None:
        status = {item.work_id: item.status for item in state.work_items}
        return next(
            (
                item
                for item in state.work_items
                if item.status is WorkStatus.PENDING
                and all(
                    status[dependency] in {WorkStatus.COMPLETED, WorkStatus.SKIPPED}
                    for dependency in item.depends_on
                )
            ),
            None,
        )

    @staticmethod
    def _item(state: ScreeningRunState, work_id: str) -> ScreeningWorkItem:
        return next(item for item in state.work_items if item.work_id == work_id)

    def _candidate(self, project_id: str, candidate_id: str) -> MaterialCandidate:
        return self._ledger.get_candidate(project_id, candidate_id)

    def _candidates(
        self, project_id: str, candidate_ids: tuple[str, ...]
    ) -> tuple[MaterialCandidate, ...]:
        return tuple(self._candidate(project_id, item) for item in candidate_ids)

    @staticmethod
    def _validate_project(project: ScreeningProject, state: ScreeningRunState) -> None:
        digest = hashlib.sha256(project.model_dump_json().encode()).hexdigest()
        if (
            project.status is not ProjectStatus.CONFIRMED
            or project.confirmed_revision != project.revision
            or project.project_id != state.project_id
            or project.revision != state.project_revision
            or digest != state.project_contract_sha256
        ):
            raise ResearchProjectError(
                "PROJECT_CONTRACT_MISMATCH",
                "run requires its exact currently confirmed project contract",
            )


def _matches(actual: Any, criterion: ScreeningCriterion) -> bool:
    expected = criterion.values
    operator = criterion.operator
    if operator is CriterionOperator.EQ:
        return bool(actual == expected[0])
    if not _is_number(actual) or not all(_is_number(value) for value in expected):
        return False
    value = float(actual)
    if operator is CriterionOperator.GTE:
        return value >= float(expected[0])
    if operator is CriterionOperator.LTE:
        return value <= float(expected[0])
    if operator is CriterionOperator.BETWEEN:
        return float(expected[0]) <= value <= float(expected[1])
    return False


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _formula(value: str) -> str:
    try:
        return Composition(value).reduced_formula
    except (TypeError, ValueError):
        return value.strip()


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
