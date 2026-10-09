"""Unit tests for the workflow graph builder (S3-M5)."""

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from materials_screening.models import (  # noqa: E402
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
)
from materials_screening.planner.models import (  # noqa: E402
    PlannerResult,
    PlannerStatus,
)
from materials_screening.repositories.base import RetrievalResult  # noqa: E402
from materials_screening.services.filter_service import FilterService  # noqa: E402
from materials_screening.services.ranking_service import RankingService  # noqa: E402
from materials_screening.services.validation_service import (
    ValidationService,  # noqa: E402
)
from materials_screening.workflow.artifact_store import (  # noqa: E402
    FileRunArtifactStore,
)
from materials_screening.workflow.context import WorkflowContext  # noqa: E402
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.graph_builder import (  # noqa: E402
    GRAPH_NAME,
    compile_workflow,
    create_workflow_builder,
)
from materials_screening.workflow.input_output import (  # noqa: E402
    WorkflowGraphInput,
    WorkflowOutput,
)
from materials_screening.workflow.nodes import (  # noqa: E402
    NODE_EXPORT_RESULTS,
    NODE_FILTER_MATERIALS,
    NODE_FINALIZE_FAILURE,
    NODE_FINALIZE_NO_RESULTS,
    NODE_FINALIZE_PLANNER_STOP,
    NODE_FINALIZE_SUCCESS,
    NODE_INITIALIZE_RUN,
    NODE_RANK_MATERIALS,
    NODE_RESOLVE_REQUEST,
    NODE_RETRIEVE_MATERIALS,
    NODE_VALIDATE_RESULTS,
)
from materials_screening.workflow.state import WorkflowStatus  # noqa: E402

_START = "__start__"
_END = "__end__"
_ALL_NODES = {
    NODE_INITIALIZE_RUN,
    NODE_RESOLVE_REQUEST,
    NODE_RETRIEVE_MATERIALS,
    NODE_FILTER_MATERIALS,
    NODE_RANK_MATERIALS,
    NODE_VALIDATE_RESULTS,
    NODE_EXPORT_RESULTS,
    NODE_FINALIZE_SUCCESS,
    NODE_FINALIZE_NO_RESULTS,
    NODE_FINALIZE_PLANNER_STOP,
    NODE_FINALIZE_FAILURE,
}
_TERMINALS = {
    NODE_FINALIZE_SUCCESS,
    NODE_FINALIZE_NO_RESULTS,
    NODE_FINALIZE_PLANNER_STOP,
    NODE_FINALIZE_FAILURE,
}


def _compiled() -> Any:
    return compile_workflow(checkpointer=InMemorySaver())


def _edges() -> list[tuple[str, str]]:
    return [(edge.source, edge.target) for edge in _compiled().get_graph().edges]


class TestGraphTopology:
    def test_node_set(self) -> None:
        nodes = set(_compiled().get_graph().nodes) - {_START, _END}
        assert nodes == _ALL_NODES

    def test_edge_set(self) -> None:
        expected = {
            (_START, NODE_INITIALIZE_RUN),
            (NODE_INITIALIZE_RUN, NODE_RESOLVE_REQUEST),
            (NODE_RESOLVE_REQUEST, NODE_RETRIEVE_MATERIALS),
            (NODE_RESOLVE_REQUEST, NODE_FINALIZE_PLANNER_STOP),
            (NODE_RESOLVE_REQUEST, NODE_FINALIZE_FAILURE),
            (NODE_RETRIEVE_MATERIALS, NODE_FILTER_MATERIALS),
            (NODE_RETRIEVE_MATERIALS, NODE_FINALIZE_FAILURE),
            (NODE_FILTER_MATERIALS, NODE_RANK_MATERIALS),
            (NODE_FILTER_MATERIALS, NODE_FINALIZE_NO_RESULTS),
            (NODE_FILTER_MATERIALS, NODE_FINALIZE_FAILURE),
            (NODE_RANK_MATERIALS, NODE_VALIDATE_RESULTS),
            (NODE_RANK_MATERIALS, NODE_FINALIZE_FAILURE),
            (NODE_VALIDATE_RESULTS, NODE_EXPORT_RESULTS),
            (NODE_VALIDATE_RESULTS, NODE_FINALIZE_FAILURE),
            (NODE_EXPORT_RESULTS, NODE_FINALIZE_SUCCESS),
            (NODE_EXPORT_RESULTS, NODE_FINALIZE_FAILURE),
            (NODE_FINALIZE_SUCCESS, _END),
            (NODE_FINALIZE_NO_RESULTS, _END),
            (NODE_FINALIZE_PLANNER_STOP, _END),
            (NODE_FINALIZE_FAILURE, _END),
        }
        assert set(_edges()) == expected

    def test_all_nodes_and_terminals_reachable(self) -> None:
        edges = _edges()
        reachable = {_START}
        changed = True
        while changed:
            changed = False
            for source, target in edges:
                if source in reachable and target not in reachable:
                    reachable.add(target)
                    changed = True
        assert _ALL_NODES | {_START, _END} <= reachable
        assert reachable >= _TERMINALS

    def test_no_business_loops(self) -> None:
        adjacency: dict[str, list[str]] = defaultdict(list)
        for source, target in _edges():
            adjacency[source].append(target)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return False
            if node in visited:
                return True
            visiting.add(node)
            for target in adjacency.get(node, ()):
                if not visit(target):
                    return False
            visiting.remove(node)
            visited.add(node)
            return True

        assert all(visit(node) for node in list(adjacency))


class TestGraphSchemas:
    def test_builder_schemas(self) -> None:
        builder = create_workflow_builder()
        assert builder.context_schema is WorkflowContext
        assert builder.input_schema is WorkflowGraphInput
        assert builder.output_schema is WorkflowOutput

    def test_compile_name_contains_workflow_v1(self) -> None:
        assert _compiled().name == GRAPH_NAME
        assert "workflow-v1" in _compiled().name

    def test_compile_with_new_saver_per_call(self) -> None:
        first = compile_workflow(checkpointer=InMemorySaver())
        second = compile_workflow(checkpointer=InMemorySaver())
        assert first is not second
        assert first.name == second.name


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _RecordsRepository:
    def __init__(self, records: tuple[MaterialRecord, ...]) -> None:
        self._records = records
        self.calls = 0

    def search(self, request: object) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=self._records,
        )

    def healthcheck(self) -> bool:
        return True


class _FakePlanner:
    def __init__(self, result: PlannerResult | None) -> None:
        self._result = result
        self.parse_calls: list[str] = []

    def parse(self, query: str) -> PlannerResult:
        self.parse_calls.append(query)
        if self._result is None:
            raise AssertionError("planner must not be called in request mode")
        return self._result


class _IdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"id-{self.count}"


def _record(material_id: str, band_gap: float) -> MaterialRecord:
    provenance = tuple(
        PropertyProvenance(
            property_name=name,
            source="materials_project",
            source_material_id=material_id,
            value_type=PropertyValueType.DFT_CALCULATED,
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
        )
        for name in ("band_gap_ev", "energy_above_hull_ev_atom", "is_metal")
    )
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=material_id,
        elements=("Si",),
        band_gap_ev=band_gap,
        energy_above_hull_ev_atom=0.0,
        is_metal=False,
        provenance=provenance,
    )


def _build_context(
    *,
    planner: _FakePlanner,
    repository: _RecordsRepository,
    run_root: Any,
) -> WorkflowContext:
    return WorkflowContext(
        planner_service=planner,
        materials_repository=repository,
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=_fixed_clock,
        id_generator=_IdGenerator(),
    )


class TestEndToEndInvocation:
    def test_request_mode_completes(self, tmp_path: Any) -> None:
        repository = _RecordsRepository((_record("mp-1", 1.5), _record("mp-2", 1.6)))
        context = _build_context(
            planner=_FakePlanner(None),
            repository=repository,
            run_root=tmp_path / "runs",
        )
        graph = compile_workflow(checkpointer=InMemorySaver())
        result = graph.invoke(
            {
                "run_id": "run-1",
                "workflow_version": "workflow-v1",
                "input_mode": "request",
                "raw_request": {
                    "limit": 10,
                    "band_gap_ev": {"min": 1.0, "max": 2.0},
                },
            },
            config={"configurable": {"thread_id": "thread-1"}, "recursion_limit": 32},
            context=context,
        )
        assert result["status"] == WorkflowStatus.COMPLETED.value
        assert result["validation_passed"] is True
        assert result["retrieved_count"] == 2
        assert result["exports"]
        assert repository.calls == 1

    def test_query_mode_planner_stop(self, tmp_path: Any) -> None:
        planner = _FakePlanner(
            PlannerResult(
                status=PlannerStatus.NEEDS_CLARIFICATION,
                query="find materials",
                clarification_question="please clarify",
            )
        )
        repository = _RecordsRepository(())
        context = _build_context(
            planner=planner,
            repository=repository,
            run_root=tmp_path / "runs",
        )
        graph = compile_workflow(checkpointer=InMemorySaver())
        result = graph.invoke(
            {
                "run_id": "run-2",
                "workflow_version": "workflow-v1",
                "input_mode": "query",
                "user_query": "find materials",
            },
            config={"configurable": {"thread_id": "thread-2"}, "recursion_limit": 32},
            context=context,
        )
        assert result["status"] == WorkflowStatus.NEEDS_CLARIFICATION.value
        assert result["planner_status"] == "needs_clarification"
        assert result["clarification_question"] == "please clarify"
        assert result["retrieved_count"] == 0
        assert result.get("exports") is None
        assert repository.calls == 0
