"""Unit tests for Stage 3 workflow context and lightweight protocols (S3-M2)."""

import dataclasses
from datetime import UTC, datetime
from typing import Any, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.runtime import Runtime  # noqa: E402

from materials_screening.llm.mock_provider import MockStructuredProvider  # noqa: E402
from materials_screening.planner.service import PlannerService  # noqa: E402
from materials_screening.planner.settings import Settings  # noqa: E402
from materials_screening.repositories.mock import MockMaterialsRepository  # noqa: E402
from materials_screening.services.export_service import ExportService  # noqa: E402
from materials_screening.services.filter_service import FilterService  # noqa: E402
from materials_screening.services.ranking_service import RankingService  # noqa: E402
from materials_screening.services.validation_service import (
    ValidationService,  # noqa: E402
)
from materials_screening.workflow.context import (  # noqa: E402
    IdGenerator,
    RunArtifactStore,
    WorkflowContext,
)
from materials_screening.workflow.state import ArtifactRef  # noqa: E402


class _FakeArtifactStore:
    """Minimal offline implementation of the RunArtifactStore protocol."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def initialize_run(self, *, run_id: str, metadata: dict[str, Any]) -> None:
        self.calls.append(("initialize_run", run_id, metadata["version"]))

    def put_json(
        self,
        *,
        run_id: str,
        name: str,
        value: object,
        schema_name: str | None = None,
        schema_version: str | None = None,
        item_count: int | None = None,
    ) -> ArtifactRef:
        self.calls.append(("put_json", name))
        return ArtifactRef(
            name=name,
            relative_path=f"artifacts/{name}.json",
            sha256="0" * 64,
            media_type="application/json",
            size_bytes=0,
            item_count=item_count,
            schema_name=schema_name,
            schema_version=schema_version,
        )

    def get_json(self, ref: ArtifactRef) -> object:
        self.calls.append(("get_json", ref.name))
        return {}

    def exists(self, ref: ArtifactRef) -> bool:
        return True

    def verify(self, ref: ArtifactRef) -> bool:
        return True


class _FakeIdGenerator:
    def __init__(self) -> None:
        self.count = 0

    def new_id(self) -> str:
        self.count += 1
        return f"id-{self.count}"


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


def _build_context() -> WorkflowContext:
    return WorkflowContext(
        planner_service=PlannerService(
            settings=Settings(llm_provider="mock"),
            provider=MockStructuredProvider(),
        ),
        materials_repository=MockMaterialsRepository(records=()),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=ExportService(),
        artifact_store=_FakeArtifactStore(),
        clock=_fixed_clock,
        id_generator=_FakeIdGenerator(),
    )


class TestWorkflowContext:
    def test_is_frozen_dataclass(self) -> None:
        assert dataclasses.is_dataclass(WorkflowContext)
        assert WorkflowContext.__dataclass_params__.frozen is True

    def test_field_names(self) -> None:
        expected = {
            "planner_service",
            "materials_repository",
            "filter_service",
            "ranking_service",
            "validation_service",
            "export_service",
            "artifact_store",
            "clock",
            "id_generator",
        }
        assert {field.name for field in dataclasses.fields(WorkflowContext)} == expected

    def test_requires_all_dependencies(self) -> None:
        with pytest.raises(TypeError):
            WorkflowContext()

    def test_holds_injected_services(self) -> None:
        context = _build_context()
        assert isinstance(context.planner_service, PlannerService)
        assert isinstance(context.materials_repository, MockMaterialsRepository)
        assert isinstance(context.filter_service, FilterService)
        assert isinstance(context.ranking_service, RankingService)
        assert isinstance(context.validation_service, ValidationService)
        assert isinstance(context.export_service, ExportService)
        assert context.clock() == _fixed_clock()
        assert context.id_generator.new_id() == "id-1"

    def test_frozen_prevents_reassignment(self) -> None:
        context = _build_context()
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.run_id = "run-1"


class TestLightweightProtocols:
    def test_fake_artifact_store_conforms(self) -> None:
        assert isinstance(_FakeArtifactStore(), RunArtifactStore)

    def test_fake_id_generator_conforms(self) -> None:
        assert isinstance(_FakeIdGenerator(), IdGenerator)

    def test_clock_is_callable(self) -> None:
        assert callable(_fixed_clock)
        assert isinstance(_fixed_clock(), datetime)


class ProbeState(TypedDict, total=False):
    generated_id: str
    clock_value: str
    artifact_name: str


def _probe_node(
    state: ProbeState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, str]:
    """Read injected dependencies from the runtime context."""
    context = runtime.context
    context.artifact_store.initialize_run(
        run_id="run-1",
        metadata={"version": "workflow-v1"},
    )
    ref = context.artifact_store.put_json(
        run_id="run-1",
        name="probe",
        value={"ok": True},
        schema_name="probe-v1",
    )
    return {
        "generated_id": context.id_generator.new_id(),
        "clock_value": context.clock().isoformat(),
        "artifact_name": ref.name,
    }


def _compile_probe_graph() -> object:
    builder = StateGraph(ProbeState, context_schema=WorkflowContext)
    builder.add_node("probe", _probe_node)
    builder.add_edge(START, "probe")
    builder.add_edge("probe", END)
    return builder.compile(
        checkpointer=InMemorySaver(),
        name="workflow-context-smoke-v1",
    )


def _thread_config(thread_id: str) -> dict[str, object]:
    return {"configurable": {"thread_id": thread_id}}


class TestRuntimeContextUsage:
    def test_context_injected_after_compile(self) -> None:
        graph = _compile_probe_graph()
        context = _build_context()
        result = graph.invoke(
            {},
            config=_thread_config("context-thread"),
            context=context,
        )
        assert result == {
            "generated_id": "id-1",
            "clock_value": "2026-08-06T00:00:00+00:00",
            "artifact_name": "probe",
        }
        store = context.artifact_store
        assert isinstance(store, _FakeArtifactStore)
        assert store.calls == [
            ("initialize_run", "run-1", "workflow-v1"),
            ("put_json", "probe"),
        ]

    def test_context_never_enters_checkpoint_state(self) -> None:
        graph = _compile_probe_graph()
        config = _thread_config("state-thread")
        graph.invoke({}, config=config, context=_build_context())

        snapshot = graph.get_state(config)
        forbidden = {
            "context",
            "planner_service",
            "materials_repository",
            "filter_service",
            "ranking_service",
            "validation_service",
            "export_service",
            "artifact_store",
            "clock",
            "id_generator",
        }
        assert forbidden.isdisjoint(snapshot.values)

        history = list(graph.get_state_history(config, limit=5))
        assert history
        assert all(forbidden.isdisjoint(item.values) for item in history)

    def test_compile_does_not_require_services(self) -> None:
        graph = _compile_probe_graph()
        assert graph is not None
