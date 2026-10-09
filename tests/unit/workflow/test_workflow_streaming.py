"""Unit tests for WorkflowRunner updates streaming (S3-M6)."""

import re
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.errors import GraphRecursionError  # noqa: E402

from materials_screening.models import (  # noqa: E402
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
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
from materials_screening.workflow.checkpointer import (  # noqa: E402
    create_checkpointer_handle,
)
from materials_screening.workflow.context import WorkflowContext  # noqa: E402
from materials_screening.workflow.errors import (  # noqa: E402
    WorkflowCheckpointError,
)
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.input_output import WorkflowInput  # noqa: E402
from materials_screening.workflow.runner import (  # noqa: E402
    StreamEvent,
    WorkflowRunner,
)
from materials_screening.workflow.settings import WorkflowSettings  # noqa: E402
from materials_screening.workflow.state import WorkflowStatus  # noqa: E402

_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _RecordsRepository:
    def __init__(self, records: tuple[MaterialRecord, ...]) -> None:
        self._records = records

    def search(self, request: object) -> RetrievalResult:
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=self._records,
        )

    def healthcheck(self) -> bool:
        return True


class _FakePlanner:
    def parse(self, query: str) -> object:
        raise AssertionError("planner must not be called in request mode")


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


def _build_context(run_root: Any) -> WorkflowContext:
    return WorkflowContext(
        planner_service=_FakePlanner(),
        materials_repository=_RecordsRepository(
            (_record("mp-1", 1.5), _record("mp-2", 1.6))
        ),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=_fixed_clock,
        id_generator=_IdGenerator(),
    )


def _build_runner(tmp_path: Any) -> WorkflowRunner:
    return WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=_build_context(tmp_path / "runs"),
        checkpointer=create_checkpointer_handle(backend="memory"),
    )


def _request_input() -> WorkflowInput:
    return WorkflowInput(
        request={
            "limit": 10,
            "band_gap_ev": {"min": 1.0, "max": 2.0},
        }
    )


class TestStreaming:
    def test_stream_yields_events_in_node_order(self, tmp_path: Any) -> None:
        runner = _build_runner(tmp_path)
        events = list(runner.stream(_request_input()))
        assert [event.node for event in events] == [
            "initialize_run",
            "resolve_request",
            "retrieve_materials",
            "filter_materials",
            "rank_materials",
            "validate_results",
            "export_results",
            "finalize_success",
        ]
        assert [event.status for event in events] == [
            "initializing",
            "ready_for_retrieval",
            "retrieving",
            "filtering",
            "ranking",
            "validating",
            "exporting",
            "completed",
        ]
        assert all(isinstance(event, StreamEvent) for event in events)
        runner.close()

    def test_stream_events_contain_only_safe_fields(self, tmp_path: Any) -> None:
        runner = _build_runner(tmp_path)
        events = list(runner.stream(_request_input()))
        assert events
        for event in events:
            dumped = event.model_dump()
            assert set(dumped) == {"node", "status", "counts", "message"}
            assert all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in dumped["counts"].values()
            )
        serialized = str([event.model_dump() for event in events])
        for sensitive in (
            "material_id",
            "band_gap_ev",
            "records",
            "structure",
            "reasoning",
            "raw_output",
            "raw_response",
            "api_key",
        ):
            assert sensitive not in serialized
        runner.close()

    def test_to_stream_event_filters_sensitive_data(self) -> None:
        chunk = {
            "type": "updates",
            "ns": (),
            "data": {
                "retrieve_materials": {
                    "status": "retrieving",
                    "events": [
                        {
                            "event_id": "run-1:retrieve_materials",
                            "run_id": "run-1",
                            "node": "retrieve_materials",
                            "event_type": "retrieved",
                            "created_at": "2026-08-06T00:00:00Z",
                            "status": "retrieving",
                            "message": "retrieved=2",
                            "metrics": {
                                "retrieved_count": 2,
                                "workflow_version": "workflow-v1",
                            },
                        }
                    ],
                    "records": [{"material_id": "mp-1", "band_gap_ev": 1.5}],
                    "raw_response": "model raw",
                    "reasoning": "secret reasoning",
                }
            },
        }
        event = WorkflowRunner._to_stream_event(chunk)
        assert event is not None
        assert event.model_dump() == {
            "node": "retrieve_materials",
            "status": "retrieving",
            "counts": {"retrieved_count": 2},
            "message": "retrieved=2",
        }

    def test_to_stream_event_ignores_non_updates_chunks(self) -> None:
        assert WorkflowRunner._to_stream_event({"type": "values", "data": {}}) is None
        assert WorkflowRunner._to_stream_event({"type": "updates", "data": {}}) is None
        assert WorkflowRunner._to_stream_event({"type": "updates"}) is None

    @pytest.mark.parametrize(
        "chunk",
        [
            {"type": "updates", "data": {"node": "not-a-dict"}},
            {"type": "updates", "data": {"node": {"events": []}}},
            {"type": "updates", "data": {"node": {"events": [42]}}},
        ],
    )
    def test_to_stream_event_handles_malformed_updates(
        self, chunk: dict[str, Any]
    ) -> None:
        assert WorkflowRunner._to_stream_event(chunk) is None

    def test_to_stream_event_non_dict_metrics_yields_empty_counts(self) -> None:
        chunk = {
            "type": "updates",
            "data": {
                "node_a": {
                    "events": [
                        {
                            "node": "node_a",
                            "status": "ok",
                            "message": "done",
                            "metrics": "not-a-dict",
                        }
                    ]
                }
            },
        }
        event = WorkflowRunner._to_stream_event(chunk)
        assert event is not None
        assert event.counts == {}

    def test_stream_uses_updates_v2(self, tmp_path: Any, monkeypatch: Any) -> None:
        runner = _build_runner(tmp_path)
        captured: dict[str, Any] = {}

        def _fake_stream(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            captured.update(kwargs)
            return []

        monkeypatch.setattr(runner._graph, "stream", _fake_stream)
        assert list(runner.stream(_request_input())) == []
        assert captured["stream_mode"] == "updates"
        assert captured["version"] == "v2"
        runner.close()

    def test_stream_recursion_error_yields_failed_event(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        runner = _build_runner(tmp_path)

        def _raise(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise GraphRecursionError("recursion limit reached")

        monkeypatch.setattr(runner._graph, "stream", _raise)
        events = list(runner.stream(_request_input()))
        assert events == [
            StreamEvent(
                node="runner",
                status=WorkflowStatus.FAILED.value,
                counts={},
                message="workflow recursion limit reached",
            )
        ]
        runner.close()

    def test_stream_checkpoint_error_yields_failed_event(
        self, tmp_path: Any, monkeypatch: Any
    ) -> None:
        runner = _build_runner(tmp_path)

        def _raise(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise WorkflowCheckpointError("checkpoint boom")

        monkeypatch.setattr(runner._graph, "stream", _raise)
        events = list(runner.stream(_request_input()))
        assert events == [
            StreamEvent(
                node="runner",
                status=WorkflowStatus.FAILED.value,
                counts={},
                message="checkpoint operation failed",
            )
        ]
        runner.close()

    def test_stream_sets_last_thread_and_get_output(self, tmp_path: Any) -> None:
        runner = _build_runner(tmp_path)
        events = list(runner.stream(_request_input()))
        assert events
        assert runner.last_thread_id is not None
        assert _UUID_PATTERN.fullmatch(runner.last_thread_id)
        output = runner.get_output(runner.last_thread_id)
        assert output.status is WorkflowStatus.COMPLETED
        assert output.validation_passed is True
        runner.close()

    def test_stream_with_explicit_thread_id(self, tmp_path: Any) -> None:
        runner = _build_runner(tmp_path)
        events = list(runner.stream(_request_input(), thread_id="explicit-stream-1"))
        assert events
        assert runner.last_thread_id == "explicit-stream-1"
        output = runner.get_output("explicit-stream-1")
        assert output.status is WorkflowStatus.COMPLETED
        runner.close()

    def test_stream_rejects_existing_thread(self, tmp_path: Any) -> None:
        runner = _build_runner(tmp_path)
        output = runner.run(_request_input())
        with pytest.raises(WorkflowCheckpointError, match="already exists"):
            list(runner.stream(_request_input(), thread_id=output.thread_id))
        runner.close()

    def test_stream_after_close_raises(self, tmp_path: Any) -> None:
        runner = _build_runner(tmp_path)
        runner.close()
        with pytest.raises(WorkflowCheckpointError):
            list(runner.stream(_request_input()))
