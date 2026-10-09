"""Unit tests for WorkflowRunner (S3-M6)."""

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
from materials_screening.workflow.checkpointer import (  # noqa: E402
    CheckpointerHandle,
    create_checkpointer_handle,
)
from materials_screening.workflow.context import WorkflowContext  # noqa: E402
from materials_screening.workflow.errors import (  # noqa: E402
    WorkflowCheckpointError,
    WorkflowErrorCode,
    WorkflowInputError,
)
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.input_output import (  # noqa: E402
    WorkflowInput,
)
from materials_screening.workflow.runner import WorkflowRunner  # noqa: E402
from materials_screening.workflow.settings import WorkflowSettings  # noqa: E402
from materials_screening.workflow.state import (  # noqa: E402
    WorkflowCheckpointView,
    WorkflowStateView,
    WorkflowStatus,
)

_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _RecordsRepository:
    def __init__(
        self,
        records: tuple[MaterialRecord, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self._records = records
        self._error = error
        self.calls = 0

    def search(self, request: object) -> RetrievalResult:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=self._records,
        )

    def healthcheck(self) -> bool:
        return True


class _FakePlanner:
    def __init__(self, result: PlannerResult | None = None) -> None:
        self._result = result

    def parse(self, query: str) -> object:
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
    repository: _RecordsRepository,
    run_root: Any,
    planner: _FakePlanner | None = None,
) -> WorkflowContext:
    return WorkflowContext(
        planner_service=planner if planner is not None else _FakePlanner(),
        materials_repository=repository,
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=WorkflowExportAdapter(run_root),
        artifact_store=FileRunArtifactStore(run_root),
        clock=_fixed_clock,
        id_generator=_IdGenerator(),
    )


def _request_input() -> WorkflowInput:
    return WorkflowInput(
        request={
            "limit": 10,
            "band_gap_ev": {"min": 1.0, "max": 2.0},
        }
    )


def _build_runner(
    *,
    run_root: Any,
    checkpointer: CheckpointerHandle,
    repository: _RecordsRepository | None = None,
) -> WorkflowRunner:
    context = _build_context(
        repository=repository
        or _RecordsRepository((_record("mp-1", 1.5), _record("mp-2", 1.6))),
        run_root=run_root,
    )
    return WorkflowRunner(
        settings=WorkflowSettings(_env_file=None),
        context=context,
        checkpointer=checkpointer,
    )


class TestRunnerRun:
    def test_run_generates_id_once(self, tmp_path: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
        )
        first = runner.run(_request_input())
        second = runner.run(_request_input())
        assert _UUID_PATTERN.fullmatch(first.run_id)
        assert first.thread_id == first.run_id
        assert first.run_id != second.run_id
        assert first.status is WorkflowStatus.COMPLETED
        assert first.validation_passed is True
        assert first.exports
        runner.close()

    def test_run_rejects_existing_thread(self, tmp_path: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
        )
        first = runner.run(_request_input())
        with pytest.raises(WorkflowCheckpointError, match="already exists"):
            runner.run(_request_input(), thread_id=first.thread_id)
        runner.close()

    def test_state_dump_contains_no_secrets(self, tmp_path: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
        )
        output = runner.run(_request_input())
        snapshot = runner._graph.get_state(
            {"configurable": {"thread_id": output.thread_id}}
        )
        dumped = str(snapshot.values).lower()
        assert "api_key" not in dumped
        assert "authorization" not in dumped
        assert "bearer" not in dumped
        runner.close()

    def test_run_stream_mode(self, tmp_path: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
        )
        output = runner.run(_request_input(), stream=True)
        assert output.status is WorkflowStatus.COMPLETED
        assert output.validation_passed is True
        assert output.exports
        runner.close()

    def test_unknown_exception_propagates(self, tmp_path: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
            repository=_RecordsRepository(error=RuntimeError("boom")),
        )
        with pytest.raises(RuntimeError, match="boom"):
            runner.run(_request_input())
        runner.close()

    def test_recursion_error_mapped(self, tmp_path: Any, monkeypatch: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
        )

        def _raise_recursion(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise GraphRecursionError("recursion limit reached")

        monkeypatch.setattr(runner._graph, "invoke", _raise_recursion)
        output = runner.run(_request_input())
        assert output.status is WorkflowStatus.FAILED
        assert output.error["code"] == WorkflowErrorCode.WORKFLOW_RECURSION_LIMIT.value
        assert output.error["exception_type"] == "GraphRecursionError"
        runner.close()

    def test_checkpoint_error_mapped(self, tmp_path: Any, monkeypatch: Any) -> None:
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=create_checkpointer_handle(backend="memory"),
        )

        def _raise_checkpoint(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise WorkflowCheckpointError("checkpoint boom")

        monkeypatch.setattr(runner._graph, "invoke", _raise_checkpoint)
        output = runner.run(_request_input())
        assert output.status is WorkflowStatus.FAILED
        assert output.error["code"] == WorkflowErrorCode.CHECKPOINT_FAILED.value
        runner.close()

    def test_query_mode_planner_stop(self, tmp_path: Any) -> None:
        planner = _FakePlanner(
            PlannerResult(
                status=PlannerStatus.NEEDS_CLARIFICATION,
                query="find materials",
                clarification_question="please clarify",
            )
        )
        repository = _RecordsRepository()
        context = _build_context(
            repository=repository,
            run_root=tmp_path / "runs",
            planner=planner,
        )
        runner = WorkflowRunner(
            settings=WorkflowSettings(_env_file=None),
            context=context,
            checkpointer=create_checkpointer_handle(backend="memory"),
        )
        output = runner.run(WorkflowInput(query="find materials"))
        assert output.status is WorkflowStatus.NEEDS_CLARIFICATION
        assert output.clarification_question == "please clarify"
        assert output.retrieved_count == 0
        assert output.exports == ()
        assert repository.calls == 0
        runner.close()


class TestRunnerStateAndHistory:
    def _completed_runner(self, tmp_path: Any, backend: str) -> WorkflowRunner:
        if backend == "sqlite":
            handle = create_checkpointer_handle(
                backend="sqlite",
                db_path=tmp_path / "checkpoints.sqlite",
                allowed_root=tmp_path,
            )
        else:
            handle = create_checkpointer_handle(backend="memory")
        runner = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=handle,
        )
        return runner

    @pytest.mark.parametrize("backend", ["memory", "sqlite"])
    def test_get_state_returns_safe_view(self, tmp_path: Any, backend: str) -> None:
        runner = self._completed_runner(tmp_path, backend)
        output = runner.run(_request_input())
        view = runner.get_state(output.thread_id)
        assert isinstance(view, WorkflowStateView)
        assert view.status == WorkflowStatus.COMPLETED.value
        assert view.retrieved_count == 2
        assert view.validation_passed is True
        assert "events" not in WorkflowStateView.model_fields
        assert "retrieval_ref" not in WorkflowStateView.model_fields
        runner.close()

    @pytest.mark.parametrize("backend", ["memory", "sqlite"])
    def test_get_history_returns_checkpoint_views(
        self, tmp_path: Any, backend: str
    ) -> None:
        runner = self._completed_runner(tmp_path, backend)
        output = runner.run(_request_input())
        history = runner.get_history(output.thread_id)
        assert isinstance(history, tuple)
        assert history
        assert all(isinstance(item, WorkflowCheckpointView) for item in history)
        assert all(item.checkpoint_id for item in history)
        assert all(isinstance(item.step, int) for item in history)
        runner.close()

    def test_invalid_thread_ids_rejected(self, tmp_path: Any) -> None:
        runner = self._completed_runner(tmp_path, "memory")
        for bad in ("", "../evil", "x" * 255):
            with pytest.raises(WorkflowInputError):
                runner.get_state(bad)
        runner.close()

    def test_config_recursion_limit_is_top_level(self, tmp_path: Any) -> None:
        runner = self._completed_runner(tmp_path, "memory")
        config = runner._build_config("thread-1")
        assert config["recursion_limit"] == 32
        assert "recursion_limit" not in config["configurable"]
        assert config["configurable"]["thread_id"] == "thread-1"
        runner.close()

    def test_to_state_view_handles_non_dict_values(self) -> None:
        view = WorkflowRunner._to_state_view(None, "thread-1")
        assert isinstance(view, WorkflowStateView)
        assert view.thread_id == "thread-1"
        assert view.status == ""
        assert view.retrieved_count == 0

    def test_close_blocks_further_use(self, tmp_path: Any) -> None:
        runner = self._completed_runner(tmp_path, "memory")
        runner.close()
        with pytest.raises(WorkflowCheckpointError):
            runner.get_state("thread-1")

    def test_runner_context_manager(self, tmp_path: Any) -> None:
        with self._completed_runner(tmp_path, "memory") as runner:
            output = runner.run(_request_input())
            assert output.status is WorkflowStatus.COMPLETED
        with pytest.raises(WorkflowCheckpointError):
            runner.get_state(output.thread_id)


class TestRunnerSqlitePersistence:
    def test_sqlite_reopen_after_close(self, tmp_path: Any) -> None:
        db_path = tmp_path / "checkpoints.sqlite"
        first_handle = create_checkpointer_handle(
            backend="sqlite",
            db_path=db_path,
            allowed_root=tmp_path,
        )
        first = _build_runner(
            run_root=tmp_path / "runs",
            checkpointer=first_handle,
        )
        output = first.run(_request_input())
        first.close()

        second_handle = create_checkpointer_handle(
            backend="sqlite",
            db_path=db_path,
            allowed_root=tmp_path,
        )
        second = WorkflowRunner(
            settings=WorkflowSettings(_env_file=None),
            context=_build_context(
                repository=_RecordsRepository((_record("mp-1", 1.5),)),
                run_root=tmp_path / "runs",
            ),
            checkpointer=second_handle,
        )
        view = second.get_state(output.thread_id)
        assert view.status == WorkflowStatus.COMPLETED.value
        history = second.get_history(output.thread_id)
        assert history
        second.close()
