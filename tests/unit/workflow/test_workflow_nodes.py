"""Unit tests for S3-M4 initialize_run and resolve_request nodes."""

from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.runtime import Runtime  # noqa: E402

from materials_screening.errors import (  # noqa: E402
    InvalidRequestError,
    RepositoryRateLimitError,
)
from materials_screening.llm.errors import LLMTimeoutError  # noqa: E402
from materials_screening.models import (  # noqa: E402
    FilterStep,
    FilterTrace,
    FloatRange,
    MaterialRecord,
    PropertyProvenance,
    PropertyValueType,
    RankedMaterial,
    ScreeningRequest,
)
from materials_screening.planner.errors import PlannerQueryError  # noqa: E402
from materials_screening.planner.models import (  # noqa: E402
    PlannerResult,
    PlannerStatus,
)
from materials_screening.repositories.base import RetrievalResult  # noqa: E402
from materials_screening.services.export_service import ExportService  # noqa: E402
from materials_screening.services.filter_service import FilterService  # noqa: E402
from materials_screening.services.ranking_service import RankingService  # noqa: E402
from materials_screening.services.validation_service import (
    ValidationService,  # noqa: E402
)
from materials_screening.workflow.artifact_store import (  # noqa: E402
    FileRunArtifactStore,
)
from materials_screening.workflow.context import WorkflowContext  # noqa: E402
from materials_screening.workflow.errors import (  # noqa: E402
    ArtifactStoreError,
    WorkflowErrorCode,
    WorkflowInputError,
    WorkflowInvariantError,
)
from materials_screening.workflow.nodes import (  # noqa: E402
    _package_version,
    filter_materials_node,
    finalize_no_results_node,
    initialize_run_node,
    rank_materials_node,
    resolve_request_node,
    retrieve_materials_node,
    validate_results_node,
)
from materials_screening.workflow.state import ArtifactRef, WorkflowStatus  # noqa: E402


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _FakePlanner:
    def __init__(
        self,
        result: PlannerResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.parse_calls: list[str] = []

    def parse(self, query: str) -> PlannerResult:
        self.parse_calls.append(query)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise AssertionError("fake planner has no result configured")
        return self._result


class _SpyRepository:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, request: object) -> object:
        self.calls.append("search")
        raise AssertionError("repository.search must not run before READY")

    def healthcheck(self) -> bool:
        self.calls.append("healthcheck")
        return True


class _SpyArtifactStore:
    def __init__(self) -> None:
        self.initialized: list[tuple[str, dict[str, Any]]] = []
        self.puts: list[str] = []

    def initialize_run(self, *, run_id: str, metadata: dict[str, Any]) -> None:
        self.initialized.append((run_id, metadata))

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
        self.puts.append(name)
        return ArtifactRef(
            name=name,
            relative_path=f"run-1/artifacts/{name}.json",
            sha256="0" * 64,
            media_type="application/json",
            size_bytes=1,
        )

    def get_json(self, ref: ArtifactRef) -> object:
        raise AssertionError("get_json is not used in S3-M4 nodes")

    def exists(self, ref: ArtifactRef) -> bool:
        return False

    def verify(self, ref: ArtifactRef) -> bool:
        return False


class _SpyIdGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def new_id(self) -> str:
        self.calls += 1
        return f"generated-{self.calls}"


class _RecordingRepository:
    def __init__(
        self,
        records: tuple[MaterialRecord, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self._records = records
        self._error = error
        self.search_calls: list[ScreeningRequest] = []

    def search(self, request: ScreeningRequest) -> RetrievalResult:
        self.search_calls.append(request)
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


class _RaisingReadStore:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def initialize_run(self, *, run_id: str, metadata: dict[str, Any]) -> None:
        return None

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
        raise AssertionError("put_json is not used with this store")

    def get_json(self, ref: ArtifactRef) -> object:
        raise self._error

    def exists(self, ref: ArtifactRef) -> bool:
        return True

    def verify(self, ref: ArtifactRef) -> bool:
        return True


def _records(count: int) -> tuple[MaterialRecord, ...]:
    return tuple(
        MaterialRecord(
            source="mock",
            material_id=f"mp-{index}",
            formula_pretty=f"F{index}",
            elements=("F",),
        )
        for index in range(1, count + 1)
    )


def _build_context(
    *,
    planner: Any = None,
    repository: Any = None,
    artifact_store: Any = None,
) -> WorkflowContext:
    return WorkflowContext(
        planner_service=planner if planner is not None else _FakePlanner(),
        materials_repository=(
            repository if repository is not None else _SpyRepository()
        ),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=ExportService(),
        artifact_store=(
            artifact_store if artifact_store is not None else _SpyArtifactStore()
        ),
        clock=_fixed_clock,
        id_generator=_SpyIdGenerator(),
    )


def _base_state(**overrides: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "run_id": "run-1",
        "workflow_version": "workflow-v1",
        "input_mode": "query",
        "user_query": "find materials",
        "output_root": "data/workflow_runs",
        "export_cif": True,
    }
    state.update(overrides)
    return state


def _runtime(context: WorkflowContext) -> Runtime[WorkflowContext]:
    return Runtime(context=context)


class TestInitializeRunNode:
    def test_initializes_run_and_returns_partial_state(self) -> None:
        artifact_store = _SpyArtifactStore()
        context = _build_context(artifact_store=artifact_store)
        update = initialize_run_node(_base_state(), _runtime(context))

        assert artifact_store.initialized == [
            (
                "run-1",
                {
                    "workflow_version": "workflow-v1",
                    "input_mode": "query",
                },
            )
        ]
        assert update["status"] == WorkflowStatus.INITIALIZING.value
        assert update["current_node"] == "initialize_run"
        assert update["started_at"] == "2026-08-06T00:00:00+00:00"
        assert update["retrieved_count"] == 0
        assert update["filtered_count"] == 0
        assert update["returned_count"] == 0
        assert update["events"] == [
            {
                "event_id": "run-1:initialize_run",
                "run_id": "run-1",
                "node": "initialize_run",
                "event_type": "started",
                "created_at": "2026-08-06T00:00:00+00:00",
                "status": "initializing",
                "message": "run started",
                "metrics": {"workflow_version": "workflow-v1"},
            }
        ]

    def test_does_not_call_planner_or_repository(self) -> None:
        planner = _FakePlanner()
        repository = _SpyRepository()
        context = _build_context(planner=planner, repository=repository)
        initialize_run_node(_base_state(), _runtime(context))
        assert planner.parse_calls == []
        assert repository.calls == []

    def test_does_not_generate_run_id(self) -> None:
        artifact_store = _SpyArtifactStore()
        context = _build_context(artifact_store=artifact_store)
        initialize_run_node(
            _base_state(run_id="runner-injected-id"),
            _runtime(context),
        )
        assert artifact_store.initialized[0][0] == "runner-injected-id"
        assert context.id_generator.calls == 0

    def test_missing_run_id_raises(self) -> None:
        context = _build_context()
        state = _base_state()
        del state["run_id"]
        with pytest.raises(WorkflowInputError):
            initialize_run_node(state, _runtime(context))

    def test_reexecution_is_idempotent(self) -> None:
        artifact_store = _SpyArtifactStore()
        context = _build_context(artifact_store=artifact_store)
        state = _base_state()
        first = initialize_run_node(state, _runtime(context))
        merged = {**state, **first}
        second = initialize_run_node(merged, _runtime(context))
        assert second["events"] == []
        assert "started_at" not in second
        assert merged["started_at"] == first["started_at"]
        assert artifact_store.initialized[0] == artifact_store.initialized[1]


def _ready_planner_result() -> PlannerResult:
    return PlannerResult(
        status=PlannerStatus.READY,
        query="find materials",
        request=ScreeningRequest(limit=10),
    )


class TestResolveRequestQueryMode:
    def test_query_mode_without_user_query_raises(self) -> None:
        context = _build_context()
        state = _base_state()
        del state["user_query"]
        with pytest.raises(WorkflowInputError):
            resolve_request_node(state, _runtime(context))

    def test_ready_result_without_request_raises(self) -> None:
        invalid_ready = PlannerResult.model_construct(
            status=PlannerStatus.READY,
            query="find materials",
            request=None,
        )
        context = _build_context(planner=_FakePlanner(result=invalid_ready))
        with pytest.raises(WorkflowInvariantError):
            resolve_request_node(_base_state(), _runtime(context))

    def test_ready_maps_to_ready_for_retrieval(self) -> None:
        planner = _FakePlanner(result=_ready_planner_result())
        context = _build_context(planner=planner)
        update = resolve_request_node(_base_state(), _runtime(context))
        assert planner.parse_calls == ["find materials"]
        assert update["status"] == WorkflowStatus.READY_FOR_RETRIEVAL.value
        assert update["planner_status"] == "ready"
        assert update["screening_request"] == ScreeningRequest(limit=10).model_dump(
            mode="json"
        )
        assert update["events"][0]["event_id"] == "run-1:resolve_request"

    @pytest.mark.parametrize(
        ("planner_status", "expected_status", "extra"),
        [
            (
                PlannerStatus.NEEDS_CLARIFICATION,
                WorkflowStatus.NEEDS_CLARIFICATION,
                {"clarification_question": "please clarify"},
            ),
            (
                PlannerStatus.INVALID,
                WorkflowStatus.INVALID_REQUEST,
                {"invalid_reasons": ("bad",)},
            ),
            (
                PlannerStatus.UNSUPPORTED,
                WorkflowStatus.UNSUPPORTED_REQUEST,
                {"unsupported_requirements": ("predict",)},
            ),
        ],
    )
    def test_non_ready_statuses_map_correctly(
        self,
        planner_status: PlannerStatus,
        expected_status: WorkflowStatus,
        extra: dict[str, Any],
    ) -> None:
        planner = _FakePlanner(
            result=PlannerResult(status=planner_status, query="q", **extra)
        )
        context = _build_context(planner=planner)
        update = resolve_request_node(_base_state(), _runtime(context))
        assert update["status"] == expected_status.value
        assert update["planner_status"] == planner_status.value
        assert "screening_request" not in update
        if planner_status is PlannerStatus.NEEDS_CLARIFICATION:
            assert update["clarification_question"] == "please clarify"

    @pytest.mark.parametrize(
        "planner_status",
        [
            PlannerStatus.READY,
            PlannerStatus.NEEDS_CLARIFICATION,
            PlannerStatus.INVALID,
            PlannerStatus.UNSUPPORTED,
        ],
    )
    def test_repository_never_called(self, planner_status: PlannerStatus) -> None:
        if planner_status is PlannerStatus.READY:
            result = _ready_planner_result()
        elif planner_status is PlannerStatus.NEEDS_CLARIFICATION:
            result = PlannerResult(
                status=planner_status,
                query="q",
                clarification_question="please clarify",
            )
        elif planner_status is PlannerStatus.INVALID:
            result = PlannerResult(
                status=planner_status,
                query="q",
                invalid_reasons=("bad",),
            )
        else:
            result = PlannerResult(
                status=planner_status,
                query="q",
                unsupported_requirements=("predict",),
            )
        repository = _SpyRepository()
        context = _build_context(
            planner=_FakePlanner(result=result),
            repository=repository,
        )
        resolve_request_node(_base_state(), _runtime(context))
        assert repository.calls == []

    def test_planner_error_maps_to_failed(self) -> None:
        planner = _FakePlanner(error=PlannerQueryError("query too long"))
        context = _build_context(planner=planner)
        update = resolve_request_node(_base_state(), _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.PLANNER_FAILED.value
        assert update["error"]["retryable"] is False
        assert update["error"]["exception_type"] == "PlannerQueryError"

    def test_llm_error_maps_to_failed(self) -> None:
        planner = _FakePlanner(error=LLMTimeoutError("timeout"))
        context = _build_context(planner=planner)
        update = resolve_request_node(_base_state(), _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.PLANNER_FAILED.value

    def test_unknown_exception_propagates(self) -> None:
        planner = _FakePlanner(error=RuntimeError("boom"))
        context = _build_context(planner=planner)
        with pytest.raises(RuntimeError, match="boom"):
            resolve_request_node(_base_state(), _runtime(context))


class TestResolveRequestRequestMode:
    def test_valid_request_maps_to_ready(self) -> None:
        planner = _FakePlanner()
        context = _build_context(planner=planner)
        state = _base_state(input_mode="request", raw_request={"limit": 5})
        update = resolve_request_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.READY_FOR_RETRIEVAL.value
        assert update["screening_request"] == ScreeningRequest(limit=5).model_dump(
            mode="json"
        )
        assert planner.parse_calls == []

    def test_invalid_request_maps_to_invalid(self) -> None:
        repository = _SpyRepository()
        context = _build_context(repository=repository)
        state = _base_state(input_mode="request", raw_request={"limit": 0})
        update = resolve_request_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.INVALID_REQUEST.value
        assert update["error"]["code"] == WorkflowErrorCode.REQUEST_INVALID.value
        assert update["error"]["exception_type"] == "ValidationError"
        assert repository.calls == []

    def test_missing_raw_request_raises(self) -> None:
        context = _build_context()
        state = _base_state(input_mode="request")
        with pytest.raises(WorkflowInputError):
            resolve_request_node(state, _runtime(context))

    def test_unknown_input_mode_raises(self) -> None:
        context = _build_context()
        state = _base_state(input_mode="bogus")
        with pytest.raises(WorkflowInputError):
            resolve_request_node(state, _runtime(context))

    def test_failure_event_id_is_stable(self) -> None:
        planner = _FakePlanner(error=PlannerQueryError("boom"))
        context = _build_context(planner=planner)
        state = _base_state()
        first = resolve_request_node(state, _runtime(context))
        merged = {**state, **first}
        second = resolve_request_node(merged, _runtime(context))
        assert first["events"][0]["event_id"] == "run-1:resolve_request"
        assert second["events"] == []
        assert first["error"] == second["error"]


class TestRetrieveMaterialsNode:
    def _state(self) -> dict[str, Any]:
        return _base_state(
            screening_request=ScreeningRequest(limit=10).model_dump(mode="json")
        )

    def test_writes_all_records_without_premature_limit(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        repository = _RecordingRepository(records=_records(5))
        context = _build_context(repository=repository, artifact_store=store)
        state = _base_state(
            screening_request=ScreeningRequest(limit=2).model_dump(mode="json")
        )
        update = retrieve_materials_node(state, _runtime(context))
        assert update["retrieved_count"] == 5
        ref = ArtifactRef.model_validate(update["retrieval_ref"])
        payload = store.get_json(ref)
        assert len(payload["records"]) == 5

    def test_state_only_contains_ref_and_count(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        context = _build_context(
            repository=_RecordingRepository(records=_records(3)),
            artifact_store=store,
        )
        update = retrieve_materials_node(self._state(), _runtime(context))
        assert set(update) == {
            "status",
            "current_node",
            "retrieval_ref",
            "retrieved_count",
            "events",
        }
        assert "records" not in update
        ref_dict = update["retrieval_ref"]
        assert "records" not in ref_dict
        assert ref_dict["item_count"] == 3

    def test_calls_repository_exactly_once_per_execution(self) -> None:
        repository = _RecordingRepository(records=_records(1))
        context = _build_context(repository=repository)
        update = retrieve_materials_node(self._state(), _runtime(context))
        assert len(repository.search_calls) == 1
        assert update["retrieved_count"] == 1

    def test_repository_error_maps_to_failed(self) -> None:
        repository = _RecordingRepository(
            error=RepositoryRateLimitError("rate limited")
        )
        context = _build_context(repository=repository)
        update = retrieve_materials_node(self._state(), _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.RETRIEVAL_FAILED.value
        assert update["error"]["exception_type"] == "RepositoryRateLimitError"
        assert "retrieval_ref" not in update

    def test_invalid_request_error_maps_to_failed(self) -> None:
        repository = _RecordingRepository(error=InvalidRequestError("rejected"))
        context = _build_context(repository=repository)
        update = retrieve_materials_node(self._state(), _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.RETRIEVAL_FAILED.value

    def test_artifact_error_maps_to_failed(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        store.put_json(run_id="run-1", name="retrieval", value={"old": True})
        context = _build_context(
            repository=_RecordingRepository(records=_records(1)),
            artifact_store=store,
        )
        update = retrieve_materials_node(self._state(), _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_WRITE_FAILED.value

    def test_unknown_exception_propagates(self) -> None:
        repository = _RecordingRepository(error=RuntimeError("boom"))
        context = _build_context(repository=repository)
        with pytest.raises(RuntimeError, match="boom"):
            retrieve_materials_node(self._state(), _runtime(context))

    def test_missing_screening_request_raises(self) -> None:
        context = _build_context(repository=_RecordingRepository())
        state = _base_state()
        with pytest.raises(WorkflowInvariantError):
            retrieve_materials_node(state, _runtime(context))

    def test_invalid_screening_request_in_state_raises(self) -> None:
        context = _build_context(repository=_RecordingRepository())
        state = _base_state(screening_request={"limit": 0})
        with pytest.raises(WorkflowInvariantError):
            retrieve_materials_node(state, _runtime(context))


def _band_gap_request() -> dict[str, Any]:
    return ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=2.0),
        limit=10,
    ).model_dump(mode="json")


def _prepare_retrieval(
    tmp_path: Any, records: tuple[MaterialRecord, ...]
) -> tuple[FileRunArtifactStore, dict[str, Any]]:
    store = FileRunArtifactStore(tmp_path / "runs")
    store.initialize_run(run_id="run-1", metadata={})
    ref = store.put_json(
        run_id="run-1",
        name="retrieval",
        value=RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=records,
        ).model_dump(mode="json"),
        schema_name="retrieval-v1",
        item_count=len(records),
    )
    return store, {"retrieval_ref": ref.model_dump(mode="json")}


def _passing_record() -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id="mp-1",
        formula_pretty="Si",
        elements=("Si",),
        band_gap_ev=1.5,
        is_metal=False,
    )


def _failing_record() -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id="mp-2",
        formula_pretty="Ge",
        elements=("Ge",),
        band_gap_ev=0.5,
        is_metal=False,
    )


class TestFilterMaterialsNode:
    def test_normal_flow_writes_artifacts(self, tmp_path: Any) -> None:
        store, refs = _prepare_retrieval(
            tmp_path, (_passing_record(), _failing_record())
        )
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieved_count=2,
            **refs,
        )
        update = filter_materials_node(state, _runtime(context))

        assert update["status"] == WorkflowStatus.FILTERING.value
        assert update["filtered_count"] == 1
        filtered_payload = store.get_json(
            ArtifactRef.model_validate(update["filtered_ref"])
        )
        assert len(filtered_payload) == 1
        assert filtered_payload[0]["material_id"] == "mp-1"
        assert filtered_payload[0]["band_gap_ev"] == 1.5
        trace_payload = store.get_json(
            ArtifactRef.model_validate(update["filter_trace_ref"])
        )
        assert trace_payload["steps"]

    def test_zero_candidates_is_normal_terminal(self, tmp_path: Any) -> None:
        store, refs = _prepare_retrieval(tmp_path, (_failing_record(),))
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieved_count=1,
            **refs,
        )
        update = filter_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.NO_RESULTS.value
        assert update["filtered_count"] == 0

    def test_state_contains_only_refs_and_counts(self, tmp_path: Any) -> None:
        store, refs = _prepare_retrieval(tmp_path, (_passing_record(),))
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieved_count=1,
            **refs,
        )
        update = filter_materials_node(state, _runtime(context))
        assert set(update) == {
            "status",
            "current_node",
            "filtered_ref",
            "filter_trace_ref",
            "filtered_count",
            "events",
        }
        assert "records" not in update
        assert "records" not in update["filtered_ref"]

    def test_corrupted_retrieval_artifact_maps_to_failed(self, tmp_path: Any) -> None:
        store, refs = _prepare_retrieval(tmp_path, (_passing_record(),))
        artifact_path = tmp_path / "runs" / "run-1" / "artifacts" / "retrieval.json"
        artifact_path.write_text('{"corrupted": true}', encoding="utf-8")
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieved_count=1,
            **refs,
        )
        update = filter_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_invalid_retrieval_payload_maps_to_failed(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="retrieval", value={"junk": 1})
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieved_count=1,
            retrieval_ref=ref.model_dump(mode="json"),
        )
        update = filter_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_artifact_write_failure_maps_to_failed(self, tmp_path: Any) -> None:
        store, refs = _prepare_retrieval(tmp_path, (_passing_record(),))
        store.put_json(run_id="run-1", name="filtered", value=[{"old": True}])
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieved_count=1,
            **refs,
        )
        update = filter_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_WRITE_FAILED.value

    def test_missing_retrieval_ref_raises(self) -> None:
        context = _build_context()
        state = _base_state(screening_request=_band_gap_request())
        with pytest.raises(WorkflowInvariantError):
            filter_materials_node(state, _runtime(context))

    def test_invalid_ref_in_state_raises(self) -> None:
        context = _build_context()
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieval_ref={"bad": 1},
        )
        with pytest.raises(WorkflowInvariantError):
            filter_materials_node(state, _runtime(context))

    def test_artifact_read_error_maps_to_failed(self) -> None:
        context = _build_context(
            artifact_store=_RaisingReadStore(ArtifactStoreError("read failed"))
        )
        state = _base_state(
            screening_request=_band_gap_request(),
            retrieval_ref={
                "name": "retrieval",
                "relative_path": "run-1/artifacts/retrieval.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
        )
        update = filter_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value


def _trace_ref(store: FileRunArtifactStore) -> dict[str, Any]:
    trace = FilterTrace(
        steps=(
            FilterStep(
                name="band_gap",
                before_count=5,
                after_count=0,
                rejection_count=5,
                reason_counts={"band_gap_below_min": 5},
            ),
        ),
        rejections=(),
    )
    ref = store.put_json(
        run_id="run-1",
        name="filter_trace",
        value=trace.model_dump(mode="json"),
        schema_name="filter_trace-v1",
    )
    return {"filter_trace_ref": ref.model_dump(mode="json")}


class TestFinalizeNoResultsNode:
    def test_reports_zero_step(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        context = _build_context(artifact_store=store)
        state = _base_state(retrieved_count=5, **_trace_ref(store))
        update = finalize_no_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.NO_RESULTS.value
        assert update["finished_at"] == "2026-08-06T00:00:00+00:00"
        assert "band_gap" in update["events"][0]["message"]
        assert "band_gap" in update["warnings"][0]
        assert "not applied automatically" in update["warnings"][0]

    def test_zero_retrieved_reports_retrieval_as_zero_step(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        context = _build_context(artifact_store=store)
        state = _base_state(retrieved_count=0, **_trace_ref(store))
        update = finalize_no_results_node(state, _runtime(context))
        assert "retrieval" in update["events"][0]["message"]

    def test_does_not_modify_screening_request(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        context = _build_context(artifact_store=store)
        state = _base_state(
            retrieved_count=5,
            screening_request=_band_gap_request(),
            **_trace_ref(store),
        )
        update = finalize_no_results_node(state, _runtime(context))
        assert "screening_request" not in update

    def test_corrupted_trace_maps_to_failed(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _trace_ref(store)
        trace_path = tmp_path / "runs" / "run-1" / "artifacts" / "filter_trace.json"
        trace_path.write_text('{"broken": true}', encoding="utf-8")
        context = _build_context(artifact_store=store)
        state = _base_state(retrieved_count=5, **refs)
        update = finalize_no_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_missing_trace_ref_raises(self) -> None:
        context = _build_context()
        state = _base_state(retrieved_count=5)
        with pytest.raises(WorkflowInvariantError):
            finalize_no_results_node(state, _runtime(context))

    def test_invalid_trace_payload_maps_to_failed(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="filter_trace", value={"junk": 1})
        context = _build_context(artifact_store=store)
        state = _base_state(
            retrieved_count=5,
            filter_trace_ref=ref.model_dump(mode="json"),
        )
        update = finalize_no_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_artifact_read_error_maps_to_failed(self) -> None:
        context = _build_context(
            artifact_store=_RaisingReadStore(ArtifactStoreError("read failed"))
        )
        state = _base_state(
            retrieved_count=5,
            filter_trace_ref={
                "name": "filter_trace",
                "relative_path": "run-1/artifacts/filter_trace.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
        )
        update = finalize_no_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_fallback_zero_step_when_no_step_reaches_zero(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        trace = FilterTrace(
            steps=(
                FilterStep(
                    name="band_gap",
                    before_count=5,
                    after_count=3,
                    rejection_count=2,
                    reason_counts={},
                ),
            ),
            rejections=(),
        )
        ref = store.put_json(
            run_id="run-1",
            name="filter_trace",
            value=trace.model_dump(mode="json"),
        )
        context = _build_context(artifact_store=store)
        state = _base_state(
            retrieved_count=5,
            filter_trace_ref=ref.model_dump(mode="json"),
        )
        update = finalize_no_results_node(state, _runtime(context))
        assert "'filter'" in update["events"][0]["message"]


def _rankable_record(material_id: str, band_gap: float, hull: float) -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=material_id,
        elements=("Si",),
        band_gap_ev=band_gap,
        energy_above_hull_ev_atom=hull,
        is_metal=False,
    )


def _prepare_filtered(
    tmp_path: Any, records: tuple[MaterialRecord, ...]
) -> tuple[FileRunArtifactStore, dict[str, Any]]:
    store = FileRunArtifactStore(tmp_path / "runs")
    store.initialize_run(run_id="run-1", metadata={})
    ref = store.put_json(
        run_id="run-1",
        name="filtered",
        value=[record.model_dump(mode="json") for record in records],
        schema_name="filtered-v1",
        item_count=len(records),
    )
    return store, {"filtered_ref": ref.model_dump(mode="json")}


def _ranking_request(limit: int = 10) -> dict[str, Any]:
    return ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=2.0),
        limit=limit,
    ).model_dump(mode="json")


def _five_rankable_records() -> tuple[MaterialRecord, ...]:
    return (
        _rankable_record("mp-1", 1.5, 0.0),
        _rankable_record("mp-2", 1.4, 0.1),
        _rankable_record("mp-3", 1.6, 0.05),
        _rankable_record("mp-4", 1.7, 0.0),
        _rankable_record("mp-5", 1.3, 0.2),
    )


class TestRankMaterialsNode:
    def test_ranks_all_then_truncates_by_limit(self, tmp_path: Any) -> None:
        store, refs = _prepare_filtered(tmp_path, _five_rankable_records())
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_ranking_request(limit=2),
            filtered_count=5,
            **refs,
        )
        update = rank_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.RANKING.value
        assert update["returned_count"] == 2
        ranked = store.get_json(ArtifactRef.model_validate(update["ranked_ref"]))
        assert len(ranked) == 2
        assert "records" not in update

    def test_order_matches_direct_ranking_service(self, tmp_path: Any) -> None:
        records = _five_rankable_records()
        store, refs = _prepare_filtered(tmp_path, records)
        context = _build_context(artifact_store=store)
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.0, max=2.0),
            limit=10,
        )
        state = _base_state(
            screening_request=request.model_dump(mode="json"),
            filtered_count=5,
            **refs,
        )
        update = rank_materials_node(state, _runtime(context))
        ranked = store.get_json(ArtifactRef.model_validate(update["ranked_ref"]))
        expected = RankingService().rank(records, request)
        assert [item["record"]["material_id"] for item in ranked] == [
            item.record.material_id for item in expected
        ]
        assert [item["rank"] for item in ranked] == [item.rank for item in expected]
        assert [item["total_score"] for item in ranked] == [
            item.total_score for item in expected
        ]

    def test_tie_break_is_deterministic(self, tmp_path: Any) -> None:
        records = (
            _rankable_record("mp-b", 1.5, 0.0),
            _rankable_record("mp-a", 1.5, 0.0),
        )
        store, refs = _prepare_filtered(tmp_path, records)
        context = _build_context(artifact_store=store)
        request = ScreeningRequest(
            band_gap_ev=FloatRange(min=1.0, max=2.0),
            limit=10,
        )
        state = _base_state(
            screening_request=request.model_dump(mode="json"),
            filtered_count=2,
            **refs,
        )
        update = rank_materials_node(state, _runtime(context))
        ranked = store.get_json(ArtifactRef.model_validate(update["ranked_ref"]))
        material_ids = [item["record"]["material_id"] for item in ranked]
        assert material_ids == ["mp-a", "mp-b"]
        expected = RankingService().rank(records, request)
        assert material_ids == [item.record.material_id for item in expected]

    def test_state_contains_only_ref_and_count(self, tmp_path: Any) -> None:
        store, refs = _prepare_filtered(tmp_path, _five_rankable_records())
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_ranking_request(),
            filtered_count=5,
            **refs,
        )
        update = rank_materials_node(state, _runtime(context))
        assert set(update) == {
            "status",
            "current_node",
            "ranked_ref",
            "returned_count",
            "events",
        }
        assert "records" not in update
        assert "records" not in update["ranked_ref"]

    def test_corrupted_filtered_artifact_maps_to_failed(self, tmp_path: Any) -> None:
        store, refs = _prepare_filtered(tmp_path, _five_rankable_records())
        filtered_path = tmp_path / "runs" / "run-1" / "artifacts" / "filtered.json"
        filtered_path.write_text('[{"corrupted": true}]', encoding="utf-8")
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_ranking_request(),
            filtered_count=5,
            **refs,
        )
        update = rank_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_invalid_filtered_payload_maps_to_failed(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        ref = store.put_json(run_id="run-1", name="filtered", value={"junk": 1})
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_ranking_request(),
            filtered_count=1,
            filtered_ref=ref.model_dump(mode="json"),
        )
        update = rank_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_artifact_write_failure_maps_to_failed(self, tmp_path: Any) -> None:
        store, refs = _prepare_filtered(tmp_path, _five_rankable_records())
        store.put_json(run_id="run-1", name="ranked", value=[{"old": True}])
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=_ranking_request(),
            filtered_count=5,
            **refs,
        )
        update = rank_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_WRITE_FAILED.value

    def test_missing_filtered_ref_raises(self) -> None:
        context = _build_context()
        state = _base_state(screening_request=_ranking_request())
        with pytest.raises(WorkflowInvariantError):
            rank_materials_node(state, _runtime(context))

    def test_artifact_read_error_maps_to_failed(self) -> None:
        context = _build_context(
            artifact_store=_RaisingReadStore(ArtifactStoreError("read failed"))
        )
        state = _base_state(
            screening_request=_ranking_request(),
            filtered_ref={
                "name": "filtered",
                "relative_path": "run-1/artifacts/filtered.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
        )
        update = rank_materials_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value


def _provenanced_record(
    material_id: str, band_gap: float, hull: float
) -> MaterialRecord:
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
        energy_above_hull_ev_atom=hull,
        is_metal=False,
        provenance=provenance,
    )


def _validate_request() -> ScreeningRequest:
    return ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=2.0),
        limit=10,
    )


def _put_validate_artifacts(
    store: FileRunArtifactStore,
    *,
    records: tuple[MaterialRecord, ...],
    trace: FilterTrace,
    ranked_items: tuple[RankedMaterial, ...],
) -> dict[str, Any]:
    retrieval_ref = store.put_json(
        run_id="run-1",
        name="retrieval",
        value=RetrievalResult(
            source="mock",
            database_version="fixture-v1",
            retrieved_at=_fixed_clock(),
            records=records,
        ).model_dump(mode="json"),
        schema_name="retrieval-v1",
        item_count=len(records),
    )
    trace_ref = store.put_json(
        run_id="run-1",
        name="filter_trace",
        value=trace.model_dump(mode="json"),
        schema_name="filter_trace-v1",
    )
    ranked_ref = store.put_json(
        run_id="run-1",
        name="ranked",
        value=[item.model_dump(mode="json") for item in ranked_items],
        schema_name="ranked-v1",
        item_count=len(ranked_items),
    )
    return {
        "retrieval_ref": retrieval_ref.model_dump(mode="json"),
        "filter_trace_ref": trace_ref.model_dump(mode="json"),
        "ranked_ref": ranked_ref.model_dump(mode="json"),
        "retrieved_count": len(records),
        "filtered_count": len(ranked_items),
    }


def _validate_state(request: ScreeningRequest, refs: dict[str, Any]) -> dict[str, Any]:
    return _base_state(
        screening_request=request.model_dump(mode="json"),
        started_at="2026-08-06T00:00:00+00:00",
        **refs,
    )


class TestValidateResultsNode:
    def test_passes_and_writes_artifacts(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (
            _provenanced_record("mp-1", 1.5, 0.0),
            _provenanced_record("mp-2", 1.6, 0.05),
        )
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.VALIDATING.value
        assert update["validation_passed"] is True
        validation_payload = store.get_json(
            ArtifactRef.model_validate(update["validation_ref"])
        )
        assert validation_payload["passed"] is True
        result_payload = store.get_json(
            ArtifactRef.model_validate(update["screening_result_ref"])
        )
        assert result_payload["validation"]["passed"] is True
        assert len(result_payload["ranked_materials"]) == 2

    def test_tampered_score_fails_without_deleting_candidates(
        self, tmp_path: Any
    ) -> None:
        request = _validate_request()
        records = (
            _provenanced_record("mp-1", 1.5, 0.0),
            _provenanced_record("mp-2", 1.6, 0.05),
        )
        filtered, trace = FilterService().apply(records, request)
        ranked_items = RankingService().rank(filtered, request)
        tampered_payload = [item.model_dump(mode="json") for item in ranked_items]
        tampered_payload[0]["total_score"] = round(
            tampered_payload[0]["total_score"] + 0.1, 8
        )
        tampered = tuple(
            RankedMaterial.model_validate(item) for item in tampered_payload
        )
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=tampered
        )
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["validation_passed"] is False
        assert update["error"]["code"] == WorkflowErrorCode.VALIDATION_FAILED.value
        result_payload = store.get_json(
            ArtifactRef.model_validate(update["screening_result_ref"])
        )
        assert len(result_payload["ranked_materials"]) == 2

    def test_missing_provenance_fails(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_rankable_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        validation_payload = store.get_json(
            ArtifactRef.model_validate(update["validation_ref"])
        )
        assert any("provenance" in error for error in validation_payload["errors"])

    def test_hard_constraint_violation_fails(self, tmp_path: Any) -> None:
        request = _validate_request()
        violating = _provenanced_record("mp-bad", 0.5, 0.0)
        ranked = RankingService().rank((violating,), request)
        trace = FilterTrace(steps=(), rejections=())
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store,
            records=(violating,),
            trace=trace,
            ranked_items=ranked,
        )
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["validation_passed"] is False
        validation_payload = store.get_json(
            ArtifactRef.model_validate(update["validation_ref"])
        )
        assert any(
            "violates hard constraints" in error
            for error in validation_payload["errors"]
        )
        result_payload = store.get_json(
            ArtifactRef.model_validate(update["screening_result_ref"])
        )
        assert len(result_payload["ranked_materials"]) == 1

    def test_corrupted_ranked_artifact_maps_to_failed(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_provenanced_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        ranked_path = tmp_path / "runs" / "run-1" / "artifacts" / "ranked.json"
        ranked_path.write_text('[{"tampered": true}]', encoding="utf-8")
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_invalid_ranked_payload_maps_to_failed(self, tmp_path: Any) -> None:
        request = _validate_request()
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        retrieval_ref = store.put_json(
            run_id="run-1",
            name="retrieval",
            value=RetrievalResult(
                source="mock",
                database_version="fixture-v1",
                retrieved_at=_fixed_clock(),
                records=(),
            ).model_dump(mode="json"),
        )
        trace_ref = store.put_json(
            run_id="run-1",
            name="filter_trace",
            value=FilterTrace(steps=(), rejections=()).model_dump(mode="json"),
        )
        ranked_ref = store.put_json(run_id="run-1", name="ranked", value={"junk": 1})
        refs = {
            "retrieval_ref": retrieval_ref.model_dump(mode="json"),
            "filter_trace_ref": trace_ref.model_dump(mode="json"),
            "ranked_ref": ranked_ref.model_dump(mode="json"),
        }
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_artifact_write_failure_maps_to_failed(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_provenanced_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        store.put_json(run_id="run-1", name="validation", value={"old": True})
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_WRITE_FAILED.value

    def test_missing_ranked_ref_raises(self) -> None:
        context = _build_context()
        state = _base_state(
            screening_request=_validate_request().model_dump(mode="json"),
            started_at="2026-08-06T00:00:00+00:00",
        )
        with pytest.raises(WorkflowInvariantError):
            validate_results_node(state, _runtime(context))

    def test_missing_started_at_raises(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_provenanced_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=request.model_dump(mode="json"),
            **refs,
        )
        with pytest.raises(WorkflowInvariantError):
            validate_results_node(state, _runtime(context))

    def test_corrupted_retrieval_artifact_maps_to_failed(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_provenanced_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        retrieval_path = tmp_path / "runs" / "run-1" / "artifacts" / "retrieval.json"
        retrieval_path.write_text('{"tampered": true}', encoding="utf-8")
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_corrupted_trace_artifact_maps_to_failed(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_provenanced_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        trace_path = tmp_path / "runs" / "run-1" / "artifacts" / "filter_trace.json"
        trace_path.write_text('{"tampered": true}', encoding="utf-8")
        context = _build_context(artifact_store=store)
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_artifact_read_error_maps_to_failed(self) -> None:
        context = _build_context(
            artifact_store=_RaisingReadStore(ArtifactStoreError("read failed"))
        )
        request = _validate_request()
        refs = {
            "retrieval_ref": {
                "name": "retrieval",
                "relative_path": "run-1/artifacts/retrieval.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
            "filter_trace_ref": {
                "name": "filter_trace",
                "relative_path": "run-1/artifacts/filter_trace.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
            "ranked_ref": {
                "name": "ranked",
                "relative_path": "run-1/artifacts/ranked.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
        }
        update = validate_results_node(
            _validate_state(request, refs), _runtime(context)
        )
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_invalid_started_at_raises(self, tmp_path: Any) -> None:
        request = _validate_request()
        records = (_provenanced_record("mp-1", 1.5, 0.0),)
        filtered, trace = FilterService().apply(records, request)
        ranked = RankingService().rank(filtered, request)
        store = FileRunArtifactStore(tmp_path / "runs")
        store.initialize_run(run_id="run-1", metadata={})
        refs = _put_validate_artifacts(
            store, records=records, trace=trace, ranked_items=ranked
        )
        context = _build_context(artifact_store=store)
        state = _base_state(
            screening_request=request.model_dump(mode="json"),
            started_at="not-a-datetime",
            **refs,
        )
        with pytest.raises(WorkflowInvariantError):
            validate_results_node(state, _runtime(context))


def test_package_version_missing_returns_none() -> None:
    assert _package_version("materials-screening-nonexistent-package") is None
