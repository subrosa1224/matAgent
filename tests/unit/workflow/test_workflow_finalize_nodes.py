"""Unit tests for S3-M4 export and finalize nodes."""

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langgraph.runtime import Runtime  # noqa: E402

from materials_screening.errors import ExportError  # noqa: E402
from materials_screening.models import (  # noqa: E402
    FilterTrace,
    FloatRange,
    MaterialRecord,
    RunMetadata,
    ScreeningRequest,
    ScreeningResult,
    ValidationReport,
)
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
    WorkflowInvariantError,
)
from materials_screening.workflow.export_adapter import (  # noqa: E402
    WorkflowExportAdapter,
)
from materials_screening.workflow.nodes import (  # noqa: E402
    export_results_node,
    finalize_failure_node,
    finalize_planner_stop_node,
    finalize_success_node,
)
from materials_screening.workflow.state import ArtifactRef, WorkflowStatus  # noqa: E402


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _SpyRepository:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, request: object) -> object:
        self.calls.append("search")
        raise AssertionError("repository must not be called")

    def healthcheck(self) -> bool:
        self.calls.append("healthcheck")
        return True


class _FakePlanner:
    def parse(self, query: str) -> object:
        raise AssertionError("planner must not be called")


class _FakeIdGenerator:
    def new_id(self) -> str:
        raise AssertionError("id generator must not be called")


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


def _build_context(
    *,
    artifact_store: FileRunArtifactStore,
    export_service: ExportService | WorkflowExportAdapter,
    repository: Any = None,
) -> WorkflowContext:
    return WorkflowContext(
        planner_service=_FakePlanner(),
        materials_repository=repository if repository is not None else _SpyRepository(),
        filter_service=FilterService(),
        ranking_service=RankingService(),
        validation_service=ValidationService(),
        export_service=export_service,
        artifact_store=artifact_store,
        clock=_fixed_clock,
        id_generator=_FakeIdGenerator(),
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


def _validated_result(run_id: str = "run-1") -> ScreeningResult:
    request = ScreeningRequest(
        band_gap_ev=FloatRange(min=1.0, max=3.0),
        limit=10,
    )
    record = MaterialRecord(
        source="mock",
        material_id="mp-1",
        formula_pretty="Fe2O3",
        elements=("Fe", "O"),
        band_gap_ev=2.0,
        is_metal=False,
    )
    ranked = RankingService().rank((record,), request)
    metadata = RunMetadata(
        run_id=run_id,
        started_at=_fixed_clock(),
        finished_at=_fixed_clock(),
        source="mock",
        database_version="fixture-v1",
        mp_api_version=None,
        pymatgen_version=None,
        application_version="test",
        query_fingerprint="abc",
    )
    return ScreeningResult(
        request=request,
        metadata=metadata,
        retrieved_count=1,
        passed_filter_count=1,
        ranked_materials=ranked,
        filter_trace=FilterTrace(steps=(), rejections=()),
        validation=ValidationReport(
            passed=True,
            checked_material_ids=("mp-1",),
        ),
    )


def _prepare_store(
    tmp_path: Any, result: ScreeningResult
) -> tuple[FileRunArtifactStore, dict[str, Any]]:
    store = FileRunArtifactStore(tmp_path / "runs")
    store.initialize_run(run_id=result.metadata.run_id, metadata={})
    ref = store.put_json(
        run_id=result.metadata.run_id,
        name="screening_result",
        value=result.model_dump(mode="json"),
        schema_name="screening_result-v1",
        item_count=len(result.ranked_materials),
    )
    return store, {"screening_result_ref": ref.model_dump(mode="json")}


def _result_state(tmp_path: Any) -> tuple[WorkflowContext, dict[str, Any]]:
    result = _validated_result()
    store, refs = _prepare_store(tmp_path, result)
    context = _build_context(
        artifact_store=store,
        export_service=WorkflowExportAdapter(tmp_path / "runs"),
    )
    state = _base_state(validation_passed=True, **refs)
    return context, state


class TestExportResultsNode:
    def test_exports_and_sets_manifest_ref(self, tmp_path: Any) -> None:
        context, state = _result_state(tmp_path)
        update = export_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.EXPORTING.value
        assert update["exports"]
        ref_dict = update["export_manifest_ref"]
        manifest = context.artifact_store.get_json(ArtifactRef.model_validate(ref_dict))
        assert manifest["run_id"] == "run-1"
        assert len(manifest["request_hash"]) == 64
        assert len(manifest["result_hash"]) == 64
        assert (tmp_path / "runs" / "run-1" / "exports").is_dir()

    def test_rejects_unvalidated_result(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(validation_passed=False)
        with pytest.raises(WorkflowInvariantError):
            export_results_node(state, _runtime(context))

    def test_requires_adapter(self, tmp_path: Any) -> None:
        result = _validated_result()
        store, refs = _prepare_store(tmp_path, result)
        context = _build_context(
            artifact_store=store,
            export_service=ExportService(),
        )
        state = _base_state(validation_passed=True, **refs)
        with pytest.raises(WorkflowInvariantError):
            export_results_node(state, _runtime(context))

    def test_export_error_maps_to_failed(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        context, state = _result_state(tmp_path)
        adapter = context.export_service

        def _boom(result: ScreeningResult) -> object:
            raise ExportError("boom")

        monkeypatch.setattr(adapter, "export", _boom)
        update = export_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.EXPORT_FAILED.value

    def test_corrupted_result_artifact_maps_to_failed(self, tmp_path: Any) -> None:
        context, state = _result_state(tmp_path)
        result_path = (
            tmp_path / "runs" / "run-1" / "artifacts" / "screening_result.json"
        )
        result_path.write_text('{"tampered": true}', encoding="utf-8")
        update = export_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_missing_result_ref_raises(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(validation_passed=True)
        with pytest.raises(WorkflowInvariantError):
            export_results_node(state, _runtime(context))

    def test_artifact_read_error_maps_to_failed(self, tmp_path: Any) -> None:
        context = _build_context(
            artifact_store=_RaisingReadStore(ArtifactStoreError("read failed")),
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(
            validation_passed=True,
            screening_result_ref={
                "name": "screening_result",
                "relative_path": "run-1/artifacts/screening_result.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
        )
        update = export_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_invalid_result_payload_maps_to_failed(self, tmp_path: Any) -> None:
        result = _validated_result()
        store, refs = _prepare_store(tmp_path, result)
        junk_ref = store.put_json(
            run_id="run-1",
            name="validation",
            value={"junk": 1},
        )
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(
            validation_passed=True,
            screening_result_ref=junk_ref.model_dump(mode="json"),
        )
        update = export_results_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value


class TestFinalizeSuccessNode:
    def _exported_state(self, tmp_path: Any) -> tuple[WorkflowContext, dict[str, Any]]:
        context, state = _result_state(tmp_path)
        update = export_results_node(state, _runtime(context))
        merged = {**state, **update}
        return context, merged

    def test_success_finalizes(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.COMPLETED.value
        assert update["finished_at"] == "2026-08-06T00:00:00+00:00"

    def test_rejects_unvalidated(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        state["validation_passed"] = False
        with pytest.raises(WorkflowInvariantError):
            finalize_success_node(state, _runtime(context))

    def test_missing_manifest_ref_raises(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        del state["export_manifest_ref"]
        with pytest.raises(WorkflowInvariantError):
            finalize_success_node(state, _runtime(context))

    def test_manifest_hash_mismatch_fails(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        manifest_path = tmp_path / "runs" / "run-1" / "exports" / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["request_hash"] = "0" * 64
        content = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        manifest_path.write_bytes(content)
        state["export_manifest_ref"]["sha256"] = hashlib.sha256(content).hexdigest()
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.EXPORT_FAILED.value
        assert "hash mismatch" in update["error"]["message"]

    def test_manifest_missing_key_file_fails(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        manifest_path = tmp_path / "runs" / "run-1" / "exports" / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        del manifest["files"]["candidates.csv"]
        content = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
        manifest_path.write_bytes(content)
        state["export_manifest_ref"]["sha256"] = hashlib.sha256(content).hexdigest()
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.EXPORT_FAILED.value
        assert "key files" in update["error"]["message"]

    def test_corrupted_manifest_fails(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        manifest_path = tmp_path / "runs" / "run-1" / "exports" / "export_manifest.json"
        manifest_path.write_text('{"tampered": true}', encoding="utf-8")
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert (
            update["error"]["code"] == WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED.value
        )

    def test_artifact_read_error_maps_to_failed(self, tmp_path: Any) -> None:
        context = _build_context(
            artifact_store=_RaisingReadStore(ArtifactStoreError("read failed")),
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(
            validation_passed=True,
            screening_result_ref={
                "name": "screening_result",
                "relative_path": "run-1/artifacts/screening_result.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
            export_manifest_ref={
                "name": "export_manifest",
                "relative_path": "run-1/exports/export_manifest.json",
                "sha256": "0" * 64,
                "media_type": "application/json",
                "size_bytes": 1,
            },
        )
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_invalid_result_payload_maps_to_failed(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        junk_ref = context.artifact_store.put_json(
            run_id="run-1",
            name="validation",
            value={"junk": 1},
        )
        state["screening_result_ref"] = junk_ref.model_dump(mode="json")
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value

    def test_non_object_manifest_maps_to_failed(self, tmp_path: Any) -> None:
        context, state = self._exported_state(tmp_path)
        list_ref = context.artifact_store.put_json(
            run_id="run-1",
            name="filter_trace",
            value=[],
        )
        state["export_manifest_ref"] = list_ref.model_dump(mode="json")
        update = finalize_success_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["error"]["code"] == WorkflowErrorCode.ARTIFACT_READ_FAILED.value


class TestFinalizePlannerStopNode:
    @pytest.mark.parametrize(
        "status",
        [
            WorkflowStatus.NEEDS_CLARIFICATION,
            WorkflowStatus.INVALID_REQUEST,
            WorkflowStatus.UNSUPPORTED_REQUEST,
        ],
    )
    def test_finalizes_planner_stop_without_mp_or_exports(
        self, tmp_path: Any, status: WorkflowStatus
    ) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        repository = _SpyRepository()
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
            repository=repository,
        )
        state = _base_state(status=status.value)
        update = finalize_planner_stop_node(state, _runtime(context))
        assert update["status"] == status.value
        assert update["finished_at"] == "2026-08-06T00:00:00+00:00"
        assert repository.calls == []
        assert not (tmp_path / "runs" / "run-1" / "exports").exists()

    def test_invalid_status_raises(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(status=WorkflowStatus.COMPLETED.value)
        with pytest.raises(WorkflowInvariantError):
            finalize_planner_stop_node(state, _runtime(context))


class TestFinalizeFailureNode:
    def test_finalizes_failed_with_safe_summary_only(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(
            status=WorkflowStatus.FAILED.value,
            error={
                "code": "RETRIEVAL_FAILED",
                "message": "safe summary",
                "retryable": False,
            },
        )
        update = finalize_failure_node(state, _runtime(context))
        assert update["status"] == WorkflowStatus.FAILED.value
        assert update["finished_at"] == "2026-08-06T00:00:00+00:00"
        assert set(update) == {
            "status",
            "current_node",
            "finished_at",
            "events",
        }
        assert "traceback" not in update
        assert "api_key" not in update

    def test_non_failed_status_raises(self, tmp_path: Any) -> None:
        store = FileRunArtifactStore(tmp_path / "runs")
        context = _build_context(
            artifact_store=store,
            export_service=WorkflowExportAdapter(tmp_path / "runs"),
        )
        state = _base_state(status=WorkflowStatus.COMPLETED.value)
        with pytest.raises(WorkflowInvariantError):
            finalize_failure_node(state, _runtime(context))
