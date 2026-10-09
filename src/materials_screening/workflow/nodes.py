"""Stage 3 workflow nodes (S3-M4: initialize_run and resolve_request)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from materials_screening import __version__
from materials_screening.errors import (
    ExportError,
    InvalidRequestError,
    RepositoryError,
)
from materials_screening.fingerprints import request_fingerprint
from materials_screening.formatting.export_files import json_text
from materials_screening.llm.errors import LLMError
from materials_screening.models import (
    FilterTrace,
    MaterialRecord,
    RankedMaterial,
    RunMetadata,
    ScreeningRequest,
    ScreeningResult,
    ValidationReport,
)
from materials_screening.planner.errors import PlannerError
from materials_screening.planner.models import PlannerStatus
from materials_screening.repositories.base import RetrievalResult
from materials_screening.workflow.artifact_store import ArtifactName
from materials_screening.workflow.context import WorkflowContext
from materials_screening.workflow.errors import (
    ArtifactIntegrityError,
    ArtifactStoreError,
    WorkflowErrorCode,
    WorkflowErrorData,
    WorkflowInputError,
    WorkflowInvariantError,
)
from materials_screening.workflow.events import emit_event
from materials_screening.workflow.export_adapter import WorkflowExportAdapter
from materials_screening.workflow.state import (
    ArtifactRef,
    WorkflowState,
    WorkflowStatus,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

NODE_INITIALIZE_RUN = "initialize_run"
NODE_RESOLVE_REQUEST = "resolve_request"
NODE_RETRIEVE_MATERIALS = "retrieve_materials"
NODE_FILTER_MATERIALS = "filter_materials"
NODE_FINALIZE_NO_RESULTS = "finalize_no_results"
NODE_RANK_MATERIALS = "rank_materials"
NODE_VALIDATE_RESULTS = "validate_results"
NODE_EXPORT_RESULTS = "export_results"
NODE_FINALIZE_SUCCESS = "finalize_success"
NODE_FINALIZE_PLANNER_STOP = "finalize_planner_stop"
NODE_FINALIZE_FAILURE = "finalize_failure"

_PLANNER_STATUS_TO_WORKFLOW: dict[PlannerStatus, WorkflowStatus] = {
    PlannerStatus.READY: WorkflowStatus.READY_FOR_RETRIEVAL,
    PlannerStatus.NEEDS_CLARIFICATION: WorkflowStatus.NEEDS_CLARIFICATION,
    PlannerStatus.INVALID: WorkflowStatus.INVALID_REQUEST,
    PlannerStatus.UNSUPPORTED: WorkflowStatus.UNSUPPORTED_REQUEST,
}


def initialize_run_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Create the run directory/manifest and initialize counters.

    The run_id is injected by the runner through the graph input; this node
    never generates identifiers and never calls Intern or Materials Project.
    """
    run_id = state.get("run_id")
    if not run_id:
        raise WorkflowInputError("run_id must be injected by the runner before invoke")
    context = runtime.context
    created_at = context.clock().isoformat()
    workflow_version = state.get("workflow_version", "")
    input_mode = state.get("input_mode", "")
    context.artifact_store.initialize_run(
        run_id=run_id,
        metadata={
            "workflow_version": workflow_version,
            "input_mode": input_mode,
        },
    )
    update: dict[str, Any] = {
        "status": WorkflowStatus.INITIALIZING.value,
        "current_node": NODE_INITIALIZE_RUN,
        "retrieved_count": 0,
        "filtered_count": 0,
        "returned_count": 0,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_INITIALIZE_RUN,
            event_type="started",
            status=WorkflowStatus.INITIALIZING.value,
            message="run started",
            metrics={"workflow_version": workflow_version},
            created_at=created_at,
        ),
    }
    if state.get("started_at") is None:
        update["started_at"] = created_at
    return update


def resolve_request_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Resolve the query via PlannerService or validate the raw request."""
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    input_mode = state.get("input_mode")
    if input_mode == "query":
        return _resolve_query(state, context, run_id, created_at)
    if input_mode == "request":
        return _resolve_request(state, context, run_id, created_at)
    raise WorkflowInputError(f"unknown input_mode: {input_mode!r}")


def retrieve_materials_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Rebuild the request, call the repository and persist all records.

    The repository already performs bounded remote retries; this node must
    not add LangGraph RetryPolicy. Records are only written to the retrieval
    artifact; the state keeps the ArtifactRef dict and the count.
    """
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    request = _rebuild_request(state)
    try:
        retrieval = context.materials_repository.search(request)
    except InvalidRequestError as exc:
        # InvalidRequestError carries a user-actionable message (e.g. a too
        # broad live query); surface it so the agent can tell the user what
        # to change instead of a generic "retry later".
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RETRIEVE_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.RETRIEVAL_FAILED,
            message=str(exc),
            exception_type=type(exc).__name__,
        )
    except RepositoryError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RETRIEVE_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.RETRIEVAL_FAILED,
            message="materials retrieval failed",
            exception_type=type(exc).__name__,
        )

    count = len(retrieval.records)
    try:
        ref = context.artifact_store.put_json(
            run_id=run_id,
            name=ArtifactName.RETRIEVAL.value,
            value=retrieval.model_dump(mode="json"),
            schema_name="retrieval-v1",
            item_count=count,
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RETRIEVE_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_WRITE_FAILED,
            message="artifact write failed",
            exception_type=type(exc).__name__,
        )

    return {
        "status": WorkflowStatus.RETRIEVING.value,
        "current_node": NODE_RETRIEVE_MATERIALS,
        "retrieval_ref": ref.model_dump(mode="json"),
        "retrieved_count": count,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_RETRIEVE_MATERIALS,
            event_type="retrieved",
            status=WorkflowStatus.RETRIEVING.value,
            message=f"retrieved={count}",
            metrics={"retrieved_count": count},
            created_at=created_at,
        ),
    }


def filter_materials_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Verify and read the retrieval artifact, then apply the hard filters.

    Zero candidates are a normal terminal signal (NO_RESULTS), not an error.
    Filtered records and the trace go to artifacts; state only keeps refs.
    """
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    try:
        payload = _read_artifact(state, context, "retrieval_ref")
    except ArtifactIntegrityError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FILTER_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED,
            message="artifact integrity check failed",
            exception_type=type(exc).__name__,
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FILTER_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact read failed",
            exception_type=type(exc).__name__,
        )
    try:
        retrieval = RetrievalResult.model_validate(payload)
    except ValidationError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FILTER_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="retrieval artifact is invalid",
            exception_type=type(exc).__name__,
        )

    request = _rebuild_request(state)
    filtered, filter_trace = context.filter_service.apply(retrieval.records, request)
    try:
        filtered_ref = context.artifact_store.put_json(
            run_id=run_id,
            name=ArtifactName.FILTERED.value,
            value=[record.model_dump(mode="json") for record in filtered],
            schema_name="filtered-v1",
            item_count=len(filtered),
        )
        filter_trace_ref = context.artifact_store.put_json(
            run_id=run_id,
            name=ArtifactName.FILTER_TRACE.value,
            value=filter_trace.model_dump(mode="json"),
            schema_name="filter_trace-v1",
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FILTER_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_WRITE_FAILED,
            message="artifact write failed",
            exception_type=type(exc).__name__,
        )

    retrieved_count = state.get("retrieved_count", len(retrieval.records))
    status = (
        WorkflowStatus.NO_RESULTS if len(filtered) == 0 else WorkflowStatus.FILTERING
    )
    return {
        "status": status.value,
        "current_node": NODE_FILTER_MATERIALS,
        "filtered_ref": filtered_ref.model_dump(mode="json"),
        "filter_trace_ref": filter_trace_ref.model_dump(mode="json"),
        "filtered_count": len(filtered),
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_FILTER_MATERIALS,
            event_type="filtered",
            status=status.value,
            message=f"filtered={len(filtered)}",
            metrics={
                "retrieved_count": retrieved_count,
                "filtered_count": len(filtered),
            },
            created_at=created_at,
        ),
    }


def finalize_no_results_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Build the zero-result summary; never relax conditions automatically."""
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    try:
        payload = _read_artifact(state, context, "filter_trace_ref")
    except ArtifactIntegrityError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_NO_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED,
            message="artifact integrity check failed",
            exception_type=type(exc).__name__,
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_NO_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact read failed",
            exception_type=type(exc).__name__,
        )
    try:
        filter_trace = FilterTrace.model_validate(payload)
    except ValidationError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_NO_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="filter_trace artifact is invalid",
            exception_type=type(exc).__name__,
        )

    retrieved_count = state.get("retrieved_count", 0)
    zero_step = _zero_step(retrieved_count, filter_trace)
    summary = (
        f"no_results: zero at '{zero_step}' (retrieved={retrieved_count}, filtered=0)"
    )
    suggestion = (
        f"no materials passed the '{zero_step}' filter step; "
        "consider relaxing conditions (not applied automatically)"
    )
    return {
        "status": WorkflowStatus.NO_RESULTS.value,
        "current_node": NODE_FINALIZE_NO_RESULTS,
        "finished_at": created_at,
        "warnings": [suggestion],
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_FINALIZE_NO_RESULTS,
            event_type="no_results",
            status=WorkflowStatus.NO_RESULTS.value,
            message=summary,
            metrics={
                "retrieved_count": retrieved_count,
                "filtered_count": 0,
            },
            created_at=created_at,
        ),
    }


def rank_materials_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Rank every filtered candidate, then truncate to request.limit.

    Ranking uses the existing RankingService (never re-implemented here), so
    the order stays identical to direct mode. State keeps only the ranked
    artifact ref and returned_count.
    """
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    try:
        payload = _read_artifact(state, context, "filtered_ref")
    except ArtifactIntegrityError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RANK_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED,
            message="artifact integrity check failed",
            exception_type=type(exc).__name__,
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RANK_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact read failed",
            exception_type=type(exc).__name__,
        )
    try:
        payload_items = cast(Sequence[Any], payload)
        records = tuple(MaterialRecord.model_validate(item) for item in payload_items)
    except (TypeError, ValidationError) as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RANK_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="filtered artifact is invalid",
            exception_type=type(exc).__name__,
        )

    request = _rebuild_request(state)
    ranked_all = context.ranking_service.rank(records, request)
    ranked_limited = ranked_all[: request.limit]
    try:
        ref = context.artifact_store.put_json(
            run_id=run_id,
            name=ArtifactName.RANKED.value,
            value=[item.model_dump(mode="json") for item in ranked_limited],
            schema_name="ranked-v1",
            item_count=len(ranked_limited),
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RANK_MATERIALS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_WRITE_FAILED,
            message="artifact write failed",
            exception_type=type(exc).__name__,
        )

    return {
        "status": WorkflowStatus.RANKING.value,
        "current_node": NODE_RANK_MATERIALS,
        "ranked_ref": ref.model_dump(mode="json"),
        "returned_count": len(ranked_limited),
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_RANK_MATERIALS,
            event_type="ranked",
            status=WorkflowStatus.RANKING.value,
            message=f"returned={len(ranked_limited)}",
            metrics={"returned_count": len(ranked_limited)},
            created_at=created_at,
        ),
    }


def validate_results_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Re-validate the ranked result with the existing ValidationService.

    Failed validation must route to finalize_failure; candidates are never
    silently removed to fake success. Provenance rules are the stage-1 rules
    because the same ValidationService is used.
    """
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()

    retrieval_payload, failure = _read_artifact_or_failure(
        state, context, "retrieval_ref", NODE_VALIDATE_RESULTS, created_at
    )
    if failure is not None:
        return failure
    trace_payload, failure = _read_artifact_or_failure(
        state, context, "filter_trace_ref", NODE_VALIDATE_RESULTS, created_at
    )
    if failure is not None:
        return failure
    ranked_payload, failure = _read_artifact_or_failure(
        state, context, "ranked_ref", NODE_VALIDATE_RESULTS, created_at
    )
    if failure is not None:
        return failure

    try:
        retrieval = RetrievalResult.model_validate(retrieval_payload)
        filter_trace = FilterTrace.model_validate(trace_payload)
        ranked_items = tuple(
            RankedMaterial.model_validate(item)
            for item in cast(Sequence[Any], ranked_payload)
        )
    except (TypeError, ValidationError) as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_VALIDATE_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact content is invalid",
            exception_type=type(exc).__name__,
        )

    request = _rebuild_request(state)
    metadata = _build_run_metadata(state, retrieval, request)
    result = ScreeningResult(
        request=request,
        metadata=metadata,
        retrieved_count=state.get("retrieved_count", len(retrieval.records)),
        passed_filter_count=state.get("filtered_count", 0),
        ranked_materials=ranked_items,
        filter_trace=filter_trace,
        validation=ValidationReport(passed=False),
    )
    validation_report = context.validation_service.validate(result)
    validated_result = result.model_copy(update={"validation": validation_report})

    try:
        validation_ref = context.artifact_store.put_json(
            run_id=run_id,
            name=ArtifactName.VALIDATION.value,
            value=validation_report.model_dump(mode="json"),
            schema_name="validation-v1",
            item_count=len(validation_report.checked_material_ids),
        )
        result_ref = context.artifact_store.put_json(
            run_id=run_id,
            name=ArtifactName.SCREENING_RESULT.value,
            value=validated_result.model_dump(mode="json"),
            schema_name="screening_result-v1",
            item_count=len(validated_result.ranked_materials),
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_VALIDATE_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_WRITE_FAILED,
            message="artifact write failed",
            exception_type=type(exc).__name__,
        )

    refs = {
        "validation_ref": validation_ref.model_dump(mode="json"),
        "screening_result_ref": result_ref.model_dump(mode="json"),
        "validation_passed": validation_report.passed,
    }
    if not validation_report.passed:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_VALIDATE_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.VALIDATION_FAILED,
            message=(
                f"validation failed with {len(validation_report.errors)} error(s)"
            ),
            exception_type="ValidationFailedError",
            extra=refs,
        )
    return {
        "status": WorkflowStatus.VALIDATING.value,
        "current_node": NODE_VALIDATE_RESULTS,
        **refs,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_VALIDATE_RESULTS,
            event_type="validated",
            status=WorkflowStatus.VALIDATING.value,
            message="validation passed",
            metrics={
                "checked": len(validation_report.checked_material_ids),
            },
            created_at=created_at,
        ),
    }


def export_results_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Export a validated result via WorkflowExportAdapter (idempotent)."""
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    if state.get("validation_passed") is not True:
        raise WorkflowInvariantError("export_results requires validation_passed=True")
    export_service = context.export_service
    if not isinstance(export_service, WorkflowExportAdapter):
        raise WorkflowInvariantError("workflow export requires WorkflowExportAdapter")
    try:
        payload = _read_artifact(state, context, "screening_result_ref")
    except ArtifactIntegrityError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_EXPORT_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED,
            message="artifact integrity check failed",
            exception_type=type(exc).__name__,
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_EXPORT_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact read failed",
            exception_type=type(exc).__name__,
        )
    try:
        result = ScreeningResult.model_validate(payload)
    except (TypeError, ValidationError) as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_EXPORT_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="screening_result artifact is invalid",
            exception_type=type(exc).__name__,
        )
    try:
        export_output = export_service.export(result)
    except ExportError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_EXPORT_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.EXPORT_FAILED,
            message="export failed",
            exception_type=type(exc).__name__,
        )
    manifest_path = export_output.run_dir / "export_manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_EXPORT_RESULTS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.EXPORT_FAILED,
            message="export manifest unreadable",
            exception_type=type(exc).__name__,
        )
    manifest_ref = ArtifactRef(
        name="export_manifest",
        relative_path=f"{run_id}/exports/export_manifest.json",
        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        media_type="application/json",
        size_bytes=len(manifest_bytes),
    )
    run_root = export_output.run_dir.parent.parent
    exports = [
        str(path.relative_to(run_root).as_posix()) for path in export_output.files
    ]
    return {
        "status": WorkflowStatus.EXPORTING.value,
        "current_node": NODE_EXPORT_RESULTS,
        "export_manifest_ref": manifest_ref.model_dump(mode="json"),
        "exports": exports,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_EXPORT_RESULTS,
            event_type="exported",
            status=WorkflowStatus.EXPORTING.value,
            message=f"exported files={len(exports)}",
            metrics={"files": len(exports)},
            created_at=created_at,
        ),
    }


def finalize_success_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Verify the export manifest and key files before marking COMPLETED."""
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    if state.get("validation_passed") is not True:
        raise WorkflowInvariantError("finalize_success requires validation_passed=True")
    if (
        state.get("screening_result_ref") is None
        or state.get("export_manifest_ref") is None
    ):
        raise WorkflowInvariantError(
            "finalize_success requires screening_result_ref and export_manifest_ref"
        )
    try:
        result_payload = _read_artifact(state, context, "screening_result_ref")
        manifest_payload = _read_artifact(state, context, "export_manifest_ref")
    except ArtifactIntegrityError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED,
            message="artifact integrity check failed",
            exception_type=type(exc).__name__,
        )
    except ArtifactStoreError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact read failed",
            exception_type=type(exc).__name__,
        )
    try:
        result = ScreeningResult.model_validate(result_payload)
    except (TypeError, ValidationError) as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="screening_result artifact is invalid",
            exception_type=type(exc).__name__,
        )
    if not isinstance(manifest_payload, dict):
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="export manifest artifact is invalid",
            exception_type="TypeError",
        )
    request_text = json_text(result.request.model_dump(mode="json"))
    result_text = json_text(result.model_dump(mode="json"))
    expected_request_hash = hashlib.sha256(request_text.encode("utf-8")).hexdigest()
    expected_result_hash = hashlib.sha256(result_text.encode("utf-8")).hexdigest()
    if (
        manifest_payload.get("request_hash") != expected_request_hash
        or manifest_payload.get("result_hash") != expected_result_hash
    ):
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.EXPORT_FAILED,
            message="export manifest hash mismatch",
            exception_type="ExportError",
        )
    files_map = manifest_payload.get("files")
    required = {"request.json", "result.json", "candidates.csv", "report.md"}
    if not isinstance(files_map, dict) or not required.issubset(files_map):
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.EXPORT_FAILED,
            message="export manifest misses key files",
            exception_type="ExportError",
        )
    return {
        "status": WorkflowStatus.COMPLETED.value,
        "current_node": NODE_FINALIZE_SUCCESS,
        "finished_at": created_at,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_FINALIZE_SUCCESS,
            event_type="completed",
            status=WorkflowStatus.COMPLETED.value,
            message="workflow completed",
            metrics={},
            created_at=created_at,
        ),
    }


def finalize_planner_stop_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Terminate for planner stops without repository access or exports."""
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    raw_status = state.get("status")
    if raw_status not in {
        WorkflowStatus.NEEDS_CLARIFICATION.value,
        WorkflowStatus.INVALID_REQUEST.value,
        WorkflowStatus.UNSUPPORTED_REQUEST.value,
    }:
        raise WorkflowInvariantError(
            f"cannot finalize planner stop from status {raw_status!r}"
        )
    return {
        "status": raw_status,
        "current_node": NODE_FINALIZE_PLANNER_STOP,
        "finished_at": created_at,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_FINALIZE_PLANNER_STOP,
            event_type="planner_stop",
            status=raw_status,
            message=f"workflow stopped: {raw_status}",
            metrics={},
            created_at=created_at,
        ),
    }


def finalize_failure_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict[str, Any]:
    """Finalize a failed run; keeps only the safe error summary in state."""
    run_id = state["run_id"]
    context = runtime.context
    created_at = context.clock().isoformat()
    if state.get("status") != WorkflowStatus.FAILED.value:
        raise WorkflowInvariantError("finalize_failure requires status failed")
    return {
        "status": WorkflowStatus.FAILED.value,
        "current_node": NODE_FINALIZE_FAILURE,
        "finished_at": created_at,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_FINALIZE_FAILURE,
            event_type="failed",
            status=WorkflowStatus.FAILED.value,
            message="workflow failed",
            metrics={},
            created_at=created_at,
        ),
    }


def _resolve_query(
    state: WorkflowState,
    context: WorkflowContext,
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    query = state.get("user_query")
    if not query:
        raise WorkflowInputError("query mode requires user_query")
    try:
        planner = context.planner_service.parse(query)
    except (PlannerError, LLMError) as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RESOLVE_REQUEST,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.PLANNER_FAILED,
            message="planner failed",
            exception_type=type(exc).__name__,
        )

    status = _PLANNER_STATUS_TO_WORKFLOW[planner.status]
    update: dict[str, Any] = {
        "status": status.value,
        "current_node": NODE_RESOLVE_REQUEST,
        "planner_result": planner.model_dump(mode="json"),
        "planner_status": planner.status.value,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_RESOLVE_REQUEST,
            event_type="resolved",
            status=status.value,
            message=f"request resolved: {planner.status.value}",
            metrics={},
            created_at=created_at,
        ),
    }
    if planner.status is PlannerStatus.READY:
        if planner.request is None:
            raise WorkflowInvariantError("READY planner result must include a request")
        update["screening_request"] = planner.request.model_dump(mode="json")
    elif planner.status is PlannerStatus.NEEDS_CLARIFICATION:
        update["clarification_question"] = planner.clarification_question
    return update


def _resolve_request(
    state: WorkflowState,
    context: WorkflowContext,
    run_id: str,
    created_at: str,
) -> dict[str, Any]:
    raw_request = state.get("raw_request")
    if raw_request is None:
        raise WorkflowInputError("request mode requires raw_request")
    try:
        request = ScreeningRequest.model_validate(raw_request)
    except ValidationError as exc:
        return _failure_update(
            state=state,
            context=context,
            run_id=run_id,
            node=NODE_RESOLVE_REQUEST,
            created_at=created_at,
            status=WorkflowStatus.INVALID_REQUEST,
            code=WorkflowErrorCode.REQUEST_INVALID,
            message="request is invalid",
            exception_type=type(exc).__name__,
        )
    return {
        "status": WorkflowStatus.READY_FOR_RETRIEVAL.value,
        "current_node": NODE_RESOLVE_REQUEST,
        "screening_request": request.model_dump(mode="json"),
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=NODE_RESOLVE_REQUEST,
            event_type="resolved",
            status=WorkflowStatus.READY_FOR_RETRIEVAL.value,
            message="request resolved: ready",
            metrics={},
            created_at=created_at,
        ),
    }


def _rebuild_request(state: WorkflowState) -> ScreeningRequest:
    """Rebuild the validated ScreeningRequest stored in state."""
    raw = state.get("screening_request")
    if raw is None:
        raise WorkflowInvariantError(
            "screening_request missing; resolve_request must run first"
        )
    try:
        return ScreeningRequest.model_validate(raw)
    except ValidationError as exc:
        raise WorkflowInvariantError("screening_request in state is invalid") from exc


def _read_artifact(
    state: WorkflowState,
    context: WorkflowContext,
    field: str,
) -> object:
    """Read and hash-verify one artifact referenced by the state."""
    raw_ref = state.get(field)
    if raw_ref is None:
        raise WorkflowInvariantError(f"{field} missing; previous node must run first")
    try:
        ref = ArtifactRef.model_validate(raw_ref)
    except ValidationError as exc:
        raise WorkflowInvariantError(f"{field} in state is invalid") from exc
    return context.artifact_store.get_json(ref)


def _read_artifact_or_failure(
    state: WorkflowState,
    context: WorkflowContext,
    field: str,
    node: str,
    created_at: str,
) -> tuple[object | None, dict[str, Any] | None]:
    """Read one artifact; on expected errors return a FAILED partial update."""
    try:
        payload = _read_artifact(state, context, field)
    except ArtifactIntegrityError as exc:
        return None, _failure_update(
            state=state,
            context=context,
            run_id=state["run_id"],
            node=node,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_INTEGRITY_FAILED,
            message="artifact integrity check failed",
            exception_type=type(exc).__name__,
        )
    except ArtifactStoreError as exc:
        return None, _failure_update(
            state=state,
            context=context,
            run_id=state["run_id"],
            node=node,
            created_at=created_at,
            status=WorkflowStatus.FAILED,
            code=WorkflowErrorCode.ARTIFACT_READ_FAILED,
            message="artifact read failed",
            exception_type=type(exc).__name__,
        )
    return payload, None


def _zero_step(retrieved_count: int, filter_trace: FilterTrace) -> str:
    """Return the step at which the candidate count reached zero."""
    if retrieved_count == 0:
        return "retrieval"
    for step in filter_trace.steps:
        if step.after_count == 0:
            return step.name
    return "filter"


def _failure_update(
    *,
    state: WorkflowState,
    context: WorkflowContext,
    run_id: str,
    node: str,
    created_at: str,
    status: WorkflowStatus,
    code: WorkflowErrorCode,
    message: str,
    exception_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    error = WorkflowErrorData(
        code=code,
        node=node,
        message=message,
        retryable=False,
        exception_type=exception_type,
        occurred_at=created_at,
    ).model_dump(mode="json")
    update = {
        "status": status.value,
        "current_node": node,
        "error": error,
        "events": emit_event(
            state=state,
            run_id=run_id,
            node=node,
            event_type="failed",
            status=status.value,
            message=message,
            metrics={},
            created_at=created_at,
        ),
    }
    if extra:
        update.update(extra)
    return update


def _build_run_metadata(
    state: WorkflowState,
    retrieval: RetrievalResult,
    request: ScreeningRequest,
) -> RunMetadata:
    """Build deterministic run metadata for the preliminary result.

    ``finished_at`` is derived from the stable ``started_at`` so replayed
    runs produce byte-identical artifacts (idempotent replay).
    """
    started_raw = state.get("started_at")
    if not started_raw:
        raise WorkflowInvariantError(
            "started_at missing; initialize_run must run first"
        )
    try:
        started_at = datetime.fromisoformat(started_raw)
    except ValueError as exc:
        raise WorkflowInvariantError("started_at in state is invalid") from exc
    return RunMetadata(
        run_id=state["run_id"],
        started_at=started_at,
        finished_at=started_at,
        source=retrieval.source,
        database_version=retrieval.database_version,
        mp_api_version=_package_version("mp-api"),
        pymatgen_version=_package_version("pymatgen"),
        application_version=_package_version("materials-screening-core") or __version__,
        query_fingerprint=request_fingerprint(request),
    )


def _package_version(package: str) -> str | None:
    try:
        return package_version(package)
    except PackageNotFoundError:
        return None
