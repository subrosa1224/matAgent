"""Unit tests for Stage 3 workflow errors (S3-M2)."""

import json

import pytest
from pydantic import ValidationError

from materials_screening.workflow.errors import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactStoreError,
    WorkflowCheckpointError,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowErrorData,
    WorkflowInputError,
    WorkflowInvariantError,
)


class TestWorkflowErrorCode:
    def test_values_match_document(self) -> None:
        expected = {
            "WORKFLOW_INPUT_INVALID",
            "PLANNER_FAILED",
            "REQUEST_INVALID",
            "RETRIEVAL_FAILED",
            "ARTIFACT_WRITE_FAILED",
            "ARTIFACT_READ_FAILED",
            "ARTIFACT_INTEGRITY_FAILED",
            "FILTER_FAILED",
            "RANKING_FAILED",
            "VALIDATION_FAILED",
            "EXPORT_FAILED",
            "CHECKPOINT_FAILED",
            "WORKFLOW_INVARIANT_FAILED",
            "WORKFLOW_RECURSION_LIMIT",
            "UNEXPECTED_WORKFLOW_ERROR",
        }
        assert {code.value for code in WorkflowErrorCode} == expected


class TestWorkflowErrorData:
    def test_valid(self) -> None:
        error = WorkflowErrorData(
            code=WorkflowErrorCode.PLANNER_FAILED,
            node="resolve_request",
            message="planner failed",
            retryable=False,
            exception_type="LLMError",
            occurred_at="2026-08-06T00:00:00Z",
        )
        assert error.code is WorkflowErrorCode.PLANNER_FAILED
        assert error.retryable is False

    def test_json_serializable(self) -> None:
        error = WorkflowErrorData(
            code=WorkflowErrorCode.PLANNER_FAILED,
            node="resolve_request",
            message="planner failed",
            retryable=False,
            exception_type="LLMError",
            occurred_at="2026-08-06T00:00:00Z",
        )
        dumped = error.model_dump(mode="json")
        assert dumped["code"] == "PLANNER_FAILED"
        assert json.loads(json.dumps(dumped)) == dumped

    def test_unknown_code_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowErrorData(
                code="NOT_A_CODE",
                node="resolve_request",
                message="boom",
                retryable=False,
                exception_type="RuntimeError",
                occurred_at="2026-08-06T00:00:00Z",
            )

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowErrorData(
                code=WorkflowErrorCode.PLANNER_FAILED,
                node="resolve_request",
                message="boom",
                retryable=False,
                exception_type="RuntimeError",
                occurred_at="2026-08-06T00:00:00Z",
                junk=1,
            )

    def test_frozen(self) -> None:
        assert WorkflowErrorData.model_config.get("frozen") is True


class TestWorkflowExceptions:
    def test_base_hierarchy(self) -> None:
        assert issubclass(WorkflowInputError, WorkflowError)
        assert issubclass(WorkflowInvariantError, WorkflowError)
        assert issubclass(WorkflowCheckpointError, WorkflowError)
        assert issubclass(ArtifactStoreError, WorkflowError)

    def test_artifact_error_hierarchy(self) -> None:
        assert issubclass(ArtifactConflictError, ArtifactStoreError)
        assert issubclass(ArtifactIntegrityError, ArtifactStoreError)

    def test_errors_are_catchable_as_base(self) -> None:
        for exc in (
            WorkflowInputError("input"),
            WorkflowInvariantError("invariant"),
            WorkflowCheckpointError("checkpoint"),
            ArtifactStoreError("store"),
            ArtifactConflictError("conflict"),
            ArtifactIntegrityError("integrity"),
        ):
            with pytest.raises(WorkflowError):
                raise exc
