"""Stage 3 workflow errors and safe error data model (S3-M2)."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class WorkflowErrorCode(StrEnum):
    """Stable machine-readable workflow error codes (STAGE3 doc section 25)."""

    WORKFLOW_INPUT_INVALID = "WORKFLOW_INPUT_INVALID"
    PLANNER_FAILED = "PLANNER_FAILED"
    REQUEST_INVALID = "REQUEST_INVALID"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    ARTIFACT_WRITE_FAILED = "ARTIFACT_WRITE_FAILED"
    ARTIFACT_READ_FAILED = "ARTIFACT_READ_FAILED"
    ARTIFACT_INTEGRITY_FAILED = "ARTIFACT_INTEGRITY_FAILED"
    FILTER_FAILED = "FILTER_FAILED"
    RANKING_FAILED = "RANKING_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXPORT_FAILED = "EXPORT_FAILED"
    CHECKPOINT_FAILED = "CHECKPOINT_FAILED"
    WORKFLOW_INVARIANT_FAILED = "WORKFLOW_INVARIANT_FAILED"
    WORKFLOW_RECURSION_LIMIT = "WORKFLOW_RECURSION_LIMIT"
    UNEXPECTED_WORKFLOW_ERROR = "UNEXPECTED_WORKFLOW_ERROR"


class WorkflowErrorData(BaseModel):
    """Safe error summary stored in state and returned to consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: WorkflowErrorCode
    node: str
    message: str
    retryable: bool
    exception_type: str
    occurred_at: str


class WorkflowError(Exception):
    """Base exception for all workflow-layer errors."""


class WorkflowInputError(WorkflowError):
    """Raised when workflow input is invalid."""


class WorkflowInvariantError(WorkflowError):
    """Raised when a state invariant or routing rule is violated."""


class WorkflowCheckpointError(WorkflowError):
    """Raised when checkpoint I/O or thread handling fails."""


class ArtifactStoreError(WorkflowError):
    """Raised when artifact storage operations fail."""


class ArtifactConflictError(ArtifactStoreError):
    """Raised when an artifact write conflicts with existing content."""


class ArtifactIntegrityError(ArtifactStoreError):
    """Raised when artifact content fails hash verification."""
