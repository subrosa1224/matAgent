"""Stage 3 workflow state models (S3-M2)."""

from enum import StrEnum
from operator import add
from pathlib import PurePosixPath
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class WorkflowStatus(StrEnum):
    """Lifecycle status of one workflow run."""

    INITIALIZING = "initializing"
    RESOLVING_REQUEST = "resolving_request"
    READY_FOR_RETRIEVAL = "ready_for_retrieval"
    RETRIEVING = "retrieving"
    FILTERING = "filtering"
    RANKING = "ranking"
    VALIDATING = "validating"
    EXPORTING = "exporting"

    NEEDS_CLARIFICATION = "needs_clarification"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_REQUEST = "unsupported_request"
    NO_RESULTS = "no_results"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_WORKFLOW_STATUSES: frozenset[WorkflowStatus] = frozenset(
    {
        WorkflowStatus.NEEDS_CLARIFICATION,
        WorkflowStatus.INVALID_REQUEST,
        WorkflowStatus.UNSUPPORTED_REQUEST,
        WorkflowStatus.NO_RESULTS,
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
    }
)


class ArtifactRef(BaseModel):
    """Reference to one artifact stored in the run artifact store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    relative_path: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    media_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    item_count: int | None = Field(default=None, ge=0)
    schema_name: str | None = None
    schema_version: str | None = None

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            raise ValueError("relative_path must be a safe relative path")
        return value


class WorkflowStateView(BaseModel):
    """Safe projection of workflow state for status/history consumers."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    thread_id: str
    workflow_version: str
    status: str
    current_node: str | None
    started_at: str | None
    finished_at: str | None
    planner_status: str | None
    retrieved_count: int
    filtered_count: int
    returned_count: int
    validation_passed: bool | None
    exports: tuple[str, ...]
    warnings: tuple[str, ...]
    error: dict[str, Any] | None


class WorkflowCheckpointView(BaseModel):
    """Safe summary of one checkpoint for history display."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(ge=0)
    checkpoint_id: str = Field(min_length=1)
    source: str
    current_node: str | None
    status: str
    next_nodes: tuple[str, ...]
    created_at: str | None


class WorkflowState(TypedDict, total=False):
    """Lightweight, JSON-serializable graph state.

    All values must stay JSON-native: no Pydantic models, ``Path``,
    ``datetime``, exceptions, services, connections or large material
    records. Large intermediate results live in the run artifact store and
    only ``ArtifactRef`` dicts are kept here.
    """

    run_id: str
    workflow_version: str
    input_mode: str
    user_query: str | None
    raw_request: dict[str, Any] | None
    output_root: str
    export_cif: bool

    status: str
    current_node: str | None
    started_at: str | None
    finished_at: str | None

    planner_result: dict[str, Any] | None
    planner_status: str | None
    clarification_question: str | None
    screening_request: dict[str, Any] | None

    retrieval_ref: dict[str, Any] | None
    filtered_ref: dict[str, Any] | None
    filter_trace_ref: dict[str, Any] | None
    ranked_ref: dict[str, Any] | None
    validation_ref: dict[str, Any] | None
    screening_result_ref: dict[str, Any] | None
    export_manifest_ref: dict[str, Any] | None

    retrieved_count: int
    filtered_count: int
    returned_count: int
    validation_passed: bool | None

    exports: list[str]
    warnings: list[str]
    error: dict[str, Any] | None
    events: Annotated[list[dict[str, Any]], add]
