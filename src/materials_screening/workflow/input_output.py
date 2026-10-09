"""Stage 3 workflow input/output models (S3-M2)."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.workflow.state import WorkflowStatus


class WorkflowInput(BaseModel):
    """User-facing workflow input; exactly one of query or request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str | None = None
    request: dict[str, Any] | None = None
    output_root: str = "data/workflow_runs"
    export_cif: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        if (self.query is None) == (self.request is None):
            raise ValueError("Exactly one of query or request must be provided")
        return self


class WorkflowGraphInput(BaseModel):
    """Internal graph input assembled by the runner before invoke."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    workflow_version: str = Field(min_length=1)
    input_mode: Literal["query", "request"]
    user_query: str | None = None
    raw_request: dict[str, Any] | None = None
    output_root: str = "data/workflow_runs"
    export_cif: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> Self:
        has_query = self.user_query is not None
        has_request = self.raw_request is not None
        if has_query == has_request:
            raise ValueError(
                "Exactly one of user_query or raw_request must be provided"
            )
        expected = "query" if has_query else "request"
        if self.input_mode != expected:
            raise ValueError("input_mode must match the provided input")
        return self


class WorkflowOutput(BaseModel):
    """Safe run output; never exposes internal artifact content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    thread_id: str
    status: WorkflowStatus
    planner_status: str | None
    request: dict[str, Any] | None
    retrieved_count: int
    filtered_count: int
    returned_count: int
    validation_passed: bool | None
    exports: tuple[str, ...]
    warnings: tuple[str, ...]
    error: dict[str, Any] | None
    clarification_question: str | None
