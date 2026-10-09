"""Stage 3.5 agent data models (S3.5-M1)."""

import json
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AGENT_ANSWER_MAX_LENGTH = 8000


class AgentFinalStatus(StrEnum):
    """Final status of one agent turn."""

    COMPLETED = "completed"
    NEEDS_USER_INPUT = "needs_user_input"
    ERROR = "error"


class AgentFinalDraft(BaseModel):
    """Structured final answer with mandatory tool evidence references."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentFinalStatus
    answer: str = Field(min_length=1, max_length=AGENT_ANSWER_MAX_LENGTH)
    active_workflow_thread_id: str | None = None
    referenced_material_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    follow_up_question: str | None = None

    @model_validator(mode="after")
    def validate_invariants(self) -> Self:
        if (
            self.status is AgentFinalStatus.NEEDS_USER_INPUT
            and not (self.follow_up_question or "").strip()
        ):
            raise ValueError("NEEDS_USER_INPUT requires a follow_up_question")
        return self


class AgentResult(BaseModel):
    """Safe result of one agent ask; never exposes the raw transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1)
    user_turn_id: str = Field(min_length=1)
    status: str
    final_status: str | None = None
    response_text: str
    active_workflow_thread_id: str | None = None
    selected_tools: tuple[str, ...] = ()
    tool_call_count: int = Field(default=0, ge=0)
    model_call_count: int = Field(default=0, ge=0)
    evidence_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: dict[str, Any] | None = None


class AgentToolCall(BaseModel):
    """One parsed tool call emitted by the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


class ToolResultStatus(StrEnum):
    """Status of one tool execution result."""

    OK = "ok"
    ERROR = "error"


class ToolErrorData(BaseModel):
    """Safe tool error summary sent back to the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    retryable: bool


class ToolResultEnvelope(BaseModel):
    """Compact, safe envelope for every tool output sent to the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ToolResultStatus
    tool_name: str
    call_id: str
    evidence_id: str
    output: dict[str, Any] | None = None
    error: ToolErrorData | None = None

    @model_validator(mode="after")
    def validate_status_content(self) -> Self:
        if self.status is ToolResultStatus.OK and self.output is None:
            raise ValueError("OK envelope requires output")
        if self.status is ToolResultStatus.ERROR and self.error is None:
            raise ValueError("ERROR envelope requires error")
        return self

    def to_json(self) -> str:
        """Compact JSON for the Responses API function_call_output item."""
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class AgentMessageItem(BaseModel):
    """Transcript item: plain user or assistant message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["message"] = "message"
    role: Literal["user", "assistant"]
    content: str


class AgentFunctionCallItem(BaseModel):
    """Transcript item: assistant function call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["function_call"] = "function_call"
    call_id: str
    name: str
    arguments: str


class AgentFunctionOutputItem(BaseModel):
    """Transcript item: tool result for a matching call id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


AgentTranscriptItem = AgentMessageItem | AgentFunctionCallItem | AgentFunctionOutputItem


class AgentEvent(BaseModel):
    """One audit event emitted by an agent node."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    conversation_id: str
    node: str
    event_type: str
    created_at: str
    status: str
    message: str
    metrics: dict[str, int | float | str | bool | None]


class AgentErrorData(BaseModel):
    """Safe error summary stored in agent state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    node: str
    message: str
    retryable: bool
    exception_type: str
    occurred_at: str
