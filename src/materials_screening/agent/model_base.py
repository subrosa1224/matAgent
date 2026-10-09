"""Agent model abstraction: request, response and protocol (S3.5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentMessageItem,
    AgentTranscriptItem,
)
from materials_screening.agent.tool_base import AgentToolDefinition

AgentModelOutputItem = AgentMessageItem | AgentFunctionCallItem


class AgentModelStatus(StrEnum):
    """Terminal status of one agent model generation."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class MaterialAgentRequest:
    """Immutable, JSON-safe inputs for one agent model call.

    ``input_items`` carry the complete multi-turn transcript (the Responses
    API is stateless). Reasoning, web search calls and raw API responses are
    never part of a request.
    """

    instructions: str
    input_items: tuple[AgentTranscriptItem, ...]
    tool_definitions: tuple[AgentToolDefinition, ...]
    final_draft_schema: dict[str, Any]
    max_output_tokens: int = 4096
    reasoning_effort: str = "none"
    temperature: float = 0.0
    allow_tool_calls: bool = True


class MaterialAgentResponse(BaseModel):
    """Non-sensitive result of one agent model call.

    Forbidden in state/checkpoint/logs: API keys, reasoning text, raw HTTP
    bodies and tool outputs. Only safe summaries are kept in ``error``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AgentModelStatus
    output_items: tuple[AgentModelOutputItem, ...]
    request_id: str
    provider: str
    model: str
    latency_ms: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    error: str | None = None

    @model_validator(mode="after")
    def validate_status_content(self) -> Self:
        if (
            self.status is not AgentModelStatus.COMPLETED
            and not (self.error or "").strip()
        ):
            raise ValueError("incomplete/failed response requires an error summary")
        return self

    @property
    def message_text(self) -> str | None:
        """Return the assistant text of the first message item, if any."""
        for item in self.output_items:
            if isinstance(item, AgentMessageItem):
                return item.content
        return None

    @property
    def tool_calls(self) -> tuple[AgentFunctionCallItem, ...]:
        """Return the function call output items, in order."""
        return tuple(
            item
            for item in self.output_items
            if isinstance(item, AgentFunctionCallItem)
        )


@runtime_checkable
class MaterialAgentModel(Protocol):
    """Contract implemented by real and mock agent models."""

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse: ...
