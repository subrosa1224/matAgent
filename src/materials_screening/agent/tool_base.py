"""Stage 3.5 agent tool abstraction (S3.5-M2)."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

if TYPE_CHECKING:
    from materials_screening.agent.context import AgentToolContext


class ToolSideEffect(StrEnum):
    """Side-effect class of one tool execution."""

    READ_ONLY = "read_only"
    CREATE_WORKFLOW_RUN = "create_workflow_run"


class AgentTool(Protocol):
    """Structural contract implemented by every whitelisted tool."""

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    side_effect: ToolSideEffect

    def execute(
        self,
        arguments: BaseModel,
        context: AgentToolContext,
    ) -> BaseModel: ...


class AgentToolDefinition(BaseModel):
    """Validated tool metadata used for model function definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]
    side_effect: ToolSideEffect
    version: str

    @model_validator(mode="after")
    def validate_parameters_schema(self) -> Self:
        schema = self.parameters
        if schema.get("type") != "object":
            raise ValueError("parameters schema must be an object")
        if schema.get("additionalProperties", True) is not False:
            raise ValueError("parameters schema must set additionalProperties=false")
        return self


def to_function_definition(
    definition: AgentToolDefinition,
) -> dict[str, Any]:
    """Convert a validated definition to an OpenAI-compatible function tool."""
    return {
        "type": "function",
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
    }
