"""Sub-agent specification models — the universal pluggable contract."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.tool_base import AgentToolDefinition


@dataclass(frozen=True)
class SubAgentSpec:
    """Immutable specification of one pluggable sub-agent.

    - name: unique id, e.g. 'materials_screening'
    - description: model-visible capability description
    - system_prompt: the sub-agent's persona/skill template
    - tool_definitions: the sub-agent's whitelisted function tools
    - runner_factory: callable returning an object with ask(message=...)
    - delegate_function_name: function name the master model uses to delegate
    """

    name: str
    description: str
    system_prompt: str
    tool_definitions: tuple[AgentToolDefinition, ...]
    runner_factory: Callable[[], Any]
    delegate_function_name: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"invalid sub-agent name: {self.name!r}")
        if not self.description.strip():
            raise ValueError(f"sub-agent {self.name!r} must have a description")
        if not self.system_prompt.strip():
            raise ValueError(f"sub-agent {self.name!r} must have a system prompt")
        if not self.delegate_function_name:
            object.__setattr__(
                self, "delegate_function_name", f"delegate_to_{self.name}"
            )


class SubAgentDefinition(BaseModel):
    """Validated definition exposed to the master model as a function tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1)
    parameters: dict[str, Any]


class SubAgentCall(BaseModel):
    """One parsed sub-agent delegation call from the master model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments_json: str


class SubAgentResultEnvelope(BaseModel):
    """Safe envelope for a sub-agent result sent back to the master model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str  # "ok" | "error"
    sub_agent_name: str
    call_id: str
    task: str  # the task that was delegated
    response_text: str  # sub-agent's final answer
    active_workflow_thread_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
