"""Stage 3.5 agent tool context (S3.5-M2)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.model_base import MaterialAgentModel
from materials_screening.agent.policy import ConversationWorkflowLink
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.workflow.context import Clock, IdGenerator

if TYPE_CHECKING:
    from materials_screening.agent.tool_executor import ToolExecutor
    from materials_screening.workflow.runner import WorkflowRunner


class WorkflowResultReader(Protocol):
    """Reads validated screening results for read-only tools (M3)."""

    def read(self, thread_id: str) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AgentToolContext:
    """Read-only dependencies passed to every tool execution."""

    workflow_runner: WorkflowRunner
    workflow_result_reader: WorkflowResultReader
    clock: Clock
    id_generator: IdGenerator
    ledger: ToolExecutionLedger
    call_id: str
    user_turn_id: str
    conversation_id: str
    active_workflow_thread_id: str | None = None
    conversation_links: tuple[ConversationWorkflowLink, ...] = ()


@dataclass(frozen=True)
class MaterialAgentContext:
    """Immutable runtime dependencies injected into every agent graph node.

    Delivered through LangGraph's ``Runtime.context``; never serialized into
    state, checkpoints or artifacts. Building this context must not construct
    real Intern or Materials Project clients; the runner wires providers by
    name right before invoking the graph.
    """

    workflow_runner: WorkflowRunner
    workflow_result_reader: WorkflowResultReader
    tool_registry: AgentToolRegistry
    tool_executor: ToolExecutor
    agent_model: MaterialAgentModel
    ledger: ToolExecutionLedger
    settings: AgentSettings
    clock: Clock
    id_generator: IdGenerator
    conversation_links: tuple[ConversationWorkflowLink, ...] = ()
    cancel_event: threading.Event | None = None
