"""Stage 3.5 agent tool policy and ownership (S3.5-M2)."""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.errors import AgentToolError
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.state import MaterialAgentState
from materials_screening.agent.tool_base import AgentTool, ToolSideEffect
from materials_screening.agent.tool_registry import ALLOWED_TOOL_NAMES

_THREAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,253}$")


class AgentPolicyError(AgentToolError):
    """Policy violation with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ConversationWorkflowLink(BaseModel):
    """Ownership record: one conversation may access its linked threads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)


class AgentToolPolicy:
    """Deterministic, stateless rules checked before every tool call."""

    def __init__(
        self,
        *,
        max_tool_calls_per_turn: int = 4,
        max_workflow_runs_per_turn: int = 1,
        max_argument_bytes: int = 8192,
    ) -> None:
        self._max_tool_calls = max_tool_calls_per_turn
        self._max_workflow_runs = max_workflow_runs_per_turn
        self._max_argument_bytes = max_argument_bytes

    def validate_call(
        self,
        *,
        tool: AgentTool,
        arguments: BaseModel,
        call_id: str,
        conversation_id: str,
        state: MaterialAgentState,
        links: Sequence[ConversationWorkflowLink],
        ledger: ToolExecutionLedger,
    ) -> None:
        """Raise AgentPolicyError when a tool call violates a rule."""
        if not call_id:
            raise AgentPolicyError("INVALID_CALL_ID", "call_id must not be empty")
        if ledger.has(call_id):
            raise AgentPolicyError(
                "DUPLICATE_CALL_ID",
                f"call_id already executed: {call_id!r}",
            )
        if tool.name not in ALLOWED_TOOL_NAMES:
            if tool.name == "web_search":
                raise AgentPolicyError(
                    "FORBIDDEN_OPERATION", "web search is not allowed"
                )
            raise AgentPolicyError(
                "UNKNOWN_TOOL",
                f"unknown or forbidden tool: {tool.name!r}",
            )
        if ledger.tool_call_count() + 1 > self._max_tool_calls:
            raise AgentPolicyError(
                "TOOL_CALL_LIMIT",
                f"tool call limit {self._max_tool_calls} reached",
            )
        if tool.side_effect is ToolSideEffect.CREATE_WORKFLOW_RUN:
            if ledger.workflow_run_count() + 1 > self._max_workflow_runs:
                raise AgentPolicyError(
                    "WORKFLOW_RUN_LIMIT",
                    f"workflow run limit {self._max_workflow_runs} reached",
                )
            if ledger.side_effect_executed():
                raise AgentPolicyError(
                    "SIDE_EFFECT_LIMIT",
                    "only one side-effect call is allowed per turn",
                )
        raw_bytes = len(arguments.model_dump_json().encode("utf-8"))
        if raw_bytes > self._max_argument_bytes:
            raise AgentPolicyError(
                "ARGUMENT_TOO_LARGE",
                f"arguments exceed {self._max_argument_bytes} bytes",
            )
        thread_id = getattr(arguments, "thread_id", None)
        if tool.side_effect is ToolSideEffect.READ_ONLY and thread_id is not None:
            self._validate_thread_access(
                thread_id=thread_id,
                conversation_id=conversation_id,
                state=state,
                links=links,
            )

    def _validate_thread_access(
        self,
        *,
        thread_id: object,
        conversation_id: str,
        state: MaterialAgentState,
        links: Sequence[ConversationWorkflowLink],
    ) -> None:
        if not isinstance(thread_id, str):
            raise AgentPolicyError("INVALID_THREAD_ID", "thread_id must be a string")
        if not _THREAD_ID_PATTERN.fullmatch(thread_id):
            raise AgentPolicyError(
                "INVALID_THREAD_ID", f"unsafe thread_id: {thread_id!r}"
            )
        allowed = self._allowed_threads(conversation_id, state, links)
        if thread_id not in allowed:
            raise AgentPolicyError(
                "OWNERSHIP_DENIED",
                f"thread {thread_id!r} is not owned by conversation "
                f"{conversation_id!r}",
            )

    @staticmethod
    def _allowed_threads(
        conversation_id: str,
        state: MaterialAgentState,
        links: Sequence[ConversationWorkflowLink],
    ) -> set[str]:
        allowed: set[str] = set()
        active = state.get("active_workflow_thread_id")
        if active is not None:
            allowed.add(active)
        allowed.update(
            link.thread_id for link in links if link.conversation_id == conversation_id
        )
        return allowed
