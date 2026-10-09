"""Stage 3.5 agent graph state (S3.5-M1)."""

from operator import add
from typing import Annotated, Any, TypedDict


class MaterialAgentState(TypedDict, total=False):
    """Lightweight, JSON-serializable agent graph state.

    Forbidden values: API keys, WorkflowRunner objects, full workflow state,
    crystal structures, tracebacks, reasoning text, raw HTTP responses and
    SQLite connections.
    """

    conversation_id: str
    user_turn_id: str
    user_message: str

    status: str
    current_node: str

    input_items: list[dict[str, Any]]
    active_workflow_thread_id: str | None

    model_call_count: int
    tool_call_count: int
    workflow_run_count: int

    pending_tool_calls: list[dict[str, Any]]
    executed_call_ids: list[str]
    evidence_ids: list[str]

    cancelled: bool

    final_draft: dict[str, Any] | None
    final_response: str | None
    error: dict[str, Any] | None

    events: Annotated[list[dict[str, Any]], add]
