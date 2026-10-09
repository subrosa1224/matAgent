"""Master agent graph state."""

from operator import add
from typing import Annotated, Any, TypedDict


class MasterAgentState(TypedDict, total=False):
    """Lightweight, JSON-serializable master agent graph state.

    Same structure as ``MaterialAgentState`` (state.py:7-39) with differences:
    - ``sub_agent_call_count`` replaces ``tool_call_count``/``workflow_run_count``
    - ``sub_agent_results`` is an Annotated list accumulator
    - ``active_workflow_thread_id`` tracks the latest screening batch returned
      by a sub-agent so follow-up turns can refer to "these materials".
    """

    conversation_id: str
    user_turn_id: str
    user_message: str
    fulltext_tasks: dict[str, dict[str, Any]]

    status: str
    current_node: str

    input_items: list[dict[str, Any]]

    model_call_count: int
    sub_agent_call_count: int

    pending_tool_calls: list[dict[str, Any]]
    executed_call_ids: list[str]
    sub_agent_results: Annotated[list[dict[str, Any]], add]
    evidence_ids: list[str]
    active_workflow_thread_id: str | None

    cancelled: bool

    final_draft: dict[str, Any] | None
    final_response: str | None
    error: dict[str, Any] | None

    events: Annotated[list[dict[str, Any]], add]
