"""Master agent graph builder."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from .master_context import MasterAgentContext
from .master_nodes import (
    NODE_CALL_MASTER_MODEL,
    NODE_EXECUTE_SUB_AGENT,
    NODE_FINALIZE_CANCELLED,
    NODE_FINALIZE_ERROR,
    NODE_FINALIZE_SUCCESS,
    NODE_PREPARE_TURN,
    NODE_VALIDATE_FINAL,
    call_master_model_node,
    execute_sub_agent_node,
    finalize_cancelled_node,
    finalize_error_node,
    finalize_success_node,
    prepare_turn_node,
    route_after_model,
    route_after_sub_agent,
    route_after_validation,
    validate_final_node,
)
from .master_state import MasterAgentState

MASTER_GRAPH_NAME = "master-agent-v1"


class MasterGraphInput(BaseModel):
    """Validated input for one master agent turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    conversation_id: str = Field(min_length=1, max_length=255)
    user_message: str = Field(min_length=1)


def create_master_agent_builder() -> StateGraph:
    """Build the master agent graph.

    Edge structure is IDENTICAL to ``create_agent_builder()``
    (graph_builder.py:48-104):
    prepare_turn -> call_master_model ->
        [execute_sub_agent | validate_final | error | cancel]
    execute_sub_agent -> call_master_model (loop)
    """
    builder = StateGraph(
        MasterAgentState,
        context_schema=MasterAgentContext,
        input_schema=MasterGraphInput,
    )
    builder.add_node(NODE_PREPARE_TURN, prepare_turn_node)
    builder.add_node(NODE_CALL_MASTER_MODEL, call_master_model_node)
    builder.add_node(NODE_EXECUTE_SUB_AGENT, execute_sub_agent_node)
    builder.add_node(NODE_VALIDATE_FINAL, validate_final_node)
    builder.add_node(NODE_FINALIZE_SUCCESS, finalize_success_node)
    builder.add_node(NODE_FINALIZE_ERROR, finalize_error_node)
    builder.add_node(NODE_FINALIZE_CANCELLED, finalize_cancelled_node)

    builder.add_edge(START, NODE_PREPARE_TURN)
    builder.add_edge(NODE_PREPARE_TURN, NODE_CALL_MASTER_MODEL)

    builder.add_conditional_edges(
        NODE_CALL_MASTER_MODEL,
        route_after_model,
        {
            "execute_sub_agent": NODE_EXECUTE_SUB_AGENT,
            "validate_final": NODE_VALIDATE_FINAL,
            "finalize_error": NODE_FINALIZE_ERROR,
            "finalize_cancelled": NODE_FINALIZE_CANCELLED,
        },
    )
    builder.add_conditional_edges(
        NODE_EXECUTE_SUB_AGENT,
        route_after_sub_agent,
        {
            "call_master_model": NODE_CALL_MASTER_MODEL,
            "finalize_error": NODE_FINALIZE_ERROR,
            "finalize_cancelled": NODE_FINALIZE_CANCELLED,
        },
    )
    builder.add_conditional_edges(
        NODE_VALIDATE_FINAL,
        route_after_validation,
        {
            "finalize_success": NODE_FINALIZE_SUCCESS,
            "finalize_error": NODE_FINALIZE_ERROR,
        },
    )

    builder.add_edge(NODE_FINALIZE_SUCCESS, END)
    builder.add_edge(NODE_FINALIZE_ERROR, END)
    builder.add_edge(NODE_FINALIZE_CANCELLED, END)

    builder.validate()
    return builder


def compile_master_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
) -> Any:
    """Compile the master agent graph with a checkpointer."""
    builder = create_master_agent_builder()
    return builder.compile(checkpointer=checkpointer, name=MASTER_GRAPH_NAME)
