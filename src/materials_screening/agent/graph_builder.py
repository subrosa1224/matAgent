"""Single agent graph builder (S3.5-M5)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import MaterialAgentContext
from materials_screening.agent.nodes import (
    NODE_CALL_AGENT_MODEL,
    NODE_EXECUTE_TOOLS,
    NODE_FINALIZE_CANCELLED,
    NODE_FINALIZE_ERROR,
    NODE_FINALIZE_SUCCESS,
    NODE_PREPARE_TURN,
    NODE_VALIDATE_FINAL,
    call_agent_model_node,
    execute_tools_node,
    finalize_cancelled_node,
    finalize_error_node,
    finalize_success_node,
    prepare_turn_node,
    route_after_model,
    route_after_tools,
    route_after_validation,
    validate_final_node,
)
from materials_screening.agent.state import MaterialAgentState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

GRAPH_NAME = "material-agent-v1"


class AgentGraphInput(BaseModel):
    """Validated input for one agent turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str = Field(min_length=1, max_length=255)
    user_message: str = Field(min_length=1)


def create_agent_builder() -> StateGraph[
    MaterialAgentState,
    MaterialAgentContext,
    AgentGraphInput,
    MaterialAgentState,
]:
    """Build the fixed single-agent graph with explicit conditional path maps."""
    builder = StateGraph(
        MaterialAgentState,
        context_schema=MaterialAgentContext,
        input_schema=AgentGraphInput,
    )
    builder.add_node(NODE_PREPARE_TURN, prepare_turn_node)
    builder.add_node(NODE_CALL_AGENT_MODEL, call_agent_model_node)
    builder.add_node(NODE_EXECUTE_TOOLS, execute_tools_node)
    builder.add_node(NODE_VALIDATE_FINAL, validate_final_node)
    builder.add_node(NODE_FINALIZE_SUCCESS, finalize_success_node)
    builder.add_node(NODE_FINALIZE_ERROR, finalize_error_node)
    builder.add_node(NODE_FINALIZE_CANCELLED, finalize_cancelled_node)

    builder.add_edge(START, NODE_PREPARE_TURN)
    builder.add_edge(NODE_PREPARE_TURN, NODE_CALL_AGENT_MODEL)

    builder.add_conditional_edges(
        NODE_CALL_AGENT_MODEL,
        route_after_model,
        {
            "execute_tools": NODE_EXECUTE_TOOLS,
            "validate_final": NODE_VALIDATE_FINAL,
            "finalize_error": NODE_FINALIZE_ERROR,
            "finalize_cancelled": NODE_FINALIZE_CANCELLED,
        },
    )
    builder.add_conditional_edges(
        NODE_EXECUTE_TOOLS,
        route_after_tools,
        {
            "call_agent_model": NODE_CALL_AGENT_MODEL,
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


def compile_agent_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[
    MaterialAgentState,
    MaterialAgentContext,
    AgentGraphInput,
    MaterialAgentState,
]:
    """Compile the agent graph with a checkpointer; no clients are created."""
    builder = create_agent_builder()
    return builder.compile(checkpointer=checkpointer, name=GRAPH_NAME)
