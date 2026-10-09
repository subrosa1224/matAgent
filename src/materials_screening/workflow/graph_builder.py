"""Fixed workflow graph builder (S3-M5)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from materials_screening.workflow.context import WorkflowContext
from materials_screening.workflow.input_output import (
    WorkflowGraphInput,
    WorkflowOutput,
)
from materials_screening.workflow.nodes import (
    NODE_EXPORT_RESULTS,
    NODE_FILTER_MATERIALS,
    NODE_FINALIZE_FAILURE,
    NODE_FINALIZE_NO_RESULTS,
    NODE_FINALIZE_PLANNER_STOP,
    NODE_FINALIZE_SUCCESS,
    NODE_INITIALIZE_RUN,
    NODE_RANK_MATERIALS,
    NODE_RESOLVE_REQUEST,
    NODE_RETRIEVE_MATERIALS,
    NODE_VALIDATE_RESULTS,
    export_results_node,
    filter_materials_node,
    finalize_failure_node,
    finalize_no_results_node,
    finalize_planner_stop_node,
    finalize_success_node,
    initialize_run_node,
    rank_materials_node,
    resolve_request_node,
    retrieve_materials_node,
    validate_results_node,
)
from materials_screening.workflow.routing import (
    route_after_export,
    route_after_filter,
    route_after_ranking,
    route_after_request,
    route_after_retrieval,
    route_after_validation,
)
from materials_screening.workflow.state import WorkflowState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

GRAPH_NAME = "materials-screening-workflow-v1"


def create_workflow_builder() -> StateGraph[
    WorkflowState, WorkflowContext, WorkflowGraphInput, WorkflowOutput
]:
    """Build the fixed workflow graph with explicit conditional path maps."""
    builder = StateGraph(
        WorkflowState,
        context_schema=WorkflowContext,
        input_schema=WorkflowGraphInput,
        output_schema=WorkflowOutput,
    )
    builder.add_node(NODE_INITIALIZE_RUN, initialize_run_node)
    builder.add_node(NODE_RESOLVE_REQUEST, resolve_request_node)
    builder.add_node(NODE_RETRIEVE_MATERIALS, retrieve_materials_node)
    builder.add_node(NODE_FILTER_MATERIALS, filter_materials_node)
    builder.add_node(NODE_RANK_MATERIALS, rank_materials_node)
    builder.add_node(NODE_VALIDATE_RESULTS, validate_results_node)
    builder.add_node(NODE_EXPORT_RESULTS, export_results_node)
    builder.add_node(NODE_FINALIZE_SUCCESS, finalize_success_node)
    builder.add_node(NODE_FINALIZE_NO_RESULTS, finalize_no_results_node)
    builder.add_node(NODE_FINALIZE_PLANNER_STOP, finalize_planner_stop_node)
    builder.add_node(NODE_FINALIZE_FAILURE, finalize_failure_node)

    builder.add_edge(START, NODE_INITIALIZE_RUN)
    builder.add_edge(NODE_INITIALIZE_RUN, NODE_RESOLVE_REQUEST)

    builder.add_conditional_edges(
        NODE_RESOLVE_REQUEST,
        route_after_request,
        {
            "retrieve_materials": NODE_RETRIEVE_MATERIALS,
            "finalize_planner_stop": NODE_FINALIZE_PLANNER_STOP,
            "finalize_failure": NODE_FINALIZE_FAILURE,
        },
    )
    builder.add_conditional_edges(
        NODE_RETRIEVE_MATERIALS,
        route_after_retrieval,
        {
            "filter_materials": NODE_FILTER_MATERIALS,
            "finalize_failure": NODE_FINALIZE_FAILURE,
        },
    )
    builder.add_conditional_edges(
        NODE_FILTER_MATERIALS,
        route_after_filter,
        {
            "rank_materials": NODE_RANK_MATERIALS,
            "finalize_no_results": NODE_FINALIZE_NO_RESULTS,
            "finalize_failure": NODE_FINALIZE_FAILURE,
        },
    )
    builder.add_conditional_edges(
        NODE_RANK_MATERIALS,
        route_after_ranking,
        {
            "validate_results": NODE_VALIDATE_RESULTS,
            "finalize_failure": NODE_FINALIZE_FAILURE,
        },
    )
    builder.add_conditional_edges(
        NODE_VALIDATE_RESULTS,
        route_after_validation,
        {
            "export_results": NODE_EXPORT_RESULTS,
            "finalize_failure": NODE_FINALIZE_FAILURE,
        },
    )
    builder.add_conditional_edges(
        NODE_EXPORT_RESULTS,
        route_after_export,
        {
            "finalize_success": NODE_FINALIZE_SUCCESS,
            "finalize_failure": NODE_FINALIZE_FAILURE,
        },
    )

    builder.add_edge(NODE_FINALIZE_SUCCESS, END)
    builder.add_edge(NODE_FINALIZE_NO_RESULTS, END)
    builder.add_edge(NODE_FINALIZE_PLANNER_STOP, END)
    builder.add_edge(NODE_FINALIZE_FAILURE, END)

    builder.validate()
    return builder


def compile_workflow(
    *,
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph[
    WorkflowState, WorkflowContext, WorkflowGraphInput, WorkflowOutput
]:
    """Compile the workflow with a checkpointer; no clients are created."""
    builder = create_workflow_builder()
    return builder.compile(checkpointer=checkpointer, name=GRAPH_NAME)
