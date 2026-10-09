"""Pure workflow routing functions (S3-M4)."""

from typing import Literal

from materials_screening.workflow.errors import WorkflowInvariantError
from materials_screening.workflow.state import WorkflowState, WorkflowStatus

RouteAfterRequest = Literal[
    "retrieve_materials",
    "finalize_planner_stop",
    "finalize_failure",
]
RouteAfterRetrieval = Literal[
    "filter_materials",
    "finalize_failure",
]
RouteAfterFilter = Literal[
    "rank_materials",
    "finalize_no_results",
    "finalize_failure",
]
RouteAfterRanking = Literal[
    "validate_results",
    "finalize_failure",
]
RouteAfterValidation = Literal[
    "export_results",
    "finalize_failure",
]
RouteAfterExport = Literal[
    "finalize_success",
    "finalize_failure",
]


def _workflow_status(state: WorkflowState) -> WorkflowStatus:
    raw_status = state.get("status")
    if raw_status is None:
        raise WorkflowInvariantError("workflow state has no status")
    try:
        return WorkflowStatus(raw_status)
    except ValueError as exc:
        raise WorkflowInvariantError(
            f"unknown workflow status: {raw_status!r}"
        ) from exc


def route_after_request(state: WorkflowState) -> RouteAfterRequest:
    """Route after resolve_request; pure, no I/O and no state mutation."""
    status = _workflow_status(state)
    if status is WorkflowStatus.READY_FOR_RETRIEVAL:
        return "retrieve_materials"
    if status in {
        WorkflowStatus.NEEDS_CLARIFICATION,
        WorkflowStatus.INVALID_REQUEST,
        WorkflowStatus.UNSUPPORTED_REQUEST,
    }:
        return "finalize_planner_stop"
    if status is WorkflowStatus.FAILED:
        return "finalize_failure"
    raise WorkflowInvariantError(f"cannot route from status {status.value!r}")


def route_after_retrieval(state: WorkflowState) -> RouteAfterRetrieval:
    """Route after retrieve_materials; pure, no I/O and no state mutation."""
    status = _workflow_status(state)
    if status is WorkflowStatus.FAILED:
        return "finalize_failure"
    if status is WorkflowStatus.RETRIEVING:
        return "filter_materials"
    raise WorkflowInvariantError(f"cannot route from status {status.value!r}")


def route_after_filter(state: WorkflowState) -> RouteAfterFilter:
    """Route after filter_materials; pure, no I/O and no state mutation."""
    status = _workflow_status(state)
    if status is WorkflowStatus.FAILED:
        return "finalize_failure"
    if status is WorkflowStatus.NO_RESULTS:
        return "finalize_no_results"
    if status is WorkflowStatus.FILTERING:
        return "rank_materials"
    raise WorkflowInvariantError(f"cannot route from status {status.value!r}")


def route_after_ranking(state: WorkflowState) -> RouteAfterRanking:
    """Route after rank_materials; pure, no I/O and no state mutation."""
    status = _workflow_status(state)
    if status is WorkflowStatus.FAILED:
        return "finalize_failure"
    if status is WorkflowStatus.RANKING:
        return "validate_results"
    raise WorkflowInvariantError(f"cannot route from status {status.value!r}")


def route_after_validation(state: WorkflowState) -> RouteAfterValidation:
    """Route after validate_results; pure, no I/O and no state mutation."""
    status = _workflow_status(state)
    if status is WorkflowStatus.FAILED:
        return "finalize_failure"
    if status is WorkflowStatus.VALIDATING:
        return "export_results"
    raise WorkflowInvariantError(f"cannot route from status {status.value!r}")


def route_after_export(state: WorkflowState) -> RouteAfterExport:
    """Route after export_results; pure, no I/O and no state mutation."""
    status = _workflow_status(state)
    if status is WorkflowStatus.FAILED:
        return "finalize_failure"
    if status is WorkflowStatus.EXPORTING:
        return "finalize_success"
    raise WorkflowInvariantError(f"cannot route from status {status.value!r}")
