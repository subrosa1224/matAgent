"""Stage 3 workflow event model and builder (S3-M2/M4)."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from materials_screening.workflow.state import WorkflowState


class WorkflowEvent(BaseModel):
    """One audit event emitted by a workflow node.

    ``created_at`` is an ISO-8601 string so the event stays JSON-native.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    run_id: str
    node: str
    event_type: str
    created_at: str
    status: str
    message: str
    metrics: dict[str, int | float | str | bool | None]


def emit_event(
    *,
    state: WorkflowState,
    run_id: str,
    node: str,
    event_type: str,
    status: str,
    message: str,
    metrics: dict[str, int | float | str | bool | None],
    created_at: str,
) -> list[dict[str, Any]]:
    """Build one event payload, deduplicated by ``(run_id, node)``.

    Nodes return only newly added events; re-executing a node after a
    checkpoint must not append the same event twice.
    """
    event = WorkflowEvent(
        event_id=f"{run_id}:{node}",
        run_id=run_id,
        node=node,
        event_type=event_type,
        created_at=created_at,
        status=status,
        message=message,
        metrics=metrics,
    ).model_dump(mode="json")
    existing_ids = {item["event_id"] for item in state.get("events", [])}
    if event["event_id"] in existing_ids:
        return []
    return [event]
