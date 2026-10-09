"""Shared helpers for agent tools (S3.5-M3)."""

from materials_screening.agent.context import AgentToolContext


def resolve_owned_thread(
    *,
    requested_thread_id: str | None,
    context: AgentToolContext,
) -> tuple[str | None, str | None]:
    """Resolve thread_id and verify conversation ownership.

    Returns ``(thread_id, error_code)``; ``error_code`` is None on success.
    """
    if requested_thread_id is not None:
        thread_id = requested_thread_id
    elif context.active_workflow_thread_id is not None:
        thread_id = context.active_workflow_thread_id
    else:
        return None, "NO_ACTIVE_WORKFLOW"
    allowed = {
        link.thread_id
        for link in context.conversation_links
        if link.conversation_id == context.conversation_id
    }
    if context.active_workflow_thread_id is not None:
        allowed.add(context.active_workflow_thread_id)
    if thread_id not in allowed:
        return thread_id, "OWNERSHIP_DENIED"
    return thread_id, None
