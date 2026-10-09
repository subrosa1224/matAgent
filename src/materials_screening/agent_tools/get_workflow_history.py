"""Agent tool: get_workflow_history (S3.5-M3)."""

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent_tools.common import resolve_owned_thread


class GetWorkflowHistoryInput(BaseModel):
    """Read safe checkpoint summaries; never checkpoint values or config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None
    limit: int = Field(default=10, ge=1, le=20)


class WorkflowHistoryStep(BaseModel):
    """One safe history step; no checkpoint id, values or config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int | None = None
    node: str | None = None
    status: str | None = None
    created_at: str | None = None


class GetWorkflowHistoryOutput(BaseModel):
    """Ordered safe checkpoint summaries for one thread."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    steps: list[WorkflowHistoryStep]
    error_code: str | None = None
    evidence_id: str = Field(min_length=1)


class GetWorkflowHistoryTool:
    """Read only safe node-level history steps of an owned thread."""

    name = "get_workflow_history"
    description = "Read the safe node-level history of the screening workflow thread."
    input_model = GetWorkflowHistoryInput
    output_model = GetWorkflowHistoryOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: GetWorkflowHistoryInput,
        context: AgentToolContext,
    ) -> GetWorkflowHistoryOutput:
        thread_id, error_code = resolve_owned_thread(
            requested_thread_id=arguments.thread_id,
            context=context,
        )
        evidence_id = context.id_generator.new_id()
        if error_code is not None or thread_id is None:
            output = GetWorkflowHistoryOutput(
                thread_id=thread_id or "",
                steps=[],
                error_code=error_code,
                evidence_id=evidence_id,
            )
        else:
            views = context.workflow_runner.get_history(
                thread_id,
                limit=arguments.limit,
            )
            if not views:
                output = GetWorkflowHistoryOutput(
                    thread_id=thread_id,
                    steps=[],
                    error_code="THREAD_NOT_FOUND",
                    evidence_id=evidence_id,
                )
            else:
                output = GetWorkflowHistoryOutput(
                    thread_id=thread_id,
                    steps=[
                        WorkflowHistoryStep(
                            step=view.step,
                            node=view.current_node,
                            status=view.status,
                            created_at=view.created_at,
                        )
                        for view in views
                    ],
                    error_code=None,
                    evidence_id=evidence_id,
                )
        self._record(context, output)
        return output

    def _record(
        self,
        context: AgentToolContext,
        output: GetWorkflowHistoryOutput,
    ) -> None:
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=output.evidence_id,
            result_json=output.model_dump_json(),
            side_effect=ToolSideEffect.READ_ONLY,
        )
