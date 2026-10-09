"""Agent tool: get_workflow_status (S3.5-M3)."""

from pydantic import BaseModel, ConfigDict, Field

from materials_screening.agent.context import AgentToolContext
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent_tools.common import resolve_owned_thread


class GetWorkflowStatusInput(BaseModel):
    """None uses the active workflow thread of the conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None


class GetWorkflowStatusOutput(BaseModel):
    """Safe status summary; never the full workflow state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str
    status: str
    current_node: str | None = None
    retrieved_count: int = 0
    filtered_count: int = 0
    returned_count: int = 0
    validation_passed: bool | None = None
    clarification_question: str | None = None
    error_code: str | None = None
    exports: list[str] = Field(default_factory=list)
    evidence_id: str = Field(min_length=1)


class GetWorkflowStatusTool:
    """Read only the safe status summary of one owned workflow thread."""

    name = "get_workflow_status"
    description = "Read the safe status summary of the screening workflow thread."
    input_model = GetWorkflowStatusInput
    output_model = GetWorkflowStatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: GetWorkflowStatusInput,
        context: AgentToolContext,
    ) -> GetWorkflowStatusOutput:
        thread_id, error_code = resolve_owned_thread(
            requested_thread_id=arguments.thread_id,
            context=context,
        )
        evidence_id = context.id_generator.new_id()
        if error_code is not None or thread_id is None:
            output = GetWorkflowStatusOutput(
                thread_id=thread_id or "",
                status="error",
                error_code=error_code,
                evidence_id=evidence_id,
            )
        else:
            view = context.workflow_runner.get_state(thread_id)
            if view.status == "":
                output = GetWorkflowStatusOutput(
                    thread_id=thread_id,
                    status="error",
                    error_code="THREAD_NOT_FOUND",
                    evidence_id=evidence_id,
                )
            else:
                output = GetWorkflowStatusOutput(
                    thread_id=thread_id,
                    status=view.status,
                    current_node=view.current_node,
                    retrieved_count=view.retrieved_count,
                    filtered_count=view.filtered_count,
                    returned_count=view.returned_count,
                    validation_passed=view.validation_passed,
                    clarification_question=None,
                    error_code=None,
                    exports=list(view.exports),
                    evidence_id=evidence_id,
                )
        self._record(context, output)
        return output

    def _record(
        self,
        context: AgentToolContext,
        output: GetWorkflowStatusOutput,
    ) -> None:
        context.ledger.record(
            call_id=context.call_id,
            tool_name=self.name,
            evidence_id=output.evidence_id,
            result_json=output.model_dump_json(),
            side_effect=ToolSideEffect.READ_ONLY,
        )
