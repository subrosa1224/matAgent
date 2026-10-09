"""Unit tests for the get_workflow_status tool (S3.5-M3)."""

import pytest

from materials_screening.agent_tools.get_workflow_status import (
    GetWorkflowStatusInput,
    GetWorkflowStatusTool,
)
from tests.unit.agent_tools.helpers import (
    FakeRunner,
    context,
    state_view,
)


class TestGetWorkflowStatusTool:
    def test_success_uses_active_thread(self) -> None:
        runner = FakeRunner(
            view=state_view(
                status="completed",
                retrieved=10,
                filtered=5,
                returned=3,
                validation=True,
                exports=("run-1/exports/request.json",),
                current_node="finalize_success",
            )
        )
        tool_context = context(runner=runner)
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id=None),
            tool_context,
        )
        assert output.thread_id == "thread-1"
        assert output.status == "completed"
        assert output.current_node == "finalize_success"
        assert output.retrieved_count == 10
        assert output.filtered_count == 5
        assert output.returned_count == 3
        assert output.validation_passed is True
        assert output.exports == ["run-1/exports/request.json"]
        assert output.error_code is None
        assert output.evidence_id == "evt_1"
        assert runner.get_state_calls == ["thread-1"]

    def test_explicit_linked_thread(self) -> None:
        runner = FakeRunner(view=state_view())
        tool_context = context(runner=runner)
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id="thread-linked"),
            tool_context,
        )
        assert output.thread_id == "thread-linked"
        assert output.error_code is None

    def test_no_active_thread(self) -> None:
        runner = FakeRunner(view=state_view())
        tool_context = context(runner=runner, active_thread=None)
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id=None),
            tool_context,
        )
        assert output.status == "error"
        assert output.error_code == "NO_ACTIVE_WORKFLOW"
        assert output.thread_id == ""

    def test_other_conversation_thread_denied(self) -> None:
        runner = FakeRunner(view=state_view())
        tool_context = context(runner=runner)
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id="thread-other"),
            tool_context,
        )
        assert output.error_code == "OWNERSHIP_DENIED"

    def test_unknown_thread(self) -> None:
        runner = FakeRunner(view=state_view(status=""))
        tool_context = context(runner=runner)
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id="thread-1"),
            tool_context,
        )
        assert output.error_code == "THREAD_NOT_FOUND"

    def test_evidence_recorded_in_ledger(self) -> None:
        runner = FakeRunner(view=state_view())
        tool_context = context(runner=runner)
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id=None),
            tool_context,
        )
        assert tool_context.ledger.evidence_ids() == (output.evidence_id,)
        assert tool_context.ledger.executed_tool_names() == ("get_workflow_status",)

    def test_output_exposes_only_safe_fields(self) -> None:
        runner = FakeRunner(view=state_view())
        output = GetWorkflowStatusTool().execute(
            GetWorkflowStatusInput(thread_id=None),
            context(runner=runner),
        )
        allowed = {
            "thread_id",
            "status",
            "current_node",
            "retrieved_count",
            "filtered_count",
            "returned_count",
            "validation_passed",
            "clarification_question",
            "error_code",
            "exports",
            "evidence_id",
        }
        assert set(output.model_dump()) == allowed
        assert "checkpoint" not in output.model_dump_json()
        assert "artifact" not in output.model_dump_json()

    def test_unknown_bug_propagates(self) -> None:
        class _BoomRunner:
            def get_state(self, thread_id: str) -> object:
                raise RuntimeError("unexpected")

        with pytest.raises(RuntimeError, match="unexpected"):
            GetWorkflowStatusTool().execute(
                GetWorkflowStatusInput(thread_id="thread-1"),
                context(runner=_BoomRunner()),
            )
