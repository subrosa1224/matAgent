"""Unit tests for the get_workflow_history tool (S3.5-M3)."""

import pytest
from pydantic import ValidationError

from materials_screening.agent_tools.get_workflow_history import (
    GetWorkflowHistoryInput,
    GetWorkflowHistoryTool,
)
from tests.unit.agent_tools.helpers import (
    FakeRunner,
    checkpoint_view,
    context,
)


class TestGetWorkflowHistoryTool:
    def test_success_maps_steps_and_passes_limit(self) -> None:
        runner = FakeRunner(
            history=(
                checkpoint_view(step=2, node="validate_results", status="validating"),
                checkpoint_view(step=1, node="filter_materials", status="filtering"),
            )
        )
        tool_context = context(runner=runner)
        output = GetWorkflowHistoryTool().execute(
            GetWorkflowHistoryInput(thread_id=None, limit=20),
            tool_context,
        )
        assert output.error_code is None
        assert [step.step for step in output.steps] == [2, 1]
        assert [step.node for step in output.steps] == [
            "validate_results",
            "filter_materials",
        ]
        assert [step.status for step in output.steps] == [
            "validating",
            "filtering",
        ]
        assert runner.get_history_calls == [("thread-1", 20)]

    @pytest.mark.parametrize("limit", [0, 21])
    def test_limit_out_of_bounds_rejected(self, limit: int) -> None:
        with pytest.raises(ValidationError):
            GetWorkflowHistoryInput(thread_id=None, limit=limit)

    @pytest.mark.parametrize("limit", [1, 20])
    def test_limit_bounds_accepted(self, limit: int) -> None:
        assert GetWorkflowHistoryInput(limit=limit).limit == limit

    def test_no_active_thread(self) -> None:
        tool_context = context(runner=FakeRunner(), active_thread=None)
        output = GetWorkflowHistoryTool().execute(
            GetWorkflowHistoryInput(thread_id=None),
            tool_context,
        )
        assert output.error_code == "NO_ACTIVE_WORKFLOW"
        assert output.steps == []

    def test_ownership_denied(self) -> None:
        tool_context = context(runner=FakeRunner())
        output = GetWorkflowHistoryTool().execute(
            GetWorkflowHistoryInput(thread_id="thread-other"),
            tool_context,
        )
        assert output.error_code == "OWNERSHIP_DENIED"

    def test_unknown_thread_empty_history(self) -> None:
        tool_context = context(runner=FakeRunner(history=()))
        output = GetWorkflowHistoryTool().execute(
            GetWorkflowHistoryInput(thread_id="thread-1"),
            tool_context,
        )
        assert output.error_code == "THREAD_NOT_FOUND"
        assert output.steps == []

    def test_steps_do_not_expose_checkpoint_data(self) -> None:
        runner = FakeRunner(history=(checkpoint_view(checkpoint_id="secret-cp"),))
        output = GetWorkflowHistoryTool().execute(
            GetWorkflowHistoryInput(thread_id=None),
            context(runner=runner),
        )
        serialized = output.model_dump_json()
        assert "secret-cp" not in serialized
        assert "checkpoint_id" not in serialized
        assert "next_nodes" not in serialized

    def test_evidence_recorded(self) -> None:
        runner = FakeRunner(history=(checkpoint_view(),))
        tool_context = context(runner=runner)
        output = GetWorkflowHistoryTool().execute(
            GetWorkflowHistoryInput(thread_id=None),
            tool_context,
        )
        assert tool_context.ledger.evidence_ids() == (output.evidence_id,)
