"""Unit tests for Stage 3 workflow input/output models (S3-M2)."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from materials_screening.workflow.input_output import (
    WorkflowGraphInput,
    WorkflowInput,
    WorkflowOutput,
)
from materials_screening.workflow.state import WorkflowStatus


class TestWorkflowInput:
    def test_query_only(self) -> None:
        workflow_input = WorkflowInput(query="find materials")
        assert workflow_input.query == "find materials"
        assert workflow_input.request is None
        assert workflow_input.output_root == "data/workflow_runs"
        assert workflow_input.export_cif is True

    def test_request_only(self) -> None:
        request = {"limit": 10}
        workflow_input = WorkflowInput(request=request, export_cif=False)
        assert workflow_input.request == request
        assert workflow_input.query is None
        assert workflow_input.export_cif is False

    def test_neither_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowInput()

    def test_both_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowInput(query="find materials", request={"limit": 10})

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowInput(query="find materials", junk=1)

    def test_frozen(self) -> None:
        assert WorkflowInput.model_config.get("frozen") is True


class TestWorkflowGraphInput:
    def test_query_mode(self) -> None:
        graph_input = WorkflowGraphInput(
            run_id="run-1",
            workflow_version="workflow-v1",
            input_mode="query",
            user_query="find materials",
        )
        assert graph_input.run_id == "run-1"
        assert graph_input.user_query == "find materials"
        assert graph_input.raw_request is None

    def test_request_mode(self) -> None:
        request = {"limit": 10}
        graph_input = WorkflowGraphInput(
            run_id="run-1",
            workflow_version="workflow-v1",
            input_mode="request",
            raw_request=request,
        )
        assert graph_input.raw_request == request
        assert graph_input.user_query is None

    def test_mode_mismatch_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowGraphInput(
                run_id="run-1",
                workflow_version="workflow-v1",
                input_mode="query",
                raw_request={"limit": 10},
            )

    def test_both_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowGraphInput(
                run_id="run-1",
                workflow_version="workflow-v1",
                input_mode="query",
                user_query="find materials",
                raw_request={"limit": 10},
            )

    def test_neither_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowGraphInput(
                run_id="run-1",
                workflow_version="workflow-v1",
                input_mode="query",
            )

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowGraphInput(
                run_id="",
                workflow_version="workflow-v1",
                input_mode="query",
                user_query="find materials",
            )

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WorkflowGraphInput(
                run_id="run-1",
                workflow_version="workflow-v1",
                input_mode="query",
                user_query="find materials",
                junk=1,
            )

    def test_frozen(self) -> None:
        assert WorkflowGraphInput.model_config.get("frozen") is True

    def test_json_serializable(self) -> None:
        graph_input = WorkflowGraphInput(
            run_id="run-1",
            workflow_version="workflow-v1",
            input_mode="query",
            user_query="find materials",
        )
        dumped = graph_input.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped


def _output(status: WorkflowStatus) -> WorkflowOutput:
    return WorkflowOutput(
        run_id="run-1",
        thread_id="thread-1",
        status=status,
        planner_status="ready",
        request={"limit": 10},
        retrieved_count=5,
        filtered_count=3,
        returned_count=2,
        validation_passed=True,
        exports=("exports/request.json",),
        warnings=(),
        error=None,
        clarification_question=None,
    )


class TestWorkflowOutput:
    @pytest.mark.parametrize("status", list(WorkflowStatus))
    def test_accepts_all_statuses(self, status: WorkflowStatus) -> None:
        output = _output(status)
        assert output.status is status

    def test_json_serializable(self) -> None:
        dumped = _output(WorkflowStatus.COMPLETED).model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        assert dumped["status"] == "completed"

    def test_frozen(self) -> None:
        assert WorkflowOutput.model_config.get("frozen") is True

    def test_extra_field_rejected(self) -> None:
        data: dict[str, Any] = _output(WorkflowStatus.COMPLETED).model_dump()
        data["junk"] = 1
        with pytest.raises(ValidationError):
            WorkflowOutput.model_validate(data)

    def test_output_exposes_no_artifact_refs(self) -> None:
        assert not any(name.endswith("_ref") for name in WorkflowOutput.model_fields)

    def test_planner_stop_output(self) -> None:
        output = WorkflowOutput(
            run_id="run-1",
            thread_id="thread-1",
            status=WorkflowStatus.NEEDS_CLARIFICATION,
            planner_status="needs_clarification",
            request=None,
            retrieved_count=0,
            filtered_count=0,
            returned_count=0,
            validation_passed=None,
            exports=(),
            warnings=(),
            error=None,
            clarification_question="please clarify",
        )
        assert output.clarification_question == "please clarify"
        assert output.retrieved_count == 0
