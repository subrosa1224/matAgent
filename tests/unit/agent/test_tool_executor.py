"""Unit tests for the deterministic tool executor (S3.5-M4 prep)."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from tests.unit.agent_tools.helpers import FakeRunner, context, state_view

from materials_screening.agent.errors import (
    AgentInvariantError,
    AgentToolError,
    WorkflowResultReadError,
)
from materials_screening.agent.models import AgentToolCall, ToolResultStatus
from materials_screening.agent.policy import AgentToolPolicy
from materials_screening.agent.state import MaterialAgentState
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent.tool_executor import (
    ToolExecutionOutcome,
    ToolExecutor,
)
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools.compare_ranked_materials import (
    CompareRankedMaterialsTool,
)
from materials_screening.agent_tools.get_workflow_history import (
    GetWorkflowHistoryTool,
)
from materials_screening.agent_tools.get_workflow_status import (
    GetWorkflowStatusTool,
)
from materials_screening.agent_tools.run_screening_workflow import (
    RunScreeningWorkflowTool,
)
from materials_screening.errors import RepositoryError
from materials_screening.workflow.input_output import WorkflowInput, WorkflowOutput
from materials_screening.workflow.state import WorkflowStatus


class _StatusInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None


class _StatusOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str


class _FailingTool:
    """Whitelisted read-only tool that raises an expected tool error."""

    name = "get_workflow_status"
    description = "failing status tool"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        raise WorkflowResultReadError("ARTIFACT_INTEGRITY", "corrupted artifact")


class _BoomTool:
    """Whitelisted tool raising a plain AgentToolError (no code)."""

    name = "get_workflow_status"
    description = "boom status tool"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        raise AgentToolError("boom")


class _RuntimeBoomTool:
    """Whitelisted tool raising an unknown programming error."""

    name = "get_workflow_status"
    description = "runtime boom status tool"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        raise RuntimeError("unexpected")


class _RepositoryFailureTool:
    name = "get_workflow_status"
    description = "repository failure"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        raise RepositoryError("raw SSL details must not reach the user")


class _NoRecordTool:
    """Whitelisted tool that forgets to register evidence."""

    name = "get_workflow_status"
    description = "silent status tool"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        return _StatusOutput(status="completed")


class _EmptyEvidenceTool:
    """Whitelisted tool that registers an empty evidence id."""

    name = "get_workflow_status"
    description = "empty evidence status tool"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        tool_context.ledger.record(
            call_id=tool_context.call_id,
            tool_name=self.name,
            evidence_id="",
            result_json='{"status":"completed"}',
            side_effect=self.side_effect,
        )
        return _StatusOutput(status="completed")


class _CorruptLedgerTool:
    """Whitelisted tool that records a non-JSON result into the ledger."""

    name = "get_workflow_status"
    description = "corrupt ledger status tool"
    input_model = _StatusInput
    output_model = _StatusOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        tool_context.ledger.record(
            call_id=tool_context.call_id,
            tool_name=self.name,
            evidence_id="evt_1",
            result_json="not json",
            side_effect=self.side_effect,
        )
        return _StatusOutput(status="completed")


class _BigOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    payload: str


class _OversizedTool:
    """Whitelisted tool returning an output that exceeds the size limit."""

    name = "get_workflow_status"
    description = "oversized status tool"
    input_model = _StatusInput
    output_model = _BigOutput
    side_effect = ToolSideEffect.READ_ONLY

    def execute(
        self,
        arguments: BaseModel,
        tool_context: Any,
    ) -> BaseModel:
        evidence_id = tool_context.id_generator.new_id()
        output = _BigOutput(evidence_id=evidence_id, payload="x" * 2000)
        tool_context.ledger.record(
            call_id=tool_context.call_id,
            tool_name=self.name,
            evidence_id=evidence_id,
            result_json=output.model_dump_json(),
            side_effect=self.side_effect,
        )
        return output


class _RunFakeRunner:
    """Runner fake exposing only ``run`` for the workflow tool."""

    def __init__(self, output: WorkflowOutput) -> None:
        self._output = output
        self.calls: list[WorkflowInput] = []

    def run(
        self,
        workflow_input: WorkflowInput,
        *args: Any,
        **kwargs: Any,
    ) -> WorkflowOutput:
        self.calls.append(workflow_input)
        return self._output


def _workflow_output() -> WorkflowOutput:
    return WorkflowOutput.model_validate(
        {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "status": WorkflowStatus.COMPLETED,
            "planner_status": "ready",
            "request": None,
            "retrieved_count": 3,
            "filtered_count": 2,
            "returned_count": 1,
            "validation_passed": True,
            "exports": (),
            "warnings": (),
            "error": None,
            "clarification_question": None,
        }
    )


def _state(*, active_thread: str | None = "thread-1") -> MaterialAgentState:
    return {
        "conversation_id": "c1",
        "user_turn_id": "turn_1",
        "active_workflow_thread_id": active_thread,
        "tool_call_count": 0,
        "workflow_run_count": 0,
    }


def _status_call(call_id: str, arguments: str = "{}") -> AgentToolCall:
    return AgentToolCall(
        call_id=call_id,
        name="get_workflow_status",
        arguments_json=arguments,
    )


def _execute_one(
    executor: ToolExecutor,
    tool_context: Any,
    call: AgentToolCall,
    *,
    state: MaterialAgentState | None = None,
) -> ToolExecutionOutcome:
    outcomes = executor.execute(
        [call],
        context=tool_context,
        state=state if state is not None else _state(),
    )
    assert len(outcomes) == 1
    return outcomes[0]


class TestToolExecutor:
    def test_valid_call_returns_ok_envelope(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(executor, tool_context, _status_call("call_1"))

        assert outcome.status is ToolResultStatus.OK
        assert outcome.envelope.error is None
        assert outcome.envelope.output is not None
        assert outcome.envelope.output["status"] == "completed"
        assert outcome.envelope.output["thread_id"] == "thread-1"
        assert outcome.evidence_id == outcome.envelope.evidence_id
        assert outcome.evidence_id
        assert outcome.output_json == outcome.envelope.to_json()
        assert tool_context.ledger.has("call_1")
        assert runner.get_state_calls == ["thread-1"]

    def test_output_json_maps_to_function_output_item(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(executor, tool_context, _status_call("call_1"))

        payload = json.loads(outcome.output_json)
        assert payload["call_id"] == "call_1"
        assert payload["tool_name"] == "get_workflow_status"
        assert payload["status"] == "ok"
        assert payload["output"]["status"] == "completed"

    def test_invalid_json_returns_error_envelope(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(
            executor, tool_context, _status_call("call_1", arguments="not json")
        )

        assert outcome.status is ToolResultStatus.ERROR
        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "INVALID_ARGUMENTS"
        assert outcome.envelope.error.retryable is False
        assert '"not json"' not in outcome.output_json
        assert runner.get_state_calls == []

    def test_non_object_json_rejected(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(executor, tool_context, _status_call("call_1", "[]"))

        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "INVALID_ARGUMENTS"
        assert runner.get_state_calls == []

    def test_extra_fields_rejected(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(
            executor,
            tool_context,
            _status_call("call_1", '{"thread_id": "thread-1", "bogus": 1}'),
        )

        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "INVALID_ARGUMENTS"
        assert runner.get_state_calls == []

    def test_arguments_too_large(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
            max_argument_bytes=8192,
        )
        huge = '{"thread_id": "' + "a" * 9000 + '"}'
        outcome = _execute_one(executor, tool_context, _status_call("call_1", huge))

        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "ARGUMENT_TOO_LARGE"
        assert runner.get_state_calls == []

    def test_unknown_tool_returns_error_envelope(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        call = AgentToolCall(
            call_id="call_1",
            name="web_search",
            arguments_json="{}",
        )
        outcome = _execute_one(executor, tool_context, call)

        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "UNKNOWN_TOOL"
        assert runner.get_state_calls == []

    def test_duplicate_call_id_reuses_previous_result(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        calls = [_status_call("call_1"), _status_call("call_1")]
        outcomes = executor.execute(calls, context=tool_context, state=_state())

        assert len(outcomes) == 2
        assert outcomes[0].status is ToolResultStatus.OK
        assert outcomes[1].status is ToolResultStatus.OK
        assert outcomes[1].evidence_id == outcomes[0].evidence_id
        assert outcomes[1].output_json == outcomes[0].output_json
        assert runner.get_state_calls == ["thread-1"]

    def test_duplicate_call_id_with_different_tool_rejected(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry(
                [GetWorkflowStatusTool(), CompareRankedMaterialsTool()]
            ),
            policy=AgentToolPolicy(),
        )
        calls = [
            _status_call("call_1"),
            AgentToolCall(
                call_id="call_1",
                name="compare_ranked_materials",
                arguments_json='{"material_ids": ["mp-1", "mp-2"]}',
            ),
        ]
        outcomes = executor.execute(calls, context=tool_context, state=_state())

        assert outcomes[0].status is ToolResultStatus.OK
        assert outcomes[1].envelope.error is not None
        assert outcomes[1].envelope.error.code == "DUPLICATE_CALL_ID"
        assert runner.get_state_calls == ["thread-1"]

    def test_workflow_run_limit_rejects_second_side_effect(self) -> None:
        runner = _RunFakeRunner(_workflow_output())
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([RunScreeningWorkflowTool()]),
            policy=AgentToolPolicy(),
        )
        calls = [
            AgentToolCall(
                call_id="call_1",
                name="run_screening_workflow",
                arguments_json='{"query": "wide gap material"}',
            ),
            AgentToolCall(
                call_id="call_2",
                name="run_screening_workflow",
                arguments_json='{"query": "direct gap material"}',
            ),
        ]
        outcomes = executor.execute(calls, context=tool_context, state=_state())

        assert outcomes[0].status is ToolResultStatus.OK
        assert outcomes[0].envelope.output is not None
        assert outcomes[0].envelope.output["thread_id"] == "thread-1"
        assert outcomes[1].envelope.error is not None
        assert outcomes[1].envelope.error.code == "WORKFLOW_RUN_LIMIT"
        assert len(runner.calls) == 1

    def test_side_effect_limit_with_relaxed_run_limit(self) -> None:
        runner = _RunFakeRunner(_workflow_output())
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([RunScreeningWorkflowTool()]),
            policy=AgentToolPolicy(max_workflow_runs_per_turn=2),
        )
        calls = [
            AgentToolCall(
                call_id="call_1",
                name="run_screening_workflow",
                arguments_json='{"query": "wide gap material"}',
            ),
            AgentToolCall(
                call_id="call_2",
                name="run_screening_workflow",
                arguments_json='{"query": "direct gap material"}',
            ),
        ]
        outcomes = executor.execute(calls, context=tool_context, state=_state())

        assert outcomes[0].status is ToolResultStatus.OK
        assert outcomes[1].envelope.error is not None
        assert outcomes[1].envelope.error.code == "SIDE_EFFECT_LIMIT"
        assert len(runner.calls) == 1

    def test_tool_failure_wrapped_in_error_envelope(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_FailingTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(executor, tool_context, _status_call("call_1"))

        assert outcome.status is ToolResultStatus.ERROR
        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "ARTIFACT_INTEGRITY"
        assert outcome.envelope.error.retryable is False
        assert "corrupted artifact" in outcome.envelope.error.message

    def test_tool_failure_without_code_uses_fallback(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_BoomTool()]),
            policy=AgentToolPolicy(),
        )
        outcome = _execute_one(executor, tool_context, _status_call("call_1"))

        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "TOOL_ERROR"
        assert outcome.envelope.error.message == "boom"

    def test_unknown_programming_error_propagates(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_RuntimeBoomTool()]),
            policy=AgentToolPolicy(),
        )
        with pytest.raises(RuntimeError, match="unexpected"):
            _execute_one(executor, tool_context, _status_call("call_1"))

    def test_repository_failure_becomes_safe_retryable_error(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_RepositoryFailureTool()]),
            policy=AgentToolPolicy(),
        )

        outcome = _execute_one(executor, tool_context, _status_call("call_1"))

        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "REPOSITORY_CONNECTION"
        assert outcome.envelope.error.retryable is True
        assert outcome.envelope.error.message == (
            "Materials Project network connection failed; please retry"
        )

    def test_oversized_output_returns_error_envelope(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_OversizedTool()]),
            policy=AgentToolPolicy(),
            max_output_bytes=256,
        )
        outcome = _execute_one(executor, tool_context, _status_call("call_1"))

        assert outcome.status is ToolResultStatus.ERROR
        assert outcome.envelope.error is not None
        assert outcome.envelope.error.code == "TOOL_OUTPUT_TOO_LARGE"
        assert outcome.envelope.output is None
        assert outcome.evidence_id == tool_context.ledger.evidence_ids()[0]
        assert "xxxx" not in outcome.output_json

    def test_output_size_limit_is_configurable(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        small = ToolExecutor(
            registry=AgentToolRegistry([_OversizedTool()]),
            policy=AgentToolPolicy(),
            max_output_bytes=64,
        )
        large = ToolExecutor(
            registry=AgentToolRegistry([_OversizedTool()]),
            policy=AgentToolPolicy(),
            max_output_bytes=65536,
        )

        small_outcome = _execute_one(small, tool_context, _status_call("call_1"))
        large_outcome = _execute_one(large, tool_context, _status_call("call_2"))

        assert small_outcome.envelope.error is not None
        assert small_outcome.envelope.error.code == "TOOL_OUTPUT_TOO_LARGE"
        assert large_outcome.status is ToolResultStatus.OK

    def test_multiple_calls_executed_in_order(self) -> None:
        runner = FakeRunner(
            view=state_view(status="completed"),
            history=(),
        )
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry(
                [GetWorkflowStatusTool(), GetWorkflowHistoryTool()]
            ),
            policy=AgentToolPolicy(),
        )
        calls = [
            AgentToolCall(
                call_id="call_1",
                name="get_workflow_status",
                arguments_json="{}",
            ),
            AgentToolCall(
                call_id="call_2",
                name="get_workflow_history",
                arguments_json='{"limit": 5}',
            ),
        ]
        outcomes = executor.execute(calls, context=tool_context, state=_state())

        assert [outcome.call_id for outcome in outcomes] == ["call_1", "call_2"]
        assert tool_context.ledger.executed_call_ids() == ("call_1", "call_2")
        assert tool_context.ledger.executed_tool_names() == (
            "get_workflow_status",
            "get_workflow_history",
        )
        for call, outcome in zip(calls, outcomes, strict=True):
            assert outcome.call_id == call.call_id
            payload = json.loads(outcome.output_json)
            assert payload["call_id"] == call.call_id
            assert payload["status"] == "ok"

    def test_tool_call_limit_enforced_sequentially(self) -> None:
        runner = FakeRunner(view=state_view(status="completed"))
        tool_context = context(runner=runner)
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(max_tool_calls_per_turn=4),
        )
        calls = [_status_call(f"call_{index}") for index in range(1, 6)]
        outcomes = executor.execute(calls, context=tool_context, state=_state())

        assert [outcome.status for outcome in outcomes[:4]] == [
            ToolResultStatus.OK,
            ToolResultStatus.OK,
            ToolResultStatus.OK,
            ToolResultStatus.OK,
        ]
        assert outcomes[4].envelope.error is not None
        assert outcomes[4].envelope.error.code == "TOOL_CALL_LIMIT"
        assert len(runner.get_state_calls) == 4

    def test_missing_evidence_is_invariant_error(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_NoRecordTool()]),
            policy=AgentToolPolicy(),
        )
        with pytest.raises(AgentInvariantError, match="did not register evidence"):
            _execute_one(executor, tool_context, _status_call("call_1"))

    def test_empty_evidence_id_is_invariant_error(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_EmptyEvidenceTool()]),
            policy=AgentToolPolicy(),
        )
        with pytest.raises(AgentInvariantError, match="empty evidence id"):
            _execute_one(executor, tool_context, _status_call("call_1"))

    def test_corrupt_stored_result_is_invariant_error(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([_CorruptLedgerTool()]),
            policy=AgentToolPolicy(),
        )
        calls = [_status_call("call_1"), _status_call("call_1")]
        with pytest.raises(AgentInvariantError, match="not valid JSON"):
            executor.execute(calls, context=tool_context, state=_state())

    def test_empty_batch_returns_empty_tuple(self) -> None:
        tool_context = context(runner=FakeRunner(view=state_view()))
        executor = ToolExecutor(
            registry=AgentToolRegistry([GetWorkflowStatusTool()]),
            policy=AgentToolPolicy(),
        )
        outcomes = executor.execute([], context=tool_context, state=_state())
        assert outcomes == ()
