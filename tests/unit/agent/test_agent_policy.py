"""Unit tests for the agent tool policy (S3.5-M2)."""

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from materials_screening.agent.errors import AgentToolError
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.policy import (
    AgentPolicyError,
    AgentToolPolicy,
    ConversationWorkflowLink,
)
from materials_screening.agent.tool_base import ToolSideEffect


class _ReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    thread_id: str | None = None


class _RunInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str


def _make_tool(
    name: str,
    *,
    input_model: type[BaseModel],
    side_effect: ToolSideEffect,
) -> Any:
    input_cls = input_model
    effect = side_effect

    class _Tool:
        description = f"{name} tool"
        input_model = input_cls
        output_model = _Output
        side_effect = effect

        def __init__(self) -> None:
            self.name = name

        def execute(
            self,
            arguments: BaseModel,
            context: object,
        ) -> BaseModel:
            raise AssertionError("policy must not execute tools")

    return _Tool()


_READ_TOOL = _make_tool(
    "get_workflow_status",
    input_model=_ReadInput,
    side_effect=ToolSideEffect.READ_ONLY,
)
_RUN_TOOL = _make_tool(
    "run_screening_workflow",
    input_model=_RunInput,
    side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
)


def _policy(**kwargs: Any) -> AgentToolPolicy:
    return AgentToolPolicy(**kwargs)


def _state(active_thread: str | None = "thread-1") -> dict[str, Any]:
    return {"active_workflow_thread_id": active_thread}


def _links() -> list[ConversationWorkflowLink]:
    return [
        ConversationWorkflowLink(
            conversation_id="c1",
            thread_id="thread-linked",
            created_at="now",
        ),
        ConversationWorkflowLink(
            conversation_id="other",
            thread_id="thread-other",
            created_at="now",
        ),
    ]


def _read_arguments(thread_id: str | None = None) -> _ReadInput:
    return _ReadInput(thread_id=thread_id)


def _run_arguments(query: str = "q") -> _RunInput:
    return _RunInput(query=query)


class TestAgentToolPolicy:
    def test_read_call_without_thread_allowed(self) -> None:
        _policy().validate_call(
            tool=_READ_TOOL,
            arguments=_read_arguments(),
            call_id="call_1",
            conversation_id="c1",
            state=_state(),
            links=_links(),
            ledger=ToolExecutionLedger(),
        )

    def test_read_call_active_thread_allowed(self) -> None:
        _policy().validate_call(
            tool=_READ_TOOL,
            arguments=_read_arguments("thread-1"),
            call_id="call_1",
            conversation_id="c1",
            state=_state("thread-1"),
            links=_links(),
            ledger=ToolExecutionLedger(),
        )

    def test_read_call_linked_thread_allowed(self) -> None:
        _policy().validate_call(
            tool=_READ_TOOL,
            arguments=_read_arguments("thread-linked"),
            call_id="call_1",
            conversation_id="c1",
            state=_state(),
            links=_links(),
            ledger=ToolExecutionLedger(),
        )

    def test_read_call_other_conversation_thread_denied(self) -> None:
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=_READ_TOOL,
                arguments=_read_arguments("thread-other"),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "OWNERSHIP_DENIED"

    def test_read_call_unknown_thread_denied(self) -> None:
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=_READ_TOOL,
                arguments=_read_arguments("thread-nobody"),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "OWNERSHIP_DENIED"

    @pytest.mark.parametrize("bad_thread", ["", "../evil", "a/b", "x" * 255])
    def test_invalid_thread_format_rejected(self, bad_thread: str) -> None:
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=_READ_TOOL,
                arguments=_read_arguments(bad_thread),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "INVALID_THREAD_ID"

    def test_tool_call_limit(self) -> None:
        ledger = ToolExecutionLedger()
        for index in range(4):
            ledger.record(
                call_id=f"c{index}",
                tool_name="get_workflow_status",
                evidence_id=f"e{index}",
                result_json="{}",
                side_effect=ToolSideEffect.READ_ONLY,
            )
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy(max_tool_calls_per_turn=4).validate_call(
                tool=_READ_TOOL,
                arguments=_read_arguments(),
                call_id="call_new",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ledger,
            )
        assert excinfo.value.code == "TOOL_CALL_LIMIT"

    def test_tool_call_within_limit_allowed(self) -> None:
        ledger = ToolExecutionLedger()
        for index in range(3):
            ledger.record(
                call_id=f"c{index}",
                tool_name="get_workflow_status",
                evidence_id=f"e{index}",
                result_json="{}",
                side_effect=ToolSideEffect.READ_ONLY,
            )
        _policy(max_tool_calls_per_turn=4).validate_call(
            tool=_READ_TOOL,
            arguments=_read_arguments(),
            call_id="call_new",
            conversation_id="c1",
            state=_state(),
            links=_links(),
            ledger=ledger,
        )

    def test_workflow_run_limit(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="c1",
            tool_name="run_screening_workflow",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy(max_workflow_runs_per_turn=1).validate_call(
                tool=_RUN_TOOL,
                arguments=_run_arguments(),
                call_id="call_new",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ledger,
            )
        assert excinfo.value.code == "WORKFLOW_RUN_LIMIT"

    def test_second_side_effect_call_rejected(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="c1",
            tool_name="run_screening_workflow",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy(max_workflow_runs_per_turn=2).validate_call(
                tool=_RUN_TOOL,
                arguments=_run_arguments(),
                call_id="call_new",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ledger,
            )
        assert excinfo.value.code == "SIDE_EFFECT_LIMIT"

    def test_duplicate_call_id_rejected(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_1",
            tool_name="get_workflow_status",
            evidence_id="e1",
            result_json="{}",
            side_effect=ToolSideEffect.READ_ONLY,
        )
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=_READ_TOOL,
                arguments=_read_arguments(),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ledger,
            )
        assert excinfo.value.code == "DUPLICATE_CALL_ID"

    def test_empty_call_id_rejected(self) -> None:
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=_READ_TOOL,
                arguments=_read_arguments(),
                call_id="",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "INVALID_CALL_ID"

    def test_argument_too_large(self) -> None:
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy(max_argument_bytes=8).validate_call(
                tool=_RUN_TOOL,
                arguments=_run_arguments("x" * 100),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "ARGUMENT_TOO_LARGE"

    def test_web_search_rejected(self) -> None:
        web_tool = _make_tool(
            "web_search",
            input_model=_ReadInput,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=web_tool,
                arguments=_read_arguments(),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "FORBIDDEN_OPERATION"

    def test_unknown_tool_rejected(self) -> None:
        unknown = _make_tool(
            "delete_run",
            input_model=_ReadInput,
            side_effect=ToolSideEffect.READ_ONLY,
        )
        with pytest.raises(AgentPolicyError) as excinfo:
            _policy().validate_call(
                tool=unknown,
                arguments=_read_arguments(),
                call_id="call_1",
                conversation_id="c1",
                state=_state(),
                links=_links(),
                ledger=ToolExecutionLedger(),
            )
        assert excinfo.value.code == "UNKNOWN_TOOL"

    def test_policy_error_is_agent_tool_error(self) -> None:
        assert issubclass(AgentPolicyError, AgentToolError)

    def test_read_without_active_thread_allowed(self) -> None:
        _policy().validate_call(
            tool=_READ_TOOL,
            arguments=_read_arguments(),
            call_id="call_1",
            conversation_id="c1",
            state=_state(None),
            links=_links(),
            ledger=ToolExecutionLedger(),
        )
