"""Unit tests for stage 3.5 agent models (S3.5-M1)."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from materials_screening.agent.models import (
    AGENT_ANSWER_MAX_LENGTH,
    AgentErrorData,
    AgentEvent,
    AgentFinalDraft,
    AgentFinalStatus,
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
    AgentResult,
    AgentToolCall,
    ToolErrorData,
    ToolResultEnvelope,
    ToolResultStatus,
)


def _final_draft(**overrides: Any) -> AgentFinalDraft:
    values: dict[str, Any] = {
        "status": AgentFinalStatus.COMPLETED,
        "answer": "done",
        "active_workflow_thread_id": None,
        "referenced_material_ids": [],
        "evidence_ids": ["evt_1"],
        "warnings": [],
        "follow_up_question": None,
    }
    values.update(overrides)
    return AgentFinalDraft.model_validate(values)


class TestAgentFinalDraft:
    def test_completed_valid(self) -> None:
        draft = _final_draft()
        assert draft.status is AgentFinalStatus.COMPLETED
        assert draft.answer == "done"
        assert draft.evidence_ids == ["evt_1"]

    def test_needs_user_input_requires_follow_up(self) -> None:
        with pytest.raises(ValidationError):
            _final_draft(
                status=AgentFinalStatus.NEEDS_USER_INPUT,
                follow_up_question=None,
            )
        draft = _final_draft(
            status=AgentFinalStatus.NEEDS_USER_INPUT,
            follow_up_question="please clarify",
        )
        assert draft.follow_up_question == "please clarify"

    def test_error_status_valid_without_follow_up(self) -> None:
        draft = _final_draft(status=AgentFinalStatus.ERROR)
        assert draft.status is AgentFinalStatus.ERROR

    def test_empty_answer_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _final_draft(answer="")

    def test_answer_length_limit(self) -> None:
        with pytest.raises(ValidationError):
            _final_draft(answer="x" * (AGENT_ANSWER_MAX_LENGTH + 1))

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _final_draft(junk=1)

    def test_frozen(self) -> None:
        assert AgentFinalDraft.model_config.get("frozen") is True

    def test_json_serializable(self) -> None:
        dumped = _final_draft().model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped


class TestAgentResult:
    def test_valid(self) -> None:
        result = AgentResult(
            conversation_id="c1",
            user_turn_id="t1",
            status="completed",
            response_text="answer",
            selected_tools=("run_screening_workflow",),
            tool_call_count=1,
            model_call_count=2,
            evidence_ids=("evt_1",),
            warnings=(),
            error=None,
        )
        assert result.tool_call_count == 1

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult(
                conversation_id="c1",
                user_turn_id="t1",
                status="completed",
                response_text="answer",
                tool_call_count=-1,
                model_call_count=0,
            )

    def test_frozen_and_extra_rejected(self) -> None:
        assert AgentResult.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            AgentResult(
                conversation_id="c1",
                user_turn_id="t1",
                status="completed",
                response_text="answer",
                junk=1,
            )


class TestAgentToolCall:
    def test_valid(self) -> None:
        call = AgentToolCall(
            call_id="call_1",
            name="get_workflow_status",
            arguments_json='{"thread_id": null}',
        )
        assert call.name == "get_workflow_status"

    def test_empty_call_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentToolCall(
                call_id="",
                name="get_workflow_status",
                arguments_json="{}",
            )

    def test_frozen_and_extra_rejected(self) -> None:
        assert AgentToolCall.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            AgentToolCall(
                call_id="call_1",
                name="get_workflow_status",
                arguments_json="{}",
                junk=1,
            )


class TestToolResultEnvelope:
    def test_ok_requires_output(self) -> None:
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.OK,
            tool_name="get_workflow_status",
            call_id="call_1",
            evidence_id="evt_1",
            output={"status": "ok"},
        )
        assert envelope.status is ToolResultStatus.OK
        with pytest.raises(ValidationError):
            ToolResultEnvelope(
                status=ToolResultStatus.OK,
                tool_name="get_workflow_status",
                call_id="call_1",
                evidence_id="evt_1",
                output=None,
            )

    def test_error_requires_error_data(self) -> None:
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.ERROR,
            tool_name="get_workflow_status",
            call_id="call_1",
            evidence_id="evt_1",
            error=ToolErrorData(
                code="NO_ACTIVE_WORKFLOW",
                message="no active workflow",
                retryable=False,
            ),
        )
        assert envelope.error is not None
        with pytest.raises(ValidationError):
            ToolResultEnvelope(
                status=ToolResultStatus.ERROR,
                tool_name="get_workflow_status",
                call_id="call_1",
                evidence_id="evt_1",
                error=None,
            )

    def test_to_json_is_compact_and_round_trips(self) -> None:
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.OK,
            tool_name="get_workflow_status",
            call_id="call_1",
            evidence_id="evt_1",
            output={"status": "ok"},
        )
        text = envelope.to_json()
        assert "\n" not in text
        parsed = json.loads(text)
        assert parsed["status"] == "ok"
        assert parsed["evidence_id"] == "evt_1"

    def test_frozen_and_extra_rejected(self) -> None:
        assert ToolResultEnvelope.model_config.get("frozen") is True
        with pytest.raises(ValidationError):
            ToolResultEnvelope(
                status=ToolResultStatus.OK,
                tool_name="get_workflow_status",
                call_id="call_1",
                evidence_id="evt_1",
                output={},
                junk=1,
            )


class TestTranscriptItems:
    def test_message_item(self) -> None:
        item = AgentMessageItem(role="user", content="hi")
        assert item.type == "message"
        with pytest.raises(ValidationError):
            AgentMessageItem(role="system", content="hi")

    def test_function_call_item(self) -> None:
        item = AgentFunctionCallItem(
            call_id="call_1",
            name="get_workflow_status",
            arguments='{"thread_id": null}',
        )
        assert item.type == "function_call"

    def test_function_output_item(self) -> None:
        item = AgentFunctionOutputItem(
            call_id="call_1",
            output='{"status":"ok"}',
        )
        assert item.type == "function_call_output"

    @pytest.mark.parametrize(
        "model",
        [
            AgentMessageItem,
            AgentFunctionCallItem,
            AgentFunctionOutputItem,
        ],
    )
    def test_frozen_and_extra_rejected(self, model: Any) -> None:
        assert model.model_config.get("frozen") is True


class TestAgentEventAndErrorData:
    def test_agent_event_valid(self) -> None:
        event = AgentEvent(
            event_id="evt_1",
            conversation_id="c1",
            node="call_agent_model",
            event_type="model_called",
            created_at="2026-08-06T00:00:00Z",
            status="running",
            message="model call 1",
            metrics={"model_call_count": 1},
        )
        dumped = event.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        assert AgentEvent.model_config.get("frozen") is True

    def test_agent_error_data_valid(self) -> None:
        error = AgentErrorData(
            code="AGENT_MODEL_FAILED",
            node="call_agent_model",
            message="safe summary",
            retryable=False,
            exception_type="LLMTimeoutError",
            occurred_at="2026-08-06T00:00:00Z",
        )
        assert error.retryable is False
        assert AgentErrorData.model_config.get("frozen") is True
