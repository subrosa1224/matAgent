"""Unit tests for the mock agent model (S3.5)."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from materials_screening.agent.errors import AgentModelError
from materials_screening.agent.mock_model import (
    MockAgentTurn,
    MockMaterialAgentModel,
    MockToolCall,
    WorkflowDrivenMockAgentModel,
)
from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentModel,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFinalDraft,
    AgentFinalStatus,
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
    ToolResultEnvelope,
    ToolResultStatus,
)
from materials_screening.agent.tool_base import (
    AgentToolDefinition,
    ToolSideEffect,
)
from materials_screening.agent.transcript import AgentTranscript


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str


def _definition(name: str = "run_screening_workflow") -> AgentToolDefinition:
    return AgentToolDefinition(
        name=name,
        description=f"Run {name}",
        parameters=_ToolInput.model_json_schema(),
        side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        version="1",
    )


def _request(
    *,
    input_items: tuple[Any, ...] | None = None,
    tool_definitions: tuple[AgentToolDefinition, ...] = (_definition(),),
) -> MaterialAgentRequest:
    items = input_items if input_items is not None else (AgentTranscript.user("hi"),)
    return MaterialAgentRequest(
        instructions="You are a materials assistant.",
        input_items=items,
        tool_definitions=tool_definitions,
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        max_output_tokens=4096,
        reasoning_effort="none",
        temperature=0.0,
    )


def _call(
    call_id: str = "call_1",
    name: str = "run_screening_workflow",
    arguments: str = '{"query":"不含 Pb 的半导体"}',
) -> MockToolCall:
    return MockToolCall(call_id=call_id, name=name, arguments=arguments)


class TestMockMaterialAgentModel:
    def test_single_function_call(self) -> None:
        model = MockMaterialAgentModel(script=[MockAgentTurn(tool_calls=(_call(),))])
        response = model.generate(_request())

        assert response.status is AgentModelStatus.COMPLETED
        assert response.message_text is None
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.call_id == "call_1"
        assert call.name == "run_screening_workflow"
        assert call.arguments == '{"query":"不含 Pb 的半导体"}'
        assert model.call_count == 1

    def test_multiple_function_calls_preserve_order(self) -> None:
        calls = (
            _call("call_1", "run_screening_workflow"),
            _call("call_2", "get_workflow_status", "{}"),
            _call("call_3", "get_screening_result", '{"material_id":"mp-1"}'),
        )
        model = MockMaterialAgentModel(script=[MockAgentTurn(tool_calls=calls)])

        response = model.generate(_request())

        assert [c.call_id for c in response.tool_calls] == [
            "call_1",
            "call_2",
            "call_3",
        ]
        assert [c.name for c in response.tool_calls] == [
            "run_screening_workflow",
            "get_workflow_status",
            "get_screening_result",
        ]

    def test_final_draft_message(self) -> None:
        draft = {
            "status": AgentFinalStatus.COMPLETED.value,
            "answer": "筛选完成，推荐 BaTiO3。",
            "referenced_material_ids": ["mp-1"],
            "evidence_ids": ["ev-1"],
            "warnings": [],
        }
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(draft)],
        )

        response = model.generate(_request())

        assert response.status is AgentModelStatus.COMPLETED
        assert response.message_text is not None
        parsed = AgentFinalDraft.model_validate(json.loads(response.message_text))
        assert parsed.status is AgentFinalStatus.COMPLETED
        assert parsed.answer == "筛选完成，推荐 BaTiO3。"
        assert response.tool_calls == ()

    def test_mixed_message_and_tool_call(self) -> None:
        turn = MockAgentTurn(
            message="正在查询材料数据。",
            tool_calls=(_call(),),
        )
        model = MockMaterialAgentModel(script=[turn])

        response = model.generate(_request())

        assert response.message_text == "正在查询材料数据。"
        assert len(response.tool_calls) == 1
        assert [type(item).__name__ for item in response.output_items] == [
            "AgentMessageItem",
            "AgentFunctionCallItem",
        ]

    def test_unknown_tool_is_returned_verbatim(self) -> None:
        known = (_definition("get_workflow_status"),)
        request = _request(tool_definitions=known)
        unknown_call = _call(name="run_screening_workflow")
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(tool_calls=(unknown_call,))]
        )

        response = model.generate(request)

        assert response.tool_calls[0].name == "run_screening_workflow"
        assert response.tool_calls[0].name not in {d.name for d in known}

    def test_invalid_arguments_are_preserved(self) -> None:
        bad_arguments = '{"query": "unterminated'
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    tool_calls=(_call(call_id="call_bad", arguments=bad_arguments),)
                )
            ]
        )

        response = model.generate(_request())

        assert response.tool_calls[0].arguments == bad_arguments
        with pytest.raises(json.JSONDecodeError):
            json.loads(response.tool_calls[0].arguments)

    def test_incomplete_turn(self) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    status=AgentModelStatus.INCOMPLETE,
                    error="max_output_tokens reached",
                )
            ]
        )

        response = model.generate(_request())

        assert response.status is AgentModelStatus.INCOMPLETE
        assert response.error == "max_output_tokens reached"
        assert response.output_items == ()

    def test_failed_turn(self) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    status=AgentModelStatus.FAILED,
                    error="rate limit exceeded",
                )
            ]
        )

        response = model.generate(_request())

        assert response.status is AgentModelStatus.FAILED
        assert response.error == "rate limit exceeded"

    def test_repeat_turn_simulates_loop(self) -> None:
        loop_call = _call(call_id="loop_1")
        model = MockMaterialAgentModel(
            repeat_turn=MockAgentTurn(tool_calls=(loop_call,))
        )

        for _ in range(4):
            response = model.generate(_request())
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0].call_id == "loop_1"

        assert model.call_count == 4

    def test_repeat_turn_used_after_script_exhaustion(self) -> None:
        first = MockAgentTurn(message="先查询。")
        loop_call = _call(call_id="loop_1")
        model = MockMaterialAgentModel(
            script=[first],
            repeat_turn=MockAgentTurn(tool_calls=(loop_call,)),
        )

        first_response = model.generate(_request())
        second_response = model.generate(_request())
        third_response = model.generate(_request())

        assert first_response.message_text == "先查询。"
        assert second_response.tool_calls[0].call_id == "loop_1"
        assert third_response.tool_calls[0].call_id == "loop_1"

    def test_fixed_tokens_and_latency(self) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    message="ok",
                    input_tokens=5,
                    output_tokens=7,
                    reasoning_tokens=1,
                    latency_ms=3,
                )
            ],
            sleep_for_latency=False,
        )

        response = model.generate(_request())

        assert response.input_tokens == 5
        assert response.output_tokens == 7
        assert response.reasoning_tokens == 1
        assert response.latency_ms == 3
        assert response.provider == "mock"
        assert response.model == "mock-material-agent"

    def test_default_usage_is_zero_latency_and_fixed(self) -> None:
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="ok")])
        response = model.generate(_request())

        assert response.latency_ms == 0
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.reasoning_tokens == 0

    def test_script_exhaustion_raises(self) -> None:
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="only turn")])
        model.generate(_request())

        with pytest.raises(AgentModelError, match="script exhausted"):
            model.generate(_request())

    def test_constructor_requires_script_or_repeat(self) -> None:
        with pytest.raises(ValueError, match="script or a repeat turn"):
            MockMaterialAgentModel()

    def test_request_ids_deterministic_and_unique_per_call(self) -> None:
        turn = MockAgentTurn(message="ok")
        first = MockMaterialAgentModel(script=[turn, turn])
        second = MockMaterialAgentModel(script=[turn, turn])

        first_a = first.generate(_request()).request_id
        second_a = second.generate(_request()).request_id
        first_b = first.generate(_request()).request_id

        assert first_a == second_a
        assert first_a != first_b
        assert first_b.startswith("mock-")

    def test_multi_turn_transcript_accepted(self) -> None:
        transcript = AgentTranscript()
        transcript.append(AgentTranscript.user("筛选半导体"))
        transcript.append(
            AgentTranscript.function_call(
                call_id="call_1",
                name="run_screening_workflow",
                arguments='{"query":"筛选半导体"}',
            )
        )
        transcript.append(
            AgentTranscript.function_output(
                call_id="call_1",
                output='{"status":"completed","thread_id":"t1"}',
            )
        )
        transcript.append(AgentTranscript.assistant("继续。"))
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="完成。")])

        response = model.generate(_request(input_items=transcript.items()))

        assert response.message_text == "完成。"

    def test_protocol_conformance(self) -> None:
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="ok")])
        assert isinstance(model, MaterialAgentModel)

    def test_rejects_raw_dict_input_items(self) -> None:
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="ok")])
        request = _request(input_items=({"type": "message"},))

        with pytest.raises(AgentModelError, match="unsupported input item"):
            model.generate(request)


class TestMockAgentTurn:
    def test_completed_turn_requires_content(self) -> None:
        with pytest.raises(ValueError, match="message or a tool call"):
            MockAgentTurn()

    def test_incomplete_turn_requires_error(self) -> None:
        with pytest.raises(ValueError, match="error summary"):
            MockAgentTurn(status=AgentModelStatus.INCOMPLETE)

    def test_failed_turn_requires_error(self) -> None:
        with pytest.raises(ValueError, match="error summary"):
            MockAgentTurn(status=AgentModelStatus.FAILED)

    def test_final_draft_builder_uses_compact_json(self) -> None:
        turn = MockAgentTurn.final_draft({"answer": "ok"})
        assert json.loads(turn.message or "") == {"answer": "ok"}
        assert turn.status is AgentModelStatus.COMPLETED


class TestMaterialAgentResponse:
    def test_failed_response_requires_error(self) -> None:
        with pytest.raises(ValidationError):
            MaterialAgentResponse(
                status=AgentModelStatus.FAILED,
                output_items=(),
                request_id="mock-1",
                provider="mock",
                model="mock-material-agent",
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(AgentMessageItem(role="assistant", content="ok"),),
                request_id="mock-1",
                provider="mock",
                model="mock-material-agent",
                junk=True,
            )

    def test_frozen(self) -> None:
        response = MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(AgentMessageItem(role="assistant", content="ok"),),
            request_id="mock-1",
            provider="mock",
            model="mock-material-agent",
        )
        assert response.model_config.get("frozen") is True
        assert response.message_text == "ok"
        assert response.tool_calls == ()


class TestWorkflowDrivenMockAgentModel:
    """Offline demo model: tool call first, draft composed from tool output."""

    def _first_call(self, model: WorkflowDrivenMockAgentModel) -> MaterialAgentResponse:
        items = (
            AgentMessageItem(role="user", content="第一条消息"),
            AgentMessageItem(role="assistant", content="历史回答"),
            AgentMessageItem(role="user", content="寻找不含 Pb 的半导体"),
        )
        return model.generate(_request(input_items=items))

    def _run_output(self, **overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "status": "completed",
            "thread_id": "run-1",
            "planner_status": "ready",
            "clarification_question": None,
            "retrieved_count": 5,
            "filtered_count": 3,
            "returned_count": 2,
            "validation_passed": True,
            "exports": [],
            "warnings": [],
            "evidence_id": "ev_1",
            "top_candidates": [
                {
                    "rank": 1,
                    "material_id": "mp-1",
                    "formula_pretty": "LiFeO2",
                    "band_gap_ev": 2.1,
                    "energy_above_hull_ev_atom": 0.01,
                    "density_g_cm3": 5.2,
                    "total_score": 0.95,
                },
                {
                    "rank": 2,
                    "material_id": "mp-2",
                    "formula_pretty": "ZnO",
                    "band_gap_ev": 3.3,
                    "energy_above_hull_ev_atom": None,
                    "density_g_cm3": None,
                    "total_score": 0.9,
                },
            ],
        }
        values.update(overrides)
        return values

    def _second_call(
        self,
        model: WorkflowDrivenMockAgentModel,
        *,
        envelope: str,
    ) -> MaterialAgentResponse:
        items = (
            AgentMessageItem(role="user", content="寻找不含 Pb 的半导体"),
            AgentFunctionCallItem(
                call_id="mock_call_1",
                name="run_screening_workflow",
                arguments='{"query":"寻找不含 Pb 的半导体"}',
            ),
            AgentFunctionOutputItem(call_id="mock_call_1", output=envelope),
        )
        return model.generate(_request(input_items=items))

    @staticmethod
    def _ok_envelope(output: dict[str, Any], *, evidence_id: str = "ev_1") -> str:
        return ToolResultEnvelope(
            status=ToolResultStatus.OK,
            tool_name="run_screening_workflow",
            call_id="mock_call_1",
            evidence_id=evidence_id,
            output=output,
        ).to_json()

    def test_first_call_emits_tool_call_with_last_user_message(self) -> None:
        model = WorkflowDrivenMockAgentModel()

        response = self._first_call(model)

        assert response.status is AgentModelStatus.COMPLETED
        assert response.message_text is None
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "run_screening_workflow"
        arguments = json.loads(call.arguments)
        assert arguments == {"query": "寻找不含 Pb 的半导体"}

    def test_concept_question_answers_without_tool_call(self) -> None:
        model = WorkflowDrivenMockAgentModel()

        response = model.generate(
            _request(input_items=(AgentMessageItem(role="user", content="你好"),))
        )

        assert response.status is AgentModelStatus.COMPLETED
        assert response.tool_calls == ()
        draft = json.loads(response.message_text or "")
        assert draft["status"] == "completed"
        assert "离线演示" in draft["answer"]
        assert draft["evidence_ids"] == []
        AgentFinalDraft.model_validate(draft)

    def test_multi_turn_re_emits_tool_call_after_draft(self) -> None:
        """Call counts accumulate across turns; behaviour follows the
        transcript, so a fresh user message triggers a new tool call."""
        model = WorkflowDrivenMockAgentModel()
        first = self._first_call(model)
        assert first.tool_calls

        # Turn 1 already has a tool result: the model composes the draft.
        turn1_output = self._ok_envelope(self._run_output())
        draft_response = self._second_call(model, envelope=turn1_output)
        assert draft_response.message_text is not None

        # Turn 2: a new user message without a tool result → tool call again.
        second_turn = model.generate(
            _request(input_items=(AgentMessageItem(role="user", content="寻找半导体"),))
        )
        assert second_turn.message_text is None
        assert len(second_turn.tool_calls) == 1
        assert json.loads(second_turn.tool_calls[0].arguments) == {
            "query": "寻找半导体"
        }

    def test_multi_turn_draft_ignores_previous_turn_envelope(self) -> None:
        """The draft parses only the current turn's tool envelope."""
        model = WorkflowDrivenMockAgentModel()
        # Turn 1: tool call → old envelope result.
        self._first_call(model)
        old = self._ok_envelope(self._run_output(evidence_id="ev_old"))
        self._second_call(model, envelope=old)
        # Turn 2: tool call → new envelope result with its own evidence id.
        second_turn = model.generate(
            _request(input_items=(AgentMessageItem(role="user", content="寻找半导体"),))
        )
        assert second_turn.tool_calls
        new = self._ok_envelope(
            self._run_output(evidence_id="ev_new"), evidence_id="ev_new"
        )
        response = model.generate(
            _request(
                input_items=(
                    AgentMessageItem(role="user", content="寻找半导体"),
                    AgentFunctionCallItem(
                        call_id="mock_call_1",
                        name="run_screening_workflow",
                        arguments='{"query":"寻找半导体"}',
                    ),
                    AgentFunctionOutputItem(call_id="mock_call_1", output=new),
                )
            )
        )

        draft = json.loads(response.message_text or "")
        assert draft["evidence_ids"] == ["ev_new"]

    def test_second_call_composes_draft_from_tool_output(self) -> None:
        model = WorkflowDrivenMockAgentModel()
        first = self._first_call(model)

        response = self._second_call(
            model,
            envelope=self._ok_envelope(self._run_output()),
        )

        assert response.status is AgentModelStatus.COMPLETED
        assert first.request_id != response.request_id
        draft = json.loads(response.message_text or "")
        assert draft["status"] == "completed"
        assert "检索 5 → 过滤 3 → 返回 2" in draft["answer"]
        assert "mp-1" in draft["answer"]
        assert "LiFeO2" in draft["answer"]
        assert draft["active_workflow_thread_id"] == "run-1"
        assert draft["referenced_material_ids"] == ["mp-1", "mp-2"]
        assert draft["evidence_ids"] == ["ev_1"]
        assert draft["follow_up_question"] is None
        # The composed draft must satisfy the final-draft schema.
        parsed = AgentFinalDraft.model_validate(draft)
        assert parsed.status is AgentFinalStatus.COMPLETED

    def test_no_results_draft(self) -> None:
        model = WorkflowDrivenMockAgentModel()
        self._first_call(model)
        output = self._run_output(
            status="no_results",
            returned_count=0,
            top_candidates=[],
        )

        response = self._second_call(model, envelope=self._ok_envelope(output))

        draft = json.loads(response.message_text or "")
        assert draft["status"] == "completed"
        assert "未找到符合条件的材料" in draft["answer"]
        assert draft["referenced_material_ids"] == []
        assert draft["evidence_ids"] == ["ev_1"]

    def test_needs_clarification_draft(self) -> None:
        model = WorkflowDrivenMockAgentModel()
        self._first_call(model)
        output = self._run_output(
            status="needs_clarification",
            clarification_question="请明确带隙范围。",
        )

        response = self._second_call(model, envelope=self._ok_envelope(output))

        draft = json.loads(response.message_text or "")
        assert draft["status"] == "needs_user_input"
        assert draft["follow_up_question"] == "请明确带隙范围。"
        AgentFinalDraft.model_validate(draft)

    def test_workflow_failed_draft_is_honest_error(self) -> None:
        model = WorkflowDrivenMockAgentModel()
        self._first_call(model)
        output = self._run_output(
            status="failed",
            warnings=["PLANNER_ERROR: fixture miss"],
        )

        response = self._second_call(model, envelope=self._ok_envelope(output))

        draft = json.loads(response.message_text or "")
        assert draft["status"] == "error"
        assert "failed" in draft["answer"]
        assert "PLANNER_ERROR" in draft["answer"]
        assert draft["referenced_material_ids"] == []
        AgentFinalDraft.model_validate(draft)

    def test_envelope_error_draft(self) -> None:
        model = WorkflowDrivenMockAgentModel()
        self._first_call(model)
        envelope = ToolResultEnvelope(
            status=ToolResultStatus.ERROR,
            tool_name="run_screening_workflow",
            call_id="mock_call_1",
            evidence_id="",
            error={
                "code": "INVALID_ARGUMENTS",
                "message": "查询过长",
                "retryable": False,
            },
        ).to_json()

        response = self._second_call(model, envelope=envelope)

        draft = json.loads(response.message_text or "")
        assert draft["status"] == "error"
        assert "INVALID_ARGUMENTS" in draft["answer"]
        assert draft["evidence_ids"] == []
        AgentFinalDraft.model_validate(draft)

    def test_without_tool_result_emits_tool_call(self) -> None:
        """Without a tool result in the current turn the model asks for one
        (emits the tool call) instead of composing an ungrounded draft."""
        model = WorkflowDrivenMockAgentModel()
        self._first_call(model)

        response = model.generate(
            _request(input_items=(AgentMessageItem(role="user", content="寻找半导体"),))
        )

        assert response.message_text is None
        assert len(response.tool_calls) == 1

    def test_deterministic_given_same_transcript(self) -> None:
        items = (
            AgentMessageItem(role="user", content="寻找半导体"),
            AgentMessageItem(role="user", content="寻找不含 Pb 的半导体"),
        )
        first = WorkflowDrivenMockAgentModel().generate(_request(input_items=items))
        second = WorkflowDrivenMockAgentModel().generate(_request(input_items=items))

        assert first.request_id == second.request_id
        assert first.tool_calls[0].arguments == second.tool_calls[0].arguments
