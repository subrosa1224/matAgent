"""Unit tests for the single agent LangGraph (S3.5-M5)."""

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START
from pydantic import ValidationError

from materials_screening.agent.context import MaterialAgentContext
from materials_screening.agent.errors import AgentInvariantError, AgentModelError
from materials_screening.agent.graph_builder import (
    AgentGraphInput,
    compile_agent_graph,
)
from materials_screening.agent.ledger import ToolExecutionLedger
from materials_screening.agent.mock_model import (
    MockAgentTurn,
    MockMaterialAgentModel,
    MockToolCall,
)
from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    ToolResultEnvelope,
    ToolResultStatus,
)
from materials_screening.agent.nodes import (
    NODE_CALL_AGENT_MODEL,
    NODE_EXECUTE_TOOLS,
    NODE_FINALIZE_CANCELLED,
    NODE_FINALIZE_ERROR,
    NODE_FINALIZE_SUCCESS,
    NODE_PREPARE_TURN,
    NODE_VALIDATE_FINAL,
    call_agent_model_node,
    execute_tools_node,
    finalize_cancelled_node,
    finalize_error_node,
    finalize_success_node,
    prepare_turn_node,
    route_after_model,
    route_after_tools,
    route_after_validation,
    validate_final_node,
)
from materials_screening.agent.policy import AgentToolPolicy
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_base import ToolSideEffect
from materials_screening.agent.tool_executor import (
    ToolExecutionOutcome,
    ToolExecutor,
)
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools import (
    CompareRankedMaterialsTool,
    GetScreeningResultTool,
    GetWorkflowHistoryTool,
    GetWorkflowStatusTool,
    RunScreeningWorkflowTool,
)
from materials_screening.sub_agents.literature.tools import (
    CandidateLiteratureScreenTool,
    UnifiedLiteratureSearchTool,
)
from materials_screening.workflow.input_output import WorkflowInput, WorkflowOutput
from materials_screening.workflow.state import WorkflowStatus


def _fixed_clock() -> datetime:
    return datetime(2026, 8, 6, 0, 0, tzinfo=UTC)


class _SeqIdGenerator:
    def __init__(self) -> None:
        self._count = 0

    def new_id(self) -> str:
        self._count += 1
        return f"id_{self._count}"


class _FakeRunner:
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


class _FakeReader:
    def read(self, thread_id: str) -> dict[str, Any]:
        return {}


@dataclass
class _FakeRuntime:
    context: MaterialAgentContext


class _RecordingModel:
    """Wraps a model and records every request it receives."""

    def __init__(self, inner: MockMaterialAgentModel) -> None:
        self.inner = inner
        self.requests: list[MaterialAgentRequest] = []

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        self.requests.append(request)
        return self.inner.generate(request)


def _settings(**overrides: Any) -> AgentSettings:
    values: dict[str, Any] = {}
    values.update(overrides)
    return AgentSettings(_env_file=None, **values)


def _workflow_output(
    status: WorkflowStatus = WorkflowStatus.COMPLETED,
    *,
    thread_id: str = "run-1",
    **overrides: Any,
) -> WorkflowOutput:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "thread_id": thread_id,
        "status": status,
        "planner_status": None,
        "request": None,
        "retrieved_count": 5,
        "filtered_count": 3,
        "returned_count": 3,
        "validation_passed": True,
        "exports": (),
        "warnings": (),
        "error": None,
        "clarification_question": None,
    }
    values.update(overrides)
    return WorkflowOutput.model_validate(values)


def _registry() -> AgentToolRegistry:
    return AgentToolRegistry(
        (
            RunScreeningWorkflowTool(),
            GetWorkflowStatusTool(),
            GetWorkflowHistoryTool(),
            GetScreeningResultTool(),
            CompareRankedMaterialsTool(),
        )
    )


def _executor(
    settings: AgentSettings,
    *,
    registry: AgentToolRegistry | None = None,
) -> ToolExecutor:
    resolved_registry = registry or _registry()
    policy = AgentToolPolicy(
        max_tool_calls_per_turn=settings.agent_max_tool_calls_per_turn,
        max_workflow_runs_per_turn=settings.agent_max_workflow_runs_per_turn,
        max_argument_bytes=settings.agent_max_argument_bytes,
    )
    return ToolExecutor(
        registry=resolved_registry,
        policy=policy,
        max_argument_bytes=settings.agent_max_argument_bytes,
        max_output_bytes=settings.agent_max_tool_output_bytes,
    )


def _context(**overrides: Any) -> MaterialAgentContext:
    settings = overrides.pop("settings", None) or _settings()
    values: dict[str, Any] = {
        "workflow_runner": _FakeRunner(_workflow_output()),
        "workflow_result_reader": _FakeReader(),
        "tool_registry": _registry(),
        "tool_executor": _executor(settings),
        "agent_model": MockMaterialAgentModel(script=[MockAgentTurn(message="ok")]),
        "ledger": ToolExecutionLedger(),
        "settings": settings,
        "clock": _fixed_clock,
        "id_generator": _SeqIdGenerator(),
        "conversation_links": (),
    }
    values.update(overrides)
    return MaterialAgentContext(**values)


def _input_state(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "conversation_id": "c1",
        "user_message": "筛选半导体",
        "status": "running",
        "current_node": NODE_PREPARE_TURN,
        "input_items": [],
        "active_workflow_thread_id": None,
        "model_call_count": 0,
        "tool_call_count": 0,
        "workflow_run_count": 0,
        "pending_tool_calls": [],
        "executed_call_ids": [],
        "evidence_ids": [],
        "final_draft": None,
        "final_response": None,
        "error": None,
        "events": [],
    }
    values.update(overrides)
    return values


def _concept_draft() -> dict[str, Any]:
    return {
        "status": "completed",
        "answer": "无机半导体是材料科学的重要领域。",
        "active_workflow_thread_id": None,
        "referenced_material_ids": [],
        "evidence_ids": [],
        "warnings": [],
        "follow_up_question": None,
    }


class TestPrepareTurnNode:
    def test_initializes_fresh_turn(self) -> None:
        context = _context()
        result = prepare_turn_node(_input_state(), _FakeRuntime(context))

        assert result["user_turn_id"] == "id_1"
        assert result["input_items"] == [
            {"type": "message", "role": "user", "content": "筛选半导体"}
        ]
        assert result["model_call_count"] == 0
        assert result["tool_call_count"] == 0
        assert result["workflow_run_count"] == 0
        assert result["pending_tool_calls"] == []
        assert result["error"] is None
        assert result["status"] == "running"
        assert len(result["events"]) == 1
        assert result["events"][0]["event_type"] == "turn_started"

    def test_rejects_unsafe_conversation_id(self) -> None:
        context = _context()
        state = _input_state(conversation_id="bad conversation!")

        result = prepare_turn_node(state, _FakeRuntime(context))

        assert result["error"]["code"] == "INVALID_CONVERSATION"
        assert result["status"] == "error"

    def test_rejects_oversized_message(self) -> None:
        context = _context(settings=_settings(agent_max_input_bytes=10))

        result = prepare_turn_node(
            _input_state(user_message="x" * 20),
            _FakeRuntime(context),
        )

        assert result["error"]["code"] == "INPUT_TOO_LARGE"


class TestCallAgentModelNode:
    def test_candidate_pool_does_not_fall_back_to_ordinary_search(self) -> None:
        registry = AgentToolRegistry(
            (
                UnifiedLiteratureSearchTool(object()),  # type: ignore[arg-type]
                CandidateLiteratureScreenTool(object()),  # type: ignore[arg-type]
            )
        )
        model = _RecordingModel(
            MockMaterialAgentModel(script=[MockAgentTurn.final_draft(_concept_draft())])
        )
        context = _context(tool_registry=registry, agent_model=model)
        message = (
            "执行候选池文献预检（不是最终推荐）：1:CsAlSiO4；2:Ca3Ga2(GeO4)3。"
            "最终列出前10名。\n独立探索池：S1:ZnO。按 A/B/C/NONE 判断。"
        )
        result = call_agent_model_node(
            _input_state(user_message=message), _FakeRuntime(context)
        )
        assert model.requests == []
        call = result["pending_tool_calls"][0]
        assert call["name"] == "screen_candidate_literature"
        arguments = json.loads(call["arguments_json"])
        assert arguments["materials"] == ["CsAlSiO4", "Ca3Ga2(GeO4)3", "ZnO"]
        assert arguments["supplementary_materials"] == ["ZnO"]
        assert arguments["final_limit"] == 10

    def test_explicit_literature_search_is_routed_without_model_choice(self) -> None:
        settings = _settings()
        literature_registry = AgentToolRegistry(
            (UnifiedLiteratureSearchTool(object()),)  # type: ignore[arg-type]
        )
        model = _RecordingModel(
            MockMaterialAgentModel(script=[MockAgentTurn.final_draft(_concept_draft())])
        )
        context = _context(
            settings=settings,
            tool_registry=literature_registry,
            tool_executor=_executor(settings, registry=literature_registry),
            agent_model=model,
        )
        message = "帮我检索近5年（2019-2024年）3D打印生物活性玻璃支架孔结构与成骨的论文"
        state = _input_state(
            user_message=message,
            input_items=[
                {
                    "type": "message",
                    "role": "user",
                    "content": message,
                }
            ],
        )

        result = call_agent_model_node(state, _FakeRuntime(context))

        assert model.requests == []
        assert result["model_call_count"] == 0
        assert result["pending_tool_calls"][0]["name"] == "literature_search"
        arguments = json.loads(result["pending_tool_calls"][0]["arguments_json"])
        assert arguments["year_from"] == 2021
        assert arguments["year_to"] == 2026
        assert "2021-2026年" in arguments["topic"]
        assert "2019-2024年" not in arguments["topic"]
        assert arguments["max_papers"] == 20
        assert arguments["sort_mode"] == "recent"

    def test_uses_configured_system_prompt(self) -> None:
        model = _RecordingModel(
            MockMaterialAgentModel(script=[MockAgentTurn.final_draft(_concept_draft())])
        )
        context = _context(
            agent_model=model,
            settings=_settings(agent_system_prompt="outlier-only prompt"),
        )

        call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert model.requests[-1].instructions == "outlier-only prompt"

    def test_first_call_allows_tools_second_call_forbids(self) -> None:
        draft = _concept_draft()
        model = _RecordingModel(
            MockMaterialAgentModel(
                script=[MockAgentTurn.final_draft(draft)],
                repeat_turn=MockAgentTurn.final_draft(draft),
            )
        )
        context = _context(agent_model=model)

        first = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert first["model_call_count"] == 1
        assert model.requests[-1].allow_tool_calls is True

        second_state = _input_state(
            model_call_count=1,
            input_items=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": '{"status":"completed"}',
                }
            ],
        )
        call_agent_model_node(second_state, _FakeRuntime(context))

        assert model.requests[-1].allow_tool_calls is False

    def test_plain_search_forbids_unrequested_second_tool(self) -> None:
        draft = _concept_draft()
        model = _RecordingModel(
            MockMaterialAgentModel(
                script=[MockAgentTurn.final_draft(draft)],
                repeat_turn=MockAgentTurn.final_draft(draft),
            )
        )
        context = _context(
            agent_model=model,
            settings=_settings(agent_allow_multi_step_tools=True),
        )
        state = _input_state(
            user_message="筛选稳定材料并返回材料 ID 和带隙",
            model_call_count=1,
            tool_call_count=1,
        )

        call_agent_model_node(state, _FakeRuntime(context))

        assert model.requests[-1].allow_tool_calls is False

    def test_explicit_statistics_allows_second_tool(self) -> None:
        draft = _concept_draft()
        model = _RecordingModel(
            MockMaterialAgentModel(
                script=[MockAgentTurn.final_draft(draft)],
                repeat_turn=MockAgentTurn.final_draft(draft),
            )
        )
        context = _context(
            agent_model=model,
            settings=_settings(agent_allow_multi_step_tools=True),
        )
        state = _input_state(
            user_message="筛选稳定材料并统计带隙分布",
            model_call_count=1,
            tool_call_count=1,
        )

        call_agent_model_node(state, _FakeRuntime(context))

        assert model.requests[-1].allow_tool_calls is True

    def test_tool_calls_become_pending(self) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    tool_calls=(
                        MockToolCall(
                            call_id="call_1",
                            name="run_screening_workflow",
                            arguments='{"query":"筛选半导体"}',
                        ),
                    )
                )
            ]
        )
        context = _context(agent_model=model)
        state = _input_state(
            input_items=[{"type": "message", "role": "user", "content": "筛选半导体"}]
        )

        result = call_agent_model_node(state, _FakeRuntime(context))

        assert result["model_call_count"] == 1
        assert result["pending_tool_calls"] == [
            {
                "call_id": "call_1",
                "name": "run_screening_workflow",
                "arguments_json": '{"query":"筛选半导体"}',
            }
        ]
        assert result["final_draft"] is None
        assert result["input_items"][-1]["type"] == "function_call"

    def test_final_draft_message_parsed(self) -> None:
        draft = _concept_draft()
        model = MockMaterialAgentModel(script=[MockAgentTurn.final_draft(draft)])
        context = _context(agent_model=model)
        state = _input_state(
            input_items=[{"type": "message", "role": "user", "content": "hi"}]
        )

        result = call_agent_model_node(state, _FakeRuntime(context))

        assert result["model_call_count"] == 1
        assert result["pending_tool_calls"] == []
        assert result["final_draft"]["answer"] == draft["answer"]
        assert result["input_items"][-1]["type"] == "message"
        assert result["input_items"][-1]["role"] == "assistant"

    def test_model_call_limit_enforced(self) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(message="should not be called")]
        )
        context = _context(
            settings=_settings(agent_max_model_calls_per_turn=1),
            agent_model=model,
        )
        state = _input_state(model_call_count=1)

        result = call_agent_model_node(state, _FakeRuntime(context))

        assert result["error"]["code"] == "MODEL_CALL_LIMIT"
        assert model.call_count == 0

    def test_transcript_size_limit_enforced(self) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(message="should not be called")]
        )
        context = _context(
            settings=_settings(agent_max_input_bytes=100),
            agent_model=model,
        )
        state = _input_state(
            input_items=[{"type": "message", "role": "user", "content": "x" * 500}]
        )

        result = call_agent_model_node(state, _FakeRuntime(context))

        assert result["error"]["code"] == "INPUT_TOO_LARGE"
        assert model.call_count == 0

    def test_incomplete_model_response_maps_to_error(self) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    status=AgentModelStatus.INCOMPLETE,
                    error="max_output_tokens reached",
                )
            ]
        )
        context = _context(agent_model=model)

        result = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert result["error"]["code"] == "MODEL_INCOMPLETE"
        assert result["error"]["message"] == "max_output_tokens reached"
        assert result["model_call_count"] == 1

    def test_agent_model_error_maps_to_safe_error(self) -> None:
        class _RaisingModel:
            def generate(self, request: object) -> object:
                raise AgentModelError("boom")

        context = _context(agent_model=_RaisingModel())

        result = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert result["error"]["code"] == "MODEL_ERROR"
        assert result["error"]["message"] == "boom"

    def test_unknown_model_exception_propagates(self) -> None:
        class _BoomModel:
            def generate(self, request: object) -> object:
                raise RuntimeError("boom")

        context = _context(agent_model=_BoomModel())

        with pytest.raises(RuntimeError, match="boom"):
            call_agent_model_node(_input_state(), _FakeRuntime(context))

    def test_invalid_final_json_maps_to_error(self) -> None:
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="{not json")])
        context = _context(agent_model=model)

        result = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert result["error"]["code"] == "INVALID_FINAL_DRAFT"

    def test_empty_model_message_maps_to_error(self) -> None:
        class _EmptyModel:
            def generate(self, request: object) -> MaterialAgentResponse:
                return MaterialAgentResponse(
                    status=AgentModelStatus.COMPLETED,
                    output_items=(),
                    request_id="req_1",
                    provider="mock",
                    model="mock",
                )

        context = _context(agent_model=_EmptyModel())

        result = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert result["error"]["code"] == "MODEL_EMPTY_OUTPUT"

    def test_unsupported_transcript_item_raises(self) -> None:
        context = _context()
        state = _input_state(input_items=[{"type": "weird", "content": "x"}])

        with pytest.raises(AgentInvariantError, match="unsupported transcript"):
            call_agent_model_node(state, _FakeRuntime(context))

    def test_existing_error_short_circuits(self) -> None:
        context = _context()
        state = _input_state(error={"code": "X", "message": "y"})

        result = call_agent_model_node(state, _FakeRuntime(context))

        assert result == {}


class TestExecuteToolsNode:
    def test_executes_and_registers_evidence(self) -> None:
        runner = _FakeRunner(_workflow_output(thread_id="run-1"))
        ledger = ToolExecutionLedger()
        context = _context(workflow_runner=runner, ledger=ledger)
        state = _input_state(
            user_turn_id="id_1",
            active_workflow_thread_id=None,
            pending_tool_calls=[
                {
                    "call_id": "call_1",
                    "name": "run_screening_workflow",
                    "arguments_json": '{"query":"筛选半导体"}',
                }
            ],
            input_items=[{"type": "message", "role": "user", "content": "筛选半导体"}],
        )

        result = execute_tools_node(state, _FakeRuntime(context))

        assert len(runner.calls) == 1
        assert runner.calls[0].query == "筛选半导体"
        assert result["tool_call_count"] == 1
        assert result["workflow_run_count"] == 1
        assert result["evidence_ids"] == ["id_1"]
        assert result["executed_call_ids"] == ["call_1"]
        assert result["active_workflow_thread_id"] == "run-1"
        assert result["input_items"][-1]["type"] == "function_call_output"
        assert result["input_items"][-1]["call_id"] == "call_1"

    def test_unknown_tool_becomes_safe_error_envelope(self) -> None:
        ledger = ToolExecutionLedger()
        context = _context(ledger=ledger)
        state = _input_state(
            user_turn_id="id_1",
            pending_tool_calls=[
                {
                    "call_id": "call_bad",
                    "name": "web_search",
                    "arguments_json": "{}",
                }
            ],
        )

        result = execute_tools_node(state, _FakeRuntime(context))

        last_item = result["input_items"][-1]
        assert last_item["type"] == "function_call_output"
        payload = json.loads(last_item["output"])
        assert payload["status"] == "error"
        assert payload["error"]["code"] == "UNKNOWN_TOOL"
        assert result["tool_call_count"] == 0

    def test_requires_pending_calls(self) -> None:
        context = _context()

        with pytest.raises(AgentInvariantError, match="pending_tool_calls"):
            execute_tools_node(_input_state(), _FakeRuntime(context))

    def test_invalid_pending_call_raises(self) -> None:
        context = _context()
        state = _input_state(
            pending_tool_calls=[{"call_id": "call_1", "arguments_json": "{}"}]
        )

        with pytest.raises(AgentInvariantError, match="pending_tool_calls"):
            execute_tools_node(state, _FakeRuntime(context))

    def test_run_outcome_with_bad_ledger_json_is_safe(self) -> None:
        class _FakeExecutor:
            def execute(self, calls: object, **kwargs: object) -> object:
                envelope = ToolResultEnvelope(
                    status=ToolResultStatus.OK,
                    tool_name="run_screening_workflow",
                    call_id="call_x",
                    evidence_id="ev_x",
                    output={},
                )
                return (
                    ToolExecutionOutcome(
                        call_id="call_x",
                        tool_name="run_screening_workflow",
                        evidence_id="ev_x",
                        status=ToolResultStatus.OK,
                        envelope=envelope,
                        output_json='{"status":"ok"}',
                    ),
                )

        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_x",
            tool_name="run_screening_workflow",
            evidence_id="ev_x",
            result_json="{bad json",
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        context = _context(tool_executor=_FakeExecutor(), ledger=ledger)
        state = _input_state(
            user_turn_id="id_1",
            pending_tool_calls=[
                {
                    "call_id": "call_x",
                    "name": "run_screening_workflow",
                    "arguments_json": "{}",
                }
            ],
        )

        result = execute_tools_node(state, _FakeRuntime(context))

        assert result["active_workflow_thread_id"] is None
        assert result["input_items"][-1]["type"] == "function_call_output"

    def test_existing_error_short_circuits(self) -> None:
        context = _context()
        state = _input_state(
            pending_tool_calls=[
                {
                    "call_id": "call_1",
                    "name": "run_screening_workflow",
                    "arguments_json": "{}",
                }
            ],
            error={"code": "X", "message": "y"},
        )

        result = execute_tools_node(state, _FakeRuntime(context))

        assert result == {}


class TestValidateFinalNode:
    def test_historical_evidence_in_same_conversation_passes(self) -> None:
        context = _context()
        draft = {
            "status": "completed",
            "answer": "IQR 离群检测完成，样本数为 669。",
            "active_workflow_thread_id": None,
            "referenced_material_ids": [],
            "evidence_ids": ["ev_previous"],
            "warnings": [],
            "follow_up_question": None,
        }
        state = _input_state(
            input_items=[
                {
                    "type": "function_call_output",
                    "call_id": "old-call",
                    "output": json.dumps(
                        {
                            "status": "ok",
                            "tool_name": "detect_material_outliers",
                            "evidence_id": "ev_previous",
                            "output": {"report": {"distribution": {"count": 669}}},
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            final_draft=draft,
        )

        result = validate_final_node(state, _FakeRuntime(context))

        assert "error" not in result
        assert result["current_node"] == NODE_VALIDATE_FINAL

    def test_valid_evidence_backed_draft_passes(self) -> None:
        ledger = ToolExecutionLedger()
        ledger.record(
            call_id="call_1",
            tool_name="run_screening_workflow",
            evidence_id="ev_run",
            result_json=_workflow_output().model_dump_json(),
            side_effect=ToolSideEffect.CREATE_WORKFLOW_RUN,
        )
        context = _context(ledger=ledger)
        draft = {
            "status": "completed",
            "answer": "筛选任务已完成。",
            "active_workflow_thread_id": "run-1",
            "referenced_material_ids": [],
            "evidence_ids": ["ev_run"],
            "warnings": [],
            "follow_up_question": None,
        }
        state = _input_state(
            active_workflow_thread_id="run-1",
            final_draft=draft,
        )

        result = validate_final_node(state, _FakeRuntime(context))

        assert "error" not in result
        assert result["current_node"] == NODE_VALIDATE_FINAL

    def test_unbacked_draft_rejected(self) -> None:
        context = _context()
        draft = _concept_draft()
        draft["answer"] = "筛选任务已完成。"
        draft["evidence_ids"] = ["ev_bogus"]
        state = _input_state(final_draft=draft)

        result = validate_final_node(state, _FakeRuntime(context))

        assert result["error"]["code"] == "FINAL_VALIDATION_FAILED"

    def test_requires_final_draft(self) -> None:
        context = _context()

        with pytest.raises(AgentInvariantError, match="final_draft"):
            validate_final_node(_input_state(), _FakeRuntime(context))

    def test_existing_error_short_circuits(self) -> None:
        context = _context()
        state = _input_state(error={"code": "X", "message": "y"})

        result = validate_final_node(state, _FakeRuntime(context))

        assert result == {}


class TestFinalizeNodes:
    def test_finalize_success_publishes_answer(self) -> None:
        context = _context()
        draft = _concept_draft()
        state = _input_state(final_draft=draft, model_call_count=2)

        result = finalize_success_node(state, _FakeRuntime(context))

        assert result["status"] == "completed"
        assert result["final_response"] == draft["answer"]
        assert result["error"] is None
        assert result["events"][0]["event_type"] == "turn_completed"
        assert result["events"][0]["metrics"]["model_call_count"] == 2

    def test_finalize_success_requires_draft(self) -> None:
        context = _context()

        with pytest.raises(AgentInvariantError, match="final_draft"):
            finalize_success_node(_input_state(), _FakeRuntime(context))

    def test_finalize_error_publishes_safe_summary(self) -> None:
        context = _context()
        state = _input_state(
            final_draft=_concept_draft(),
            error={"code": "MODEL_CALL_LIMIT", "message": "limit reached"},
        )

        result = finalize_error_node(state, _FakeRuntime(context))

        assert result["status"] == "error"
        assert result["final_response"] == "limit reached"
        assert result["final_draft"] is None
        assert result["pending_tool_calls"] == []
        assert result["error"]["code"] == "MODEL_CALL_LIMIT"
        assert result["events"][0]["event_type"] == "turn_error"

    def test_finalize_error_without_error_uses_unknown(self) -> None:
        context = _context()

        result = finalize_error_node(_input_state(), _FakeRuntime(context))

        assert result["error"]["code"] == "UNKNOWN_ERROR"


class TestRouting:
    def test_route_after_model(self) -> None:
        assert route_after_model(_input_state(error={"code": "X"})) == "finalize_error"
        assert (
            route_after_model(_input_state(pending_tool_calls=[{"call_id": "c"}]))
            == "execute_tools"
        )
        assert route_after_model(_input_state(final_draft={})) == "validate_final"
        with pytest.raises(AgentInvariantError):
            route_after_model(_input_state())

    def test_route_after_tools(self) -> None:
        assert route_after_tools(_input_state(error={"code": "X"})) == "finalize_error"
        assert route_after_tools(_input_state()) == "call_agent_model"

    def test_route_after_validation(self) -> None:
        assert (
            route_after_validation(_input_state(error={"code": "X"}))
            == "finalize_error"
        )
        assert route_after_validation(_input_state()) == "finalize_success"


class TestGraphCompile:
    def test_node_set_and_edges(self) -> None:
        compiled = compile_agent_graph(checkpointer=InMemorySaver())
        graph = compiled.get_graph()

        assert set(graph.nodes) == {
            START,
            END,
            NODE_PREPARE_TURN,
            NODE_CALL_AGENT_MODEL,
            NODE_EXECUTE_TOOLS,
            NODE_VALIDATE_FINAL,
            NODE_FINALIZE_SUCCESS,
            NODE_FINALIZE_ERROR,
            NODE_FINALIZE_CANCELLED,
        }
        edges = {(edge.source, edge.target) for edge in graph.edges}
        assert (START, NODE_PREPARE_TURN) in edges
        assert (NODE_PREPARE_TURN, NODE_CALL_AGENT_MODEL) in edges
        assert (NODE_FINALIZE_SUCCESS, END) in edges
        assert (NODE_FINALIZE_ERROR, END) in edges
        assert (NODE_FINALIZE_CANCELLED, END) in edges
        assert {
            target for source, target in edges if source == NODE_CALL_AGENT_MODEL
        } == {
            NODE_EXECUTE_TOOLS,
            NODE_VALIDATE_FINAL,
            NODE_FINALIZE_ERROR,
            NODE_FINALIZE_CANCELLED,
        }
        assert {target for source, target in edges if source == NODE_EXECUTE_TOOLS} == {
            NODE_CALL_AGENT_MODEL,
            NODE_FINALIZE_ERROR,
            NODE_FINALIZE_CANCELLED,
        }
        assert {
            target for source, target in edges if source == NODE_VALIDATE_FINAL
        } == {NODE_FINALIZE_SUCCESS, NODE_FINALIZE_ERROR}

    def test_compiled_graph_invokes_concept_answer(self) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        context = _context(agent_model=model)
        graph = compile_agent_graph(checkpointer=InMemorySaver())

        result = graph.invoke(
            {"conversation_id": "c1", "user_message": "介绍一下半导体"},
            config={"thread_id": "agent_c1", "recursion_limit": 16},
            context=context,
        )

        assert result["status"] == "completed"
        assert result["final_response"] == "无机半导体是材料科学的重要领域。"
        assert model.call_count == 1


class TestGraphFlow:
    def _graph(self) -> Any:
        return compile_agent_graph(checkpointer=InMemorySaver())

    def test_tool_then_final_draft(self) -> None:
        run_call = MockToolCall(
            call_id="call_1",
            name="run_screening_workflow",
            arguments='{"query":"筛选半导体"}',
        )
        draft = {
            "status": "completed",
            "answer": "筛选任务已完成。",
            "active_workflow_thread_id": "run-1",
            "referenced_material_ids": [],
            # prepare_turn consumes id_1 (user_turn_id) and id_2 (event);
            # the run tool registers its evidence as id_3.
            "evidence_ids": ["id_3"],
            "warnings": [],
            "follow_up_question": None,
        }
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(tool_calls=(run_call,)),
                MockAgentTurn.final_draft(draft),
            ]
        )
        runner = _FakeRunner(_workflow_output(thread_id="run-1"))
        context = _context(agent_model=model, workflow_runner=runner)
        graph = self._graph()

        result = graph.invoke(
            {"conversation_id": "c1", "user_message": "筛选半导体"},
            config={"thread_id": "agent_c1", "recursion_limit": 16},
            context=context,
        )

        assert result["status"] == "completed"
        assert result["final_response"] == "筛选任务已完成。"
        assert result["evidence_ids"] == ["id_3"]
        assert result["active_workflow_thread_id"] == "run-1"
        assert len(runner.calls) == 1
        assert runner.calls[0].query == "筛选半导体"
        assert result["events"][-1]["event_type"] == "turn_completed"

    def test_loop_stops_at_model_call_limit(self) -> None:
        loop_call = MockToolCall(
            call_id="loop_1",
            name="run_screening_workflow",
            arguments='{"query":"重复"}',
        )
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(tool_calls=(loop_call,))],
            repeat_turn=MockAgentTurn(tool_calls=(loop_call,)),
        )
        runner = _FakeRunner(_workflow_output())
        context = _context(agent_model=model, workflow_runner=runner)
        graph = self._graph()

        result = graph.invoke(
            {"conversation_id": "c1", "user_message": "重复执行"},
            config={"thread_id": "agent_c1", "recursion_limit": 16},
            context=context,
        )

        assert result["status"] == "error"
        assert result["error"]["code"] == "MODEL_CALL_LIMIT"
        assert result["model_call_count"] == 4
        # The repeated call id is reused idempotently; the runner runs once.
        assert len(runner.calls) == 1

    def test_recursion_limit_raises_graph_recursion_error(self) -> None:
        loop_call = MockToolCall(
            call_id="loop_1",
            name="run_screening_workflow",
            arguments='{"query":"重复"}',
        )
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(tool_calls=(loop_call,))],
            repeat_turn=MockAgentTurn(tool_calls=(loop_call,)),
        )
        context = _context(
            agent_model=model,
            settings=_settings(agent_max_model_calls_per_turn=10),
        )
        graph = self._graph()

        with pytest.raises(GraphRecursionError):
            graph.invoke(
                {"conversation_id": "c1", "user_message": "重复执行"},
                config={"thread_id": "agent_c1", "recursion_limit": 6},
                context=context,
            )

    def test_validation_failure_ends_in_error(self) -> None:
        draft = _concept_draft()
        draft["answer"] = "筛选任务已完成。"
        draft["evidence_ids"] = ["ev_bogus"]
        model = MockMaterialAgentModel(script=[MockAgentTurn.final_draft(draft)])
        context = _context(agent_model=model)
        graph = self._graph()

        result = graph.invoke(
            {"conversation_id": "c1", "user_message": "筛选半导体"},
            config={"thread_id": "agent_c1", "recursion_limit": 16},
            context=context,
        )

        assert result["status"] == "error"
        assert result["error"]["code"] == "FINAL_VALIDATION_FAILED"
        assert result["final_draft"] is None

    def test_model_incomplete_ends_in_error(self) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    status=AgentModelStatus.INCOMPLETE,
                    error="max_output_tokens reached",
                )
            ]
        )
        context = _context(agent_model=model)
        graph = self._graph()

        result = graph.invoke(
            {"conversation_id": "c1", "user_message": "筛选半导体"},
            config={"thread_id": "agent_c1", "recursion_limit": 16},
            context=context,
        )

        assert result["status"] == "error"
        assert result["error"]["code"] == "MODEL_INCOMPLETE"


class TestAgentGraphInput:
    def test_valid_input(self) -> None:
        value = AgentGraphInput(conversation_id="c1", user_message="hi")
        assert value.conversation_id == "c1"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentGraphInput(conversation_id="c1", user_message="hi", junk=True)

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentGraphInput(conversation_id="c1", user_message="")


class TestCancellation:
    def test_call_agent_model_stops_before_model_call(self) -> None:
        event = threading.Event()
        event.set()
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="不应调用")])
        recording = _RecordingModel(model)
        context = _context(agent_model=recording, cancel_event=event)

        result = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert result["cancelled"] is True
        assert result["pending_tool_calls"] == []
        assert recording.requests == []
        assert route_after_model(_input_state(**result)) == "finalize_cancelled"

    def test_call_agent_model_ignores_unset_event(self) -> None:
        event = threading.Event()
        model = MockMaterialAgentModel(script=[MockAgentTurn(message="ok")])
        recording = _RecordingModel(model)
        context = _context(agent_model=recording, cancel_event=event)

        result = call_agent_model_node(_input_state(), _FakeRuntime(context))

        assert result.get("cancelled") is not True
        assert len(recording.requests) == 1

    def test_execute_tools_stops_before_tool_execution(self) -> None:
        event = threading.Event()
        event.set()
        runner = _FakeRunner(_workflow_output())
        context = _context(workflow_runner=runner, cancel_event=event)
        state = _input_state(
            user_turn_id="id_1",
            pending_tool_calls=[
                {
                    "call_id": "call_1",
                    "name": "run_screening_workflow",
                    "arguments_json": '{"query":"筛选半导体"}',
                }
            ],
        )

        result = execute_tools_node(state, _FakeRuntime(context))

        assert result["cancelled"] is True
        assert runner.calls == []
        assert route_after_tools(_input_state(**result)) == "finalize_cancelled"

    def test_finalize_cancelled_publishes_safe_summary(self) -> None:
        context = _context()
        state = _input_state(model_call_count=2, tool_call_count=1)

        result = finalize_cancelled_node(state, _FakeRuntime(context))

        assert result["status"] == "cancelled"
        assert result["error"] is None
        assert result["final_draft"] is None
        assert "已停止" in result["final_response"]
        assert result["events"][0]["event_type"] == "turn_cancelled"

    def test_full_turn_with_cancel_before_tools_ends_cancelled(self) -> None:
        run_call = MockToolCall(
            call_id="call_1",
            name="run_screening_workflow",
            arguments='{"query":"筛选半导体"}',
        )
        # A single-turn script: any second model call would raise, so reaching
        # finalize_cancelled proves the loop stopped after the cancellation.
        model = MockMaterialAgentModel(script=[MockAgentTurn(tool_calls=(run_call,))])
        event = threading.Event()
        graph = compile_agent_graph(checkpointer=InMemorySaver())
        runner = _FakeRunner(_workflow_output())
        context = _context(
            agent_model=model,
            workflow_runner=runner,
            cancel_event=event,
        )

        stream = graph.stream(
            {"conversation_id": "c1", "user_message": "筛选半导体"},
            config={"thread_id": "agent_c1", "recursion_limit": 16},
            context=context,
            stream_mode="updates",
            version="v2",
        )
        next(stream)  # prepare_turn
        next(stream)  # call_agent_model: emits the tool call
        event.set()
        list(stream)  # execute_tools sees the cancel and ends the turn

        snapshot = graph.get_state({"configurable": {"thread_id": "agent_c1"}})
        values = snapshot.values or {}
        assert values["status"] == "cancelled"
        assert values["final_draft"] is None
        assert values["error"] is None
        assert runner.calls == []
