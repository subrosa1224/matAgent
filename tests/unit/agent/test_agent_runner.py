"""Unit tests for the MaterialAgentRunner (S3.5-M6)."""

import re
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from materials_screening.agent.conversation_store import SqliteConversationStore
from materials_screening.agent.errors import AgentConversationError
from materials_screening.agent.mock_model import (
    MockAgentTurn,
    MockMaterialAgentModel,
    MockToolCall,
)
from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentModel,
    MaterialAgentRequest,
)
from materials_screening.agent.runner import (
    ConversationView,
    MaterialAgentRunner,
)
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_registry import AgentToolRegistry
from materials_screening.agent_tools import (
    CompareRankedMaterialsTool,
    GetScreeningResultTool,
    GetWorkflowHistoryTool,
    GetWorkflowStatusTool,
    RunScreeningWorkflowTool,
)
from materials_screening.workflow.input_output import WorkflowInput, WorkflowOutput
from materials_screening.workflow.state import WorkflowStatus


def _tick_clock(start: datetime | None = None) -> Callable[[], datetime]:
    current = start or datetime(2026, 8, 6, 0, 0, tzinfo=UTC)

    def clock() -> datetime:
        nonlocal current
        value = current
        current = current.replace(second=current.second + 1)
        return value

    return clock


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


class _RecordingModel:
    """Wraps a model and records every request for history assertions."""

    def __init__(self, inner: MaterialAgentModel) -> None:
        self._inner = inner
        self.requests: list[MaterialAgentRequest] = []

    def generate(self, request: MaterialAgentRequest) -> object:
        self.requests.append(request)
        return self._inner.generate(request)


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


def _build(
    tmp_path: Path,
    *,
    model: MockMaterialAgentModel | None = None,
    settings: AgentSettings | None = None,
    store: SqliteConversationStore | None = None,
    checkpointer: object | None = None,
    recording: bool = False,
    **overrides: Any,
) -> tuple[
    MaterialAgentRunner,
    SqliteConversationStore,
    MockMaterialAgentModel,
    _RecordingModel | None,
]:
    resolved_settings = settings or _settings()
    resolved_store = store or SqliteConversationStore(
        tmp_path / "agent_conversations.sqlite",
        clock=_tick_clock(),
    )
    resolved_model = model or MockMaterialAgentModel(
        script=[MockAgentTurn(message="ok")]
    )
    wrapper = _RecordingModel(resolved_model) if recording else None
    agent_model: MaterialAgentModel = wrapper or resolved_model
    values: dict[str, Any] = {
        "settings": resolved_settings,
        "store": resolved_store,
        "workflow_runner": _FakeRunner(_workflow_output()),
        "workflow_result_reader": _FakeReader(),
        "tool_registry": _registry(),
        "agent_model": agent_model,
        "checkpointer": checkpointer or InMemorySaver(),
        "clock": _tick_clock(),
        "id_generator": _SeqIdGenerator(),
    }
    values.update(overrides)
    runner = MaterialAgentRunner(**values)
    return runner, resolved_store, resolved_model, wrapper


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


def _run_then_draft() -> MockMaterialAgentModel:
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
        # prepare consumes id_1/id_2; the run tool evidence is id_3.
        "evidence_ids": ["id_3"],
        "warnings": [],
        "follow_up_question": None,
    }
    return MockMaterialAgentModel(
        script=[
            MockAgentTurn(tool_calls=(run_call,)),
            MockAgentTurn.final_draft(draft),
        ]
    )


def _run_draft_then_concept() -> MockMaterialAgentModel:
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
        "evidence_ids": ["id_3"],
        "warnings": [],
        "follow_up_question": None,
    }
    return MockMaterialAgentModel(
        script=[
            MockAgentTurn(tool_calls=(run_call,)),
            MockAgentTurn.final_draft(draft),
            MockAgentTurn.final_draft(_concept_draft()),
        ]
    )


class TestStartConversation:
    def test_start_conversation_creates_and_returns_id(self, tmp_path: Path) -> None:
        runner, store, _, _ = _build(tmp_path)

        conversation_id = runner.start_conversation()

        assert re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,253}", conversation_id)
        assert store.exists(conversation_id) is True
        assert store.get(conversation_id).turn_count == 0

    def test_default_id_generator_produces_compact_ids(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(
            tmp_path,
            model=model,
            id_generator=None,
        )

        conversation_id = runner.start_conversation()

        assert re.fullmatch(r"[0-9a-f]{4}", conversation_id)
        assert store.exists(conversation_id) is True

    def test_recursion_limit_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            _build(tmp_path, recursion_limit=0)


class TestAskSingleTurn:
    def test_concept_answer_completes(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(tmp_path, model=model)

        result = runner.ask(message="介绍一下半导体", conversation_id="c1")

        assert result.status == "completed"
        assert result.response_text == "无机半导体是材料科学的重要领域。"
        assert result.model_call_count == 1
        assert result.user_turn_id
        assert result.error is None
        assert store.get("c1").turn_count == 1

    def test_ask_creates_missing_conversation(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(tmp_path, model=model)

        result = runner.ask(message="hi", conversation_id="new_c")

        assert result.status == "completed"
        assert store.exists("new_c") is True
        assert store.get("new_c").turn_count == 1

    def test_ask_without_conversation_starts_one(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(tmp_path, model=model)

        result = runner.ask(message="hi")

        assert result.status == "completed"
        assert store.exists(result.conversation_id) is True


class TestMultiTurn:
    def test_second_turn_receives_full_history(self, tmp_path: Path) -> None:
        model = _run_draft_then_concept()
        runner, store, _, recording = _build(
            tmp_path,
            model=model,
            recording=True,
        )
        assert recording is not None

        first = runner.ask(message="筛选半导体", conversation_id="c1")
        second = runner.ask(message="好的，请继续。", conversation_id="c1")

        assert first.status == "completed"
        assert second.status == "completed"
        assert store.get("c1").turn_count == 2
        last_request = recording.requests[-1]
        items = last_request.input_items
        contents = [item.model_dump(mode="json") for item in items]
        assert contents[0]["content"] == "筛选半导体"
        assert contents[-1]["content"] == "好的，请继续。"
        assert [item["type"] for item in contents] == [
            "message",
            "function_call",
            "function_call_output",
            "message",
            "message",
        ]

    def test_recovery_after_reopen_preserves_history(self, tmp_path: Path) -> None:
        store_path = tmp_path / "agent_conversations.sqlite"
        checkpoint_path = tmp_path / "agent_checkpoints.sqlite"

        store1 = SqliteConversationStore(store_path, clock=_tick_clock())
        connection1 = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        saver1 = SqliteSaver(connection1)
        runner1, _, _, _ = _build(
            tmp_path,
            model=_run_then_draft(),
            store=store1,
            checkpointer=saver1,
        )
        first = runner1.ask(message="筛选半导体", conversation_id="c1")
        assert first.status == "completed"
        connection1.close()
        store1.close()

        store2 = SqliteConversationStore(store_path, clock=_tick_clock())
        connection2 = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        saver2 = SqliteSaver(connection2)
        concept = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner2, _, _, recording = _build(
            tmp_path,
            model=concept,
            store=store2,
            checkpointer=saver2,
            recording=True,
        )

        second = runner2.ask(message="继续", conversation_id="c1")

        assert second.status == "completed"
        assert store2.get("c1").turn_count == 2
        assert store2.owns_workflow("c1", "run-1") is True
        assert store2.get("c1").active_workflow_thread_id == "run-1"
        assert recording is not None
        contents = [
            item.model_dump(mode="json") for item in recording.requests[-1].input_items
        ]
        assert contents[0]["content"] == "筛选半导体"
        assert contents[-1]["content"] == "继续"
        connection2.close()
        store2.close()


class TestConcurrency:
    def test_concurrent_ask_returns_busy(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(tmp_path, model=model)
        lock = store.conversation_lock("c1")
        assert lock.acquire(blocking=False) is True

        result = runner.ask(message="hi", conversation_id="c1")

        assert result.status == "error"
        assert result.error is not None
        assert result.error["code"] == "CONVERSATION_BUSY"
        assert model.call_count == 0
        assert store.exists("c1") is False
        lock.release()

    def test_different_conversations_are_not_blocked(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(tmp_path, model=model)
        lock = store.conversation_lock("c1")
        assert lock.acquire(blocking=False) is True

        result = runner.ask(message="hi", conversation_id="c2")

        assert result.status == "completed"
        lock.release()


class TestCancellation:
    def test_cancelled_before_start_yields_final_cancelled(
        self,
        tmp_path: Path,
    ) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, store, _, _ = _build(tmp_path, model=model)
        event = threading.Event()
        event.set()

        events = list(
            runner.ask_stream(
                message="hi",
                conversation_id="c1",
                cancel_event=event,
            )
        )

        final = events[-1]
        assert final.is_final is True
        assert final.result is not None
        assert final.result.status == "cancelled"
        assert "已停止" in final.result.response_text
        assert model.call_count == 0
        # The cancelled turn never started, so the conversation is not created.
        assert store.exists("c1") is False

    def test_cancel_between_model_and_tools_ends_cancelled(
        self,
        tmp_path: Path,
    ) -> None:
        run_call = MockToolCall(
            call_id="call_1",
            name="run_screening_workflow",
            arguments='{"query":"筛选半导体"}',
        )
        # A single-turn script: any second model call would raise, so reaching
        # the cancelled final state proves the loop stopped at the boundary.
        model = MockMaterialAgentModel(script=[MockAgentTurn(tool_calls=(run_call,))])
        workflow_runner = _FakeRunner(_workflow_output())
        runner, store, _, recording = _build(
            tmp_path,
            model=model,
            workflow_runner=workflow_runner,
            recording=True,
        )
        event = threading.Event()
        stream = runner.ask_stream(
            message="筛选半导体",
            conversation_id="c1",
            cancel_event=event,
        )
        next(stream)  # prepare_turn
        next(stream)  # call_agent_model: model decides to call the tool
        event.set()
        events = list(stream)

        assert recording is not None
        assert recording.requests[0].allow_tool_calls is True
        assert model.call_count == 1
        assert workflow_runner.calls == []
        final = events[-1]
        assert final.is_final is True
        assert final.result is not None
        assert final.result.status == "cancelled"
        assert final.result.selected_tools == ()
        # The cancelled turn is recorded; the lock is released (next ask works).
        assert store.get("c1").turn_count == 1

    def test_ask_after_cancelled_turn_is_not_busy(self, tmp_path: Path) -> None:
        run_call = MockToolCall(
            call_id="call_1",
            name="run_screening_workflow",
            arguments='{"query":"筛选半导体"}',
        )
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(tool_calls=(run_call,)),
                MockAgentTurn.final_draft(_concept_draft()),
                MockAgentTurn(tool_calls=(run_call,)),
                MockAgentTurn.final_draft(_concept_draft()),
            ]
        )
        runner, store, _, _ = _build(tmp_path, model=model)
        event = threading.Event()
        stream = runner.ask_stream(
            message="筛选半导体",
            conversation_id="c1",
            cancel_event=event,
        )
        next(stream)  # prepare_turn
        event.set()
        list(stream)  # cancelled final state

        result = runner.ask(message="hi", conversation_id="c1")

        assert result.status == "completed"
        assert store.get("c1").turn_count == 2


class TestErrors:
    def test_turn_limit(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner, _, _, _ = _build(
            tmp_path,
            model=model,
            settings=_settings(agent_max_conversation_turns=1),
        )

        first = runner.ask(message="hi", conversation_id="c1")
        second = runner.ask(message="hi", conversation_id="c1")

        assert first.status == "completed"
        assert second.status == "error"
        assert second.error is not None
        assert second.error["code"] == "TURN_LIMIT"
        assert model.call_count == 1

    def test_recursion_limit_mapped_to_safe_result(self, tmp_path: Path) -> None:
        loop_call = MockToolCall(
            call_id="loop_1",
            name="run_screening_workflow",
            arguments='{"query":"重复"}',
        )
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(tool_calls=(loop_call,))],
            repeat_turn=MockAgentTurn(tool_calls=(loop_call,)),
        )
        runner, _, _, _ = _build(
            tmp_path,
            model=model,
            settings=_settings(agent_max_model_calls_per_turn=10),
            recursion_limit=4,
        )

        result = runner.ask(message="重复执行", conversation_id="c1")

        assert result.status == "error"
        assert result.error is not None
        assert result.error["code"] == "RECURSION_LIMIT"
        assert result.response_text == "agent recursion limit reached"

    def test_recursion_error_with_failing_state_read(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        loop_call = MockToolCall(
            call_id="loop_1",
            name="run_screening_workflow",
            arguments='{"query":"重复"}',
        )
        model = MockMaterialAgentModel(
            script=[MockAgentTurn(tool_calls=(loop_call,))],
            repeat_turn=MockAgentTurn(tool_calls=(loop_call,)),
        )
        runner, _, _, _ = _build(
            tmp_path,
            model=model,
            settings=_settings(agent_max_model_calls_per_turn=10),
            recursion_limit=4,
        )

        def _broken_get_state(config: object) -> object:
            raise RuntimeError("boom")

        monkeypatch.setattr(runner._graph, "get_state", _broken_get_state)

        result = runner.ask(message="重复执行", conversation_id="c1")

        assert result.status == "error"
        assert result.error is not None
        assert result.error["code"] == "RECURSION_LIMIT"
        assert result.user_turn_id

    def test_empty_message_rejected(self, tmp_path: Path) -> None:
        runner, _, _, _ = _build(tmp_path)

        result = runner.ask(message="   ", conversation_id="c1")

        assert result.error is not None
        assert result.error["code"] == "EMPTY_MESSAGE"

    def test_model_error_maps_to_safe_result(self, tmp_path: Path) -> None:
        model = MockMaterialAgentModel(
            script=[
                MockAgentTurn(
                    status=AgentModelStatus.INCOMPLETE,
                    error="max_output_tokens reached",
                )
            ]
        )
        runner, _, _, _ = _build(tmp_path, model=model)

        result = runner.ask(message="筛选半导体", conversation_id="c1")

        assert result.status == "error"
        assert result.error is not None
        assert result.error["code"] == "MODEL_INCOMPLETE"
        assert result.response_text == "max_output_tokens reached"

    def test_validation_failure_maps_to_safe_result(self, tmp_path: Path) -> None:
        draft = _concept_draft()
        draft["answer"] = "筛选任务已完成。"
        draft["evidence_ids"] = ["ev_bogus"]
        model = MockMaterialAgentModel(script=[MockAgentTurn.final_draft(draft)])
        runner, _, _, _ = _build(tmp_path, model=model)

        result = runner.ask(message="筛选半导体", conversation_id="c1")

        assert result.status == "error"
        assert result.error is not None
        assert result.error["code"] == "FINAL_VALIDATION_FAILED"
        assert "筛选任务已完成。" not in result.response_text


class TestConversationView:
    def test_view_shows_turns_links_and_active(self, tmp_path: Path) -> None:
        runner, store, _, _ = _build(tmp_path, model=_run_then_draft())
        runner.ask(message="筛选半导体", conversation_id="c1")

        view = runner.get_conversation("c1")

        assert isinstance(view, ConversationView)
        assert view.conversation_id == "c1"
        assert view.turn_count == 1
        assert view.active_workflow_thread_id == "run-1"
        assert view.workflow_threads == ("run-1",)

    def test_view_missing_conversation_raises(self, tmp_path: Path) -> None:
        runner, _, _, _ = _build(tmp_path)

        with pytest.raises(AgentConversationError, match="not found") as exc:
            runner.get_conversation("missing")

        assert exc.value.code == "CONVERSATION_NOT_FOUND"

    def test_conversation_status_reads_last_terminal_status(
        self, tmp_path: Path
    ) -> None:
        runner, _, _, _ = _build(tmp_path)

        assert runner.conversation_status("c1") is None

        model = MockMaterialAgentModel(
            script=[MockAgentTurn.final_draft(_concept_draft())]
        )
        runner2, _, _, _ = _build(tmp_path, model=model)
        runner2.ask(message="hi", conversation_id="c1")

        assert runner2.conversation_status("c1") == "completed"


class TestSafeResult:
    def test_result_exposes_no_raw_artifacts(self, tmp_path: Path) -> None:
        runner, _, _, _ = _build(tmp_path, model=_run_then_draft())

        result = runner.ask(message="筛选半导体", conversation_id="c1")

        dumped = result.model_dump()
        assert "arguments" not in dumped
        assert "reasoning" not in dumped
        assert "raw_response" not in dumped
        assert "input_items" not in dumped
        assert "pending_tool_calls" not in dumped
        assert result.response_text == "筛选任务已完成。"
        assert result.selected_tools == ("run_screening_workflow",)
        assert result.evidence_ids == ("id_3",)
        assert result.tool_call_count == 1
        assert result.model_call_count == 2
        assert result.active_workflow_thread_id == "run-1"
