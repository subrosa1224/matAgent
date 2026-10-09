"""Unit tests for the Gradio web UI (agent_ui_gradio.py).

The whole module is skipped when the optional ``web-ui`` extra is not
installed; the offline quality gate does not require gradio.
"""

import json
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("gradio")

import materials_screening.agent_ui_gradio as ui  # noqa: E402
from materials_screening.agent.models import AgentResult  # noqa: E402


class _FakeStreamEvent:
    """Duck-typed AgentStreamEvent; the UI only reads these four fields."""

    def __init__(
        self,
        *,
        is_final: bool = False,
        node: str = "",
        message: str = "",
        result: AgentResult | None = None,
    ) -> None:
        self.is_final = is_final
        self.node = node
        self.message = message
        self.result = result


class _FakeRunner:
    """Stands in for MaterialAgentRunner; records ask_stream arguments."""

    def __init__(self, result: AgentResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str | None, threading.Event | None]] = []

    def ask_stream(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        self.calls.append((message, conversation_id, cancel_event))
        yield _FakeStreamEvent(node="prepare_turn", message="正在准备...")
        yield _FakeStreamEvent(
            node="runner",
            message="完成",
            is_final=True,
            result=self.result,
        )


def _result(
    *,
    status: str = "completed",
    response_text: str = "完成",
    conversation_id: str = "conv_abc",
    error: dict[str, Any] | None = None,
) -> AgentResult:
    return AgentResult(
        conversation_id=conversation_id,
        user_turn_id="u1",
        status=status,
        response_text=response_text,
        error=error,
    )


def _install_fake(monkeypatch: pytest.MonkeyPatch, result: AgentResult) -> _FakeRunner:
    fake = _FakeRunner(result)
    monkeypatch.setattr(ui, "_global_runner", fake)
    monkeypatch.setattr(ui, "_conversation_id", None)
    monkeypatch.setattr(ui, "_turn_in_progress", False)
    monkeypatch.setattr(ui, "_cancel_event", None)
    return fake


class TestChatRespond:
    def test_reuses_same_conversation_across_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _install_fake(monkeypatch, _result())

        first = list(ui._chat_respond("寻找半导体", []))

        # The first message has no conversation yet; the result's id is kept.
        assert fake.calls[0][1] is None
        assert ui._conversation_id == "conv_abc"
        assert first[-1][0][-1]["content"] == "完成"

        history = first[-1][0]
        second = list(ui._chat_respond("为什么第一名更好？", history))

        assert fake.calls[1][1] == "conv_abc"
        assert second[-1][0][-1]["content"] == "完成"

    def test_passes_fresh_cancel_event_per_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _install_fake(monkeypatch, _result())

        list(ui._chat_respond("寻找半导体", []))

        event = fake.calls[0][2]
        assert isinstance(event, threading.Event)
        assert event.is_set() is False
        # The event is cleared after the turn finishes.
        assert ui._cancel_event is None
        assert ui._turn_in_progress is False

    def test_blocks_second_submit_while_busy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fake = _install_fake(monkeypatch, _result())
        monkeypatch.setattr(ui, "_turn_in_progress", True)

        outputs = list(ui._chat_respond("寻找半导体", []))

        history = outputs[0][0]
        assert "仍在处理中" in history[-1]["content"]
        assert fake.calls == []

    def test_requires_initialized_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ui, "_global_runner", None)

        outputs = list(ui._chat_respond("寻找半导体", []))

        assert "初始化" in outputs[0][0][-1]["content"]

    def test_cancelled_result_is_prefixed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _result(
            status="cancelled",
            response_text="已停止：不再执行后续步骤（进行中的步骤将完成）",
        )
        _install_fake(monkeypatch, result)

        outputs = list(ui._chat_respond("寻找半导体", []))

        assert "⏹️" in outputs[-1][0][-1]["content"]
        # A cancelled turn is still remembered so follow-ups keep working.
        assert ui._conversation_id == "conv_abc"

    def test_turn_limit_suggests_reset(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _result(
            status="error",
            response_text="会话达到 20 轮上限",
            error={"code": "TURN_LIMIT", "message": "limit", "retryable": False},
        )
        _install_fake(monkeypatch, result)

        outputs = list(ui._chat_respond("寻找半导体", []))

        content = outputs[-1][0][-1]["content"]
        assert "TURN_LIMIT" in content
        assert "重置" in content
        assert "新会话" in content


class TestEventHandlers:
    def test_on_stop_sets_cancel_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        event = threading.Event()
        monkeypatch.setattr(ui, "_cancel_event", event)

        out = ui._on_stop()

        assert event.is_set() is True
        assert "停止请求已发送" in (out or "")

    def test_on_stop_without_active_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ui, "_cancel_event", None)

        out = ui._on_stop()

        assert "没有正在进行的任务" in (out or "")

    def test_on_reset_starts_new_conversation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ui, "_conversation_id", "conv_abc")

        out = ui._on_reset()

        assert ui._conversation_id is None
        assert "新会话" in out[2]


class TestResultReader:
    def test_reads_valid_result_json(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(ui, "_run_root", tmp_path)
        run_dir = tmp_path / "run_1"
        run_dir.mkdir()
        (run_dir / "result.json").write_text(
            json.dumps({"retrieved_count": 5, "returned_count": 3}),
            encoding="utf-8",
        )

        payload = ui._read_result("run_1")

        assert payload == {"retrieved_count": 5, "returned_count": 3}

    def test_returns_none_for_missing_or_invalid(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(ui, "_run_root", tmp_path)

        assert ui._read_result("missing") is None

        run_dir = tmp_path / "run_1"
        run_dir.mkdir()
        (run_dir / "result.json").write_text("not json", encoding="utf-8")

        assert ui._read_result("run_1") is None


class TestErrorExplain:
    def test_model_error_maps_to_chinese(self) -> None:
        text = ui._explain_error(
            "MODEL_ERROR",
            "Intern agent final output is not valid JSON",
        )

        assert "模型输出异常" in text
        assert "重试" in text
        # The stable English code stays visible for debugging/bug reports.
        assert "MODEL_ERROR" in text

    def test_unknown_code_falls_back(self) -> None:
        text = ui._explain_error("WEIRD_CODE", "boom")

        assert "处理失败" in text
        assert "WEIRD_CODE" in text

    def test_thread_notice_reports_workflow_result(self) -> None:
        text = ui._explain_error("MODEL_ERROR", "x", "run-1")

        assert "data/workflow_runs/run-1" in text
        assert "已执行完成" in text

    def test_no_thread_notice_without_thread(self) -> None:
        text = ui._explain_error("MODEL_ERROR", "x")

        assert "workflow_runs" not in text

    def test_chat_respond_shows_chinese_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = _result(
            status="error",
            response_text="Intern agent final output is not valid JSON",
            error={
                "code": "MODEL_ERROR",
                "message": "Intern agent final output is not valid JSON",
                "retryable": False,
            },
        )
        _install_fake(monkeypatch, result)

        outputs = list(ui._chat_respond("帮我找钠离子电池正极候选", []))

        content = outputs[-1][0][-1]["content"]
        assert "模型输出异常" in content
        assert "MODEL_ERROR" in content


class TestProgress:
    def test_shows_conversation_id(self) -> None:
        html = ui._progress([], [], None, "conv_abc")

        assert "conv_abc" in html

    def test_placeholder_without_anything(self) -> None:
        html = ui._progress([], [], None)

        assert "等待执行" in html

    def test_card_wraps_sections(self) -> None:
        html = ui._progress(
            ["调用工具"],
            [{"icon": "🔍", "label": "工作流执行完成", "detail": "检索 5 → 返回 3"}],
            "run-1",
        )

        assert 'class="status-card"' in html
        assert "🔧 工具调用" in html
        assert "📊 中间结果" in html
        assert "thread: run-1" in html


class TestFlowchart:
    def test_step_states_marked(self) -> None:
        html = ui._flowchart(["call_agent_model"], ["prepare_turn"], None)

        assert 'class="flow-step done"' in html
        assert 'class="flow-step active"' in html
        assert html.count('class="flow-step ') == 6

    def test_error_step_marked(self) -> None:
        html = ui._flowchart([], ["prepare_turn"], "finalize_error")

        assert 'class="flow-step error"' in html
        assert "✕" in html

    def test_pending_steps_have_no_state(self) -> None:
        html = ui._flowchart([], [], None)

        assert 'class="flow-step "' in html
