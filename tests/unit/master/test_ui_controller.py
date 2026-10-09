"""Tests for the UI-neutral MasterAgent streaming adapter."""

from __future__ import annotations

from materials_screening.agent.models import AgentResult
from materials_screening.master.master_runner import AgentStreamEvent
from materials_screening.master.ui_controller import stream_master_request


class _Runner:
    def ask_stream(self, *, message: str, conversation_id: str | None):
        assert message == "检索论文"
        assert conversation_id == "conversation-1"
        yield AgentStreamEvent(
            node="call_master_model",
            message="Master 正在分析意图...",
        )
        yield AgentStreamEvent(
            node="runner",
            message="完成",
            is_final=True,
            result=AgentResult(
                conversation_id="conversation-1",
                user_turn_id="turn-1",
                status="completed",
                final_status="completed",
                response_text="文献结果",
                selected_tools=("delegate_to_literature",),
            ),
        )


def test_stream_master_request_preserves_conversation_and_delegate() -> None:
    updates = list(
        stream_master_request(
            _Runner(), message="检索论文", conversation_id="conversation-1"
        )
    )
    assert updates[0].status == "判断意图"
    assert updates[-1].is_final is True
    assert updates[-1].conversation_id == "conversation-1"
    assert updates[-1].delegated_agent == "literature"
    assert updates[-1].result is not None
    assert updates[-1].result.response_text == "文献结果"
