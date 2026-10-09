"""UI-neutral streaming adapter for :class:`MasterAgentRunner`.

The adapter deliberately contains no domain keyword routing. It translates
runner events into a small presentation contract reusable by any frontend.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Event
from typing import Any

from pydantic import BaseModel, ConfigDict

from materials_screening.agent.models import AgentResult


class MasterUiUpdate(BaseModel):
    """One safe progress update for a user interface."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    status: str
    detail: str
    conversation_id: str | None = None
    delegated_agent: str | None = None
    is_final: bool = False
    result: AgentResult | None = None


def stream_master_request(
    runner: Any,
    *,
    message: str,
    conversation_id: str | None,
    artifact_refs: tuple[str, ...] = (),
    task_id: str | None = None,
) -> Iterator[MasterUiUpdate]:
    """Stream one request through the real Master runner."""

    delegated_agent: str | None = None
    resolved_conversation_id = conversation_id
    arguments: dict[str, Any] = {
        "message": message,
        "conversation_id": conversation_id,
    }
    if artifact_refs:
        arguments["artifact_refs"] = artifact_refs
    if task_id is not None:
        arguments["task_id"] = task_id
    cancel = Event()
    if getattr(getattr(runner, "_preview_processor", None), "auto_batches", False):
        arguments["cancel_event"] = cancel
    stream = runner.ask_stream(**arguments)
    try:
        for event in stream:
            result = event.result
            if result is not None:
                resolved_conversation_id = result.conversation_id
                delegated_agent = (
                    _delegated_agent(result.selected_tools) or delegated_agent
                )
            delegated_agent = (
                _delegated_agent_from_event(event.message) or delegated_agent
            )
            yield MasterUiUpdate(
                status=_event_status(event.node, event.is_final, result),
                detail=event.message,
                conversation_id=resolved_conversation_id,
                delegated_agent=delegated_agent,
                is_final=event.is_final,
                result=result,
            )
    finally:
        cancel.set()
        if hasattr(stream, "close"):
            stream.close()


def _event_status(node: str, is_final: bool, result: AgentResult | None) -> str:
    if is_final:
        if result is not None and result.status == "cancelled":
            return "已停止"
        if result is not None and result.final_status == "needs_user_input":
            return "需要补充信息"
        if result is not None and (result.status == "error" or result.error):
            return "执行失败"
        return "已完成"
    return {
        "prepare_turn": "准备请求",
        "call_master_model": "判断意图",
        "execute_sub_agent": "子 Agent 执行中",
        "validate_final": "校验回答",
        "finalize_success": "整理结果",
        "finalize_error": "处理错误",
        "fulltext_preview": "全文预览",
    }.get(node, "运行中")


def _delegated_agent(selected_tools: tuple[str, ...]) -> str | None:
    for tool in selected_tools:
        if tool in {"delegate_to_materials_database", "materials_database"}:
            return "materials_database"
        if tool in {"delegate_to_literature", "literature"}:
            return "literature"
        if tool in {"delegate_to_data_analysis", "data_analysis"}:
            return "data_analysis"
    return None


def _delegated_agent_from_event(message: str) -> str | None:
    lowered = message.lower()
    if "materials database" in lowered or "materials_database" in lowered:
        return "materials_database"
    if "literature" in lowered:
        return "literature"
    if "data analysis" in lowered or "data_analysis" in lowered:
        return "data_analysis"
    return None
