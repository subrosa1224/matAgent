"""Deterministic offline mock of the agent model (S3.5).

The mock never calls Intern or any network service. Responses come from an
ordered script; an optional repeat turn simulates a looping model until the
graph's per-turn model call limit stops it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from materials_screening.agent.errors import AgentModelError
from materials_screening.agent.model_base import (
    AgentModelOutputItem,
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
    AgentTranscriptItem,
)


@dataclass(frozen=True)
class MockToolCall:
    """One scripted function call emitted by the mock model."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class MockAgentTurn:
    """One scripted model generation turn."""

    status: AgentModelStatus = AgentModelStatus.COMPLETED
    message: str | None = None
    tool_calls: tuple[MockToolCall, ...] = ()
    error: str | None = None
    input_tokens: int = 10
    output_tokens: int = 20
    reasoning_tokens: int = 0
    latency_ms: int = 0

    def __post_init__(self) -> None:
        if (
            self.status is not AgentModelStatus.COMPLETED
            and not (self.error or "").strip()
        ):
            raise ValueError("incomplete/failed turn requires an error summary")
        if (
            self.status is AgentModelStatus.COMPLETED
            and not self.message
            and not (self.tool_calls)
        ):
            raise ValueError("completed turn requires a message or a tool call")

    @staticmethod
    def final_draft(draft: dict[str, Any], **overrides: Any) -> MockAgentTurn:
        """Build a completed turn whose message is compact JSON of a draft."""
        message = json.dumps(
            draft,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return MockAgentTurn(message=message, **overrides)


class MockMaterialAgentModel:
    """Scripted offline model for all normal agent graph tests.

    ``script`` is consumed one turn per ``generate`` call. When the script is
    exhausted, ``repeat_turn`` (if provided) is returned forever, which is how
    a looping model is simulated. Usage counters are fixed per turn; latency is
    reported and only actually slept when ``sleep_for_latency`` is enabled.
    """

    def __init__(
        self,
        *,
        script: Sequence[MockAgentTurn] | None = None,
        repeat_turn: MockAgentTurn | None = None,
        provider: str = "mock",
        model: str = "mock-material-agent",
        sleep_for_latency: bool = False,
    ) -> None:
        if script is None and repeat_turn is None:
            raise ValueError("provide at least a script or a repeat turn")
        self._script = tuple(script or ())
        self._repeat_turn = repeat_turn
        self._provider = provider
        self._model = model
        self._sleep_for_latency = sleep_for_latency
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of generate calls made so far."""
        return self._call_count

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        """Return the next scripted turn; never performs real model calls."""
        self._validate_request(request)
        self._call_count += 1
        turn = self._turn_for_call(self._call_count)
        if self._sleep_for_latency and turn.latency_ms > 0:
            time.sleep(turn.latency_ms / 1000.0)
        return MaterialAgentResponse(
            status=turn.status,
            output_items=self._build_output_items(turn),
            request_id=self._request_id(turn),
            provider=self._provider,
            model=self._model,
            latency_ms=turn.latency_ms,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            reasoning_tokens=turn.reasoning_tokens,
            error=turn.error,
        )

    def _validate_request(self, request: MaterialAgentRequest) -> None:
        for item in request.input_items:
            if not isinstance(item, AgentTranscriptItem):
                raise AgentModelError(
                    f"unsupported input item type: {type(item).__name__}"
                )

    def _turn_for_call(self, call_index: int) -> MockAgentTurn:
        if call_index <= len(self._script):
            return self._script[call_index - 1]
        if self._repeat_turn is not None:
            return self._repeat_turn
        raise AgentModelError(f"mock script exhausted after {len(self._script)} turns")

    def _build_output_items(
        self,
        turn: MockAgentTurn,
    ) -> tuple[AgentModelOutputItem, ...]:
        items: list[AgentModelOutputItem] = []
        if turn.message is not None:
            items.append(AgentMessageItem(role="assistant", content=turn.message))
        for call in turn.tool_calls:
            items.append(
                AgentFunctionCallItem(
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                )
            )
        return tuple(items)

    def _request_id(self, turn: MockAgentTurn) -> str:
        payload = json.dumps(
            {
                "call": self._call_count,
                "status": turn.status.value,
                "message": turn.message,
                "tool_calls": [
                    {"call_id": c.call_id, "name": c.name, "arguments": c.arguments}
                    for c in turn.tool_calls
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"mock-{digest}"


_SCREENING_INTENT_MARKERS = (
    "筛选",
    "寻找",
    "查找",
    "有哪些",
    "找出",
    "检索",
    "搜索",
    "带隙",
    "band gap",
    "禁带",
    "能带",
    "hull",
    "凸包",
    "能量高于",
    "密度",
    "density",
    "稳定",
    "stable",
    "半导体",
    "semiconductor",
    "材料",
    "materials",
    "物质",
    "含",
    "不含",
    "元素",
    "element",
    "比较",
    "compare",
    "钙钛矿",
    "perovskite",
    "氧化物",
    "oxide",
    "硫化物",
    "氮化物",
    "卤化物",
    "光伏",
    "太阳能",
    "电池",
    "光催化",
    "热电",
    "磁性",
    "透明导电",
    "透明电极",
    "发光",
    "红外",
    "立方",
    "晶系",
    "空间群",
    "spacegroup",
    "chemsys",
    "正极",
    "负极",
    "电解质",
    "二维",
    "迁移率",
    "热导率",
    "导热",
    "直接带隙",
    "n 型",
    "p 型",
    "轻质",
    "结构材料",
    "薄膜",
    "单晶",
)


class WorkflowDrivenMockAgentModel:
    """Offline demo model that drives the real (mock) screening workflow.

    Used by the CLI/UI in mock mode so offline demos answer with genuine
    workflow results instead of a canned reply. The first call emits a
    ``run_screening_workflow`` tool call carrying the last user message when
    the message looks like a screening request; otherwise it answers directly
    without calling any tool (concept questions stay conversational). Every
    later call parses the tool's ``ToolResultEnvelope`` from the transcript and
    composes a deterministic final draft from the actual workflow output
    (counts, top candidates, evidence ids). Zero network; deterministic given
    the same transcript. Only ids present in the tool evidence are cited, so
    the final validator accepts the draft.
    """

    _TOOL_NAME = "run_screening_workflow"
    _CALL_ID = "mock_call_1"
    _CONCEPT_ANSWER = (
        "（离线演示）当前为离线演示模式：材料类查询会基于内置示例数据执行"
        "完整流程并给出真实候选；切换到 intern 提供商后可获得完整对话能力。"
    )

    def __init__(
        self,
        *,
        provider: str = "mock",
        model: str = "mock-material-agent-workflow",
    ) -> None:
        self._provider = provider
        self._model = model
        self._call_count = 0

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        self._call_count += 1
        if self._turn_has_tool_result(request):
            # The current turn already executed the tool; compose the final
            # draft from its result. Call counts accumulate across turns, so
            # behaviour is decided by the transcript, not by call ordering.
            return self._draft_turn(request)
        query = self._last_user_message(request)
        if not query or not self._is_screening_query(query):
            # Concept/chit-chat question: answer directly, never call a
            # tool; the demo workflow only makes sense for screening asks.
            return self._draft(status="completed", answer=self._CONCEPT_ANSWER)
        return self._tool_call_turn(query)

    @staticmethod
    def _turn_has_tool_result(request: MaterialAgentRequest) -> bool:
        """True when the last user message is followed by a tool output.

        A user message starts a new turn; anything after it (assistant tool
        call plus its output) belongs to that turn.
        """
        last_user_index = -1
        for index, item in enumerate(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                last_user_index = index
        if last_user_index < 0:
            return False
        for item in request.input_items[last_user_index + 1 :]:
            if isinstance(item, AgentFunctionOutputItem):
                return True
        return False

    @staticmethod
    def _is_screening_query(query: str) -> bool:
        """Heuristic screening intent; keeps the offline demo from firing the
        workflow on concept questions while covering the quick-sample asks."""
        lowered = query.lower()
        return any(marker in lowered for marker in _SCREENING_INTENT_MARKERS)

    def _tool_call_turn(self, query: str) -> MaterialAgentResponse:
        item = AgentFunctionCallItem(
            call_id=self._CALL_ID,
            name=self._TOOL_NAME,
            arguments=json.dumps(
                {"query": query},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        return self._response((item,))

    def _draft_turn(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        envelope = self._last_tool_envelope(request)
        if envelope is None:
            return self._draft(
                status="error",
                answer="（离线演示）未获得工具结果，请重试。",
            )
        if envelope.get("status") == "error":
            detail = self._error_detail(envelope.get("error"))
            return self._draft(status="error", answer=f"（离线演示）{detail}")
        output = envelope.get("output")
        if not isinstance(output, dict):
            return self._draft(
                status="error",
                answer="（离线演示）工具返回了无法解析的结果。",
            )
        workflow_status = str(output.get("status", ""))
        evidence_id = str(output.get("evidence_id") or "")
        evidence = (evidence_id,) if evidence_id else ()
        thread_id = output.get("thread_id")
        thread_id = str(thread_id) if isinstance(thread_id, str) and thread_id else None
        if workflow_status in {"completed", "no_results"}:
            candidates = output.get("top_candidates")
            rows = (
                [c for c in candidates if isinstance(c, dict)]
                if isinstance(candidates, list)
                else []
            )
            referenced = tuple(
                str(c["material_id"])
                for c in rows
                if isinstance(c.get("material_id"), str) and c["material_id"]
            )
            if workflow_status == "completed" and rows:
                answer = self._completed_answer(output, rows)
            else:
                answer = "（离线演示）未找到符合条件的材料。"
            return self._draft(
                status="completed",
                answer=answer,
                thread_id=thread_id,
                referenced=referenced,
                evidence=evidence,
            )
        if workflow_status == "needs_clarification":
            question = output.get("clarification_question")
            question = (
                str(question)
                if isinstance(question, str) and question
                else "请补充筛选条件（如带隙范围、元素组成等）。"
            )
            return self._draft(
                status="needs_user_input",
                answer="（离线演示）需要补充筛选条件。",
                follow_up=question,
            )
        warnings = output.get("warnings")
        extra = (
            "；".join(str(w) for w in warnings[:2])
            if isinstance(warnings, list) and warnings
            else ""
        )
        return self._draft(
            status="error",
            answer=(
                f"（离线演示）工作流未完成：{workflow_status}"
                + (f"。{extra}" if extra else "")
            ),
            thread_id=thread_id,
            evidence=evidence,
        )

    @staticmethod
    def _completed_answer(output: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        lines = [
            "（离线演示）筛选完成："
            f"检索 {output.get('retrieved_count', 0)} → "
            f"过滤 {output.get('filtered_count', 0)} → "
            f"返回 {output.get('returned_count', 0)} 条。",
            "推荐候选：",
        ]
        for row in rows:
            parts = [f"{int(row.get('rank', 0))}. {row.get('material_id', '')}"]
            formula = str(row.get("formula_pretty", "") or "")
            if formula:
                parts.append(formula)
            if row.get("band_gap_ev") is not None:
                parts.append(f"带隙 {row['band_gap_ev']} eV")
            if row.get("energy_above_hull_ev_atom") is not None:
                parts.append(f"hull {row['energy_above_hull_ev_atom']} eV/atom")
            if row.get("density_g_cm3") is not None:
                parts.append(f"密度 {row['density_g_cm3']} g/cm³")
            parts.append(f"分数 {row.get('total_score', 0.0)}")
            lines.append("；".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _error_detail(error: Any) -> str:
        if isinstance(error, dict):
            code = str(error.get("code", "ERROR"))
            message = str(error.get("message", "工具执行失败"))
            return f"{code}: {message}"
        return "工具执行失败"

    def _draft(
        self,
        *,
        status: str,
        answer: str,
        thread_id: str | None = None,
        referenced: tuple[str, ...] = (),
        evidence: tuple[str, ...] = (),
        follow_up: str | None = None,
    ) -> MaterialAgentResponse:
        draft = {
            "status": status,
            "answer": answer,
            "active_workflow_thread_id": thread_id,
            "referenced_material_ids": list(referenced),
            "evidence_ids": list(evidence),
            "warnings": [],
            "follow_up_question": follow_up,
        }
        item = AgentMessageItem(
            role="assistant",
            content=json.dumps(
                draft,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        return self._response((item,))

    def _response(
        self,
        output_items: tuple[AgentMessageItem | AgentFunctionCallItem, ...],
    ) -> MaterialAgentResponse:
        payload = json.dumps(
            {
                "call": self._call_count,
                "provider": self._provider,
                "model": self._model,
                "output": [item.model_dump(mode="json") for item in output_items],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=output_items,
            request_id=f"mock-{digest}",
            provider=self._provider,
            model=self._model,
        )

    @staticmethod
    def _last_user_message(request: MaterialAgentRequest) -> str | None:
        for item in reversed(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                return item.content
        return None

    @staticmethod
    def _last_tool_envelope(request: MaterialAgentRequest) -> dict[str, Any] | None:
        """Parse the latest tool envelope of the current turn only."""
        last_user_index = -1
        for index, item in enumerate(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                last_user_index = index
        for item in reversed(request.input_items[last_user_index + 1 :]):
            if not isinstance(item, AgentFunctionOutputItem):
                continue
            try:
                payload = json.loads(item.output)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None
