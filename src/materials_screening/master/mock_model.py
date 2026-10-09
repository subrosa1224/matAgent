"""Deterministic offline model for Master Agent delegation demos."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)

_OUTLIER_MARKERS = ("离群", "异常值", "异常检测", "离群检测", "outlier", "anomaly")
_LITERATURE_MARKERS = (
    "文献",
    "论文",
    "综述",
    "参考文献",
    "doi",
    "literature",
    "paper",
    "publication",
    "review article",
)
_DATA_ANALYSIS_MARKERS = (
    "dataset-",
    "csv",
    "xlsx",
    "json数据",
    "数据集",
    "数据质量",
    "缺失值",
    "重复值",
    "相关性",
    "显著性",
    "t检验",
    "anova",
    "实验数据",
    "数据清洗",
    "箱线图",
    "散点图",
    "热力图",
)
_DATABASE_MARKERS = (
    "材料",
    "筛选",
    "查询",
    "比较",
    "统计",
    "带隙",
    "密度",
    "形成能",
    "稳定",
    "化学式",
    "materials project",
    "material",
    "database",
    "band gap",
    "mp-",
)


class MasterMockAgentModel:
    """Route offline requests to registered sub-agents and summarize results."""

    def __init__(self) -> None:
        self._call_count = 0

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        self._call_count += 1
        if self._has_sub_agent_result(request):
            return self._final_from_sub_agent(request)

        query = self._last_user_message(request)
        if query:
            if self._is_data_analysis_query(query):
                return self._delegate(query, "delegate_to_data_analysis")
            if self._is_literature_query(query):
                return self._delegate(query, "delegate_to_literature")
            if self._is_database_query(query):
                return self._delegate(query, "delegate_to_materials_database")
            return self._final(
                "请说明需要查询材料数据库，还是检索与分析文献。",
                status="needs_user_input",
                follow_up_question="您希望使用材料数据库、文献与知识，还是数据分析？",
            )
        return self._final("（离线演示）请提供需要分析的材料问题。")

    @staticmethod
    def _is_outlier_query(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in _OUTLIER_MARKERS)

    @staticmethod
    def _is_literature_query(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in _LITERATURE_MARKERS)

    @staticmethod
    def _is_data_analysis_query(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in _DATA_ANALYSIS_MARKERS)

    @staticmethod
    def _is_database_query(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in _DATABASE_MARKERS) or any(
            marker in lowered for marker in _OUTLIER_MARKERS
        )

    def _delegate(self, task: str, name: str) -> MaterialAgentResponse:
        return self._response(
            (
                AgentFunctionCallItem(
                    call_id="mock_master_call_1",
                    name=name,
                    arguments=json.dumps({"task": task}, ensure_ascii=False),
                ),
            )
        )

    def _final_from_sub_agent(
        self,
        request: MaterialAgentRequest,
    ) -> MaterialAgentResponse:
        envelope = self._last_sub_agent_envelope(request)
        if envelope is None or envelope.get("status") != "ok":
            error = envelope.get("error") if envelope else None
            detail = str(error.get("message", "")) if isinstance(error, dict) else ""
            return self._final(
                "（离线演示）子 Agent 执行失败"
                + (f"：{detail}" if detail else "，请重试。"),
                status="error",
            )
        answer = str(
            envelope.get("response_text") or "（离线演示）子 Agent 未返回结果。"
        )
        return self._final(
            answer,
            evidence_ids=[str(x) for x in envelope.get("evidence_ids", [])],
            warnings=[str(x) for x in envelope.get("warnings", [])],
        )

    def _final(
        self,
        answer: str,
        *,
        status: str = "completed",
        evidence_ids: list[str] | None = None,
        warnings: list[str] | None = None,
        follow_up_question: str | None = None,
    ) -> MaterialAgentResponse:
        draft = {
            "status": status,
            "answer": answer,
            "referenced_material_ids": [],
            "evidence_ids": evidence_ids or [],
            "warnings": warnings or [],
            "follow_up_question": follow_up_question,
        }
        return self._response(
            (
                AgentMessageItem(
                    role="assistant", content=json.dumps(draft, ensure_ascii=False)
                ),
            )
        )

    def _response(
        self,
        output_items: tuple[AgentMessageItem | AgentFunctionCallItem, ...],
    ) -> MaterialAgentResponse:
        digest = hashlib.sha256(
            json.dumps(
                [item.model_dump(mode="json") for item in output_items],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:12]
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=output_items,
            request_id=f"mock-master-{digest}",
            provider="mock",
            model="mock-master-agent",
        )

    @staticmethod
    def _last_user_message(request: MaterialAgentRequest) -> str | None:
        for item in reversed(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                return item.content
        return None

    @staticmethod
    def _has_sub_agent_result(request: MaterialAgentRequest) -> bool:
        return any(
            isinstance(item, AgentFunctionOutputItem)
            for item in MasterMockAgentModel._current_turn_items(request)
        )

    @staticmethod
    def _last_sub_agent_envelope(
        request: MaterialAgentRequest,
    ) -> dict[str, Any] | None:
        for item in reversed(MasterMockAgentModel._current_turn_items(request)):
            if not isinstance(item, AgentFunctionOutputItem):
                continue
            try:
                payload = json.loads(item.output)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _current_turn_items(request: MaterialAgentRequest) -> tuple[Any, ...]:
        items = tuple(request.input_items)
        for index in range(len(items) - 1, -1, -1):
            item = items[index]
            if isinstance(item, AgentMessageItem) and item.role == "user":
                return items[index + 1 :]
        return items
