"""Deterministic offline model for LiteratureAgent tests and mock CLI mode."""

from __future__ import annotations

import json

from materials_screening.agent.intern_model import InternAgentModel
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


class LiteratureMockModel:
    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        outputs = [
            item
            for item in request.input_items
            if isinstance(item, AgentFunctionOutputItem)
        ]
        if outputs:
            envelope = json.loads(outputs[-1].output)
            output = envelope.get("output", envelope)
            evidence = envelope.get("evidence_id") or output.get("evidence_id")
            answer = _summarize_output(output)
            content = {
                "status": "completed",
                "answer": answer,
                "referenced_material_ids": [],
                "evidence_ids": [evidence] if evidence else [],
                "warnings": list(output.get("warnings", [])),
            }
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentMessageItem(
                        role="assistant",
                        content=json.dumps(content, ensure_ascii=False),
                    ),
                ),
                request_id="mock-literature-final",
                provider="mock",
                model="mock",
            )
        topic = "materials science"
        for item in reversed(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                topic = item.content[:1000]
                break
        name = "openalex_search"
        arguments = json.dumps({"topic": topic, "max_papers": 20})
        if "候选池文献预检" in topic:
            name = "screen_candidate_literature"
            arguments = InternAgentModel._normalize_tool_arguments(request, name, "{}")
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentFunctionCallItem(
                    call_id="mock-lit-search",
                    name=name,
                    arguments=arguments,
                ),
            ),
            request_id="mock-literature-call",
            provider="mock",
            model="mock",
        )


def _summarize_output(output: dict[str, object]) -> str:
    finalists = output.get("finalists")
    if isinstance(finalists, list):
        return InternAgentModel._candidate_screen_answer(output)
    query_id = output.get("query_id", "—")
    papers = output.get("papers")
    paper_rows = papers if isinstance(papers, list) else []
    lines = [f"离线文献检索已完成，查询快照为 `{query_id}`。"]
    if not paper_rows:
        lines.append(
            "当前离线文献源未返回候选论文，因此本轮只能确认文献 Agent 已被调用，"
            "不能据此评价任何候选材料的紫外探测可行性。"
        )
    else:
        lines.append(f"共返回 {len(paper_rows)} 篇候选论文：")
        for index, paper in enumerate(paper_rows[:10], 1):
            if not isinstance(paper, dict):
                continue
            lines.append(
                f"{index}. {paper.get('title', '题名未知')}"
                f"（{paper.get('year', '年份未知')}）"
            )
            grade = paper.get("application_evidence_grade")
            reason = paper.get("application_evidence_reason")
            if grade:
                lines.append(f"   紫外应用证据：{grade} 级；{reason or '理由未提供'}")
    warnings = output.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append("注意：" + "；".join(str(item) for item in warnings))
    return "\n".join(lines)
