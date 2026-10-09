"""Render generic query finals solely from current-turn database tool results."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
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

from .models import NUMERIC_FIELDS, OUTPUT_FIELDS, GetQueryResultInput

_TOOLS = frozenset(
    {
        "search_materials",
        "get_query_result",
        "get_material_details",
        "describe_materials",
        "compare_materials",
    }
)
_QUERY_ID = re.compile(r"query-[A-Za-z0-9_-]+\Z")


def is_search_request(message: str) -> bool:
    if re.match(r"\s*(?:解释|说明|(?:what is|explain)\b)", message, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"筛选|查询|查找|寻找|\b(?:search|find|screen)\b", message, re.IGNORECASE
        )
    )


def previous_query_read(request: MaterialAgentRequest, message: str):
    """Exact repeat only: turn a validated historical reference into a NEW read.

    No fuzzy matching, merging queries, or accepting a model-written marker.
    The read must still succeed this turn before its evidence can be published.
    """
    if not request.allow_tool_calls or not any(
        tool.name == "get_query_result" for tool in request.tool_definitions
    ):
        return None
    if re.search(r"重新查询|刷新|最新|\b(?:refresh|latest)\b", message, re.IGNORECASE):
        return None
    indexes = [
        i
        for i, x in enumerate(request.input_items)
        if isinstance(x, AgentMessageItem) and x.role == "user"
    ]
    for pos in range(len(indexes) - 2, -1, -1):
        begin, end = indexes[pos], indexes[pos + 1]
        if request.input_items[begin].content != message:
            continue
        items = request.input_items[begin + 1 : end]
        calls = {
            x.call_id: x.name for x in items if isinstance(x, AgentFunctionCallItem)
        }
        payloads = []
        for x in items:
            if not isinstance(x, AgentFunctionOutputItem):
                continue
            name = calls.get(x.call_id)
            if name not in {"search_materials", "get_query_result"}:
                continue
            try:
                envelope = json.loads(x.output)
            except ValueError:
                continue
            if (
                not isinstance(envelope, dict)
                or envelope.get("status") != "ok"
                or envelope.get("tool_name", name) != name
            ):
                continue
            payload = envelope.get("output")
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("query_id"), str)
                and _QUERY_ID.fullmatch(payload["query_id"])
            ):
                payloads.append(payload)
        if not payloads:
            continue
        if len({x["query_id"] for x in payloads}) != 1:
            return None
        fields = list(
            dict.fromkeys(
                f
                for p in payloads
                for f in (
                    p["fields"]
                    if isinstance(p.get("fields"), list)
                    else [
                        key
                        for row in p.get("materials", [])
                        if isinstance(row, dict)
                        for key in row
                    ]
                )
                if isinstance(f, str) and f in OUTPUT_FIELDS
            )
        )
        args = GetQueryResultInput(
            query_id=payloads[-1]["query_id"],
            limit=100,
            fields=tuple(fields) or ("material_id", "formula_pretty"),
        )
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentFunctionCallItem(
                    call_id="repeat-query-read",
                    name="get_query_result",
                    arguments=args.model_dump_json(),
                ),
            ),
            request_id="database-repeat-read",
            provider="deterministic",
            model="current-turn-evidence-v1",
        )
    return None


def deliver_query_evidence(
    response: MaterialAgentResponse,
    items: Sequence[object],
    *,
    require_query: bool = False,
) -> MaterialAgentResponse:
    """Keep tool selection/errors intact; replace only successful query finals."""
    if response.status != AgentModelStatus.COMPLETED or response.tool_calls:
        return response
    try:
        draft = json.loads(response.message_text or "")
    except (ValueError, TypeError):
        return response
    if not isinstance(draft, dict) or draft.get("status") != "completed":
        return response
    calls = {
        item.call_id: item.name
        for item in items
        if isinstance(item, AgentFunctionCallItem)
    }
    evidence: list[tuple[str, dict[str, Any], str]] = []
    for item in items:
        if not isinstance(item, AgentFunctionOutputItem):
            continue
        name = calls.get(item.call_id)
        if name not in _TOOLS:
            # Do not swallow a requested export/outlier operation in this adapter.
            return response
        try:
            envelope = json.loads(item.output)
        except ValueError:
            return response
        if not isinstance(envelope, dict):
            return response
        if envelope.get("tool_name", name) != name:
            return response
        if envelope.get("status") != "ok":
            continue
        payload = envelope.get("output")
        evidence_id = envelope.get("evidence_id")
        if not isinstance(payload, dict) or not isinstance(evidence_id, str):
            return response
        evidence.append((name, payload, evidence_id))
    queries = [
        row for row in evidence if row[0] in {"search_materials", "get_query_result"}
    ]
    if not queries:
        if require_query:
            return _response(
                "本轮没有成功的数据库查询或快照读取证据，不能发布历史材料结果或继续交接。",
                [],
                [],
                ["需要本轮成功工具证据；未自动重试失败步骤。"],
                status="error",
            )
        return response
    query_values = [row[1].get("query_id") for row in queries]
    if any(not isinstance(q, str) or not _QUERY_ID.fullmatch(q) for q in query_values):
        return response
    query_ids = set(query_values)
    if len(query_ids) != 1:
        return _response(
            "本轮返回多个独立查询快照，不能自动合并或任选一个继续交接。",
            [],
            [row[2] for row in queries],
            ["请明确后续分析使用哪个查询快照。"],
            status="error",
        )
    query_id = next(iter(query_ids))
    latest = queries[-1][1]
    materials = latest.get("materials")
    if not isinstance(materials, list):
        return response
    matched_count = next(
        (
            payload["matched_count"]
            for _, payload, _ in reversed(queries)
            if "matched_count" in payload
        ),
        "未提供",
    )
    # Pagination has no fields/count metadata; use returned keys and original
    # search counts, never recompute a full-query count from a page.
    fields = latest.get("fields")
    if not isinstance(fields, list):
        fields = list(
            dict.fromkeys(
                key
                for row in materials
                if isinstance(row, dict)
                for key in row
                if key in OUTPUT_FIELDS
            )
        )
    warnings = [
        warning
        for _, payload, _ in evidence
        for warning in payload.get("warnings", [])
        if isinstance(warning, str)
    ]
    warnings.append(
        "数据库计算属性只支持初筛；不能据此推断容量、电压、循环性能或已实验合成。"
    )
    count_text = f"匹配 {matched_count} 条"
    if matched_count == "未提供" and isinstance(latest.get("total"), int):
        count_text = f"快照保存 {latest['total']} 条"
    if not any(name == "search_materials" for name, _, _ in queries):
        warnings.insert(0, "本轮读取已有查询快照，不代表重新获取数据库最新数据。")
    lines = [
        f"数据库查询快照：`{query_id}`。",
        f"{count_text}；本页工具返回 {len(materials)} 条，摘要最多展示20条。",
        "以下表格只复制工具返回字段，不根据展示页计算总体统计或性能排名。",
        *(
            warnings[:1]
            if not any(name == "search_materials" for name, _, _ in queries)
            else []
        ),
        "",
        f"MATERIAL_QUERY_HANDOFF: query_id={query_id}",
    ]
    snapshot = Path("data/material_queries") / query_id / "records.jsonl"
    if snapshot.is_file():
        lines.append(f"[完整查询记录]({snapshot.resolve().as_posix()})")
    material_ids: list[str] = []
    block = _materials_table(materials, fields, material_ids)
    lines.extend(("", _bounded(block, 3300)))
    for name, payload, _ in evidence:
        if name == "describe_materials" and payload.get("query_id") == query_id:
            lines.extend(("", _statistics_table(payload)))
        elif name in {"get_material_details", "compare_materials"}:
            rows = payload.get("materials")
            if isinstance(rows, list):
                block = _materials_table(rows, payload.get("fields", []), material_ids)
                lines.extend(("", "补充工具字段：", _bounded(block, 1000)))
    lines.extend(("", "结论边界：", warnings[-1]))
    return _response(
        _bounded("\n".join(lines), 7400),
        list(dict.fromkeys(material_ids)),
        list(dict.fromkeys(row[2] for row in evidence)),
        list(dict.fromkeys(warnings)),
    )


def _materials_table(rows: list[Any], fields: Any, ids: list[str]) -> str:
    rows = [row for row in rows if isinstance(row, dict)][:20]
    columns = ["material_id", "formula_pretty"]
    columns.extend(
        field
        for field in (fields if isinstance(fields, list) else [])
        if field in OUTPUT_FIELDS and field not in columns
    )
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        material_id = row.get("material_id")
        if isinstance(material_id, str) and re.fullmatch(
            r"mp-[A-Za-z0-9-]+", material_id
        ):
            ids.append(material_id)
        lines.append("| " + " | ".join(_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _statistics_table(payload: dict[str, Any]) -> str:
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        return "工具未返回可展示的统计字段。"
    keys = ("count", "missing", "mean", "std", "min", "median", "max")
    lines = [
        f"工具统计（分析 `{_cell(payload.get('analysis_id'))}`）：",
        "| 字段 | " + " | ".join(keys) + " |",
        "|" + "---|" * 8,
    ]
    for field, values in statistics.items():
        if field in NUMERIC_FIELDS and isinstance(values, dict):
            lines.append(
                "| "
                + field
                + " | "
                + " | ".join(_cell(values.get(k)) for k in keys)
                + " |"
            )
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")[:180]


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return (
        text[:limit].rsplit("\n", 1)[0] + "\n（摘要截短；完整记录保留在工具快照中。）"
    )


def _response(answer, material_ids, evidence_ids, warnings, *, status="completed"):
    return MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(
            AgentMessageItem(
                role="assistant",
                content=json.dumps(
                    {
                        "status": status,
                        "answer": answer,
                        "referenced_material_ids": material_ids,
                        "evidence_ids": evidence_ids,
                        "warnings": warnings,
                        "follow_up_question": None,
                    },
                    ensure_ascii=False,
                ),
            ),
        ),
        request_id="database-evidence-final",
        provider="deterministic",
        model="tool-evidence-delivery-v1",
    )
