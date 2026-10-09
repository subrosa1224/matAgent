"""Deterministic execution for complete, common database filter requests."""

from __future__ import annotations

import json
import re

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentModel,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.sub_agents.materials_database.evidence_delivery import (
    _response,
    deliver_query_evidence,
    is_search_request,
    previous_query_read,
)
from materials_screening.sub_agents.materials_database.formula_screening import (
    parse_formula_screening,
)
from materials_screening.sub_agents.materials_database.models import (
    FilterOperator,
    PropertyFilter,
    SearchMaterialsInput,
    SortRule,
)
from materials_screening.sub_agents.materials_database.multi_query_handoff import (
    deliver_multi_query_handoff,
    explicit_multi_query_calls,
)
from materials_screening.sub_agents.materials_database.screening_handoff import (
    SCREENING_HANDOFF_SCOPE,
    scoped_screening_handoff,
)

_STANDARD_OXIDE = re.compile(
    r"(?:筛选|查询|寻找|查找).*(?:带隙|band\s*gap)\s*(?:大于|>|高于)\s*"
    r"(?P<gap>\d+(?:\.\d+)?)\s*(?:eV)?.*(?:稳定).*(?:氧化物)",
    re.IGNORECASE,
)
_STANDARD_OXIDE_REVERSED = re.compile(
    r"(?:筛选|查询|寻找|查找).*(?:稳定).*(?:氧化物).*"
    r"(?:带隙|band\s*gap)\s*(?:大于|>|高于)\s*"
    r"(?P<gap>\d+(?:\.\d+)?)\s*(?:eV)?",
    re.IGNORECASE,
)
_BAND_GAP_RANGE = re.compile(
    r"(?:带隙|band\s*gap)\s*(?:为|在)?\s*"
    r"(?P<minimum>\d+(?:\.\d+)?)\s*(?:[~\-–—～〜]|至|到)\s*"
    r"(?P<maximum>\d+(?:\.\d+)?)\s*(?:eV)?",
    re.IGNORECASE,
)
_HULL_MAXIMUM = re.compile(
    r"(?:凸包能|energy[_\s-]*above[_\s-]*hull)\s*"
    r"(?:不高于|不大于|小于等于|<=|≤)\s*"
    r"(?P<maximum>\d+(?:\.\d+)?)\s*(?:eV\s*/\s*atom)?",
    re.IGNORECASE,
)
_EXCLUDED_ELEMENTS = re.compile(
    r"(?:不含|排除|without|exclude)\s*"
    r"(?P<elements>(?:[A-Z][a-z]?\s*[,、/\s]*){1,12})",
    re.IGNORECASE,
)

# Materials Project does not expose an "oxygen is the sole anion" switch.  For
# an ordinary oxide request, exclude common competing anion-forming elements so
# fluorides, chlorides, nitrides, sulfides, carbonates, etc. do not leak in.
_NON_OXIDE_ANION_ELEMENTS = (
    "H",
    "C",
    "N",
    "F",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
    "Te",
    "At",
)
_OXIDE_SCOPE_WARNING = (
    "氧化物范围采用元素排除近似：排除 "
    + "、".join(_NON_OXIDE_ANION_ELEMENTS)
    + "，不是严格的氧化态或结构分类；可能排除这些元素的有效氧化物，"
    "也不能保证排除所有含氧盐。需要这些材料时应明确类别并另行检索。"
)


def _nonmetal_filter(message: str) -> tuple[PropertyFilter, ...]:
    if any(
        marker in message.casefold()
        for marker in (
            "非金属",
            "半导体",
            "nonmetal",
            "non-metal",
            "semiconductor",
        )
    ) or re.search(r"is_metal\s*=\s*false", message, re.IGNORECASE):
        return (
            PropertyFilter(field="is_metal", operator=FilterOperator.EQ, value=False),
        )
    return ()


class DeterministicDatabaseModel:
    """Wrap a model and bypass it only for fully specified safe filters."""

    def __init__(self, delegate: MaterialAgentModel) -> None:
        self._delegate = delegate

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        handoff = scoped_screening_handoff(request)
        if handoff is not None:
            return handoff
        user_message, current_items = _current_turn(request)
        multi_calls = explicit_multi_query_calls(user_message)
        if multi_calls is not None:
            if not current_items and request.allow_tool_calls:
                return MaterialAgentResponse(
                    status=AgentModelStatus.COMPLETED,
                    output_items=multi_calls,
                    request_id="explicit-multi-query",
                    provider="deterministic",
                    model="explicit-multi-formula-v1",
                )
            seed = MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentMessageItem(
                        role="assistant", content='{"status":"completed"}'
                    ),
                ),
                request_id="explicit-multi-handoff",
                provider="deterministic",
                model="explicit-multi-formula-v1",
            )
            verified = deliver_multi_query_handoff(seed, current_items, user_message)
            if verified is not None:
                return verified
            return _response(
                "数据库初筛尚未通过完整性和筛选条件核验，暂不能继续候选分析与文献检索。",
                [],
                [],
                ["查询交接未完成；未放宽筛选条件。"],
                status="error",
            )
        formula_search = (
            parse_formula_screening(user_message[: -len(SCREENING_HANDOFF_SCOPE)])
            if user_message.endswith(SCREENING_HANDOFF_SCOPE)
            else None
        )
        if formula_search is not None:
            if not current_items and request.allow_tool_calls:
                return MaterialAgentResponse(
                    status=AgentModelStatus.COMPLETED,
                    output_items=(
                        AgentFunctionCallItem(
                            call_id="scoped-formula-search",
                            name="search_materials",
                            arguments=formula_search.model_dump_json(),
                        ),
                    ),
                    request_id="scoped-formula-call",
                    provider="deterministic",
                    model="explicit-formula-filter-v1",
                )
            # A successful tool call is insufficient: handoff above must have
            # verified parameters, paired evidence, metadata and ALL stored rows.
            # Do not send an unverified snapshot to later agents or retry blindly.
            return _response(
                "本步未通过筛选条件与查询快照核验，未将结果交给后续分析。"
                "查询失败、没有候选、证据缺失或条件不一致时不会放宽原条件。",
                [],
                [],
                ["筛选步骤未完成；不得据此宣称完整任务完成。"],
                status="error",
            )
        search = parse_standard_search(user_message)
        if search is None:
            requires_query = is_search_request(user_message)
            if requires_query and not current_items:
                read = previous_query_read(request, user_message)
                if read is not None:
                    return read
            if requires_query and any(
                isinstance(item, AgentFunctionCallItem)
                and item.call_id == "repeat-query-read"
                and item.name == "get_query_result"
                for item in current_items
            ):
                # This adapter owns the exact-repeat read. Finish from its NEW
                # tool evidence, not an optional model-generated export question.
                # Errors remain errors; generic model clarifications are untouched.
                seed = MaterialAgentResponse(
                    status=AgentModelStatus.COMPLETED,
                    output_items=(
                        AgentMessageItem(
                            role="assistant", content='{"status":"completed"}'
                        ),
                    ),
                    request_id="repeat-query-final",
                    provider="deterministic",
                    model="current-turn-evidence-v1",
                )
                return deliver_query_evidence(seed, current_items, require_query=True)
            response = self._delegate.generate(request)
            cohort = deliver_multi_query_handoff(response, current_items, user_message)
            if cohort is not None:
                return cohort
            return deliver_query_evidence(
                response,
                current_items,
                require_query=requires_query,
            )
        outputs = [
            item for item in current_items if isinstance(item, AgentFunctionOutputItem)
        ]
        if not outputs:
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentFunctionCallItem(
                        call_id="deterministic-oxide-search",
                        name="search_materials",
                        arguments=search.model_dump_json(),
                    ),
                ),
                request_id="deterministic-oxide-call",
                provider="deterministic",
                model="structured-filter-v1",
            )
        if "BAND_GAP_EXPLORATION_POOL" in user_message:
            supplemental = _band_gap_exploration_search(search)
            if supplemental is not None and len(outputs) == 1:
                first = json.loads(outputs[0].output)
                if first.get("status") == "ok":
                    return MaterialAgentResponse(
                        status=AgentModelStatus.COMPLETED,
                        output_items=(
                            AgentFunctionCallItem(
                                call_id="deterministic-exploration-search",
                                name="search_materials",
                                arguments=supplemental.model_dump_json(),
                            ),
                        ),
                        request_id="deterministic-exploration-call",
                        provider="deterministic",
                        model="structured-filter-v1",
                    )
            if len(outputs) >= 2:
                original = _final_from_tool(outputs[0], search)
                content = json.loads(original.message_text)
                supplement = json.loads(outputs[-1].output)
                if supplement.get("status") == "ok":
                    query_id = (supplement.get("output") or {}).get("query_id")
                    content["answer"] += (
                        "\n\n独立带隙探索池：保留凸包能与禁用元素条件，"
                        "另查 0 < 计算带隙 ≤ 原上限的二元氧化物。"
                        "不使用固定带隙校正，不并入原严格达标名单或统计；"
                        "实验光学带隙仍需全文核验。\n"
                        f"SUPPLEMENTARY_QUERY_HANDOFF: query_id={query_id}"
                    )
                    evidence_id = supplement.get("evidence_id")
                    if evidence_id:
                        content["evidence_ids"].append(str(evidence_id))
                else:
                    content["warnings"].append(
                        "独立带隙探索池查询失败；原严格结果仍保留。"
                    )
                return MaterialAgentResponse(
                    status=AgentModelStatus.COMPLETED,
                    output_items=(
                        AgentMessageItem(
                            role="assistant",
                            content=json.dumps(content, ensure_ascii=False),
                        ),
                    ),
                    request_id="deterministic-two-pools-final",
                    provider="deterministic",
                    model="structured-filter-v1",
                )
        return _final_from_tool(outputs[-1], search)


def _band_gap_exploration_search(
    search: SearchMaterialsInput,
) -> SearchMaterialsInput | None:
    if not any(
        item.field == "band_gap_ev" and item.operator == FilterOperator.GTE
        for item in search.filters
    ):
        return None
    filters = tuple(
        item
        for item in search.filters
        if not (
            item.field == "band_gap_ev"
            and item.operator in {FilterOperator.GTE, FilterOperator.GT}
        )
    )
    return search.model_copy(
        update={
            "num_elements": 2,
            "filters": (
                *filters,
                PropertyFilter(
                    field="band_gap_ev", operator=FilterOperator.GT, value=0
                ),
                *(
                    (
                        PropertyFilter(
                            field="is_metal", operator=FilterOperator.EQ, value=False
                        ),
                    )
                    if not any(
                        item.field == "is_metal" and item.value is False
                        for item in filters
                    )
                    else ()
                ),
            ),
        }
    )


def parse_standard_search(message: str) -> SearchMaterialsInput | None:
    """Parse one unambiguous oxide filter; return None for all other requests."""

    # Do not override an explicitly broader/mixed-anion material scope with
    # the ordinary oxide approximation; let the delegate interpret it.
    if any(
        marker in message
        for marker in (
            "氧氟化物",
            "氧硫化物",
            "氧氮化物",
            "磷酸盐",
            "硫酸盐",
            "碳酸盐",
            "氢氧化物",
        )
    ):
        return None
    screening = _parse_bounded_oxide_screening(message)
    if screening is not None:
        return screening

    match = _STANDARD_OXIDE.search(message) or _STANDARD_OXIDE_REVERSED.search(message)
    if match is None:
        return None
    gap = float(match.group("gap"))
    return SearchMaterialsInput(
        required_elements=("O",),
        excluded_elements=_NON_OXIDE_ANION_ELEMENTS,
        filters=(
            PropertyFilter(field="band_gap_ev", operator=FilterOperator.GT, value=gap),
            PropertyFilter(field="is_stable", operator=FilterOperator.EQ, value=True),
            *_nonmetal_filter(message),
        ),
        limit=20,
        fields=(
            "material_id",
            "formula_pretty",
            "band_gap_ev",
            "is_stable",
            "energy_above_hull_ev_atom",
            "is_metal",
        ),
    )


def _parse_bounded_oxide_screening(message: str) -> SearchMaterialsInput | None:
    """Parse the bounded oxide-screening shape used by full-chain tasks."""

    normalized = message.casefold()
    if "氧化物" not in normalized or not any(
        marker in normalized for marker in ("筛选", "查找", "寻找", "screen")
    ):
        return None
    gap_match = _BAND_GAP_RANGE.search(message)
    hull_match = _HULL_MAXIMUM.search(message)
    if gap_match is None or hull_match is None:
        return None
    minimum = float(gap_match.group("minimum"))
    maximum = float(gap_match.group("maximum"))
    if minimum > maximum:
        return None
    excluded: tuple[str, ...] = ()
    excluded_match = _EXCLUDED_ELEMENTS.search(message)
    if excluded_match is not None:
        excluded = tuple(
            dict.fromkeys(re.findall(r"[A-Z][a-z]?", excluded_match.group("elements")))
        )
    return SearchMaterialsInput(
        required_elements=("O",),
        excluded_elements=tuple(dict.fromkeys((*excluded, *_NON_OXIDE_ANION_ELEMENTS))),
        filters=(
            PropertyFilter(
                field="band_gap_ev", operator=FilterOperator.GTE, value=minimum
            ),
            PropertyFilter(
                field="band_gap_ev", operator=FilterOperator.LTE, value=maximum
            ),
            PropertyFilter(
                field="energy_above_hull_ev_atom",
                operator=FilterOperator.LTE,
                value=float(hull_match.group("maximum")),
            ),
            *_nonmetal_filter(message),
        ),
        sort=(
            SortRule(field="energy_above_hull_ev_atom", direction="asc"),
            SortRule(field="band_gap_ev", direction="desc"),
        ),
        limit=20,
        fields=(
            "material_id",
            "formula_pretty",
            "band_gap_ev",
            "energy_above_hull_ev_atom",
            "density_g_cm3",
            "formation_energy_ev_atom",
            "crystal_system",
            "is_gap_direct",
            "is_metal",
        ),
    )


def _current_turn(
    request: MaterialAgentRequest,
) -> tuple[str, tuple[object, ...]]:
    items = tuple(request.input_items)
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, AgentMessageItem) and item.role == "user":
            return item.content, items[index + 1 :]
    return "", items


def _final_from_tool(
    output_item: AgentFunctionOutputItem, search: SearchMaterialsInput
) -> MaterialAgentResponse:
    try:
        envelope = json.loads(output_item.output)
    except json.JSONDecodeError:
        return _final("数据库工具返回无法解析，请重试。", status="error")
    if envelope.get("status") != "ok":
        error = envelope.get("error") or {}
        return _final(
            "数据库查询失败：" + str(error.get("message") or "未知错误"),
            status="error",
        )
    payload = envelope.get("output") or {}
    materials = payload.get("materials") or []
    gap = next(
        (
            item.value
            for item in search.filters
            if item.field == "band_gap_ev" and item.operator == FilterOperator.GT
        ),
        "—",
    )
    lines = [
        _criteria_summary(search, gap),
        "",
        f"匹配 {payload.get('matched_count', len(materials))} 条，"
        f"本次返回 {len(materials)} 条。",
        "",
        "| Material ID | 化学式 | 带隙 (eV) | 密度 (g/cm³) | E hull (eV/atom) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in materials:
        lines.append(
            f"| {row.get('material_id', '—')} | {row.get('formula_pretty', '—')} | "
            f"{_value(row.get('band_gap_ev'))} | {_value(row.get('density_g_cm3'))} | "
            f"{_value(row.get('energy_above_hull_ev_atom'))} |"
        )
    warnings = [str(item) for item in payload.get("warnings") or []]
    if set(_NON_OXIDE_ANION_ELEMENTS).issubset(search.excluded_elements):
        warnings.append(_OXIDE_SCOPE_WARNING)
    if warnings:
        lines.extend(("", "注意：" + "；".join(warnings)))
    query_id = payload.get("query_id")
    if isinstance(query_id, str) and query_id:
        lines.extend(("", f"MATERIAL_QUERY_HANDOFF: query_id={query_id}"))
    material_ids = [
        str(row["material_id"]) for row in materials if row.get("material_id")
    ]
    return _final(
        "\n".join(lines),
        evidence_ids=[str(envelope.get("evidence_id"))]
        if envelope.get("evidence_id")
        else [],
        material_ids=material_ids,
        warnings=warnings,
    )


def _criteria_summary(search: SearchMaterialsInput, legacy_gap: object) -> str:
    filters = {(item.field, item.operator.value): item.value for item in search.filters}
    minimum = filters.get(("band_gap_ev", "gte"))
    maximum = filters.get(("band_gap_ev", "lte"))
    hull = filters.get(("energy_above_hull_ev_atom", "lte"))
    nonmetal = "is_metal = false、" if filters.get(("is_metal", "eq")) is False else ""
    if minimum is not None and maximum is not None and hull is not None:
        excluded = "、".join(search.excluded_elements) or "无额外元素"
        return (
            "已按以下条件查询 Materials Project："
            f"含 O、{nonmetal}{minimum} ≤ band_gap_ev ≤ {maximum} eV、"
            f"energy_above_hull_ev_atom ≤ {hull} eV/atom，"
            f"排除 {excluded}（氧化物元素排除近似，非严格化学分类）。"
        )
    return (
        f"已按以下条件查询 Materials Project：含 O、{nonmetal}"
        f"band_gap_ev > {legacy_gap} eV、"
        "is_stable = true（热力学稳定），并排除常见非氧阴离子元素"
        "（氧化物元素排除近似，非严格化学分类）。"
    )


def _final(
    answer: str,
    *,
    status: str = "completed",
    evidence_ids: list[str] | None = None,
    material_ids: list[str] | None = None,
    warnings: list[str] | None = None,
) -> MaterialAgentResponse:
    content = {
        "status": status,
        "answer": answer,
        "referenced_material_ids": material_ids or [],
        "evidence_ids": evidence_ids or [],
        "warnings": warnings or [],
        "follow_up_question": None,
    }
    return MaterialAgentResponse(
        status=AgentModelStatus.COMPLETED,
        output_items=(
            AgentMessageItem(
                role="assistant", content=json.dumps(content, ensure_ascii=False)
            ),
        ),
        request_id="deterministic-oxide-final",
        provider="deterministic",
        model="structured-filter-v1",
    )


def _value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).lower() if isinstance(value, bool) else str(value)
